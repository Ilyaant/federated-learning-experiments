from __future__ import annotations

import logging
from collections import deque
from contextlib import contextmanager
from pathlib import Path
from typing import Deque, Dict, Iterable, List, Sequence

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .history import LiveHistoryWriter, save_model
from .metrics import AverageMeter, evaluate
from .utils import (
    average_state_dicts,
    clone_state_dict,
    cosine_learning_rate,
    get_device,
    load_state_dict,
)


logger = logging.getLogger(__name__)

_REPORTED_METRICS = {
    "loss",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "image_accuracy",
    "image_precision",
    "image_recall",
    "image_f1",
    "confusion_matrix",
    "image_confusion_matrix",
}

_DEFAULT_SELECTION_FALLBACKS = (
    "val_f1",
    "val_accuracy",
    "val_image_f1",
    "val_image_accuracy",
)


def prefix_metrics(metrics: Dict[str, float | str], prefix: str) -> Dict[str, float | str]:
    prefixed: Dict[str, float | str] = {}
    for key, value in metrics.items():
        if key not in _REPORTED_METRICS:
            continue
        prefixed[f"{prefix}{key}"] = value
    return prefixed


def numeric_metrics(metrics: Dict[str, float | str]) -> Dict[str, float]:
    return {
        key: float(value)
        for key, value in metrics.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    }


class Trainer:
    def __init__(
        self,
        model,
        train_dataset,
        val_dataset,
        test_dataset,
        optimizer,
        criterion,
        eval_criterion=None,
        batch_size: int = 32,
        epochs: int = 100,
        num_workers: int = 4,
        num_classes: int = 5,
        aggregation: str = "average_probability",
        device=None,
        initial_lr: float = 0.0001,
        min_lr: float = 1e-6,
        max_grad_norm: float | None = 1.0,
        tta: bool = False,
        eval_train: bool = False,
        eval_train_every: int = 1,
        eval_val_every: int = 1,
        eval_test: str = "end",
        selection_metric: str = "val_f1",
        early_stopping_patience: int = 0,
        min_delta: float = 0.0,
        swa_window: int = 0,
        swa_mode: str = "best",
        swa_eval_every: int = 0,
        save_dir: str | Path | None = None,
    ):
        if eval_test not in {"end", "every", "never"}:
            raise ValueError("eval_test must be 'end', 'every' or 'never'")
        if swa_mode not in {"best", "last"}:
            raise ValueError("swa_mode must be 'best' or 'last'")

        self.device = device or get_device()
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.eval_criterion = (
            eval_criterion if eval_criterion is not None else criterion
        )
        self.epochs = epochs
        self.num_classes = num_classes
        self.aggregation = aggregation
        self.initial_lr = initial_lr
        self.min_lr = min_lr
        self.max_grad_norm = max_grad_norm
        self.tta = tta
        self.eval_train = eval_train
        self.eval_train_every = max(1, int(eval_train_every))
        self.eval_val_every = max(1, int(eval_val_every))
        self.eval_test = eval_test
        self.selection_metric = selection_metric
        self.early_stopping_patience = int(early_stopping_patience)
        self.min_delta = float(min_delta)
        self.swa_window = int(swa_window)
        self.swa_mode = swa_mode
        self.swa_eval_every = max(0, int(swa_eval_every))
        self.current_lr = initial_lr
        for group in self.optimizer.param_groups:
            group.setdefault("initial_lr", group["lr"])

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset

        # persistent_workers is intentionally off: workers must be
        # re-forked each epoch to pick up set_epoch() resampling.
        loader_kwargs = {
            "batch_size": batch_size,
            "num_workers": num_workers,
            "pin_memory": torch.cuda.is_available(),
        }
        self.train_loader = DataLoader(
            train_dataset,
            shuffle=True,
            drop_last=False,
            **loader_kwargs,
        )
        self.train_eval_loader = DataLoader(
            train_dataset,
            shuffle=False,
            drop_last=False,
            **loader_kwargs,
        )
        self.val_loader = DataLoader(
            val_dataset,
            shuffle=False,
            drop_last=False,
            **loader_kwargs,
        )
        self.test_loader = DataLoader(
            test_dataset,
            shuffle=False,
            drop_last=False,
            **loader_kwargs,
        )

        self.save_dir = Path(save_dir) if save_dir is not None else None
        self.history = (
            LiveHistoryWriter(self.save_dir) if self.save_dir is not None else None
        )
        self.best_score = -float("inf")
        self.best_epoch = 0
        self.best_state = None
        self.best_val_metrics: Dict[str, float | str] = {}
        self._swa_snapshots: Deque[dict[str, torch.Tensor]] = deque(
            maxlen=max(1, self.swa_window)
        )
        self._swa_scored: List[tuple[float, int, dict[str, torch.Tensor]]] = []
        self.swa_state: dict[str, torch.Tensor] | None = None
        self.stopped_epoch: int | None = None

    def _selection_candidates(self) -> Sequence[str]:
        seen = set()
        ordered = []
        for key in (self.selection_metric, *_DEFAULT_SELECTION_FALLBACKS):
            if key not in seen:
                seen.add(key)
                ordered.append(key)
        return ordered

    def _selection_score(
        self,
        metrics: Dict[str, float | str],
    ) -> tuple[str | None, float | None]:
        for key in self._selection_candidates():
            value = metrics.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return key, float(value)
        return None, None

    def _set_learning_rate(self, epoch: int) -> float:
        self.current_lr = cosine_learning_rate(
            epoch,
            self.epochs,
            self.initial_lr,
            self.min_lr,
        )
        scale = self.current_lr / self.initial_lr if self.initial_lr else 1.0
        for group in self.optimizer.param_groups:
            group["lr"] = float(group.get("initial_lr", self.initial_lr)) * scale
        return self.current_lr

    def train_one_epoch(self, epoch: int) -> float:
        if hasattr(self.train_dataset, "set_epoch"):
            self.train_dataset.set_epoch(epoch)

        self.model.train()
        loss_meter = AverageMeter()

        for batch in self.train_loader:
            images = batch["image"].to(self.device, non_blocking=True)
            labels = batch["label"].to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)
            logits = self.model(images)
            loss = self.criterion(logits, labels)
            loss.backward()
            if self.max_grad_norm is not None:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.optimizer.step()

            loss_meter.update(loss.item(), images.size(0))

        return loss_meter.avg

    def _run_evaluation(self, dataloader, tta: bool | None = None):
        return evaluate(
            self.model,
            dataloader,
            self.eval_criterion,
            self.device,
            num_classes=self.num_classes,
            aggregation=self.aggregation,
            tta=self.tta if tta is None else tta,
        )

    def _evaluate_train(self):
        dataset = self.train_dataset
        restored_fraction = None
        restored_transform = getattr(dataset, "transform", None)

        if hasattr(dataset, "epoch_fraction") and dataset.epoch_fraction < 1.0:
            restored_fraction = dataset.epoch_fraction
            dataset.epoch_fraction = 1.0
            dataset.set_epoch(0)

        if hasattr(dataset, "transform"):
            dataset.transform = None

        try:
            metrics = self._run_evaluation(self.train_eval_loader, tta=False)
        finally:
            if hasattr(dataset, "transform"):
                dataset.transform = restored_transform
            if restored_fraction is not None:
                dataset.epoch_fraction = restored_fraction
                dataset.set_epoch(0)

        return metrics

    @contextmanager
    def _loaded_weights(self, state_dict: dict[str, torch.Tensor]):
        current = clone_state_dict(self.model.state_dict())
        load_state_dict(self.model, state_dict)
        try:
            yield
        finally:
            load_state_dict(self.model, current)

    def _update_swa(self, score: float, epoch: int) -> None:
        if self.swa_window <= 0:
            return
        state = clone_state_dict(self.model.state_dict())
        if self.swa_mode == "last":
            self._swa_snapshots.append(state)
            self.swa_state = average_state_dicts(list(self._swa_snapshots))
            return

        self._swa_scored.append((score, epoch, state))
        self._swa_scored.sort(key=lambda item: item[0], reverse=True)
        self._swa_scored = self._swa_scored[: self.swa_window]
        self.swa_state = average_state_dicts(
            [item[2] for item in self._swa_scored]
        )

    def _swa_count(self) -> int:
        if self.swa_mode == "last":
            return len(self._swa_snapshots)
        return len(self._swa_scored)

    def _should_eval_swa(self, epoch: int, is_last: bool) -> bool:
        if self.swa_window <= 0 or self.swa_state is None:
            return False
        if is_last or self.swa_eval_every <= 0:
            return is_last
        return epoch % self.swa_eval_every == 0

    def _evaluate_swa(self, splits: Iterable[str]) -> Dict[str, float | str]:
        assert self.swa_state is not None
        loaders = {"val": self.val_loader, "test": self.test_loader}
        metrics: Dict[str, float | str] = {}
        with self._loaded_weights(self.swa_state):
            for split in splits:
                result = self._run_evaluation(loaders[split])
                metrics.update(prefix_metrics(result, f"swa_{split}_"))
        return metrics

    def _maybe_save_best(
        self,
        epoch: int,
        metrics: Dict[str, float | str],
        score: float,
        metric_name: str,
    ) -> bool:
        if not score > self.best_score + self.min_delta:
            return False

        self.best_score = score
        self.best_epoch = epoch
        self.best_state = clone_state_dict(self.model.state_dict())
        self.best_val_metrics = {
            key: value
            for key, value in metrics.items()
            if str(key).startswith("val_")
        }
        if self.history is not None:
            self.history.set_best(epoch, score, metric_name)
        if self.save_dir is not None:
            save_model(self.model, self.save_dir / "model_best.pt")
            logger.info(
                "Epoch %s: new best %s %.4f, saved %s",
                epoch,
                metric_name,
                score,
                self.save_dir / "model_best.pt",
            )
        return True

    def _should_eval_train(self, epoch: int, is_last: bool) -> bool:
        if not self.eval_train:
            return False
        return is_last or epoch % self.eval_train_every == 0

    def _should_eval_val(self, epoch: int, is_last: bool) -> bool:
        return is_last or epoch % self.eval_val_every == 0

    def fit(self) -> LiveHistoryWriter | None:
        logger.info(
            "Starting training for %s epochs on %s (selection=%s, eval_test=%s)",
            self.epochs,
            self.device,
            self.selection_metric,
            self.eval_test,
        )
        epochs_without_improve = 0

        for epoch in range(1, self.epochs + 1):
            is_last = epoch >= self.epochs
            lr = self._set_learning_rate(epoch)
            logger.info("Epoch %s/%s learning rate set to %.6g", epoch, self.epochs, lr)

            train_loss = self.train_one_epoch(epoch)
            logger.info("Epoch %s/%s train_loss=%.6f", epoch, self.epochs, train_loss)

            epoch_metrics: Dict[str, float | str] = {
                "train_loss": float(train_loss),
                "lr": float(lr),
            }

            if self._should_eval_train(epoch, is_last):
                train_metrics = self._evaluate_train()
                train_prefixed = prefix_metrics(train_metrics, "train_")
                train_prefixed.pop("train_loss", None)
                epoch_metrics.update(train_prefixed)

            improved = False
            selected_name, selected_score = None, None
            if self._should_eval_val(epoch, is_last):
                val_metrics = self._run_evaluation(self.val_loader)
                epoch_metrics.update(prefix_metrics(val_metrics, "val_"))
                selected_name, selected_score = self._selection_score(epoch_metrics)
                if selected_score is not None:
                    self._update_swa(selected_score, epoch)
                    improved = self._maybe_save_best(
                        epoch,
                        epoch_metrics,
                        selected_score,
                        selected_name or self.selection_metric,
                    )

            if self.eval_test == "every":
                test_metrics = self._run_evaluation(self.test_loader)
                epoch_metrics.update(prefix_metrics(test_metrics, "test_"))

            swa_splits = ["val"]
            if self.eval_test == "every":
                swa_splits.append("test")
            if self._should_eval_swa(epoch, is_last=False) and self.swa_state is not None:
                swa_metrics = self._evaluate_swa(swa_splits)
                epoch_metrics.update(swa_metrics)

            if self.history is not None:
                self.history.update(epoch, epoch_metrics)

            logger.info(
                "Epoch %s/%s %s=%s val_image_f1=%s",
                epoch,
                self.epochs,
                self.selection_metric,
                epoch_metrics.get(self.selection_metric),
                epoch_metrics.get("val_image_f1"),
            )

            if selected_score is not None:
                if improved:
                    epochs_without_improve = 0
                else:
                    epochs_without_improve += 1
                    if (
                        self.early_stopping_patience > 0
                        and epochs_without_improve >= self.early_stopping_patience
                    ):
                        self.stopped_epoch = epoch
                        logger.info(
                            "Early stopping at epoch %s (%s did not improve for %s evals)",
                            epoch,
                            self.selection_metric,
                            self.early_stopping_patience,
                        )
                        break

        self._finalize()
        return self.history

    def _finalize(self) -> None:
        self._save_final_models()
        summary = self._held_out_summary()
        if self.history is not None:
            self.history.set_summary(summary)
        logger.info("Run summary: %s", summary)

    def _held_out_summary(self) -> Dict[str, object]:
        summary: Dict[str, object] = {
            "selection_metric": self.selection_metric,
            "best_epoch": self.best_epoch,
            "best_score": None if self.best_epoch == 0 else float(self.best_score),
            "stopped_epoch": self.stopped_epoch,
            "eval_test": self.eval_test,
        }
        if self.best_val_metrics:
            summary["best_val"] = numeric_metrics(self.best_val_metrics)

        chosen = "best"
        chosen_test: Dict[str, float] = {}
        best_test: Dict[str, float] = {}
        swa_val: Dict[str, float] = {}
        swa_test: Dict[str, float] = {}
        swa_val_score = None

        if self.best_state is not None and self.eval_test != "never":
            with self._loaded_weights(self.best_state):
                best_test = numeric_metrics(
                    prefix_metrics(self._run_evaluation(self.test_loader), "test_")
                )
            summary["best_test"] = best_test
            chosen_test = best_test
            logger.info(
                "Held-out test of best epoch %s: %s",
                self.best_epoch,
                {k: v for k, v in best_test.items() if "confusion" not in k},
            )

        if self.swa_state is not None:
            swa_splits = ["val"]
            if self.eval_test != "never":
                swa_splits.append("test")
            swa_metrics = self._evaluate_swa(swa_splits)
            swa_val = numeric_metrics(
                {k: v for k, v in swa_metrics.items() if k.startswith("swa_val_")}
            )
            swa_test = numeric_metrics(
                {k: v for k, v in swa_metrics.items() if k.startswith("swa_test_")}
            )
            summary["swa_val"] = swa_val
            if swa_test:
                summary["swa_test"] = swa_test
            _, swa_val_score = self._selection_score(
                {k.replace("swa_", "", 1): v for k, v in swa_metrics.items()}
            )
            if (
                swa_val_score is not None
                and swa_val_score > self.best_score
                and swa_test
            ):
                chosen = "swa"
                chosen_test = {
                    key.replace("swa_", "", 1): value
                    for key, value in swa_test.items()
                }
            logger.info(
                "SWA (%s, %s snapshots) val score=%s",
                self.swa_mode,
                self._swa_count(),
                swa_val_score,
            )

        summary["selected_weights"] = chosen
        summary["selected_test"] = chosen_test
        if chosen_test:
            summary["test_image_f1"] = chosen_test.get("test_image_f1")
            summary["test_f1"] = chosen_test.get("test_f1")
        if self.best_val_metrics:
            summary["val_image_f1"] = numeric_metrics(self.best_val_metrics).get(
                "val_image_f1"
            )
            summary["val_f1"] = numeric_metrics(self.best_val_metrics).get("val_f1")
        summary["swa_val_score"] = swa_val_score
        return summary

    def _save_final_models(self) -> None:
        if self.save_dir is None:
            return

        save_model(self.model, self.save_dir / "model_final.pt")
        logger.info("Saved final model to %s", self.save_dir / "model_final.pt")

        if self.best_state is not None:
            logger.info(
                "Best model is from epoch %s (%s=%.4f)",
                self.best_epoch,
                self.selection_metric,
                self.best_score,
            )

        if self.swa_state is not None:
            current = clone_state_dict(self.model.state_dict())
            load_state_dict(self.model, self.swa_state)
            save_model(self.model, self.save_dir / "model_swa.pt")
            load_state_dict(self.model, current)
            logger.info(
                "Saved SWA model (%s, %s snapshots) to %s",
                self.swa_mode,
                self._swa_count(),
                self.save_dir / "model_swa.pt",
            )
