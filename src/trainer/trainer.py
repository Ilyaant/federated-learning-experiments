from __future__ import annotations

import logging
from collections import deque
from contextlib import contextmanager
from pathlib import Path
from typing import Deque, Dict

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


def prefix_metrics(metrics: Dict[str, float | str], prefix: str) -> Dict[str, float | str]:
    prefixed: Dict[str, float | str] = {}
    for key, value in metrics.items():
        if key not in _REPORTED_METRICS:
            continue
        prefixed[f"{prefix}{key}"] = value
    return prefixed


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
        eval_train: bool = True,
        swa_window: int = 0,
        swa_eval_every: int = 1,
        save_dir: str | Path | None = None,
    ):
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
        self.swa_window = int(swa_window)
        self.swa_eval_every = max(1, int(swa_eval_every))
        self.current_lr = initial_lr

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
        # Sequential pass over the same dataset for train-split evaluation:
        # patches of one image are visited together, so the per-image LRU
        # cache is hit instead of re-decoding the JPEG for every patch.
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
        self._swa_snapshots: Deque[dict[str, torch.Tensor]] = deque(
            maxlen=max(1, self.swa_window)
        )
        self.swa_state: dict[str, torch.Tensor] | None = None

    def _set_learning_rate(self, epoch: int) -> float:
        self.current_lr = cosine_learning_rate(
            epoch,
            self.epochs,
            self.initial_lr,
            self.min_lr,
        )
        for group in self.optimizer.param_groups:
            group["lr"] = self.current_lr
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

    def _update_swa(self) -> None:
        if self.swa_window <= 0:
            return
        self._swa_snapshots.append(clone_state_dict(self.model.state_dict()))
        self.swa_state = average_state_dicts(list(self._swa_snapshots))

    def _should_eval_swa(self, epoch: int) -> bool:
        if self.swa_window <= 0 or self.swa_state is None:
            return False
        is_last = epoch >= self.epochs
        return is_last or epoch % self.swa_eval_every == 0

    def _evaluate_swa(self) -> Dict[str, float | str]:
        assert self.swa_state is not None
        metrics: Dict[str, float | str] = {}
        with self._loaded_weights(self.swa_state):
            for split, loader in (("val", self.val_loader), ("test", self.test_loader)):
                result = self._run_evaluation(loader)
                metrics.update(prefix_metrics(result, f"swa_{split}_"))
        return metrics

    def _maybe_save_best(self, epoch: int, metrics: Dict[str, float | str]) -> None:
        val_score = None
        selected_metric = "val_f1"
        for metric_key in ("val_f1", "val_image_f1", "val_image_accuracy", "val_accuracy"):
            value = metrics.get(metric_key)
            if isinstance(value, (int, float)):
                val_score = float(value)
                selected_metric = metric_key
                break
        if val_score is None or val_score <= self.best_score:
            return

        self.best_score = val_score
        self.best_epoch = epoch
        self.best_state = clone_state_dict(self.model.state_dict())
        if self.history is not None:
            self.history.set_best(epoch, val_score, selected_metric)
        if self.save_dir is not None:
            save_model(self.model, self.save_dir / "model_best.pt")
            logger.info(
                "Epoch %s: new best val score %.4f, saved %s",
                epoch,
                val_score,
                self.save_dir / "model_best.pt",
            )

    def fit(self) -> LiveHistoryWriter | None:
        logger.info("Starting training for %s epochs on %s", self.epochs, self.device)

        for epoch in range(1, self.epochs + 1):
            lr = self._set_learning_rate(epoch)
            logger.info("Epoch %s/%s learning rate set to %.6g", epoch, self.epochs, lr)

            train_loss = self.train_one_epoch(epoch)
            logger.info("Epoch %s/%s train_loss=%.6f", epoch, self.epochs, train_loss)

            epoch_metrics: Dict[str, float | str] = {
                "train_loss": float(train_loss),
                "lr": float(lr),
            }

            if self.eval_train:
                train_metrics = self._evaluate_train()
                train_prefixed = prefix_metrics(train_metrics, "train_")
                train_prefixed.pop("train_loss", None)
                epoch_metrics.update(train_prefixed)

            val_metrics = self._run_evaluation(self.val_loader)
            test_metrics = self._run_evaluation(self.test_loader)
            epoch_metrics.update(prefix_metrics(val_metrics, "val_"))
            epoch_metrics.update(prefix_metrics(test_metrics, "test_"))

            self._update_swa()
            if self._should_eval_swa(epoch):
                swa_metrics = self._evaluate_swa()
                epoch_metrics.update(swa_metrics)
                logger.info(
                    "Epoch %s SWA(last %s epochs) metrics: %s",
                    epoch,
                    len(self._swa_snapshots),
                    {
                        key: value
                        for key, value in swa_metrics.items()
                        if not str(key).endswith("confusion_matrix")
                    },
                )

            self._maybe_save_best(epoch, epoch_metrics)
            if self.history is not None:
                self.history.update(epoch, epoch_metrics)

            logger.info(
                "Epoch %s/%s val_f1=%s test_f1=%s",
                epoch,
                self.epochs,
                epoch_metrics.get("val_f1"),
                epoch_metrics.get("test_f1"),
            )

        self._save_final_models()
        return self.history

    def _save_final_models(self) -> None:
        if self.save_dir is None:
            return

        save_model(self.model, self.save_dir / "model_final.pt")
        logger.info("Saved final model to %s", self.save_dir / "model_final.pt")

        if self.best_state is not None:
            logger.info(
                "Best model is from epoch %s (val_score=%.4f)",
                self.best_epoch,
                self.best_score,
            )

        if self.swa_state is not None:
            current = clone_state_dict(self.model.state_dict())
            load_state_dict(self.model, self.swa_state)
            save_model(self.model, self.save_dir / "model_swa.pt")
            load_state_dict(self.model, current)
            logger.info(
                "Saved SWA model (mean of last %s epochs) to %s",
                len(self._swa_snapshots),
                self.save_dir / "model_swa.pt",
            )
