from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .config import Config
from .dataset import PerImagePatchSampler, TexturePatchDataset
from .logging_utils import (
    CsvLogger,
    format_metrics,
    get_logger,
    save_json,
)
from .metrics import (
    METRIC_NAMES,
    aggregate_image_predictions,
    classification_metrics,
    format_confusion_matrix,
    format_per_class,
)
from .model import split_parameters

LEVELS = ("patch", "image")

EPOCH_CSV_FIELDS = (
    ["epoch", "lr", "epoch_time_sec", "train_loss"]
    + [f"train_patch_{m}" for m in METRIC_NAMES]
    + ["val_loss"]
    + [f"val_patch_{m}" for m in METRIC_NAMES]
    + [f"val_image_{m}" for m in METRIC_NAMES]
    + ["val_n_patches", "val_n_images", "is_best"]
)

FINAL_CSV_FIELDS = [
    "checkpoint",
    "split",
    "level",
    "n",
    "loss",
    *METRIC_NAMES,
]


class Trainer:
    """Trains a patch classifier and scores it on patches and whole images."""

    def __init__(
        self,
        config: Config,
        model: nn.Module,
        class_names: Sequence[str],
        train_loader: DataLoader,
        train_sampler: PerImagePatchSampler,
        val_loader: DataLoader,
        test_loader: DataLoader,
        run_dir: str | Path,
        device: torch.device,
    ):
        self.config = config
        self.model = model.to(device)
        self.class_names = list(class_names)
        self.train_loader = train_loader
        self.train_sampler = train_sampler
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.run_dir = Path(run_dir)
        self.device = device
        self.logger = get_logger()

        train_cfg = config.get("train", {})
        eval_cfg = config.get("evaluation", {})
        log_cfg = config.get("logging", {})

        self.epochs = int(train_cfg.get("epochs", 1))
        self.max_grad_norm = train_cfg.get("max_grad_norm")
        self.log_every = int(log_cfg.get("log_every_n_batches", 20))
        self.save_last = bool(log_cfg.get("save_last", True))
        self.use_amp = bool(train_cfg.get("amp", False)) and device.type == "cuda"
        self.aggregation = eval_cfg.get("aggregation", "mean_prob")
        self.average = eval_cfg.get("metrics_average", "macro")
        self.selection_metric = eval_cfg.get("selection_metric", "val_patch_f1")
        self.minimize = self.selection_metric == "val_loss"
        self.patience = eval_cfg.get("early_stopping_patience")
        self.min_delta = float(eval_cfg.get("min_delta", 0.0))

        self.criterion = self._build_criterion(train_cfg)
        self.optimizer = self._build_optimizer(train_cfg)
        self.scheduler = self._build_scheduler(train_cfg)
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        self.epoch_csv = CsvLogger(self.run_dir / log_cfg.get("metrics_csv", "metrics.csv"), EPOCH_CSV_FIELDS)
        self.final_csv_path = self.run_dir / log_cfg.get("final_csv", "final_metrics.csv")
        self.history: List[Dict[str, object]] = []
        self.best_value: Optional[float] = None
        self.best_epoch: Optional[int] = None

    # ------------------------------------------------------------- builders
    def _build_criterion(self, train_cfg: dict) -> nn.Module:
        weight = None
        if train_cfg.get("class_weights", False):
            dataset: TexturePatchDataset = self.train_loader.dataset  # type: ignore[assignment]
            counts = dataset.class_distribution()
            total = sum(counts.values())
            weights = [total / (len(self.class_names) * counts.get(i, 1)) for i in range(len(self.class_names))]
            weight = torch.tensor(weights, dtype=torch.float32, device=self.device)
            self.logger.info("Class weights: %s", {c: round(w, 3) for c, w in zip(self.class_names, weights)})
        return nn.CrossEntropyLoss(
            weight=weight,
            label_smoothing=float(train_cfg.get("label_smoothing", 0.0)),
        )

    def _build_optimizer(self, train_cfg: dict) -> torch.optim.Optimizer:
        lr = float(train_cfg.get("lr", 1e-4))
        backbone_lr = train_cfg.get("backbone_lr")
        weight_decay = float(train_cfg.get("weight_decay", 0.0))

        backbone, head = split_parameters(self.model)
        groups = []
        if backbone:
            groups.append({"params": backbone, "lr": float(backbone_lr) if backbone_lr else lr, "name": "backbone"})
        if head:
            groups.append({"params": head, "lr": lr, "name": "head"})
        for group in groups:
            group["initial_lr"] = group["lr"]
        return torch.optim.AdamW(groups, weight_decay=weight_decay)

    def _build_scheduler(self, train_cfg: dict) -> torch.optim.lr_scheduler.LambdaLR:
        steps_per_epoch = max(1, len(self.train_loader))
        total_steps = max(1, self.epochs * steps_per_epoch)
        warmup_steps = int(float(train_cfg.get("warmup_epochs", 0.0)) * steps_per_epoch)
        min_lr = float(train_cfg.get("min_lr", 0.0))
        base_lrs = [group["initial_lr"] for group in self.optimizer.param_groups]

        def factor_for(base_lr: float):
            floor = min_lr / base_lr if base_lr > 0 else 0.0

            def factor(step: int) -> float:
                if warmup_steps > 0 and step < warmup_steps:
                    return max(floor, (step + 1) / warmup_steps)
                progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
                progress = min(1.0, max(0.0, progress))
                return floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * progress))

            return factor

        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, [factor_for(lr) for lr in base_lrs])

    # --------------------------------------------------------------- epochs
    def _current_lr(self) -> float:
        return float(self.optimizer.param_groups[-1]["lr"])

    def train_one_epoch(self, epoch: int) -> Dict[str, object]:
        self.model.train()
        self.train_sampler.set_epoch(epoch)

        total_loss, seen = 0.0, 0
        preds: List[int] = []
        labels: List[int] = []
        num_batches = len(self.train_loader)
        start = time.time()

        for batch_idx, batch in enumerate(self.train_loader, start=1):
            images = batch["image"].to(self.device, non_blocking=True)
            targets = batch["label"].to(self.device, non_blocking=True)

            with torch.autocast(device_type=self.device.type, enabled=self.use_amp):
                logits = self.model(images)
                loss = self.criterion(logits, targets)

            self.optimizer.zero_grad(set_to_none=True)
            self.scaler.scale(loss).backward()
            if self.max_grad_norm:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), float(self.max_grad_norm))
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()

            batch_size = targets.size(0)
            total_loss += loss.item() * batch_size
            seen += batch_size
            preds.extend(logits.argmax(dim=1).tolist())
            labels.extend(targets.tolist())

            if self.log_every and (batch_idx % self.log_every == 0 or batch_idx == num_batches):
                running_acc = float(np.mean(np.array(preds) == np.array(labels)))
                self.logger.info(
                    "epoch %d | batch %d/%d | loss %.4f | acc %.4f | lr %.2e | %.0fs",
                    epoch, batch_idx, num_batches, total_loss / max(1, seen), running_acc,
                    self._current_lr(), time.time() - start,
                )

        metrics = classification_metrics(labels, preds, self.class_names, self.average)
        return {"loss": total_loss / max(1, seen), "patch": metrics}

    @torch.no_grad()
    def evaluate(self, loader: DataLoader, split: str) -> Dict[str, object]:
        """Loss + patch-level metrics + image-level metrics for one split."""
        self.model.eval()
        dataset: TexturePatchDataset = loader.dataset  # type: ignore[assignment]

        total_loss, seen = 0.0, 0
        all_probs: List[np.ndarray] = []
        labels: List[int] = []
        image_ids: List[int] = []

        for batch in loader:
            images = batch["image"].to(self.device, non_blocking=True)
            targets = batch["label"].to(self.device, non_blocking=True)

            with torch.autocast(device_type=self.device.type, enabled=self.use_amp):
                logits = self.model(images)
                loss = self.criterion(logits, targets)

            total_loss += loss.item() * targets.size(0)
            seen += targets.size(0)
            all_probs.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
            labels.extend(targets.tolist())
            image_ids.extend(batch["image_id"].tolist())

        probs = np.concatenate(all_probs, axis=0) if all_probs else np.zeros((0, len(self.class_names)))
        patch_preds = probs.argmax(axis=1).tolist() if len(probs) else []
        patch_metrics = classification_metrics(labels, patch_preds, self.class_names, self.average)

        per_image = aggregate_image_predictions(probs, image_ids, self.aggregation)
        image_true = [dataset.samples[image_id][1] for image_id in per_image]
        image_pred = [int(vec.argmax()) for vec in per_image.values()]
        image_metrics = classification_metrics(image_true, image_pred, self.class_names, self.average)

        return {
            "split": split,
            "loss": total_loss / max(1, seen),
            "patch": patch_metrics,
            "image": image_metrics,
            "n_patches": int(seen),
            "n_images": len(per_image),
        }

    # ------------------------------------------------------------ selection
    def _selection_value(self, val_result: Dict[str, object]) -> float:
        if self.selection_metric == "val_loss":
            return float(val_result["loss"])
        _, level, metric = self.selection_metric.split("_", 2)
        return float(val_result[level][metric])

    def _is_improvement(self, value: float) -> bool:
        if self.best_value is None:
            return True
        if self.minimize:
            return value < self.best_value - self.min_delta
        return value > self.best_value + self.min_delta

    def _save_checkpoint(self, name: str, epoch: int, value: float) -> Path:
        path = self.run_dir / name
        torch.save(
            {
                "model_state": self.model.state_dict(),
                "epoch": epoch,
                "class_names": self.class_names,
                "selection_metric": self.selection_metric,
                "selection_value": value,
                "config": self.config,
            },
            path,
        )
        return path

    def _log_split(self, prefix: str, result: Dict[str, object]) -> None:
        self.logger.info(
            "%s | loss %.4f | patch[%d]: %s | image[%d]: %s",
            prefix, result["loss"], result["n_patches"], format_metrics(result["patch"]),
            result["n_images"], format_metrics(result["image"]),
        )

    # ------------------------------------------------------------------ fit
    def fit(self) -> Dict[str, object]:
        self.logger.info(
            "Training %d epochs | selection metric %s | patience %s | amp %s",
            self.epochs, self.selection_metric, self.patience, self.use_amp,
        )
        epochs_without_improvement = 0

        for epoch in range(1, self.epochs + 1):
            start = time.time()
            train_result = self.train_one_epoch(epoch)
            val_result = self.evaluate(self.val_loader, "val")
            elapsed = time.time() - start

            value = self._selection_value(val_result)
            is_best = self._is_improvement(value)
            if is_best:
                self.best_value, self.best_epoch = value, epoch
                epochs_without_improvement = 0
                self._save_checkpoint("model_best.pt", epoch, value)
            else:
                epochs_without_improvement += 1
            if self.save_last:
                self._save_checkpoint("model_last.pt", epoch, value)

            self.logger.info(
                "epoch %d/%d done in %.0fs | train loss %.4f | train patch: %s",
                epoch, self.epochs, elapsed, train_result["loss"], format_metrics(train_result["patch"]),
            )
            self._log_split(f"epoch {epoch} val", val_result)
            self.logger.info(
                "%s = %.4f (best %.4f @ epoch %s)%s",
                self.selection_metric, value, self.best_value, self.best_epoch, " *" if is_best else "",
            )

            row = self._epoch_row(epoch, elapsed, train_result, val_result, is_best)
            self.epoch_csv.write(row)
            self.history.append(
                {
                    **row,
                    "train": train_result,
                    "val": val_result,
                }
            )
            save_json(self.history, self.run_dir / "history.json")

            if self.patience and epochs_without_improvement >= int(self.patience):
                self.logger.info(
                    "Early stopping: no %s improvement for %d epochs", self.selection_metric, self.patience
                )
                break

        return {"best_epoch": self.best_epoch, "best_value": self.best_value}

    def _epoch_row(
        self,
        epoch: int,
        elapsed: float,
        train_result: Dict[str, object],
        val_result: Dict[str, object],
        is_best: bool,
    ) -> Dict[str, object]:
        row: Dict[str, object] = {
            "epoch": epoch,
            "lr": self._current_lr(),
            "epoch_time_sec": elapsed,
            "train_loss": train_result["loss"],
            "val_loss": val_result["loss"],
            "val_n_patches": val_result["n_patches"],
            "val_n_images": val_result["n_images"],
            "is_best": is_best,
        }
        for metric in METRIC_NAMES:
            row[f"train_patch_{metric}"] = train_result["patch"][metric]
            for level in LEVELS:
                row[f"val_{level}_{metric}"] = val_result[level][metric]
        return row

    # ------------------------------------------------------------ final eval
    def final_evaluation(
        self,
        checkpoint: str = "model_best.pt",
        train_eval_loader: Optional[DataLoader] = None,
    ) -> Dict[str, object]:
        """Score the selected checkpoint on val and test (and optionally train)."""
        path = self.run_dir / checkpoint
        if path.exists():
            state = torch.load(path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(state["model_state"])
            epoch = state.get("epoch")
            self.logger.info("Loaded %s (epoch %s) for final evaluation", checkpoint, epoch)
        else:
            epoch = None
            self.logger.warning("Checkpoint %s not found, evaluating current weights", checkpoint)

        loaders = {"val": self.val_loader, "test": self.test_loader}
        if train_eval_loader is not None:
            loaders = {"train": train_eval_loader, **loaders}

        results: Dict[str, object] = {}
        final_csv = CsvLogger(self.final_csv_path, FINAL_CSV_FIELDS)
        for split, loader in loaders.items():
            result = self.evaluate(loader, split)
            results[split] = result
            self._log_split(f"FINAL {split}", result)
            for level in LEVELS:
                metrics = result[level]
                self.logger.info(
                    "FINAL %s %s-level per class:\n%s\n%s",
                    split, level, format_per_class(metrics["per_class"]),
                    format_confusion_matrix(metrics["confusion_matrix"], self.class_names),
                )
                final_csv.write(
                    {
                        "checkpoint": checkpoint,
                        "split": split,
                        "level": level,
                        "n": metrics["n"],
                        "loss": result["loss"] if level == "patch" else None,
                        **{m: metrics[m] for m in METRIC_NAMES},
                    }
                )

        summary = {
            "checkpoint": checkpoint,
            "checkpoint_epoch": epoch,
            "selection_metric": self.selection_metric,
            "best_value": self.best_value,
            "best_epoch": self.best_epoch,
            "epochs_run": len(self.history),
            "class_names": self.class_names,
            "results": results,
        }
        save_json(summary, self.run_dir / "summary.json")
        return summary
