from __future__ import annotations

import logging
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List

import flwr as fl
import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .metrics import AverageMeter, evaluate
from .utils import get_device


logger = logging.getLogger(__name__)


def _create_client_logger(
    client_id: int | None,
    log_dir: str | Path | None,
) -> logging.Logger:
    if client_id is None or log_dir is None:
        return logger

    client_logger = logging.getLogger(f"{__name__}.client_{client_id}")
    client_logger.setLevel(logging.INFO)
    client_logger.propagate = False

    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = (log_dir / f"client_{client_id}.log").resolve()

    if not any(
        isinstance(handler, logging.FileHandler)
        and Path(handler.baseFilename).resolve() == log_path
        for handler in client_logger.handlers
    ):
        handler = logging.FileHandler(log_path, encoding="utf-8")
        handler.setLevel(logging.INFO)
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
            )
        )
        client_logger.addHandler(handler)

    return client_logger


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
}


def _prefixed_metrics(metrics: Dict[str, float], prefix: str) -> Dict[str, float]:
    return {
        f"{prefix}{key}": float(value)
        for key, value in metrics.items()
        if key in _REPORTED_METRICS
    }


class FlowerClient(fl.client.NumPyClient):
    def __init__(
        self,
        model,
        train_dataset,
        val_dataset,
        test_dataset,
        optimizer,
        criterion,
        batch_size: int = 32,
        local_epochs: int = 1,
        num_workers: int = 4,
        num_classes: int = 5,
        aggregation: str = "average_probability",
        device=None,
        client_id: int | None = None,
        log_dir: str | Path | None = None,
        initial_lr: float = 0.0001,
        min_lr: float = 1e-6,
        total_rounds: int = 100,
        max_grad_norm: float | None = 1.0,
    ):
        self.logger = _create_client_logger(client_id, log_dir)
        self.initial_lr = initial_lr
        self.min_lr = min_lr
        self.total_rounds = total_rounds
        self.max_grad_norm = max_grad_norm
        self.device = device or get_device()
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.local_epochs = local_epochs
        self.num_classes = num_classes
        self.aggregation = aggregation

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

    def get_parameters(self, config=None):
        return [
            value.detach().cpu().numpy()
            for value in self.model.state_dict().values()
        ]

    def set_parameters(self, parameters: List[np.ndarray]):
        state_dict = OrderedDict(
            (key, torch.from_numpy(value))
            for key, value in zip(
                self.model.state_dict().keys(),
                parameters,
            )
        )
        self.model.load_state_dict(state_dict, strict=True)

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

    def _run_evaluation(self, dataloader):
        return evaluate(
            self.model,
            dataloader,
            self.criterion,
            self.device,
            num_classes=self.num_classes,
            aggregation=self.aggregation,
        )

    def _evaluate_train(self):
        dataset = self.train_dataset
        restored_fraction = None

        if hasattr(dataset, "epoch_fraction") and dataset.epoch_fraction < 1.0:
            restored_fraction = dataset.epoch_fraction
            dataset.epoch_fraction = 1.0
            dataset.set_epoch(0)

        metrics = self._run_evaluation(self.train_loader)

        if restored_fraction is not None:
            dataset.epoch_fraction = restored_fraction

        return metrics

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        server_round = config.get("server_round", 1)
        
        if self.total_rounds > 1 and isinstance(server_round, int):
            progress = (server_round - 1) / (self.total_rounds - 1)
            progress = min(1.0, max(0.0, progress))
            current_lr = self.min_lr + 0.5 * (self.initial_lr - self.min_lr) * (1 + math.cos(math.pi * progress))
            for g in self.optimizer.param_groups:
                g['lr'] = current_lr
            self.logger.info("Round %s learning rate set to %.6f", server_round, current_lr)

        self.logger.info("Starting local training for round %s", server_round)

        train_loss = 0.0
        for epoch in range(self.local_epochs):
            train_loss = self.train_one_epoch(epoch)
            self.logger.info(
                "Round %s local epoch %s/%s: train_loss=%.6f",
                server_round,
                epoch + 1,
                self.local_epochs,
                train_loss,
            )

        # Test metrics are reported from evaluate(), which Flower
        # calls every round anyway; no need to compute them twice.
        train_metrics = self._evaluate_train()
        val_metrics = self._run_evaluation(self.val_loader)

        train_prefixed = _prefixed_metrics(train_metrics, "train_")
        train_prefixed.pop("train_loss", None)

        fit_metrics = {
            "train_loss": float(train_loss),
            **train_prefixed,
            **_prefixed_metrics(val_metrics, "val_"),
        }
        self.logger.info(
            "Round %s local fit metrics: %s",
            server_round,
            fit_metrics,
        )

        return (
            self.get_parameters(),
            self.train_dataset.num_images,
            fit_metrics,
        )

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        server_round = config.get("server_round", "unknown")

        test_metrics = self._run_evaluation(self.test_loader)

        test_prefixed = _prefixed_metrics(test_metrics, "test_")
        test_prefixed.pop("test_loss", None)
        self.logger.info(
            "Round %s local test metrics: loss=%.6f metrics=%s",
            server_round,
            test_metrics["loss"],
            test_prefixed,
        )

        return (
            float(test_metrics["loss"]),
            self.test_dataset.num_images,
            test_prefixed,
        )
