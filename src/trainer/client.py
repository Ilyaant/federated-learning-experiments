from __future__ import annotations

from collections import OrderedDict
from typing import Dict, List

import flwr as fl
import numpy as np
import torch
from torch.utils.data import DataLoader

from .metrics import AverageMeter, evaluate
from .utils import get_device


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
    ):
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

        loader_kwargs = {
            "batch_size": batch_size,
            "num_workers": num_workers,
            "pin_memory": torch.cuda.is_available(),
            "persistent_workers": num_workers > 0,
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

        train_loss = 0.0
        for epoch in range(self.local_epochs):
            train_loss = self.train_one_epoch(epoch)

        train_metrics = self._evaluate_train()
        val_metrics = self._run_evaluation(self.val_loader)
        test_metrics = self._run_evaluation(self.test_loader)

        train_prefixed = _prefixed_metrics(train_metrics, "train_")
        train_prefixed.pop("train_loss", None)

        fit_metrics = {
            "train_loss": float(train_loss),
            **train_prefixed,
            **_prefixed_metrics(val_metrics, ""),
            **_prefixed_metrics(test_metrics, "test_"),
        }

        return (
            self.get_parameters(),
            self.train_dataset.num_images,
            fit_metrics,
        )

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)

        val_metrics = self._run_evaluation(self.val_loader)
        test_metrics = self._run_evaluation(self.test_loader)

        test_prefixed = _prefixed_metrics(test_metrics, "")
        test_prefixed.pop("loss", None)

        return (
            float(test_metrics["loss"]),
            self.test_dataset.num_images,
            {
                **test_prefixed,
                **_prefixed_metrics(val_metrics, "val_"),
            },
        )
