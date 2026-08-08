from __future__ import annotations

from collections import OrderedDict
from typing import Dict, List

import flwr as fl
import numpy as np
import torch
from torch.utils.data import DataLoader

from .metrics import AverageMeter, evaluate
from .utils import get_device


class FlowerClient(fl.client.NumPyClient):
    def __init__(
        self,
        model,
        train_dataset,
        val_dataset,
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

    def fit(self, parameters, config):
        self.set_parameters(parameters)

        train_loss = 0.0
        for epoch in range(self.local_epochs):
            train_loss = self.train_one_epoch(epoch)

        metrics = evaluate(
            self.model,
            self.val_loader,
            self.criterion,
            self.device,
            num_classes=self.num_classes,
            aggregation=self.aggregation,
        )

        return (
            self.get_parameters(),
            self.train_dataset.num_images,
            {
                "train_loss": float(train_loss),
                "loss": float(metrics["loss"]),
                "patch_accuracy": float(metrics["patch_accuracy"]),
                "accuracy": float(metrics["accuracy"]),
                "precision": float(metrics["precision"]),
                "recall": float(metrics["recall"]),
                "f1": float(metrics["f1"]),
            },
        )

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)

        metrics = evaluate(
            self.model,
            self.val_loader,
            self.criterion,
            self.device,
            num_classes=self.num_classes,
            aggregation=self.aggregation,
        )

        return (
            float(metrics["loss"]),
            self.val_dataset.num_images,
            {
                "patch_accuracy": float(metrics["patch_accuracy"]),
                "accuracy": float(metrics["accuracy"]),
                "precision": float(metrics["precision"]),
                "recall": float(metrics["recall"]),
                "f1": float(metrics["f1"]),
            },
        )
