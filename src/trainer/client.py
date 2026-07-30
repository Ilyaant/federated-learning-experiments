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
        device=None,
    ):
        self.device = device or get_device()

        self.model = model.to(self.device)

        self.optimizer = optimizer
        self.criterion = criterion

        self.local_epochs = local_epochs

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=num_workers > 0,
        )

        self.val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=num_workers > 0,
        )

    def get_parameters(self, config=None):
        return [
            value.detach().cpu().numpy()
            for value in self.model.state_dict().values()
        ]

    def set_parameters(
        self,
        parameters: List[np.ndarray],
    ):
        params_dict = zip(
            self.model.state_dict().keys(),
            parameters,
        )

        state_dict = OrderedDict()

        for key, value in params_dict:
            tensor = torch.from_numpy(value)
            state_dict[key] = tensor

        self.model.load_state_dict(
            state_dict,
            strict=True,
        )

    def train_one_epoch(self):
        self.model.train()

        loss_meter = AverageMeter()

        for images, labels in self.train_loader:
            images = images.to(
                self.device,
                non_blocking=True,
            )

            labels = labels.to(
                self.device,
                non_blocking=True,
            )

            self.optimizer.zero_grad(set_to_none=True)

            logits = self.model(images)

            loss = self.criterion(
                logits,
                labels,
            )

            loss.backward()

            self.optimizer.step()

            loss_meter.update(
                loss.item(),
                images.size(0),
            )

        return loss_meter.avg

    def fit(
        self,
        parameters,
        config,
    ):
        self.set_parameters(parameters)

        train_loss = 0.0

        for _ in range(self.local_epochs):
            train_loss = self.train_one_epoch()

        metrics = evaluate(
            self.model,
            self.val_loader,
            self.criterion,
            self.device,
        )

        return (
            self.get_parameters(),
            len(self.train_loader.dataset),
            {
                "train_loss": float(train_loss),
                "loss": float(metrics["loss"]),
                "accuracy": float(metrics["accuracy"]),
                "precision": float(metrics["precision"]),
                "recall": float(metrics["recall"]),
                "f1": float(metrics["f1"]),
            },
        )

    def evaluate(
        self,
        parameters,
        config,
    ):
        self.set_parameters(parameters)

        metrics = evaluate(
            self.model,
            self.val_loader,
            self.criterion,
            self.device,
        )

        return (
            float(metrics["loss"]),
            len(self.val_loader.dataset),
            {
                "accuracy": float(metrics["accuracy"]),
                "precision": float(metrics["precision"]),
                "recall": float(metrics["recall"]),
                "f1": float(metrics["f1"]),
            },
        )


def create_client(
    model,
    train_dataset,
    val_dataset,
    optimizer,
    criterion,
    batch_size=32,
    local_epochs=1,
    num_workers=4,
):
    return FlowerClient(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        optimizer=optimizer,
        criterion=criterion,
        batch_size=batch_size,
        local_epochs=local_epochs,
        num_workers=num_workers,
    )
