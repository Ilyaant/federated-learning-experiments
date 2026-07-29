from __future__ import annotations

from typing import Dict, Optional

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .aggregation import build_aggregator
from .metrics import MetricAccumulator


class LocalTrainer:

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: torch.device,
        num_classes: int,
        aggregation: str = "majority_vote",
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    ):

        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.device = device

        self.num_classes = num_classes
        self.aggregation = aggregation

    ####################################################################
    # TRAIN
    ####################################################################

    def fit(
        self,
        train_loader: DataLoader,
        epochs: int,
    ) -> Dict[str, float]:

        train_dataset = train_loader.dataset

        epoch_metrics = None

        for epoch in range(epochs):

            if hasattr(train_dataset, "set_epoch"):
                train_dataset.set_epoch(epoch)

            epoch_metrics = self._train_epoch(train_loader)

            if self.scheduler is not None:
                self.scheduler.step()

        return epoch_metrics

    ####################################################################
    # ONE TRAIN EPOCH
    ####################################################################

    def _train_epoch(
        self,
        loader: DataLoader,
    ):

        self.model.train()

        metrics = MetricAccumulator()

        progress = tqdm(
            loader,
            leave=False,
        )

        for batch in progress:

            images = batch["image"].to(self.device)
            labels = batch["label"].to(self.device)

            self.optimizer.zero_grad()

            logits = self.model(images)

            loss = self.criterion(
                logits,
                labels,
            )

            loss.backward()

            self.optimizer.step()

            metrics.update(
                logits,
                labels,
                loss,
            )

            result = metrics.compute()

            progress.set_postfix(
                loss=f"{result.loss:.4f}",
                acc=f"{100*result.accuracy:.2f}",
            )

        return metrics.compute().to_dict()

    ####################################################################
    # VALIDATION / TEST
    ####################################################################

    @torch.no_grad()
    def evaluate(
        self,
        loader: DataLoader,
    ):

        self.model.eval()

        patch_metrics = MetricAccumulator()

        aggregator = build_aggregator(
            self.aggregation,
            self.num_classes,
        )

        for batch in tqdm(
            loader,
            leave=False,
        ):

            images = batch["image"].to(self.device)
            labels = batch["label"].to(self.device)

            logits = self.model(images)

            loss = self.criterion(
                logits,
                labels,
            )

            patch_metrics.update(
                logits,
                labels,
                loss,
            )

            aggregator.update(
                batch["image_id"],
                logits,
                labels,
            )

        patch_result = patch_metrics.compute()

        image_result = aggregator.compute()

        return {
            "loss": patch_result.loss,
            "patch_accuracy": patch_result.accuracy,
            "patch_precision": patch_result.precision,
            "patch_recall": patch_result.recall,
            "patch_f1": patch_result.f1,
            "accuracy": image_result["accuracy"],
            "precision": image_result["precision"],
            "recall": image_result["recall"],
            "f1": image_result["f1"],
            "confusion_matrix": image_result["confusion_matrix"],
        }
