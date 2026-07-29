from __future__ import annotations

from typing import Dict

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .aggregation import build_aggregator
from .metrics import MetricAccumulator


class LocalTrainer:

    def __init__(
        self,
        device: torch.device,
        num_classes: int,
        aggregation: str = "average_probability",
    ):

        self.device = device
        self.num_classes = num_classes
        self.aggregation = aggregation

    ####################################################################
    # Train
    ####################################################################

    def train_epoch(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        loader: DataLoader,
    ) -> Dict[str, float]:

        model.train()

        metrics = MetricAccumulator()

        progress = tqdm(loader, leave=False)

        for batch in progress:

            images = batch["image"].to(self.device)
            labels = batch["label"].to(self.device)

            optimizer.zero_grad(set_to_none=True)

            logits = model(images)

            loss = criterion(
                logits,
                labels,
            )

            loss.backward()

            optimizer.step()

            metrics.update(
                logits,
                labels,
                loss,
            )

            current = metrics.compute()

            progress.set_postfix(
                loss=f"{current.loss:.4f}",
                acc=f"{100*current.accuracy:.2f}",
            )

        return metrics.compute().to_dict()

    ####################################################################
    # Validation
    ####################################################################

    @torch.no_grad()
    def validate(
        self,
        model: nn.Module,
        criterion: nn.Module,
        loader: DataLoader,
    ):

        return self._evaluate(
            model,
            criterion,
            loader,
        )

    ####################################################################
    # Test
    ####################################################################

    @torch.no_grad()
    def test(
        self,
        model: nn.Module,
        criterion: nn.Module,
        loader: DataLoader,
    ):

        return self._evaluate(
            model,
            criterion,
            loader,
        )

    ####################################################################
    # Shared evaluation
    ####################################################################

    @torch.no_grad()
    def _evaluate(
        self,
        model,
        criterion,
        loader,
    ):

        model.eval()

        patch_metrics = MetricAccumulator()

        aggregator = build_aggregator(
            self.aggregation,
            self.num_classes,
        )

        for batch in tqdm(loader, leave=False):

            images = batch["image"].to(self.device)
            labels = batch["label"].to(self.device)

            logits = model(images)

            loss = criterion(
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

        patch = patch_metrics.compute()

        image = aggregator.compute()

        return {
            "loss": patch.loss,
            "patch_accuracy": patch.accuracy,
            "patch_precision": patch.precision,
            "patch_recall": patch.recall,
            "patch_f1": patch.f1,
            "accuracy": image["accuracy"],
            "precision": image["precision"],
            "recall": image["recall"],
            "f1": image["f1"],
            "confusion_matrix": image["confusion_matrix"],
        }