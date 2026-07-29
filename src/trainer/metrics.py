from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)


@dataclass
class ClassificationMetrics:
    loss: float
    accuracy: float
    precision: float
    recall: float
    f1: float
    confusion_matrix: np.ndarray

    def to_dict(self):

        return {
            "loss": float(self.loss),
            "accuracy": float(self.accuracy),
            "precision": float(self.precision),
            "recall": float(self.recall),
            "f1": float(self.f1),
        }


class MetricAccumulator:

    def __init__(self):

        self.reset()

    ####################################################################
    # Update
    ####################################################################

    def update(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        loss: torch.Tensor,
    ):

        self.loss_sum += (
            loss.item() * targets.size(0)
        )

        self.num_samples += targets.size(0)

        preds = torch.argmax(
            logits,
            dim=1,
        )

        self.targets.extend(
            targets.detach().cpu().tolist()
        )

        self.predictions.extend(
            preds.detach().cpu().tolist()
        )

    ####################################################################
    # Compute
    ####################################################################

    def compute(self) -> ClassificationMetrics:

        if self.num_samples == 0:

            raise RuntimeError(
                "No samples accumulated."
            )

        precision, recall, f1, _ = (
            precision_recall_fscore_support(
                self.targets,
                self.predictions,
                average="macro",
                zero_division=0,
            )
        )

        return ClassificationMetrics(
            loss=self.loss_sum / self.num_samples,
            accuracy=accuracy_score(
                self.targets,
                self.predictions,
            ),
            precision=precision,
            recall=recall,
            f1=f1,
            confusion_matrix=confusion_matrix(
                self.targets,
                self.predictions,
            ),
        )

    ####################################################################
    # Reset
    ####################################################################

    def reset(self):

        self.loss_sum = 0.0
        self.num_samples = 0

        self.targets = []
        self.predictions = []


########################################################################
# Utilities
########################################################################

def merge_metrics(metrics_list):

    """
    Weighted average of metrics from several clients.
    """

    total = sum(
        n
        for _, n in metrics_list
    )

    result = {}

    for key in [
        "loss",
        "accuracy",
        "precision",
        "recall",
        "f1",
    ]:

        value = 0.0

        for metrics, n in metrics_list:

            value += metrics[key] * n

        result[key] = value / total

    return result