from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Dict, List

import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def classification_summary(
    y_true,
    y_pred,
) -> Dict[str, float]:
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def _softmax(logits: np.ndarray) -> np.ndarray:
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=1, keepdims=True)


class PatchAggregator(ABC):
    """Aggregates patch predictions into image-level predictions."""

    def __init__(self, num_classes: int):
        self.num_classes = num_classes
        self.image_probs: Dict[int, List[np.ndarray]] = defaultdict(list)
        self.image_labels: Dict[int, int] = {}

    def update(self, image_ids, logits, labels):
        if torch.is_tensor(image_ids):
            image_ids = image_ids.cpu().numpy()
        if torch.is_tensor(labels):
            labels = labels.cpu().numpy()
        if torch.is_tensor(logits):
            logits = logits.detach().cpu().numpy()

        probs = _softmax(np.asarray(logits))

        for image_id, prob, label in zip(image_ids, probs, labels):
            self.image_probs[int(image_id)].append(prob)
            self.image_labels[int(image_id)] = int(label)

    @abstractmethod
    def _predict(self, probs: np.ndarray) -> int:
        """Predict image class from an array of patch probabilities."""

    def aggregate(self):
        y_true = []
        y_pred = []

        for image_id in sorted(self.image_labels):
            probs = np.stack(self.image_probs[image_id], axis=0)
            y_true.append(self.image_labels[image_id])
            y_pred.append(self._predict(probs))

        return np.asarray(y_true), np.asarray(y_pred)

    def compute(self) -> Dict[str, float]:
        y_true, y_pred = self.aggregate()
        return classification_summary(y_true, y_pred)

    def reset(self):
        self.image_probs.clear()
        self.image_labels.clear()


class MajorityVoteAggregator(PatchAggregator):
    def _predict(self, probs: np.ndarray) -> int:
        votes = np.bincount(
            probs.argmax(axis=1),
            minlength=self.num_classes,
        )
        return int(votes.argmax())


class AverageProbabilityAggregator(PatchAggregator):
    def _predict(self, probs: np.ndarray) -> int:
        return int(probs.mean(axis=0).argmax())


class MaxProbabilityAggregator(PatchAggregator):
    def _predict(self, probs: np.ndarray) -> int:
        most_confident = probs.max(axis=1).argmax()
        return int(probs[most_confident].argmax())


_AGGREGATORS = {
    "majority": MajorityVoteAggregator,
    "majority_vote": MajorityVoteAggregator,
    "vote": MajorityVoteAggregator,
    "average": AverageProbabilityAggregator,
    "average_probability": AverageProbabilityAggregator,
    "mean_probability": AverageProbabilityAggregator,
    "max": MaxProbabilityAggregator,
    "max_probability": MaxProbabilityAggregator,
}


def build_aggregator(method: str, num_classes: int) -> PatchAggregator:
    try:
        return _AGGREGATORS[method.lower()](num_classes)
    except KeyError:
        raise ValueError(f"Unknown aggregation method: {method}") from None
