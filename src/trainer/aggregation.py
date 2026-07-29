from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Dict, List

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)


class PatchAggregator(ABC):
    """
    Base class for aggregating patch predictions into
    image-level predictions.
    """

    def __init__(self, num_classes: int):

        self.num_classes = num_classes

        self.image_logits: Dict[int, List[np.ndarray]] = defaultdict(list)
        self.image_predictions: Dict[int, List[int]] = defaultdict(list)
        self.image_labels: Dict[int, int] = {}

    ##################################################################

    def update(
        self,
        image_ids,
        logits,
        labels,
    ):

        if torch.is_tensor(image_ids):
            image_ids = image_ids.cpu().numpy()

        if torch.is_tensor(labels):
            labels = labels.cpu().numpy()

        if torch.is_tensor(logits):
            logits = logits.detach().cpu().numpy()

        preds = np.argmax(logits, axis=1)

        for image_id, logit, pred, label in zip(
            image_ids,
            logits,
            preds,
            labels,
        ):

            self.image_logits[int(image_id)].append(logit)
            self.image_predictions[int(image_id)].append(int(pred))
            self.image_labels[int(image_id)] = int(label)

    ##################################################################

    @abstractmethod
    def aggregate(self):
        pass

    ##################################################################

    def compute(self):

        y_true, y_pred = self.aggregate()

        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        )

        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "confusion_matrix": confusion_matrix(
                y_true,
                y_pred,
            ),
            "y_true": y_true,
            "y_pred": y_pred,
        }

    ##################################################################

    def reset(self):

        self.image_logits.clear()
        self.image_predictions.clear()
        self.image_labels.clear()


########################################################################
# Majority Vote
########################################################################

class MajorityVoteAggregator(PatchAggregator):

    def aggregate(self):

        y_true = []
        y_pred = []

        for image_id in sorted(self.image_labels):

            preds = np.asarray(
                self.image_predictions[image_id]
            )

            vote = np.bincount(
                preds,
                minlength=self.num_classes,
            ).argmax()

            y_true.append(
                self.image_labels[image_id]
            )

            y_pred.append(
                int(vote)
            )

        return np.asarray(y_true), np.asarray(y_pred)


########################################################################
# Average Probability
########################################################################

class AverageProbabilityAggregator(PatchAggregator):

    def aggregate(self):

        y_true = []
        y_pred = []

        for image_id in sorted(self.image_labels):

            logits = np.stack(
                self.image_logits[image_id],
                axis=0,
            )

            logits = logits - logits.max(
                axis=1,
                keepdims=True,
            )

            exp = np.exp(logits)

            probs = exp / exp.sum(
                axis=1,
                keepdims=True,
            )

            mean_prob = probs.mean(axis=0)

            prediction = int(
                np.argmax(mean_prob)
            )

            y_true.append(
                self.image_labels[image_id]
            )

            y_pred.append(
                prediction
            )

        return np.asarray(y_true), np.asarray(y_pred)


########################################################################
# Maximum Probability
########################################################################

class MaxProbabilityAggregator(PatchAggregator):

    def aggregate(self):

        y_true = []
        y_pred = []

        for image_id in sorted(self.image_labels):

            logits = np.stack(
                self.image_logits[image_id],
                axis=0,
            )

            logits = logits - logits.max(
                axis=1,
                keepdims=True,
            )

            exp = np.exp(logits)

            probs = exp / exp.sum(
                axis=1,
                keepdims=True,
            )

            idx = np.argmax(
                probs.max(axis=1)
            )

            prediction = int(
                np.argmax(
                    probs[idx]
                )
            )

            y_true.append(
                self.image_labels[image_id]
            )

            y_pred.append(
                prediction
            )

        return np.asarray(y_true), np.asarray(y_pred)


########################################################################
# Factory
########################################################################

def build_aggregator(
    method: str,
    num_classes: int,
):

    method = method.lower()

    if method in [
        "majority",
        "majority_vote",
        "vote",
    ]:
        return MajorityVoteAggregator(num_classes)

    if method in [
        "average",
        "average_probability",
        "mean_probability",
    ]:
        return AverageProbabilityAggregator(num_classes)

    if method in [
        "max",
        "max_probability",
    ]:
        return MaxProbabilityAggregator(num_classes)

    raise ValueError(
        f"Unknown aggregation method: {method}"
    )
