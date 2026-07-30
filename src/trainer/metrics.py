from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.value = 0.0
        self.sum = 0.0
        self.count = 0
        self.avg = 0.0

    def update(
        self,
        value: float,
        n: int = 1,
    ):
        self.value = value
        self.sum += value * n
        self.count += n
        self.avg = self.sum / max(1, self.count)


class ClassificationMetrics:
    def __init__(
        self,
        num_classes: int,
    ):
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        self.targets: List[int] = []
        self.predictions: List[int] = []

    @torch.no_grad()
    def update(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ):
        preds = torch.argmax(logits, dim=1)

        self.targets.extend(
            targets.cpu().tolist()
        )

        self.predictions.extend(
            preds.cpu().tolist()
        )

    def confusion_matrix(self):
        cm = np.zeros(
            (
                self.num_classes,
                self.num_classes,
            ),
            dtype=np.int64,
        )

        for target, pred in zip(
            self.targets,
            self.predictions,
        ):
            cm[target, pred] += 1

        return cm

    def accuracy(self):
        if len(self.targets) == 0:
            return 0.0

        return float(
            np.mean(
                np.asarray(self.targets)
                == np.asarray(self.predictions)
            )
        )

    def precision(self):
        cm = self.confusion_matrix()

        scores = []

        for cls in range(self.num_classes):
            tp = cm[cls, cls]
            fp = cm[:, cls].sum() - tp

            scores.append(
                tp / (tp + fp)
                if tp + fp > 0
                else 0.0
            )

        return float(np.mean(scores))

    def recall(self):
        cm = self.confusion_matrix()

        scores = []

        for cls in range(self.num_classes):
            tp = cm[cls, cls]
            fn = cm[cls, :].sum() - tp

            scores.append(
                tp / (tp + fn)
                if tp + fn > 0
                else 0.0
            )

        return float(np.mean(scores))

    def f1(self):
        p = self.precision()
        r = self.recall()

        if p + r == 0:
            return 0.0

        return 2 * p * r / (p + r)

    def per_class_accuracy(self):
        cm = self.confusion_matrix()

        scores = {}

        for cls in range(self.num_classes):
            total = cm[cls].sum()

            if total == 0:
                scores[cls] = 0.0
            else:
                scores[cls] = float(
                    cm[cls, cls] / total
                )

        return scores

    def summary(self):
        return {
            "accuracy": self.accuracy(),
            "precision": self.precision(),
            "recall": self.recall(),
            "f1": self.f1(),
        }


@torch.no_grad()
def evaluate(
    model,
    dataloader,
    criterion,
    device,
):
    model.eval()

    loss_meter = AverageMeter()

    num_classes = (
        model.num_classes
        if hasattr(model, "num_classes")
        else 5
    )

    metrics = ClassificationMetrics(
        num_classes=num_classes,
    )

    for images, labels in dataloader:
        images = images.to(
            device,
            non_blocking=True,
        )

        labels = labels.to(
            device,
            non_blocking=True,
        )

        outputs = model(images)

        loss = criterion(
            outputs,
            labels,
        )

        loss_meter.update(
            loss.item(),
            images.size(0),
        )

        metrics.update(
            outputs,
            labels,
        )

    result = metrics.summary()

    result["loss"] = loss_meter.avg
    result["per_class_accuracy"] = (
        metrics.per_class_accuracy()
    )

    return result
