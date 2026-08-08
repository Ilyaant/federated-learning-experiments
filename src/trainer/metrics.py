from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch

from .aggregation import build_aggregator


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.value = 0.0
        self.sum = 0.0
        self.count = 0
        self.avg = 0.0

    def update(self, value: float, n: int = 1):
        self.value = value
        self.sum += value * n
        self.count += n
        self.avg = self.sum / max(1, self.count)


class ClassificationMetrics:
    def __init__(self, num_classes: int):
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        self.targets: List[int] = []
        self.predictions: List[int] = []

    @torch.no_grad()
    def update(self, logits: torch.Tensor, targets: torch.Tensor):
        preds = torch.argmax(logits, dim=1)
        self.targets.extend(targets.cpu().tolist())
        self.predictions.extend(preds.cpu().tolist())

    def confusion_matrix(self):
        cm = np.zeros(
            (self.num_classes, self.num_classes),
            dtype=np.int64,
        )
        for target, pred in zip(self.targets, self.predictions):
            cm[target, pred] += 1
        return cm

    def accuracy(self) -> float:
        if not self.targets:
            return 0.0
        return float(
            np.mean(
                np.asarray(self.targets) == np.asarray(self.predictions)
            )
        )

    def precision(self) -> float:
        cm = self.confusion_matrix()
        scores = []
        for cls in range(self.num_classes):
            tp = cm[cls, cls]
            fp = cm[:, cls].sum() - tp
            scores.append(tp / (tp + fp) if tp + fp > 0 else 0.0)
        return float(np.mean(scores))

    def recall(self) -> float:
        cm = self.confusion_matrix()
        scores = []
        for cls in range(self.num_classes):
            tp = cm[cls, cls]
            fn = cm[cls, :].sum() - tp
            scores.append(tp / (tp + fn) if tp + fn > 0 else 0.0)
        return float(np.mean(scores))

    def f1(self) -> float:
        p, r = self.precision(), self.recall()
        return 0.0 if p + r == 0 else 2 * p * r / (p + r)

    def summary(self) -> Dict[str, float]:
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
    num_classes: int,
    aggregation: str = "average_probability",
) -> Dict[str, float]:
    model.eval()

    loss_meter = AverageMeter()
    patch_metrics = ClassificationMetrics(num_classes)
    image_aggregator = build_aggregator(aggregation, num_classes)

    for batch in dataloader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        image_ids = batch["image_id"]

        logits = model(images)
        loss = criterion(logits, labels)

        loss_meter.update(loss.item(), images.size(0))
        patch_metrics.update(logits, labels)
        image_aggregator.update(image_ids, logits, labels)

    patch = patch_metrics.summary()
    image = image_aggregator.compute()

    return {
        "loss": loss_meter.avg,
        "patch_accuracy": patch["accuracy"],
        "patch_precision": patch["precision"],
        "patch_recall": patch["recall"],
        "patch_f1": patch["f1"],
        "accuracy": image["accuracy"],
        "precision": image["precision"],
        "recall": image["recall"],
        "f1": image["f1"],
    }
