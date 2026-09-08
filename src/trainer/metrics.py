from __future__ import annotations

from typing import Dict, List

import torch

from .aggregation import (
    build_aggregator,
    build_confusion_matrix,
    classification_summary,
    serialize_confusion_matrix,
)


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


@torch.no_grad()
def evaluate(
    model,
    dataloader,
    criterion,
    device,
    num_classes: int,
    aggregation: str = "average_probability",
) -> Dict[str, float | str]:
    model.eval()

    loss_meter = AverageMeter()
    patch_targets: List[int] = []
    patch_predictions: List[int] = []
    image_aggregator = build_aggregator(aggregation, num_classes)

    for batch in dataloader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        image_ids = batch["image_id"]

        logits = model(images)
        loss = criterion(logits, labels)

        loss_meter.update(loss.item(), images.size(0))
        patch_targets.extend(labels.cpu().tolist())
        patch_predictions.extend(
            torch.argmax(logits, dim=1).cpu().tolist()
        )
        image_aggregator.update(image_ids, logits, labels)

    patch = classification_summary(patch_targets, patch_predictions)
    patch_confusion = build_confusion_matrix(
        patch_targets,
        patch_predictions,
        num_classes,
    )
    image_targets, image_predictions = image_aggregator.aggregate()
    image = classification_summary(image_targets, image_predictions)
    image_confusion = build_confusion_matrix(
        image_targets,
        image_predictions,
        num_classes,
    )

    # Primary metrics are patch-level; image-level metrics are
    # kept under the "image_" prefix for reference.
    return {
        "loss": loss_meter.avg,
        "accuracy": patch["accuracy"],
        "precision": patch["precision"],
        "recall": patch["recall"],
        "f1": patch["f1"],
        "confusion_matrix": serialize_confusion_matrix(patch_confusion),
        "image_accuracy": image["accuracy"],
        "image_precision": image["precision"],
        "image_recall": image["recall"],
        "image_f1": image["f1"],
        "image_confusion_matrix": serialize_confusion_matrix(
            image_confusion
        ),
    }
