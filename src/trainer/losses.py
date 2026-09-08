from __future__ import annotations

from typing import Mapping

import torch


def compute_class_weights(
    class_counts: Mapping[int, int],
    num_classes: int,
    power: float = 0.5,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Build normalized, softened inverse-frequency class weights.

    ``power=1`` is standard inverse frequency, ``power=0.5`` is inverse
    square root, and ``power=0`` disables frequency reweighting.
    """
    if num_classes <= 0:
        raise ValueError("num_classes must be positive")
    if not 0.0 <= power <= 1.0:
        raise ValueError("class weight power must be between 0 and 1")

    counts = torch.tensor(
        [class_counts.get(class_id, 0) for class_id in range(num_classes)],
        dtype=torch.float32,
        device=device,
    )
    if torch.any(counts <= 0):
        missing = [
            class_id
            for class_id, count in enumerate(counts.tolist())
            if count <= 0
        ]
        raise ValueError(f"Missing training samples for classes: {missing}")

    total = counts.sum()
    inverse_frequency = total / (num_classes * counts)
    weights = inverse_frequency.pow(power)
    return weights / weights.mean()


__all__ = ["compute_class_weights"]
