from __future__ import annotations

import math
import random
from typing import Mapping, Sequence

import numpy as np
import torch


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")

    if (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        return torch.device("mps")

    return torch.device("cpu")


def cosine_learning_rate(
    epoch: int,
    total_epochs: int,
    initial_lr: float,
    min_lr: float,
) -> float:
    """Cosine decay from ``initial_lr`` at epoch 1 to ``min_lr`` at the last epoch."""
    if total_epochs <= 1:
        return float(initial_lr)
    progress = (epoch - 1) / (total_epochs - 1)
    progress = min(1.0, max(0.0, progress))
    return float(
        min_lr
        + 0.5 * (initial_lr - min_lr) * (1.0 + math.cos(math.pi * progress))
    )


def clone_state_dict(
    state_dict: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in state_dict.items()}


def average_state_dicts(
    snapshots: Sequence[Mapping[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    """Element-wise mean of several state dicts, keeping each tensor's dtype."""
    if not snapshots:
        raise ValueError("Nothing to average")

    averaged: dict[str, torch.Tensor] = {}
    for key in snapshots[0]:
        stacked = torch.stack(
            [snapshot[key].detach().cpu().float() for snapshot in snapshots]
        )
        averaged[key] = stacked.mean(dim=0).to(dtype=snapshots[0][key].dtype)
    return averaged


def load_state_dict(model: torch.nn.Module, state_dict: Mapping[str, torch.Tensor]) -> None:
    device = next(model.parameters()).device
    model.load_state_dict(
        {key: value.to(device) for key, value in state_dict.items()},
        strict=True,
    )
