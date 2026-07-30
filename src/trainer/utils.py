from __future__ import annotations

import random
from collections import OrderedDict
from pathlib import Path
from typing import Iterable, List

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


def get_parameters(model) -> List[np.ndarray]:
    return [
        parameter.detach().cpu().numpy()
        for parameter in model.state_dict().values()
    ]


def set_parameters(
    model,
    parameters: Iterable[np.ndarray],
):
    state_dict = OrderedDict()

    for key, value in zip(
        model.state_dict().keys(),
        parameters,
    ):
        state_dict[key] = torch.from_numpy(value)

    model.load_state_dict(
        state_dict,
        strict=True,
    )


def count_parameters(model):
    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )


def save_checkpoint(
    model,
    optimizer,
    epoch: int,
    path: str | Path,
    metrics=None,
):
    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict()
            if optimizer is not None
            else None,
            "metrics": metrics,
        },
        path,
    )


def load_checkpoint(
    model,
    optimizer,
    path: str | Path,
    device="cpu",
):
    checkpoint = torch.load(
        path,
        map_location=device,
    )

    model.load_state_dict(
        checkpoint["model"]
    )

    if (
        optimizer is not None
        and checkpoint["optimizer"] is not None
    ):
        optimizer.load_state_dict(
            checkpoint["optimizer"]
        )

    return checkpoint


class EarlyStopping:
    def __init__(
        self,
        patience: int = 10,
        mode: str = "max",
        min_delta: float = 0.0,
    ):
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta

        self.best = None
        self.counter = 0

    def step(
        self,
        value: float,
    ):
        if self.best is None:
            self.best = value
            return False

        if self.mode == "max":
            improved = (
                value > self.best + self.min_delta
            )
        else:
            improved = (
                value < self.best - self.min_delta
            )

        if improved:
            self.best = value
            self.counter = 0
        else:
            self.counter += 1

        return self.counter >= self.patience


class MetricTracker:
    def __init__(self):
        self.history = {}

    def update(
        self,
        metrics: dict,
    ):
        for key, value in metrics.items():
            self.history.setdefault(
                key,
                [],
            ).append(float(value))

    def latest(self):
        return {
            key: values[-1]
            for key, values in self.history.items()
        }

    def best(
        self,
        key: str,
        mode: str = "max",
    ):
        if key not in self.history:
            return None

        if mode == "max":
            return max(self.history[key])

        return min(self.history[key])

    def state_dict(self):
        return self.history

    def load_state_dict(
        self,
        state,
    ):
        self.history = state
