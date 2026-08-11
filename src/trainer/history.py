from __future__ import annotations

import csv
import json
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Dict

import torch
from flwr.common import Parameters, parameters_to_ndarrays
from flwr.server.history import History


def history_to_dict(history: History) -> Dict:
    return {
        "losses_distributed": history.losses_distributed,
        "losses_centralized": history.losses_centralized,
        "metrics_distributed_fit": history.metrics_distributed_fit,
        "metrics_distributed": history.metrics_distributed,
        "metrics_centralized": history.metrics_centralized,
    }


def save_history(history: History, save_dir: str | Path) -> None:
    """Dump the Flower History to history.json and a per-round
    wide-format metrics.csv inside save_dir."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    with open(save_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(history_to_dict(history), f, indent=2)

    rows: Dict[int, Dict[str, float]] = defaultdict(dict)

    # losses_distributed holds the loss returned by clients' evaluate(),
    # which in this project is computed on the test split.
    for rnd, loss in history.losses_distributed:
        rows[rnd]["test_loss"] = float(loss)

    for metrics in (
        history.metrics_distributed_fit,
        history.metrics_distributed,
    ):
        for name, series in metrics.items():
            for rnd, value in series:
                rows[rnd][name] = float(value)

    if not rows:
        return

    columns = ["round"] + sorted({key for row in rows.values() for key in row})

    with open(save_dir / "metrics.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for rnd in sorted(rows):
            writer.writerow({"round": rnd, **rows[rnd]})


def save_global_model(
    parameters: Parameters,
    model: torch.nn.Module,
    path: str | Path,
) -> None:
    """Load aggregated Flower parameters into the model and save its
    state_dict to path."""
    state_dict = OrderedDict(
        (key, torch.from_numpy(value))
        for key, value in zip(
            model.state_dict().keys(),
            parameters_to_ndarrays(parameters),
        )
    )
    model.load_state_dict(state_dict, strict=True)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
