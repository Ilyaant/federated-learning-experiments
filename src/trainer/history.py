from __future__ import annotations

import csv
import json
import logging
from collections import OrderedDict, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Mapping

import torch
from flwr.common.logger import configure as configure_flower_logging
from flwr.common import Parameters, parameters_to_ndarrays
from flwr.server.history import History


logger = logging.getLogger(__name__)


def configure_file_logging(
    save_dir: str | Path,
    filename: str = "experiment.log",
    identifier: str = "federated-texture",
) -> Path:
    """Write application and Flower messages to a file immediately."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    log_path = (save_dir / filename).resolve()

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    if not any(
        isinstance(handler, logging.FileHandler)
        and Path(handler.baseFilename).resolve() == log_path
        for handler in root_logger.handlers
    ):
        handler = logging.FileHandler(log_path, encoding="utf-8")
        handler.setLevel(logging.INFO)
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
            )
        )
        root_logger.addHandler(handler)

    configure_flower_logging(identifier=identifier, filename=str(log_path))
    logger.info("File logging initialized: %s", log_path)
    return log_path


class LiveHistoryWriter:
    """Persist aggregated federated metrics while training is running."""

    def __init__(self, save_dir: str | Path):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.fit_metrics: Dict[int, Dict[str, float]] = defaultdict(dict)
        self.evaluate_metrics: Dict[int, Dict[str, float]] = defaultdict(dict)
        self.distributed_losses: Dict[int, float] = {}
        self.flush()

    @staticmethod
    def _as_float_metrics(metrics: Mapping) -> Dict[str, float]:
        return {str(name): float(value) for name, value in metrics.items()}

    def update_fit(self, server_round: int, metrics: Mapping) -> None:
        self.fit_metrics[server_round].update(self._as_float_metrics(metrics))
        self.flush()
        logger.info(
            "Round %s fit metrics saved at %s: %s",
            server_round,
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            dict(self.fit_metrics[server_round]),
        )

    def update_evaluate(
        self,
        server_round: int,
        loss: float | None,
        metrics: Mapping,
    ) -> None:
        if loss is not None:
            self.distributed_losses[server_round] = float(loss)
        self.evaluate_metrics[server_round].update(
            self._as_float_metrics(metrics)
        )
        self.flush()
        logger.info(
            "Round %s evaluation metrics saved at %s: loss=%s metrics=%s",
            server_round,
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            loss,
            dict(self.evaluate_metrics[server_round]),
        )

    @staticmethod
    def _metric_series(
        rows: Mapping[int, Mapping[str, float]],
    ) -> Dict[str, list[tuple[int, float]]]:
        series: Dict[str, list[tuple[int, float]]] = defaultdict(list)
        for server_round in sorted(rows):
            for name, value in rows[server_round].items():
                series[name].append((server_round, value))
        return dict(series)

    def _history_dict(self) -> Dict:
        return {
            "losses_distributed": sorted(self.distributed_losses.items()),
            "losses_centralized": [],
            "metrics_distributed_fit": self._metric_series(self.fit_metrics),
            "metrics_distributed": self._metric_series(
                self.evaluate_metrics
            ),
            "metrics_centralized": {},
        }

    def _rows(self) -> Dict[int, Dict[str, float]]:
        rows: Dict[int, Dict[str, float]] = defaultdict(dict)
        for server_round, metrics in self.fit_metrics.items():
            rows[server_round].update(metrics)
        for server_round, metrics in self.evaluate_metrics.items():
            rows[server_round].update(metrics)
        for server_round, loss in self.distributed_losses.items():
            rows[server_round]["test_loss"] = loss
        return rows

    def flush(self) -> None:
        """Atomically refresh snapshots so readers never see partial files."""
        history_path = self.save_dir / "history.json"
        history_tmp = history_path.with_suffix(".json.tmp")
        with open(history_tmp, "w", encoding="utf-8") as file:
            json.dump(self._history_dict(), file, indent=2)
        history_tmp.replace(history_path)

        rows = self._rows()
        metrics_path = self.save_dir / "metrics.csv"
        metrics_tmp = metrics_path.with_suffix(".csv.tmp")
        columns = ["round"] + sorted(
            {name for metrics in rows.values() for name in metrics}
        )
        with open(metrics_tmp, "w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=columns)
            writer.writeheader()
            for server_round in sorted(rows):
                writer.writerow(
                    {"round": server_round, **rows[server_round]}
                )
        metrics_tmp.replace(metrics_path)


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
