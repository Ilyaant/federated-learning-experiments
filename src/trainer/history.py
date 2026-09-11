from __future__ import annotations

import csv
import json
import logging
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Mapping

import torch


logger = logging.getLogger(__name__)


def configure_file_logging(
    save_dir: str | Path,
    filename: str = "experiment.log",
) -> Path:
    """Write application messages to a file immediately."""
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

    has_console = any(
        type(handler) is logging.StreamHandler for handler in root_logger.handlers
    )
    if not has_console:
        stream = logging.StreamHandler()
        stream.setLevel(logging.INFO)
        stream.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
        )
        root_logger.addHandler(stream)

    logger.info("File logging initialized: %s", log_path)
    return log_path


def reset_file_logging() -> None:
    """Drop file handlers so consecutive runs in one process do not mix logs."""
    root_logger = logging.getLogger()
    for handler in list(root_logger.handlers):
        if isinstance(handler, logging.FileHandler):
            handler.close()
            root_logger.removeHandler(handler)


class LiveHistoryWriter:
    """Persist per-epoch metrics while training is running."""

    def __init__(self, save_dir: str | Path):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.metrics: Dict[int, Dict[str, float]] = defaultdict(dict)
        self.extras: Dict[int, Dict[str, object]] = defaultdict(dict)
        self.best: Dict[str, float | int] | None = None
        self.summary: Dict[str, object] | None = None
        self.flush()

    def update(self, epoch: int, metrics: Mapping) -> None:
        for name, value in metrics.items():
            if isinstance(value, bool):
                self.metrics[epoch][str(name)] = float(value)
            elif isinstance(value, (int, float)):
                self.metrics[epoch][str(name)] = float(value)
            else:
                self.extras[epoch][str(name)] = value
        self.flush()
        logger.info(
            "Epoch %s metrics saved at %s: %s",
            epoch,
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            dict(self.metrics[epoch]),
        )

    def set_best(self, epoch: int, score: float, metric: str = "val_f1") -> None:
        self.best = {"epoch": int(epoch), "score": float(score), "metric": metric}
        self.flush()

    def set_summary(self, summary: Mapping) -> None:
        self.summary = dict(summary)
        self.flush()

    def _history_dict(self) -> Dict:
        series: Dict[str, list[tuple[int, float]]] = defaultdict(list)
        for epoch in sorted(self.metrics):
            for name, value in self.metrics[epoch].items():
                series[name].append([epoch, value])
        payload = {"metrics": dict(series)}
        if self.extras:
            payload["extras"] = {
                str(epoch): extras for epoch, extras in sorted(self.extras.items())
            }
        if self.best is not None:
            payload["best"] = self.best
        if self.summary is not None:
            payload["summary"] = self.summary
        return payload

    def flush(self) -> None:
        """Atomically refresh snapshots so readers never see partial files."""
        history_path = self.save_dir / "history.json"
        history_tmp = history_path.with_suffix(".json.tmp")
        with open(history_tmp, "w", encoding="utf-8") as file:
            json.dump(self._history_dict(), file, indent=2)
        history_tmp.replace(history_path)

        metrics_path = self.save_dir / "metrics.csv"
        metrics_tmp = metrics_path.with_suffix(".csv.tmp")
        columns = ["epoch"] + sorted(
            {name for row in self.metrics.values() for name in row}
        )
        with open(metrics_tmp, "w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=columns)
            writer.writeheader()
            for epoch in sorted(self.metrics):
                writer.writerow({"epoch": epoch, **self.metrics[epoch]})
        metrics_tmp.replace(metrics_path)

        if self.summary is not None:
            summary_path = self.save_dir / "summary.json"
            summary_tmp = summary_path.with_suffix(".json.tmp")
            with open(summary_tmp, "w", encoding="utf-8") as file:
                json.dump(self.summary, file, indent=2, ensure_ascii=False)
            summary_tmp.replace(summary_path)


def save_model(model: torch.nn.Module, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
