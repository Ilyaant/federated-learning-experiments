from __future__ import annotations

import csv
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence

LOGGER_NAME = "texture"


def setup_logger(log_path: str | Path, level: int = logging.INFO) -> logging.Logger:
    """Logger writing the same records to ``log_path`` and stdout."""
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(level)
    logger.propagate = False
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def get_logger() -> logging.Logger:
    return logging.getLogger(LOGGER_NAME)


class CsvLogger:
    """Append-only CSV with a fixed header; missing keys are written empty."""

    def __init__(self, path: str | Path, fieldnames: Sequence[str]):
        self.path = Path(path)
        self.fieldnames = list(fieldnames)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w", newline="", encoding="utf-8") as file:
            csv.DictWriter(file, fieldnames=self.fieldnames).writeheader()

    def write(self, row: Dict[str, Any]) -> None:
        with open(self.path, "a", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=self.fieldnames, extrasaction="ignore")
            writer.writerow({key: _format(row.get(key)) for key in self.fieldnames})

    def write_many(self, rows: Iterable[Dict[str, Any]]) -> None:
        for row in rows:
            self.write(row)


def _format(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    if isinstance(value, bool):
        return int(value)
    return value


def save_json(data: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False, default=_json_default)


def _json_default(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return str(value)


def format_metrics(metrics: Dict[str, Any], keys: Optional[Sequence[str]] = None) -> str:
    keys = keys or ("accuracy", "precision", "recall", "f1")
    return " ".join(f"{key}={metrics[key]:.4f}" for key in keys if key in metrics)
