from __future__ import annotations

import csv
import itertools
import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import yaml


logger = logging.getLogger(__name__)

_UNSAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")


def load_yaml(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as file:
        loaded = yaml.safe_load(file) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected a mapping in {path}")
    return loaded


def coerce_override_value(raw: str):
    try:
        value = yaml.safe_load(raw)
    except yaml.YAMLError:
        value = raw
    if isinstance(value, str):
        lowered = value.lower()
        if lowered == "true":
            return True
        if lowered == "false":
            return False
        try:
            return int(value)
        except ValueError:
            pass
        try:
            return float(value)
        except ValueError:
            pass
    return value


def parse_override(spec: str) -> tuple[str, Any]:
    if "=" not in spec:
        raise ValueError(f"Override must be key=value, got {spec!r}")
    key, raw = spec.split("=", 1)
    key = key.strip()
    if not key:
        raise ValueError(f"Override is missing a key: {spec!r}")
    return key, coerce_override_value(raw.strip())


def set_nested(cfg: dict, dotted_key: str, value) -> dict:
    keys = [part for part in dotted_key.split(".") if part]
    if not keys:
        raise ValueError("Empty override key")
    cursor = cfg
    for key in keys[:-1]:
        next_value = cursor.get(key)
        if not isinstance(next_value, dict):
            cursor[key] = {}
        cursor = cursor[key]
    cursor[keys[-1]] = value
    return cfg


def apply_dotted_overrides(cfg: dict, overrides: Mapping[str, Any] | None) -> dict:
    for key, value in (overrides or {}).items():
        if key in {"name", "run_name"}:
            continue
        set_nested(cfg, key, value)
    return cfg


def flatten_overrides(overrides: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        key: value
        for key, value in overrides.items()
        if key not in {"name", "run_name"}
    }


def sanitize_run_name(name: str) -> str:
    cleaned = _UNSAFE_NAME.sub("_", name).strip("._-")
    return cleaned or "run"


def name_from_overrides(overrides: Mapping[str, Any], index: int) -> str:
    explicit = overrides.get("name") or overrides.get("run_name")
    if explicit:
        return sanitize_run_name(str(explicit))
    parts = []
    for key, value in flatten_overrides(overrides).items():
        short = key.split(".")[-1]
        parts.append(f"{short}{value}")
    label = sanitize_run_name("_".join(str(part) for part in parts))
    return f"{index:02d}_{label}" if label != "run" else f"{index:02d}_run"


def resolve_save_dir(cfg: dict, run_name: str | None = None) -> Path:
    logging_cfg = cfg.setdefault("logging", {})
    base = Path(logging_cfg.get("save_dir", "logs/runs"))
    unique = bool(logging_cfg.get("unique_dir", True))
    name = run_name or logging_cfg.get("run_name")
    if unique:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        folder = f"{name}_{stamp}" if name else stamp
        return base / folder
    return base


def expand_grid(grid: Mapping[str, Sequence]) -> List[dict]:
    if not grid:
        return [{}]
    keys = list(grid)
    values = [list(grid[key]) for key in keys]
    combos = []
    for combo in itertools.product(*values):
        combos.append(dict(zip(keys, combo)))
    return combos


def load_sweep_jobs(sweep: Mapping[str, Any]) -> List[dict]:
    jobs: List[dict] = []
    if sweep.get("grid"):
        jobs.extend(expand_grid(sweep["grid"]))
    for run in sweep.get("runs") or []:
        if not isinstance(run, dict):
            raise ValueError("Each item in sweep.runs must be a mapping")
        jobs.append(dict(run))
    if not jobs:
        jobs.append({})
    named = []
    used = set()
    for index, job in enumerate(jobs, start=1):
        name = name_from_overrides(job, index)
        original = name
        suffix = 2
        while name in used:
            name = f"{original}_{suffix}"
            suffix += 1
        used.add(name)
        job = dict(job)
        job["name"] = name
        named.append(job)
    return named


def write_sweep_summary(rows: Sequence[Mapping[str, Any]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    columns: List[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    preferred = [
        "name",
        "save_dir",
        "best_score",
        "selection_metric",
        "best_epoch",
        "val_f1",
        "test_f1",
        "val_image_f1",
        "test_image_f1",
        "selected_weights",
        "stopped_epoch",
        "status",
    ]
    ordered = [col for col in preferred if col in columns]
    ordered.extend(col for col in columns if col not in ordered)

    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=ordered)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    ranking_path = path.with_name("ranking.json")
    ranked = sorted(
        (row for row in rows if row.get("best_score") is not None),
        key=lambda row: float(row["best_score"]),
        reverse=True,
    )
    payload = {
        "selection_metric": next(
            (row.get("selection_metric") for row in rows if row.get("selection_metric")),
            "val_f1",
        ),
        "best_run": ranked[0] if ranked else None,
        "ranking": ranked,
    }
    with open(ranking_path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)


def flatten_job_overrides(job: Mapping[str, Any]) -> Dict[str, str]:
    return {
        f"override.{key}": value if isinstance(value, str) else json.dumps(value)
        for key, value in flatten_overrides(job).items()
    }


def pick_best_run(rows: Iterable[Mapping[str, Any]]) -> dict | None:
    ranked = [
        dict(row)
        for row in rows
        if isinstance(row.get("best_score"), (int, float))
    ]
    if not ranked:
        return None
    return max(ranked, key=lambda row: float(row["best_score"]))
