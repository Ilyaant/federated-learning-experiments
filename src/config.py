from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, Iterable, List

import yaml

Config = Dict[str, Any]


def load_config(path: str | Path) -> Config:
    with open(path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Config root must be a mapping, got {type(config)}")
    return config


def save_config(config: Config, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        yaml.safe_dump(
            config,
            file,
            sort_keys=False,
            allow_unicode=True,
            default_flow_style=False,
        )


def get_path(config: Config, dotted: str, default: Any = None) -> Any:
    node: Any = config
    for key in dotted.split("."):
        if not isinstance(node, dict) or key not in node:
            return default
        node = node[key]
    return node


def set_path(config: Config, dotted: str, value: Any) -> None:
    keys = dotted.split(".")
    node = config
    for key in keys[:-1]:
        if key not in node or not isinstance(node[key], dict):
            node[key] = {}
        node = node[key]
    node[keys[-1]] = value


def _parse_value(raw: str) -> Any:
    """Parse ``key=value`` right-hand side with YAML rules (1e-4, true, null, [1,2])."""
    try:
        return yaml.safe_load(raw)
    except yaml.YAMLError:
        return raw


def apply_overrides(config: Config, overrides: Iterable[str] | None) -> Config:
    result = copy.deepcopy(config)
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Override must look like key.path=value, got {item!r}")
        key, raw = item.split("=", 1)
        set_path(result, key.strip(), _parse_value(raw.strip()))
    return result


def validate_config(config: Config) -> List[str]:
    """Return a list of human-readable problems (empty list = OK)."""
    problems: List[str] = []

    for split_key in ("train_overlap", "eval_overlap"):
        overlap = get_path(config, f"patches.{split_key}", 0.0)
        if not 0 <= float(overlap) < 1:
            problems.append(f"patches.{split_key} must satisfy 0 <= overlap < 1")

    if int(get_path(config, "patches.size", 224)) <= 0:
        problems.append("patches.size must be positive")

    ratios = [
        float(get_path(config, f"dataset.{name}", 0.0))
        for name in ("train_ratio", "val_ratio", "test_ratio")
    ]
    if get_path(config, "dataset.regroup", True) and abs(sum(ratios) - 1.0) > 1e-6:
        problems.append("dataset.train_ratio + val_ratio + test_ratio must equal 1")

    aggregation = get_path(config, "evaluation.aggregation", "mean_prob")
    if aggregation not in {"mean_prob", "majority_vote"}:
        problems.append("evaluation.aggregation must be mean_prob or majority_vote")

    selection = get_path(config, "evaluation.selection_metric", "val_patch_f1")
    allowed = {
        "val_patch_f1",
        "val_patch_accuracy",
        "val_image_f1",
        "val_image_accuracy",
        "val_loss",
    }
    if selection not in allowed:
        problems.append(f"evaluation.selection_metric must be one of {sorted(allowed)}")

    pad_mode = get_path(config, "patches.pad_mode", "reflect")
    if pad_mode not in {"reflect", "replicate"}:
        problems.append("patches.pad_mode must be reflect or replicate")

    norm_mode = get_path(config, "patches.normalize.mode", "fixed")
    allowed_norm_modes = {"fixed", "patch_mean", "patch_median", "patch_zscore"}
    if norm_mode not in allowed_norm_modes:
        problems.append(f"patches.normalize.mode must be one of {sorted(allowed_norm_modes)}")

    contrast_prob = float(get_path(config, "augmentation.contrast.prob", 0.0) or 0.0)
    if contrast_prob > 0:
        scale = get_path(config, "augmentation.contrast.scale", [0.8, 1.25])
        if (
            not isinstance(scale, (list, tuple))
            or len(scale) != 2
            or float(scale[0]) <= 0
            or float(scale[0]) > float(scale[1])
        ):
            problems.append("augmentation.contrast.scale must be [min_scale, max_scale] with 0 < min <= max")

    weights = get_path(config, "model.weights")
    if weights not in (None, ""):
        path = Path(str(weights))
        if not path.is_file():
            problems.append(f"model.weights file not found: {path}")

    return problems
