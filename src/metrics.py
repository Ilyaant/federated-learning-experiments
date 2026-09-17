from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

METRIC_NAMES = ("accuracy", "precision", "recall", "f1")


def classification_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    class_names: Sequence[str],
    average: str = "macro",
) -> Dict[str, object]:
    """accuracy / precision / recall / f1 plus per-class scores and a confusion matrix."""
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    labels = list(range(len(class_names)))

    if len(y_true) == 0:
        empty = {name: float("nan") for name in METRIC_NAMES}
        empty.update({"n": 0, "per_class": {}, "confusion_matrix": []})
        return empty

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=average, zero_division=0
    )
    per_p, per_r, per_f, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "n": int(len(y_true)),
        "per_class": {
            class_names[i]: {
                "precision": float(per_p[i]),
                "recall": float(per_r[i]),
                "f1": float(per_f[i]),
                "support": int(support[i]),
            }
            for i in labels
        },
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def aggregate_image_predictions(
    probs: np.ndarray,
    image_ids: Sequence[int],
    method: str = "mean_prob",
) -> Dict[int, np.ndarray]:
    """Combine patch probabilities ``[N, C]`` into one probability vector per image.

    ``mean_prob`` averages patch probabilities; ``majority_vote`` turns each
    patch into a one-hot vote and normalizes the vote counts.
    """
    probs = np.asarray(probs, dtype=np.float64)
    image_ids = np.asarray(image_ids)
    num_classes = probs.shape[1]

    aggregated: Dict[int, np.ndarray] = {}
    for image_id in np.unique(image_ids):
        rows = probs[image_ids == image_id]
        if method == "majority_vote":
            votes = np.bincount(rows.argmax(axis=1), minlength=num_classes).astype(np.float64)
            aggregated[int(image_id)] = votes / votes.sum()
        elif method == "mean_prob":
            aggregated[int(image_id)] = rows.mean(axis=0)
        else:
            raise ValueError(f"Unknown aggregation method: {method}")
    return aggregated


def format_confusion_matrix(matrix: List[List[int]], class_names: Sequence[str]) -> str:
    width = max(6, max(len(name) for name in class_names) + 1)
    header = " " * width + "".join(f"{name:>{width}}" for name in class_names)
    rows = [
        f"{name:>{width}}" + "".join(f"{value:>{width}d}" for value in row)
        for name, row in zip(class_names, matrix)
    ]
    return "\n".join(["rows = true, cols = predicted", header, *rows])


def format_per_class(per_class: Dict[str, Dict[str, float]]) -> str:
    lines = [f"{'class':>10} {'prec':>7} {'rec':>7} {'f1':>7} {'n':>6}"]
    for name, scores in per_class.items():
        lines.append(
            f"{name:>10} {scores['precision']:7.4f} {scores['recall']:7.4f} "
            f"{scores['f1']:7.4f} {scores['support']:6d}"
        )
    return "\n".join(lines)
