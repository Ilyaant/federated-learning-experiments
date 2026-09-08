from __future__ import annotations

import logging
from collections import deque
from pathlib import Path
from typing import Callable, Deque, Dict, List, Optional, Tuple

import flwr as fl
import numpy as np
from flwr.common import (
    Metrics,
    NDArrays,
    Parameters,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)

from .aggregation import (
    classification_summary_from_confusion,
    deserialize_confusion_matrix,
)
from .history import LiveHistoryWriter, save_global_model


logger = logging.getLogger(__name__)


def aggregate_metrics(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Aggregate scalar metrics and reconstruct exact global macro metrics."""
    if len(metrics) == 0:
        return {}

    aggregated: Dict[str, float] = {}
    scalar_names = {
        name
        for _, client_metrics in metrics
        for name, value in client_metrics.items()
        if not name.endswith("confusion_matrix")
        and isinstance(value, (bool, int, float))
    }

    for name in scalar_names:
        weighted_sum = 0.0
        metric_weight = 0
        for num_examples, client_metrics in metrics:
            value = client_metrics.get(name)
            if isinstance(value, (bool, int, float)):
                weighted_sum += num_examples * float(value)
                metric_weight += num_examples
        if metric_weight:
            aggregated[name] = weighted_sum / metric_weight

    summed_matrices: Dict[str, np.ndarray] = {}
    for _, client_metrics in metrics:
        for name, value in client_metrics.items():
            if not name.endswith("confusion_matrix"):
                continue
            if not isinstance(value, (str, bytes)):
                raise TypeError(f"{name} must be serialized as str or bytes")

            matrix = deserialize_confusion_matrix(value)
            if name in summed_matrices:
                if summed_matrices[name].shape != matrix.shape:
                    raise ValueError(
                        f"Inconsistent confusion matrix shape for {name}"
                    )
                summed_matrices[name] += matrix
            else:
                summed_matrices[name] = matrix.copy()

    for name, matrix in summed_matrices.items():
        metric_prefix = name.removesuffix("confusion_matrix")
        summary = classification_summary_from_confusion(matrix)
        for metric_name, value in summary.items():
            aggregated[f"{metric_prefix}{metric_name}"] = value

    return aggregated


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Backward-compatible alias for the project metric aggregator."""
    return aggregate_metrics(metrics)


def average_ndarrays(snapshots: List[NDArrays]) -> NDArrays:
    """Element-wise mean of several parameter lists, keeping each
    layer's dtype (integer buffers such as num_batches_tracked included)."""
    if not snapshots:
        raise ValueError("Nothing to average")
    averaged: NDArrays = []
    for layers in zip(*snapshots):
        stacked = np.stack([np.asarray(layer, dtype=np.float64) for layer in layers])
        averaged.append(stacked.mean(axis=0).astype(layers[0].dtype))
    return averaged


CentralEvalFn = Callable[[int, NDArrays], Tuple[float, Dict[str, float]]]


class TrackingFedAvg(fl.server.strategy.FedAvg):
    """FedAvg that keeps the latest aggregated parameters (so the final
    global model can be saved), tracks the best round by validation score
    and optionally maintains a running average of the last ``swa_window``
    global models (SWA over rounds).

    Training always continues from the plain FedAvg parameters; the SWA
    model is a side product that is evaluated centrally through
    ``central_eval_fn`` and saved separately.
    """

    def __init__(
        self,
        save_dir=None,
        model_fn: Optional[Callable] = None,
        swa_window: int = 0,
        central_eval_fn: Optional[CentralEvalFn] = None,
        swa_eval_every: int = 1,
        num_rounds: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.save_dir = Path(save_dir) if save_dir is not None else None
        self.model_fn = model_fn
        self.latest_parameters = None
        self.best_parameters = None
        self.best_score = -float("inf")
        self.best_round = 0
        self.swa_window = int(swa_window)
        self.central_eval_fn = central_eval_fn
        self.swa_eval_every = max(1, int(swa_eval_every))
        self.num_rounds = num_rounds
        self._swa_snapshots: Deque[NDArrays] = deque(
            maxlen=max(1, self.swa_window)
        )
        self.swa_parameters: Optional[Parameters] = None
        self.history_writer = (
            LiveHistoryWriter(save_dir) if save_dir is not None else None
        )

    def _update_swa(self, parameters: Parameters) -> None:
        if self.swa_window <= 0:
            return
        self._swa_snapshots.append(parameters_to_ndarrays(parameters))
        self.swa_parameters = ndarrays_to_parameters(
            average_ndarrays(list(self._swa_snapshots))
        )

    def aggregate_fit(self, server_round, results, failures):
        parameters, metrics = super().aggregate_fit(
            server_round,
            results,
            failures,
        )

        if parameters is not None:
            self.latest_parameters = parameters
            self._update_swa(parameters)

            val_score = None
            if metrics:
                for metric_key in (
                    "val_f1",
                    "val_image_f1",
                    "val_image_accuracy",
                    "val_accuracy",
                ):
                    if metric_key in metrics:
                        val_score = float(metrics[metric_key])
                        break

            if val_score is not None and val_score > self.best_score:
                self.best_score = val_score
                self.best_round = server_round
                self.best_parameters = parameters
                if self.save_dir is not None and self.model_fn is not None:
                    save_global_model(
                        parameters,
                        self.model_fn(),
                        self.save_dir / "model_best.pt",
                    )
                    logger.info(
                        "Round %s: new best val score %.4f, saved %s",
                        server_round,
                        val_score,
                        self.save_dir / "model_best.pt",
                    )

        if self.history_writer is not None:
            self.history_writer.update_fit(server_round, metrics)

        return parameters, metrics

    def aggregate_evaluate(self, server_round, results, failures):
        loss, metrics = super().aggregate_evaluate(
            server_round,
            results,
            failures,
        )

        if self.history_writer is not None:
            self.history_writer.update_evaluate(server_round, loss, metrics)

        return loss, metrics

    def evaluate(self, server_round, parameters):
        """Centralized evaluation hook: Flower calls it after every
        aggregate_fit (and once at round 0). We use it to score the SWA
        model on the full validation/test splits."""
        if (
            self.central_eval_fn is None
            or self.swa_parameters is None
            or server_round == 0
        ):
            return None

        is_last = self.num_rounds is not None and server_round >= self.num_rounds
        if server_round % self.swa_eval_every != 0 and not is_last:
            return None

        loss, metrics = self.central_eval_fn(
            server_round,
            parameters_to_ndarrays(self.swa_parameters),
        )
        logger.info(
            "Round %s SWA(last %s rounds) centralized metrics: %s",
            server_round,
            len(self._swa_snapshots),
            metrics,
        )
        if self.history_writer is not None:
            self.history_writer.update_centralized(server_round, loss, metrics)

        return float(loss), metrics


def create_strategy(
    num_clients: int,
    initial_parameters: Optional[NDArrays] = None,
    save_dir=None,
    model_fn: Optional[Callable] = None,
    swa_window: int = 0,
    central_eval_fn: Optional[CentralEvalFn] = None,
    swa_eval_every: int = 1,
    num_rounds: Optional[int] = None,
) -> TrackingFedAvg:
    return TrackingFedAvg(
        save_dir=save_dir,
        model_fn=model_fn,
        swa_window=swa_window,
        central_eval_fn=central_eval_fn,
        swa_eval_every=swa_eval_every,
        num_rounds=num_rounds,
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=num_clients,
        min_evaluate_clients=num_clients,
        min_available_clients=num_clients,
        initial_parameters=(
            ndarrays_to_parameters(initial_parameters)
            if initial_parameters is not None
            else None
        ),
        fit_metrics_aggregation_fn=aggregate_metrics,
        evaluate_metrics_aggregation_fn=aggregate_metrics,
        on_fit_config_fn=lambda server_round: {
            "server_round": server_round,
        },
        on_evaluate_config_fn=lambda server_round: {
            "server_round": server_round,
        },
        accept_failures=False,
    )
