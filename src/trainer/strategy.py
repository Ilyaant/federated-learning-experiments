from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import flwr as fl
from flwr.common import (
    Metrics,
    NDArrays,
    ndarrays_to_parameters,
)

from .history import LiveHistoryWriter, save_global_model


logger = logging.getLogger(__name__)


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    if len(metrics) == 0:
        return {}

    total_examples = sum(num_examples for num_examples, _ in metrics)

    aggregated: Dict[str, float] = {}

    metric_names = set()
    for _, metric in metrics:
        metric_names.update(metric.keys())

    for name in metric_names:
        value = 0.0
        for num_examples, metric in metrics:
            if name in metric:
                value += num_examples * float(metric[name])
        aggregated[name] = value / total_examples

    return aggregated


class TrackingFedAvg(fl.server.strategy.FedAvg):
    """FedAvg that keeps the latest aggregated parameters, so the
    final global model can be saved after training."""

    def __init__(
        self,
        save_dir=None,
        model_fn: Optional[Callable] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.save_dir = Path(save_dir) if save_dir is not None else None
        self.model_fn = model_fn
        self.latest_parameters = None
        self.best_parameters = None
        self.best_score = -float("inf")
        self.best_round = 0
        self.history_writer = (
            LiveHistoryWriter(save_dir) if save_dir is not None else None
        )

    def aggregate_fit(self, server_round, results, failures):
        parameters, metrics = super().aggregate_fit(
            server_round,
            results,
            failures,
        )

        if parameters is not None:
            self.latest_parameters = parameters

            val_score = None
            if metrics:
                for metric_key in (
                    "val_image_f1",
                    "val_f1",
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


def create_strategy(
    num_clients: int,
    initial_parameters: Optional[NDArrays] = None,
    save_dir=None,
    model_fn: Optional[Callable] = None,
) -> TrackingFedAvg:
    return TrackingFedAvg(
        save_dir=save_dir,
        model_fn=model_fn,
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
        fit_metrics_aggregation_fn=weighted_average,
        evaluate_metrics_aggregation_fn=weighted_average,
        on_fit_config_fn=lambda server_round: {
            "server_round": server_round,
        },
        on_evaluate_config_fn=lambda server_round: {
            "server_round": server_round,
        },
        accept_failures=False,
    )
