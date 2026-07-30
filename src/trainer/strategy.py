from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import flwr as fl
from flwr.common import (
    Metrics,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
)


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


class FedAvgStrategy(fl.server.strategy.FedAvg):
    def __init__(
        self,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        initial_parameters: Optional[Parameters] = None,
        accept_failures: bool = False,
    ):
        super().__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            initial_parameters=initial_parameters,
            fit_metrics_aggregation_fn=weighted_average,
            evaluate_metrics_aggregation_fn=weighted_average,
            accept_failures=accept_failures,
        )


def create_strategy(
    num_clients: int,
    initial_parameters: Optional[NDArrays] = None,
):
    parameters = None

    if initial_parameters is not None:
        parameters = ndarrays_to_parameters(initial_parameters)

    return FedAvgStrategy(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=num_clients,
        min_evaluate_clients=num_clients,
        min_available_clients=num_clients,
        initial_parameters=parameters,
    )
