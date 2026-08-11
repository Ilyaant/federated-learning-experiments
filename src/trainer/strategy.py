from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import flwr as fl
from flwr.common import (
    Metrics,
    NDArrays,
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


def create_strategy(
    num_clients: int,
    initial_parameters: Optional[NDArrays] = None,
) -> fl.server.strategy.FedAvg:
    return fl.server.strategy.FedAvg(
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
        accept_failures=False,
    )
