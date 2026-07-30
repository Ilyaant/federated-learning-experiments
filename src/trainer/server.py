from __future__ import annotations

from typing import Optional

import flwr as fl

from .strategy import create_strategy


def start_server(
    server_address: str,
    num_rounds: int,
    num_clients: int,
    initial_parameters=None,
):
    strategy = create_strategy(
        num_clients=num_clients,
        initial_parameters=initial_parameters,
    )

    config = fl.server.ServerConfig(
        num_rounds=num_rounds,
    )

    fl.server.start_server(
        server_address=server_address,
        config=config,
        strategy=strategy,
    )


def run(
    cfg: dict,
    initial_parameters=None,
):
    start_server(
        server_address=cfg["server"]["address"],
        num_rounds=cfg["federated"]["rounds"],
        num_clients=cfg["federated"]["num_clients"],
        initial_parameters=initial_parameters,
    )


if __name__ == "__main__":
    default_cfg = {
        "server": {
            "address": "127.0.0.1:8080",
        },
        "federated": {
            "rounds": 20,
            "num_clients": 5,
        },
    }

    run(default_cfg)
