from __future__ import annotations

import flwr as fl

from .strategy import create_strategy


class FederatedServer:
    def __init__(
        self,
        server_address: str,
        num_rounds: int,
        num_clients: int,
        initial_parameters=None,
    ):
        self.server_address = server_address
        self.num_rounds = num_rounds

        self.strategy = create_strategy(
            num_clients=num_clients,
            initial_parameters=initial_parameters,
        )

    def run(self):
        config = fl.server.ServerConfig(
            num_rounds=self.num_rounds,
        )

        fl.server.start_server(
            server_address=self.server_address,
            config=config,
            strategy=self.strategy,
        )


def start_server(
    server_address: str = "127.0.0.1:8080",
    num_rounds: int = 10,
    num_clients: int = 2,
    initial_parameters=None,
):
    server = FederatedServer(
        server_address=server_address,
        num_rounds=num_rounds,
        num_clients=num_clients,
        initial_parameters=initial_parameters,
    )

    server.run()


if __name__ == "__main__":
    start_server()
