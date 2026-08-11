from __future__ import annotations

import argparse
from pathlib import Path

import flwr as fl
import torch
import yaml
from flwr.common import Context
from flwr.common.constant import PARTITION_ID_KEY

from src.datasets.partition import FederatedPartitioner
from src.datasets.preprocessing import load_split
from src.datasets.texture_patch_dataset import TexturePatchDataset
from src.models import create_model
from src.trainer.client import FlowerClient
from src.trainer.strategy import create_strategy
from src.trainer.utils import get_device, seed_everything


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Federated texture classification with Flower",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/texture.yaml",
    )
    parser.add_argument(
        "--mode",
        choices=["server", "client", "simulation"],
        required=True,
    )
    parser.add_argument("--client-id", type=int, default=0)
    return parser.parse_args()


def partition_split(
    samples,
    num_clients: int,
    seed: int,
    strategy: str = "stratified",
):
    partitioner = FederatedPartitioner(samples, num_clients, seed=seed)

    if strategy == "iid":
        return partitioner.iid()
    if strategy == "dirichlet":
        return partitioner.dirichlet()
    return partitioner.stratified()


def build_datasets(cfg: dict, client_id: int):
    root = cfg["dataset"]["root"]
    num_clients = cfg["federated"]["num_clients"]
    seed = cfg["seed"]
    strategy = cfg["federated"].get("partition", "stratified")
    patch_kwargs = {
        "patch_size": cfg["dataset"]["patch_size"],
        "overlap": cfg["dataset"]["overlap"],
        "grayscale": cfg["dataset"]["grayscale"],
        "normalize": cfg["dataset"]["normalize"],
        "cache_size": cfg["dataset"].get("cache_size", 64),
        "seed": seed,
    }

    train_partitions = partition_split(
        load_split(root, "train"),
        num_clients,
        seed,
        strategy,
    )
    val_partitions = partition_split(
        load_split(root, "val"),
        num_clients,
        seed,
        strategy,
    )
    test_partitions = partition_split(
        load_split(root, "test"),
        num_clients,
        seed,
        strategy,
    )

    train_dataset = TexturePatchDataset(
        train_partitions[client_id],
        epoch_fraction=cfg["train"].get("epoch_fraction", 1.0),
        balanced_per_image=cfg["train"].get("balanced_per_image", True),
        with_replacement=cfg["train"].get("with_replacement", False),
        **patch_kwargs,
    )
    val_dataset = TexturePatchDataset(
        val_partitions[client_id],
        **patch_kwargs,
    )
    test_dataset = TexturePatchDataset(
        test_partitions[client_id],
        **patch_kwargs,
    )

    return train_dataset, val_dataset, test_dataset


def build_model(cfg: dict):
    return create_model(
        num_classes=cfg["model"]["num_classes"],
        pretrained=cfg["model"]["pretrained"],
        grayscale=cfg["dataset"]["grayscale"],
        freeze_backbone=cfg["model"]["freeze_backbone"],
        dropout=cfg["model"]["dropout"],
        drop_path_rate=cfg["model"]["drop_path_rate"],
    )


def build_flower_client(
    cfg: dict,
    client_id: int,
    num_workers: int | None = None,
) -> FlowerClient:
    device = get_device()
    train_dataset, val_dataset, test_dataset = build_datasets(cfg, client_id)
    model = build_model(cfg).to(device)

    return FlowerClient(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        optimizer=torch.optim.AdamW(
            model.parameters(),
            lr=cfg["train"]["lr"],
            weight_decay=cfg["train"]["weight_decay"],
        ),
        criterion=torch.nn.CrossEntropyLoss(),
        batch_size=cfg["train"]["batch_size"],
        local_epochs=cfg["train"]["local_epochs"],
        num_workers=(
            num_workers
            if num_workers is not None
            else cfg["train"]["num_workers"]
        ),
        num_classes=cfg["model"]["num_classes"],
        aggregation=cfg["train"].get("aggregation", "average_probability"),
        device=device,
    )


def run_server(cfg: dict):
    fl.server.start_server(
        server_address=cfg["server"]["address"],
        config=fl.server.ServerConfig(
            num_rounds=cfg["federated"]["rounds"],
        ),
        strategy=create_strategy(
            num_clients=cfg["federated"]["num_clients"],
        ),
    )


def run_client(cfg: dict, client_id: int):
    client = build_flower_client(cfg, client_id)

    fl.client.start_numpy_client(
        server_address=cfg["server"]["address"],
        client=client,
    )


def run_simulation(cfg: dict):
    num_clients = cfg["federated"]["num_clients"]
    sim_cfg = cfg.get("simulation", {})
    client_cache: dict[int, FlowerClient] = {}

    def client_fn(context: Context):
        client_id = int(context.node_config[PARTITION_ID_KEY])
        if client_id not in client_cache:
            client_cache[client_id] = build_flower_client(
                cfg,
                client_id,
                num_workers=sim_cfg.get("num_workers", 0),
            )
        return client_cache[client_id].to_client()

    return fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=num_clients,
        config=fl.server.ServerConfig(
            num_rounds=cfg["federated"]["rounds"],
        ),
        strategy=create_strategy(num_clients=num_clients),
        client_resources=sim_cfg.get(
            "client_resources",
            {"num_cpus": 1, "num_gpus": 0.0},
        ),
        ray_init_args=sim_cfg.get(
            "ray_init_args",
            {"ignore_reinit_error": True, "include_dashboard": False},
        ),
    )


def main():
    args = parse_args()
    cfg = load_config(args.config)

    seed_everything(cfg["seed"])
    Path(cfg["logging"]["save_dir"]).mkdir(parents=True, exist_ok=True)

    if args.mode == "server":
        run_server(cfg)
    elif args.mode == "simulation":
        run_simulation(cfg)
    else:
        run_client(cfg, args.client_id)


if __name__ == "__main__":
    main()
