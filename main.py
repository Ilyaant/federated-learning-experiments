from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import flwr as fl
import torch
import yaml

from datasets.partition import FederatedPartitioner
from datasets.preprocessing import load_split
from datasets.texture_patch_dataset import TexturePatchDataset
from models import create_model
from trainer.client import FlowerClient
from trainer.server import start_server
from trainer.utils import get_device, seed_everything


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
        choices=["server", "client"],
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


def run_server(cfg: dict):
    start_server(
        server_address=cfg["server"]["address"],
        num_rounds=cfg["federated"]["rounds"],
        num_clients=cfg["federated"]["num_clients"],
    )


def run_client(cfg: dict, client_id: int):
    device = get_device()
    train_dataset, val_dataset, test_dataset = build_datasets(cfg, client_id)
    model = build_model(cfg).to(device)

    client = FlowerClient(
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
        num_workers=cfg["train"]["num_workers"],
        num_classes=cfg["model"]["num_classes"],
        aggregation=cfg["train"].get("aggregation", "average_probability"),
        device=device,
    )

    fl.client.start_numpy_client(
        server_address=cfg["server"]["address"],
        client=client,
    )


def main():
    args = parse_args()
    cfg = load_config(args.config)

    seed_everything(cfg["seed"])
    Path(cfg["logging"]["save_dir"]).mkdir(parents=True, exist_ok=True)

    if args.mode == "server":
        run_server(cfg)
    else:
        run_client(cfg, args.client_id)


if __name__ == "__main__":
    main()
