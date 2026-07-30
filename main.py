from __future__ import annotations

import argparse
from pathlib import Path

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


def load_config(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        default="configs/texture.yaml",
    )

    parser.add_argument(
        "--mode",
        choices=[
            "server",
            "client",
        ],
        required=True,
    )

    parser.add_argument(
        "--client-id",
        type=int,
        default=0,
    )

    return parser.parse_args()


def build_datasets(cfg, client_id):
    train_samples = load_split(
        cfg["dataset"]["root"],
        "train",
    )

    val_samples = load_split(
        cfg["dataset"]["root"],
        "val",
    )

    partitioner = FederatedPartitioner(
        train_samples,
        cfg["federated"]["num_clients"],
        seed=cfg["seed"],
    )

    train_partitions = partitioner.stratified()

    partitioner = FederatedPartitioner(
        val_samples,
        cfg["federated"]["num_clients"],
        seed=cfg["seed"],
    )

    val_partitions = partitioner.stratified()

    train_dataset = TexturePatchDataset(
        train_partitions[client_id],
        patch_size=cfg["dataset"]["patch_size"],
        overlap=cfg["dataset"]["overlap"],
        grayscale=cfg["dataset"]["grayscale"],
        normalize=cfg["dataset"]["normalize"],
    )

    val_dataset = TexturePatchDataset(
        val_partitions[client_id],
        patch_size=cfg["dataset"]["patch_size"],
        overlap=cfg["dataset"]["overlap"],
        grayscale=cfg["dataset"]["grayscale"],
        normalize=cfg["dataset"]["normalize"],
    )

    return train_dataset, val_dataset


def build_model(cfg):
    return create_model(
        num_classes=cfg["model"]["num_classes"],
        pretrained=cfg["model"]["pretrained"],
        grayscale=cfg["dataset"]["grayscale"],
        freeze_backbone=cfg["model"]["freeze_backbone"],
        dropout=cfg["model"]["dropout"],
        drop_path_rate=cfg["model"]["drop_path_rate"],
    )


def run_server(cfg):
    start_server(
        server_address=cfg["server"]["address"],
        num_rounds=cfg["federated"]["rounds"],
        num_clients=cfg["federated"]["num_clients"],
    )


def run_client(cfg, client_id):
    device = get_device()

    train_dataset, val_dataset = build_datasets(
        cfg,
        client_id,
    )

    model = build_model(cfg).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"],
    )

    criterion = torch.nn.CrossEntropyLoss()

    client = FlowerClient(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        optimizer=optimizer,
        criterion=criterion,
        batch_size=cfg["train"]["batch_size"],
        local_epochs=cfg["train"]["local_epochs"],
        num_workers=cfg["train"]["num_workers"],
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

    Path(cfg["logging"]["save_dir"]).mkdir(
        parents=True,
        exist_ok=True,
    )

    if args.mode == "server":
        run_server(cfg)
    else:
        run_client(
            cfg,
            args.client_id,
        )


if __name__ == "__main__":
    main()
