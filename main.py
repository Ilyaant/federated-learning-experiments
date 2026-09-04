from __future__ import annotations

import argparse
import logging
from datetime import datetime
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
from src.trainer.history import (
    configure_file_logging,
    save_global_model,
    save_history,
)
from src.trainer.strategy import create_strategy
from src.trainer.utils import get_device, seed_everything


logger = logging.getLogger(__name__)


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


import torchvision.transforms as T

class AddGaussianNoise:
    def __init__(self, std=0.05, p=0.2):
        self.std = std
        self.p = p

    def __call__(self, tensor):
        if torch.rand(1).item() < self.p:
            noise = torch.randn_like(tensor) * self.std
            return torch.clamp(tensor + noise, -1.0, 1.0)
        return tensor

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

    train_transform = None
    aug_cfg = cfg["dataset"].get("augmentation", {})
    if aug_cfg.get("enabled", False):
        transforms_list = []
        if aug_cfg.get("hflip_prob", 0.0) > 0:
            transforms_list.append(T.RandomHorizontalFlip(p=aug_cfg["hflip_prob"]))
        if aug_cfg.get("vflip_prob", 0.0) > 0:
            transforms_list.append(T.RandomVerticalFlip(p=aug_cfg["vflip_prob"]))
        if aug_cfg.get("rotation_degrees", 0.0) > 0:
            transforms_list.append(T.RandomRotation(degrees=aug_cfg["rotation_degrees"]))
        if aug_cfg.get("blur_prob", 0.0) > 0:
            transforms_list.append(T.RandomApply([T.GaussianBlur(kernel_size=aug_cfg.get("blur_kernel_size", 5))], p=aug_cfg["blur_prob"]))
        if aug_cfg.get("noise_prob", 0.0) > 0:
            transforms_list.append(AddGaussianNoise(std=aug_cfg.get("noise_std", 0.05), p=aug_cfg["noise_prob"]))
        
        if transforms_list:
            train_transform = T.Compose(transforms_list)

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
        transform=train_transform,
        **patch_kwargs,
    )
    val_dataset = TexturePatchDataset(
        val_partitions[client_id],
        transform=None,
        **patch_kwargs,
    )
    test_dataset = TexturePatchDataset(
        test_partitions[client_id],
        transform=None,
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
        client_id=client_id,
        log_dir=cfg["logging"]["save_dir"],
        initial_lr=cfg["train"]["lr"],
        min_lr=cfg["train"].get("min_lr", 1e-6),
        total_rounds=cfg["federated"]["rounds"],
        max_grad_norm=cfg["train"].get("max_grad_norm", 1.0),
    )


def save_results(cfg: dict, history, strategy) -> None:
    save_dir = Path(cfg["logging"]["save_dir"])

    save_history(history, save_dir)
    logger.info("Saved final history.json and metrics.csv to %s", save_dir)

    if strategy.latest_parameters is not None:
        save_global_model(
            strategy.latest_parameters,
            build_model(cfg),
            save_dir / "model_final.pt",
        )
        logger.info(
            "Saved final global model to %s",
            save_dir / "model_final.pt",
        )
    else:
        logger.warning("No aggregated parameters available; model not saved")


def run_server(cfg: dict):
    strategy = create_strategy(
        num_clients=cfg["federated"]["num_clients"],
        save_dir=cfg["logging"]["save_dir"],
    )

    history = fl.server.start_server(
        server_address=cfg["server"]["address"],
        config=fl.server.ServerConfig(
            num_rounds=cfg["federated"]["rounds"],
        ),
        strategy=strategy,
    )

    save_results(cfg, history, strategy)


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

    strategy = create_strategy(
        num_clients=num_clients,
        save_dir=cfg["logging"]["save_dir"],
    )

    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=num_clients,
        config=fl.server.ServerConfig(
            num_rounds=cfg["federated"]["rounds"],
        ),
        strategy=strategy,
        client_resources=sim_cfg.get(
            "client_resources",
            {"num_cpus": 2, "num_gpus": 0.5},
        ),
        ray_init_args=sim_cfg.get(
            "ray_init_args",
            {"ignore_reinit_error": True, "include_dashboard": False},
        ),
    )

    save_results(cfg, history, strategy)
    return history


def main():
    args = parse_args()
    cfg = load_config(args.config)

    seed_everything(cfg["seed"])
    save_dir = Path(cfg["logging"]["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)
    log_filename = cfg["logging"].get("log_file", "experiment.log")
    if args.mode == "client":
        log_filename = f"client_{args.client_id}.log"
    configure_file_logging(
        save_dir,
        filename=log_filename,
        identifier=f"{args.mode}-{args.client_id}",
    )
    logger.info(
        "Experiment started at %s",
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    )
    device = get_device()
    logger.info(
        "Starting mode=%s client_id=%s with config=%s device=%s",
        args.mode,
        args.client_id,
        args.config,
        device,
    )

    if args.mode == "server":
        run_server(cfg)
    elif args.mode == "simulation":
        run_simulation(cfg)
    else:
        run_client(cfg, args.client_id)


if __name__ == "__main__":
    main()
