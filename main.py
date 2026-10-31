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
from src.trainer.losses import compute_class_weights
from src.trainer.metrics import evaluate
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


class RandomRotation90:
    """Rotate by a random multiple of 90 degrees (0, 90, 180, 270; lossless for square patches)."""

    def __init__(self, p: float = 1.0):
        self.p = p

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() < self.p:
            k = int(torch.randint(0, 4, (1,)).item())
            if k > 0:
                return torch.rot90(tensor, k, [-2, -1])
        return tensor


class RandomContrastScale:
    """Scale patch contrast around its spatial mean to simulate session contrast variation."""

    def __init__(self, scale_range=(0.9, 1.1), p: float = 0.3, normalized: bool = True):
        if len(scale_range) != 2 or float(scale_range[0]) <= 0 or float(scale_range[0]) > float(scale_range[1]):
            raise ValueError(f"Invalid contrast scale_range: {scale_range!r}")
        self.min_scale = float(scale_range[0])
        self.max_scale = float(scale_range[1])
        self.p = float(p)
        self.low = -1.0 if normalized else 0.0

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        if (self.min_scale != 1.0 or self.max_scale != 1.0) and torch.rand(1).item() < self.p:
            alpha = self.min_scale + torch.rand(1).item() * (self.max_scale - self.min_scale)
            mean = tensor.mean(dim=(-2, -1), keepdim=True)
            return torch.clamp((tensor - mean) * alpha + mean, self.low, 1.0)
        return tensor


class RandomIlluminationGradient:
    """Add a smooth linear brightness gradient across the patch to simulate residual vignetting."""

    def __init__(self, max_strength: float = 0.02, p: float = 0.2, normalized: bool = True):
        self.max_strength = float(max_strength) * (2.0 if normalized else 1.0)
        self.p = float(p)
        self.low = -1.0 if normalized else 0.0

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.max_strength > 0 and torch.rand(1).item() < self.p:
            _, h, w = tensor.shape
            gx = (torch.rand(1).item() * 2.0 - 1.0) * self.max_strength
            gy = (torch.rand(1).item() * 2.0 - 1.0) * self.max_strength
            ys = torch.linspace(-1.0, 1.0, h, device=tensor.device, dtype=tensor.dtype).view(1, h, 1)
            xs = torch.linspace(-1.0, 1.0, w, device=tensor.device, dtype=tensor.dtype).view(1, 1, w)
            grad = gy * ys + gx * xs
            return torch.clamp(tensor + grad, self.low, 1.0)
        return tensor


class AddGaussianNoise:
    def __init__(self, std=0.05, p=0.2, normalized: bool = True):
        self.std = float(std)
        self.p = float(p)
        self.low = -1.0 if normalized else 0.0

    def __call__(self, tensor):
        if self.std > 0 and torch.rand(1).item() < self.p:
            noise = torch.randn_like(tensor) * self.std
            return torch.clamp(tensor + noise, self.low, 1.0)
        return tensor


def patch_kwargs_from_config(cfg: dict) -> dict:
    """Patch extraction settings shared by every dataset in the run."""
    return {
        "patch_size": cfg["dataset"]["patch_size"],
        "overlap": cfg["dataset"]["overlap"],
        "grayscale": cfg["dataset"]["grayscale"],
        "normalize": cfg["dataset"]["normalize"],
        "cache_size": cfg["dataset"].get("cache_size", 64),
        "downscale": cfg["dataset"].get("downscale", 1.0),
        "seed": cfg["seed"],
    }


def build_train_transform(cfg: dict):
    aug_cfg = cfg["dataset"].get("augmentation", {})
    if not aug_cfg.get("enabled", False):
        return None

    normalized = bool(cfg["dataset"].get("normalize", True))
    transforms_list = []
    if aug_cfg.get("rot90_prob", 0.0) > 0:
        transforms_list.append(RandomRotation90(p=float(aug_cfg["rot90_prob"])))
    if aug_cfg.get("hflip_prob", 0.0) > 0:
        transforms_list.append(T.RandomHorizontalFlip(p=float(aug_cfg["hflip_prob"])))
    if aug_cfg.get("vflip_prob", 0.0) > 0:
        transforms_list.append(T.RandomVerticalFlip(p=float(aug_cfg["vflip_prob"])))
    if aug_cfg.get("rotation_degrees", 0.0) > 0:
        transforms_list.append(T.RandomRotation(degrees=float(aug_cfg["rotation_degrees"])))
    if aug_cfg.get("blur_prob", 0.0) > 0:
        transforms_list.append(
            T.RandomApply(
                [T.GaussianBlur(kernel_size=int(aug_cfg.get("blur_kernel_size", 5)))],
                p=float(aug_cfg["blur_prob"]),
            )
        )

    contrast_cfg = aug_cfg.get("contrast") or {}
    contrast_prob = float(contrast_cfg.get("prob", aug_cfg.get("contrast_prob", 0.0)))
    if contrast_prob > 0:
        scale_range = contrast_cfg.get("scale", aug_cfg.get("contrast_scale", (0.9, 1.1)))
        transforms_list.append(
            RandomContrastScale(
                scale_range=tuple(float(s) for s in scale_range),
                p=contrast_prob,
                normalized=normalized,
            )
        )

    illum_cfg = aug_cfg.get("illumination_gradient") or {}
    illum_prob = float(illum_cfg.get("prob", aug_cfg.get("illumination_prob", 0.0)))
    if illum_prob > 0:
        max_strength = float(illum_cfg.get("max_strength", aug_cfg.get("illumination_max_strength", 0.02)))
        transforms_list.append(
            RandomIlluminationGradient(
                max_strength=max_strength,
                p=illum_prob,
                normalized=normalized,
            )
        )

    if aug_cfg.get("noise_prob", 0.0) > 0:
        transforms_list.append(
            AddGaussianNoise(
                std=float(aug_cfg.get("noise_std", 0.05)),
                p=float(aug_cfg["noise_prob"]),
                normalized=normalized,
            )
        )

    return T.Compose(transforms_list) if transforms_list else None


def build_datasets(cfg: dict, client_id: int):
    root = cfg["dataset"]["root"]
    num_clients = cfg["federated"]["num_clients"]
    seed = cfg["seed"]
    strategy = cfg["federated"].get("partition", "stratified")
    patch_kwargs = patch_kwargs_from_config(cfg)
    train_transform = build_train_transform(cfg)

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
        grid_jitter=bool(cfg["dataset"].get("train_grid_jitter", False)),
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

    num_classes = cfg["model"]["num_classes"]
    weights = None
    if cfg["train"].get("weighted_loss", True):
        global_distribution: dict[int, int] = {}
        for _, label in load_split(cfg["dataset"]["root"], "train"):
            global_distribution[label] = (
                global_distribution.get(label, 0) + 1
            )
        weights = compute_class_weights(
            global_distribution,
            num_classes,
            power=cfg["train"].get("class_weight_power", 0.5),
            device=device,
        )
        logger.info(
            "Using global class distribution %s with weights %s",
            global_distribution,
            [round(value, 4) for value in weights.detach().cpu().tolist()],
        )

    criterion = torch.nn.CrossEntropyLoss(
        weight=weights,
        label_smoothing=cfg["train"].get("label_smoothing", 0.0),
    )
    eval_criterion = torch.nn.CrossEntropyLoss()

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
        criterion=criterion,
        eval_criterion=eval_criterion,
        batch_size=cfg["train"]["batch_size"],
        local_epochs=cfg["train"]["local_epochs"],
        num_workers=(
            num_workers
            if num_workers is not None
            else cfg["train"]["num_workers"]
        ),
        num_classes=num_classes,
        aggregation=cfg["train"].get("aggregation", "average_probability"),
        device=device,
        client_id=client_id,
        log_dir=cfg["logging"]["save_dir"],
        initial_lr=cfg["train"]["lr"],
        min_lr=cfg["train"].get("min_lr", 1e-6),
        total_rounds=cfg["federated"]["rounds"],
        max_grad_norm=cfg["train"].get("max_grad_norm", 1.0),
        tta=cfg.get("evaluation", {}).get("tta", False),
    )


def build_central_eval_fn(cfg: dict):
    """Server-side evaluation of a parameter vector on the *full*
    validation and test splits (all images, no client partitioning).

    Used to score the SWA model; metrics are prefixed ``swa_val_`` /
    ``swa_test_``. Datasets and the model are built lazily on first call so
    that server-only modes do not pay for them unless needed.
    """
    state: dict = {}

    def _setup():
        device = get_device()
        patch_kwargs = patch_kwargs_from_config(cfg)
        root = cfg["dataset"]["root"]
        loader_kwargs = {
            "batch_size": cfg["train"]["batch_size"],
            "shuffle": False,
            "num_workers": cfg.get("simulation", {}).get("num_workers", 0),
            "pin_memory": torch.cuda.is_available(),
        }
        state["device"] = device
        state["model"] = build_model(cfg).to(device)
        state["loaders"] = {
            split: torch.utils.data.DataLoader(
                TexturePatchDataset(load_split(root, split), **patch_kwargs),
                **loader_kwargs,
            )
            for split in ("val", "test")
        }
        state["criterion"] = torch.nn.CrossEntropyLoss()

    def central_evaluate(server_round: int, parameters):
        if not state:
            _setup()
        model = state["model"]
        state_dict = {
            key: torch.from_numpy(value)
            for key, value in zip(model.state_dict().keys(), parameters)
        }
        model.load_state_dict(state_dict, strict=True)

        metrics: dict[str, float] = {}
        test_loss = 0.0
        for split, loader in state["loaders"].items():
            result = evaluate(
                model,
                loader,
                state["criterion"],
                state["device"],
                num_classes=cfg["model"]["num_classes"],
                aggregation=cfg["train"].get("aggregation", "average_probability"),
                tta=cfg.get("evaluation", {}).get("tta", False),
            )
            for key, value in result.items():
                if key.endswith("confusion_matrix"):
                    continue  # strings are not Flower/CSV scalars
                metrics[f"swa_{split}_{key}"] = float(value)
            if split == "test":
                test_loss = float(result["loss"])

        return test_loss, metrics

    return central_evaluate


def strategy_kwargs_from_config(cfg: dict) -> dict:
    eval_cfg = cfg.get("evaluation", {})
    swa_window = int(eval_cfg.get("swa_window", 0))
    return {
        "num_clients": cfg["federated"]["num_clients"],
        "save_dir": cfg["logging"]["save_dir"],
        "model_fn": lambda: build_model(cfg),
        "swa_window": swa_window,
        "central_eval_fn": build_central_eval_fn(cfg) if swa_window > 0 else None,
        "swa_eval_every": int(eval_cfg.get("swa_eval_every", 1)),
        "num_rounds": cfg["federated"]["rounds"],
    }


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

    if getattr(strategy, "best_parameters", None) is not None:
        save_global_model(
            strategy.best_parameters,
            build_model(cfg),
            save_dir / "model_best.pt",
        )
        logger.info(
            "Saved best global model (round %s, val_score=%.4f) to %s",
            strategy.best_round,
            strategy.best_score,
            save_dir / "model_best.pt",
        )

    if getattr(strategy, "swa_parameters", None) is not None:
        save_global_model(
            strategy.swa_parameters,
            build_model(cfg),
            save_dir / "model_swa.pt",
        )
        logger.info(
            "Saved SWA global model (mean of last %s rounds) to %s",
            strategy.swa_window,
            save_dir / "model_swa.pt",
        )


def run_server(cfg: dict):
    strategy = create_strategy(**strategy_kwargs_from_config(cfg))

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

    strategy = create_strategy(**strategy_kwargs_from_config(cfg))

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
