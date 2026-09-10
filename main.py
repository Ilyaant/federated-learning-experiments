from __future__ import annotations

import argparse
import logging
from datetime import datetime
from pathlib import Path

import torch
import yaml

from src.datasets.preprocessing import load_split
from src.datasets.texture_patch_dataset import TexturePatchDataset
from src.datasets.transforms import build_train_transform
from src.models import create_model
from src.trainer.history import configure_file_logging
from src.trainer.losses import compute_class_weights
from src.trainer.trainer import Trainer
from src.trainer.utils import get_device, seed_everything


logger = logging.getLogger(__name__)


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Centralized texture classification",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/texture.yaml",
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--save-dir", type=str, default=None)
    parser.add_argument("--downscale", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def apply_overrides(cfg: dict, args) -> dict:
    if args.epochs is not None:
        cfg.setdefault("train", {})["epochs"] = args.epochs
    if args.save_dir is not None:
        cfg.setdefault("logging", {})["save_dir"] = args.save_dir
    if args.downscale is not None:
        cfg.setdefault("dataset", {})["downscale"] = args.downscale
    if args.seed is not None:
        cfg["seed"] = args.seed
    return cfg


def dump_config(cfg: dict, save_dir: Path) -> None:
    path = save_dir / "config.yaml"
    with open(path, "w", encoding="utf-8") as file:
        yaml.safe_dump(cfg, file, sort_keys=False, allow_unicode=True)
    logger.info("Wrote resolved config to %s", path)


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


def build_datasets(cfg: dict):
    root = cfg["dataset"]["root"]
    patch_kwargs = patch_kwargs_from_config(cfg)
    train_transform = build_train_transform(cfg["dataset"].get("augmentation", {}))

    train_dataset = TexturePatchDataset(
        load_split(root, "train"),
        epoch_fraction=cfg["train"].get("epoch_fraction", 1.0),
        balanced_per_image=cfg["train"].get("balanced_per_image", True),
        with_replacement=cfg["train"].get("with_replacement", False),
        transform=train_transform,
        **patch_kwargs,
    )
    val_dataset = TexturePatchDataset(
        load_split(root, "val"),
        transform=None,
        **patch_kwargs,
    )
    test_dataset = TexturePatchDataset(
        load_split(root, "test"),
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


def build_trainer(cfg: dict) -> Trainer:
    device = get_device()
    train_dataset, val_dataset, test_dataset = build_datasets(cfg)
    model = build_model(cfg).to(device)

    logger.info("Train %s", train_dataset)
    logger.info("Val %s", val_dataset)
    logger.info("Test %s", test_dataset)

    num_classes = cfg["model"]["num_classes"]
    weights = None
    if cfg["train"].get("weighted_loss", True):
        distribution = train_dataset.class_distribution()
        weights = compute_class_weights(
            distribution,
            num_classes,
            power=cfg["train"].get("class_weight_power", 0.5),
            device=device,
        )
        logger.info(
            "Using class distribution %s with weights %s",
            distribution,
            [round(value, 4) for value in weights.detach().cpu().tolist()],
        )

    criterion = torch.nn.CrossEntropyLoss(
        weight=weights,
        label_smoothing=cfg["train"].get("label_smoothing", 0.0),
    )
    eval_criterion = torch.nn.CrossEntropyLoss()
    eval_cfg = cfg.get("evaluation", {})

    return Trainer(
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
        epochs=cfg["train"]["epochs"],
        num_workers=cfg["train"]["num_workers"],
        num_classes=num_classes,
        aggregation=cfg["train"].get("aggregation", "average_probability"),
        device=device,
        initial_lr=cfg["train"]["lr"],
        min_lr=cfg["train"].get("min_lr", 1e-6),
        max_grad_norm=cfg["train"].get("max_grad_norm", 1.0),
        tta=eval_cfg.get("tta", False),
        eval_train=eval_cfg.get("eval_train", True),
        swa_window=int(eval_cfg.get("swa_window", 0)),
        swa_eval_every=int(eval_cfg.get("swa_eval_every", 1)),
        save_dir=cfg["logging"]["save_dir"],
    )


def main():
    args = parse_args()
    cfg = apply_overrides(load_config(args.config), args)

    seed_everything(cfg["seed"])
    save_dir = Path(cfg["logging"]["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)
    configure_file_logging(
        save_dir,
        filename=cfg["logging"].get("log_file", "experiment.log"),
    )
    dump_config(cfg, save_dir)
    logger.info(
        "Experiment started at %s",
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    )
    device = get_device()
    logger.info("Starting training with config=%s device=%s", args.config, device)

    trainer = build_trainer(cfg)
    trainer.fit()
    logger.info("Experiment finished at %s", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


if __name__ == "__main__":
    main()
