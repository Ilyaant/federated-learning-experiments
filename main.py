from __future__ import annotations

import argparse
import logging
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import torch
import yaml

from src.datasets.preprocessing import load_splits, write_split_manifest
from src.datasets.texture_patch_dataset import TexturePatchDataset
from src.datasets.transforms import build_train_transform
from src.models import create_model
from src.trainer.experiments import (
    apply_dotted_overrides,
    flatten_job_overrides,
    load_sweep_jobs,
    load_yaml,
    parse_override,
    pick_best_run,
    resolve_save_dir,
    write_sweep_summary,
)
from src.trainer.history import configure_file_logging, reset_file_logging
from src.trainer.losses import compute_class_weights
from src.trainer.trainer import Trainer
from src.trainer.utils import build_optimizer, get_device, seed_everything


logger = logging.getLogger(__name__)


def load_config(path: str) -> dict:
    return load_yaml(path)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Centralized texture classification experiments",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/texture.yaml",
    )
    parser.add_argument(
        "--sweep",
        type=str,
        default=None,
        help="YAML with a grid and/or named runs to search hyperparameters",
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--save-dir", type=str, default=None)
    parser.add_argument("--downscale", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--selection-metric", type=str, default=None)
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Nested config override, e.g. train.lr=3e-5 (repeatable)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned sweep jobs without training",
    )
    return parser.parse_args(argv)


def apply_overrides(cfg: dict, args) -> dict:
    if getattr(args, "epochs", None) is not None:
        cfg.setdefault("train", {})["epochs"] = args.epochs
    if getattr(args, "save_dir", None) is not None:
        logging_cfg = cfg.setdefault("logging", {})
        logging_cfg["save_dir"] = args.save_dir
        logging_cfg["unique_dir"] = False
    if getattr(args, "downscale", None) is not None:
        cfg.setdefault("dataset", {})["downscale"] = args.downscale
    if getattr(args, "seed", None) is not None:
        cfg["seed"] = args.seed
    if getattr(args, "run_name", None):
        cfg.setdefault("logging", {})["run_name"] = args.run_name
    if getattr(args, "patience", None) is not None:
        cfg.setdefault("evaluation", {})["early_stopping_patience"] = args.patience
    if getattr(args, "selection_metric", None):
        cfg.setdefault("evaluation", {})["selection_metric"] = args.selection_metric
    for spec in getattr(args, "override", None) or []:
        key, value = parse_override(spec)
        apply_dotted_overrides(cfg, {key: value})
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
    dataset_cfg = cfg["dataset"]
    splits, manifest = load_splits(
        dataset_cfg["root"],
        regroup=dataset_cfg.get("regroup", True),
        group_by=dataset_cfg.get("group_by", "sample"),
        train_ratio=dataset_cfg.get("train_ratio", 0.7),
        val_ratio=dataset_cfg.get("val_ratio", 0.15),
        test_ratio=dataset_cfg.get("test_ratio", 0.15),
        seed=dataset_cfg.get("split_seed", cfg["seed"]),
    )
    patch_kwargs = patch_kwargs_from_config(cfg)
    train_transform = build_train_transform(dataset_cfg.get("augmentation", {}))

    train_dataset = TexturePatchDataset(
        splits["train"],
        epoch_fraction=cfg["train"].get("epoch_fraction", 1.0),
        balanced_per_image=cfg["train"].get("balanced_per_image", True),
        with_replacement=cfg["train"].get("with_replacement", False),
        transform=train_transform,
        **patch_kwargs,
    )
    val_dataset = TexturePatchDataset(
        splits["val"],
        transform=None,
        **patch_kwargs,
    )
    test_dataset = TexturePatchDataset(
        splits["test"],
        transform=None,
        **patch_kwargs,
    )
    return train_dataset, val_dataset, test_dataset, manifest


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
    train_dataset, val_dataset, test_dataset, manifest = build_datasets(cfg)
    save_dir = cfg.get("logging", {}).get("save_dir")
    if save_dir is not None:
        write_split_manifest(manifest, Path(save_dir) / "split_manifest.json")
    model = build_model(cfg).to(device)

    logger.info("Train %s", train_dataset)
    logger.info("Val %s", val_dataset)
    logger.info("Test %s", test_dataset)
    logger.info(
        "Split counts %s group_by=%s",
        manifest.get("counts"),
        manifest.get("group_by"),
    )

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
    optimizer = build_optimizer(model, cfg)
    logger.info(
        "Optimizer groups %s",
        [
            {
                "lr": group["lr"],
                "n_params": sum(parameter.numel() for parameter in group["params"]),
            }
            for group in optimizer.param_groups
        ],
    )

    return Trainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        optimizer=optimizer,
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
        eval_train=eval_cfg.get("eval_train", False),
        eval_train_every=int(eval_cfg.get("eval_train_every", 1)),
        eval_val_every=int(eval_cfg.get("eval_val_every", 1)),
        eval_test=str(eval_cfg.get("eval_test", "end")),
        selection_metric=eval_cfg.get("selection_metric", "val_f1"),
        early_stopping_patience=int(eval_cfg.get("early_stopping_patience", 0)),
        min_delta=float(eval_cfg.get("min_delta", 0.0)),
        swa_window=int(eval_cfg.get("swa_window", 0)),
        swa_mode=str(eval_cfg.get("swa_mode", "best")),
        swa_eval_every=int(eval_cfg.get("swa_eval_every", 0)),
        save_dir=save_dir,
    )


def run_experiment(cfg: dict) -> dict:
    seed_everything(cfg["seed"])
    save_dir = resolve_save_dir(cfg)
    save_dir.mkdir(parents=True, exist_ok=True)
    cfg = deepcopy(cfg)
    cfg.setdefault("logging", {})["save_dir"] = str(save_dir)
    cfg["logging"]["unique_dir"] = False

    reset_file_logging()
    configure_file_logging(
        save_dir,
        filename=cfg["logging"].get("log_file", "experiment.log"),
    )
    dump_config(cfg, save_dir)
    logger.info(
        "Experiment started at %s save_dir=%s",
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        save_dir,
    )
    device = get_device()
    logger.info("Starting training device=%s", device)

    trainer = build_trainer(cfg)
    history = trainer.fit()
    logger.info(
        "Experiment finished at %s",
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    )

    summary = {}
    if history is not None and history.summary:
        summary = dict(history.summary)
    summary["save_dir"] = str(save_dir)
    summary["seed"] = cfg["seed"]
    return summary


def _apply_sweep_defaults(cfg: dict, sweep: dict) -> dict:
    apply_dotted_overrides(cfg, sweep.get("apply") or {})
    if sweep.get("selection_metric"):
        cfg.setdefault("evaluation", {})["selection_metric"] = sweep["selection_metric"]
    return cfg


def run_sweep(sweep_path: str, args) -> dict | None:
    sweep = load_yaml(sweep_path)
    base_cfg = apply_overrides(load_config(sweep.get("base", args.config)), args)
    _apply_sweep_defaults(base_cfg, sweep)

    jobs = load_sweep_jobs(sweep)
    save_root = Path(sweep.get("save_root", "logs/sweeps"))
    sweep_name = sweep.get("name") or Path(sweep_path).stem
    sweep_dir = save_root / f"{sweep_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    sweep_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Sweep %s: %s jobs -> %s", sweep_name, len(jobs), sweep_dir)
    for job in jobs:
        logger.info("  planned run %s overrides=%s", job["name"], flatten_job_overrides(job))

    if getattr(args, "dry_run", False):
        write_sweep_summary(
            [
                {"name": job["name"], "status": "planned", **flatten_job_overrides(job)}
                for job in jobs
            ],
            sweep_dir / "summary.csv",
        )
        print(f"Dry run: {len(jobs)} jobs would be written to {sweep_dir}")
        return None

    rows = []
    for job in jobs:
        cfg = deepcopy(base_cfg)
        apply_dotted_overrides(cfg, job)
        cfg.setdefault("logging", {})
        cfg["logging"]["save_dir"] = str(sweep_dir / job["name"])
        cfg["logging"]["unique_dir"] = False
        cfg["logging"]["run_name"] = job["name"]
        logger.info("Starting sweep run %s", job["name"])
        try:
            summary = run_experiment(cfg)
            row = {
                "name": job["name"],
                "status": "ok",
                **flatten_job_overrides(job),
                **{
                    key: summary.get(key)
                    for key in (
                        "save_dir",
                        "best_score",
                        "selection_metric",
                        "best_epoch",
                        "val_f1",
                        "test_f1",
                        "val_image_f1",
                        "test_image_f1",
                        "selected_weights",
                        "stopped_epoch",
                        "seed",
                    )
                },
            }
        except Exception as exc:
            logger.exception("Sweep run %s failed", job["name"])
            row = {
                "name": job["name"],
                "status": "failed",
                "error": str(exc),
                **flatten_job_overrides(job),
            }
        rows.append(row)
        write_sweep_summary(rows, sweep_dir / "summary.csv")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    best = pick_best_run(rows)
    if best:
        logger.info(
            "Best sweep run %s %s=%.4f test_f1=%s dir=%s",
            best.get("name"),
            best.get("selection_metric"),
            float(best["best_score"]),
            best.get("test_f1"),
            best.get("save_dir"),
        )
        print(
            f"Best run: {best.get('name')} "
            f"{best.get('selection_metric')}={best.get('best_score')} "
            f"test_f1={best.get('test_f1')} "
            f"({best.get('save_dir')})"
        )
    else:
        print(f"Sweep finished without a successful run. See {sweep_dir}")
    return best


def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    if args.sweep:
        run_sweep(args.sweep, args)
        return

    cfg = apply_overrides(load_config(args.config), args)
    run_experiment(cfg)


if __name__ == "__main__":
    main()
