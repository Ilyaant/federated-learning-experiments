from __future__ import annotations

import random
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import Config, get_path, save_config, validate_config
from .dataset import PerImagePatchSampler, TexturePatchDataset, limit_images_per_class
from .logging_utils import get_logger, save_json, setup_logger
from .model import build_model, count_parameters, resolve_device
from .preprocessing import (
    SPLIT_NAMES,
    find_classes,
    load_cifar100_splits,
    load_splits,
    write_split_manifest,
)
from .trainer import Trainer
from .transforms import (
    build_eval_transform,
    build_train_transform,
    describe_transform,
    needs_rotation_context,
)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _worker_init(worker_id: int) -> None:
    seed = (torch.initial_seed() + worker_id) % (2**32)
    random.seed(seed)
    np.random.seed(seed)


def make_run_dir(config: Config) -> Path:
    log_cfg = config.get("logging", {})
    save_dir = Path(log_cfg.get("save_dir", "logs/runs"))
    run_name = log_cfg.get("run_name", "run")
    if log_cfg.get("timestamp_dir", True):
        run_name = f"{run_name}_{datetime.now():%Y%m%d_%H%M%S}"
    run_dir = save_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def resolve_class_names(root: Path) -> list[str]:
    """Class folders in a fixed (sorted) order, from train/ or from the root."""
    train_root = root / "train"
    return find_classes(train_root if train_root.is_dir() else root)


def is_cifar100_dataset(ds_cfg: dict) -> bool:
    name = str(ds_cfg.get("name", "")).lower().replace("-", "").replace("_", "")
    if name == "cifar100":
        return True
    root = str(ds_cfg.get("root", "")).lower().replace("-", "").replace("_", "")
    return "cifar100" in root


def build_datasets(config: Config) -> Tuple[Dict[str, TexturePatchDataset], dict, list[str]]:
    ds_cfg = config.get("dataset", {})
    patch_cfg = config.get("patches", {})
    aug_cfg = config.get("augmentation", {})

    cifar100 = is_cifar100_dataset(ds_cfg)
    if cifar100:
        root = Path(ds_cfg.get("root", "data/cifar100"))
        samples, manifest, class_names = load_cifar100_splits(
            root,
            regroup=bool(ds_cfg.get("regroup", False)),
            group_by=ds_cfg.get("group_by", "sample"),
            train_ratio=float(ds_cfg.get("train_ratio", 0.7)),
            val_ratio=float(ds_cfg.get("val_ratio", 0.15)),
            test_ratio=float(ds_cfg.get("test_ratio", 0.15)),
            seed=int(ds_cfg.get("split_seed", 42)),
            download=bool(ds_cfg.get("download", True)),
        )
    else:
        root = Path(ds_cfg.get("root", "data/dataset2_exp_prepared"))
        class_names = resolve_class_names(root)
        if not class_names:
            raise FileNotFoundError(f"No class folders found under {root}")

        samples, manifest = load_splits(
            root,
            regroup=bool(ds_cfg.get("regroup", True)),
            group_by=ds_cfg.get("group_by", "sample"),
            train_ratio=float(ds_cfg.get("train_ratio", 0.7)),
            val_ratio=float(ds_cfg.get("val_ratio", 0.15)),
            test_ratio=float(ds_cfg.get("test_ratio", 0.15)),
            seed=int(ds_cfg.get("split_seed", 42)),
        )
    max_per_class = ds_cfg.get("max_images_per_class")
    samples = {split: limit_images_per_class(items, max_per_class) for split, items in samples.items()}

    grayscale = bool(ds_cfg.get("grayscale", False if cifar100 else True))
    channels = 1 if grayscale else 3
    slice_patches = False if cifar100 else bool(patch_cfg.get("enabled", True))
    patch_size = int(patch_cfg.get("size") or 224)
    normalize_cfg = patch_cfg.get("normalize")

    # CIFAR frames stay whole images; scale them to patches.size so the
    # backbone sees its training resolution instead of the native 32x32.
    resize_to = patch_size if cifar100 else None
    common = dict(
        patch_size=patch_size,
        include_edges=bool(patch_cfg.get("include_edges", True)),
        grayscale=grayscale,
        pad_mode=patch_cfg.get("pad_mode", "reflect"),
        cache_size=int(patch_cfg.get("cache_size", 512)),
        slice_patches=slice_patches,
        resize_to=resize_to,
    )
    transform_size = patch_size
    if not slice_patches and resize_to is None and samples["train"]:
        first_src = samples["train"][0][0]
        if isinstance(first_src, np.ndarray):
            transform_size = int(first_src.shape[1])
        elif isinstance(first_src, Path):
            from .patches import read_image_size
            transform_size = int(read_image_size(first_src)[0])

    train_transform = build_train_transform(aug_cfg, normalize_cfg, transform_size, channels)
    eval_transform = build_eval_transform(normalize_cfg, channels)

    datasets = {
        "train": TexturePatchDataset(
            samples["train"],
            overlap=float(patch_cfg.get("train_overlap", 0.0)),
            transform=train_transform,
            rotation_context=needs_rotation_context(aug_cfg),
            **common,
        )
    }
    for split in ("val", "test"):
        datasets[split] = TexturePatchDataset(
            samples[split],
            overlap=float(patch_cfg.get("eval_overlap", 0.0)),
            transform=eval_transform,
            **common,
        )
    # Train images without augmentation, for optional end-of-run scoring.
    datasets["train_eval"] = TexturePatchDataset(
        samples["train"],
        overlap=float(patch_cfg.get("eval_overlap", 0.0)),
        transform=eval_transform,
        **common,
    )
    return datasets, manifest, class_names


def _shared_memory_bytes() -> int:
    try:
        return int(shutil.disk_usage("/dev/shm").total)
    except OSError:
        return 0


def _sample_bytes(dataset: TexturePatchDataset) -> int:
    channels = 1 if dataset.grayscale else 3
    side = int(dataset.patch_size)
    return channels * side * side * 4


def _fit_workers(requested: int, batch_size: int, sample_bytes: int, shm_bytes: int) -> Tuple[int, int]:
    """Pick workers and prefetch so queued batches stay inside /dev/shm.

    DataLoader workers place prefetched batches in shared memory. A 224x224
    batch is large enough to overflow the 64MB default and kill a worker
    with a bus error.
    """
    requested = max(0, int(requested))
    if requested == 0 or shm_bytes <= 0:
        return requested, 2
    budget = int(shm_bytes * 0.5)
    batch_bytes = max(1, int(batch_size)) * max(1, int(sample_bytes))
    for prefetch in (2, 1):
        workers = min(requested, budget // (batch_bytes * prefetch))
        if workers >= 1:
            return int(workers), prefetch
    return 0, 2


def build_loaders(
    config: Config,
    datasets: Dict[str, TexturePatchDataset],
    seed: int,
) -> Tuple[Dict[str, DataLoader], PerImagePatchSampler]:
    train_cfg = config.get("train", {})
    eval_cfg = config.get("evaluation", {})
    patch_cfg = config.get("patches", {})
    pin_memory = torch.cuda.is_available()

    train_sampler = PerImagePatchSampler(
        datasets["train"],
        patches_per_image=patch_cfg.get("train_patches_per_image"),
        seed=seed,
        shuffle=True,
    )
    generator = torch.Generator()
    generator.manual_seed(seed)

    requested_train = int(train_cfg.get("num_workers", 0))
    requested_eval = int(eval_cfg.get("num_workers", requested_train))
    sample_bytes = _sample_bytes(datasets["train"])
    shm_bytes = _shared_memory_bytes()
    train_batch = int(train_cfg.get("batch_size", 32))
    eval_batch = int(eval_cfg.get("batch_size", 64))
    train_workers, train_prefetch = _fit_workers(requested_train, train_batch, sample_bytes, shm_bytes)
    eval_workers, eval_prefetch = _fit_workers(requested_eval, eval_batch, sample_bytes, shm_bytes)
    if train_workers != requested_train or eval_workers != requested_eval:
        get_logger().info(
            "DataLoader workers reduced to fit /dev/shm (%dMB): "
            "train %d -> %d (prefetch %d), eval %d -> %d (prefetch %d)",
            shm_bytes // (1024 * 1024),
            requested_train, train_workers, train_prefetch if train_workers else 0,
            requested_eval, eval_workers, eval_prefetch if eval_workers else 0,
        )

    train_kwargs = {}
    if train_workers > 0:
        train_kwargs["prefetch_factor"] = train_prefetch
    eval_kwargs = {}
    if eval_workers > 0:
        eval_kwargs["prefetch_factor"] = eval_prefetch

    loaders = {
        "train": DataLoader(
            datasets["train"],
            batch_size=train_batch,
            sampler=train_sampler,
            num_workers=train_workers,
            pin_memory=pin_memory,
            drop_last=False,
            persistent_workers=train_workers > 0,
            worker_init_fn=_worker_init,
            generator=generator,
            **train_kwargs,
        )
    }
    for split in ("val", "test"):
        loaders[split] = DataLoader(
            datasets[split],
            batch_size=eval_batch,
            shuffle=False,
            num_workers=eval_workers,
            pin_memory=pin_memory,
            persistent_workers=False,
            worker_init_fn=_worker_init,
            **eval_kwargs,
        )

    # Train scored in eval mode: a fixed (seeded, never reshuffled) subset of
    # grid patches per image, ordered by image so the decode cache is reused.
    train_eval_sampler = PerImagePatchSampler(
        datasets["train_eval"],
        patches_per_image=eval_cfg.get("train_eval_patches_per_image"),
        seed=seed,
        shuffle=False,
    )
    loaders["train_eval"] = DataLoader(
        datasets["train_eval"],
        batch_size=eval_batch,
        sampler=train_eval_sampler,
        num_workers=eval_workers,
        pin_memory=pin_memory,
        persistent_workers=False,
        worker_init_fn=_worker_init,
        **eval_kwargs,
    )
    return loaders, train_sampler


def _log_dataset_summary(logger, name: str, dataset: TexturePatchDataset, class_names: list[str]) -> None:
    images = {class_names[k]: v for k, v in dataset.class_distribution().items()}
    patches = {class_names[k]: v for k, v in dataset.patch_class_distribution().items()}
    logger.info(
        "%-10s %d images, %d patches | images per class %s | patches per class %s",
        name, dataset.num_images, dataset.num_patches, images, patches,
    )


def run_experiment(config: Config, dry_run: bool = False) -> Dict[str, object]:
    problems = validate_config(config)
    if problems:
        raise ValueError("Invalid config:\n  - " + "\n  - ".join(problems))

    seed = int(config.get("seed", 42))
    seed_everything(seed)

    run_dir = make_run_dir(config)
    logger = setup_logger(run_dir / get_path(config, "logging.log_file", "experiment.log"))
    save_config(config, run_dir / "config.yaml")

    device = resolve_device(config.get("device", "auto"))
    logger.info("Run directory: %s", run_dir)
    logger.info("Device: %s | torch %s | threads %d", device, torch.__version__, torch.get_num_threads())
    logger.info("Seed: %d", seed)

    started = time.time()
    datasets, manifest, class_names = build_datasets(config)
    write_split_manifest(manifest, run_dir / "split_manifest.json")
    logger.info("Classes (%d): %s", len(class_names), class_names)
    logger.info(
        "Split: regroup=%s group_by=%s counts=%s",
        manifest.get("regroup"), manifest.get("group_by"), manifest.get("counts"),
    )
    if datasets["train"].slice_patches:
        logger.info(
            "Patches: size %d | train overlap %.2f | eval overlap %.2f | train patches/image/epoch %s",
            datasets["train"].patch_size, datasets["train"].grid.overlap,
            datasets["val"].grid.overlap, get_path(config, "patches.train_patches_per_image"),
        )
    else:
        if datasets["train"].resize_to:
            native = datasets["train"].image_sizes[0]
            logger.info(
                "Patches: disabled (whole images resized %dx%d -> %dx%d)",
                native[0], native[1],
                datasets["train"].resize_to, datasets["train"].resize_to,
            )
        else:
            logger.info(
                "Patches: disabled (whole images %dx%d)",
                datasets["train"].patch_size, datasets["train"].patch_size,
            )
    logger.info("Train transform: %s", describe_transform(datasets["train"].transform))
    logger.info("Eval transform:  %s", describe_transform(datasets["val"].transform))
    for split in SPLIT_NAMES:
        _log_dataset_summary(logger, split, datasets[split], class_names)
    logger.info("Datasets built in %.1fs", time.time() - started)

    loaders, train_sampler = build_loaders(config, datasets, seed)
    logger.info(
        "Train patches per epoch: %d (%d batches of %d)",
        len(train_sampler), len(loaders["train"]), get_path(config, "train.batch_size", 32),
    )
    logger.info(
        "Train eval (no aug): every %s epoch(s) on %d patches (%s per image)",
        get_path(config, "evaluation.eval_train_every", 0),
        len(loaders["train_eval"].sampler),
        get_path(config, "evaluation.train_eval_patches_per_image") or "all",
    )

    model = build_model(
        config.get("model", {}),
        num_classes=len(class_names),
        in_chans=1 if datasets["train"].grayscale else 3,
    )
    total, trainable = count_parameters(model)
    logger.info(
        "Model: %s | pretrained=%s | params %.2fM (trainable %.2fM)",
        get_path(config, "model.name"), get_path(config, "model.pretrained"), total / 1e6, trainable / 1e6,
    )

    if dry_run:
        logger.info("Dry run: datasets and model built, skipping training")
        return {"run_dir": str(run_dir), "dry_run": True}

    trainer = Trainer(
        config=config,
        model=model,
        class_names=class_names,
        train_loader=loaders["train"],
        train_sampler=train_sampler,
        val_loader=loaders["val"],
        test_loader=loaders["test"],
        train_eval_loader=loaders["train_eval"],
        run_dir=run_dir,
        device=device,
    )
    fit_result = trainer.fit()
    logger.info("Training finished: best %s=%.4f at epoch %s", trainer.selection_metric, fit_result["best_value"], fit_result["best_epoch"])

    summary = trainer.final_evaluation(
        "model_best.pt",
        include_train=bool(get_path(config, "evaluation.eval_train_at_end", False)),
    )
    summary["run_dir"] = str(run_dir)
    summary["total_time_sec"] = time.time() - started
    save_json(summary, run_dir / "summary.json")
    logger.info("Total time: %.0fs | results in %s", summary["total_time_sec"], run_dir)
    return summary
