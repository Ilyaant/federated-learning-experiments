from __future__ import annotations

from collections import defaultdict
from functools import lru_cache
import random
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
import torchvision
from PIL import Image, ImageOps

VALID_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
}

SPLIT_NAMES = ("train", "val", "test")
SampleSource = Union[Path, np.ndarray]
Sample = Tuple[SampleSource, int]


def find_classes(dataset_root: Path) -> List[str]:
    return sorted(
        directory.name
        for directory in dataset_root.iterdir()
        if directory.is_dir()
    )


def collect_images(dataset_root: Path) -> Dict[str, List[Path]]:
    images = {}

    for cls in find_classes(dataset_root):
        class_dir = dataset_root / cls
        files = sorted(
            file
            for file in class_dir.rglob("*")
            if file.suffix.lower() in VALID_EXTENSIONS
        )
        images[cls] = files

    return images


def _copy_preserving_structure(
    src: Path,
    class_dir: Path,
    dst_root: Path,
    split: str,
    cls: str,
) -> None:
    rel = src.relative_to(class_dir)
    dst = dst_root / split / cls / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def convert_to_grayscale(
    src_root: str | Path,
    dst_root: str | Path,
) -> None:
    src_root = Path(src_root)
    dst_root = Path(dst_root)

    for cls, images in collect_images(src_root).items():
        class_dir = src_root / cls

        for image_path in images:
            rel = image_path.relative_to(class_dir)
            dst = dst_root / cls / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            img = Image.open(image_path).convert("L")
            img = ImageOps.autocontrast(img, cutoff=1)
            img.save(dst)


def split_dataset(
    dataset_root: str | Path,
    output_root: str | Path,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> None:
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("train + val + test must equal 1")

    random.seed(seed)
    dataset_root = Path(dataset_root)
    output_root = Path(output_root)

    for cls, images in collect_images(dataset_root).items():
        class_dir = dataset_root / cls
        shuffled = images.copy()
        random.shuffle(shuffled)

        n = len(shuffled)
        n_train = int(train_ratio * n)
        n_val = int(val_ratio * n)

        splits = {
            "train": shuffled[:n_train],
            "val": shuffled[n_train : n_train + n_val],
            "test": shuffled[n_train + n_val :],
        }

        for split, split_images in splits.items():
            for image in split_images:
                _copy_preserving_structure(
                    image,
                    class_dir,
                    output_root,
                    split,
                    cls,
                )


def prepare_dataset(
    raw_root: str | Path,
    output_root: str | Path,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> None:
    raw_root = Path(raw_root)
    output_root = Path(output_root)
    grayscale_root = output_root / "_grayscale_tmp"

    if grayscale_root.exists():
        shutil.rmtree(grayscale_root)

    convert_to_grayscale(raw_root, grayscale_root)
    split_dataset(
        grayscale_root,
        output_root,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )
    shutil.rmtree(grayscale_root)


def _resolve_cifar100_root(dataset_root: str | Path) -> Path | None:
    root = Path(dataset_root)
    if root.name == "cifar-100-python" and root.is_dir():
        return root.parent
    if root.is_dir() and (root / "cifar-100-python").is_dir() and not (root / "train").is_dir():
        return root
    root_norm = str(root).lower().replace("-", "").replace("_", "")
    if "cifar100" in root_norm:
        return root
    return None


def is_cifar100_dataset(dataset_cfg_or_root: dict | str | Path) -> bool:
    if isinstance(dataset_cfg_or_root, dict):
        name = (
            str(dataset_cfg_or_root.get("name", ""))
            .lower()
            .replace("-", "")
            .replace("_", "")
        )
        if name:
            return name == "cifar100"
        root = dataset_cfg_or_root.get("root", "")
        return _resolve_cifar100_root(root) is not None
    return _resolve_cifar100_root(dataset_cfg_or_root) is not None


def _split_counts(
    n: int,
    train_ratio: float,
    val_ratio: float,
) -> Tuple[int, int, int]:
    if n <= 0:
        return 0, 0, 0
    if n == 1:
        return 1, 0, 0
    if n == 2:
        return 1, 1, 0

    n_train = int(train_ratio * n)
    n_val = int(val_ratio * n)
    n_test = n - n_train - n_val

    n_train = max(n_train, 1)
    n_val = max(n_val, 1)
    n_test = max(n_test, 1)

    while n_train + n_val + n_test > n:
        if n_train >= n_val and n_train >= n_test and n_train > 1:
            n_train -= 1
        elif n_test >= n_val and n_test > 1:
            n_test -= 1
        elif n_val > 1:
            n_val -= 1
        else:
            break

    n_train += n - (n_train + n_val + n_test)
    return n_train, n_val, n_test


@lru_cache(maxsize=4)
def _cached_cifar100_splits(
    cifar_root: str,
    regroup: bool = False,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    download: bool = False,
) -> Dict[str, List[Tuple[np.ndarray, int]]]:
    train_ds = torchvision.datasets.CIFAR100(
        root=cifar_root,
        train=True,
        download=download,
    )
    test_ds = torchvision.datasets.CIFAR100(
        root=cifar_root,
        train=False,
        download=download,
    )

    rng = random.Random(seed)
    assigned: Dict[str, List[Tuple[np.ndarray, int]]] = {
        split: [] for split in SPLIT_NAMES
    }

    if regroup:
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
            raise ValueError("train + val + test must equal 1")
        per_class: Dict[int, List[Tuple[np.ndarray, int]]] = defaultdict(list)
        for img, label in zip(train_ds.data, train_ds.targets):
            per_class[int(label)].append((img, int(label)))
        for img, label in zip(test_ds.data, test_ds.targets):
            per_class[int(label)].append((img, int(label)))

        for cls in sorted(per_class):
            items = list(per_class[cls])
            rng.shuffle(items)
            n_train, n_val, n_test = _split_counts(
                len(items),
                train_ratio,
                val_ratio,
            )
            assigned["train"].extend(items[:n_train])
            assigned["val"].extend(items[n_train : n_train + n_val])
            assigned["test"].extend(
                items[n_train + n_val : n_train + n_val + n_test]
            )
    else:
        per_class = defaultdict(list)
        for img, label in zip(train_ds.data, train_ds.targets):
            per_class[int(label)].append((img, int(label)))

        val_frac = val_ratio if 0.0 < val_ratio < 1.0 else 0.15
        for cls in sorted(per_class):
            items = list(per_class[cls])
            rng.shuffle(items)
            n_val = (
                max(1, int(round(len(items) * val_frac)))
                if len(items) > 1
                else 0
            )
            n_train = len(items) - n_val
            assigned["train"].extend(items[:n_train])
            assigned["val"].extend(items[n_train:])

        for img, label in zip(test_ds.data, test_ds.targets):
            assigned["test"].append((img, int(label)))

    return assigned


def load_cifar100_splits(
    dataset_root: str | Path = "data",
    regroup: bool = False,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    download: bool = False,
) -> Dict[str, List[Tuple[np.ndarray, int]]]:
    """Load CIFAR-100 from ``dataset_root`` and split into train/val/test."""
    resolved = _resolve_cifar100_root(dataset_root)
    cifar_root = resolved if resolved is not None else Path(dataset_root)
    cached = _cached_cifar100_splits(
        str(cifar_root.resolve()),
        regroup=bool(regroup),
        train_ratio=float(train_ratio),
        val_ratio=float(val_ratio),
        test_ratio=float(test_ratio),
        seed=int(seed),
        download=bool(download),
    )
    return {split: list(items) for split, items in cached.items()}


def load_split(
    dataset_root: str | Path,
    split: str,
    *,
    dataset_name: str | None = None,
    regroup: bool = False,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    download: bool = False,
) -> List[Sample]:
    if split not in SPLIT_NAMES:
        raise ValueError(f"Unknown split: {split}")

    is_cifar = (
        str(dataset_name).lower().replace("-", "").replace("_", "") == "cifar100"
        if dataset_name
        else is_cifar100_dataset(dataset_root)
    )
    if is_cifar:
        splits = load_cifar100_splits(
            dataset_root=dataset_root,
            regroup=regroup,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            seed=seed,
            download=download,
        )
        return list(splits[split])

    split_root = Path(dataset_root) / split
    classes = find_classes(split_root)
    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}

    samples: List[Sample] = []
    for cls in classes:
        class_dir = split_root / cls
        for file in sorted(
            f
            for f in class_dir.rglob("*")
            if f.suffix.lower() in VALID_EXTENSIONS
        ):
            samples.append((file, class_to_idx[cls]))

    return samples


if __name__ == "__main__":
    prepare_dataset(
        raw_root="data/dataset2_exp",
        output_root="data/dataset2_exp_prepared"
    )
