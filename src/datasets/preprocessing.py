from __future__ import annotations

import random
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image

VALID_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
}


def find_classes(dataset_root: Path) -> List[str]:
    return sorted(
        [
            directory.name
            for directory in dataset_root.iterdir()
            if directory.is_dir()
        ]
    )


def collect_images(
    dataset_root: Path,
) -> Dict[str, List[Path]]:
    images = {}

    for cls in find_classes(dataset_root):
        class_dir = dataset_root / cls

        files = [
            file
            for file in class_dir.rglob("*")
            if file.suffix.lower() in VALID_EXTENSIONS
        ]

        files.sort()

        images[cls] = files

    return images


def convert_to_grayscale(
    src_root: str | Path,
    dst_root: str | Path,
):
    src_root = Path(src_root)
    dst_root = Path(dst_root)

    dataset = collect_images(src_root)

    for cls, images in dataset.items():
        output_dir = dst_root / cls
        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        for image_path in images:
            image = Image.open(image_path).convert("L")

            image.save(
                output_dir / image_path.name
            )


def split_dataset(
    dataset_root: str | Path,
    output_root: str | Path,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
):
    if abs(
        train_ratio + val_ratio + test_ratio - 1
    ) > 1e-6:
        raise ValueError(
            "train + val + test must equal 1"
        )

    random.seed(seed)

    dataset_root = Path(dataset_root)
    output_root = Path(output_root)

    dataset = collect_images(dataset_root)

    for split in [
        "train",
        "val",
        "test",
    ]:
        for cls in dataset:
            (
                output_root
                / split
                / cls
            ).mkdir(
                parents=True,
                exist_ok=True,
            )

    for cls, images in dataset.items():
        images = images.copy()

        random.shuffle(images)

        n = len(images)

        n_train = int(train_ratio * n)
        n_val = int(val_ratio * n)

        train = images[:n_train]
        val = images[
            n_train : n_train + n_val
        ]
        test = images[
            n_train + n_val :
        ]

        mapping = {
            "train": train,
            "val": val,
            "test": test,
        }

        for split, split_images in mapping.items():
            for image in split_images:
                shutil.copy2(
                    image,
                    output_root
                    / split
                    / cls
                    / image.name,
                )


def prepare_dataset(
    raw_root: str | Path,
    output_root: str | Path,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
):
    raw_root = Path(raw_root)
    output_root = Path(output_root)

    grayscale_root = (
        output_root / "_grayscale_tmp"
    )

    if grayscale_root.exists():
        shutil.rmtree(grayscale_root)

    convert_to_grayscale(
        raw_root,
        grayscale_root,
    )

    split_dataset(
        grayscale_root,
        output_root,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )

    shutil.rmtree(grayscale_root)


def load_split(
    dataset_root: str | Path,
    split: str,
) -> List[Tuple[Path, int]]:
    dataset_root = Path(dataset_root)

    split_root = dataset_root / split

    classes = find_classes(split_root)

    class_to_idx = {
        cls: idx
        for idx, cls in enumerate(classes)
    }

    samples = []

    for cls in classes:
        class_dir = split_root / cls

        files = [
            file
            for file in class_dir.rglob("*")
            if file.suffix.lower() in VALID_EXTENSIONS
        ]

        files.sort()

        for file in files:
            samples.append(
                (
                    file,
                    class_to_idx[cls],
                )
            )

    return samples
