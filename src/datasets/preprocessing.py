from __future__ import annotations

import random
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image, ImageOps

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


def load_split(
    dataset_root: str | Path,
    split: str,
) -> List[Tuple[Path, int]]:
    split_root = Path(dataset_root) / split
    classes = find_classes(split_root)
    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}

    samples = []
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
