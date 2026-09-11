from __future__ import annotations

import json
import random
import re
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

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
LabeledPath = Tuple[Path, str]
LabeledSample = Tuple[Path, int]

_FRAME_SUFFIX = re.compile(r"-\d+-\d+$")


def find_classes(dataset_root: Path) -> List[str]:
    return sorted(
        directory.name
        for directory in dataset_root.iterdir()
        if directory.is_dir() and not directory.name.startswith("_")
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


def dataset_has_splits(dataset_root: str | Path) -> bool:
    root = Path(dataset_root)
    return (root / "train").is_dir()


def sample_group_id(path: str | Path) -> str:
    """Group frames of the same capture so they stay in one split.

    Names like ``15_11_17(2-1)пп(20)-1-3.jpg`` share the prefix before the
    trailing ``-frame-crop`` indices.
    """
    stem = Path(path).stem
    if _FRAME_SUFFIX.search(stem):
        return stem.rsplit("-", 2)[0]
    return stem


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


def collect_labeled_paths(dataset_root: str | Path) -> List[LabeledPath]:
    """Return ``(path, class_name)`` for every image under ``dataset_root``.

    Accepts either class folders or an existing train/val/test layout.
    """
    root = Path(dataset_root)
    labeled: List[LabeledPath] = []

    if dataset_has_splits(root):
        for split in SPLIT_NAMES:
            split_root = root / split
            if not split_root.is_dir():
                continue
            for cls, files in collect_images(split_root).items():
                labeled.extend((path, cls) for path in files)
        return labeled

    for cls, files in collect_images(root).items():
        labeled.extend((path, cls) for path in files)
    return labeled


def split_labeled_by_group(
    labeled: Sequence[LabeledPath],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    group_by: str = "sample",
) -> Tuple[Dict[str, List[LabeledPath]], dict]:
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("train + val + test must equal 1")
    if group_by not in {"sample", "none"}:
        raise ValueError("group_by must be 'sample' or 'none'")

    rng = random.Random(seed)
    per_class: Dict[str, List[LabeledPath]] = defaultdict(list)
    for path, cls in labeled:
        per_class[cls].append((path, cls))

    assigned: Dict[str, List[LabeledPath]] = {split: [] for split in SPLIT_NAMES}
    groups_by_split: Dict[str, Dict[str, List[str]]] = {
        split: defaultdict(list) for split in SPLIT_NAMES
    }

    for cls in sorted(per_class):
        items = list(per_class[cls])
        if group_by == "none":
            buckets: List[Tuple[str, List[LabeledPath]]] = [
                (str(path), [(path, item_cls)]) for path, item_cls in items
            ]
        else:
            grouped: Dict[str, List[LabeledPath]] = defaultdict(list)
            for path, item_cls in items:
                grouped[sample_group_id(path)].append((path, item_cls))
            buckets = [
                (group_id, grouped[group_id])
                for group_id in sorted(grouped)
            ]

        rng.shuffle(buckets)
        n_train, n_val, n_test = _split_counts(
            len(buckets),
            train_ratio,
            val_ratio,
        )
        slices = {
            "train": buckets[:n_train],
            "val": buckets[n_train : n_train + n_val],
            "test": buckets[n_train + n_val : n_train + n_val + n_test],
        }
        for split, split_buckets in slices.items():
            for group_id, group_items in split_buckets:
                assigned[split].extend(group_items)
                groups_by_split[split][cls].append(group_id)

    manifest = {
        "group_by": group_by,
        "split_seed": seed,
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
        "counts": {
            split: {
                "images": len(assigned[split]),
                "groups": sum(len(v) for v in groups_by_split[split].values()),
            }
            for split in SPLIT_NAMES
        },
        "groups": {
            split: {cls: ids for cls, ids in groups_by_split[split].items()}
            for split in SPLIT_NAMES
        },
    }
    return assigned, manifest


def _to_indexed_samples(
    assigned: Dict[str, List[LabeledPath]],
) -> Dict[str, List[LabeledSample]]:
    classes = sorted({cls for items in assigned.values() for _, cls in items})
    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
    return {
        split: sorted(
            ((path, class_to_idx[cls]) for path, cls in items),
            key=lambda item: (item[1], str(item[0])),
        )
        for split, items in assigned.items()
    }


def load_splits(
    dataset_root: str | Path,
    regroup: bool = True,
    group_by: str = "sample",
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[Dict[str, List[LabeledSample]], dict]:
    """Load train/val/test samples, optionally regrouped without file leakage."""
    root = Path(dataset_root)

    if not regroup and dataset_has_splits(root):
        samples = {split: load_split(root, split) for split in SPLIT_NAMES}
        manifest = {
            "group_by": "none",
            "regroup": False,
            "counts": {
                split: {"images": len(items), "groups": None}
                for split, items in samples.items()
            },
        }
        return samples, manifest

    labeled = collect_labeled_paths(root)
    if not labeled:
        raise FileNotFoundError(f"No images found under {root}")

    assigned, manifest = split_labeled_by_group(
        labeled,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
        group_by=group_by,
    )
    manifest["regroup"] = True
    return _to_indexed_samples(assigned), manifest


def write_split_manifest(manifest: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2, ensure_ascii=False)


def split_dataset(
    dataset_root: str | Path,
    output_root: str | Path,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    group_by: str = "sample",
) -> dict:
    dataset_root = Path(dataset_root)
    output_root = Path(output_root)
    labeled = collect_labeled_paths(dataset_root)
    assigned, manifest = split_labeled_by_group(
        labeled,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
        group_by=group_by,
    )

    class_dirs = {}
    if dataset_has_splits(dataset_root):
        for path, cls in labeled:
            class_dirs.setdefault(cls, path.parent)
    else:
        for cls in find_classes(dataset_root):
            class_dirs[cls] = dataset_root / cls

    for split, items in assigned.items():
        for path, cls in items:
            _copy_preserving_structure(
                path,
                class_dirs[cls],
                output_root,
                split,
                cls,
            )

    write_split_manifest(manifest, output_root / "split_manifest.json")
    return manifest


def prepare_dataset(
    raw_root: str | Path,
    output_root: str | Path,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    group_by: str = "sample",
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
        group_by=group_by,
    )
    shutil.rmtree(grayscale_root)


def load_split(
    dataset_root: str | Path,
    split: str,
) -> List[LabeledSample]:
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


def groups_leak_across_splits(
    assigned: Dict[str, Iterable[LabeledPath]],
) -> List[str]:
    membership: Dict[str, set] = defaultdict(set)
    for split, items in assigned.items():
        for path, cls in items:
            membership[f"{cls}/{sample_group_id(path)}"].add(split)
    return sorted(key for key, splits in membership.items() if len(splits) > 1)


if __name__ == "__main__":
    prepare_dataset(
        raw_root="data/dataset2_exp",
        output_root="data/dataset2_exp_prepared",
    )
