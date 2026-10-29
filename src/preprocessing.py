from __future__ import annotations

import json
import random
import re
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from torchvision.transforms import functional as TF

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
DEFAULT_PREPARED_OUTPUT_ROOT = "data/dataset2_exp_prepared_flatfield"


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


def specimen_prefix(group_id: str) -> str:
    """Physical specimen identifier shared by sub-series ``(k-1)`` and ``(k-2)``."""
    return group_id.split("-")[0]


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


def flatfield_correct_grayscale(
    img: Image.Image,
    sigma: float = 64.0,
    contrast_gain: float = 4.0,
    preserve_mean: bool = True,
    cutoff_percent: float = 0.0,
    downscale: int = 8,
) -> Image.Image:
    """Remove low-frequency microscope vignetting in float32 while preserving true class brightness/contrast.

    When ``preserve_mean=True`` and ``contrast_gain > 0``, removes the smooth background
    illumination gradient ``bg`` and scales the local texture residual by a fixed
    ``contrast_gain`` around the image's mean intensity: ``mean(arr) + contrast_gain * (arr - bg)``.
    This eliminates corner vignetting without inflating noise in low-contrast ``clear`` gels.
    """
    gray = img.convert("L")
    arr = np.asarray(gray, dtype=np.float32)
    h, w = arr.shape

    if sigma > 0:
        scale = max(1, int(downscale))
        small_sigma = max(0.5, float(sigma) / scale)
        radius = max(1, int(round(3.0 * small_sigma)))
        ksize = 2 * radius + 1

        tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)
        small = F.interpolate(
            tensor,
            size=(max(1, h // scale), max(1, w // scale)),
            mode="area",
        )
        pad_h = min(radius, max(0, small.shape[-2] - 1))
        pad_w = min(radius, max(0, small.shape[-1] - 1))
        small_padded = F.pad(small, (pad_w, pad_w, pad_h, pad_h), mode="reflect")
        blurred = TF.gaussian_blur(
            small_padded,
            kernel_size=[ksize, ksize],
            sigma=[small_sigma, small_sigma],
        )
        if pad_h > 0 or pad_w > 0:
            blurred = blurred[
                :,
                :,
                pad_h : pad_h + small.shape[-2],
                pad_w : pad_w + small.shape[-1],
            ]
        bg = (
            F.interpolate(blurred, size=(h, w), mode="bicubic", align_corners=False)
            .squeeze(0)
            .squeeze(0)
            .numpy()
        )
        residual = arr - bg
    else:
        residual = arr - float(arr.mean())

    if preserve_mean and cutoff_percent <= 0:
        base = float(arr.mean()) if preserve_mean else 128.0
        stretched = np.clip(base + residual * float(contrast_gain), 0.0, 255.0)
        return Image.fromarray(stretched.round().astype(np.uint8), mode="L")

    signal = residual
    if cutoff_percent > 0:
        q_low, q_high = np.percentile(
            signal, [float(cutoff_percent), 100.0 - float(cutoff_percent)]
        )
    else:
        q_low, q_high = float(signal.min()), float(signal.max())

    span = max(float(q_high - q_low), 1e-3)
    stretched = np.clip((signal - q_low) / span * 255.0, 0.0, 255.0)
    return Image.fromarray(stretched.round().astype(np.uint8), mode="L")


def convert_to_grayscale(
    src_root: str | Path,
    dst_root: str | Path,
    flatfield_sigma: float = 64.0,
    contrast_gain: float = 4.0,
    preserve_mean: bool = True,
    cutoff_percent: float = 0.0,
    output_ext: str = ".png",
) -> None:
    src_root = Path(src_root)
    dst_root = Path(dst_root)

    for cls, images in collect_images(src_root).items():
        class_dir = src_root / cls

        for image_path in images:
            rel = image_path.relative_to(class_dir)
            if output_ext:
                rel = rel.with_suffix(output_ext)
            dst = dst_root / cls / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            with Image.open(image_path) as raw_img:
                if flatfield_sigma and flatfield_sigma > 0:
                    img = flatfield_correct_grayscale(
                        raw_img,
                        sigma=flatfield_sigma,
                        contrast_gain=contrast_gain,
                        preserve_mean=preserve_mean,
                        cutoff_percent=cutoff_percent,
                    )
                else:
                    img = ImageOps.autocontrast(
                        raw_img.convert("L"), cutoff=max(1.0, cutoff_percent)
                    )

            if dst.suffix.lower() == ".png":
                img.save(dst, format="PNG", compress_level=1)
            elif dst.suffix.lower() in {".jpg", ".jpeg"}:
                img.save(dst, quality=98, subsampling=0)
            else:
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


def _partition_group_buckets(
    buckets: Sequence[Tuple[str, List[LabeledPath]]],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    rng: random.Random,
) -> Dict[str, List[Tuple[str, List[LabeledPath]]]]:
    """Split group buckets into train/val/test balancing image counts and keeping every specimen in train."""
    n = len(buckets)
    shuffled = list(buckets)
    rng.shuffle(shuffled)

    if n <= 2:
        n_train, n_val, n_test = _split_counts(n, train_ratio, val_ratio)
        return {
            "train": shuffled[:n_train],
            "val": shuffled[n_train : n_train + n_val],
            "test": shuffled[n_train + n_val : n_train + n_val + n_test],
        }

    total_imgs = sum(len(items) for _, items in shuffled)
    target_val = val_ratio * total_imgs
    target_test = test_ratio * total_imgs
    target_train = train_ratio * total_imgs

    by_specimen: Dict[str, List[Tuple[str, List[LabeledPath]]]] = defaultdict(list)
    for group_id, items in shuffled:
        by_specimen[specimen_prefix(group_id)].append((group_id, items))

    best_score = float("inf")
    best_slices: Dict[str, List[Tuple[str, List[LabeledPath]]]] | None = None

    num_trials = min(256, max(16, n * 16))
    for _ in range(num_trials):
        train_locked: List[Tuple[str, List[LabeledPath]]] = []
        holdout_pool: List[Tuple[str, List[LabeledPath]]] = []
        specs = list(by_specimen.items())
        rng.shuffle(specs)
        for _, glist in specs:
            glist_copy = list(glist)
            rng.shuffle(glist_copy)
            train_locked.append(glist_copy[0])
            holdout_pool.extend(glist_copy[1:])

        rng.shuffle(holdout_pool)
        m = len(holdout_pool)
        if m < 2:
            break

        for i in range(1, m):
            for j in range(i + 1, m + 1):
                va_cand = holdout_pool[:i]
                te_cand = holdout_pool[i:j]
                tr_cand = train_locked + holdout_pool[j:]
                c_tr = sum(len(b[1]) for b in tr_cand)
                c_va = sum(len(b[1]) for b in va_cand)
                c_te = sum(len(b[1]) for b in te_cand)
                penalty = 0.0
                if len(va_cand) < 2 and m >= 4:
                    penalty += 10.0
                if len(te_cand) < 2 and m >= 4:
                    penalty += 10.0
                score = (
                    penalty
                    + abs(c_tr - target_train)
                    + 2.0 * abs(c_va - target_val)
                    + 2.0 * abs(c_te - target_test)
                )
                if score < best_score:
                    best_score = score
                    best_slices = {
                        "train": tr_cand,
                        "val": va_cand,
                        "test": te_cand,
                    }

    assert best_slices is not None
    return best_slices


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
        else:
            grouped: Dict[str, List[LabeledPath]] = defaultdict(list)
            for path, item_cls in items:
                grouped[sample_group_id(path)].append((path, item_cls))
            buckets = [
                (group_id, grouped[group_id])
                for group_id in sorted(grouped)
            ]
            slices = _partition_group_buckets(
                buckets,
                train_ratio=train_ratio,
                val_ratio=val_ratio,
                test_ratio=test_ratio,
                rng=rng,
            )

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
    flatfield_sigma: float = 64.0,
    contrast_gain: float = 4.0,
    preserve_mean: bool = True,
    cutoff_percent: float = 0.0,
    output_ext: str = ".png",
) -> dict:
    raw_root = Path(raw_root)
    output_root = Path(output_root)
    if output_root.exists():
        shutil.rmtree(output_root)
    grayscale_root = output_root / "_grayscale_tmp"

    convert_to_grayscale(
        raw_root,
        grayscale_root,
        flatfield_sigma=flatfield_sigma,
        contrast_gain=contrast_gain,
        preserve_mean=preserve_mean,
        cutoff_percent=cutoff_percent,
        output_ext=output_ext,
    )
    manifest = split_dataset(
        grayscale_root,
        output_root,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
        group_by=group_by,
    )
    shutil.rmtree(grayscale_root)
    return manifest


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
    import argparse

    parser = argparse.ArgumentParser(
        description="Preprocess raw cryogel micrographs (flat-field + contrast gain + grouped split)."
    )
    parser.add_argument(
        "--raw-root",
        default="data/dataset2_exp",
        help="Path to raw dataset directory",
    )
    parser.add_argument(
        "--output-root",
        default=DEFAULT_PREPARED_OUTPUT_ROOT,
        help="Path to output prepared dataset directory",
    )
    parser.add_argument(
        "--flatfield-sigma",
        type=float,
        default=64.0,
        help="Gaussian sigma for flat-field background subtraction (0 disables flat-field)",
    )
    parser.add_argument(
        "--contrast-gain",
        type=float,
        default=4.0,
        help="Fixed contrast amplification factor around mean brightness",
    )
    parser.add_argument(
        "--cutoff-percent",
        type=float,
        default=0.0,
        help="Percentile cutoff for per-image contrast stretching (0 uses fixed contrast-gain)",
    )
    parser.add_argument(
        "--output-ext",
        default=".png",
        help="Output image extension (.png for lossless, .jpg for high-quality JPEG)",
    )
    parser.add_argument(
        "--group-by",
        default="sample",
        choices=["sample", "none"],
        help="Group frames of the same sample capture into the same split",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for split")
    args = parser.parse_args()

    prepare_dataset(
        raw_root=args.raw_root,
        output_root=args.output_root,
        seed=args.seed,
        group_by=args.group_by,
        flatfield_sigma=args.flatfield_sigma,
        contrast_gain=args.contrast_gain,
        cutoff_percent=args.cutoff_percent,
        output_ext=args.output_ext,
    )
