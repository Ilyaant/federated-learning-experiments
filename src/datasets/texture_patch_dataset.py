from __future__ import annotations

import math
from collections import OrderedDict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF
from torchvision.transforms.functional import InterpolationMode

from .patch_extractor import PatchExtractor

Sample = Tuple[Path, int]
PatchEntry = Tuple[int, int, int, str, float]
AugmentationVariant = Tuple[str, float]


class TexturePatchDataset(Dataset):
    """
    Lazy patch dataset for federated learning.

    Stores image paths and patch coordinates; loads images on demand.
    Supports optional epoch subsampling (training) and LRU image caching.
    """

    def __init__(
        self,
        samples: Sequence[Tuple[str | Path, int]],
        patch_size: int = 224,
        overlap: float = 0.5,
        grayscale: bool = True,
        normalize: bool = True,
        transform: Optional[Callable] = None,
        epoch_fraction: float = 1.0,
        balanced_per_image: bool = True,
        with_replacement: bool = False,
        seed: int = 42,
        cache_size: int = 64,
        augmentation: Optional[dict] = None,
    ):
        self.samples: List[Sample] = [
            (Path(path), int(label)) for path, label in samples
        ]
        self.transform = transform
        self.epoch_fraction = epoch_fraction
        self.balanced_per_image = balanced_per_image
        self.with_replacement = with_replacement
        self.seed = seed
        self.cache_size = cache_size
        self.augmentation = dict(augmentation or {})

        self.extractor = PatchExtractor(
            patch_size=patch_size,
            overlap=overlap,
            grayscale=grayscale,
            normalize=normalize,
        )

        self._all_patches: List[PatchEntry] = []
        self._active_patches: List[PatchEntry] = []
        self._image_cache: OrderedDict[str, torch.Tensor] = OrderedDict()
        self._augmentation_variants = self._build_augmentation_variants()

        self._build_patch_index()
        self.set_epoch(0)

    def _build_augmentation_variants(self) -> List[AugmentationVariant]:
        if not self.augmentation.get("enabled", False):
            return [("original", 0.0)]

        variants: List[AugmentationVariant] = []
        if self.augmentation.get("include_original", True):
            variants.append(("original", 0.0))

        for angle in self.augmentation.get("rotation_angles", [-15.0, 15.0]):
            variants.append(("rotation", float(angle)))

        blur_cfg = self.augmentation.get("blur", {})
        if blur_cfg.get("enabled", True):
            kernel_size = int(blur_cfg.get("kernel_size", 5))
            sigma = float(blur_cfg.get("sigma", 1.0))
            if kernel_size <= 0 or kernel_size % 2 == 0:
                raise ValueError("blur kernel_size must be a positive odd number")
            if sigma <= 0:
                raise ValueError("blur sigma must be positive")
            variants.append(("blur", sigma))

        noise_cfg = self.augmentation.get("noise", {})
        if noise_cfg.get("enabled", True):
            std = float(noise_cfg.get("std", 0.05))
            if std <= 0:
                raise ValueError("noise std must be positive")
            variants.append(("noise", std))

        if not variants:
            raise ValueError("augmentation must define at least one patch variant")

        return variants

    def _build_patch_index(self) -> None:
        self._all_patches.clear()

        for image_idx, (path, _) in enumerate(self.samples):
            for x, y in self.extractor.coordinates(path):
                for kind, value in self._augmentation_variants:
                    self._all_patches.append((image_idx, x, y, kind, value))

    def set_epoch(self, epoch: int) -> None:
        if self.epoch_fraction >= 1.0:
            self._active_patches = list(self._all_patches)
            return

        rng = np.random.default_rng(self.seed + epoch)
        active: List[PatchEntry] = []

        if self.balanced_per_image:
            by_image: Dict[int, List[PatchEntry]] = {}
            for entry in self._all_patches:
                by_image.setdefault(entry[0], []).append(entry)

            for entries in by_image.values():
                k = max(1, math.ceil(len(entries) * self.epoch_fraction))
                indices = rng.choice(
                    len(entries),
                    size=k if self.with_replacement else min(k, len(entries)),
                    replace=self.with_replacement,
                )
                active.extend(entries[int(i)] for i in indices)
        else:
            total = len(self._all_patches)
            k = max(1, math.ceil(total * self.epoch_fraction))
            indices = rng.choice(
                total,
                size=k if self.with_replacement else min(k, total),
                replace=self.with_replacement,
            )
            active = [self._all_patches[int(i)] for i in indices]

        rng.shuffle(active)
        self._active_patches = active

    def _load_image(self, path: Path) -> torch.Tensor:
        """Load, normalize and cache the full image tensor, so that
        extracting each of its patches is a cheap slicing operation."""
        key = str(path)

        if key in self._image_cache:
            self._image_cache.move_to_end(key)
            return self._image_cache[key]

        tensor = self.extractor.load_image(path)
        self._image_cache[key] = tensor

        if len(self._image_cache) > self.cache_size:
            self._image_cache.popitem(last=False)

        return tensor

    def __len__(self) -> int:
        return len(self._active_patches)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        image_idx, x, y, augmentation, value = self._active_patches[idx]
        path, label = self.samples[image_idx]

        patch = self.extractor.extract_patch(
            self._load_image(path),
            x,
            y,
        )
        patch = self._augment_patch(
            patch,
            augmentation,
            value,
            image_idx,
            x,
            y,
        )

        if self.transform is not None:
            patch = self.transform(patch)

        return {
            "image": patch,
            "label": label,
            "image_id": image_idx,
        }

    def _augment_patch(
        self,
        patch: torch.Tensor,
        kind: str,
        value: float,
        image_idx: int,
        x: int,
        y: int,
    ) -> torch.Tensor:
        if kind == "original":
            return patch
        if kind == "rotation":
            return TF.rotate(
                patch,
                angle=value,
                interpolation=InterpolationMode.BILINEAR,
            )
        if kind == "blur":
            kernel_size = int(
                self.augmentation.get("blur", {}).get("kernel_size", 5)
            )
            return TF.gaussian_blur(
                patch,
                kernel_size=[kernel_size, kernel_size],
                sigma=[value, value],
            )
        if kind == "noise":
            # Derive a stable per-patch seed so validation and test results
            # do not depend on DataLoader worker count or access order.
            noise_seed = (
                self.seed * 1_000_003
                + image_idx * 100_003
                + x * 1_009
                + y * 9_176
            ) % (2**63 - 1)
            generator = torch.Generator(device=patch.device)
            generator.manual_seed(noise_seed)
            noise = torch.randn(
                patch.shape,
                dtype=patch.dtype,
                device=patch.device,
                generator=generator,
            )
            lower, upper = (-1.0, 1.0) if self.extractor.normalize else (0.0, 1.0)
            return (patch + noise * value).clamp(lower, upper)

        raise ValueError(f"Unknown augmentation kind: {kind}")

    @property
    def num_images(self) -> int:
        return len(self.samples)

    @property
    def num_patches(self) -> int:
        return len(self._all_patches)

    @property
    def num_base_patches(self) -> int:
        return len(self._all_patches) // len(self._augmentation_variants)

    def class_distribution(self) -> Dict[int, int]:
        distribution: Dict[int, int] = {}
        for _, label in self.samples:
            distribution[label] = distribution.get(label, 0) + 1
        return distribution

    def subset(self, image_indices: Sequence[int]) -> TexturePatchDataset:
        return TexturePatchDataset(
            samples=[self.samples[i] for i in image_indices],
            patch_size=self.extractor.patch_size,
            overlap=self.extractor.overlap,
            grayscale=self.extractor.grayscale,
            normalize=self.extractor.normalize,
            transform=self.transform,
            epoch_fraction=self.epoch_fraction,
            balanced_per_image=self.balanced_per_image,
            with_replacement=self.with_replacement,
            seed=self.seed,
            cache_size=self.cache_size,
            augmentation=self.augmentation,
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"images={self.num_images}, "
            f"patches={self.num_patches}, "
            f"base_patches={self.num_base_patches}, "
            f"variants={len(self._augmentation_variants)}, "
            f"active={len(self)}, "
            f"patch_size={self.extractor.patch_size}, "
            f"overlap={self.extractor.overlap})"
        )
