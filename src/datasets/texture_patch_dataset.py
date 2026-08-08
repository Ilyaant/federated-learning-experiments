from __future__ import annotations

import math
from collections import OrderedDict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from .patch_extractor import PatchExtractor

Sample = Tuple[Path, int]
PatchEntry = Tuple[int, int, int]


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

        self.extractor = PatchExtractor(
            patch_size=patch_size,
            overlap=overlap,
            grayscale=grayscale,
            normalize=normalize,
        )

        self._all_patches: List[PatchEntry] = []
        self._active_patches: List[PatchEntry] = []
        self._image_cache: OrderedDict[str, Image.Image] = OrderedDict()

        self._build_patch_index()
        self.set_epoch(0)

    def _build_patch_index(self) -> None:
        self._all_patches.clear()

        for image_idx, (path, _) in enumerate(self.samples):
            for x, y in self.extractor.coordinates(path):
                self._all_patches.append((image_idx, x, y))

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

    def _load_image(self, path: Path) -> Image.Image:
        key = str(path)

        if key in self._image_cache:
            self._image_cache.move_to_end(key)
            return self._image_cache[key]

        mode = "L" if self.extractor.grayscale else "RGB"
        image = Image.open(path).convert(mode)
        self._image_cache[key] = image

        if len(self._image_cache) > self.cache_size:
            self._image_cache.popitem(last=False)

        return image

    def __len__(self) -> int:
        return len(self._active_patches)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        image_idx, x, y = self._active_patches[idx]
        path, label = self.samples[image_idx]

        patch = self.extractor.extract_patch(
            self._load_image(path),
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

    @property
    def num_images(self) -> int:
        return len(self.samples)

    @property
    def num_patches(self) -> int:
        return len(self._all_patches)

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
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"images={self.num_images}, "
            f"patches={self.num_patches}, "
            f"active={len(self)}, "
            f"patch_size={self.extractor.patch_size}, "
            f"overlap={self.extractor.overlap})"
        )
