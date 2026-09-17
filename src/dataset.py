from __future__ import annotations

from collections import Counter, OrderedDict
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from .patches import (
    PatchGrid,
    context_size_for_rotation,
    extract_patch,
    load_image_uint8,
    read_image_size,
)

Sample = Tuple[Path, int]
PatchEntry = Tuple[int, int, int]  # image index, x, y


class TexturePatchDataset(Dataset):
    """Grid patches of whole texture images.

    Every item is one ``patch_size`` square taken from the grid of one image.
    Images are decoded lazily and kept in a per-worker LRU cache as ``uint8``
    so that cutting many patches from the same image is cheap.
    """

    def __init__(
        self,
        samples: Sequence[Tuple[str | Path, int]],
        patch_size: int = 224,
        overlap: float = 0.0,
        include_edges: bool = True,
        grayscale: bool = True,
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        rotation_context: bool = False,
        pad_mode: str = "reflect",
        cache_size: int = 512,
    ):
        self.samples: List[Sample] = [(Path(p), int(label)) for p, label in samples]
        self.grid = PatchGrid(patch_size, overlap, include_edges)
        self.grayscale = grayscale
        self.transform = transform
        self.pad_mode = pad_mode
        self.cache_size = int(cache_size)
        self.context_size = (
            context_size_for_rotation(patch_size) if rotation_context else None
        )

        self.image_sizes: List[Tuple[int, int]] = [
            read_image_size(path) for path, _ in self.samples
        ]
        self.patches: List[PatchEntry] = []
        self.patches_by_image: List[List[int]] = []
        for image_idx, (width, height) in enumerate(self.image_sizes):
            indices = []
            for x, y in self.grid.coordinates(width, height):
                indices.append(len(self.patches))
                self.patches.append((image_idx, x, y))
            self.patches_by_image.append(indices)

        self._cache: "OrderedDict[int, torch.Tensor]" = OrderedDict()

    # ----------------------------------------------------------------- info
    @property
    def patch_size(self) -> int:
        return self.grid.patch_size

    @property
    def num_images(self) -> int:
        return len(self.samples)

    @property
    def num_patches(self) -> int:
        return len(self.patches)

    @property
    def labels(self) -> List[int]:
        return [label for _, label in self.samples]

    def class_distribution(self) -> Dict[int, int]:
        return dict(sorted(Counter(self.labels).items()))

    def patch_class_distribution(self) -> Dict[int, int]:
        counts: Counter = Counter()
        for image_idx, _, _ in self.patches:
            counts[self.samples[image_idx][1]] += 1
        return dict(sorted(counts.items()))

    # ------------------------------------------------------------- loading
    def _load_image(self, image_idx: int) -> torch.Tensor:
        if self.cache_size > 0 and image_idx in self._cache:
            self._cache.move_to_end(image_idx)
            return self._cache[image_idx]

        tensor = load_image_uint8(self.samples[image_idx][0], self.grayscale)
        if self.cache_size > 0:
            self._cache[image_idx] = tensor
            while len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)
        return tensor

    def __len__(self) -> int:
        return len(self.patches)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        image_idx, x, y = self.patches[idx]
        _, label = self.samples[image_idx]

        patch = extract_patch(
            self._load_image(image_idx),
            x,
            y,
            self.patch_size,
            context_size=self.context_size,
            pad_mode=self.pad_mode,
        )
        if self.transform is not None:
            patch = self.transform(patch)

        return {
            "image": patch,
            "label": label,
            "image_id": image_idx,
            "x": x,
            "y": y,
        }

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(images={self.num_images}, "
            f"patches={self.num_patches}, patch_size={self.patch_size}, "
            f"overlap={self.grid.overlap}, context={self.context_size})"
        )


class PerImagePatchSampler(Sampler[int]):
    """Shuffled sampler that draws up to ``patches_per_image`` grid patches per image.

    Living in the main process, it is re-evaluated every epoch (unlike the
    dataset copy inside persistent DataLoader workers), so ``set_epoch`` gives
    a fresh, reproducible subset each time.
    """

    def __init__(
        self,
        dataset: TexturePatchDataset,
        patches_per_image: Optional[int] = None,
        seed: int = 42,
        shuffle: bool = True,
    ):
        self.dataset = dataset
        self.patches_per_image = patches_per_image
        self.seed = int(seed)
        self.shuffle = shuffle
        self.epoch = 0
        self._indices = self._draw()

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
        self._indices = self._draw()

    def _draw(self) -> List[int]:
        rng = np.random.default_rng(self.seed + self.epoch)
        chosen: List[int] = []
        for indices in self.dataset.patches_by_image:
            if self.patches_per_image is None or self.patches_per_image >= len(indices):
                chosen.extend(indices)
            else:
                picked = rng.choice(len(indices), size=self.patches_per_image, replace=False)
                chosen.extend(indices[int(i)] for i in picked)
        if self.shuffle:
            rng.shuffle(chosen)
        return chosen

    def __iter__(self) -> Iterator[int]:
        return iter(self._indices)

    def __len__(self) -> int:
        return len(self._indices)


def limit_images_per_class(
    samples: Sequence[Tuple[Path, int]],
    max_per_class: Optional[int],
) -> List[Tuple[Path, int]]:
    if not max_per_class:
        return list(samples)
    taken: Counter = Counter()
    limited: List[Tuple[Path, int]] = []
    for path, label in samples:
        if taken[label] < max_per_class:
            limited.append((path, label))
            taken[label] += 1
    return limited
