from __future__ import annotations

from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import Dataset

from .patch_extractor import PatchExtractor


class TexturePatchDataset(Dataset):
    """
    Dataset of image patches.

    Unlike the previous implementation, this dataset does NOT store every patch
    tensor in memory. It stores only the coordinates of every patch and loads
    the corresponding image lazily. This makes it suitable for federated
    learning where every client owns only a subset of images.
    """

    def __init__(
        self,
        samples: Sequence[Tuple[str | Path, int]],
        patch_size: int = 224,
        overlap: float = 0.5,
        grayscale: bool = True,
        normalize: bool = True,
        transform: Optional[Callable] = None,
    ):
        self.samples = [
            (Path(path), int(label))
            for path, label in samples
        ]

        self.transform = transform

        self.extractor = PatchExtractor(
            patch_size=patch_size,
            overlap=overlap,
            grayscale=grayscale,
            normalize=normalize,
        )

        self.patch_index: List[Tuple[int, int, int]] = []

        self._build_patch_index()

    def _build_patch_index(self):
        self.patch_index.clear()

        for image_idx, (path, _) in enumerate(self.samples):
            coords = self.extractor.get_patch_coordinates(path)

            for x, y in coords:
                self.patch_index.append(
                    (
                        image_idx,
                        x,
                        y,
                    )
                )

    def __len__(self):
        return len(self.patch_index)

    def __getitem__(self, idx):
        image_idx, x, y = self.patch_index[idx]

        image_path, label = self.samples[image_idx]

        patch = self.extractor.extract_patch(
            image_path,
            x,
            y,
        )

        if self.transform is not None:
            patch = self.transform(patch)

        return patch, label

    @property
    def targets(self):
        return [
            self.samples[i][1]
            for i, _, _ in self.patch_index
        ]

    @property
    def image_paths(self):
        return [
            self.samples[i][0]
            for i, _, _ in self.patch_index
        ]

    @property
    def num_images(self):
        return len(self.samples)

    @property
    def num_patches(self):
        return len(self.patch_index)

    def patches_for_image(
        self,
        image_index: int,
    ):
        return [
            idx
            for idx, (img_idx, _, _) in enumerate(self.patch_index)
            if img_idx == image_index
        ]

    def get_image(
        self,
        image_index: int,
    ):
        return self.samples[image_index]

    def subset(
        self,
        image_indices: Sequence[int],
    ):
        subset_samples = [
            self.samples[idx]
            for idx in image_indices
        ]

        return TexturePatchDataset(
            samples=subset_samples,
            patch_size=self.extractor.patch_size,
            overlap=self.extractor.overlap,
            grayscale=self.extractor.grayscale,
            normalize=self.extractor.normalize,
            transform=self.transform,
        )

    def class_distribution(self):
        distribution = {}

        for _, label in self.samples:
            distribution[label] = distribution.get(label, 0) + 1

        return distribution

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"images={self.num_images}, "
            f"patches={self.num_patches}, "
            f"patch_size={self.extractor.patch_size}, "
            f"overlap={self.extractor.overlap})"
        )
