from __future__ import annotations

import math
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from .patch_extractor import PatchExtractor


class TextureTrainDataset(Dataset):
    """
    Dataset used for local client training.

    Workflow
    --------
    * Images are stored only once.
    * Patch coordinates are computed once during initialization.
    * At the beginning of each epoch a subset of coordinates is sampled.
    * __getitem__ extracts only the requested patch.

    This avoids storing thousands of cropped patches in memory.
    """

    def __init__(
        self,
        samples: List[dict],
        extractor: PatchExtractor,
        transform: Optional[Callable] = None,
        epoch_fraction: float = 1.0,
        reshuffle_each_epoch: bool = True,
        balanced_per_image: bool = True,
        with_replacement: bool = False,
        seed: int = 42,
    ):

        self.samples = samples
        self.extractor = extractor
        self.transform = transform

        self.epoch_fraction = epoch_fraction
        self.reshuffle_each_epoch = reshuffle_each_epoch
        self.balanced_per_image = balanced_per_image
        self.with_replacement = with_replacement

        self.seed = seed

        self._coordinates: Dict[int, List[tuple]] = {}
        self._epoch_samples: List[tuple] = []

        self._prepare_coordinates()
        self.set_epoch(0)

    ####################################################################
    # Initialization
    ####################################################################

    def _prepare_coordinates(self):

        for image_idx, sample in enumerate(self.samples):

            image = Image.open(sample["path"]).convert("L")

            coords = self.extractor.get_patch_coordinates(image)

            self._coordinates[image_idx] = coords

    ####################################################################
    # Epoch sampling
    ####################################################################

    def set_epoch(self, epoch: int):

        rng = np.random.default_rng(self.seed + epoch)

        self._epoch_samples = []

        if self.balanced_per_image:

            for image_idx, coords in self._coordinates.items():

                n = len(coords)

                k = max(
                    1,
                    math.ceil(
                        n * self.epoch_fraction
                    ),
                )

                if self.with_replacement:

                    chosen = rng.choice(
                        n,
                        size=k,
                        replace=True,
                    )

                else:

                    chosen = rng.choice(
                        n,
                        size=min(k, n),
                        replace=False,
                    )

                for idx in chosen:

                    self._epoch_samples.append(
                        (
                            image_idx,
                            coords[int(idx)],
                        )
                    )

        else:

            pool = []

            for image_idx, coords in self._coordinates.items():

                for coord in coords:

                    pool.append(
                        (
                            image_idx,
                            coord,
                        )
                    )

            total = len(pool)

            k = max(
                1,
                math.ceil(
                    total * self.epoch_fraction
                ),
            )

            if self.with_replacement:

                chosen = rng.choice(
                    total,
                    size=k,
                    replace=True,
                )

            else:

                chosen = rng.choice(
                    total,
                    size=min(k, total),
                    replace=False,
                )

            self._epoch_samples = [
                pool[int(i)]
                for i in chosen
            ]

        if self.reshuffle_each_epoch:

            rng.shuffle(self._epoch_samples)

    ####################################################################
    # Dataset API
    ####################################################################

    def __len__(self):

        return len(self._epoch_samples)

    def __getitem__(self, index):

        image_idx, (x, y) = self._epoch_samples[index]

        sample = self.samples[image_idx]

        image = Image.open(sample["path"]).convert("L")

        patch = self.extractor.extract_patch(
            image=image,
            x=x,
            y=y,
        )

        if self.transform is not None:

            patch = self.transform(patch)

        return {
            "image": patch,
            "label": sample["label"],
            "image_id": image_idx,
            "coords": (x, y),
        }

    ####################################################################
    # Statistics
    ####################################################################

    @property
    def total_possible_patches(self):

        return sum(
            len(v)
            for v in self._coordinates.values()
        )

    @property
    def current_epoch_patches(self):

        return len(self._epoch_samples)

    def summary(self):

        print("=" * 60)
        print("TextureTrainDataset")
        print("=" * 60)

        print(f"Images              : {len(self.samples)}")
        print(f"All patches         : {self.total_possible_patches}")
        print(f"Epoch patches       : {self.current_epoch_patches}")
        print(f"Epoch fraction      : {self.epoch_fraction}")
        print(f"Balanced            : {self.balanced_per_image}")
        print(f"Replacement         : {self.with_replacement}")
