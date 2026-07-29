from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, List, Optional

from PIL import Image
from torch.utils.data import Dataset

from .patch_extractor import PatchExtractor, PatchCoordinate


class TextureInferenceDataset(Dataset):
    """
    Dataset used for validation and testing.

    Unlike TextureTrainDataset:
        * uses ALL patches;
        * coordinates are computed once;
        * patches are extracted lazily in __getitem__.
    """

    def __init__(
        self,
        samples: List[dict],
        extractor: PatchExtractor,
        transform: Optional[Callable] = None,
        cache_images=True,
        cache_size=64,
    ):

        self.samples = samples
        self.extractor = extractor
        self.transform = transform
        self.cache_images = cache_images
        self.cache_size = cache_size
        self.image_cache = OrderedDict()

        self.index = []

        self._build_index()

    ####################################################################
    # Build patch index
    ####################################################################

    def _build_index(self):

        for image_id, sample in enumerate(self.samples):

            coords = self.extractor.get_patch_coordinates(
                sample["path"]
            )

            for patch_id, coord in enumerate(coords):

                self.index.append(
                    {
                        "image_id": image_id,
                        "patch_id": patch_id,
                        "coord": coord,
                    }
                )

    ####################################################################
    # Dataset API
    ####################################################################

    def __len__(self):

        return len(self.index)

    
    def _get_image(self, image_path):

        image_path = str(image_path)

        if not self.cache_images:
            return Image.open(image_path).convert("L")

        if image_path in self.image_cache:

            image = self.image_cache.pop(image_path)
            self.image_cache[image_path] = image

            return image

        image = Image.open(image_path).convert("L")

        self.image_cache[image_path] = image

        if len(self.image_cache) > self.cache_size:
            self.image_cache.popitem(last=False)

        return image


    def __getitem__(self, idx):

        info = self.index[idx]

        sample = self.samples[info["image_id"]]

        image = self._get_image(sample["path"])

        patch = self.extractor.extract_patch(
            image=image,
            x=info["coord"].x,
            y=info["coord"].y,
        )

        if self.transform is not None:

            patch = self.transform(patch)

        return {
            "image": patch,
            "label": sample["label"],
            "image_id": info["image_id"],
            "patch_id": info["patch_id"],
            "coords": (
                info["coord"].x,
                info["coord"].y,
            ),
        }

    ####################################################################
    # Statistics
    ####################################################################

    @property
    def num_images(self):

        return len(self.samples)

    @property
    def num_patches(self):

        return len(self.index)

    def summary(self):

        print("=" * 60)
        print("TextureInferenceDataset")
        print("=" * 60)

        print(f"Images : {self.num_images}")
        print(f"Patches: {self.num_patches}")
