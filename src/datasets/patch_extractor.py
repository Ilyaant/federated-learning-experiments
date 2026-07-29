from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import List, Tuple, Union

import torch
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor


class PatchExtractor:

    def __init__(
        self,
        patch_size: int = 224,
        overlap: float = 0.5,
        grayscale: bool = True,
        normalize: bool = True,
        cache_size: int = 64,
    ):

        self.patch_size = patch_size
        self.overlap = overlap
        self.grayscale = grayscale
        self.normalize = normalize

        self.stride = int(
            patch_size * (1.0 - overlap)
        )

        if self.stride <= 0:
            raise ValueError("Invalid overlap.")

        self.cache_size = cache_size
        self.tensor_cache = OrderedDict()

    ####################################################################
    # Tensor cache
    ####################################################################

    def load_tensor(
        self,
        image: Union[str, Path, Image.Image],
    ) -> torch.Tensor:

        if isinstance(image, Image.Image):
            return self._pil_to_tensor(image)

        key = str(image)

        if key in self.tensor_cache:

            tensor = self.tensor_cache.pop(key)
            self.tensor_cache[key] = tensor

            return tensor

        img = Image.open(key)

        if self.grayscale:
            img = img.convert("L")

        tensor = self._pil_to_tensor(img)

        self.tensor_cache[key] = tensor

        if len(self.tensor_cache) > self.cache_size:
            self.tensor_cache.popitem(last=False)

        return tensor

    ####################################################################
    # Internal
    ####################################################################

    def _pil_to_tensor(
        self,
        image: Image.Image,
    ) -> torch.Tensor:

        tensor = pil_to_tensor(image)

        if self.normalize:
            tensor = tensor.float() / 255.0

        return tensor

    ####################################################################
    # Coordinates
    ####################################################################

    def get_patch_coordinates(
        self,
        image: Union[str, Path, Image.Image],
    ) -> List[Tuple[int, int]]:

        if isinstance(image, Image.Image):
            width, height = image.size
        else:
            tensor = self.load_tensor(image)
            _, height, width = tensor.shape

        xs = self._positions(width)
        ys = self._positions(height)

        coords = []

        for y in ys:
            for x in xs:
                coords.append((x, y))

        return coords

    ####################################################################
    # Extract one patch
    ####################################################################

    def extract_patch(
        self,
        image: Union[str, Path, Image.Image, torch.Tensor],
        x: int,
        y: int,
    ) -> torch.Tensor:

        if torch.is_tensor(image):
            tensor = image
        else:
            tensor = self.load_tensor(image)

        return tensor[
            :,
            y:y + self.patch_size,
            x:x + self.patch_size,
        ]

    ####################################################################
    # Extract all patches
    ####################################################################

    def extract_all(
        self,
        image,
    ):

        tensor = self.load_tensor(image)

        coords = self.get_patch_coordinates(image)

        return [
            (
                self.extract_patch(
                    tensor,
                    x,
                    y,
                ),
                (x, y),
            )
            for x, y in coords
        ]

    ####################################################################
    # Helpers
    ####################################################################

    def _positions(
        self,
        length: int,
    ):

        if length <= self.patch_size:
            return [0]

        pos = list(
            range(
                0,
                length - self.patch_size + 1,
                self.stride,
            )
        )

        last = length - self.patch_size

        if pos[-1] != last:
            pos.append(last)

        return pos

    def clear_cache(self):

        self.tensor_cache.clear()