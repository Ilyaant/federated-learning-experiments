from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Union, Optional

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as F


@dataclass
class PatchSample:
    """
    One extracted patch.
    """

    image: torch.Tensor
    x: int
    y: int
    width: int
    height: int


class PatchExtractor:
    """
    Extract overlapping patches from texture images.

    Parameters
    ----------
    patch_size : int
        Patch size.

    overlap : float
        Overlap between neighbouring patches.
        0.5 -> stride = patch_size // 2

    grayscale : bool
        Convert image to grayscale.

    normalize : bool
        Convert to float tensor in [0,1].

    return_coords : bool
        Return PatchSample or only tensors.
    """

    def __init__(
        self,
        patch_size: int = 224,
        overlap: float = 0.5,
        grayscale: bool = True,
        normalize: bool = True,
        return_coords: bool = True,
    ) -> None:

        self.patch_size = patch_size
        self.overlap = overlap
        self.stride = int(patch_size * (1.0 - overlap))

        if self.stride <= 0:
            raise ValueError("Invalid overlap.")

        self.grayscale = grayscale
        self.normalize = normalize
        self.return_coords = return_coords

    def __call__(
        self,
        image: Union[str, Path, Image.Image],
    ):
        image = self._load(image)

        patches = []

        for x, y in self._coordinates(image.size):

            patch = image.crop(
                (
                    x,
                    y,
                    x + self.patch_size,
                    y + self.patch_size,
                )
            )

            tensor = F.pil_to_tensor(patch)

            if self.normalize:
                tensor = tensor.float() / 255.0

            if self.return_coords:
                patches.append(
                    PatchSample(
                        image=tensor,
                        x=x,
                        y=y,
                        width=self.patch_size,
                        height=self.patch_size,
                    )
                )
            else:
                patches.append(tensor)

        return patches

    def _load(
        self,
        image: Union[str, Path, Image.Image],
    ) -> Image.Image:

        if isinstance(image, (str, Path)):
            image = Image.open(image)

        if self.grayscale:
            image = image.convert("L")

        return image

    def _coordinates(
        self,
        size: Tuple[int, int],
    ) -> List[Tuple[int, int]]:

        width, height = size

        xs = self._axis_positions(width)
        ys = self._axis_positions(height)

        coords = []

        for y in ys:
            for x in xs:
                coords.append((x, y))

        return coords

    def _axis_positions(
        self,
        length: int,
    ) -> List[int]:

        if length <= self.patch_size:
            return [0]

        positions = list(
            range(
                0,
                length - self.patch_size + 1,
                self.stride,
            )
        )

        last = length - self.patch_size

        if positions[-1] != last:
            positions.append(last)

        return positions

    @property
    def config(self):

        return {
            "patch_size": self.patch_size,
            "stride": self.stride,
            "overlap": self.overlap,
            "grayscale": self.grayscale,
            "normalize": self.normalize,
        }


if __name__ == "__main__":

    extractor = PatchExtractor(
        patch_size=224,
        overlap=0.5,
    )

    patches = extractor("example.png")

    print(f"Extracted {len(patches)} patches")

    print(patches[0].image.shape)
    print(patches[0].x, patches[0].y)