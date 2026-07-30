from __future__ import annotations

from pathlib import Path
from typing import List, Sequence, Tuple, Union

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import functional as TF


class PatchExtractor:
    def __init__(
        self,
        patch_size: int = 224,
        overlap: float = 0.5,
        grayscale: bool = True,
        normalize: bool = True,
        pad_mode: str = "replicate",
    ):
        if not 0 <= overlap < 1:
            raise ValueError("overlap must satisfy 0 <= overlap < 1")

        self.patch_size = patch_size
        self.overlap = overlap
        self.grayscale = grayscale
        self.normalize = normalize
        self.pad_mode = pad_mode

        self.stride = int(patch_size * (1 - overlap))
        self.stride = max(1, self.stride)

    def load_image(
        self,
        image: Union[str, Path, Image.Image],
    ) -> torch.Tensor:
        if isinstance(image, (str, Path)):
            image = Image.open(image)

        if self.grayscale:
            image = image.convert("L")
        else:
            image = image.convert("RGB")

        tensor = TF.to_tensor(image)

        if not self.normalize:
            tensor = tensor * 255.0

        return tensor

    def _positions(
        self,
        size: int,
    ) -> List[int]:
        if size <= self.patch_size:
            return [0]

        positions = list(
            range(
                0,
                size - self.patch_size + 1,
                self.stride,
            )
        )

        if positions[-1] != size - self.patch_size:
            positions.append(size - self.patch_size)

        return positions

    def coordinates(
        self,
        image: Union[str, Path, Image.Image, torch.Tensor],
    ) -> List[Tuple[int, int]]:
        if torch.is_tensor(image):
            _, h, w = image.shape
        else:
            tensor = self.load_image(image)
            _, h, w = tensor.shape

        xs = self._positions(w)
        ys = self._positions(h)

        return [(x, y) for y in ys for x in xs]

    def extract_patch(
        self,
        image: Union[str, Path, Image.Image, torch.Tensor],
        x: int,
        y: int,
    ) -> torch.Tensor:
        if torch.is_tensor(image):
            tensor = image
        else:
            tensor = self.load_image(image)

        _, h, w = tensor.shape

        pad_h = max(0, self.patch_size - h)
        pad_w = max(0, self.patch_size - w)

        if pad_h or pad_w:
            tensor = F.pad(
                tensor,
                (0, pad_w, 0, pad_h),
                mode=self.pad_mode,
            )

        patch = tensor[
            :,
            y : y + self.patch_size,
            x : x + self.patch_size,
        ]

        if (
            patch.shape[1] != self.patch_size
            or patch.shape[2] != self.patch_size
        ):
            patch = F.pad(
                patch,
                (
                    0,
                    self.patch_size - patch.shape[2],
                    0,
                    self.patch_size - patch.shape[1],
                ),
                mode=self.pad_mode,
            )

        return patch

    def extract(
        self,
        image: Union[str, Path, Image.Image, torch.Tensor],
    ) -> List[torch.Tensor]:
        if torch.is_tensor(image):
            tensor = image
        else:
            tensor = self.load_image(image)

        patches = []

        for x, y in self.coordinates(tensor):
            patches.append(
                self.extract_patch(
                    tensor,
                    x,
                    y,
                )
            )

        return patches

    def extract_with_coordinates(
        self,
        image: Union[str, Path, Image.Image, torch.Tensor],
    ) -> List[Tuple[torch.Tensor, Tuple[int, int]]]:
        if torch.is_tensor(image):
            tensor = image
        else:
            tensor = self.load_image(image)

        output = []

        for x, y in self.coordinates(tensor):
            output.append(
                (
                    self.extract_patch(
                        tensor,
                        x,
                        y,
                    ),
                    (x, y),
                )
            )

        return output

    def num_patches(
        self,
        image: Union[str, Path, Image.Image, torch.Tensor],
    ) -> int:
        return len(self.coordinates(image))
