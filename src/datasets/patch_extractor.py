from __future__ import annotations

from pathlib import Path
from typing import List, Tuple, Union

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
        self.stride = max(1, int(patch_size * (1 - overlap)))

    def load_image(
        self,
        image: Union[str, Path, Image.Image],
    ) -> torch.Tensor:
        if isinstance(image, (str, Path)):
            image = Image.open(image)

        image = image.convert("L" if self.grayscale else "RGB")
        tensor = TF.to_tensor(image)

        if self.normalize:
            channels = 1 if self.grayscale else 3
            mean = [0.5] * channels
            std = [0.5] * channels
            tensor = TF.normalize(tensor, mean=mean, std=std)

        return tensor

    def _positions(self, size: int) -> List[int]:
        if size <= self.patch_size:
            return [0]

        positions = list(
            range(0, size - self.patch_size + 1, self.stride)
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
            _, h, w = self.load_image(image).shape

        xs = self._positions(w)
        ys = self._positions(h)
        return [(x, y) for y in ys for x in xs]

    def extract_patch(
        self,
        image: Union[str, Path, Image.Image, torch.Tensor],
        x: int,
        y: int,
    ) -> torch.Tensor:
        tensor = image if torch.is_tensor(image) else self.load_image(image)
        _, h, w = tensor.shape

        pad_h = max(0, self.patch_size - h)
        pad_w = max(0, self.patch_size - w)

        if pad_h or pad_w:
            tensor = F.pad(
                tensor,
                (0, pad_w, 0, pad_h),
                mode=self.pad_mode,
            )

        return tensor[
            :,
            y : y + self.patch_size,
            x : x + self.patch_size,
        ]

    def extract(
        self,
        image: Union[str, Path, Image.Image, torch.Tensor],
    ) -> List[torch.Tensor]:
        tensor = image if torch.is_tensor(image) else self.load_image(image)
        return [
            self.extract_patch(tensor, x, y)
            for x, y in self.coordinates(tensor)
        ]
