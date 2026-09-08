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
        downscale: float = 1.0,
    ):
        if not 0 <= overlap < 1:
            raise ValueError("overlap must satisfy 0 <= overlap < 1")
        if downscale < 1.0:
            raise ValueError("downscale must be >= 1 (1 keeps native scale)")

        self.patch_size = patch_size
        self.overlap = overlap
        self.grayscale = grayscale
        self.normalize = normalize
        self.pad_mode = pad_mode
        # The whole image is shrunk by ``downscale`` before patching, so a
        # single ``patch_size`` patch covers ``downscale`` times more of the
        # original field of view (more texture context per patch).
        self.downscale = float(downscale)
        self.stride = max(1, int(patch_size * (1 - overlap)))

    def scaled_size(self, width: int, height: int) -> Tuple[int, int]:
        """Image size after downscaling (never smaller than 1x1)."""
        if self.downscale == 1.0:
            return width, height
        return (
            max(1, round(width / self.downscale)),
            max(1, round(height / self.downscale)),
        )

    def load_image(
        self,
        image: Union[str, Path, Image.Image],
    ) -> torch.Tensor:
        if isinstance(image, (str, Path)):
            image = Image.open(image)

        mode = "L" if self.grayscale else "RGB"
        scaled = self.scaled_size(*image.size)

        if scaled != image.size and hasattr(image, "draft"):
            # JPEG-only fast path: ask the decoder for the smallest DCT
            # scale (1/2, 1/4, 1/8) that is still >= the target size, so
            # most of the shrinking happens during decoding. The final
            # LANCZOS resize below brings it to the exact size.
            image.draft(mode, scaled)

        image = image.convert(mode)

        if scaled != image.size:
            # LANCZOS anti-aliases when shrinking; nearest/bilinear would
            # alias the fine fibrous texture we want to classify.
            image = image.resize(scaled, Image.LANCZOS)

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

    def coordinates_for_size(
        self,
        width: int,
        height: int,
    ) -> List[Tuple[int, int]]:
        xs = self._positions(width)
        ys = self._positions(height)
        return [(x, y) for y in ys for x in xs]

    def coordinates(
        self,
        image: Union[str, Path, Image.Image, torch.Tensor],
    ) -> List[Tuple[int, int]]:
        if torch.is_tensor(image):
            # Tensors come out of load_image() and are already downscaled.
            _, h, w = image.shape
            return self.coordinates_for_size(w, h)

        if isinstance(image, Image.Image):
            w, h = image.size
        else:
            # Reads only the image header, without decoding pixels.
            with Image.open(image) as opened:
                w, h = opened.size

        return self.coordinates_for_size(*self.scaled_size(w, h))

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
