from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import InterpolationMode
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
        slice_patches: bool = True,
        resize_to: Optional[int] = None,
    ):
        if not 0 <= overlap < 1:
            raise ValueError("overlap must satisfy 0 <= overlap < 1")
        if downscale < 1.0:
            raise ValueError("downscale must be >= 1 (1 keeps native scale)")
        if resize_to is not None and int(resize_to) <= 0:
            raise ValueError("resize_to must be positive")

        self.patch_size = int(patch_size)
        self.overlap = float(overlap)
        self.grayscale = bool(grayscale)
        self.normalize = bool(normalize)
        self.pad_mode = pad_mode
        # The whole image is shrunk by ``downscale`` before patching, so a
        # single ``patch_size`` patch covers ``downscale`` times more of the
        # original field of view (more texture context per patch).
        self.downscale = float(downscale)
        self.slice_patches = bool(slice_patches)
        self.resize_to = int(resize_to) if resize_to is not None else None
        self.stride = max(1, int(self.patch_size * (1 - self.overlap)))

    def scaled_size(self, width: int, height: int) -> Tuple[int, int]:
        """Image size after resizing or downscaling (never smaller than 1x1)."""
        if self.resize_to is not None:
            return (self.resize_to, self.resize_to)
        if self.downscale == 1.0:
            return width, height
        return (
            max(1, round(width / self.downscale)),
            max(1, round(height / self.downscale)),
        )

    def load_image(
        self,
        image: Union[str, Path, Image.Image, np.ndarray, torch.Tensor],
    ) -> torch.Tensor:
        if torch.is_tensor(image):
            tensor = (
                image.float().div(255.0)
                if image.dtype == torch.uint8
                else image.float()
            )
            if tensor.ndim == 2:
                tensor = tensor.unsqueeze(0)
            if self.grayscale and tensor.shape[0] == 3:
                tensor = TF.rgb_to_grayscale(tensor)
            elif not self.grayscale and tensor.shape[0] == 1:
                tensor = tensor.expand(3, -1, -1).contiguous()
            h, w = tensor.shape[-2:]
            scaled_w, scaled_h = self.scaled_size(w, h)
            if (scaled_w, scaled_h) != (w, h):
                tensor = TF.resize(
                    tensor,
                    [scaled_h, scaled_w],
                    interpolation=InterpolationMode.BICUBIC,
                    antialias=True,
                ).clamp(0.0, 1.0)
        else:
            if isinstance(image, (str, Path)):
                image = Image.open(image)
            elif isinstance(image, np.ndarray):
                image = Image.fromarray(image)

            mode = "L" if self.grayscale else "RGB"
            orig_w, orig_h = image.size
            scaled = self.scaled_size(orig_w, orig_h)

            if (
                scaled != image.size
                and scaled[0] < orig_w
                and scaled[1] < orig_h
                and hasattr(image, "draft")
            ):
                # JPEG-only fast path: ask the decoder for the smallest DCT
                # scale (1/2, 1/4, 1/8) that is still >= the target size, so
                # most of the shrinking happens during decoding. The final
                # LANCZOS resize below brings it to the exact size.
                image.draft(mode, scaled)

            image = image.convert(mode)

            if scaled != image.size:
                # Use BICUBIC when resizing to a fixed resolution (e.g. CIFAR-100 -> 224),
                # and LANCZOS when downscaling large texture images.
                resample = (
                    Image.BICUBIC
                    if self.resize_to is not None
                    else Image.LANCZOS
                )
                image = image.resize(scaled, resample)

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
        image: Union[str, Path, Image.Image, np.ndarray, torch.Tensor],
    ) -> List[Tuple[int, int]]:
        if not self.slice_patches:
            return [(0, 0)]

        if torch.is_tensor(image):
            # Tensors come out of load_image() and are already downscaled.
            _, h, w = image.shape
            return self.coordinates_for_size(w, h)

        if isinstance(image, np.ndarray):
            h, w = image.shape[:2]
        elif isinstance(image, Image.Image):
            w, h = image.size
        else:
            # Reads only the image header, without decoding pixels.
            with Image.open(image) as opened:
                w, h = opened.size

        return self.coordinates_for_size(*self.scaled_size(w, h))

    def extract_patch(
        self,
        image: Union[str, Path, Image.Image, np.ndarray, torch.Tensor],
        x: int,
        y: int,
    ) -> torch.Tensor:
        tensor = image if torch.is_tensor(image) else self.load_image(image)
        if not self.slice_patches:
            return tensor
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
