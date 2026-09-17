from __future__ import annotations

import math
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import functional as TF

Coordinate = Tuple[int, int]


class PatchGrid:
    """Regular grid of ``patch_size`` squares over an image.

    ``overlap`` is the fraction of ``patch_size`` shared by neighbours; with
    ``include_edges`` the last row/column is shifted to touch the border so the
    whole image is covered even if its size is not a multiple of the stride.
    """

    def __init__(
        self,
        patch_size: int = 224,
        overlap: float = 0.0,
        include_edges: bool = True,
    ):
        if not 0 <= overlap < 1:
            raise ValueError("overlap must satisfy 0 <= overlap < 1")
        if patch_size <= 0:
            raise ValueError("patch_size must be positive")

        self.patch_size = int(patch_size)
        self.overlap = float(overlap)
        self.include_edges = include_edges
        self.stride = max(1, int(round(self.patch_size * (1.0 - self.overlap))))

    def positions(self, length: int) -> List[int]:
        if length <= self.patch_size:
            return [0]

        positions = list(range(0, length - self.patch_size + 1, self.stride))
        last = length - self.patch_size
        if self.include_edges and positions[-1] != last:
            positions.append(last)
        return positions

    def coordinates(self, width: int, height: int) -> List[Coordinate]:
        xs = self.positions(width)
        ys = self.positions(height)
        return [(x, y) for y in ys for x in xs]

    def count(self, width: int, height: int) -> int:
        return len(self.positions(width)) * len(self.positions(height))


def read_image_size(path: str | Path) -> Tuple[int, int]:
    """(width, height) from the file header, without decoding pixels."""
    with Image.open(path) as image:
        return image.size


def load_image_uint8(path: str | Path, grayscale: bool = True) -> torch.Tensor:
    """Decode an image into a ``uint8`` tensor ``[C, H, W]``."""
    mode = "L" if grayscale else "RGB"
    with Image.open(path) as image:
        image = image.convert(mode)
        return TF.pil_to_tensor(image)


def crop_with_padding(
    image: torch.Tensor,
    x0: int,
    y0: int,
    width: int,
    height: int,
    pad_mode: str = "reflect",
) -> torch.Tensor:
    """Crop ``[C, height, width]`` at (x0, y0), padding outside the image.

    Works for ``uint8`` or float input and returns a float tensor in [0, 1].
    Coordinates may be negative or exceed the image; missing pixels are filled
    with ``reflect``/``replicate`` padding so texture statistics stay natural.
    """
    _, img_h, img_w = image.shape
    x1, y1 = x0 + width, y0 + height

    cx0, cy0 = max(x0, 0), max(y0, 0)
    cx1, cy1 = min(x1, img_w), min(y1, img_h)
    if cx0 >= cx1 or cy0 >= cy1:
        raise ValueError(
            f"Crop ({x0},{y0},{width},{height}) does not intersect image {img_w}x{img_h}"
        )

    crop = image[:, cy0:cy1, cx0:cx1]
    crop = crop.float().div_(255.0) if crop.dtype == torch.uint8 else crop.float()

    pads = (cx0 - x0, x1 - cx1, cy0 - y0, y1 - cy1)  # left, right, top, bottom
    if any(pads):
        _, crop_h, crop_w = crop.shape
        mode = pad_mode
        # reflect requires pad < dim; fall back for tiny crops.
        if mode == "reflect" and (
            max(pads[0], pads[1]) >= crop_w or max(pads[2], pads[3]) >= crop_h
        ):
            mode = "replicate"
        crop = F.pad(crop.unsqueeze(0), pads, mode=mode).squeeze(0)

    return crop


def context_size_for_rotation(patch_size: int) -> int:
    """Side of the square that still contains a ``patch_size`` square after any rotation."""
    return int(math.ceil(patch_size * math.sqrt(2.0)))


def extract_patch(
    image: torch.Tensor,
    x: int,
    y: int,
    patch_size: int,
    context_size: int | None = None,
    pad_mode: str = "reflect",
) -> torch.Tensor:
    """Float patch ``[C, S, S]`` at grid position (x, y).

    With ``context_size`` the crop is enlarged around the patch centre so it
    can be rotated by an arbitrary angle and centre-cropped back to
    ``patch_size`` without black corners.
    """
    size = context_size or patch_size
    if size == patch_size:
        return crop_with_padding(image, x, y, patch_size, patch_size, pad_mode)

    offset = (size - patch_size) // 2
    return crop_with_padding(image, x - offset, y - offset, size, size, pad_mode)
