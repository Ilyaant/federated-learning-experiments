from __future__ import annotations

from typing import Callable, List, Sequence

import torch
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

Transform = Callable[[torch.Tensor], torch.Tensor]


class RandomRotate90:
    """Rotate by a random multiple of 90 degrees (lossless for square patches)."""

    def __init__(self, p: float = 1.0):
        self.p = p

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() < self.p:
            k = int(torch.randint(1, 4, (1,)).item())
            return torch.rot90(tensor, k, dims=[-2, -1])
        return tensor


class RandomRotateAndCenterCrop:
    """Rotate an oversized patch by a random angle and crop the centre.

    The input must be at least ``output_size * sqrt(2)`` wide so the output
    never contains pixels outside the source image (no black corners).
    """

    def __init__(self, degrees: float, output_size: int, p: float = 1.0):
        self.degrees = float(degrees)
        self.output_size = int(output_size)
        self.p = p

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.degrees > 0 and torch.rand(1).item() < self.p:
            angle = (torch.rand(1).item() * 2.0 - 1.0) * self.degrees
            tensor = TF.rotate(
                tensor,
                angle,
                interpolation=InterpolationMode.BILINEAR,
                expand=False,
            )
        return TF.center_crop(tensor, [self.output_size, self.output_size])


class AddGaussianNoise:
    """Additive Gaussian noise in [0, 1] brightness units."""

    def __init__(self, std: float = 0.03, p: float = 0.3):
        self.std = float(std)
        self.p = p

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.std > 0 and torch.rand(1).item() < self.p:
            noise = torch.randn_like(tensor) * self.std
            return torch.clamp(tensor + noise, 0.0, 1.0)
        return tensor


def build_normalize(normalize_cfg: dict | None, channels: int) -> Transform:
    mean = list((normalize_cfg or {}).get("mean", [0.5] * channels))
    std = list((normalize_cfg or {}).get("std", [0.5] * channels))
    if len(mean) == 1 and channels > 1:
        mean = mean * channels
    if len(std) == 1 and channels > 1:
        std = std * channels
    return T.Normalize(mean=mean, std=std)


def needs_rotation_context(aug_cfg: dict | None) -> bool:
    """True when train patches must be cut with extra margin for free rotation."""
    if not aug_cfg or not aug_cfg.get("enabled", False):
        return False
    rotation = aug_cfg.get("rotation") or {}
    return float(rotation.get("degrees", 0.0)) > 0 and float(rotation.get("prob", 0.0)) > 0


def build_train_transform(
    aug_cfg: dict | None,
    normalize_cfg: dict | None,
    patch_size: int,
    channels: int,
) -> Transform:
    """Augmentations for training patches (input: float ``[C, S, S]`` in [0, 1]).

    Order: free-angle rotation + centre crop -> rot90 -> flips -> blur ->
    noise -> normalize. Photometric transforms run before normalization so
    their parameters are in brightness units.
    """
    steps: List[Transform] = []
    enabled = bool(aug_cfg and aug_cfg.get("enabled", False))

    if enabled:
        rotation = aug_cfg.get("rotation") or {}
        rot_prob = float(rotation.get("prob", 0.0))

        if needs_rotation_context(aug_cfg):
            steps.append(
                RandomRotateAndCenterCrop(
                    degrees=float(rotation["degrees"]),
                    output_size=patch_size,
                    p=rot_prob,
                )
            )
        if rotation.get("rot90", False) and rot_prob > 0:
            steps.append(RandomRotate90(p=rot_prob))

        if float(aug_cfg.get("hflip_prob", 0.0)) > 0:
            steps.append(T.RandomHorizontalFlip(p=float(aug_cfg["hflip_prob"])))
        if float(aug_cfg.get("vflip_prob", 0.0)) > 0:
            steps.append(T.RandomVerticalFlip(p=float(aug_cfg["vflip_prob"])))

        blur = aug_cfg.get("gaussian_blur") or {}
        if float(blur.get("prob", 0.0)) > 0:
            sigma: Sequence[float] = blur.get("sigma", (0.1, 2.0))
            steps.append(
                T.RandomApply(
                    [
                        T.GaussianBlur(
                            kernel_size=int(blur.get("kernel_size", 5)),
                            sigma=tuple(float(s) for s in sigma),
                        )
                    ],
                    p=float(blur["prob"]),
                )
            )

        noise = aug_cfg.get("gaussian_noise") or {}
        if float(noise.get("prob", 0.0)) > 0:
            steps.append(
                AddGaussianNoise(
                    std=float(noise.get("std", 0.03)),
                    p=float(noise["prob"]),
                )
            )

    steps.append(build_normalize(normalize_cfg, channels))
    return T.Compose(steps)


def build_eval_transform(normalize_cfg: dict | None, channels: int) -> Transform:
    return T.Compose([build_normalize(normalize_cfg, channels)])


def describe_transform(transform: Transform) -> str:
    if isinstance(transform, T.Compose):
        return " -> ".join(type(step).__name__ for step in transform.transforms)
    return type(transform).__name__
