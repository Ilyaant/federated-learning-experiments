from __future__ import annotations

from typing import Callable, List, Sequence

import torch
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

Transform = Callable[[torch.Tensor], torch.Tensor]


class RandomRotate90:
    """Rotate by a random multiple of 90 degrees (0, 90, 180, 270; lossless for square patches)."""

    def __init__(self, p: float = 1.0):
        self.p = p

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() < self.p:
            k = int(torch.randint(0, 4, (1,)).item())
            if k > 0:
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


class RandomContrastScale:
    """Scale patch contrast around its spatial mean to simulate session contrast variation."""

    def __init__(self, scale_range: Sequence[float] = (0.8, 1.25), p: float = 0.5):
        if len(scale_range) != 2 or float(scale_range[0]) <= 0 or float(scale_range[0]) > float(scale_range[1]):
            raise ValueError(f"Invalid contrast scale_range: {scale_range!r}")
        self.min_scale = float(scale_range[0])
        self.max_scale = float(scale_range[1])
        self.p = float(p)

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        if (self.min_scale != 1.0 or self.max_scale != 1.0) and torch.rand(1).item() < self.p:
            alpha = self.min_scale + torch.rand(1).item() * (self.max_scale - self.min_scale)
            mean = tensor.mean(dim=(-2, -1), keepdim=True)
            return torch.clamp((tensor - mean) * alpha + mean, 0.0, 1.0)
        return tensor


class RandomIlluminationGradient:
    """Add a smooth linear brightness gradient across the patch to simulate residual vignetting."""

    def __init__(self, max_strength: float = 0.05, p: float = 0.3):
        self.max_strength = float(max_strength)
        self.p = float(p)

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.max_strength > 0 and torch.rand(1).item() < self.p:
            _, h, w = tensor.shape
            gx = (torch.rand(1).item() * 2.0 - 1.0) * self.max_strength
            gy = (torch.rand(1).item() * 2.0 - 1.0) * self.max_strength
            ys = torch.linspace(-1.0, 1.0, h, device=tensor.device, dtype=tensor.dtype).view(1, h, 1)
            xs = torch.linspace(-1.0, 1.0, w, device=tensor.device, dtype=tensor.dtype).view(1, 1, w)
            grad = gy * ys + gx * xs
            return torch.clamp(tensor + grad, 0.0, 1.0)
        return tensor


NORMALIZE_MODES = ("fixed", "patch_mean", "patch_median", "patch_zscore")


class PerPatchNormalize:
    """Center each patch by its own spatial mean/median to remove vignetting.

    Modes:
    - ``patch_mean``: ``(x - mean(x)) / std`` (removes per-patch brightness offset,
      preserves contrast differences between classes).
    - ``patch_median``: ``(x - median(x)) / std`` (robust to outliers).
    - ``patch_zscore``: ``(x - mean(x)) / (std(x) + eps)`` (full per-patch z-score).
    """

    def __init__(
        self,
        mode: str = "patch_mean",
        std: Sequence[float] = (0.5,),
        eps: float = 1e-6,
    ):
        if mode not in {"patch_mean", "patch_median", "patch_zscore"}:
            raise ValueError(f"Unsupported PerPatchNormalize mode: {mode!r}")
        self.mode = mode
        self.std = torch.tensor(list(std), dtype=torch.float32).view(-1, 1, 1)
        self.eps = float(eps)

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.mode == "patch_median":
            flat = tensor.reshape(tensor.shape[0], -1)
            center = flat.median(dim=-1).values.view(-1, 1, 1)
        else:
            center = tensor.mean(dim=(-2, -1), keepdim=True)

        centered = tensor - center
        if self.mode == "patch_zscore":
            scale = tensor.std(dim=(-2, -1), keepdim=True).clamp_min(self.eps)
            return centered / scale

        scale = self.std.to(device=tensor.device, dtype=tensor.dtype)
        return centered / scale

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(mode={self.mode!r})"


def build_normalize(normalize_cfg: dict | None, channels: int) -> Transform:
    cfg = normalize_cfg or {}
    mode = str(cfg.get("mode", "fixed"))
    mean = list(cfg.get("mean", [0.5] * channels))
    std = list(cfg.get("std", [0.5] * channels))
    if len(mean) == 1 and channels > 1:
        mean = mean * channels
    if len(std) == 1 and channels > 1:
        std = std * channels
    if mode == "fixed":
        return T.Normalize(mean=mean, std=std)
    return PerPatchNormalize(
        mode=mode,
        std=std,
        eps=float(cfg.get("eps", 1e-6)),
    )


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

        contrast = aug_cfg.get("contrast") or {}
        if float(contrast.get("prob", 0.0)) > 0:
            scale_range: Sequence[float] = contrast.get("scale", (0.8, 1.25))
            steps.append(
                RandomContrastScale(
                    scale_range=tuple(float(s) for s in scale_range),
                    p=float(contrast["prob"]),
                )
            )

        illum = aug_cfg.get("illumination_gradient") or {}
        if float(illum.get("prob", 0.0)) > 0:
            steps.append(
                RandomIlluminationGradient(
                    max_strength=float(illum.get("max_strength", 0.05)),
                    p=float(illum["prob"]),
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
        return " -> ".join(
            repr(step) if isinstance(step, PerPatchNormalize) else type(step).__name__
            for step in transform.transforms
        )
    return repr(transform) if isinstance(transform, PerPatchNormalize) else type(transform).__name__
