from __future__ import annotations

import torch
import torchvision.transforms as T


class RandomRotation90:
    def __init__(self, p: float = 0.75):
        self.p = p

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() < self.p:
            k = int(torch.randint(1, 4, (1,)).item())
            return torch.rot90(tensor, k, [-2, -1])
        return tensor


class AddGaussianNoise:
    def __init__(self, std: float = 0.05, p: float = 0.2):
        self.std = std
        self.p = p

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() < self.p:
            noise = torch.randn_like(tensor) * self.std
            return torch.clamp(tensor + noise, -1.0, 1.0)
        return tensor


def build_train_transform(aug_cfg: dict | None):
    if not aug_cfg or not aug_cfg.get("enabled", False):
        return None

    transforms_list = []
    if aug_cfg.get("hflip_prob", 0.0) > 0:
        transforms_list.append(T.RandomHorizontalFlip(p=aug_cfg["hflip_prob"]))
    if aug_cfg.get("vflip_prob", 0.0) > 0:
        transforms_list.append(T.RandomVerticalFlip(p=aug_cfg["vflip_prob"]))
    if aug_cfg.get("rot90_prob", 0.0) > 0:
        transforms_list.append(RandomRotation90(p=aug_cfg["rot90_prob"]))
    if aug_cfg.get("rotation_degrees", 0.0) > 0:
        transforms_list.append(T.RandomRotation(degrees=aug_cfg["rotation_degrees"]))
    if aug_cfg.get("blur_prob", 0.0) > 0:
        transforms_list.append(
            T.RandomApply(
                [T.GaussianBlur(kernel_size=aug_cfg.get("blur_kernel_size", 5))],
                p=aug_cfg["blur_prob"],
            )
        )
    if aug_cfg.get("noise_prob", 0.0) > 0:
        transforms_list.append(
            AddGaussianNoise(
                std=aug_cfg.get("noise_std", 0.05),
                p=aug_cfg["noise_prob"],
            )
        )

    return T.Compose(transforms_list) if transforms_list else None
