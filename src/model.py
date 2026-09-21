from __future__ import annotations

from typing import Iterable, List, Tuple

import timm
import torch
from torch import nn

_TEXTURE_CNN_NAMES = {"texture_cnn", "tcnn"}


class TextureCNN(nn.Module):
    """T-CNN (Andrearczyk & Whelan, 2016): AlexNet conv stack plus an energy layer.

    The energy layer is the mean of each feature map, so the classifier sees
    filter energies rather than spatial layout. ``depth`` is the number of
    conv layers (T-CNN-1 .. T-CNN-5). Pooling stays only between conv layers;
    the pool after the last conv is replaced by the energy layer, so any
    patch size is valid.
    """

    _CONV = (
        (96, 11, 4, 2),
        (256, 5, 1, 2),
        (384, 3, 1, 1),
        (384, 3, 1, 1),
        (256, 3, 1, 1),
    )
    _POOL_AFTER = {0, 1}

    def __init__(
        self,
        num_classes: int,
        in_chans: int = 1,
        depth: int = 3,
        dropout: float = 0.5,
        fc_dim: int = 4096,
    ) -> None:
        super().__init__()
        if not 1 <= depth <= len(self._CONV):
            raise ValueError(f"texture_cnn depth must be 1..{len(self._CONV)}, got {depth}")

        layers: List[nn.Module] = []
        channels = in_chans
        for index, (out_channels, kernel, stride, padding) in enumerate(self._CONV[:depth]):
            layers += [
                nn.Conv2d(channels, out_channels, kernel, stride=stride, padding=padding),
                nn.ReLU(inplace=True),
            ]
            if index < 2:
                layers.append(nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2.0))
            if index in self._POOL_AFTER and index < depth - 1:
                layers.append(nn.MaxPool2d(kernel_size=3, stride=2))
            channels = out_channels

        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(channels, fc_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fc_dim, fc_dim),
            nn.ReLU(inplace=True),
            nn.Linear(fc_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x).mean(dim=(2, 3)))


def build_model(model_cfg: dict, num_classes: int, in_chans: int) -> nn.Module:
    """FastViT (any timm model) or Texture CNN, adapted to ``in_chans`` and ``num_classes``.

    With ``in_chans=1`` timm folds the pretrained RGB stem weights into a
    single channel, so ImageNet features are kept for grayscale input.
    ``texture_cnn`` / ``tcnn`` is trained from scratch: it has no checkpoint.
    """
    name = model_cfg.get("name", "fastvit_t8")
    if name in _TEXTURE_CNN_NAMES:
        if model_cfg.get("pretrained", False):
            raise ValueError("texture_cnn has no pretrained checkpoint; set model.pretrained: false")
        model: nn.Module = TextureCNN(
            num_classes=num_classes,
            in_chans=in_chans,
            depth=int(model_cfg.get("depth", 3)),
            dropout=float(model_cfg.get("dropout", 0.5)),
            fc_dim=int(model_cfg.get("fc_dim", 4096)),
        )
    else:
        model = timm.create_model(
            name,
            pretrained=bool(model_cfg.get("pretrained", True)),
            num_classes=num_classes,
            in_chans=in_chans,
            drop_rate=float(model_cfg.get("dropout", 0.0)),
            drop_path_rate=float(model_cfg.get("drop_path_rate", 0.0)),
        )

    if model_cfg.get("freeze_backbone", False):
        head_params = {id(p) for p in head_parameters(model)}
        for param in model.parameters():
            if id(param) not in head_params:
                param.requires_grad_(False)

    return model


def head_parameters(model: nn.Module) -> Iterable[nn.Parameter]:
    head = getattr(model, "head", None)
    if head is None:
        head = model.get_classifier()
    return head.parameters()


def split_parameters(model: nn.Module) -> Tuple[List[nn.Parameter], List[nn.Parameter]]:
    """(backbone, head) trainable parameters."""
    head_ids = {id(p) for p in head_parameters(model)}
    backbone, head = [], []
    for param in model.parameters():
        if not param.requires_grad:
            continue
        (head if id(param) in head_ids else backbone).append(param)
    return backbone, head


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def resolve_device(name: str | None) -> torch.device:
    if name in (None, "auto"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)
