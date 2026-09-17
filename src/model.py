from __future__ import annotations

from typing import Iterable, List, Tuple

import timm
import torch
from torch import nn


def build_model(model_cfg: dict, num_classes: int, in_chans: int) -> nn.Module:
    """FastViT-T8 (or any timm model) adapted to ``in_chans`` and ``num_classes``.

    With ``in_chans=1`` timm folds the pretrained RGB stem weights into a
    single channel, so ImageNet features are kept for grayscale input.
    """
    model = timm.create_model(
        model_cfg.get("name", "fastvit_t8"),
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
