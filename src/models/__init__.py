from __future__ import annotations

import timm
import torch.nn as nn


def create_model(
    num_classes: int = 5,
    pretrained: bool = True,
    grayscale: bool = True,
    freeze_backbone: bool = False,
    dropout: float = 0.0,
    drop_path_rate: float = 0.1,
) -> nn.Module:
    # timm adapts pretrained stem weights to in_chans=1 by averaging
    # the RGB channels, so no manual layer surgery is needed.
    model = timm.create_model(
        "fastvit_t8.apple_in1k",
        pretrained=pretrained,
        in_chans=1 if grayscale else 3,
        num_classes=num_classes,
        drop_rate=dropout,
        drop_path_rate=drop_path_rate,
    )

    if freeze_backbone:
        for parameter in model.parameters():
            parameter.requires_grad = False
        for parameter in model.get_classifier().parameters():
            parameter.requires_grad = True

    return model


__all__ = ["create_model"]
