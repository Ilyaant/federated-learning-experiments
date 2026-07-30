from __future__ import annotations

import torch
import torch.nn as nn
import timm


class FastViTT8(nn.Module):
    def __init__(
        self,
        num_classes: int = 5,
        pretrained: bool = True,
        grayscale: bool = True,
        freeze_backbone: bool = False,
        dropout: float = 0.0,
        drop_path_rate: float = 0.1,
    ):
        super().__init__()

        self.num_classes = num_classes

        self.backbone = timm.create_model(
            "fastvit_t8.apple_in1k",
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
            drop_rate=dropout,
            drop_path_rate=drop_path_rate,
        )

        self.feature_dim = self.backbone.num_features

        self.classifier = nn.Linear(
            self.feature_dim,
            num_classes,
        )

        if grayscale:
            self._adapt_input_layer()

        if freeze_backbone:
            self.freeze_backbone()

    def _adapt_input_layer(self):
        conv = None

        for module in self.backbone.modules():
            if isinstance(module, nn.Conv2d):
                conv = module
                break

        if conv is None:
            raise RuntimeError(
                "Unable to locate first Conv2d layer."
            )

        new_conv = nn.Conv2d(
            in_channels=1,
            out_channels=conv.out_channels,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            dilation=conv.dilation,
            groups=conv.groups,
            bias=conv.bias is not None,
        )

        with torch.no_grad():
            new_conv.weight.copy_(
                conv.weight.mean(dim=1, keepdim=True)
            )

            if conv.bias is not None:
                new_conv.bias.copy_(conv.bias)

        for name, module in self.backbone.named_modules():
            for child_name, child in module.named_children():
                if child is conv:
                    setattr(module, child_name, new_conv)
                    return

        raise RuntimeError(
            "Unable to replace first Conv2d layer."
        )

    def freeze_backbone(self):
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False

        for parameter in self.classifier.parameters():
            parameter.requires_grad = True

    def unfreeze_backbone(self):
        for parameter in self.parameters():
            parameter.requires_grad = True

    def forward_features(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return self.backbone(x)

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        features = self.forward_features(x)
        logits = self.classifier(features)
        return logits


def create_model(
    num_classes: int = 5,
    pretrained: bool = True,
    grayscale: bool = True,
    freeze_backbone: bool = False,
    dropout: float = 0.0,
    drop_path_rate: float = 0.1,
):
    return FastViTT8(
        num_classes=num_classes,
        pretrained=pretrained,
        grayscale=grayscale,
        freeze_backbone=freeze_backbone,
        dropout=dropout,
        drop_path_rate=drop_path_rate,
    )
