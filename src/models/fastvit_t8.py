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

    def _make_grayscale_conv(self, conv: nn.Conv2d) -> nn.Conv2d:
        new_conv = nn.Conv2d(
            in_channels=1,
            out_channels=conv.out_channels,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            dilation=conv.dilation,
            groups=1 if conv.in_channels == conv.groups else conv.groups,
            bias=conv.bias is not None,
        )

        with torch.no_grad():
            new_conv.weight.copy_(
                conv.weight.mean(dim=1, keepdim=True)
            )

            if conv.bias is not None:
                new_conv.bias.copy_(conv.bias)

        return new_conv

    @staticmethod
    def _replace_module(
        root: nn.Module,
        qualified_name: str,
        new_module: nn.Module,
    ) -> None:
        parts = qualified_name.split(".")
        parent = root

        for part in parts[:-1]:
            parent = getattr(parent, part)

        setattr(parent, parts[-1], new_module)

    def _adapt_input_layer(self):
        input_convs = [
            (name, module)
            for name, module in self.backbone.named_modules()
            if isinstance(module, nn.Conv2d) and module.in_channels == 3
        ]

        if not input_convs:
            raise RuntimeError(
                "Unable to locate input Conv2d layers."
            )

        for name, conv in input_convs:
            self._replace_module(
                self.backbone,
                name,
                self._make_grayscale_conv(conv),
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
