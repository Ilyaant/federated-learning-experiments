"""TwistNet-2D (Lian et al., 2026): ResNet-18 with spiral-twisted channel interaction.

Stages 1–2 are standard residual blocks. Stages 3–4 are TwistBlocks: a residual
path plus a gated multi-head STCI branch (directions 0°, 45°, 90°, 135°).
The network is trained from scratch; there is no published checkpoint.
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import nn


def _conv3x3(in_ch: int, out_ch: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False)


def _conv1x1(in_ch: int, out_ch: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_ch, out_ch, 1, stride, bias=False)


class BasicBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, downsample: Optional[nn.Module] = None) -> None:
        super().__init__()
        self.conv1 = _conv3x3(in_ch, out_ch, stride)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = _conv3x3(out_ch, out_ch)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.downsample(x) if self.downsample is not None else x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)


class SpiralTwist(nn.Module):
    """Depthwise 3×3 shift initialized to blend the center with one neighbor."""

    _ANGLES = (0, 45, 90, 135)

    def __init__(self, dim: int, direction: int) -> None:
        super().__init__()
        self.direction = direction % 4
        self.dwconv = nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False)
        self.scale = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.reset_directional_kernel()

    def reset_directional_kernel(self) -> None:
        with torch.no_grad():
            weight = self.dwconv.weight
            weight.zero_()
            center = weight.shape[-1] // 2
            weight[:, :, center, center] = 0.5
            angle = math.radians(self._ANGLES[self.direction])
            nx = center + int(round(math.cos(angle)))
            ny = center + int(round(math.sin(angle)))
            if 0 <= ny < weight.shape[-2] and 0 <= nx < weight.shape[-1]:
                weight[:, :, ny, nx] = 0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dwconv(x) * self.scale


class SpiralTwistedInteractionHead(nn.Module):
    """One STCI head: reduce channels, shift, then upper-triangular products."""

    def __init__(self, in_ch: int, c_red: int, direction: int, use_spiral: bool = True) -> None:
        super().__init__()
        self.reduce = nn.Sequential(
            _conv1x1(in_ch, c_red),
            nn.BatchNorm2d(c_red),
            nn.ReLU(inplace=True),
        )
        self.twist: nn.Module = SpiralTwist(c_red, direction) if use_spiral else nn.Identity()

        idx_i, idx_j = [], []
        for i in range(c_red):
            for j in range(i, c_red):
                idx_i.append(i)
                idx_j.append(j)
        self.register_buffer("idx_i", torch.tensor(idx_i))
        self.register_buffer("idx_j", torch.tensor(idx_j))
        self.out_dim = c_red + len(idx_i)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.reduce(x)
        z_twist = self.twist(z)
        z_norm = F.normalize(z, p=2, dim=1, eps=1e-6)
        z_twist_norm = F.normalize(z_twist, p=2, dim=1, eps=1e-6)
        interactions = z_norm[:, self.idx_i] * z_twist_norm[:, self.idx_j]
        return torch.cat([z_norm, interactions], dim=1)


class AdaptiveInteractionSelection(nn.Module):
    """SE-style gate on the concatenated interaction channels."""

    def __init__(self, dim: int, reduction: int = 4) -> None:
        super().__init__()
        mid = max(dim // reduction, 16)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(dim, mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, dim),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.fc(x).view(x.size(0), -1, 1, 1)


class MultiHeadSpiralTwistedInteraction(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        num_heads: int = 4,
        c_red_list: Optional[Sequence[int]] = None,
        use_ais: bool = True,
        use_spiral: bool = True,
    ) -> None:
        super().__init__()
        reductions = list(c_red_list) if c_red_list is not None else [8] * num_heads
        heads = min(num_heads, 4)
        self.heads = nn.ModuleList(
            [
                SpiralTwistedInteractionHead(
                    in_ch,
                    reductions[i % len(reductions)],
                    direction=i % 4,
                    use_spiral=use_spiral,
                )
                for i in range(heads)
            ]
        )
        total_dim = sum(head.out_dim for head in self.heads)
        self.ais: nn.Module = AdaptiveInteractionSelection(total_dim) if use_ais else nn.Identity()
        num_groups = min(32, total_dim)
        while total_dim % num_groups != 0 and num_groups > 1:
            num_groups -= 1
        self.norm = nn.GroupNorm(num_groups, total_dim)
        self.proj = nn.Sequential(_conv1x1(total_dim, out_ch), nn.BatchNorm2d(out_ch))

    def reset_projection(self) -> None:
        nn.init.normal_(self.proj[0].weight, std=0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        inter = torch.cat([head(x) for head in self.heads], dim=1)
        return self.proj(self.norm(self.ais(inter)))


class TwistBlock(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        num_heads: int = 4,
        c_red_list: Optional[Sequence[int]] = None,
        use_ais: bool = True,
        use_spiral: bool = True,
        gate_init: float = -2.0,
    ) -> None:
        super().__init__()
        self.conv1 = _conv3x3(in_ch, out_ch, stride)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = _conv3x3(out_ch, out_ch)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)
        self.mhstci = MultiHeadSpiralTwistedInteraction(
            out_ch,
            out_ch,
            num_heads=num_heads,
            c_red_list=c_red_list,
            use_ais=use_ais,
            use_spiral=use_spiral,
        )
        self.gate = nn.Parameter(torch.tensor(float(gate_init)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.downsample(x) if self.downsample is not None else x
        hidden = self.relu(self.bn1(self.conv1(x)))
        main = self.bn2(self.conv2(hidden))
        interaction = self.mhstci(hidden) * torch.sigmoid(self.gate)
        return self.relu(main + interaction + identity)


class TwistNet(nn.Module):
    """TwistNet-18: widths (64, 128, 256, 512), two blocks per stage."""

    def __init__(
        self,
        num_classes: int,
        in_chans: int = 1,
        layers: Sequence[int] = (2, 2, 2, 2),
        base_width: int = 64,
        twist_stages: Tuple[int, ...] = (3, 4),
        num_heads: int = 4,
        c_red: int = 8,
        use_ais: bool = True,
        use_spiral: bool = True,
        gate_init: float = -2.0,
        stem_type: str = "resnet",
    ) -> None:
        super().__init__()
        self.twist_stages = set(twist_stages)
        self.num_heads = num_heads
        self.c_red_list = [c_red] * num_heads
        self.use_ais = use_ais
        self.use_spiral = use_spiral
        self.gate_init = gate_init

        if stem_type == "resnet":
            self.stem = nn.Sequential(
                nn.Conv2d(in_chans, base_width, 7, stride=2, padding=3, bias=False),
                nn.BatchNorm2d(base_width),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            )
        elif stem_type == "lightweight":
            self.stem = nn.Sequential(
                nn.Conv2d(in_chans, base_width // 2, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(base_width // 2),
                nn.ReLU(inplace=True),
                nn.Conv2d(base_width // 2, base_width, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(base_width),
                nn.ReLU(inplace=True),
            )
        else:
            raise ValueError(f"stem_type must be resnet or lightweight, got {stem_type!r}")

        widths = [base_width * (2 ** i) for i in range(4)]
        self.in_ch = base_width
        self.layer1 = self._make_layer(1, widths[0], layers[0], stride=1)
        self.layer2 = self._make_layer(2, widths[1], layers[1], stride=2)
        self.layer3 = self._make_layer(3, widths[2], layers[2], stride=2)
        self.layer4 = self._make_layer(4, widths[3], layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(widths[3], num_classes)
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        # Kaiming above replaces the paper's special inits; put them back.
        for module in self.modules():
            if isinstance(module, SpiralTwist):
                module.reset_directional_kernel()
            elif isinstance(module, MultiHeadSpiralTwistedInteraction):
                module.reset_projection()

    def _make_layer(self, stage_id: int, out_ch: int, blocks: int, stride: int) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.in_ch != out_ch:
            downsample = nn.Sequential(_conv1x1(self.in_ch, out_ch, stride), nn.BatchNorm2d(out_ch))
        use_twist = stage_id in self.twist_stages
        layers: List[nn.Module] = []
        if use_twist:
            layers.append(
                TwistBlock(
                    self.in_ch,
                    out_ch,
                    stride,
                    downsample,
                    num_heads=self.num_heads,
                    c_red_list=self.c_red_list,
                    use_ais=self.use_ais,
                    use_spiral=self.use_spiral,
                    gate_init=self.gate_init,
                )
            )
        else:
            layers.append(BasicBlock(self.in_ch, out_ch, stride, downsample))
        self.in_ch = out_ch
        for _ in range(1, blocks):
            if use_twist:
                layers.append(
                    TwistBlock(
                        self.in_ch,
                        out_ch,
                        num_heads=self.num_heads,
                        c_red_list=self.c_red_list,
                        use_ais=self.use_ais,
                        use_spiral=self.use_spiral,
                        gate_init=self.gate_init,
                    )
                )
            else:
                layers.append(BasicBlock(self.in_ch, out_ch))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return self.head(self.avgpool(x).flatten(1))
