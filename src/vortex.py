"""VORTEX (Scabini et al., 2025): orderless randomized encodings of ViT tokens.

Spatial tokens from every backbone block are stacked and column-normalized.
Each image is then encoded by a soup of randomized autoencoders: a fixed LCG
projection, a sigmoid, and a least-squares decoder whose weights are the image
descriptor. The soup sums those weights. The ViT stays frozen; a linear layer
maps the descriptor to logits so the existing trainer can replace the paper's
SVM.
"""

from __future__ import annotations

from typing import List, Optional

import timm
import torch
import torch.nn.functional as F
from torch import nn

VORTEX_NAMES = {"vortex", "vortex_b", "vortex_s", "vortex_beit"}

_BACKBONE_FOR_NAME = {
    "vortex": "vit_base_patch16_224.orig_in21k",
    "vortex_b": "vit_base_patch16_224.orig_in21k",
    "vortex_s": "vit_small_patch16_224.augreg_in1k",
    "vortex_beit": "beitv2_base_patch16_224.in1k_ft_in22k_in1k",
}

_LCG_A = 75.0
_LCG_B = 74.0
_LCG_C = float((1 << 16) + 1)


def _native_img_size(backbone_name: str) -> int:
    cfg = timm.get_pretrained_cfg(backbone_name)
    if cfg is not None and getattr(cfg, "input_size", None):
        return int(cfg.input_size[-1])
    return 224


def _lcg_matrix(rows: int, cols: int, seed: int) -> torch.Tensor:
    """Z-scored LCG slice used by the official VORTEX encoder.

    The stream starts at 0 with ``x <- (75x + 74) mod (2^16+1)``. Encoder ``k``
    reads ``q * d`` values at offset ``k * q * d``.
    """
    length = rows * cols
    if length == 1:
        return torch.ones(rows, cols)
    values = torch.zeros(seed + length, dtype=torch.float64)
    for index in range(1, values.numel()):
        values[index] = (_LCG_A * values[index - 1] + _LCG_B) % _LCG_C
    sample = values[seed : seed + length]
    sample = (sample - sample.mean()) / sample.std(unbiased=True)
    return sample.reshape(rows, cols).float()


def _make_orthogonal(tensor: torch.Tensor) -> torch.Tensor:
    """QR orthogonalization used by VORTEX (sign-corrected)."""
    rows = tensor.size(0)
    flattened = tensor.reshape(rows, -1)
    transposed = rows < flattened.shape[1]
    if transposed:
        flattened = flattened.t()
    orthogonal, upper = torch.linalg.qr(flattened)
    orthogonal = orthogonal * torch.diag(upper).sign()
    if transposed:
        orthogonal = orthogonal.t()
    return orthogonal


class RandomizedAutoencoder(nn.Module):
    """One frozen random encoder and a per-image least-squares decoder."""

    def __init__(self, dim: int, hidden: int, seed: int) -> None:
        super().__init__()
        weight = _make_orthogonal(_lcg_matrix(hidden, dim, seed))
        self.register_buffer("weight", weight)
        self.register_buffer("bias", torch.ones(hidden))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """``tokens`` is ``[B, N, D]``. Returns decoder weights ``[B, Q*D]``."""
        values = tokens.float()
        hidden = torch.sigmoid(F.linear(values, self.weight, self.bias))
        solution = torch.linalg.lstsq(hidden, values).solution
        return solution.reshape(solution.shape[0], -1).to(dtype=tokens.dtype)


class VORTEX(nn.Module):
    """Frozen ViT plus a VORTEX token soup and a linear classifier.

    ``m`` is the number of randomized autoencoders in the soup (the paper uses
    16). ``rae_hidden`` is the encoder width ``q``; the paper keeps ``q=1`` so
    the descriptor stays the same size as the ViT embedding.
    """

    def __init__(
        self,
        num_classes: int,
        in_chans: int = 3,
        backbone: str = "vit_base_patch16_224.orig_in21k",
        pretrained: bool = True,
        m: int = 16,
        rae_hidden: int = 1,
        dropout: float = 0.0,
        token_norm: bool = True,
        train_backbone: bool = False,
    ) -> None:
        super().__init__()
        if int(m) < 1:
            raise ValueError("model.m must be a positive number of RAEs")
        if int(rae_hidden) < 1:
            raise ValueError("model.rae_hidden must be positive")
        self.m = int(m)
        self.rae_hidden = int(rae_hidden)
        self.token_norm = bool(token_norm)
        self.train_backbone = bool(train_backbone)

        self.backbone = _create_backbone(backbone, pretrained=pretrained, in_chans=in_chans)
        if not hasattr(self.backbone, "forward_intermediates"):
            raise ValueError(f"VORTEX backbone must expose intermediate tokens, got {backbone!r}")
        patch = getattr(getattr(self.backbone, "patch_embed", None), "patch_size", 16)
        self.patch_size = int(patch[0] if isinstance(patch, (tuple, list)) else patch)
        self.embed_dim = int(getattr(self.backbone, "embed_dim", self.backbone.num_features))

        self.encoders = nn.ModuleList(
            RandomizedAutoencoder(
                dim=self.embed_dim,
                hidden=self.rae_hidden,
                seed=index * self.rae_hidden * self.embed_dim,
            )
            for index in range(self.m)
        )
        self.feature_dim = self.rae_hidden * self.embed_dim
        self.dropout = nn.Dropout(float(dropout))
        self.head = nn.Linear(self.feature_dim, num_classes)
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)
        if not self.train_backbone:
            self.backbone.requires_grad_(False)
            self.backbone.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if not self.train_backbone:
            self.backbone.eval()
        return self

    def _fit_patch(self, images: torch.Tensor) -> torch.Tensor:
        patch = self.patch_size
        height, width = images.shape[-2:]
        if height < patch or width < patch:
            side = max(height, width, patch)
            images = F.interpolate(images, size=(side, side), mode="bicubic", align_corners=False)
            height, width = images.shape[-2:]
        pad_h = (patch - height % patch) % patch
        pad_w = (patch - width % patch) % patch
        if pad_h or pad_w:
            images = F.pad(images, (0, pad_w, 0, pad_h), mode="reflect")
        return images

    def _spatial_tokens(self, images: torch.Tensor) -> torch.Tensor:
        if self.train_backbone:
            layers = self._intermediate_maps(images)
        else:
            with torch.no_grad():
                layers = self._intermediate_maps(images)
        maps = torch.stack(layers, dim=1)
        tokens = maps.flatten(3).transpose(2, 3).flatten(1, 2)
        if self.token_norm:
            tokens = F.normalize(tokens.float(), p=2, dim=1, eps=1e-10).to(dtype=maps.dtype)
        return tokens

    def _intermediate_maps(self, images: torch.Tensor) -> List[torch.Tensor]:
        layers = self.backbone.forward_intermediates(
            images,
            indices=None,
            norm=False,
            output_fmt="NCHW",
            intermediates_only=True,
        )
        if not layers:
            raise RuntimeError(f"{type(self.backbone).__name__} returned no intermediate maps")
        channels = {tuple(layer.shape[1:]) for layer in layers}
        if len(channels) != 1:
            raise ValueError(
                "VORTEX expects every block to emit the same map shape; "
                f"got {sorted(channels)}"
            )
        return layers

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        tokens = self._spatial_tokens(images)
        encoded = [encoder(tokens) for encoder in self.encoders]
        return torch.stack(encoded, dim=0).sum(dim=0)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4:
            raise ValueError(f"VORTEX expected a batch of images [B, C, H, W], got {tuple(images.shape)}")
        features = self.encode(self._fit_patch(images))
        return self.head(self.dropout(features))


def _create_backbone(name: str, pretrained: bool, in_chans: int) -> nn.Module:
    kwargs = dict(
        pretrained=pretrained,
        num_classes=0,
        in_chans=in_chans,
        img_size=_native_img_size(name),
    )
    try:
        return timm.create_model(name, dynamic_img_size=True, **kwargs)
    except TypeError:
        return timm.create_model(name, **kwargs)


def build_vortex(model_cfg: dict, num_classes: int, in_chans: int) -> VORTEX:
    name = str(model_cfg.get("name", "vortex")).lower().replace("-", "_")
    backbone = model_cfg.get("backbone") or _BACKBONE_FOR_NAME.get(name, _BACKBONE_FOR_NAME["vortex"])
    soup = model_cfg.get("m", model_cfg.get("num_encoders", 16))
    return VORTEX(
        num_classes=num_classes,
        in_chans=in_chans,
        backbone=str(backbone),
        pretrained=bool(model_cfg.get("pretrained", True)),
        m=int(soup),
        rae_hidden=int(model_cfg.get("rae_hidden", 1)),
        dropout=float(model_cfg.get("dropout", 0.0)),
        token_norm=bool(model_cfg.get("token_norm", True)),
        train_backbone=bool(model_cfg.get("train_backbone", False)),
    )
