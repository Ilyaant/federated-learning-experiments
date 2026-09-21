"""HiPerViT (de Sá & Bruno, 2026): multi-scale ViT with a second-order token.

A shared ViT reads a global view and a local crop of the same image. Tokens from
the chosen blocks stay as a spatial sequence. A Count-Sketch bilinear pool of
those tokens becomes one statistical token, which a short transformer mixes
with the spatial tokens. Learnable latent queries then distill the sequence
with cross-attention, and a linear layer maps the averaged latents to logits.

The module follows the default interact-then-distill wiring (topology A).
``distill_interact`` and ``late_fusion`` are the two other wirings from the
paper. ``forward`` returns logits, not probabilities.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple, Union

import timm
import torch
import torch.nn.functional as F
from timm.models.vision_transformer import Block
from torch import nn
from torchvision.transforms import InterpolationMode, v2

HIPERVIT_NAMES = {
    "hipervit",
    "hiper_vit",
    "hi_per_vit",
    "hipervit_s",
    "hipervit_b",
    "hipervit_l",
    "hipervit_s14",
    "hipervit_b14",
    "hipervit_l14",
}

_BACKBONE_FOR_NAME = {
    "hipervit": "vit_small_patch14_dinov2.lvd142m",
    "hiper_vit": "vit_small_patch14_dinov2.lvd142m",
    "hi_per_vit": "vit_small_patch14_dinov2.lvd142m",
    "hipervit_s": "vit_small_patch14_dinov2.lvd142m",
    "hipervit_s14": "vit_small_patch14_dinov2.lvd142m",
    "hipervit_b": "vit_base_patch14_dinov2.lvd142m",
    "hipervit_b14": "vit_base_patch14_dinov2.lvd142m",
    "hipervit_l": "vit_large_patch14_dinov2.lvd142m",
    "hipervit_l14": "vit_large_patch14_dinov2.lvd142m",
}

_TOPOLOGIES = {
    "interact_distill": "interact_distill",
    "i2d": "interact_distill",
    "a": "interact_distill",
    "pre_interact": "interact_distill",
    "distill_interact": "distill_interact",
    "d2i": "distill_interact",
    "b": "distill_interact",
    "late_fusion": "late_fusion",
    "late": "late_fusion",
    "c": "late_fusion",
}

StageSpec = Union[str, int, Sequence[int]]


def _native_img_size(backbone_name: str) -> int:
    cfg = timm.get_pretrained_cfg(backbone_name)
    if cfg is not None and getattr(cfg, "input_size", None):
        return int(cfg.input_size[-1])
    return 224


def _fraction_block(depth: int, fraction: float) -> int:
    """1-indexed block nearest to ``fraction`` of the backbone depth."""
    return max(1, min(depth, int(round(fraction * depth))))


def resolve_stages(stages: Optional[StageSpec], depth: int) -> List[int]:
    """Map a config value to 0-indexed ViT block indices.

    Positive numbers follow the paper and are 1-indexed. Negative numbers
    count from the end (``-1`` is the last block). Names ``early``, ``mid``,
    ``late`` and ``multi`` pick blocks at 25%, 50% and 100% of the depth.
    """
    if stages is None:
        stages = [1, 5]
    if isinstance(stages, str):
        key = stages.lower().strip()
        named = {
            "early": [_fraction_block(depth, 0.25)],
            "mid": [_fraction_block(depth, 0.5)],
            "late": [depth],
            "multi": [
                _fraction_block(depth, 0.25),
                _fraction_block(depth, 0.5),
                depth,
            ],
            "all": [
                _fraction_block(depth, 0.25),
                _fraction_block(depth, 0.5),
                depth,
            ],
        }
        if key.replace("-", "_") in named:
            stages = named[key.replace("-", "_")]
        else:
            parts = [part for part in key.replace(" ", "").split(",") if part]
            if len(parts) == 1 and parts[0].count("-") == 1 and not parts[0].startswith("-"):
                parts = parts[0].split("-")
            stages = [int(part) for part in parts]
    elif isinstance(stages, int):
        stages = [stages]

    if not isinstance(stages, Sequence) or isinstance(stages, (str, bytes)) or len(stages) == 0:
        raise ValueError(f"model.stages must list ViT blocks, got {stages!r}")

    blocks: List[int] = []
    for stage in stages:
        index = int(stage)
        if index == 0 or index < -depth or index > depth:
            raise ValueError(f"ViT stage {index} is outside 1..{depth} (depth {depth})")
        block = depth + index if index < 0 else index - 1
        if block not in blocks:
            blocks.append(block)
    return blocks


def resolve_topology(name: str) -> str:
    key = str(name).lower().replace("-", "_").strip()
    if key not in _TOPOLOGIES:
        choices = "interact_distill, distill_interact, late_fusion"
        raise ValueError(f"model.topology must be one of {choices}, got {name!r}")
    return _TOPOLOGIES[key]


class CompactBilinearPooling(nn.Module):
    """Count-Sketch approximation of the sum of token outer products.

    Two fixed random sketches are multiplied in the Fourier domain and summed
    over tokens, then signed-square-root and L2 normalised. The sketch is
    invariant to token order.
    """

    def __init__(self, dim: int, sketch_dim: int, seed: int = 0, eps: float = 1e-6) -> None:
        super().__init__()
        if dim <= 0 or sketch_dim < 2:
            raise ValueError(f"sketch_dim must be >= 2 and dim > 0, got dim={dim}, sketch_dim={sketch_dim}")
        generator = torch.Generator()
        generator.manual_seed(int(seed))
        self.sketch_dim = int(sketch_dim)
        self.eps = float(eps)
        self.register_buffer("index_a", torch.randint(0, sketch_dim, (dim,), generator=generator))
        self.register_buffer("index_b", torch.randint(0, sketch_dim, (dim,), generator=generator))
        signs_a = torch.randint(0, 2, (dim,), generator=generator).float().mul_(2).sub_(1)
        signs_b = torch.randint(0, 2, (dim,), generator=generator).float().mul_(2).sub_(1)
        self.register_buffer("sign_a", signs_a)
        self.register_buffer("sign_b", signs_b)

    def _sketch(self, tokens: torch.Tensor, index: torch.Tensor, sign: torch.Tensor) -> torch.Tensor:
        batch, length, channels = tokens.shape
        sketched = tokens.new_zeros(batch, length, self.sketch_dim)
        sketched.scatter_add_(
            -1,
            index.view(1, 1, channels).expand(batch, length, channels),
            tokens * sign.view(1, 1, channels),
        )
        return sketched

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """``tokens`` is ``[B, N, C]``. Returns ``[B, sketch_dim]``."""
        dtype = tokens.dtype
        values = tokens.float()
        spectrum = torch.fft.rfft(self._sketch(values, self.index_a, self.sign_a), dim=-1)
        spectrum = spectrum * torch.fft.rfft(self._sketch(values, self.index_b, self.sign_b), dim=-1)
        pooled = torch.fft.irfft(spectrum.sum(dim=1), n=self.sketch_dim, dim=-1)
        pooled = torch.sign(pooled) * torch.sqrt(pooled.abs() + self.eps)
        return F.normalize(pooled, p=2, dim=-1, eps=self.eps).to(dtype)


class CrossAttention(nn.Module):
    """Pre-norm cross-attention. Queries attend to the visual sequence."""

    def __init__(self, dim: int, num_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"embed_dim {dim} must be divisible by num_heads {num_heads}")
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)

    def forward(self, queries: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        query = self.norm_q(queries)
        context = self.norm_kv(context)
        update, _ = self.attn(query, context, context, need_weights=False)
        return queries + update


def _transformer(depth: int, dim: int, num_heads: int, dropout: float, drop_path: float) -> nn.Module:
    if depth <= 0:
        return nn.Identity()
    return nn.Sequential(
        *[
            Block(
                dim,
                num_heads,
                qkv_bias=True,
                proj_drop=dropout,
                attn_drop=dropout,
                drop_path=drop_path,
            )
            for _ in range(depth)
        ]
    )


def _init_fusion(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        nn.init.trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)


class HiPerViTHead(nn.Module):
    """Statistical token, cross-order mixing, latent distillation, classifier."""

    def __init__(
        self,
        dim: int,
        num_classes: int,
        num_stages: int,
        num_heads: int,
        sketch_dim: int,
        num_latents: int,
        interaction_layers: int,
        latent_layers: int,
        topology: str,
        dropout: float,
        drop_path: float,
        sketch_seed: int,
    ) -> None:
        super().__init__()
        self.topology = topology
        self.num_latents = int(num_latents)
        self.srm = CompactBilinearPooling(dim, sketch_dim, seed=sketch_seed)
        self.srm_proj = nn.Linear(sketch_dim * num_stages, dim)
        self.spatial_proj = nn.Linear(dim, dim)
        self.cross_attn = CrossAttention(dim, num_heads, dropout=dropout)
        self.latent_tower = _transformer(latent_layers, dim, num_heads, dropout, drop_path)
        self.latents = nn.Parameter(torch.zeros(1, self.num_latents, dim))
        self.dropout = nn.Dropout(dropout)
        classifier_in = dim * 2 if topology == "late_fusion" else dim
        self.classifier = nn.Linear(classifier_in, num_classes)
        if topology == "late_fusion":
            self.cls_token = None
            self.encoder = None
        else:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
            self.encoder = _transformer(interaction_layers, dim, num_heads, dropout, drop_path)
        self.apply(_init_fusion)
        nn.init.trunc_normal_(self.latents, std=0.02)
        if self.cls_token is not None:
            nn.init.trunc_normal_(self.cls_token, std=0.02)

    def _statistical_token(self, stage_tokens: Sequence[torch.Tensor]) -> torch.Tensor:
        sketched = torch.cat([self.srm(tokens) for tokens in stage_tokens], dim=-1)
        return self.srm_proj(sketched).unsqueeze(1)

    def _distill(self, context: torch.Tensor) -> torch.Tensor:
        batch = context.shape[0]
        latents = self.latents.expand(batch, -1, -1)
        latents = self.cross_attn(latents, context)
        return self.latent_tower(latents)

    def forward(self, stage_tokens: Sequence[torch.Tensor]) -> torch.Tensor:
        spatial = self.spatial_proj(torch.cat(list(stage_tokens), dim=1))
        statistical = self._statistical_token(stage_tokens)
        if self.topology == "interact_distill":
            cls = self.cls_token.expand(spatial.shape[0], -1, -1)
            mixed = self.encoder(torch.cat([cls, statistical, spatial], dim=1))
            pooled = self._distill(mixed).mean(dim=1)
            features = pooled
        elif self.topology == "distill_interact":
            distilled = self._distill(spatial)
            cls = self.cls_token.expand(spatial.shape[0], -1, -1)
            mixed = self.encoder(torch.cat([cls, statistical, distilled], dim=1))
            pooled = mixed[:, 2:].mean(dim=1)
            features = pooled
        else:
            pooled = self._distill(spatial).mean(dim=1)
            features = torch.cat([pooled, statistical.squeeze(1)], dim=-1)
        return self.classifier(self.dropout(features))


class HiPerViT(nn.Module):
    """Shared-weight multi-scale ViT plus the HiPerViT fusion head.

    ``global_size`` and ``local_size`` are the side lengths of the two views.
    ``None`` keeps the incoming image size, which is the patch size used by
    the training pipeline. During training the local view is a random resized
    crop with area scale ``local_scale``; at evaluation it is a center crop.
    """

    def __init__(
        self,
        num_classes: int,
        in_chans: int = 3,
        backbone: str = "vit_small_patch14_dinov2.lvd142m",
        pretrained: bool = True,
        stages: Optional[StageSpec] = None,
        sketch_dim: int = 512,
        num_latents: int = 64,
        interaction_layers: int = 1,
        latent_layers: int = 1,
        topology: str = "interact_distill",
        global_size: Optional[int] = None,
        local_size: Optional[int] = None,
        local_scale: Sequence[float] = (0.5, 1.0),
        local_ratio: Sequence[float] = (1.0, 1.0),
        num_heads: Optional[int] = None,
        dropout: float = 0.0,
        drop_path_rate: float = 0.0,
        sketch_seed: int = 0,
    ) -> None:
        super().__init__()
        self.topology = resolve_topology(topology)
        self.global_size = None if global_size in (None, 0) else int(global_size)
        self.local_size = None if local_size in (None, 0) else int(local_size)
        self.local_scale = _pair("model.local_scale", local_scale, upper=1.0)
        self.local_ratio = _pair("model.local_ratio", local_ratio, upper=None)
        if self.global_size is not None and self.global_size <= 0:
            raise ValueError("model.global_size must be positive")
        if self.local_size is not None and self.local_size <= 0:
            raise ValueError("model.local_size must be positive")
        if int(num_latents) < 1:
            raise ValueError("model.num_latents must be positive")
        if int(interaction_layers) < 0 or int(latent_layers) < 0:
            raise ValueError("interaction_layers and latent_layers must be >= 0")

        self.backbone = timm.create_model(
            backbone,
            pretrained=pretrained,
            num_classes=0,
            in_chans=in_chans,
            img_size=_native_img_size(backbone),
            dynamic_img_size=True,
        )
        if not hasattr(self.backbone, "blocks") or not hasattr(self.backbone, "forward_intermediates"):
            raise ValueError(f"HiPerViT backbone must be a timm ViT, got {backbone!r}")

        patch = self.backbone.patch_embed.patch_size
        self.patch_size = int(patch[0] if isinstance(patch, (tuple, list)) else patch)
        depth = len(self.backbone.blocks)
        self.block_indices = resolve_stages(stages, depth)
        # Blocks past the deepest read have no gradient. Drop them so AdamW
        # weight decay cannot touch unused pretrained weights.
        last_block = max(self.block_indices)
        if last_block < depth - 1:
            self.backbone.blocks = self.backbone.blocks[: last_block + 1]
        self.embed_dim = int(self.backbone.embed_dim)
        heads = int(num_heads) if num_heads else _backbone_heads(self.backbone)
        if self.embed_dim % heads != 0:
            raise ValueError(f"embed_dim {self.embed_dim} must be divisible by num_heads {heads}")

        self.head = HiPerViTHead(
            dim=self.embed_dim,
            num_classes=num_classes,
            num_stages=len(self.block_indices),
            num_heads=heads,
            sketch_dim=int(sketch_dim),
            num_latents=int(num_latents),
            interaction_layers=int(interaction_layers),
            latent_layers=int(latent_layers),
            topology=self.topology,
            dropout=float(dropout),
            drop_path=float(drop_path_rate),
            sketch_seed=int(sketch_seed),
        )

    def _resize(self, images: torch.Tensor, size: Optional[int]) -> torch.Tensor:
        if size is None or (images.shape[-2] == size and images.shape[-1] == size):
            return images
        down = size < images.shape[-2] or size < images.shape[-1]
        return F.interpolate(
            images,
            size=(size, size),
            mode="bicubic",
            align_corners=False,
            antialias=down,
        )

    def _center_view(self, images: torch.Tensor, size: Optional[int]) -> torch.Tensor:
        if size is None:
            return images
        height, width = images.shape[-2:]
        if height == size and width == size:
            return images
        if height >= size and width >= size:
            top = (height - size) // 2
            left = (width - size) // 2
            return images[:, :, top : top + size, left : left + size]
        return self._resize(images, size)

    def _random_view(self, images: torch.Tensor) -> torch.Tensor:
        height, width = images.shape[-2:]
        size = (height, width) if self.local_size is None else (self.local_size, self.local_size)
        crop = v2.RandomResizedCrop(
            size=size,
            scale=self.local_scale,
            ratio=self.local_ratio,
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        )
        return crop(images)

    def _fit_patch(self, images: torch.Tensor) -> torch.Tensor:
        patch = self.patch_size
        height, width = images.shape[-2:]
        if height < patch or width < patch:
            images = self._resize(images, max(height, width, patch))
            height, width = images.shape[-2:]
        pad_h = (patch - height % patch) % patch
        pad_w = (patch - width % patch) % patch
        if pad_h or pad_w:
            images = F.pad(images, (0, pad_w, 0, pad_h), mode="reflect")
        return images

    def _views(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        global_view = self._fit_patch(self._resize(images, self.global_size))
        if self.training:
            local_view = self._fit_patch(self._random_view(images))
        else:
            local_view = self._fit_patch(self._center_view(images, self.local_size))
        return global_view, local_view

    def _stage_tokens(self, images: torch.Tensor) -> List[torch.Tensor]:
        taken = self.backbone.forward_intermediates(
            images,
            indices=self.block_indices,
            norm=True,
            stop_early=True,
            output_fmt="NLC",
            intermediates_only=True,
        )
        by_index = {index: tokens for index, tokens in zip(sorted(self.block_indices), taken)}
        return [by_index[index] for index in self.block_indices]

    def _multiscale_tokens(self, images: torch.Tensor) -> List[torch.Tensor]:
        global_view, local_view = self._views(images)
        if global_view is local_view:
            stages = self._stage_tokens(global_view)
            return [torch.cat([tokens, tokens], dim=1) for tokens in stages]
        if global_view.shape[-2:] == local_view.shape[-2:]:
            batch = global_view.shape[0]
            stages = self._stage_tokens(torch.cat([global_view, local_view], dim=0))
            return [torch.cat([tokens[:batch], tokens[batch:]], dim=1) for tokens in stages]
        global_stages = self._stage_tokens(global_view)
        local_stages = self._stage_tokens(local_view)
        return [
            torch.cat([global_tokens, local_tokens], dim=1)
            for global_tokens, local_tokens in zip(global_stages, local_stages)
        ]

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4:
            raise ValueError(f"HiPerViT expected a batch of images [B, C, H, W], got {tuple(images.shape)}")
        return self.head(self._multiscale_tokens(images))


def _pair(name: str, value: Sequence[float], upper: Optional[float]) -> Tuple[float, float]:
    pair = tuple(float(item) for item in value)
    if len(pair) != 2 or pair[0] <= 0 or pair[0] > pair[1]:
        raise ValueError(f"{name} must be a pair lo <= hi with lo > 0, got {value!r}")
    if upper is not None and pair[1] > upper:
        raise ValueError(f"{name} hi must be <= {upper}, got {value!r}")
    return pair[0], pair[1]


def _backbone_heads(backbone: nn.Module) -> int:
    attention = backbone.blocks[0].attn
    if hasattr(attention, "num_heads"):
        return int(attention.num_heads)
    return max(1, int(backbone.embed_dim) // 64)


def _optional_int(value: object) -> Optional[int]:
    if value in (None, 0, "null"):
        return None
    return int(value)


def build_hipervit(model_cfg: dict, num_classes: int, in_chans: int) -> HiPerViT:
    name = str(model_cfg.get("name", "hipervit")).lower().replace("-", "_")
    backbone = model_cfg.get("backbone") or _BACKBONE_FOR_NAME.get(name, "vit_small_patch14_dinov2.lvd142m")
    return HiPerViT(
        num_classes=num_classes,
        in_chans=in_chans,
        backbone=str(backbone),
        pretrained=bool(model_cfg.get("pretrained", True)),
        stages=model_cfg.get("stages"),
        sketch_dim=int(model_cfg.get("sketch_dim", 512)),
        num_latents=int(model_cfg.get("num_latents", 64)),
        interaction_layers=int(model_cfg.get("interaction_layers", 1)),
        latent_layers=int(model_cfg.get("latent_layers", 1)),
        topology=str(model_cfg.get("topology", "interact_distill")),
        global_size=_optional_int(model_cfg.get("global_size")),
        local_size=_optional_int(model_cfg.get("local_size")),
        local_scale=model_cfg.get("local_scale", (0.5, 1.0)),
        local_ratio=model_cfg.get("local_ratio", (1.0, 1.0)),
        num_heads=model_cfg.get("num_heads"),
        dropout=float(model_cfg.get("dropout", 0.0)),
        drop_path_rate=float(model_cfg.get("drop_path_rate", 0.0)),
        sketch_seed=int(model_cfg.get("sketch_seed", 0)),
    )
