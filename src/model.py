from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import timm
import torch
from timm.models import adapt_input_conv
from torch import nn

from .hipervit import HIPERVIT_NAMES, build_hipervit
from .logging_utils import get_logger
from .twistnet import TwistNet
from .vortex import VORTEX_NAMES, build_vortex

_TEXTURE_CNN_NAMES = {"texture_cnn", "tcnn"}
_TWISTNET_NAMES = {"twistnet18", "twistnet", "twistnet2d", "twistnet_2d"}


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
    """FastViT (any timm model), Texture CNN, TwistNet-2D, HiPerViT, or VORTEX.

    With ``in_chans=1`` timm folds the pretrained RGB stem weights into a
    single channel, so ImageNet features are kept for grayscale input.
    ``model.weights`` loads a local state dict instead of the timm checkpoint
    (for example ``models/brodatz_fastvit_t8.pth``). A classifier whose width
    does not match ``num_classes`` is left randomly initialized.
    ``texture_cnn`` / ``tcnn`` and ``twistnet18`` are trained from scratch.
    ``hipervit`` keeps a pretrained ViT backbone when ``pretrained`` is true
    and trains the statistical-token fusion head from scratch.
    ``vortex`` freezes the ViT and trains only the linear head on the
    orderless token encoding.
    """
    name = str(model_cfg.get("name", "fastvit_t8")).lower().replace("-", "_")
    weights = model_cfg.get("weights") or None
    custom = (
        name in _TEXTURE_CNN_NAMES
        or name in _TWISTNET_NAMES
        or name in HIPERVIT_NAMES
        or name in VORTEX_NAMES
    )
    if weights and custom:
        raise ValueError(
            f"{name} does not load model.weights; use it with a timm model such as fastvit_t8"
        )
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
    elif name in _TWISTNET_NAMES:
        if model_cfg.get("pretrained", False):
            raise ValueError("twistnet18 has no pretrained checkpoint; set model.pretrained: false")
        model = TwistNet(
            num_classes=num_classes,
            in_chans=in_chans,
            num_heads=int(model_cfg.get("num_heads", 4)),
            c_red=int(model_cfg.get("c_red", 8)),
            stem_type=str(model_cfg.get("stem_type", "resnet")),
        )
    elif name in HIPERVIT_NAMES:
        model = build_hipervit(model_cfg, num_classes=num_classes, in_chans=in_chans)
    elif name in VORTEX_NAMES:
        model = build_vortex(model_cfg, num_classes=num_classes, in_chans=in_chans)
    else:
        use_timm_pretrained = bool(model_cfg.get("pretrained", True)) and not weights
        model = timm.create_model(
            name,
            pretrained=use_timm_pretrained,
            num_classes=num_classes,
            in_chans=in_chans,
            drop_rate=float(model_cfg.get("dropout", 0.0)),
            drop_path_rate=float(model_cfg.get("drop_path_rate", 0.0)),
        )
        if weights:
            if model_cfg.get("pretrained", True):
                get_logger().info(
                    "model.weights=%s is used instead of the timm ImageNet checkpoint",
                    weights,
                )
            load_model_weights(model, weights, in_chans)

    if model_cfg.get("freeze_backbone", False):
        head_params = {id(p) for p in head_parameters(model)}
        for param in model.parameters():
            if id(param) not in head_params:
                param.requires_grad_(False)

    return model


def load_model_weights(model: nn.Module, path: str | Path, in_chans: int) -> None:
    """Load a timm state dict, folding an RGB stem and skipping a foreign head.

    Accepts a raw ``state_dict`` (as saved for the Brodatz FastViT) or a
    mapping with ``model_state``, ``state_dict``, or ``model``. The Brodatz
    checkpoint is FastViT-T8 with a 3-channel stem and a 64-class head. For
    grayscale input the two stem convolutions are summed over RGB, the same
    way timm adapts ImageNet weights. The classification layer is copied only
    when its shape matches the model being trained.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"model.weights file not found: {path}")

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = _strip_module_prefix(_checkpoint_state_dict(checkpoint))
    model_state = model.state_dict()
    classifiers = _classifier_names(model)
    dropped_head: List[str] = []
    for name in classifiers:
        for suffix in ("weight", "bias"):
            key = f"{name}.{suffix}"
            if key not in state:
                continue
            if key not in model_state or tuple(state[key].shape) != tuple(model_state[key].shape):
                state.pop(key)
                dropped_head.append(key)

    adapted = _adapt_stem(state, model_state, model, in_chans)
    filtered: Dict[str, torch.Tensor] = {}
    unexpected: List[str] = []
    mismatched: List[str] = []
    for key, value in state.items():
        if key not in model_state:
            unexpected.append(key)
            continue
        if tuple(value.shape) != tuple(model_state[key].shape):
            mismatched.append(
                f"{key}: checkpoint {tuple(value.shape)} vs model {tuple(model_state[key].shape)}"
            )
            continue
        filtered[key] = value
    if mismatched:
        raise ValueError("Cannot load model.weights, shape mismatch:\n  - " + "\n  - ".join(mismatched))

    head_keys = {f"{name}.{suffix}" for name in classifiers for suffix in ("weight", "bias")}
    missing = [key for key in model_state if key not in filtered]
    missing_backbone = [key for key in missing if key not in head_keys]
    if missing_backbone:
        preview = ", ".join(missing_backbone[:8])
        extra = "" if len(missing_backbone) <= 8 else f" (+{len(missing_backbone) - 8} more)"
        raise ValueError(f"model.weights is missing backbone tensors: {preview}{extra}")

    model.load_state_dict(filtered, strict=False)
    logger = get_logger()
    logger.info(
        "Loaded %s: %d tensors, new head %s, stem adapted %s",
        path,
        len(filtered),
        dropped_head or "no",
        adapted or "no",
    )
    if unexpected:
        logger.info("Ignored %d unexpected checkpoint keys", len(unexpected))


def _checkpoint_state_dict(checkpoint: object) -> dict:
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Unsupported checkpoint type: {type(checkpoint).__name__}")
    for key in ("model_state", "state_dict", "model"):
        nested = checkpoint.get(key)
        if isinstance(nested, dict) and _is_state_dict(nested):
            return nested
    if _is_state_dict(checkpoint):
        return checkpoint
    raise ValueError("Checkpoint must be a state_dict or a mapping with model_state, state_dict, or model")


def _is_state_dict(mapping: dict) -> bool:
    return bool(mapping) and all(torch.is_tensor(value) for value in mapping.values())


def _strip_module_prefix(state: dict) -> dict:
    if state and all(key.startswith("module.") for key in state):
        return {key[len("module."):]: value for key, value in state.items()}
    return state


def _classifier_names(model: nn.Module) -> Tuple[str, ...]:
    configured = (getattr(model, "pretrained_cfg", None) or {}).get("classifier", "head.fc")
    if isinstance(configured, str):
        return (configured,)
    return tuple(configured)


def _adapt_stem(
    state: dict,
    model_state: dict,
    model: nn.Module,
    in_chans: int,
) -> List[str]:
    if in_chans == 3:
        return []
    first_conv = (getattr(model, "pretrained_cfg", None) or {}).get("first_conv") or ()
    if isinstance(first_conv, str):
        first_conv = (first_conv,)
    adapted: List[str] = []
    for name in first_conv:
        key = f"{name}.weight"
        if key not in state or key not in model_state:
            continue
        if tuple(state[key].shape) == tuple(model_state[key].shape):
            continue
        state[key] = adapt_input_conv(in_chans, state[key])
        adapted.append(key)
    return adapted


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
