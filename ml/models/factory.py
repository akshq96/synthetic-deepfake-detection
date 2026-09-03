"""Model factory: one entrypoint for every architecture this project compares.

CNN (EfficientNetV2 / ConvNeXt) and ViT are both built via `timm`, both reduced
to a single-logit binary head (real=0 vs fake=1, BCEWithLogitsLoss), so
`ml/training/train.py` never needs to know which family it's training —
architecture is purely a config choice (`model.name`).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

# Config `model.name` -> timm model id. Kept as an explicit allowlist (rather
# than passing arbitrary strings straight to timm) so a typo fails fast with a
# clear message instead of timm's own error, and so XAI (Phase 7) has a fixed
# set of architectures to special-case target layers for.
SUPPORTED_MODELS: dict[str, str] = {
    "efficientnetv2_s": "tf_efficientnetv2_s",
    "convnext_tiny": "convnext_tiny",
    "vit_base_patch16_224": "vit_base_patch16_224",
}

CNN_MODELS = {"efficientnetv2_s", "convnext_tiny"}
VIT_MODELS = {"vit_base_patch16_224"}


@dataclass(frozen=True)
class ModelConfig:
    name: str
    pretrained: bool = True
    drop_rate: float = 0.0

    @classmethod
    def from_dict(cls, d: dict) -> "ModelConfig":
        return cls(
            name=d["name"],
            pretrained=bool(d.get("pretrained", True)),
            drop_rate=float(d.get("drop_rate", 0.0)),
        )


def is_vit(model_name: str) -> bool:
    return model_name in VIT_MODELS


def is_cnn(model_name: str) -> bool:
    return model_name in CNN_MODELS


def build_model(config: ModelConfig | dict) -> nn.Module:
    """Build a binary-classification model (single logit output) from config.

    `config.name` must be one of SUPPORTED_MODELS' keys (not a raw timm id) —
    this keeps the config file's model names stable/documented even if the
    underlying timm model id ever changes.
    """
    if isinstance(config, dict):
        config = ModelConfig.from_dict(config)

    if config.name not in SUPPORTED_MODELS:
        raise ValueError(
            f"unknown model name {config.name!r}; supported: {sorted(SUPPORTED_MODELS)}"
        )

    import timm

    timm_id = SUPPORTED_MODELS[config.name]
    model = timm.create_model(
        timm_id,
        pretrained=config.pretrained,
        num_classes=1,
        drop_rate=config.drop_rate,
    )
    return model


def build_untrained_debug_model(model_name: str) -> nn.Module:
    """Build a model with `pretrained=False` for fast CPU tests — avoids a
    network call to fetch pretrained weights in unit tests / CI.
    """
    return build_model(ModelConfig(name=model_name, pretrained=False))


def forward_logits(model: nn.Module, images: torch.Tensor) -> torch.Tensor:
    """Run the model and squeeze the single-logit output to shape (batch,)."""
    out = model(images)
    return out.squeeze(-1)
