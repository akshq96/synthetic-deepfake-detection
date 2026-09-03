from __future__ import annotations

import numpy as np
import pytest
import torch

from ml.models.factory import build_untrained_debug_model
from ml.xai.attention_rollout import AttentionRolloutExplainer


def test_attention_rollout_shape_and_range():
    model = build_untrained_debug_model("vit_base_patch16_224")
    model.eval()
    explainer = AttentionRolloutExplainer()

    x = torch.randn(2, 3, 224, 224)
    heatmap = explainer.explain(model, x)

    assert heatmap.shape == (2, 224, 224)
    assert heatmap.dtype == np.float32
    assert heatmap.min() >= 0.0 - 1e-5
    assert heatmap.max() <= 1.0 + 1e-5


def test_attention_rollout_restores_fused_attn_flag():
    model = build_untrained_debug_model("vit_base_patch16_224")
    original_flags = [block.attn.fused_attn for block in model.blocks]
    explainer = AttentionRolloutExplainer()
    x = torch.randn(1, 3, 224, 224)
    explainer.explain(model, x)
    restored_flags = [block.attn.fused_attn for block in model.blocks]
    assert restored_flags == original_flags


def test_attention_rollout_restores_training_mode():
    model = build_untrained_debug_model("vit_base_patch16_224")
    model.train()
    explainer = AttentionRolloutExplainer()
    x = torch.randn(1, 3, 224, 224)
    explainer.explain(model, x)
    assert model.training is True


def test_attention_rollout_rejects_non_vit_model():
    from ml.models.factory import build_untrained_debug_model as build

    cnn_model = build("efficientnetv2_s")
    explainer = AttentionRolloutExplainer()
    x = torch.randn(1, 3, 64, 64)
    with pytest.raises(ValueError, match="requires a timm VisionTransformer"):
        explainer.explain(cnn_model, x)


def test_attention_rollout_with_discard_ratio():
    model = build_untrained_debug_model("vit_base_patch16_224")
    model.eval()
    explainer = AttentionRolloutExplainer(discard_ratio=0.5)
    x = torch.randn(1, 3, 224, 224)
    heatmap = explainer.explain(model, x)
    assert heatmap.shape == (1, 224, 224)
