from __future__ import annotations

import pytest
import torch

from ml.models.factory import (
    SUPPORTED_MODELS,
    ModelConfig,
    build_model,
    build_untrained_debug_model,
    is_cnn,
    is_vit,
)


@pytest.mark.parametrize("model_name", list(SUPPORTED_MODELS.keys()))
def test_build_model_forward_pass_shape(model_name):
    model = build_untrained_debug_model(model_name)
    model.eval()
    x = torch.randn(2, 3, 64, 64) if is_cnn(model_name) else torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, 1)


def test_build_model_unknown_name_raises():
    with pytest.raises(ValueError, match="unknown model name"):
        build_model(ModelConfig(name="not_a_real_model"))


def test_build_model_from_dict():
    model = build_model({"name": "efficientnetv2_s", "pretrained": False})
    assert model is not None


def test_is_cnn_is_vit_partition_supported_models():
    for name in SUPPORTED_MODELS:
        assert is_cnn(name) != is_vit(name)  # exactly one is true
