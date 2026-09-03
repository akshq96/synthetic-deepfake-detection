from __future__ import annotations

import numpy as np
import pytest
import torch

from ml.models.factory import build_untrained_debug_model
from ml.xai.gradcam import GradCAMExplainer


@pytest.mark.parametrize("model_name", ["efficientnetv2_s", "convnext_tiny"])
def test_gradcam_explain_shape_and_range(model_name):
    model = build_untrained_debug_model(model_name)
    model.eval()
    explainer = GradCAMExplainer(model_name)

    x = torch.randn(2, 3, 64, 64)
    heatmap = explainer.explain(model, x, target_class=1)

    assert heatmap.shape == (2, 64, 64)
    assert heatmap.dtype == np.float32
    assert heatmap.min() >= 0.0
    assert heatmap.max() <= 1.0 + 1e-5


def test_gradcam_unknown_model_raises():
    with pytest.raises(ValueError, match="no Grad-CAM target layer registered"):
        GradCAMExplainer("not_a_real_model")


def test_gradcam_requires_grad_enabled():
    model = build_untrained_debug_model("efficientnetv2_s")
    model.eval()
    explainer = GradCAMExplainer("efficientnetv2_s")
    x = torch.randn(1, 3, 64, 64)
    with torch.no_grad(), pytest.raises(RuntimeError, match="autograd disabled"):
        explainer.explain(model, x)


def test_gradcam_restores_model_training_mode():
    model = build_untrained_debug_model("efficientnetv2_s")
    model.train()
    explainer = GradCAMExplainer("efficientnetv2_s")
    x = torch.randn(1, 3, 64, 64)
    explainer.explain(model, x)
    assert model.training is True  # restored to its original (train) mode
