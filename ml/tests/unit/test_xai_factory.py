from __future__ import annotations

import pytest

from ml.xai.attention_rollout import AttentionRolloutExplainer
from ml.xai.factory import get_explainer
from ml.xai.gradcam import GradCAMExplainer


@pytest.mark.parametrize("model_name", ["efficientnetv2_s", "convnext_tiny"])
def test_get_explainer_returns_gradcam_for_cnn(model_name):
    explainer = get_explainer(model_name)
    assert isinstance(explainer, GradCAMExplainer)


def test_get_explainer_returns_attention_rollout_for_vit():
    explainer = get_explainer("vit_base_patch16_224")
    assert isinstance(explainer, AttentionRolloutExplainer)


def test_get_explainer_unknown_model_raises():
    with pytest.raises(ValueError, match="no explainer available"):
        get_explainer("not_a_real_model")
