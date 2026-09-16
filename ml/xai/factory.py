"""One entrypoint for "give me an explainer for this model" — the API/
frontend layer calls this instead of branching on architecture itself.
"""

from __future__ import annotations

from ml.models.factory import CNN_MODELS, VIT_MODELS
from ml.models.pretrained_detector import PRETRAINED_MODEL_NAMES
from ml.xai.attention_rollout import AttentionRolloutExplainer
from ml.xai.gradcam import GradCAMExplainer
from ml.xai.hf_attention_rollout import HFAttentionRolloutExplainer
from ml.xai.interface import Explainer


def get_explainer(model_name: str) -> Explainer:
    if model_name in CNN_MODELS:
        return GradCAMExplainer(model_name)
    if model_name in VIT_MODELS:
        return AttentionRolloutExplainer()
    if model_name in PRETRAINED_MODEL_NAMES:
        return HFAttentionRolloutExplainer()
    raise ValueError(
        f"no explainer available for model {model_name!r}; supported CNNs: "
        f"{sorted(CNN_MODELS)}, supported ViTs: {sorted(VIT_MODELS)}, "
        f"plus {sorted(PRETRAINED_MODEL_NAMES)!r}"
    )
