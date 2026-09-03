"""One entrypoint for "give me an explainer for this model" — the API/
frontend layer calls this instead of branching on architecture itself.
"""

from __future__ import annotations

from ml.models.factory import CNN_MODELS, VIT_MODELS
from ml.xai.attention_rollout import AttentionRolloutExplainer
from ml.xai.gradcam import GradCAMExplainer
from ml.xai.interface import Explainer


def get_explainer(model_name: str) -> Explainer:
    if model_name in CNN_MODELS:
        return GradCAMExplainer(model_name)
    if model_name in VIT_MODELS:
        return AttentionRolloutExplainer()
    raise ValueError(
        f"no explainer available for model {model_name!r}; supported CNNs: "
        f"{sorted(CNN_MODELS)}, supported ViTs: {sorted(VIT_MODELS)}"
    )
