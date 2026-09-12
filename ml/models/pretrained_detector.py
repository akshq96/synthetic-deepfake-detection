"""A real, already-trained deepfake detector, used as this project's default
model until a from-scratch checkpoint is trained on a licensed benchmark
dataset (see docs/methodology.md's "no real training yet" caveat).

This project's own CNN/ViT models (ml/models/factory.py) are only ever as
good as what they were trained on — with no real dataset downloaded yet,
`efficientnetv2_s`/`vit_base_patch16_224` checkpoints in this repo are
pipeline-validation artifacts (trained on a synthetic fixture set), not real
detectors, and give effectively arbitrary verdicts on real photos/videos.
This module wraps a public, already-trained deepfake-classification model
(a ViT fine-tuned on real deepfake data by its authors) so uploads get a
genuinely meaningful verdict today, without requiring local training.

Kept behind the exact same nn.Module-with-a-single-fake-logit contract as
every other model in this project (see ml/models/predictor.py) so it plugs
into the unchanged Predictor/calibration/abstain pipeline, and into the
unchanged video face-tracking pipeline, with no other code changes.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

# Real/fake face binary classifier, fine-tuned by its authors on real
# deepfake-vs-real image data (not something trained by this project).
# id2label = {0: "Real", 1: "Fake"} — confirmed via its published config.
PRETRAINED_MODEL_ID = "dima806/deepfake_vs_real_image_detection"

# This project's own preprocessing (ml/datasets/image_dataset.py) always
# normalizes with plain ImageNet stats before a model ever sees the tensor.
# The pretrained model's own processor may use different stats — resolved at
# load time by reading its real preprocessor_config, not assumed.
_PROJECT_MEAN = (0.485, 0.456, 0.406)
_PROJECT_STD = (0.229, 0.224, 0.225)

MODEL_NAME = "pretrained_vit_deepfake"


class PretrainedViTDeepfakeDetector(nn.Module):
    """Wraps a Hugging Face ViT deepfake classifier as a single-logit binary
    model: forward(x) -> (batch, 1) where a higher value means more "fake".

    Re-normalizes the incoming (already ImageNet-normalized, by this
    project's own transform) tensor to whatever mean/std/size the wrapped
    model's own processor actually specifies, entirely with differentiable
    tensor ops (no PIL round-trip) — Grad-CAM/gradient-based explainers can
    still backprop through this if a caller wants to attempt it, though none
    is wired up for this architecture yet (see ml/xai/factory.py).
    """

    def __init__(self, model_id: str = PRETRAINED_MODEL_ID):
        super().__init__()
        from transformers import AutoImageProcessor, AutoModelForImageClassification

        # "eager" (not the default fused "sdpa" kernel) so attention weights
        # are actually materialized — required for HFAttentionRolloutExplainer
        # (ml/xai/hf_attention_rollout.py) to read them via output_attentions.
        self.hf_model = AutoModelForImageClassification.from_pretrained(model_id, attn_implementation="eager")
        self.hf_model.eval()

        processor = AutoImageProcessor.from_pretrained(model_id)
        mean = getattr(processor, "image_mean", None) or [0.5, 0.5, 0.5]
        std = getattr(processor, "image_std", None) or [0.5, 0.5, 0.5]
        size_cfg = getattr(processor, "size", None) or {}
        target_size = size_cfg.get("height") or size_cfg.get("shortest_edge") or 224

        self.register_buffer("_hf_mean", torch.tensor(mean).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("_hf_std", torch.tensor(std).view(1, 3, 1, 1), persistent=False)
        self.register_buffer(
            "_project_mean", torch.tensor(_PROJECT_MEAN).view(1, 3, 1, 1), persistent=False
        )
        self.register_buffer(
            "_project_std", torch.tensor(_PROJECT_STD).view(1, 3, 1, 1), persistent=False
        )
        self._target_size = int(target_size)

        id2label = {int(k): v.lower() for k, v in self.hf_model.config.id2label.items()}
        fake_indices = [i for i, name in id2label.items() if "fake" in name or "deepfake" in name]
        real_indices = [i for i, name in id2label.items() if "real" in name]
        if not fake_indices or not real_indices:
            raise ValueError(
                f"could not determine real/fake label indices from {model_id}'s "
                f"id2label={self.hf_model.config.id2label!r}"
            )
        self._fake_index = fake_indices[0]
        self._real_index = real_indices[0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Undo this project's ImageNet normalization, then apply the
        # pretrained model's own — an exact re-normalization, not a guess.
        x = x * self._project_std + self._project_mean
        x = (x - self._hf_mean) / self._hf_std
        if x.shape[-1] != self._target_size or x.shape[-2] != self._target_size:
            x = F.interpolate(
                x, size=(self._target_size, self._target_size), mode="bilinear", align_corners=False
            )

        logits = self.hf_model(pixel_values=x).logits  # (batch, num_labels)
        # sigmoid(fake_logit - real_logit) == softmax(logits)[fake_index] —
        # an exact identity, giving a single-logit output compatible with
        # this project's Predictor/Calibration convention (see
        # ml/models/predictor.py) without approximation.
        fake_logit = logits[:, self._fake_index] - logits[:, self._real_index]
        return fake_logit.unsqueeze(-1)
