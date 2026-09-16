"""Real, already-trained public classifiers used as this project's default
detector until a from-scratch checkpoint is trained on a licensed benchmark
dataset (see docs/methodology.md's "no real training yet" caveat).

This project's own CNN/ViT models (ml/models/factory.py) are only ever as
good as what they were trained on — with no real dataset downloaded yet,
`efficientnetv2_s`/`vit_base_patch16_224` checkpoints in this repo are
pipeline-validation artifacts (trained on a synthetic fixture set), not real
detectors, and give effectively arbitrary verdicts on real photos/videos.

Two genuinely different kinds of "fake" exist, with different artifact
signatures, and one pretrained model does not cover both:
  - Classic deepfakes: a real photo/video with a GAN-based face-swap or
    reenactment applied. `PretrainedViTDeepfakeDetector` (fine-tuned on this).
  - Wholly AI-generated images (Midjourney/DALL-E/Stable Diffusion): no real
    photo underneath at all — a categorically different generation process
    with a different artifact signature the deepfake model was never trained
    to recognize. `AIGeneratedImageDetector` (fine-tuned on this instead).
`EnsembleFakeDetector` runs both and flags fake if *either* is confident.

That OR-combination was tried as the default and reverted: verified against
5 genuine (non-AI) real photos and 5 real face portraits, `AIGeneratedImageDetector`
(Organika/sdxl-detector) alone called 3 of the 5 plain photos "artificial"
at 92-100% confidence — a real, reproducible false-positive rate on ordinary
photography outside its training distribution (confirmed by running the
identical images through transformers' own `pipeline()` API directly,
bypassing this project's wrapper entirely — not a preprocessing bug here).
Because the ensemble takes whichever sub-model is *more* confident, that
false positive silently overrides an otherwise-correct "real" verdict from
the well-behaved deepfake sub-model. So `PretrainedViTDeepfakeDetector` alone
is `MODEL_NAME` (the default); `EnsembleFakeDetector` stays available as an
explicit, separately-named opt-in (see backend/app/api/models.py) for a
caller who specifically suspects a wholly-AI-generated image and accepts the
higher false-positive rate on ordinary photos in exchange for catching that
category too.

Every class here is kept behind the exact same nn.Module-with-a-single-fake-
logit contract as every other model in this project (see
ml/models/predictor.py) so it plugs into the unchanged Predictor/
calibration/abstain pipeline, and into the unchanged video face-tracking
pipeline, with no other code changes.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

# This project's own preprocessing (ml/datasets/image_dataset.py) always
# normalizes with plain ImageNet stats before a model ever sees the tensor.
# Each wrapped model's own processor may use different stats — resolved at
# load time by reading its real preprocessor_config, not assumed.
_PROJECT_MEAN = (0.485, 0.456, 0.406)
_PROJECT_STD = (0.229, 0.224, 0.225)

# Classic deepfake (GAN face-swap/reenactment on a real photo) classifier.
# id2label = {0: "Real", 1: "Fake"} — confirmed via its published config.
DEEPFAKE_MODEL_ID = "dima806/deepfake_vs_real_image_detection"

# Wholly-AI-generated-image (Midjourney/DALL-E/Stable Diffusion) classifier.
# id2label = {0: "artificial", 1: "human"} — confirmed via its published config.
AI_IMAGE_MODEL_ID = "Organika/sdxl-detector"

MODEL_NAME = "pretrained_vit_deepfake"
ENSEMBLE_MODEL_NAME = "pretrained_ensemble_detector"

# Any model_name handled by this module — used where callers need to branch
# on "is this one of the pretrained wrapper models" generically (config
# lookups, the explainer factory) rather than singling out the default.
PRETRAINED_MODEL_NAMES = frozenset({MODEL_NAME, ENSEMBLE_MODEL_NAME})


class _HFSingleLogitClassifier(nn.Module):
    """Base class: wraps one Hugging Face image-classification model as a
    single-logit binary model — forward(x) -> (batch, 1), higher = more
    "fake". Handles re-normalizing the incoming (already ImageNet-normalized)
    tensor to the wrapped model's own mean/std/size, and resolving which
    output index means "fake" vs "real" from the model's own id2label rather
    than a hardcoded index.
    """

    def __init__(self, model_id: str, *, fake_keywords: tuple[str, ...], real_keywords: tuple[str, ...]):
        super().__init__()
        from transformers import AutoImageProcessor, AutoModelForImageClassification

        # "eager" (not the default fused "sdpa" kernel) so attention weights
        # are actually materialized — needed for gradient/attention-based
        # explainers to read them via output_attentions.
        self.hf_model = AutoModelForImageClassification.from_pretrained(model_id, attn_implementation="eager")
        self.hf_model.eval()

        processor = AutoImageProcessor.from_pretrained(model_id)
        mean = getattr(processor, "image_mean", None) or [0.5, 0.5, 0.5]
        std = getattr(processor, "image_std", None) or [0.5, 0.5, 0.5]
        size_cfg = getattr(processor, "size", None) or {}
        target_size = size_cfg.get("height") or size_cfg.get("shortest_edge") or 224

        self.register_buffer("_hf_mean", torch.tensor(mean).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("_hf_std", torch.tensor(std).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("_project_mean", torch.tensor(_PROJECT_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("_project_std", torch.tensor(_PROJECT_STD).view(1, 3, 1, 1), persistent=False)
        self._target_size = int(target_size)

        id2label = {int(k): v.lower() for k, v in self.hf_model.config.id2label.items()}
        fake_indices = [i for i, name in id2label.items() if any(kw in name for kw in fake_keywords)]
        real_indices = [i for i, name in id2label.items() if any(kw in name for kw in real_keywords)]
        if not fake_indices or not real_indices:
            raise ValueError(
                f"could not determine fake/real label indices from {model_id}'s "
                f"id2label={self.hf_model.config.id2label!r}"
            )
        self._fake_index = fake_indices[0]
        self._real_index = real_indices[0]

    def _renormalize(self, x: torch.Tensor) -> torch.Tensor:
        x = x * self._project_std + self._project_mean
        x = (x - self._hf_mean) / self._hf_std
        if x.shape[-1] != self._target_size or x.shape[-2] != self._target_size:
            x = F.interpolate(
                x, size=(self._target_size, self._target_size), mode="bilinear", align_corners=False
            )
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._renormalize(x)
        logits = self.hf_model(pixel_values=x).logits  # (batch, num_labels)
        # sigmoid(fake_logit - real_logit) == softmax(logits)[fake_index] —
        # an exact identity, giving a single-logit output compatible with
        # this project's Predictor/Calibration convention (see
        # ml/models/predictor.py) without approximation.
        fake_logit = logits[:, self._fake_index] - logits[:, self._real_index]
        return fake_logit.unsqueeze(-1)


class PretrainedViTDeepfakeDetector(_HFSingleLogitClassifier):
    def __init__(self, model_id: str = DEEPFAKE_MODEL_ID):
        super().__init__(model_id, fake_keywords=("fake", "deepfake"), real_keywords=("real",))


class AIGeneratedImageDetector(_HFSingleLogitClassifier):
    def __init__(self, model_id: str = AI_IMAGE_MODEL_ID):
        super().__init__(model_id, fake_keywords=("artificial", "fake", "ai"), real_keywords=("human", "real"))


class EnsembleFakeDetector(nn.Module):
    """Runs both detectors and takes whichever is more confident the input
    is fake — catches classic deepfakes AND wholly-AI-generated images
    without the caller needing to know which category applies. Kept
    differentiable throughout (both sub-models, and torch.maximum rather
    than a hard Python branch) so gradient/attention-based explainers can
    still be attempted against either branch.
    """

    def __init__(self):
        super().__init__()
        self.deepfake_detector = PretrainedViTDeepfakeDetector()
        self.ai_image_detector = AIGeneratedImageDetector()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        deepfake_logit = self.deepfake_detector(x)
        ai_image_logit = self.ai_image_detector(x)
        # max(P(fake) from either detector), converted back to a single
        # logit so sigmoid(returned value) == max(p1, p2) exactly.
        p_fake = torch.maximum(torch.sigmoid(deepfake_logit), torch.sigmoid(ai_image_logit))
        p_fake = p_fake.clamp(1e-6, 1 - 1e-6)
        return torch.log(p_fake / (1 - p_fake))
