"""Attention rollout (Abnar & Zuidema, 2020) for the pretrained Hugging Face
ViT wrapper (ml/models/pretrained_detector.py) — same algorithm as
ml/xai/attention_rollout.py, but that explainer is hard-coded against timm's
specific module layout (`.blocks[i].attn.attn_drop`, a `fused_attn` flag to
flip). A Hugging Face ViTForImageClassification has a different internal
structure entirely, but natively supports `output_attentions=True`, which
hands back every layer's attention weights directly with no hooks needed —
simpler than replicating timm's fused-kernel workaround for a model that
doesn't have that problem.

`EnsembleFakeDetector` (the current default) wraps two sub-models: a plain
ViT deepfake classifier and a Swin-based AI-generated-image classifier. This
explainer only visualizes the ViT sub-model (`.deepfake_detector`) — Swin's
windowed/hierarchical attention doesn't fit the same flat-patch-sequence
rollout math, and adapting it is future work. So this heatmap reflects "what
looked deepfake-like", not "what looked AI-generated", even on inputs where
the AI-image sub-model was what actually drove the fake verdict.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from ml.xai.interface import Explainer


class HFAttentionRolloutExplainer(Explainer):
    def explain(self, model: nn.Module, input_tensor: torch.Tensor, *, target_class: int = 1) -> np.ndarray:
        del target_class  # attention rollout is class-agnostic

        # EnsembleFakeDetector exposes the ViT sub-model as .deepfake_detector;
        # a bare PretrainedViTDeepfakeDetector exposes .hf_model directly —
        # support both so this explainer works either way.
        vit_wrapper = getattr(model, "deepfake_detector", model)
        hf_model = getattr(vit_wrapper, "hf_model", None)
        if hf_model is None:
            raise ValueError(
                "HFAttentionRolloutExplainer requires a model with an `hf_model` "
                "attribute, or a `.deepfake_detector` sub-model that has one "
                "(ml.models.pretrained_detector.PretrainedViTDeepfakeDetector / EnsembleFakeDetector)"
            )

        was_training = hf_model.training
        hf_model.eval()

        # Re-normalize exactly as the wrapper's own forward() does, so the
        # explainer sees the same pixel values the classification itself saw.
        x = input_tensor * vit_wrapper._project_std + vit_wrapper._project_mean
        x = (x - vit_wrapper._hf_mean) / vit_wrapper._hf_std
        if x.shape[-1] != vit_wrapper._target_size or x.shape[-2] != vit_wrapper._target_size:
            x = torch.nn.functional.interpolate(
                x, size=(vit_wrapper._target_size, vit_wrapper._target_size), mode="bilinear", align_corners=False
            )

        with torch.no_grad():
            outputs = hf_model(pixel_values=x, output_attentions=True)
        hf_model.train(was_training)

        attentions = outputs.attentions  # tuple of (B, heads, N, N), one per layer
        if not attentions:
            raise RuntimeError(
                "the wrapped model returned no attention weights for output_attentions=True"
            )

        batch_size, _, n_tokens, _ = attentions[0].shape
        rollout = torch.eye(n_tokens).unsqueeze(0).repeat(batch_size, 1, 1)
        identity = torch.eye(n_tokens).unsqueeze(0)

        for layer_attn in attentions:
            attn = layer_attn.mean(dim=1)  # average over heads -> (B, N, N)
            attn = 0.5 * attn + 0.5 * identity  # residual-connection correction
            attn = attn / attn.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            rollout = attn @ rollout

        cls_attention = rollout[:, 0, 1:]  # CLS token's attention to each patch token
        n_patches = cls_attention.shape[-1]
        grid_size = int(round(n_patches**0.5))
        if grid_size * grid_size != n_patches:
            raise RuntimeError(
                f"attention rollout expected a square patch grid, got {n_patches} patch "
                f"tokens (not a perfect square)"
            )

        heatmap_grid = cls_attention.reshape(batch_size, 1, grid_size, grid_size)
        heatmap = torch.nn.functional.interpolate(
            heatmap_grid, size=input_tensor.shape[-2:], mode="bilinear", align_corners=False
        ).squeeze(1)

        heatmap = heatmap - heatmap.amin(dim=(1, 2), keepdim=True)
        heatmap = heatmap / heatmap.amax(dim=(1, 2), keepdim=True).clamp_min(1e-8)
        return heatmap.numpy().astype(np.float32)
