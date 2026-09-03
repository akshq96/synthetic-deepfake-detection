"""Attention rollout (Abnar & Zuidema, 2020) for timm Vision Transformers.

timm's ViT attention module uses PyTorch's fused
`scaled_dot_product_attention` kernel by default (`attn.fused_attn`), which
never materializes the attention weight matrix — so it's invisible to a
forward hook. This explainer temporarily flips `fused_attn = False` on every
block (forcing the manual softmax(QK^T)@V path, which does compute and briefly
expose the attention matrix through `attn.attn_drop`) for the duration of one
forward pass, captures it via hooks, then restores the original flag —
verified end to end in tests rather than assumed.

Algorithm: average each layer's per-head attention over heads, mix with the
identity (accounting for the residual connection around attention) and
row-normalize, then multiply the per-layer matrices together in order. The
resulting CLS-token row says how much each output attends, through the whole
network, to each input patch token.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from ml.xai.interface import Explainer


class AttentionRolloutExplainer(Explainer):
    def __init__(self, discard_ratio: float = 0.0):
        """`discard_ratio`: fraction of lowest attention values to zero out
        per layer before rollout (Abnar & Zuidema's noted variant to reduce
        noise from near-uniform low-attention entries). 0.0 = no discarding.
        """
        self._discard_ratio = discard_ratio

    def explain(
        self, model: nn.Module, input_tensor: torch.Tensor, *, target_class: int = 1
    ) -> np.ndarray:
        del target_class  # attention rollout is class-agnostic (see module docstring)

        # `.blocks` alone isn't a reliable ViT signal — timm's CNN backbones
        # (e.g. EfficientNetV2) also expose a top-level `.blocks` Sequential,
        # just of conv stages, not attention blocks. `.patch_embed` is ViT-specific.
        if not (hasattr(model, "blocks") and hasattr(model, "patch_embed")):
            raise ValueError(
                "AttentionRolloutExplainer requires a timm VisionTransformer-style "
                "model with `.blocks` (attention blocks) and `.patch_embed` attributes"
            )

        was_training = model.training
        model.eval()

        captured: dict[int, torch.Tensor] = {}
        original_fused_flags: dict[int, bool] = {}
        hooks = []

        def make_hook(layer_idx: int):
            def hook(_module, _inputs, output):
                captured[layer_idx] = output.detach()

            return hook

        for i, block in enumerate(model.blocks):
            original_fused_flags[i] = block.attn.fused_attn
            block.attn.fused_attn = False
            hooks.append(block.attn.attn_drop.register_forward_hook(make_hook(i)))

        try:
            with torch.no_grad():
                model(input_tensor)
        finally:
            for h in hooks:
                h.remove()
            for i, block in enumerate(model.blocks):
                block.attn.fused_attn = original_fused_flags[i]
            model.train(was_training)

        if len(captured) != len(model.blocks):
            raise RuntimeError(
                f"expected to capture attention from {len(model.blocks)} blocks, got "
                f"{len(captured)} — the model's attention implementation may differ "
                f"from what this explainer assumes."
            )

        n_tokens = captured[0].shape[-1]
        batch_size = captured[0].shape[0]
        rollout = torch.eye(n_tokens).unsqueeze(0).repeat(batch_size, 1, 1)

        for i in sorted(captured):
            attn = captured[i].mean(dim=1)  # average over heads -> (B, N, N)
            if self._discard_ratio > 0:
                attn = self._discard_lowest(attn, self._discard_ratio)
            identity = torch.eye(n_tokens).unsqueeze(0)
            attn = 0.5 * attn + 0.5 * identity  # residual-connection correction
            attn = attn / attn.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            rollout = attn @ rollout

        cls_attention = rollout[:, 0, 1:]  # CLS token's attention to each patch token
        n_patches = cls_attention.shape[-1]
        grid_size = int(round(n_patches**0.5))
        if grid_size * grid_size != n_patches:
            raise RuntimeError(
                f"attention rollout expected a square patch grid, got {n_patches} patch "
                f"tokens (not a perfect square) — non-square patch grids aren't supported"
            )

        heatmap_grid = cls_attention.reshape(batch_size, 1, grid_size, grid_size)
        target_size = input_tensor.shape[-2:]
        heatmap = torch.nn.functional.interpolate(
            heatmap_grid, size=target_size, mode="bilinear", align_corners=False
        ).squeeze(1)

        heatmap = heatmap - heatmap.amin(dim=(1, 2), keepdim=True)
        heatmap = heatmap / heatmap.amax(dim=(1, 2), keepdim=True).clamp_min(1e-8)
        return heatmap.numpy().astype(np.float32)

    @staticmethod
    def _discard_lowest(attn: torch.Tensor, ratio: float) -> torch.Tensor:
        flat = attn.flatten(1)
        n_discard = int(flat.shape[1] * ratio)
        if n_discard == 0:
            return attn
        threshold = flat.kthvalue(n_discard, dim=1, keepdim=True).values
        mask = flat >= threshold
        return (flat * mask).reshape(attn.shape)
