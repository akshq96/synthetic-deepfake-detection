"""Shared explainability interface: the API/frontend can request a heatmap
for any model type + input without knowing whether the underlying model is
a CNN (Grad-CAM++) or a ViT (attention rollout).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import torch
from torch import nn


class Explainer(ABC):
    @abstractmethod
    def explain(
        self, model: nn.Module, input_tensor: torch.Tensor, *, target_class: int = 1
    ) -> np.ndarray:
        """`input_tensor`: shape (N, C, H, W), already normalized the same
        way the model was trained (see ml.datasets.image_dataset). Returns a
        (N, H, W) float array in [0, 1] — a per-sample heatmap at the input's
        spatial resolution, higher = more influence on the prediction.

        `target_class`: 1 = explain "why fake", 0 = explain "why real" — the
        same 0/1 convention as ml.data_pipeline.schema (LABEL_REAL=0,
        LABEL_FAKE=1).
        """
        raise NotImplementedError
