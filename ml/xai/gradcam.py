"""Grad-CAM++ explainer for CNN architectures, via the `pytorch-grad-cam`
library. Target layer (the last spatial conv feature map before pooling)
differs by architecture — timm's internal module names aren't part of any
public "which layer is the last conv" contract, so this lookup is pinned
here and verified by a unit test per supported CNN, rather than guessed at
call time.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from ml.xai.interface import Explainer

_TARGET_LAYER_LOOKUP = {
    "efficientnetv2_s": lambda model: model.conv_head,
    "convnext_tiny": lambda model: model.stages[-1],
}


class GradCAMExplainer(Explainer):
    def __init__(self, model_name: str):
        if model_name not in _TARGET_LAYER_LOOKUP:
            raise ValueError(
                f"no Grad-CAM target layer registered for {model_name!r}; "
                f"supported: {sorted(_TARGET_LAYER_LOOKUP)}"
            )
        self._model_name = model_name

    def explain(
        self, model: nn.Module, input_tensor: torch.Tensor, *, target_class: int = 1
    ) -> np.ndarray:
        from pytorch_grad_cam import GradCAMPlusPlus
        from pytorch_grad_cam.utils.model_targets import BinaryClassifierOutputTarget

        target_layer = _TARGET_LAYER_LOOKUP[self._model_name](model)
        was_training = model.training
        model.eval()
        try:
            cam = GradCAMPlusPlus(model=model, target_layers=[target_layer])
            targets = [BinaryClassifierOutputTarget(target_class)] * input_tensor.shape[0]
            # pytorch-grad-cam needs gradients through the input; it manages
            # its own no_grad/grad context internally, but the caller may be
            # inside a torch.no_grad() block (e.g. a batch-inference loop) —
            # guard against that producing an all-zero heatmap silently.
            if not torch.is_grad_enabled():
                raise RuntimeError(
                    "GradCAMExplainer.explain() called with autograd disabled "
                    "(e.g. inside torch.no_grad()); Grad-CAM requires gradients."
                )
            grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
        finally:
            model.train(was_training)
        return grayscale_cam.astype(np.float32)
