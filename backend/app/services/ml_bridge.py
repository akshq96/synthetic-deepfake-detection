"""The seam between the FastAPI app and the `ml` research package.

Inference (image/video detect) runs in-process and synchronously — fast
enough on CPU for single-request use, and this is a solo/local-first tool
(see the project plan's job-execution-model decision). Training/experiment
runs go through backend/app/services/job_launcher.py instead (subprocess).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import cv2
import torch
from torch import nn

from backend.app.core.config import settings
from ml.evaluation.evaluate import load_calibration_if_exists, load_trained_model
from ml.models.predictor import Predictor
from ml.training.calibrate import Calibration
from ml.xai.factory import get_explainer
from ml.xai.interface import Explainer
from ml.xai.overlay import save_overlay


class NoTrainedModelConfiguredError(RuntimeError):
    pass


def _require_checkpoint_configured() -> tuple[str, str]:
    if not settings.default_checkpoint_path:
        raise NoTrainedModelConfiguredError(
            "No trained checkpoint is configured (settings.default_checkpoint_path is unset). "
            "Train a model first (see ml/training/train.py or the Synthetic Data Lab), then set "
            "DEFAULT_CHECKPOINT_PATH (and DEFAULT_CALIBRATION_PATH) in the environment/.env."
        )
    return settings.default_checkpoint_path, settings.default_model_name


@lru_cache(maxsize=4)
def _load_model_cached(checkpoint_path: str, model_name: str) -> nn.Module:
    from ml.models.factory import ModelConfig

    return load_trained_model(checkpoint_path, ModelConfig(name=model_name, pretrained=False))


def get_default_predictor(*, abstain_margin: float = 0.1) -> tuple[Predictor, str, int]:
    """Returns (predictor, model_name, image_size) for the currently
    configured default checkpoint. Raises NoTrainedModelConfiguredError if
    none is set.
    """
    checkpoint_path, model_name = _require_checkpoint_configured()
    model = _load_model_cached(checkpoint_path, model_name)
    calibration = (
        load_calibration_if_exists(settings.default_calibration_path)
        if settings.default_calibration_path
        else Calibration.identity()
    )
    predictor = Predictor(model, calibration=calibration, abstain_margin=abstain_margin)
    return predictor, model_name, settings.default_image_size


def get_default_explainer() -> Explainer:
    _, model_name = _require_checkpoint_configured()
    return get_explainer(model_name)


def get_default_model() -> nn.Module:
    checkpoint_path, model_name = _require_checkpoint_configured()
    return _load_model_cached(checkpoint_path, model_name)


def detect_image(image_path: str | Path, *, save_heatmap: bool = True) -> dict:
    """Runs the default model on one image. Returns a dict matching
    PredictionOut's fields (minus DB-only fields like id/created_at).
    """
    from ml.datasets.image_dataset import default_transform

    predictor, model_name, image_size = get_default_predictor()

    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise ValueError(f"could not read image at {image_path!r}")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    transform = default_transform(image_size, train=False)
    image_tensor = transform(image_rgb).unsqueeze(0)

    prediction = predictor.predict_batch(image_tensor)[0]

    heatmap_path = None
    if save_heatmap:
        explainer = get_default_explainer()
        model = get_default_model()
        heatmap = explainer.explain(model, image_tensor, target_class=1)[0]
        out_path = settings.heatmaps_dir / f"{Path(image_path).stem}_heatmap.png"
        save_overlay(image_bgr, heatmap, out_path)
        heatmap_path = str(out_path)

    return {
        "input_type": "image",
        "model_name": model_name,
        "label": prediction.label,
        "confidence": prediction.confidence,
        "fake_probability": prediction.fake_probability,
        "abstained": prediction.abstained,
        "heatmap_path": heatmap_path,
    }


def detect_video(video_path: str | Path, *, top_k_suspicious: int = 5, save_heatmaps: bool = True) -> dict:
    from ml.video.pipeline import analyze_video

    _, model_name = _require_checkpoint_configured()
    model = get_default_model()
    calibration = (
        load_calibration_if_exists(settings.default_calibration_path)
        if settings.default_calibration_path
        else Calibration.identity()
    )
    explainer = get_default_explainer() if save_heatmaps else None
    heatmap_dir = settings.heatmaps_dir if save_heatmaps else None

    result = analyze_video(
        str(video_path),
        model,
        calibration=calibration,
        explainer=explainer,
        heatmap_output_dir=heatmap_dir,
        image_size=settings.default_image_size,
        top_k_suspicious=top_k_suspicious,
    )

    # Video-level confidence/fake_probability: the max across tracks flagged
    # fake (or, if none fake, the min real-probability track) — i.e. whichever
    # track drove the verdict, consistent with aggregate_video's "any track
    # fake" rule.
    track_probs = [
        r["fake_probability"] for r in result.track_results.values() if r["fake_probability"] is not None
    ]
    if not track_probs:
        video_fake_probability, video_confidence = 0.5, 0.5
    elif result.video_label == "fake":
        video_fake_probability = max(track_probs)
        video_confidence = video_fake_probability
    else:
        video_fake_probability = max(track_probs)
        video_confidence = 1.0 - video_fake_probability

    return {
        "input_type": "video",
        "model_name": model_name,
        "label": result.video_label,
        "confidence": video_confidence,
        "fake_probability": video_fake_probability,
        "abstained": result.video_label == "abstain",
        "heatmap_path": None,
        "suspicious_frames": result.suspicious_frames,
    }
