"""The seam between the FastAPI app and the `ml` research package.

Inference (image/video detect) runs in-process and synchronously — fast
enough on CPU for single-request use, and this is a solo/local-first tool
(see the project plan's job-execution-model decision). Training/experiment
runs go through backend/app/services/job_launcher.py instead (subprocess).
"""

from __future__ import annotations

import time
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn

from backend.app.core.config import settings
from ml.data_pipeline.face_detector import MediapipeFaceDetector
from ml.evaluation.evaluate import load_calibration_if_exists, load_trained_model
from ml.models.predictor import Predictor
from ml.training.calibrate import Calibration
from ml.xai.factory import get_explainer
from ml.xai.interface import Explainer
from ml.xai.overlay import save_heatmap_only, save_original, save_overlay


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


def _resolve_checkpoint(checkpoint_path: str | None, model_name: str | None) -> tuple[str, str]:
    """Resolves an optional model-registry override (Milestone B's model
    selector / Milestone D's live-camera model param) against the configured
    default. `checkpoint_path=None` always falls back to the default,
    regardless of `model_name` — matching the previous, override-less
    behavior exactly when no override is passed."""
    if checkpoint_path is not None:
        return checkpoint_path, model_name or settings.default_model_name
    return _require_checkpoint_configured()


@lru_cache(maxsize=4)
def _load_model_cached(checkpoint_path: str, model_name: str) -> nn.Module:
    from ml.models.factory import ModelConfig

    return load_trained_model(checkpoint_path, ModelConfig(name=model_name, pretrained=False))


def get_predictor(
    *, checkpoint_path: str | None = None, model_name: str | None = None, abstain_margin: float = 0.1
) -> tuple[Predictor, str, int]:
    """Returns (predictor, model_name, image_size) for `checkpoint_path` if
    given, else the configured default. Raises NoTrainedModelConfiguredError
    if no override is given and none is configured."""
    resolved_checkpoint, resolved_model_name = _resolve_checkpoint(checkpoint_path, model_name)
    model = _load_model_cached(resolved_checkpoint, resolved_model_name)
    calibration = (
        load_calibration_if_exists(settings.default_calibration_path)
        if settings.default_calibration_path
        else Calibration.identity()
    )
    predictor = Predictor(model, calibration=calibration, abstain_margin=abstain_margin)
    return predictor, resolved_model_name, settings.default_image_size


def get_default_predictor(*, abstain_margin: float = 0.1) -> tuple[Predictor, str, int]:
    """Returns (predictor, model_name, image_size) for the currently
    configured default checkpoint. Raises NoTrainedModelConfiguredError if
    none is set.
    """
    return get_predictor(abstain_margin=abstain_margin)


def get_explainer_for(model_name: str) -> Explainer:
    return get_explainer(model_name)


def get_default_explainer() -> Explainer:
    _, model_name = _require_checkpoint_configured()
    return get_explainer(model_name)


def get_model_for(checkpoint_path: str | None, model_name: str | None) -> nn.Module:
    resolved_checkpoint, resolved_model_name = _resolve_checkpoint(checkpoint_path, model_name)
    return _load_model_cached(resolved_checkpoint, resolved_model_name)


def get_default_model() -> nn.Module:
    checkpoint_path, model_name = _require_checkpoint_configured()
    return _load_model_cached(checkpoint_path, model_name)


def detect_image_array(
    image_bgr: np.ndarray,
    *,
    checkpoint_path: str | None = None,
    model_name: str | None = None,
    save_heatmap: bool = True,
    heatmap_stem: str = "live",
    count_faces: bool = True,
) -> dict:
    """Runs a model on one in-memory BGR image array. The file-based
    `detect_image()` (image uploads) and the live-camera WebSocket endpoint
    both delegate here, so single-image inference exists in exactly one
    place. `checkpoint_path`/`model_name` select a non-default model (the
    Detect page's model selector); omitted, both fall back to the configured
    default exactly as before this parameter existed.
    """
    from ml.datasets.image_dataset import default_transform

    start = time.perf_counter()
    predictor, resolved_model_name, image_size = get_predictor(
        checkpoint_path=checkpoint_path, model_name=model_name
    )

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    transform = default_transform(image_size, train=False)
    image_tensor = transform(image_rgb).unsqueeze(0)

    prediction = predictor.predict_batch(image_tensor)[0]

    heatmap_path = None
    original_path = None
    heatmap_only_path = None
    if save_heatmap:
        explainer = get_explainer_for(resolved_model_name)
        model = get_model_for(checkpoint_path, model_name)
        heatmap = explainer.explain(model, image_tensor, target_class=1)[0]
        overlay_out = settings.heatmaps_dir / f"{heatmap_stem}_heatmap.png"
        save_overlay(image_bgr, heatmap, overlay_out)
        heatmap_path = str(overlay_out)
        original_out = settings.heatmaps_dir / f"{heatmap_stem}_original.png"
        save_original(image_bgr, original_out)
        original_path = str(original_out)
        heatmap_only_out = settings.heatmaps_dir / f"{heatmap_stem}_heatmap_only.png"
        save_heatmap_only(heatmap, image_bgr.shape[:2], heatmap_only_out)
        heatmap_only_path = str(heatmap_only_out)

    n_faces_detected = len(MediapipeFaceDetector().detect(image_bgr)) if count_faces else None
    height, width = image_bgr.shape[:2]
    processing_time_ms = (time.perf_counter() - start) * 1000.0

    return {
        "input_type": "image",
        "model_name": resolved_model_name,
        "label": prediction.label,
        "confidence": prediction.confidence,
        "fake_probability": prediction.fake_probability,
        "abstained": prediction.abstained,
        "heatmap_path": heatmap_path,
        "original_path": original_path,
        "heatmap_only_path": heatmap_only_path,
        "width": width,
        "height": height,
        "n_faces_detected": n_faces_detected,
        "processing_time_ms": processing_time_ms,
    }


def detect_image(
    image_path: str | Path,
    *,
    save_heatmap: bool = True,
    checkpoint_path: str | None = None,
    model_name: str | None = None,
) -> dict:
    """Runs a model on one image file. Returns a dict matching PredictionOut's
    fields (minus DB-only fields like id/created_at).
    """
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise ValueError(f"could not read image at {image_path!r}")
    return detect_image_array(
        image_bgr,
        checkpoint_path=checkpoint_path,
        model_name=model_name,
        save_heatmap=save_heatmap,
        heatmap_stem=Path(image_path).stem,
    )


def detect_video(
    video_path: str | Path,
    *,
    top_k_suspicious: int = 5,
    save_heatmaps: bool = True,
    checkpoint_path: str | None = None,
    model_name: str | None = None,
) -> dict:
    from ml.video.pipeline import analyze_video

    start = time.perf_counter()
    resolved_checkpoint, resolved_model_name = _resolve_checkpoint(checkpoint_path, model_name)
    model = get_model_for(checkpoint_path, model_name)
    calibration = (
        load_calibration_if_exists(settings.default_calibration_path)
        if settings.default_calibration_path
        else Calibration.identity()
    )
    explainer = get_explainer_for(resolved_model_name) if save_heatmaps else None
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

    processing_time_ms = (time.perf_counter() - start) * 1000.0

    return {
        "input_type": "video",
        "model_name": resolved_model_name,
        "label": result.video_label,
        "confidence": video_confidence,
        "fake_probability": video_fake_probability,
        "abstained": result.video_label == "abstain",
        "heatmap_path": None,
        "original_path": None,
        "heatmap_only_path": None,
        "suspicious_frames": result.suspicious_frames,
        "frame_scores": result.frame_scores,
        "n_frames_total": result.n_frames_total,
        "n_frames_analyzed": result.n_frames_sampled,
        "width": result.video_width,
        "height": result.video_height,
        "duration_seconds": result.duration_seconds,
        "n_faces_detected": result.n_tracks,
        "processing_time_ms": processing_time_ms,
    }
