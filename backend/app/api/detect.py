from __future__ import annotations

from fastapi import APIRouter, Depends, Form, HTTPException, Query, UploadFile
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.api.models import resolve_model
from backend.app.db.base import get_db
from backend.app.models.frame_score import FrameScore
from backend.app.models.prediction import Prediction
from backend.app.models.suspicious_frame import SuspiciousFrame
from backend.app.schemas.prediction import PredictionOut
from backend.app.services import ml_bridge, storage

router = APIRouter(prefix="/api/detect", tags=["detect"])

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def _require_extension(filename: str | None, allowed: set[str], kind: str) -> None:
    from pathlib import Path

    suffix = Path(filename or "").suffix.lower()
    if suffix not in allowed:
        raise HTTPException(
            status_code=400, detail=f"unsupported {kind} file extension {suffix!r}; expected one of {sorted(allowed)}"
        )


def _resolve_model_run(model_run_id: str | None) -> tuple[str | None, str | None]:
    """(checkpoint_path, model_name) overrides for ml_bridge, or (None, None)
    to use the configured default — the Detect page's model selector is
    optional, so a request with no model_run_id behaves exactly as before
    this parameter existed."""
    if not model_run_id:
        return None, None
    return resolve_model(model_run_id)


@router.post("/image", response_model=PredictionOut)
async def detect_image(
    file: UploadFile, model_run_id: str | None = Form(None), db: Session = Depends(get_db)
) -> PredictionOut:
    _require_extension(file.filename, IMAGE_EXTENSIONS, "image")
    try:
        saved_path = await storage.save_upload(file, subdir="images")
    except storage.UploadTooLargeError as e:
        raise HTTPException(status_code=413, detail=str(e)) from e

    checkpoint_path, model_name = _resolve_model_run(model_run_id)
    try:
        result = ml_bridge.detect_image(saved_path, checkpoint_path=checkpoint_path, model_name=model_name)
    except ml_bridge.NoTrainedModelConfiguredError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    prediction = Prediction(
        input_type="image",
        file_path=storage.relative_to_artifacts(saved_path),
        model_name=result["model_name"],
        label=result["label"],
        confidence=result["confidence"],
        fake_probability=result["fake_probability"],
        abstained=result["abstained"],
        heatmap_path=storage.relative_to_artifacts(result["heatmap_path"]) if result["heatmap_path"] else None,
        original_path=storage.relative_to_artifacts(result["original_path"]) if result["original_path"] else None,
        heatmap_only_path=(
            storage.relative_to_artifacts(result["heatmap_only_path"]) if result["heatmap_only_path"] else None
        ),
        video_width=result["width"],
        video_height=result["height"],
        n_faces_detected=result["n_faces_detected"],
        processing_time_ms=result["processing_time_ms"],
        model_run_id=model_run_id,
    )
    db.add(prediction)
    db.commit()
    db.refresh(prediction)
    return PredictionOut.model_validate(prediction)


@router.post("/video", response_model=PredictionOut)
async def detect_video(
    file: UploadFile, model_run_id: str | None = Form(None), db: Session = Depends(get_db)
) -> PredictionOut:
    _require_extension(file.filename, VIDEO_EXTENSIONS, "video")
    try:
        saved_path = await storage.save_upload(file, subdir="videos")
    except storage.UploadTooLargeError as e:
        raise HTTPException(status_code=413, detail=str(e)) from e

    checkpoint_path, model_name = _resolve_model_run(model_run_id)
    try:
        result = ml_bridge.detect_video(saved_path, checkpoint_path=checkpoint_path, model_name=model_name)
    except ml_bridge.NoTrainedModelConfiguredError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    prediction = Prediction(
        input_type="video",
        file_path=storage.relative_to_artifacts(saved_path),
        model_name=result["model_name"],
        label=result["label"],
        confidence=result["confidence"],
        fake_probability=result["fake_probability"],
        abstained=result["abstained"],
        heatmap_path=None,
        n_frames_total=result["n_frames_total"],
        n_frames_analyzed=result["n_frames_analyzed"],
        video_width=result["width"],
        video_height=result["height"],
        duration_seconds=result["duration_seconds"],
        n_faces_detected=result["n_faces_detected"],
        processing_time_ms=result["processing_time_ms"],
        model_run_id=model_run_id,
    )
    db.add(prediction)
    db.flush()  # assigns prediction.id without committing yet

    for frame in result["suspicious_frames"]:
        db.add(
            SuspiciousFrame(
                prediction_id=prediction.id,
                track_id=frame["track_id"],
                frame_index=frame["frame_index"],
                timestamp=frame["timestamp"],
                fake_probability=frame["fake_probability"],
                heatmap_path=storage.relative_to_artifacts(frame["heatmap_path"]) if frame["heatmap_path"] else None,
                original_path=(
                    storage.relative_to_artifacts(frame["original_path"]) if frame["original_path"] else None
                ),
                heatmap_only_path=(
                    storage.relative_to_artifacts(frame["heatmap_only_path"])
                    if frame["heatmap_only_path"]
                    else None
                ),
            )
        )
    for score in result["frame_scores"]:
        db.add(
            FrameScore(
                prediction_id=prediction.id,
                track_id=score["track_id"],
                frame_index=score["frame_index"],
                timestamp=score["timestamp"],
                fake_probability=score["fake_probability"],
            )
        )
    db.commit()
    db.refresh(prediction)
    return PredictionOut.model_validate(prediction)


@router.get("", response_model=list[PredictionOut])
def list_predictions(
    limit: int = Query(50, le=200), offset: int = Query(0, ge=0), db: Session = Depends(get_db)
) -> list[PredictionOut]:
    """Every past detection, most-recent-first — including ones with no
    generated report, which the Reports page can't show. Powers the Saved
    Analyses page."""
    predictions = (
        db.execute(select(Prediction).order_by(Prediction.created_at.desc()).offset(offset).limit(limit))
        .scalars()
        .all()
    )
    return [PredictionOut.model_validate(p) for p in predictions]


@router.get("/{prediction_id}", response_model=PredictionOut)
def get_prediction(prediction_id: str, db: Session = Depends(get_db)) -> PredictionOut:
    prediction = db.get(Prediction, prediction_id)
    if prediction is None:
        raise HTTPException(status_code=404, detail=f"prediction {prediction_id!r} not found")
    return PredictionOut.model_validate(prediction)
