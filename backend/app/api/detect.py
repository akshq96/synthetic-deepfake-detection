from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, UploadFile
from sqlalchemy.orm import Session

from backend.app.db.base import get_db
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


@router.post("/image", response_model=PredictionOut)
async def detect_image(file: UploadFile, db: Session = Depends(get_db)) -> PredictionOut:
    _require_extension(file.filename, IMAGE_EXTENSIONS, "image")
    try:
        saved_path = await storage.save_upload(file, subdir="images")
    except storage.UploadTooLargeError as e:
        raise HTTPException(status_code=413, detail=str(e)) from e

    try:
        result = ml_bridge.detect_image(saved_path)
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
    )
    db.add(prediction)
    db.commit()
    db.refresh(prediction)
    return PredictionOut.model_validate(prediction)


@router.post("/video", response_model=PredictionOut)
async def detect_video(file: UploadFile, db: Session = Depends(get_db)) -> PredictionOut:
    _require_extension(file.filename, VIDEO_EXTENSIONS, "video")
    try:
        saved_path = await storage.save_upload(file, subdir="videos")
    except storage.UploadTooLargeError as e:
        raise HTTPException(status_code=413, detail=str(e)) from e

    try:
        result = ml_bridge.detect_video(saved_path)
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
            )
        )
    db.commit()
    db.refresh(prediction)
    return PredictionOut.model_validate(prediction)


@router.get("/{prediction_id}", response_model=PredictionOut)
def get_prediction(prediction_id: str, db: Session = Depends(get_db)) -> PredictionOut:
    prediction = db.get(Prediction, prediction_id)
    if prediction is None:
        raise HTTPException(status_code=404, detail=f"prediction {prediction_id!r} not found")
    return PredictionOut.model_validate(prediction)
