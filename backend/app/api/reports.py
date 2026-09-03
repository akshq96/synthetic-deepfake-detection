from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from backend.app.core.config import settings
from backend.app.db.base import get_db
from backend.app.models.prediction import Prediction
from backend.app.models.report import Report
from backend.app.schemas.report import ReportCreateRequest, ReportOut
from backend.app.services import storage
from backend.app.services.report_service import generate_report_pdf

router = APIRouter(prefix="/api/reports", tags=["reports"])


@router.post("/generate", response_model=ReportOut)
def generate_report(payload: ReportCreateRequest, db: Session = Depends(get_db)) -> ReportOut:
    prediction = db.get(Prediction, payload.prediction_id)
    if prediction is None:
        raise HTTPException(status_code=404, detail=f"prediction {payload.prediction_id!r} not found")

    output_path = settings.reports_dir / f"{prediction.id}.pdf"
    generate_report_pdf(prediction, output_path)

    report = Report(prediction_id=prediction.id, pdf_path=storage.relative_to_artifacts(output_path))
    db.add(report)
    db.commit()
    db.refresh(report)
    return ReportOut.model_validate(report)


@router.get("/{report_id}/export")
def export_report(report_id: str, db: Session = Depends(get_db)) -> FileResponse:
    report = db.get(Report, report_id)
    if report is None:
        raise HTTPException(status_code=404, detail=f"report {report_id!r} not found")

    path = storage.resolve_artifact_path(report.pdf_path)
    if not path.exists():
        raise HTTPException(status_code=404, detail="report PDF file is missing on disk")
    return FileResponse(path, media_type="application/pdf", filename=path.name)
