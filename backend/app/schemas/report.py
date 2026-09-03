from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict


class ReportCreateRequest(BaseModel):
    prediction_id: str


class ReportOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    prediction_id: str
    pdf_path: str
    created_at: datetime
