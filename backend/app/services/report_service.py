"""Renders a one-page forensic report PDF for a single prediction —
metrics/label summary plus the heatmap image, if one was saved. Uses
reportlab (pure-Python, no system-library dependency) rather than
weasyprint, which needs Cairo/Pango installed at the OS level.
"""

from __future__ import annotations

from pathlib import Path

from backend.app.core.config import settings
from backend.app.models.prediction import Prediction


def generate_report_pdf(prediction: Prediction, output_path: str | Path) -> Path:
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.units import inch
    from reportlab.pdfgen import canvas

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    c = canvas.Canvas(str(output_path), pagesize=letter)
    width, height = letter

    c.setFont("Helvetica-Bold", 16)
    c.drawString(1 * inch, height - 1 * inch, "Deepfake Detection Forensic Report")

    c.setFont("Helvetica", 10)
    y = height - 1.5 * inch
    fields = [
        ("Prediction ID", prediction.id),
        ("Input type", prediction.input_type),
        ("Model", prediction.model_name),
        ("Label", prediction.label),
        ("Confidence", f"{prediction.confidence:.4f}"),
        ("Fake probability", f"{prediction.fake_probability:.4f}"),
        ("Abstained", str(prediction.abstained)),
        ("Created at (UTC)", prediction.created_at.isoformat()),
    ]
    for name, value in fields:
        c.drawString(1 * inch, y, f"{name}: {value}")
        y -= 0.25 * inch

    y -= 0.25 * inch
    if prediction.heatmap_path:
        heatmap_path = settings.artifacts_root / prediction.heatmap_path
        if heatmap_path.exists():
            c.setFont("Helvetica-Bold", 11)
            c.drawString(1 * inch, y, "Explainability heatmap:")
            y -= 0.2 * inch
            image_size = 3 * inch
            try:
                c.drawImage(
                    str(heatmap_path),
                    1 * inch,
                    y - image_size,
                    width=image_size,
                    height=image_size,
                    preserveAspectRatio=True,
                )
            except Exception:
                # Best-effort: a corrupt/unreadable image shouldn't prevent
                # the rest of the report (the textual summary above) from
                # being generated and downloadable.
                c.setFont("Helvetica-Oblique", 9)
                c.drawString(1 * inch, y - 0.2 * inch, "(heatmap image could not be embedded)")

    c.showPage()
    c.save()
    return output_path
