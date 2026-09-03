"""Local-filesystem storage for uploads/heatmaps/reports. DB rows always
store paths relative to `settings.artifacts_root` (never absolute), so the
whole `artifacts/` tree stays portable — see the project plan's storage
design note. Swapping to S3-compatible storage later is a matter of
replacing this module's functions behind the same signatures, not touched
now (no speculative abstraction beyond that documented intent).
"""

from __future__ import annotations

import uuid
from pathlib import Path

from fastapi import UploadFile

from backend.app.core.config import settings

MAX_UPLOAD_SIZE_BYTES = settings.max_upload_size_bytes


class UploadTooLargeError(ValueError):
    pass


def _unique_filename(original_name: str) -> str:
    suffix = Path(original_name).suffix
    return f"{uuid.uuid4()}{suffix}"


async def save_upload(upload: UploadFile, *, subdir: str = "") -> Path:
    """Streams `upload` to disk under uploads_dir/<subdir>/, enforcing
    max_upload_size_bytes without ever buffering the whole file in memory.
    Returns the absolute path written.
    """
    dest_dir = settings.uploads_dir / subdir if subdir else settings.uploads_dir
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / _unique_filename(upload.filename or "upload")

    total_bytes = 0
    chunk_size = 1024 * 1024
    with open(dest_path, "wb") as f:
        while chunk := await upload.read(chunk_size):
            total_bytes += len(chunk)
            if total_bytes > MAX_UPLOAD_SIZE_BYTES:
                f.close()
                dest_path.unlink(missing_ok=True)
                raise UploadTooLargeError(
                    f"upload exceeds max size of {MAX_UPLOAD_SIZE_BYTES} bytes"
                )
            f.write(chunk)

    return dest_path


def relative_to_artifacts(path: str | Path) -> str:
    path = Path(path)
    try:
        return str(path.relative_to(settings.artifacts_root))
    except ValueError:
        return str(path)  # already relative, or outside artifacts_root (unusual, kept as-is)


def resolve_artifact_path(relative_path: str) -> Path:
    return settings.artifacts_root / relative_path
