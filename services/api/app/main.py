import os
from contextlib import asynccontextmanager
from typing import Any, Dict

import anyio
from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile

from services.api.app.config import MAX_UPLOAD_BYTES, ensure_upload_dir_exists, is_allowed_extension
from services.api.app.database import create_db_and_tables
from services.api.app.db import get_session
from services.api.app.detector import Detector, DetectorInput, DummyDetector, OnnxAudioDetector
from services.api.app.media import media_type_from_filename
from services.api.app.repository import (
    count_analyses,
    create_analysis_record,
    get_analysis_by_id,
    get_analysis_record_by_id,
    list_analyses,
    update_analysis_record,
)
from services.api.app.schemas import AnalysisRecordOut, AnalysisReportOut
from services.api.app.auth import ROLE_ANALYST, User
from services.api.app.utils import sanitize_filename, stream_save_and_sha256

# A static version for now, can be replaced with git hash later
APP_VERSION = "0.1.0"

ONNX_MODEL_PATH = os.environ.get("ONNX_MODEL_PATH", None)
REQUIRE_MODEL = os.environ.get("REQUIRE_MODEL", "false").lower() in {"true", "1", "yes"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Runs once at application startup
    create_db_and_tables()
    ensure_upload_dir_exists()

    # Validate model if required
    if REQUIRE_MODEL and not ONNX_MODEL_PATH:
        raise RuntimeError("REQUIRE_MODEL=true but ONNX_MODEL_PATH is not set")
    if ONNX_MODEL_PATH:
        from pathlib import Path

        if not Path(ONNX_MODEL_PATH).exists():
            raise FileNotFoundError(f"ONNX model not found: {ONNX_MODEL_PATH}")

    yield
    # Runs once at application shutdown (nothing to do yet)


app = FastAPI(
    title="Aegis-AI API",
    version=APP_VERSION,
    description="API for deepfake detection and analysis.",
    lifespan=lifespan,
)


def get_detector() -> Detector:
    if ONNX_MODEL_PATH:
        return OnnxAudioDetector(model_path=ONNX_MODEL_PATH)
    return DummyDetector()


DETECTOR_TIMEOUT_SECONDS = float(os.environ.get("DETECTOR_TIMEOUT_SECONDS", "2.0"))


@app.middleware("http")
async def attach_request_user(request: Request, call_next):
    """Attach a request-scoped User to request.state.user.

    Header format: X-User: user_id:role1,role2
    Example: X-User: alice:ADMIN,ANALYST

    If missing, defaults to dev user: dev-user with role ANALYST.
    """

    header = request.headers.get("X-User")
    if not header:
        request.state.user = User(id="dev-user", roles=[ROLE_ANALYST])
    else:
        user_id = header
        roles = []
        if ":" in header:
            user_id, roles_part = header.split(":", 1)
            roles = [r.strip() for r in roles_part.split(",") if r.strip()]
        request.state.user = User(id=user_id.strip() or "dev-user", roles=roles)

    return await call_next(request)


@app.get("/health")
def health_check():
    """Simple health check endpoint."""
    return {"status": "ok"}


@app.get("/__debug/user")
async def debug_user(request: Request):
    """Debug endpoint to inspect request-scoped user (dev only)."""
    u = getattr(request.state, "user", None)
    return {"id": getattr(u, "id", None), "roles": getattr(u, "roles", None)}


@app.get("/version")
def version():
    """Returns the application version."""
    return {"version": APP_VERSION}


@app.get("/v1/models")
def list_models():
    """Return currently loaded model info."""

    if ONNX_MODEL_PATH:
        from pathlib import Path

        model_path = Path(ONNX_MODEL_PATH)
        # Extract version from filename (e.g., V1.0.0.onnx -> V1.0.0)
        stem = model_path.stem  # e.g., "latest" or "V1.0.0"
        current_version = stem if stem.startswith("V") else "unknown"

        return {
            "audio": {
                "current": current_version,
                "path": str(model_path),
            }
        }
    else:
        return {
            "audio": {
                "current": "DummyDetector",
                "path": None,
            }
        }


@app.get("/v1/analysis")
def list_analysis(limit: int = 20, offset: int = 0):
    if limit < 1 or limit > 200:
        raise HTTPException(status_code=400, detail="invalid limit")
    if offset < 0:
        raise HTTPException(status_code=400, detail="invalid offset")

    with get_session() as session:
        total = count_analyses(session)
        rows = list_analyses(session, limit=limit, offset=offset)

    items = [
        AnalysisRecordOut(
            id=str(r.id),
            created_at=r.created_at,
            media_type=r.media_type,
            filename=r.filename,
            sha256=r.sha256,
            duration_s=r.duration_s,
            verdict=r.verdict,
            confidence=r.confidence,
            explanations=r.explanations,
            signals=r.signals,
            model_version=r.model_version,
            processing_ms=r.processing_ms,
        )
        for r in rows
    ]

    return {"items": items, "limit": limit, "offset": offset, "total": total}


@app.post("/v1/analysis/upload")
async def upload(file: UploadFile | None = File(None)):
    """Upload a media file for analysis.

    Validates extension and max size, streams to disk while hashing, then creates
    an AnalysisRecord with an initial SUSPICIOUS verdict.
    """

    # Missing file is handled by FastAPI validation, but keep explicit guard.
    if file is None:
        raise HTTPException(status_code=400, detail="missing file")

    original_name = file.filename or ""
    safe_name = sanitize_filename(original_name)

    if not is_allowed_extension(safe_name):
        raise HTTPException(status_code=415, detail="extension not allowed")

    # Ensure upload directory exists (startup should handle this, but be defensive)
    upload_dir = ensure_upload_dir_exists()

    # Save under uploads/ with a temp-ish name; actual uniqueness handled by sha.
    dest_path = upload_dir / safe_name

    # Stream file to disk while hashing
    result = stream_save_and_sha256(fileobj=file.file, destination_path=dest_path)

    if result.size_bytes > MAX_UPLOAD_BYTES:
        # Remove oversized file to avoid leaving junk on disk
        try:
            dest_path.unlink(missing_ok=True)  # py3.8+: missing_ok supported
        except TypeError:
            # for older python compatibility
            if dest_path.exists():
                dest_path.unlink()
        raise HTTPException(status_code=413, detail="file too large")

    media_type = media_type_from_filename(safe_name)

    with get_session() as session:
        rec = create_analysis_record(
            session,
            media_type=media_type,
            filename=safe_name,
            sha256=result.sha256,
            duration_s=None,
            verdict="SUSPICIOUS",
            confidence=0.0,
        )

    return {"analysis_id": str(rec.id), "sha256": result.sha256}


@app.get("/v1/analysis/{analysis_id}")
def get_analysis(analysis_id: str) -> AnalysisRecordOut:
    with get_session() as session:
        rec = get_analysis_record_by_id(session, analysis_id)
        if rec is None:
            raise HTTPException(status_code=404, detail="analysis not found")

        return AnalysisRecordOut(
            id=str(rec.id),
            created_at=rec.created_at,
            media_type=rec.media_type,
            filename=rec.filename,
            sha256=rec.sha256,
            duration_s=rec.duration_s,
            verdict=rec.verdict,
            confidence=rec.confidence,
            explanations=rec.explanations,
            signals=rec.signals,
            model_version=rec.model_version,
            processing_ms=rec.processing_ms,
        )


@app.get("/v1/analysis/{analysis_id}/report")
def get_analysis_report(analysis_id: str) -> AnalysisReportOut:
    with get_session() as session:
        rec = get_analysis_record_by_id(session, analysis_id)
        if rec is None:
            raise HTTPException(status_code=404, detail="analysis not found")

    from services.api.app.reporting import build_report

    return build_report(rec)


@app.post("/v1/analysis/{analysis_id}/run")
async def run_analysis(analysis_id: str, detector: Detector = Depends(get_detector)):
    """Run detector inference for an existing uploaded analysis record."""

    with get_session() as session:
        rec = get_analysis_record_by_id(session, analysis_id)
        if rec is None:
            raise HTTPException(status_code=404, detail="analysis not found")

        # Build detector input
        file_path = str(ensure_upload_dir_exists() / rec.filename)
        metadata: Dict[str, Any] = {
            "analysis_id": str(rec.id),
            "media_type": rec.media_type,
            "filename": rec.filename,
            "sha256": rec.sha256,
            "created_at": rec.created_at.isoformat(),
        }

    det_in = DetectorInput(file_path=file_path, metadata=metadata)

    # Run detector in a thread with timeout (DummyDetector is sync)
    try:
        with anyio.fail_after(DETECTOR_TIMEOUT_SECONDS):
            det_out = await anyio.to_thread.run_sync(detector.run, det_in, abandon_on_cancel=True)
    except TimeoutError:
        # Persist failure state for auditability
        with get_session() as session:
            rec = get_analysis_record_by_id(session, analysis_id)
            if rec is not None:
                rec.verdict = "FAILED"
                rec.confidence = 0.0
                rec.explanations = ["Detector timeout"]
                rec.signals = {"error": "timeout"}
                rec.model_version = "timeout"
                update_analysis_record(session, rec)

        raise HTTPException(status_code=504, detail="detector timeout")

    # Persist detector output
    with get_session() as session:
        rec = get_analysis_record_by_id(session, analysis_id)
        if rec is None:
            raise HTTPException(status_code=404, detail="analysis not found")

        rec.verdict = det_out.verdict
        rec.confidence = float(det_out.confidence)
        rec.explanations = det_out.explanations
        rec.signals = det_out.signals
        rec.model_version = det_out.model_version

        rec = update_analysis_record(session, rec)

    return {
        "analysis_id": str(rec.id),
        "verdict": rec.verdict,
        "confidence": rec.confidence,
        "model_version": rec.model_version,
    }
