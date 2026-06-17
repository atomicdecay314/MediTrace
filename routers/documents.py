import logging
import uuid
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy.orm import Session as DBSession

from config import settings
from database import SessionLocal, get_db
from models import RawSource, Session
from schemas import DocumentDetailOut, DocumentOut, DocumentUploadOut
from services.doc_pipeline import detect_kind, process_document

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/sessions", tags=["documents"])

_UPLOAD_STATUSES = {"active", "interview_complete"}


def _ocr_task(source_id: str, content: bytes, mime: str, kind: str) -> None:
    db = SessionLocal()
    source = None
    try:
        source = db.get(RawSource, source_id)
        if not source:
            return
        process_document(source, content, mime, kind)
        db.commit()
    except Exception as exc:
        logger.error("OCR task failed for source %s: %s", source_id, exc)
        try:
            if source:
                source.ocr_status = "error"
                source.extraction_meta = {"error": str(exc), "warnings": []}
                db.commit()
        except Exception:
            pass
    finally:
        db.close()


@router.post("/{session_id}/documents", status_code=202, response_model=DocumentUploadOut)
async def upload_document(
    session_id: str,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    kind: str | None = Form(None),
    db: DBSession = Depends(get_db),
):
    session = db.get(Session, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    if session.status not in _UPLOAD_STATUSES:
        raise HTTPException(status_code=409, detail="Session is not accepting documents")

    content = await file.read()
    mime = file.content_type or "application/octet-stream"
    filename = file.filename or "upload"

    detected_kind = kind or detect_kind(filename, mime, content[:16])

    source_id = str(uuid.uuid4())
    suffix = Path(filename).suffix or ""
    storage_path = str(Path(settings.UPLOAD_DIR) / f"{source_id}{suffix}")
    Path(storage_path).write_bytes(content)

    source = RawSource(
        id=source_id,
        session_id=session_id,
        kind=detected_kind,
        filename=filename,
        storage_path=storage_path,
        ocr_status="pending",
        extraction_meta={},
    )
    db.add(source)
    db.commit()

    background_tasks.add_task(_ocr_task, source_id, content, mime, detected_kind)

    return DocumentUploadOut(source_id=source_id, kind=detected_kind, ocr_status="pending")


@router.get("/{session_id}/documents", response_model=list[DocumentOut])
def list_documents(session_id: str, db: DBSession = Depends(get_db)):
    if not db.get(Session, session_id):
        raise HTTPException(status_code=404, detail="Session not found")

    sources = (
        db.query(RawSource)
        .filter(RawSource.session_id == session_id)
        .order_by(RawSource.created_at)
        .all()
    )
    return [
        DocumentOut(
            source_id=s.id,
            filename=s.filename,
            kind=s.kind,
            ocr_status=s.ocr_status,
            warnings=(s.extraction_meta or {}).get("warnings") or [],
        )
        for s in sources
    ]


@router.get("/{session_id}/documents/{source_id}", response_model=DocumentDetailOut)
def get_document(session_id: str, source_id: str, db: DBSession = Depends(get_db)):
    if not db.get(Session, session_id):
        raise HTTPException(status_code=404, detail="Session not found")

    source = db.get(RawSource, source_id)
    if not source or source.session_id != session_id:
        raise HTTPException(status_code=404, detail="Document not found")

    return DocumentDetailOut(
        source_id=source.id,
        filename=source.filename,
        kind=source.kind,
        ocr_status=source.ocr_status,
        extracted_text=source.extracted_text,
        extraction_meta=source.extraction_meta or {},
    )
