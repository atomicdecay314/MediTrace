import logging
import uuid

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session as DBSession

from database import get_db
from models import Event, RawSource, Session
from schemas import EventOut, ExtractOut
from services import extractor

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/sessions", tags=["timeline"])

_EXTRACT_STATUSES = {"active", "interview_complete"}


@router.post("/{session_id}/events/extract", response_model=ExtractOut)
def extract_events(session_id: str, db: DBSession = Depends(get_db)):
    session = db.get(Session, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    if session.status not in _EXTRACT_STATUSES:
        raise HTTPException(status_code=409, detail="Session is not in an extractable state")

    # Idempotent: delete all prior events for this session before re-extracting
    # PHASE 5: preserve manually curated events; for now everything is auto-extracted
    db.query(Event).filter(Event.session_id == session_id).delete(
        synchronize_session=False
    )
    db.flush()

    counts: dict = {"interview": 0, "documents": {}}

    # Interview extraction
    try:
        interview_events = extractor.extract_from_interview(session)
        for e in interview_events:
            db.add(Event(id=str(uuid.uuid4()), **e))
        counts["interview"] = len(interview_events)
    except Exception as exc:
        logger.error("Interview extraction failed for session %s: %s", session_id, exc)

    # Document extraction (only sources with ocr_status="done")
    sources = (
        db.query(RawSource)
        .filter(RawSource.session_id == session_id, RawSource.ocr_status == "done")
        .all()
    )
    for source in sources:
        try:
            doc_events = extractor.extract_from_document(source)
            for e in doc_events:
                db.add(Event(id=str(uuid.uuid4()), **e))
            counts["documents"][source.id] = len(doc_events)
        except Exception as exc:
            logger.error("Document extraction failed for source %s: %s", source.id, exc)
            counts["documents"][source.id] = 0

    db.commit()

    total = counts["interview"] + sum(counts["documents"].values())
    return ExtractOut(counts=counts, total=total)


@router.get("/{session_id}/events", response_model=list[EventOut])
def list_events(session_id: str, db: DBSession = Depends(get_db)):
    if not db.get(Session, session_id):
        raise HTTPException(status_code=404, detail="Session not found")

    events = (
        db.query(Event)
        .filter(Event.session_id == session_id)
        .all()
    )
    # Sort: known dates first (ascending), then unknowns; tiebreak by confidence desc
    events.sort(key=lambda e: (e.date_start is None, e.date_start, -e.date_confidence))
    return [EventOut.model_validate(e) for e in events]
