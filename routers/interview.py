import logging
import uuid

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy.orm import Session as DBSession

from database import get_db
from models import Event, Session
from schemas import ExtractionRetryIn, InterviewTurnIn
from services.extractor import ExtractionTransientError, extract_single_turn
from services.gemini_client import GeminiParseError
from services.interviewer import init_state, run_turn
from services.whisper_client import WhisperError, whisper_client

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/sessions", tags=["interview"])


def _run_per_turn_extraction(
    message: str, session_id: str, db: DBSession
) -> tuple[bool, int]:
    """Extract events from a single patient message and persist them.

    Returns (extraction_ok, event_count).
    Never raises — failures are signalled via extraction_ok=False so the
    conversation reply is always returned to the client.
    """
    if not message or not message.strip():
        return True, 0   # empty message (e.g. initial greeting) — not a failure
    try:
        events = extract_single_turn(message, session_id)
        for e in events:
            db.add(Event(id=str(uuid.uuid4()), **e))
        if events:
            db.commit()
        return True, len(events)
    except ExtractionTransientError:
        logger.warning(
            "Per-turn extraction failed after retries for session %s", session_id
        )
        return False, 0
    except Exception as exc:
        logger.error(
            "Unexpected per-turn extraction error for session %s: %s", session_id, exc
        )
        return False, 0


@router.post("/{session_id}/interview/turn")
def interview_turn(
    session_id: str, payload: InterviewTurnIn, db: DBSession = Depends(get_db)
):
    session = db.get(Session, session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    if session.status not in ("active", "interview_complete"):
        raise HTTPException(status_code=409, detail="Session is not accepting interview turns")

    if not session.interview_state:
        session.interview_state = init_state()

    try:
        result = run_turn(session, payload.message)
    except GeminiParseError as exc:
        logger.error("Gemini parse error in interview turn: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))
    except RuntimeError as exc:
        logger.error("Runtime error in interview turn: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))

    db.add(session)
    db.commit()

    # Per-turn extraction — runs AFTER the turn is committed so the conversation
    # is always saved even if extraction fails.
    extraction_ok, extraction_count = _run_per_turn_extraction(
        payload.message, session_id, db
    )

    return {
        "reply": result.reply,
        "coverage": result.updated_coverage,
        "interview_complete": result.interview_complete,
        "follow_up_reason": result.follow_up_reason,
        "extraction_ok": extraction_ok,
        "extraction_count": extraction_count,
    }


@router.post("/{session_id}/interview/voice-turn")
async def voice_turn(
    session_id: str,
    audio: UploadFile = File(...),
    db: DBSession = Depends(get_db),
):
    session = db.get(Session, session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    if session.status not in ("active", "interview_complete"):
        raise HTTPException(status_code=409, detail="Session is not accepting interview turns")

    audio_bytes = await audio.read()
    mime_type = audio.content_type or "audio/webm"

    try:
        transcript_text = whisper_client.transcribe(audio_bytes, mime_type)
    except WhisperError as exc:
        logger.error("Whisper error in voice turn: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))

    if not transcript_text.strip():
        raise HTTPException(
            status_code=422, detail="Audio contained no speech — please try again."
        )

    if not session.interview_state:
        session.interview_state = init_state()

    try:
        result = run_turn(session, transcript_text)
    except GeminiParseError as exc:
        logger.error("Gemini parse error in voice turn: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))
    except RuntimeError as exc:
        logger.error("Runtime error in voice turn: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))

    db.add(session)
    db.commit()

    extraction_ok, extraction_count = _run_per_turn_extraction(
        transcript_text, session_id, db
    )

    return {
        "transcript": transcript_text,
        "reply": result.reply,
        "coverage": result.updated_coverage,
        "interview_complete": result.interview_complete,
        "follow_up_reason": result.follow_up_reason,
        "extraction_ok": extraction_ok,
        "extraction_count": extraction_count,
    }


@router.post("/{session_id}/interview/retry-extraction")
def retry_extraction(
    session_id: str, payload: ExtractionRetryIn, db: DBSession = Depends(get_db)
):
    """Re-run per-turn extraction for a specific patient message that previously failed.
    Always returns 200 so the frontend can distinguish retry-failed from server-error.
    """
    session = db.get(Session, session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    extraction_ok, extraction_count = _run_per_turn_extraction(
        payload.message, session_id, db
    )
    return {"ok": extraction_ok, "count": extraction_count}


@router.post("/{session_id}/interview/complete")
def complete_interview(session_id: str, db: DBSession = Depends(get_db)):
    session = db.get(Session, session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    if session.status != "active":
        raise HTTPException(
            status_code=409, detail="Interview is already past active state"
        )

    session.status = "interview_complete"
    db.add(session)
    db.commit()

    return {"status": session.status}
