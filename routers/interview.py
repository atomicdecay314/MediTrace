import logging

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session as DBSession

from database import get_db
from models import Session
from schemas import InterviewTurnIn
from services.gemini_client import GeminiParseError
from services.interviewer import init_state, run_turn

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/sessions", tags=["interview"])


@router.post("/{session_id}/interview/turn")
def interview_turn(
    session_id: str, payload: InterviewTurnIn, db: DBSession = Depends(get_db)
):
    session = db.get(Session, session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    if session.status != "active":
        raise HTTPException(status_code=409, detail="Session is not active")

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

    return {
        "reply": result.reply,
        "coverage": result.updated_coverage,
        "interview_complete": result.interview_complete,
        "follow_up_reason": result.follow_up_reason,
    }


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
