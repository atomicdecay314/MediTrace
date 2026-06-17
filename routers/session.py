import uuid

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import func, select
from sqlalchemy.orm import Session as DBSession

from database import get_db
from models import Conflict, Event, RawSource, Session
from schemas import SessionCreate, SessionOut

router = APIRouter(prefix="/api/sessions", tags=["sessions"])


def _build_out(session: Session, db: DBSession) -> SessionOut:
    source_count = db.scalar(
        select(func.count()).where(RawSource.session_id == session.id)
    ) or 0
    event_count = db.scalar(
        select(func.count()).where(Event.session_id == session.id)
    ) or 0
    conflict_count = db.scalar(
        select(func.count()).where(Conflict.session_id == session.id)
    ) or 0
    return SessionOut(
        id=session.id,
        status=session.status,
        patient_label=session.patient_label,
        interview_state=session.interview_state or {},
        transcript=session.transcript or [],
        counts={"sources": source_count, "events": event_count, "conflicts": conflict_count},
    )


@router.post("", status_code=201, response_model=SessionOut)
def create_session(payload: SessionCreate, db: DBSession = Depends(get_db)):
    session = Session(
        id=str(uuid.uuid4()),
        patient_label=payload.patient_label,
        status="active",
        transcript=[],
        interview_state={},
    )
    db.add(session)
    db.commit()
    db.refresh(session)
    return _build_out(session, db)


@router.get("/{session_id}", response_model=SessionOut)
def get_session(session_id: str, db: DBSession = Depends(get_db)):
    session = db.get(Session, session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return _build_out(session, db)
