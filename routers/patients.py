from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session as DBSession

from database import get_db
from models import Patient, Session
from schemas import PatientOut

router = APIRouter(prefix="/api/patients", tags=["patients"])


@router.get("", response_model=list[PatientOut])
def list_patients(db: DBSession = Depends(get_db)):
    """Return all patients ordered newest-first, each with their most-recent session id."""
    patients = db.query(Patient).order_by(Patient.created_at.desc()).all()
    result: list[PatientOut] = []
    for p in patients:
        latest = (
            db.query(Session)
            .filter(Session.patient_id == p.id)
            .order_by(Session.created_at.desc())
            .first()
        )
        result.append(
            PatientOut(
                id=p.id,
                name=p.name,
                age=p.age,
                sex=p.sex,
                created_at=p.created_at.isoformat(),
                latest_session_id=latest.id if latest else None,
            )
        )
    return result
