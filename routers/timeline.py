import logging
import uuid
from collections import defaultdict

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from sqlalchemy.orm import Session as DBSession

from database import SessionLocal, get_db
from models import Conflict, Event, RawSource, Session
from schemas import (
    CanonicalEventOut, ConflictOut, ConflictResolveIn,
    EventOut, EventPatchIn, EventPatchOut, ExtractOut, TimelineOut,
)
from services import extractor, fusion

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/sessions", tags=["timeline"])

# Allow re-extraction from any non-processing state so fixes can be re-run.
# "processing" is the only hard block (a fusion run is in-flight).
_EXTRACT_STATUSES = {"active", "interview_complete", "timeline_generated", "failed"}
_GENERATE_STATUSES = {"active", "interview_complete", "processing", "failed", "timeline_generated"}


# ── Phase 4: extract + list raw events (unchanged) ───────────────────────────

@router.post("/{session_id}/events/extract", response_model=ExtractOut)
def extract_events(session_id: str, db: DBSession = Depends(get_db)):
    session = db.get(Session, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    if session.status not in _EXTRACT_STATUSES:
        raise HTTPException(status_code=409, detail="Session is not in an extractable state")

    # Note whether we need to reset fusion state, but do NOT flush yet.
    # Flushing before extraction expired the session object, making
    # session.transcript reload unreliably on the second call (SQLite
    # mid-transaction reload). With autoflush=False we hold all writes
    # until after extraction is complete.
    reset_from_generated = session.status == "timeline_generated"
    if reset_from_generated:
        session.status = "interview_complete"

    # ── GATHER ALL NEW EVENTS FIRST (no DB writes yet) ─────────────────
    new_events: list[dict] = []
    counts: dict = {"interview": 0, "documents": {}}

    # Interview — session.transcript is read from the original db.get() load;
    # no flush has occurred so the object is not expired.
    try:
        interview_events = extractor.extract_from_interview(session)
        new_events.extend(interview_events)
        counts["interview"] = len(interview_events)
        if not interview_events:
            n_turns = len(session.transcript or [])
            logger.warning(
                "extract_from_interview returned 0 events for session %s "
                "(transcript has %d turns) — check LLM response", session_id, n_turns
            )
    except Exception as exc:
        logger.error("Interview extraction failed for session %s: %s", session_id, exc)

    # Documents
    sources = (
        db.query(RawSource)
        .filter(RawSource.session_id == session_id, RawSource.ocr_status == "done")
        .all()
    )
    for source in sources:
        try:
            doc_events = extractor.extract_from_document(source)
            new_events.extend(doc_events)
            counts["documents"][source.id] = len(doc_events)
        except Exception as exc:
            logger.error("Document extraction failed for source %s: %s", source.id, exc)
            counts["documents"][source.id] = 0

    # ── ATOMIC REPLACE: clear stale data, insert fresh events ──────────
    # Only delete UNRESOLVED conflicts — resolved conflicts survive re-extraction.
    if reset_from_generated:
        db.query(Conflict).filter(
            Conflict.session_id == session_id,
            Conflict.resolution == "unresolved",
        ).delete(synchronize_session=False)

    # Preserve manually_edited events — they survive re-extraction untouched
    # and will be re-clustered with the fresh events during re-fusion.
    db.query(Event).filter(
        Event.session_id == session_id,
        Event.manually_edited == False,  # noqa: E712
    ).delete(synchronize_session=False)

    for e in new_events:
        db.add(Event(id=str(uuid.uuid4()), **e))

    db.commit()
    total = counts["interview"] + sum(counts["documents"].values())
    return ExtractOut(counts=counts, total=total)


@router.patch("/{session_id}/events/{event_id}", response_model=EventPatchOut)
def patch_event(
    session_id: str,
    event_id: str,
    payload: EventPatchIn,
    db: DBSession = Depends(get_db),
):
    """
    Partially update a single event. Sets manually_edited=True so that
    re-extraction keeps the event and re-fusion does not overwrite the
    user's changes.

    Fusion-owned fields (dedup_key, cluster_id, is_canonical, event_type,
    source_id) are not editable via this endpoint.

    Canonical-edit scope: the edit sticks to this event only, not to other
    cluster members. Cluster members are raw source events; editing the
    canonical corrects the merged view without corrupting sources.
    """
    event = db.get(Event, event_id)
    if not event or event.session_id != session_id:
        raise HTTPException(status_code=404, detail="Event not found")

    # Apply only fields the user explicitly included in the request body
    fs = payload.model_fields_set
    if "description" in fs and payload.description is not None:
        event.description = payload.description
    if "date_start" in fs:
        event.date_start = payload.date_start
    if "date_end" in fs:
        event.date_end = payload.date_end
    if "date_raw" in fs and payload.date_raw is not None:
        event.date_raw = payload.date_raw
    if "date_precision" in fs and payload.date_precision is not None:
        event.date_precision = payload.date_precision
    if "date_confidence" in fs and payload.date_confidence is not None:
        event.date_confidence = payload.date_confidence
    if "confidence" in fs and payload.confidence is not None:
        event.confidence = payload.confidence
    if "structured" in fs and payload.structured is not None:
        # Merge rather than replace so existing keys (is_negation etc.) aren't lost
        event.structured = {**(event.structured or {}), **payload.structured}

    event.manually_edited = True
    db.commit()
    return EventPatchOut.model_validate(event)


@router.get("/{session_id}/events", response_model=list[EventOut])
def list_events(session_id: str, db: DBSession = Depends(get_db)):
    if not db.get(Session, session_id):
        raise HTTPException(status_code=404, detail="Session not found")
    events = db.query(Event).filter(Event.session_id == session_id).all()
    events.sort(key=lambda e: (e.date_start is None, e.date_start, -e.date_confidence))
    return [EventOut.model_validate(e) for e in events]


# ── Phase 5: timeline generate / get / conflict resolve ───────────────────────

def _fusion_task(session_id: str) -> None:
    """Background task — runs in its own DB session; single transaction."""
    db = SessionLocal()
    try:
        fusion.run_fusion(session_id, db)
        session = db.get(Session, session_id)
        if session:
            session.status = "timeline_generated"
        db.commit()
        logger.info("Fusion completed for session %s", session_id)
    except Exception as exc:
        logger.error("Fusion failed for session %s: %s", session_id, exc)
        db.rollback()
        # Mark failed in a fresh attempt
        try:
            session = db.get(Session, session_id)
            if session:
                session.status = "failed"
                db.commit()
        except Exception:
            db.rollback()
    finally:
        db.close()


@router.post("/{session_id}/timeline/generate")
async def generate_timeline(
    session_id: str,
    background_tasks: BackgroundTasks,
    db: DBSession = Depends(get_db),
):
    """
    Phase 6B guard decisions:

    P1 — Interview-only (no docs): VALID. A voice/text intake alone is legitimate.
    P2 — Docs-only (no interview): VALID. extract_from_interview returns [] cleanly.
    P1+P2 — Truly empty (no interview turns AND no done docs): 422 — not a 500 or
         silent empty timeline.
    P3 — Pending OCR: 409 naming the pending files. "Pending" = ocr_status=='pending'
         on any RawSource. Error-status sources are excluded (they just don't
         contribute to the timeline).
    P4 — Retryability: fusion runs in its own transaction; on failure it rolls back
         all cluster/conflict writes and sets status='failed'. Events from the last
         successful extraction are preserved. The caller can safely retry generate.
         A status=='processing' guard prevents duplicate concurrent runs.
    """
    session = db.get(Session, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    # P4: deduplicate concurrent generate requests
    if session.status == "processing":
        raise HTTPException(
            status_code=409,
            detail="Timeline generation is already in progress — wait for it to finish",
        )

    # P3: pending OCR — generating from partial data would produce a misleading result
    pending_sources = (
        db.query(RawSource)
        .filter(
            RawSource.session_id == session_id,
            RawSource.ocr_status == "pending",
        )
        .all()
    )
    if pending_sources:
        raise HTTPException(
            status_code=409,
            detail={
                "error": "Documents are still being processed — wait for OCR to complete",
                "pending_sources": [
                    {"id": s.id, "filename": s.filename or "(unnamed)"}
                    for s in pending_sources
                ],
            },
        )

    # P1+P2: distinguish "truly empty session" (422) from "has content but not
    # yet extracted" (409).  A session has content if it has at least one
    # patient-turn in the transcript OR at least one successfully OCR'd document.
    has_interview_turns = any(
        t.get("role") == "user" for t in (session.transcript or [])
    )
    done_doc_count = (
        db.query(RawSource)
        .filter(
            RawSource.session_id == session_id,
            RawSource.ocr_status == "done",
        )
        .count()
    )
    if not has_interview_turns and done_doc_count == 0:
        raise HTTPException(
            status_code=422,
            detail=(
                "Nothing to generate from: no interview has been conducted "
                "and no documents have been successfully processed."
            ),
        )

    # Content exists but /events/extract has not been run yet (or produced nothing)
    event_count = db.query(Event).filter(Event.session_id == session_id).count()
    if event_count == 0:
        raise HTTPException(
            status_code=409,
            detail=(
                "No events found — run POST /events/extract first "
                "to extract events from the interview and/or documents."
            ),
        )

    session.status = "processing"
    db.commit()

    background_tasks.add_task(_fusion_task, session_id)
    return {"status": "processing"}


@router.get("/{session_id}/timeline", response_model=TimelineOut)
def get_timeline(session_id: str, db: DBSession = Depends(get_db)):
    session = db.get(Session, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    # Canonical events, sorted
    canonical = (
        db.query(Event)
        .filter(Event.session_id == session_id, Event.is_canonical == True)
        .all()
    )
    canonical.sort(key=lambda e: (e.date_start is None, e.date_start, -e.date_confidence))

    # Cluster stats in one pass
    all_events = db.query(Event).filter(Event.session_id == session_id).all()
    cluster_sizes: dict[str, int] = defaultdict(int)
    cluster_srcs: dict[str, set[str]] = defaultdict(set)
    for e in all_events:
        if e.cluster_id:
            cluster_sizes[e.cluster_id] += 1
            if e.source_id:
                cluster_srcs[e.cluster_id].add(e.source_id)

    event_outs = []
    for e in canonical:
        cid = e.cluster_id or ''
        event_outs.append(CanonicalEventOut(
            id=e.id,
            event_type=e.event_type,
            description=e.description,
            date_start=e.date_start,
            date_end=e.date_end,
            date_raw=e.date_raw or '',
            date_precision=e.date_precision or '',
            date_confidence=e.date_confidence,
            confidence=e.confidence,
            cluster_id=e.cluster_id,
            cluster_size=cluster_sizes.get(cid, 1),
            source_ids=list(cluster_srcs.get(cid, set())),
            structured=e.structured or {},
            is_negation=bool((e.structured or {}).get('is_negation', False)),
            manually_edited=e.manually_edited,
        ))

    conflicts = (
        db.query(Conflict)
        .filter(Conflict.session_id == session_id)
        .all()
    )
    conflict_outs = [
        ConflictOut(
            id=c.id,
            conflict_type=c.conflict_type,
            detail=c.detail,
            event_a_id=c.event_a_id,
            event_b_id=c.event_b_id,
            resolution=c.resolution,
        )
        for c in conflicts
    ]

    return TimelineOut(
        status=session.status,
        events=event_outs,
        conflicts=conflict_outs,
    )


@router.patch("/{session_id}/conflicts/{conflict_id}")
def resolve_conflict(
    session_id: str,
    conflict_id: str,
    payload: ConflictResolveIn,
    db: DBSession = Depends(get_db),
):
    """
    Resolve a conflict. Optionally declare which event wins as canonical.

    canonical_choice="a"|"b": sets is_canonical on the winner and marks it
    manually_edited=True so re-fusion preserves the user's canonical choice.

    Resolved conflicts are NOT deleted on re-fusion (run_fusion only deletes
    resolution='unresolved' rows). The dedup check in _apply_cluster prevents
    them from being re-created as unresolved. Resolution is permanent.
    """
    if payload.resolution not in ("a_wins", "b_wins", "both_noted"):
        raise HTTPException(status_code=422, detail="resolution must be a_wins, b_wins, or both_noted")
    conflict = db.get(Conflict, conflict_id)
    if not conflict or conflict.session_id != session_id:
        raise HTTPException(status_code=404, detail="Conflict not found")

    conflict.resolution = payload.resolution

    # Apply canonical choice when provided
    if payload.canonical_choice in ("a", "b"):
        winner_id = conflict.event_a_id if payload.canonical_choice == "a" else conflict.event_b_id
        loser_id  = conflict.event_b_id if payload.canonical_choice == "a" else conflict.event_a_id
        winner = db.get(Event, winner_id)
        loser  = db.get(Event, loser_id)
        if winner and loser:
            if winner.cluster_id and winner.cluster_id == loser.cluster_id:
                # Same cluster: flip canonical within the cluster
                winner.is_canonical = True
                loser.is_canonical  = False
            # Mark winner manually_edited so re-fusion preserves it as canonical
            winner.manually_edited = True

    db.commit()
    return {
        "id": conflict.id,
        "resolution": conflict.resolution,
        "canonical_choice": payload.canonical_choice,
    }
