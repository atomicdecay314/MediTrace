import uuid
from datetime import datetime, timezone
from enum import Enum as PyEnum

from sqlalchemy import (
    Boolean, Date, DateTime, Enum, Float, ForeignKey,
    Index, JSON, String, Text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from database import Base


def _uuid() -> str:
    return str(uuid.uuid4())


def _now() -> datetime:
    return datetime.now(timezone.utc)


class SessionStatus(str, PyEnum):
    active = "active"
    interview_complete = "interview_complete"
    processing = "processing"
    timeline_generated = "timeline_generated"
    failed = "failed"


class Session(Base):
    __tablename__ = "sessions"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    patient_label: Mapped[str | None] = mapped_column(String, nullable=True)
    status: Mapped[str] = mapped_column(
        Enum(SessionStatus), default=SessionStatus.active, nullable=False
    )
    transcript: Mapped[list] = mapped_column(JSON, default=list)
    interview_state: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now, onupdate=_now
    )

    sources: Mapped[list["RawSource"]] = relationship("RawSource", back_populates="session")
    events: Mapped[list["Event"]] = relationship("Event", back_populates="session")
    conflicts: Mapped[list["Conflict"]] = relationship("Conflict", back_populates="session")


class RawSource(Base):
    __tablename__ = "raw_sources"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    session_id: Mapped[str] = mapped_column(String, ForeignKey("sessions.id"), nullable=False)
    kind: Mapped[str] = mapped_column(String, nullable=False)
    filename: Mapped[str | None] = mapped_column(String, nullable=True)
    storage_path: Mapped[str | None] = mapped_column(String, nullable=True)
    ocr_status: Mapped[str] = mapped_column(String, default="pending", nullable=False)
    extracted_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    extraction_meta: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    session: Mapped["Session"] = relationship("Session", back_populates="sources")
    events: Mapped[list["Event"]] = relationship("Event", back_populates="source")

    __table_args__ = (Index("ix_raw_sources_session_id", "session_id"),)


class Event(Base):
    __tablename__ = "events"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    session_id: Mapped[str] = mapped_column(String, ForeignKey("sessions.id"), nullable=False)
    source_id: Mapped[str | None] = mapped_column(String, ForeignKey("raw_sources.id"), nullable=True)
    event_type: Mapped[str] = mapped_column(String, nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    date_start: Mapped[datetime | None] = mapped_column(Date, nullable=True)
    date_end: Mapped[datetime | None] = mapped_column(Date, nullable=True)
    date_raw: Mapped[str] = mapped_column(String, default="", nullable=False)
    date_confidence: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    date_precision: Mapped[str] = mapped_column(String, default="", nullable=False)
    dedup_key: Mapped[str] = mapped_column(String, default="", nullable=False)
    cluster_id: Mapped[str | None] = mapped_column(String, nullable=True)
    is_canonical: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    structured: Mapped[dict] = mapped_column(JSON, default=dict)
    confidence: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    manually_edited: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    session: Mapped["Session"] = relationship("Session", back_populates="events")
    source: Mapped["RawSource | None"] = relationship("RawSource", back_populates="events")

    __table_args__ = (
        Index("ix_events_session_id", "session_id"),
        Index("ix_events_dedup_key", "dedup_key"),
    )


class Conflict(Base):
    __tablename__ = "conflicts"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    session_id: Mapped[str] = mapped_column(String, ForeignKey("sessions.id"), nullable=False)
    event_a_id: Mapped[str] = mapped_column(String, ForeignKey("events.id"), nullable=False)
    event_b_id: Mapped[str] = mapped_column(String, ForeignKey("events.id"), nullable=False)
    conflict_type: Mapped[str] = mapped_column(String, nullable=False)
    detail: Mapped[str] = mapped_column(Text, nullable=False)
    resolution: Mapped[str] = mapped_column(String, default="unresolved", nullable=False)

    session: Mapped["Session"] = relationship("Session", back_populates="conflicts")

    __table_args__ = (Index("ix_conflicts_session_id", "session_id"),)
