from __future__ import annotations

import warnings

from pydantic import BaseModel, Field, field_validator

# PHASE 2 (complete): InterviewTurnIn, InterviewTurnOut
# PHASE 3 (complete): DocumentUploadOut, DocumentOut, DocumentDetailOut
# PHASE 4 (complete): EventOut, ExtractOut
# PHASE 5 (complete): CanonicalEventOut, ConflictOut, TimelineOut, ConflictResolveIn
# PHASE 6A (complete): EventPatchIn, EventPatchOut — manual event edits
# PHASE 7B (complete): ProvenanceMemberOut

_COVERAGE_TOPICS = [
    "chief_complaint",
    "chronic_conditions",
    "medications",
    "surgeries",
    "hospitalizations",
    "allergies",
    "family_history",
]


# ── Phase 8: Patient system ────────────────────────────────────────────────────

class PatientCreate(BaseModel):
    name: str
    age: int | None = None
    sex: str | None = None     # free string — "M" / "F" / "Other" / custom


class PatientOut(BaseModel):
    id: str
    name: str
    age: int | None
    sex: str | None
    created_at: str            # ISO-8601 UTC
    latest_session_id: str | None = None


class SessionCreate(BaseModel):
    patient_label: str | None = None   # legacy field; kept for backwards compat
    # Phase 8: patient identity fields (when provided, a Patient row is created)
    name: str | None = None
    age: int | None = None
    sex: str | None = None


class SessionOut(BaseModel):
    id: str
    status: str
    patient_label: str | None
    interview_state: dict
    transcript: list
    counts: dict  # {sources: int, events: int, conflicts: int}
    # Phase 8: patient identity (null-safe for pre-Phase-8 sessions)
    patient_id: str | None = None
    patient_name: str | None = None
    patient_age: int | None = None
    patient_sex: str | None = None

    model_config = {"from_attributes": True}


from datetime import date as _date


class EventOut(BaseModel):
    id: str
    event_type: str
    description: str
    date_start: _date | None
    date_end: _date | None
    date_raw: str
    date_precision: str
    date_confidence: float
    confidence: float
    source_id: str | None
    structured: dict = Field(default_factory=dict)
    source_snippet: str | None = None

    model_config = {"from_attributes": True}


class ExtractOut(BaseModel):
    counts: dict
    total: int


# ── Phase 6A: manual event edits ─────────────────────────────────────────────

class EventPatchIn(BaseModel):
    """
    Partial update for a single Event. Only user-editable fields accepted.
    Fusion-owned fields (dedup_key, cluster_id, is_canonical, event_type,
    source_id) are silently ignored even if present in the request body.
    Use model_fields_set to distinguish 'explicitly set to None' from 'omitted'.
    """
    description: str | None = None
    date_start: _date | None = None
    date_end: _date | None = None
    date_raw: str | None = None
    date_precision: str | None = None
    date_confidence: float | None = None
    confidence: float | None = None
    structured: dict | None = None


class EventPatchOut(BaseModel):
    id: str
    event_type: str
    description: str
    date_start: _date | None
    date_end: _date | None
    date_raw: str
    date_precision: str
    date_confidence: float
    confidence: float
    source_id: str | None
    structured: dict = Field(default_factory=dict)
    is_canonical: bool
    cluster_id: str | None
    manually_edited: bool
    source_snippet: str | None = None

    model_config = {"from_attributes": True}


class CanonicalEventOut(BaseModel):
    id: str
    event_type: str
    description: str
    date_start: _date | None
    date_end: _date | None
    date_raw: str
    date_precision: str
    date_confidence: float
    confidence: float
    cluster_id: str | None
    cluster_size: int = 1
    source_ids: list[str] = Field(default_factory=list)
    structured: dict = Field(default_factory=dict)
    is_negation: bool = False       # patient denial — kept for contradiction detection only
    manually_edited: bool = False   # user has manually patched this event
    source_snippet: str | None = None

    model_config = {"from_attributes": True}


class ConflictOut(BaseModel):
    id: str
    conflict_type: str
    detail: str
    event_a_id: str
    event_b_id: str
    resolution: str

    model_config = {"from_attributes": True}


class TimelineOut(BaseModel):
    status: str
    events: list[CanonicalEventOut]
    conflicts: list[ConflictOut]


class ConflictResolveIn(BaseModel):
    resolution: str                    # a_wins | b_wins | both_noted
    canonical_choice: str | None = None  # "a" | "b" — applies is_canonical + manually_edited


# ── Phase 7C: Clinical summary ────────────────────────────────────────────────

class SummarySectionEntryOut(BaseModel):
    """One cited clinical claim inside a summary section."""
    text: str
    event_ids: list[str]  # canonical event IDs from the timeline that back this claim


class SummarySectionOut(BaseModel):
    """One category block in the summary (e.g. 'Active Conditions')."""
    heading: str
    entries: list[SummarySectionEntryOut]


class SummaryOut(BaseModel):
    """Full clinical summary response — narrative + provenance map."""
    sections: list[SummarySectionOut]
    conflicts_note: str                              # prose, empty string if no conflicts
    data_quality_note: str                           # caveats about vague dates / low confidence
    citations: dict[str, list["ProvenanceMemberOut"]]  # event_id → cluster members
    generated_at: str                                # ISO-8601 UTC timestamp


class ProvenanceMemberOut(BaseModel):
    """One cluster member returned by GET /events/{id}/provenance."""
    event_id: str
    is_canonical: bool
    source_id: str | None
    source_kind: str | None    # None when source is interview
    source_label: str | None   # RawSource.filename, or None
    source_snippet: str | None
    is_negation: bool
    confidence: float
    description: str


class DocumentUploadOut(BaseModel):
    source_id: str
    kind: str
    ocr_status: str


class DocumentOut(BaseModel):
    source_id: str
    filename: str | None
    kind: str
    ocr_status: str
    warnings: list[str] = Field(default_factory=list)


class DocumentDetailOut(BaseModel):
    source_id: str
    filename: str | None
    kind: str
    ocr_status: str
    extracted_text: str | None
    extraction_meta: dict = Field(default_factory=dict)


class ExtractionRetryIn(BaseModel):
    """Body for POST /interview/retry-extraction — re-runs per-turn extraction."""
    message: str


class InterviewTurnIn(BaseModel):
    message: str


class InterviewTurnOut(BaseModel):
    reply: str = ""
    updated_coverage: dict[str, str] = Field(default_factory=dict)
    open_threads: list[str] = Field(default_factory=list)
    next_target_topic: str = ""
    follow_up_reason: str = ""
    interview_complete: bool = False
    completion_justification: str | None = None

    @field_validator("reply", mode="before")
    @classmethod
    def _warn_empty_reply(cls, v: object) -> str:
        if not v:
            warnings.warn("LLM returned empty reply field")
        return str(v) if v else ""

    @field_validator("updated_coverage", mode="before")
    @classmethod
    def _fill_missing_topics(cls, v: object) -> dict:
        if not isinstance(v, dict):
            warnings.warn("LLM returned non-dict updated_coverage; resetting")
            v = {}
        for topic in _COVERAGE_TOPICS:
            if topic not in v:
                warnings.warn(f"LLM missing coverage for topic '{topic}'; defaulting to uncovered")
                v[topic] = "uncovered"
        return v
