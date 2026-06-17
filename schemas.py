from __future__ import annotations

import warnings

from pydantic import BaseModel, Field, field_validator

# PHASE 2 (complete): InterviewTurnIn, InterviewTurnOut
# PHASE 3 (complete): DocumentUploadOut, DocumentOut, DocumentDetailOut
# PHASE 4 (complete): EventOut, ExtractOut
# PHASE 5: FusedTimelineOut, ConflictOut

_COVERAGE_TOPICS = [
    "chief_complaint",
    "chronic_conditions",
    "medications",
    "surgeries",
    "hospitalizations",
    "allergies",
    "family_history",
]


class SessionCreate(BaseModel):
    patient_label: str | None = None


class SessionOut(BaseModel):
    id: str
    status: str
    patient_label: str | None
    interview_state: dict
    transcript: list
    counts: dict  # {sources: int, events: int, conflicts: int}

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

    model_config = {"from_attributes": True}


class ExtractOut(BaseModel):
    counts: dict
    total: int


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
