from __future__ import annotations

import warnings

from pydantic import BaseModel, Field, field_validator

# PHASE 2 (complete): InterviewTurnIn, InterviewTurnOut
# PHASE 2 remaining: AudioUploadOut, TranscriptOut
# PHASE 3: RawSourceOut, DocumentUploadOut, OcrStatusOut
# PHASE 4: EventOut, ConflictOut, TimelineOut
# PHASE 5: FusionResultOut

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
