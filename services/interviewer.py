import logging
from pathlib import Path

from models import Session
from schemas import InterviewTurnOut
from services.gemini_client import GeminiParseError, gemini_client

logger = logging.getLogger(__name__)

REQUIRED_TOPICS = [
    "chief_complaint",
    "chronic_conditions",
    "medications",
    "surgeries",
    "hospitalizations",
    "allergies",
    "family_history",
]

_interview_prompt = (
    Path(__file__).parent.parent / "prompts" / "interview.txt"
).read_text()


def init_state() -> dict:
    return {
        "coverage_map": {topic: "uncovered" for topic in REQUIRED_TOPICS},
        "turn_count": 0,
        "open_threads": [],
    }


def run_turn(session: Session, user_message: str) -> InterviewTurnOut:
    state = dict(session.interview_state) if session.interview_state else init_state()
    transcript = list(session.transcript) if session.transcript else []

    payload = {
        "user_message": user_message,
        "coverage_map": state.get(
            "coverage_map", {t: "uncovered" for t in REQUIRED_TOPICS}
        ),
        "open_threads": state.get("open_threads", []),
        "recent_turns": transcript[-6:],
        "turn_count": state.get("turn_count", 0),
    }

    raw = gemini_client.json_completion(_interview_prompt, payload)
    result = InterviewTurnOut(**raw)

    # Reassign dicts/lists to trigger SQLAlchemy JSON dirty-tracking
    session.interview_state = {
        "coverage_map": result.updated_coverage,
        "open_threads": result.open_threads,
        "turn_count": state.get("turn_count", 0) + 1,
    }

    new_transcript = list(transcript)
    if user_message.strip():
        new_transcript.append({"role": "user", "content": user_message})
    new_transcript.append({"role": "assistant", "content": result.reply})
    session.transcript = new_transcript

    return result


def is_complete(state: dict) -> bool:
    coverage = state.get("coverage_map", {})
    return all(v != "uncovered" for v in coverage.values())
