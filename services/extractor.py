"""
Extract Event candidates from OCR output and interview transcripts.
Returns plain dicts suitable for passing directly to Event(**dict).
Does NOT persist to DB — the router handles that.
"""

from __future__ import annotations

import logging
import random
import re
import time
from pathlib import Path
from typing import Any

from google.genai.errors import ClientError, ServerError

from services import dates as dates_mod
from services.gemini_client import GeminiParseError, gemini_client

logger = logging.getLogger(__name__)

_extract_prompt = (
    Path(__file__).parent.parent / "prompts" / "extract_interview.txt"
).read_text()

_LEGIBILITY_CONFIDENCE = {"clear": 0.90, "unclear": 0.50, "unreadable": 0.20}
_VALID_EVENT_TYPES = {
    "Diagnosis", "Medication", "LabResult",
    "Surgery", "Hospitalization", "Symptom", "Consultation",
}

_NULL_LIKE = {"", "null", "none", "n/a", "no sample", "not done", "-"}

# Heuristic patterns that strongly signal a denial regardless of LLM is_negation flag.
# These catch cases where the model inverts a negation into a positive assertion.
_DENIAL_DESC_RE = re.compile(
    r"^patient\s+denies\b|^denies\b|^no\s+history\b|^never\s+had\b|"
    r"^no\s+h/o\b|^no\s+known\b",
    re.I,
)


# ── Per-turn extraction retry helpers ────────────────────────────────────────
# Same 503/429/500 retry pattern as the summary service (7C).

_EXT_MAX_RETRIES = 3
_EXT_BASE_DELAY  = 1.0   # seconds; doubles each attempt
_EXT_JITTER      = 0.5   # max random seconds added per delay


class ExtractionTransientError(Exception):
    """Raised when per-turn event extraction fails after retries (provider overload)."""


def _ext_is_transient(exc: Exception) -> bool:
    if isinstance(exc, ServerError) and exc.code in (500, 503):
        return True
    if isinstance(exc, ClientError) and exc.code == 429:
        return True
    return False


def _ext_retry_after(exc: Exception) -> float | None:
    try:
        resp = getattr(exc, "response", None)
        if resp is not None:
            raw = resp.headers.get("retry-after") or resp.headers.get("Retry-After")
            if raw:
                return float(raw)
    except Exception:
        pass
    return None


def _ext_call_with_retry(prompt: str, payload: dict) -> dict:
    """Call gemini_client.json_completion with exponential backoff on transient errors.

    GeminiParseError (bad JSON) is never retried — retrying won't help.
    Non-transient ClientErrors (400/401/403) are not retried either.
    Raises ExtractionTransientError after _EXT_MAX_RETRIES exhausted.
    """
    last_exc: Exception | None = None
    for attempt in range(_EXT_MAX_RETRIES + 1):
        try:
            return gemini_client.json_completion(prompt, payload)
        except GeminiParseError:
            raise
        except Exception as exc:
            if not _ext_is_transient(exc) or attempt == _EXT_MAX_RETRIES:
                last_exc = exc
                break
            delay = _ext_retry_after(exc) or (
                _EXT_BASE_DELAY * (2 ** attempt) + random.uniform(0, _EXT_JITTER)
            )
            logger.warning(
                "Interview extraction hit transient error (attempt %d/%d, code=%s) "
                "— retrying in %.1fs",
                attempt + 1, _EXT_MAX_RETRIES,
                getattr(exc, "code", "?"),
                delay,
            )
            time.sleep(delay)
            last_exc = exc

    if _ext_is_transient(last_exc):
        raise ExtractionTransientError(
            "Event extraction is temporarily unavailable — your answer was saved, "
            "please retry capture."
        ) from last_exc
    raise last_exc  # type: ignore[misc]


def _is_null_value(v: Any) -> bool:
    return v is None or str(v).strip().lower() in _NULL_LIKE


def _base(source_id: str | None, session_id: str, event_type: str,
          description: str, date_raw: str, date_result: dict,
          structured: dict, confidence: float,
          source_snippet: str | None = None) -> dict:
    return {
        "session_id": session_id,
        "source_id": source_id,
        "event_type": event_type,
        "description": description,
        "date_raw": date_raw,
        "date_start": date_result["date_start"],
        "date_end": date_result["date_end"],
        "date_precision": date_result["date_precision"],
        "date_confidence": date_result["date_confidence"],
        "structured": structured,
        "confidence": confidence,
        "is_canonical": True,
        "dedup_key": "",
        "cluster_id": None,
        "source_snippet": source_snippet or None,
    }


# ── Document extraction ──────────────────────────────────────────────────────

def extract_from_document(raw_source) -> list[dict]:
    """Extract Event dicts from a completed RawSource row."""
    if raw_source.ocr_status != "done":
        return []
    meta = raw_source.extraction_meta or {}
    kind = raw_source.kind
    sid = raw_source.session_id
    src = raw_source.id
    extracted_text = raw_source.extracted_text or ""

    if kind == "image_handwritten":
        return _from_prescription(meta, sid, src)
    if kind == "pdf_typed":
        return _from_typed_doc(meta, extracted_text, sid, src)
    return []


def _from_prescription(meta: dict, session_id: str, source_id: str) -> list[dict]:
    events: list[dict] = []
    date_raw = meta.get("prescribed_date") or ""
    dr = dates_mod.normalize(date_raw)

    for m in meta.get("medications") or []:
        leg = m.get("legibility", "clear")
        if leg == "unreadable":
            continue
        conf = _LEGIBILITY_CONFIDENCE.get(leg, 0.60)
        desc = m.get("normalized_guess") or m.get("raw_text") or m.get("drug_name")
        if not desc or _is_null_value(desc):
            continue
        structured = {k: m.get(k) for k in
                      ("raw_text", "normalized_guess", "strength", "dose",
                       "frequency", "duration", "legibility")}
        # raw_text is the verbatim prescription line — promote it as provenance
        snippet = m.get("raw_text") or None
        events.append(_base(source_id, session_id, "Medication",
                            desc, date_raw, dr, structured, conf,
                            source_snippet=snippet))

    for dx in meta.get("diagnoses") or []:
        if not dx or _is_null_value(dx):
            continue
        events.append(_base(source_id, session_id, "Diagnosis",
                            dx, date_raw, dr, {"source": "prescription"}, 0.80))

    return events


_MED_EXTRACT_SYS = (
    "Extract all medications from this typed prescription or clinical document text. "
    "Return ONLY valid JSON: "
    '{"medications":[{"drug_name":"<name>","strength":"<e.g. 500mg or null>",'
    '"dose":"<e.g. 1 tab or null>","frequency":"<e.g. OD/BD/TDS or null>","duration":"<or null>"}]} '
    "If no medications found, return {\"medications\":[]}."
)


def _fallback_meds_from_text(text: str) -> list[dict]:
    """LLM fallback: extract medications from extracted_text for existing typed prescriptions."""
    try:
        result = gemini_client.json_completion(_MED_EXTRACT_SYS, text[:4000])
        return result.get("medications") or []
    except Exception as exc:
        logger.warning("Medication fallback extraction failed: %s", exc)
        return []


def _from_typed_doc(meta: dict, extracted_text: str, session_id: str, source_id: str) -> list[dict]:
    events: list[dict] = []
    date_raw = meta.get("report_date") or ""
    dr = dates_mod.normalize(date_raw)

    for t in meta.get("tests") or []:
        name = t.get("name")
        value = t.get("value")
        if not name or _is_null_value(value):
            continue
        unit_str = f" {t['unit']}" if t.get("unit") else ""
        ref_str = f" (ref {t['reference_range']})" if t.get("reference_range") else ""
        flag_str = f" [{t['flag']}]" if t.get("flag") else ""
        desc = f"{name}: {value}{unit_str}{ref_str}{flag_str}"
        structured = {k: t.get(k) for k in
                      ("name", "value", "unit", "reference_range", "flag")}
        events.append(_base(source_id, session_id, "LabResult",
                            desc, date_raw, dr, structured, 0.85,
                            source_snippet=t.get("source_text") or None))

    for dx in meta.get("diagnoses") or []:
        # Diagnoses may be a plain string (old OCR cache) or {"text":..., "source_text":...} (new prompt)
        if isinstance(dx, dict):
            dx_text = dx.get("text") or ""
            dx_snippet = dx.get("source_text") or None
        else:
            dx_text = dx or ""
            dx_snippet = None
        if not dx_text or _is_null_value(dx_text):
            continue
        events.append(_base(source_id, session_id, "Diagnosis",
                            dx_text, date_raw, dr, {"source": "typed_document"}, 0.80,
                            source_snippet=dx_snippet))

    # Medications — present in meta when document was OCR'd with updated ocr_typed.txt,
    # OR recovered via LLM fallback for existing prescriptions OCR'd with the old prompt.
    meds = meta.get("medications") or []
    if not meds and meta.get("document_type") == "prescription" and extracted_text:
        logger.info("Typed prescription has no medications in meta; running LLM fallback")
        meds = _fallback_meds_from_text(extracted_text)

    for m in meds:
        drug_name = m.get("drug_name") or m.get("name")
        if not drug_name or _is_null_value(drug_name):
            continue
        strength = f" {m['strength']}" if m.get("strength") else ""
        desc = f"{drug_name}{strength}"
        structured = {
            "drug_name": drug_name,
            "normalized_guess": drug_name,  # align with handwritten schema for fusion
            "strength": m.get("strength"),
            "dose": m.get("dose"),
            "frequency": m.get("frequency"),
            "duration": m.get("duration"),
        }
        events.append(_base(source_id, session_id, "Medication",
                            desc, date_raw, dr, structured, 0.85,
                            source_snippet=m.get("source_text") or None))

    return events


# ── Interview extraction ─────────────────────────────────────────────────────

def extract_from_interview(session) -> list[dict]:
    """Extract Event dicts from a Session's transcript via LLM."""
    transcript = session.transcript or []
    if not transcript:
        logger.warning("extract_from_interview: transcript is empty for session %s", session.id)
        return []

    # Build the transcript array — pass as a JSON dict so the LLM
    # clearly receives the data (raw text as a user message confused the model
    # because the system prompt says "below" but the text was in a separate turn).
    transcript_entries = [
        {
            "role": "Patient" if t.get("role") == "user" else "Clinician",
            "text": t.get("content", ""),
        }
        for t in transcript
        if t.get("content", "").strip()
    ]
    if not transcript_entries:
        logger.warning("extract_from_interview: no non-empty turns for session %s", session.id)
        return []

    payload = {"transcript": transcript_entries}
    logger.info("extract_from_interview: %d turns for session %s", len(transcript_entries), session.id)

    try:
        result = gemini_client.json_completion(_extract_prompt, payload)
        raw_events = result.get("events") or []
        logger.info("extract_from_interview: LLM returned %d events", len(raw_events))
    except GeminiParseError as exc:
        logger.error("Interview extraction parse error: %s", exc)
        return []
    except Exception as exc:
        logger.error("Interview extraction failed: %s", exc)
        return []

    events: list[dict] = []
    for e in raw_events:
        event_type = e.get("event_type", "Symptom")
        if event_type not in _VALID_EVENT_TYPES:
            event_type = "Symptom"
        description = (e.get("description") or "").strip()
        if not description:
            continue
        date_raw = (e.get("date_raw") or "").strip()
        is_negation = bool(e.get("is_negation", False))
        source_snippet = (e.get("source_turn") or "").strip() or None
        structured = dict(e.get("structured") or {})

        # ── Defensive negation repair ──────────────────────────────────────
        # If the LLM inverted the negation (said is_negation=false but the
        # description signals a denial), force is_negation=True.
        if not is_negation and _DENIAL_DESC_RE.match(description):
            is_negation = True
            logger.warning("Forced is_negation=True based on description text: %r", description)

        # If is_negation=True, ensure description reads "Patient denies X"
        # and structured.negated_claim = "X" (without the prefix).
        if is_negation:
            # Extract the denied thing
            negated_claim = structured.get("negated_claim") or ""
            if not negated_claim:
                # Derive from description: strip known denial prefixes
                negated_claim = re.sub(
                    r"^patient\s+denies\s+|^denies\s+|^no\s+history\s+of\s+|"
                    r"^never\s+had\s+|^no\s+h/o\s+|^no\s+known\s+",
                    "",
                    description,
                    flags=re.I,
                ).strip() or description
            structured["negated_claim"] = negated_claim
            # Canonical denial description so it's never confused with a positive event
            description = f"Patient denies {negated_claim}"

        structured["is_negation"] = is_negation

        # For Medication events: ensure normalized_guess is set so fusion dedup
        # can match interview drug names against document drug names.
        if event_type == "Medication" and "normalized_guess" not in structured:
            drug_name = structured.get("drug_name") or description
            structured["normalized_guess"] = drug_name.strip().lower()

        confidence = min(float(e.get("confidence") or 0.6), 0.6)  # cap self-reported
        dr = dates_mod.normalize(date_raw) if date_raw else dates_mod.unknown()
        events.append(_base(None, session.id, event_type,
                            description, date_raw, dr, structured, confidence,
                            source_snippet=source_snippet))

    return events


# ── Per-turn extraction (Phase 8) ────────────────────────────────────────────

def extract_single_turn(message: str, session_id: str) -> list[dict]:
    """Extract Event dicts from a single patient message, with retry on transient errors.

    Unlike extract_from_interview (which silently returns [] on failure so the
    batch Extract button never appears to error), this function RAISES
    ExtractionTransientError if the LLM is unavailable after retries. The caller
    decides how to surface that to the user.

    Returns [] legitimately when the message contains no medical events.
    """
    if not message or not message.strip():
        return []

    payload = {"transcript": [{"role": "Patient", "text": message.strip()}]}
    logger.info("extract_single_turn: extracting events from 1-turn message for session %s", session_id)

    try:
        result = _ext_call_with_retry(_extract_prompt, payload)
    except ExtractionTransientError:
        raise  # let the router show the "capture failed" badge
    except GeminiParseError as exc:
        logger.error("extract_single_turn parse error: %s", exc)
        return []  # model returned bad JSON — not a transient error; treat as 0 events

    raw_events = result.get("events") or []
    logger.info("extract_single_turn: LLM returned %d events for session %s", len(raw_events), session_id)

    # ── Event processing — identical logic to extract_from_interview ──────────
    events: list[dict] = []
    for e in raw_events:
        event_type = e.get("event_type", "Symptom")
        if event_type not in _VALID_EVENT_TYPES:
            event_type = "Symptom"
        description = (e.get("description") or "").strip()
        if not description:
            continue
        date_raw = (e.get("date_raw") or "").strip()
        is_negation = bool(e.get("is_negation", False))
        source_snippet = (e.get("source_turn") or "").strip() or None
        structured = dict(e.get("structured") or {})

        if not is_negation and _DENIAL_DESC_RE.match(description):
            is_negation = True
            logger.warning("extract_single_turn: forced is_negation=True: %r", description)

        if is_negation:
            negated_claim = structured.get("negated_claim") or ""
            if not negated_claim:
                negated_claim = re.sub(
                    r"^patient\s+denies\s+|^denies\s+|^no\s+history\s+of\s+|"
                    r"^never\s+had\s+|^no\s+h/o\s+|^no\s+known\s+",
                    "",
                    description,
                    flags=re.I,
                ).strip() or description
            structured["negated_claim"] = negated_claim
            description = f"Patient denies {negated_claim}"

        structured["is_negation"] = is_negation

        if event_type == "Medication" and "normalized_guess" not in structured:
            drug_name = structured.get("drug_name") or description
            structured["normalized_guess"] = drug_name.strip().lower()

        confidence = min(float(e.get("confidence") or 0.6), 0.6)
        dr = dates_mod.normalize(date_raw) if date_raw else dates_mod.unknown()
        events.append(_base(None, session_id, event_type,
                            description, date_raw, dr, structured, confidence,
                            source_snippet=source_snippet))

    return events
