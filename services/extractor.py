"""
Extract Event candidates from OCR output and interview transcripts.
Returns plain dicts suitable for passing directly to Event(**dict).
Does NOT persist to DB — the router handles that.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

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


def _is_null_value(v: Any) -> bool:
    return v is None or str(v).strip().lower() in _NULL_LIKE


def _base(source_id: str | None, session_id: str, event_type: str,
          description: str, date_raw: str, date_result: dict,
          structured: dict, confidence: float) -> dict:
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
        events.append(_base(source_id, session_id, "Medication",
                            desc, date_raw, dr, structured, conf))

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
                            desc, date_raw, dr, structured, 0.85))

    for dx in meta.get("diagnoses") or []:
        if not dx or _is_null_value(dx):
            continue
        events.append(_base(source_id, session_id, "Diagnosis",
                            dx, date_raw, dr, {"source": "typed_document"}, 0.80))

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
                            desc, date_raw, dr, structured, 0.85))

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
        structured = e.get("structured") or {}
        confidence = min(float(e.get("confidence") or 0.6), 0.6)  # cap self-reported

        dr = dates_mod.normalize(date_raw) if date_raw else dates_mod.unknown()
        events.append(_base(None, session.id, event_type,
                            description, date_raw, dr, structured, confidence))

    return events
