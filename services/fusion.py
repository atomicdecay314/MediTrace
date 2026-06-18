"""
Phase 5 — Timeline fusion service.
Pipeline (single DB transaction):
  A. Compute dedup keys, cluster events deterministically.
  B. Per multi-event cluster: LLM merge + intra-cluster conflict detection.
  C. Cross-cluster contradiction pass.
"""

from __future__ import annotations

import logging
import re
import uuid
from collections import defaultdict
from datetime import date
from pathlib import Path

from pydantic import BaseModel, Field
from sqlalchemy.orm import Session as DBSession

from models import Conflict, Event, RawSource
from services.gemini_client import GeminiParseError, gemini_client

logger = logging.getLogger(__name__)

# ── Text normalisation ───────────────────────────────────────────────────────

_FORM_RE = re.compile(
    r'\b(tablet|tab|tabs|capsule|cap|caps|injection|inj|syrup|syr|'
    r'suspension|susp|ampule|amp|ampoule|cream|ointment|oint|'
    r'drops|drp|gtt|inhaler|inh|patch|solution|soln|forte|plus|'
    r'sr|mr|xr|er|ds|forte)\b',
    re.I,
)
_DOSE_RE = re.compile(
    r'\d+(\.\d+)?\s*(mg|ml|g|mcg|iu|miu|units?|%|mmol|meq)(/\s*(ml|kg))?',
    re.I,
)
_DX_PREFIX = re.compile(
    r'^(k/c/o|k\.c\.o|known case of|c/o|c\.o|h/o|h\.o|history of|'
    r'presenting with|presenting c/o|complaints? of|complaints?:?)\s+',
    re.I,
)
_DX_SYNONYMS: dict[str, str] = {
    'htn':                        'hypertension',
    'high blood pressure':        'hypertension',
    'elevated blood pressure':    'hypertension',
    'hypertensive':               'hypertension',
    'dm':                         'type 2 diabetes',
    't2dm':                       'type 2 diabetes',
    'type 2 dm':                  'type 2 diabetes',
    'type 2 diabetes mellitus':   'type 2 diabetes',
    'type ii diabetes':           'type 2 diabetes',
    'diabetes mellitus type 2':   'type 2 diabetes',
    'diabetes mellitus':          'diabetes',
    'ihd':                        'ischemic heart disease',
    'cad':                        'coronary artery disease',
    'gerd':                       'gastroesophageal reflux disease',
    'acid reflux':                'gastroesophageal reflux disease',
    'copd':                       'chronic obstructive pulmonary disease',
    'uti':                        'urinary tract infection',
    'pain abdomen':               'abdominal pain',
    'c/o pain abdomen':           'abdominal pain',
    'abdo pain':                  'abdominal pain',
    'cholecystitis':              'cholecystitis',
    'gallbladder problem':        'cholecystitis',
    'gallbladder disease':        'cholecystitis',
    'gallstones':                 'cholelithiasis',
}


def _norm_med(text: str) -> str:
    t = text.lower().strip()
    t = _FORM_RE.sub(' ', t)
    t = _DOSE_RE.sub(' ', t)
    t = re.sub(r'[^a-z0-9\s]', ' ', t)
    return re.sub(r'\s+', ' ', t).strip()


def _norm_dx(text: str) -> str:
    t = _DX_PREFIX.sub('', text.lower().strip()).strip()
    for k, v in _DX_SYNONYMS.items():
        if k in t:
            return v
    return t


def _norm_lab(desc: str) -> str:
    return desc.split(':')[0].lower().strip()


def _entity_key(e: Event) -> str:
    struct = e.structured or {}

    # Negation events get their own key space so they never cluster with
    # the positive events they contradict.
    if struct.get('is_negation'):
        if e.event_type == 'Medication':
            name = struct.get('drug_name') or struct.get('normalized_guess') or e.description
            return 'neg_med|' + _norm_med(name)
        return 'neg|' + e.event_type.lower() + '|' + _norm_dx(e.description)

    if e.event_type == 'Medication':
        # normalized_guess is set by extractor for both document and interview meds;
        # drug_name is the LLM-provided clean name for interview meds.
        name = (struct.get('normalized_guess') or
                struct.get('drug_name') or
                struct.get('raw_text') or
                e.description)
        return 'med|' + _norm_med(name)
    if e.event_type == 'Diagnosis':
        return 'dx|' + _norm_dx(e.description)
    if e.event_type == 'LabResult':
        return 'lab|' + _norm_lab(e.description)
    return e.event_type.lower() + '|' + re.sub(r'\s+', ' ', e.description.lower())[:80]


def _date_bucket(e: Event) -> str:
    """
    Coarse date bucket.
    - LabResult: full ISO date — two LabResults with the same test but different
      dates must NEVER merge.
    - Others: year only, or "" if unknown (unknown matches any year).
    """
    if e.event_type == 'LabResult':
        return e.date_start.isoformat() if e.date_start else f'_nodate_{e.id}'
    return str(e.date_start.year) if e.date_start else ''


def _can_cluster(a: Event, b: Event) -> bool:
    ba, bb = _date_bucket(a), _date_bucket(b)
    if a.event_type == 'LabResult':
        return ba == bb          # LabResults: must share exact date
    return not ba or not bb or ba == bb   # Others: unknown date matches any year


# ── Union-find ───────────────────────────────────────────────────────────────

def _cluster_group(events: list[Event]) -> list[list[Event]]:
    n = len(events)
    if n == 0:
        return []
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(n):
        for j in range(i + 1, n):
            if _can_cluster(events[i], events[j]):
                pi, pj = find(i), find(j)
                if pi != pj:
                    parent[pi] = pj

    groups: dict[int, list[Event]] = defaultdict(list)
    for i in range(n):
        groups[find(i)].append(events[i])
    return list(groups.values())


# ── Source trust ─────────────────────────────────────────────────────────────

def _build_kind_map(events: list[Event], db: DBSession) -> dict[str, str]:
    src_ids = {e.source_id for e in events if e.source_id}
    if not src_ids:
        return {}
    sources = db.query(RawSource).filter(RawSource.id.in_(src_ids)).all()
    return {s.id: s.kind for s in sources}


def _trust(source_id: str | None, kind_map: dict[str, str]) -> int:
    if source_id is None:
        return 0                          # interview
    return {'pdf_typed': 2, 'image_handwritten': 1}.get(kind_map.get(source_id, ''), 0)


# ── LLM fusion per cluster ────────────────────────────────────────────────────

_fusion_prompt: str | None = None


def _get_fusion_prompt() -> str:
    global _fusion_prompt
    if _fusion_prompt is None:
        _fusion_prompt = (Path(__file__).parent.parent / 'prompts' / 'fusion.txt').read_text()
    return _fusion_prompt


class _FusionCanon(BaseModel):
    event_type: str = ''
    description: str = ''
    date_start: str | None = None
    date_end: str | None = None
    date_precision: str = 'unknown'
    date_confidence: float = 0.0
    structured: dict = Field(default_factory=dict)
    source_ids: list[str] = Field(default_factory=list)
    confidence: float = 0.0


class _FusionConflict(BaseModel):
    event_ref_a: str = ''
    event_ref_b: str = ''
    conflict_type: str = 'contradiction'
    detail: str = ''


class _FusionResult(BaseModel):
    canonical: _FusionCanon = Field(default_factory=_FusionCanon)
    split_out: list[str] = Field(default_factory=list)
    conflicts: list[_FusionConflict] = Field(default_factory=list)


def _parse_date(s: str | None) -> date | None:
    if not s:
        return None
    try:
        return date.fromisoformat(str(s)[:10])
    except ValueError:
        return None


def _call_fusion_llm(cluster: list[Event], kind_map: dict[str, str]) -> _FusionResult:
    candidates = [
        {
            'id': e.id,
            'event_type': e.event_type,
            'description': e.description,
            'date_start': e.date_start.isoformat() if e.date_start else None,
            'date_end': e.date_end.isoformat() if e.date_end else None,
            'date_precision': e.date_precision,
            'date_confidence': e.date_confidence,
            'date_raw': e.date_raw,
            'structured': e.structured or {},
            'source_id': e.source_id,
            'source_kind': kind_map.get(e.source_id, 'interview') if e.source_id else 'interview',
            'confidence': e.confidence,
        }
        for e in cluster
    ]
    raw = gemini_client.json_completion(_get_fusion_prompt(), {'candidates': candidates})
    return _FusionResult(**raw)


# ── Apply cluster result ──────────────────────────────────────────────────────

def _record_intra_conflicts(
    result: _FusionResult, cluster: list[Event], session_id: str, db: DBSession
) -> None:
    """Record intra-cluster conflicts returned by the LLM. Skips already-resolved pairs."""
    id_map = {e.id: e for e in cluster}
    src_to_ev = {e.source_id: e for e in cluster if e.source_id}

    def _resolve(ref: str) -> str | None:
        if ref in id_map:
            return ref
        if ref in src_to_ev:
            return src_to_ev[ref].id
        return None

    for fc in result.conflicts:
        a_id = _resolve(fc.event_ref_a)
        b_id = _resolve(fc.event_ref_b)
        if not (a_id and b_id and a_id != b_id):
            continue
        # Don't re-create a conflict that the user already resolved
        existing = db.query(Conflict).filter(
            Conflict.session_id == session_id,
            Conflict.event_a_id.in_([a_id, b_id]),
            Conflict.event_b_id.in_([a_id, b_id]),
        ).first()
        if not existing:
            db.add(Conflict(
                id=str(uuid.uuid4()),
                session_id=session_id,
                event_a_id=a_id,
                event_b_id=b_id,
                conflict_type=fc.conflict_type,
                detail=fc.detail,
                resolution='unresolved',
            ))


def _apply_cluster(
    cluster: list[Event],
    result: _FusionResult,
    kind_map: dict[str, str],
    session_id: str,
    db: DBSession,
) -> None:
    split_ids = set(result.split_out)
    in_cluster = [e for e in cluster if e.id not in split_ids]
    split_out  = [e for e in cluster if e.id in split_ids]

    cluster_id = str(uuid.uuid4())

    def rank(e: Event) -> tuple:
        return (_trust(e.source_id, kind_map), e.confidence, e.date_confidence)

    # ── Manually-edited events win as canonical unconditionally ────────────
    # A manually_edited event was patched by the user (or pinned by conflict
    # resolution). Its description, dates, structured, and confidence must
    # not be overwritten by the LLM result.
    manually_edited_in_cluster = [e for e in in_cluster if e.manually_edited]
    if manually_edited_in_cluster:
        canon_ev = manually_edited_in_cluster[0]
        canon_ev.cluster_id = cluster_id
        canon_ev.is_canonical = True
        for e in in_cluster:
            if e.id != canon_ev.id:
                e.cluster_id   = cluster_id
                e.is_canonical = False
        for e in split_out:
            e.cluster_id   = str(uuid.uuid4())
            e.is_canonical = True
        # Still record any intra-cluster conflicts the LLM detected
        _record_intra_conflicts(result, cluster, session_id, db)
        return

    # ── Normal fusion: LLM picks canonical and edits its fields ───────────
    ranked = sorted(in_cluster, key=rank, reverse=True)
    canon_ev = ranked[0] if ranked else cluster[0]

    c = result.canonical
    canon_ev.cluster_id   = cluster_id
    canon_ev.is_canonical = True
    canon_ev.description  = c.description or canon_ev.description
    canon_ev.date_start   = _parse_date(c.date_start) or canon_ev.date_start
    canon_ev.date_end     = _parse_date(c.date_end) or canon_ev.date_end
    canon_ev.date_precision  = c.date_precision or canon_ev.date_precision
    canon_ev.date_confidence = c.date_confidence or canon_ev.date_confidence
    canon_ev.structured   = {**(canon_ev.structured or {}), **(c.structured or {})}
    if c.confidence:
        canon_ev.confidence = c.confidence

    # Date inheritance fallback
    if not canon_ev.date_start:
        dated = sorted(
            [e for e in in_cluster if e.date_start and e.id != canon_ev.id],
            key=rank, reverse=True,
        )
        if dated:
            src = dated[0]
            canon_ev.date_start      = src.date_start
            canon_ev.date_end        = src.date_end
            canon_ev.date_precision  = src.date_precision
            canon_ev.date_confidence = src.date_confidence

    for e in in_cluster:
        if e.id != canon_ev.id:
            e.cluster_id   = cluster_id
            e.is_canonical = False

    for e in split_out:
        e.cluster_id   = str(uuid.uuid4())
        e.is_canonical = True

    _record_intra_conflicts(result, cluster, session_id, db)


# ── Cross-cluster contradiction pass ──────────────────────────────────────────

_CONTRADICT_SYS = """You are a clinical contradiction detector. You receive:
  patient_denials — things the patient explicitly said they do NOT have / never had
  documented_events — diagnoses and lab results from medical records

Find pairs where what the patient denied clearly contradicts a documented finding.
Apply clinical synonym reasoning:
  "cholesterol problems" ↔ dyslipidemia / hyperlipidemia / high LDL / elevated total cholesterol / hypercholesterolemia
  "heart problems" ↔ CAD / ischemic heart disease / myocardial infarction / angina
  "kidney problems" ↔ CKD / nephropathy / renal failure
  "sugar problems" / "blood sugar" ↔ diabetes / hyperglycemia
  "breathing problems" ↔ COPD / asthma / dyspnea

For each contradiction: cite BOTH the denial AND the specific documented finding.
For lab results include the value and flag (e.g. "LDL 162 mg/dL [HIGH]").

Return ONLY valid JSON:
{
  "contradictions": [
    {
      "denial_event_id": "<id from patient_denials>",
      "documented_event_id": "<id from documented_events>",
      "detail": "<e.g. Patient denies cholesterol problems; records show Dyslipidemia and LDL 162 mg/dL [HIGH], Total Cholesterol 245 mg/dL [HIGH]>"
    }
  ]
}
If no contradictions found: {"contradictions": []}"""


def _contradiction_pass(canonical: list[Event], session_id: str, db: DBSession) -> None:
    # Only check patient denials (is_negation=True) against documented positives.
    # Passing ALL events previously caused the LLM to miss clinical synonym links.
    negations = [e for e in canonical if (e.structured or {}).get('is_negation')]
    if not negations:
        return

    documented = [
        e for e in canonical
        if not (e.structured or {}).get('is_negation')
        and e.event_type in ('Diagnosis', 'LabResult')
    ]
    if not documented:
        return

    payload = {
        'patient_denials': [
            {
                'id': e.id,
                # Use negated_claim (just the denied thing, e.g. "cholesterol problems")
                # NOT the full description ("Patient denies cholesterol problems")
                # so the LLM synonym match is clean.
                'denied_claim': (e.structured or {}).get('negated_claim') or
                                re.sub(r'^patient\s+denies\s+', '', e.description, flags=re.I).strip(),
                'event_type': e.event_type,
            }
            for e in negations
        ],
        'documented_events': [
            {'id': e.id, 'event_type': e.event_type, 'description': e.description}
            for e in documented
        ],
    }
    try:
        raw = gemini_client.json_completion(_CONTRADICT_SYS, payload)
        items = raw.get('contradictions') or []
    except Exception as exc:
        logger.warning('Contradiction pass failed: %s', exc)
        return

    ev_map = {e.id: e for e in canonical}
    for item in items:
        a_id = item.get('denial_event_id', '')
        b_id = item.get('documented_event_id', '')
        if a_id in ev_map and b_id in ev_map and a_id != b_id:
            existing = db.query(Conflict).filter(
                Conflict.session_id == session_id,
                Conflict.event_a_id.in_([a_id, b_id]),
                Conflict.event_b_id.in_([a_id, b_id]),
            ).first()
            if not existing:
                db.add(Conflict(
                    id=str(uuid.uuid4()),
                    session_id=session_id,
                    event_a_id=a_id,
                    event_b_id=b_id,
                    conflict_type='contradiction',
                    detail=item.get('detail', 'Contradiction detected'),
                    resolution='unresolved',
                ))


# ── Main entry point ──────────────────────────────────────────────────────────

def run_fusion(session_id: str, db: DBSession) -> None:
    """
    Full pipeline in one transaction (caller commits/rolls back).
    A → dedup + cluster
    B → LLM merge per multi-event cluster
    C → cross-cluster contradiction pass
    """
    events = db.query(Event).filter(Event.session_id == session_id).all()
    if not events:
        return

    kind_map = _build_kind_map(events, db)

    # Clear prior fusion state — but preserve resolved conflicts so user
    # resolutions survive re-fusion (only unresolved ones get re-generated).
    db.query(Conflict).filter(
        Conflict.session_id == session_id,
        Conflict.resolution == 'unresolved',
    ).delete(synchronize_session=False)

    # Reset clustering state for all events. manually_edited events keep their
    # user-edited content (description, dates, structured) — only cluster
    # bookkeeping is reset here; _apply_cluster enforces the field preservation.
    for e in events:
        e.is_canonical = True   # will be overridden in Steps A/B
        e.cluster_id = None     # will be assigned in Steps A/B
        e.dedup_key = ''        # will be recomputed in Step A
    db.flush()

    # STEP A: compute dedup keys and group by entity
    entity_groups: dict[str, list[Event]] = defaultdict(list)
    for e in events:
        ek = _entity_key(e)
        e.dedup_key = f'{e.event_type}|{ek}|{_date_bucket(e)}'
        entity_groups[ek].append(e)

    all_clusters: list[list[Event]] = []
    for group in entity_groups.values():
        all_clusters.extend(_cluster_group(group))

    # Assign singleton cluster IDs immediately
    for cluster in all_clusters:
        if len(cluster) == 1:
            cluster[0].cluster_id = str(uuid.uuid4())

    # STEP B: LLM merge for multi-event clusters
    for cluster in all_clusters:
        if len(cluster) <= 1:
            continue
        try:
            result = _call_fusion_llm(cluster, kind_map)
            _apply_cluster(cluster, result, kind_map, session_id, db)
        except Exception as exc:
            logger.error('Fusion LLM failed for cluster (size %d): %s', len(cluster), exc)
            # Fallback: manually_edited wins; otherwise highest-trust/confidence
            manually_edited = [e for e in cluster if e.manually_edited]
            best = manually_edited[0] if manually_edited else max(
                cluster, key=lambda e: (_trust(e.source_id, kind_map), e.confidence)
            )
            cid = str(uuid.uuid4())
            for e in cluster:
                e.cluster_id   = cid
                e.is_canonical = e.id == best.id
            # Date inheritance fallback
            if not best.date_start:
                dated = [e for e in cluster if e.date_start and e.id != best.id]
                if dated:
                    src = max(dated, key=lambda e: (_trust(e.source_id, kind_map), e.date_confidence))
                    best.date_start      = src.date_start
                    best.date_end        = src.date_end
                    best.date_precision  = src.date_precision
                    best.date_confidence = src.date_confidence

    db.flush()

    # STEP C: cross-cluster contradiction pass
    canonical = [e for e in events if e.is_canonical]
    _contradiction_pass(canonical, session_id, db)
    db.flush()
