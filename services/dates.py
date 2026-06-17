"""
Deterministic-first date normalizer for MediTrace.

normalize() has NO database or LLM dependency in its main path.
An optional llm_resolver hook is accepted but never called in tests.
"""

from __future__ import annotations

import re
from calendar import monthrange
from datetime import date, timedelta
from typing import Callable, TypedDict

__all__ = ["normalize", "unknown", "DateResult"]


class DateResult(TypedDict):
    date_start: date | None
    date_end: date | None
    date_precision: str   # exact | month | year | approx | unknown
    date_confidence: float


MONTH_NAMES: dict[str, int] = {
    "january": 1,  "february": 2,  "march": 3,     "april": 4,
    "may": 5,      "june": 6,      "july": 7,       "august": 8,
    "september": 9,"october": 10,  "november": 11,  "december": 12,
    "jan": 1, "feb": 2, "mar": 3, "apr": 4,
    "jun": 6, "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}


def unknown() -> DateResult:
    return {"date_start": None, "date_end": None,
            "date_precision": "unknown", "date_confidence": 0.0}


def _last_day(year: int, month: int) -> date:
    return date(year, month, monthrange(year, month)[1])


def _expand_year(y: int) -> int:
    """Expand 2-digit year: <50 → 20xx, ≥50 → 19xx."""
    if y >= 100:
        return y
    return 2000 + y if y < 50 else 1900 + y


def _make_exact(year: int, month: int, day: int) -> DateResult | None:
    try:
        d = date(year, month, day)
        return {"date_start": d, "date_end": d,
                "date_precision": "exact", "date_confidence": 0.95}
    except ValueError:
        return None


def _make_month(year: int, month: int, confidence: float = 0.65) -> DateResult:
    return {"date_start": date(year, month, 1),
            "date_end": _last_day(year, month),
            "date_precision": "month", "date_confidence": confidence}


# ── Rule 1: explicit full dates ─────────────────────────────────────────────

def _try_explicit(text: str) -> DateResult | None:
    t = text.strip()

    # "DD/MM/YYYY", "DD-MM-YYYY", "DD.MM.YYYY", "YYYY-MM-DD"
    m = re.match(r'^(\d{1,4})[/\-.](\d{1,2})[/\-.](\d{1,4})$', t)
    if m:
        a, b, c = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if a > 31:                  # YYYY-MM-DD (ISO)
            return _make_exact(a, b, c)
        else:                       # c is the year
            year = _expand_year(c)
            # DD/MM default (Indian clinical context); a > 12 forces day
            day, month = (a, b) if a > 12 else (a, b)  # default DD/MM
            return _make_exact(year, month, day)

    # "DD MonthName YYYY|YY"  e.g. "13 July 2021", "14 Jun 26"
    m = re.match(r'^(\d{1,2})\s+([A-Za-z]+)\s+(\d{2,4})$', t, re.I)
    if m:
        day, mon_str, yr = int(m.group(1)), m.group(2).lower(), int(m.group(3))
        month = MONTH_NAMES.get(mon_str)
        if month and 1 <= day <= 31:
            return _make_exact(_expand_year(yr), month, day)

    # "MonthName DD YYYY"  e.g. "July 13 2021"
    m = re.match(r'^([A-Za-z]+)\s+(\d{1,2})\s+(\d{2,4})$', t, re.I)
    if m:
        mon_str, day, yr = m.group(1).lower(), int(m.group(2)), int(m.group(3))
        month = MONTH_NAMES.get(mon_str)
        if month and 1 <= day <= 31:
            return _make_exact(_expand_year(yr), month, day)

    return None


# ── Rule 2: month + year ────────────────────────────────────────────────────

def _try_month_year(text: str) -> DateResult | None:
    tl = text.strip().lower()

    # "early/mid/late YYYY"
    m = re.match(r'^(early|mid|late)\s+(\d{4})$', tl)
    if m:
        q, year = m.group(1), int(m.group(2))
        spans = {"early": (1, 4), "mid": (5, 8), "late": (9, 12)}
        s, e = spans[q]
        return {"date_start": date(year, s, 1),
                "date_end": _last_day(year, e),
                "date_precision": "month", "date_confidence": 0.50}

    # "MonthName YYYY" or "YYYY MonthName"
    for pat in [r'^([A-Za-z]+)\s*,?\s*(\d{4})$', r'^(\d{4})\s+([A-Za-z]+)$']:
        m = re.match(pat, tl)
        if m:
            g1, g2 = m.group(1), m.group(2)
            mon_str, yr_str = (g2, g1) if g1.isdigit() else (g1, g2)
            month = MONTH_NAMES.get(mon_str.lower())
            if month:
                return _make_month(int(yr_str), month, 0.65)

    # "MonName YY"  e.g. "Jun 26" → June 2026
    m = re.match(r'^([A-Za-z]+)\s+(\d{2})$', tl)
    if m:
        mon_str, yr2 = m.group(1), int(m.group(2))
        month = MONTH_NAMES.get(mon_str)
        if month:
            return _make_month(_expand_year(yr2), month, 0.65)

    return None


# ── Rule 3: year only ───────────────────────────────────────────────────────

def _try_year_only(text: str) -> DateResult | None:
    tl = text.strip().lower()
    m = re.match(r'^(?:in\s+|year\s+)?(\d{4})$', tl)
    if m:
        year = int(m.group(1))
        if 1900 <= year <= 2100:
            return {"date_start": date(year, 1, 1), "date_end": date(year, 12, 31),
                    "date_precision": "year", "date_confidence": 0.60}
    return None


# ── Rule 4: relative expressions ────────────────────────────────────────────

def _try_relative(text: str, ref: date) -> DateResult | None:
    tl = text.strip().lower()

    if tl == "yesterday":
        d = ref - timedelta(days=1)
        return {"date_start": d, "date_end": d,
                "date_precision": "exact", "date_confidence": 0.90}

    if tl == "today":
        return {"date_start": ref, "date_end": ref,
                "date_precision": "exact", "date_confidence": 0.90}

    if re.match(r'^last\s+week$', tl):
        end = ref - timedelta(days=ref.weekday() + 1)
        return {"date_start": end - timedelta(days=6), "date_end": end,
                "date_precision": "approx", "date_confidence": 0.70}

    if re.match(r'^last\s+month$', tl):
        yr, mo = (ref.year - 1, 12) if ref.month == 1 else (ref.year, ref.month - 1)
        return _make_month(yr, mo, 0.75)

    if re.match(r'^last\s+year$', tl):
        yr = ref.year - 1
        return {"date_start": date(yr, 1, 1), "date_end": date(yr, 12, 31),
                "date_precision": "year", "date_confidence": 0.75}

    # "last [MonthName]"
    m = re.match(r'^last\s+([A-Za-z]+)$', tl)
    if m:
        month = MONTH_NAMES.get(m.group(1))
        if month:
            yr = ref.year if month < ref.month else ref.year - 1
            return _make_month(yr, month, 0.70)

    # "N years/months/weeks/days ago"
    m = re.match(r'^(\d+)\s+(year|month|week|day)s?\s+ago$', tl)
    if m:
        n, unit = int(m.group(1)), m.group(2)
        if unit == "year":
            try:
                d = date(ref.year - n, ref.month, ref.day)
            except ValueError:
                d = _last_day(ref.year - n, ref.month)
            return {"date_start": d, "date_end": d,
                    "date_precision": "approx", "date_confidence": 0.65}
        if unit == "month":
            total = ref.year * 12 + (ref.month - 1) - n
            yr, mo = divmod(total, 12)
            mo += 1
            try:
                d = date(yr, mo, ref.day)
            except ValueError:
                d = _last_day(yr, mo)
            return {"date_start": d, "date_end": d,
                    "date_precision": "approx", "date_confidence": 0.65}
        if unit == "week":
            d = ref - timedelta(weeks=n)
            return {"date_start": d, "date_end": d,
                    "date_precision": "approx", "date_confidence": 0.70}
        if unit == "day":
            d = ref - timedelta(days=n)
            return {"date_start": d, "date_end": d,
                    "date_precision": "exact", "date_confidence": 0.85}

    # "a few / several / many years ago"
    m = re.match(r'^(a\s+few|few|several|some|many)\s+years?\s+ago$', tl)
    if m:
        phrase = m.group(1)
        ranges = {
            "a few": (2, 4), "few": (2, 4),
            "several": (3, 6), "some": (3, 6),
            "many": (5, 10),
        }
        lo, hi = ranges.get(phrase, (2, 4))
        return {"date_start": date(ref.year - hi, 1, 1),
                "date_end": date(ref.year - lo, 12, 31),
                "date_precision": "approx", "date_confidence": 0.30}

    # "a few / several months ago"
    m = re.match(r'^(a\s+few|few|several)\s+months?\s+ago$', tl)
    if m:
        lo, hi = 2, 4
        s_total = ref.year * 12 + (ref.month - 1) - hi
        e_total = ref.year * 12 + (ref.month - 1) - lo
        s_yr, s_mo = divmod(s_total, 12); s_mo += 1
        e_yr, e_mo = divmod(e_total, 12); e_mo += 1
        return {"date_start": date(s_yr, s_mo, 1),
                "date_end": _last_day(e_yr, e_mo),
                "date_precision": "approx", "date_confidence": 0.35}

    if re.search(r'\brecently\b', tl):
        return {"date_start": ref - timedelta(days=90), "date_end": ref,
                "date_precision": "approx", "date_confidence": 0.30}

    return None


# ── Rule 5: age-relative ────────────────────────────────────────────────────

def _try_age_relative(text: str, patient_age: int, ref: date) -> DateResult | None:
    tl = text.lower()
    birth_year = ref.year - patient_age

    age_patterns: list[tuple[str, int, int]] = [
        (r'\bteenager\b|\bteen\b|\badolescent\b', 13, 19),
        (r'\bchild(hood)?\b|\bkid\b', 5, 12),
        (r'\bbaby\b|\binfant\b|\btoddler\b', 0, 3),
        (r'\byoung adult\b', 18, 30),
        (r'\b20s\b', 20, 29),
        (r'\b30s\b', 30, 39),
        (r'\b40s\b', 40, 49),
    ]
    for pattern, age_min, age_max in age_patterns:
        if re.search(pattern, tl):
            yr_start = max(birth_year + age_min, 1900)
            yr_end = min(birth_year + age_max, ref.year)
            return {"date_start": date(yr_start, 1, 1),
                    "date_end": date(yr_end, 12, 31),
                    "date_precision": "approx", "date_confidence": 0.40}

    # "when I was N years old"
    m = re.search(r'\bwhen\s+i\s+was\s+(\d+)\s+years?\s+old\b', tl)
    if m:
        age = int(m.group(1))
        yr = birth_year + age
        if 1900 <= yr <= ref.year:
            return {"date_start": date(yr, 1, 1), "date_end": date(yr, 12, 31),
                    "date_precision": "approx", "date_confidence": 0.50}

    return None


# ── Public entry point ──────────────────────────────────────────────────────

def normalize(
    date_raw: str,
    reference_date: date | None = None,
    patient_age: int | None = None,
    llm_resolver: Callable[[str], DateResult] | None = None,
) -> DateResult:
    """
    Normalize a raw date string to a structured DateResult.

    Priority: explicit → month+year → year-only → relative → age-relative → LLM → unknown.
    All rules are deterministic; llm_resolver is optional and never called in tests.
    """
    if not date_raw or not date_raw.strip():
        return unknown()

    ref = reference_date or date.today()
    text = date_raw.strip()

    result = (
        _try_explicit(text)
        or _try_month_year(text)
        or _try_year_only(text)
        or _try_relative(text, ref)
        or (patient_age is not None and _try_age_relative(text, patient_age, ref))
        or None
    )

    if result:
        return result

    if llm_resolver:
        try:
            return llm_resolver(date_raw)
        except Exception:
            pass

    return unknown()
