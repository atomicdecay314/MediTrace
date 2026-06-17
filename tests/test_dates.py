"""Unit tests for services/dates.py — no LLM, no DB."""
from datetime import date
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from services.dates import normalize

REF = date(2025, 6, 17)


# ── Explicit full dates ──────────────────────────────────────────────────────

def test_dmy_slash():
    r = normalize("13/7/2021", REF)
    assert r["date_start"] == date(2021, 7, 13)
    assert r["date_precision"] == "exact"
    assert r["date_confidence"] >= 0.9

def test_iso_date():
    r = normalize("2021-07-13", REF)
    assert r["date_start"] == date(2021, 7, 13)
    assert r["date_precision"] == "exact"

def test_dmy_text_month():
    r = normalize("13 July 2021", REF)
    assert r["date_start"] == date(2021, 7, 13)
    assert r["date_precision"] == "exact"

def test_dmy_abbr_2digit_year():
    r = normalize("14 Jun 26", REF)
    assert r["date_start"] == date(2026, 6, 14)
    assert r["date_precision"] == "exact"

def test_mdy_text_month():
    r = normalize("July 13 2021", REF)
    assert r["date_start"] == date(2021, 7, 13)
    assert r["date_precision"] == "exact"

def test_ddmm_first_gt12_forces_day():
    # 15/3/2020 — first component 15 > 12, must be day
    r = normalize("15/3/2020", REF)
    assert r["date_start"] == date(2020, 3, 15)
    assert r["date_precision"] == "exact"

def test_ddmm_default_ambiguous():
    # 3/7/2020 — default DD/MM → 3rd July (not 7th March)
    r = normalize("3/7/2020", REF)
    assert r["date_start"] == date(2020, 7, 3)

def test_dmy_dot_separator():
    r = normalize("5.11.2023", REF)
    assert r["date_start"] == date(2023, 11, 5)


# ── Month + year ─────────────────────────────────────────────────────────────

def test_month_year_full():
    r = normalize("March 2022", REF)
    assert r["date_start"] == date(2022, 3, 1)
    assert r["date_end"] == date(2022, 3, 31)
    assert r["date_precision"] == "month"

def test_month_abbr_4digit_year():
    r = normalize("Jun 2026", REF)
    assert r["date_start"] == date(2026, 6, 1)
    assert r["date_precision"] == "month"

def test_month_abbr_2digit_year():
    # "Jun 26" → June 2026 (month+year, not June 26th)
    r = normalize("Jun 26", REF)
    assert r["date_start"] == date(2026, 6, 1)
    assert r["date_end"] == date(2026, 6, 30)
    assert r["date_precision"] == "month"

def test_early_year():
    r = normalize("early 2020", REF)
    assert r["date_start"] == date(2020, 1, 1)
    assert r["date_end"].year == 2020
    assert r["date_end"].month <= 4
    assert r["date_precision"] == "month"

def test_mid_year():
    r = normalize("mid 2020", REF)
    assert r["date_start"] == date(2020, 5, 1)
    assert r["date_end"].month >= 8

def test_late_year():
    r = normalize("late 2020", REF)
    assert r["date_start"] == date(2020, 9, 1)
    assert r["date_end"] == date(2020, 12, 31)


# ── Year only ────────────────────────────────────────────────────────────────

def test_year_only():
    r = normalize("2019", REF)
    assert r["date_start"] == date(2019, 1, 1)
    assert r["date_end"] == date(2019, 12, 31)
    assert r["date_precision"] == "year"
    assert r["date_confidence"] == pytest.approx(0.60)

def test_in_year():
    r = normalize("in 2019", REF)
    assert r["date_start"] == date(2019, 1, 1)
    assert r["date_precision"] == "year"


# ── Relative expressions ─────────────────────────────────────────────────────

def test_yesterday():
    r = normalize("yesterday", REF)
    assert r["date_start"] == date(2025, 6, 16)
    assert r["date_precision"] == "exact"

def test_last_week():
    r = normalize("last week", REF)
    assert r["date_precision"] == "approx"
    assert r["date_start"] < REF

def test_last_month():
    r = normalize("last month", REF)
    assert r["date_start"] == date(2025, 5, 1)
    assert r["date_end"] == date(2025, 5, 31)
    assert r["date_precision"] == "month"

def test_last_year():
    r = normalize("last year", REF)
    assert r["date_start"] == date(2024, 1, 1)
    assert r["date_end"] == date(2024, 12, 31)
    assert r["date_precision"] == "year"

def test_last_named_month_past():
    # ref = June 2025; "last March" → March 2025
    r = normalize("last March", REF)
    assert r["date_start"] == date(2025, 3, 1)
    assert r["date_end"] == date(2025, 3, 31)
    assert r["date_precision"] == "month"

def test_last_named_month_wraps_year():
    # ref = February 2025; "last March" → March 2024
    r = normalize("last March", date(2025, 2, 10))
    assert r["date_start"] == date(2024, 3, 1)

def test_n_years_ago():
    r = normalize("2 years ago", REF)
    assert r["date_start"] == date(2023, 6, 17)
    assert r["date_precision"] == "approx"

def test_n_months_ago():
    r = normalize("3 months ago", REF)
    assert r["date_start"] == date(2025, 3, 17)
    assert r["date_precision"] == "approx"

def test_n_days_ago():
    r = normalize("5 days ago", REF)
    assert r["date_start"] == date(2025, 6, 12)
    assert r["date_precision"] == "exact"

def test_a_few_years_ago():
    r = normalize("a few years ago", REF)
    assert r["date_start"] == date(2021, 1, 1)
    assert r["date_end"] == date(2023, 12, 31)
    assert r["date_precision"] == "approx"
    assert r["date_confidence"] <= 0.35

def test_several_years_ago():
    r = normalize("several years ago", REF)
    assert r["date_start"].year <= 2022
    assert r["date_precision"] == "approx"

def test_few_months_ago():
    r = normalize("a few months ago", REF)
    assert r["date_precision"] == "approx"
    assert r["date_start"] < REF

def test_recently():
    r = normalize("recently diagnosed", REF)
    assert r["date_precision"] == "approx"
    assert r["date_start"] < REF


# ── Age-relative ─────────────────────────────────────────────────────────────

def test_age_teenager():
    # patient_age=35, ref=2025 → birth_year=1990; teenager=13-19 → 2003-2009
    r = normalize("when I was a teenager", REF, patient_age=35)
    assert r["date_start"] == date(2003, 1, 1)
    assert r["date_end"] == date(2009, 12, 31)
    assert r["date_precision"] == "approx"
    assert r["date_confidence"] == pytest.approx(0.40)

def test_age_child():
    r = normalize("as a child", REF, patient_age=40)
    assert r["date_start"].year == 1985 + 5   # birth 1985 + 5
    assert r["date_precision"] == "approx"

def test_age_relative_no_age_returns_unknown():
    # Without patient_age, age-relative falls through to unknown
    r = normalize("when I was a teenager", REF)
    assert r["date_precision"] == "unknown"


# ── Unknown / empty ───────────────────────────────────────────────────────────

def test_empty_string():
    r = normalize("")
    assert r["date_start"] is None
    assert r["date_precision"] == "unknown"
    assert r["date_confidence"] == 0.0

def test_none_like():
    r = normalize("   ")
    assert r["date_precision"] == "unknown"

def test_garbage():
    r = normalize("xyz not a date at all")
    assert r["date_precision"] == "unknown"

def test_no_llm_fallback_is_safe():
    # Confirm unknown result is returned without crashing when no llm_resolver
    r = normalize("some unparseable text 🎲")
    assert r["date_precision"] == "unknown"
    assert r["date_start"] is None
