"""Tests for composite outcome endpoints (easyicu.outcomes).

Most logic depends on per-database death-time tables, so the substantive
checks are real-data gated (``--run-real`` + ``EASYICU_DATA_PATH``):
they assert plausible mortality bands, horizon monotonicity, and that
databases without death follow-up return empty.
"""

from __future__ import annotations

import pytest


@pytest.mark.needs_real_data
@pytest.mark.parametrize(
    "database,lo28,hi28",
    [
        ("miiv", 0.08, 0.25),
        ("mimic", 0.06, 0.22),
        ("sic", 0.03, 0.15),
        ("aumc", 0.05, 0.20),
    ],
)
def test_horizon_mortality_plausible_and_monotonic(database, lo28, hi28):
    from easyicu.scores.outcomes import load_outcomes

    out = load_outcomes(database)
    assert not out.empty
    m28 = out["mort_28d"].mean()
    m90 = out["mort_90d"].mean()
    m365 = out["mort_365d"].mean()
    assert lo28 <= m28 <= hi28, f"{database} 28d mortality {m28:.3f} out of band"
    assert m28 <= m90 <= m365 + 1e-9, "horizons must be monotonic"
    assert "icu_free_days_28" not in out.columns


@pytest.mark.needs_real_data
@pytest.mark.parametrize("database", ["hirid"])
def test_no_followup_returns_empty(database):
    # eICU is excluded: it has no horizon mortality but DOES expose a native
    # ventilator-free-days endpoint (see test_eicu_ventilator_free_days).
    from easyicu.scores.outcomes import load_outcomes

    assert load_outcomes(database).empty


@pytest.mark.needs_real_data
def test_icu_readmission_is_unavailable_without_history_completeness():
    from easyicu.scores.outcomes import load_outcomes

    out = load_outcomes("miiv")
    assert "icu_readmission" not in out.columns


@pytest.mark.needs_real_data
def test_outcomes_via_load_concepts():
    from easyicu import load_concepts

    df = load_concepts(["mort_28d", "icu_free_days_28"], database="miiv")
    assert "mort_28d" in df.columns and "stay_id" in df.columns
    assert "icu_free_days_28" not in df.columns


@pytest.mark.needs_real_data
def test_eicu_ventilator_free_days_are_structurally_unavailable():
    from easyicu.scores.outcomes import load_outcomes

    out = load_outcomes("eicu")
    assert out.empty
    assert "vent_free_days_28" not in out.columns


@pytest.mark.needs_real_data
def test_vent_free_days_not_supported_for_mimic():
    # MIMIC mech_vent is too fragmented for a defensible VFD -> not exposed.
    from easyicu.scores.outcomes import load_outcomes

    out = load_outcomes("miiv")
    assert "vent_free_days_28" not in out.columns
