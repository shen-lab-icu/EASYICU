"""Tests for microbiology culture-positivity + extra lab overlay concepts.

Structural checks run offline; value checks are real-data gated.
"""

from __future__ import annotations

import pytest


@pytest.mark.needs_real_data
@pytest.mark.parametrize("database,lab,lo,hi", [
    ("miiv", "ldh", 50, 1000),       # plausible median band
    ("miiv", "osmolality", 250, 330),
    ("eicu", "ldh", 50, 1000),
    ("sic", "ferritin", 50, 2000),
    ("aumc", "lipase", 5, 500),
])
def test_extra_lab_median_plausible(database, lab, lo, hi):
    from easyicu import load_concepts

    df = load_concepts([lab], database=database)
    vals = df[lab].dropna()
    assert len(vals) > 0
    assert lo <= vals.median() <= hi


@pytest.mark.needs_real_data
@pytest.mark.parametrize("database", ["miiv", "mimic", "eicu"])
def test_microbiology_positivity_bands(database):
    from easyicu.microbiology import load_microbiology

    out = load_microbiology(database)
    assert not out.empty
    assert "culture_positive" in out.columns
    assert "bld_culture_positive" in out.columns
    cp = out["culture_positive"].mean()
    bp = out["bld_culture_positive"].mean()
    assert 0.0 < cp <= 1.0
    assert bp <= cp + 1e-9          # blood-positive is a subset of any-positive


@pytest.mark.needs_real_data
@pytest.mark.parametrize("database", ["sic", "hirid", "aumc"])
def test_microbiology_na_databases(database):
    from easyicu.microbiology import load_microbiology

    assert load_microbiology(database).empty


@pytest.mark.needs_real_data
def test_micro_via_load_concepts():
    from easyicu import load_concepts

    df = load_concepts(["bld_culture_positive"], database="miiv")
    assert "bld_culture_positive" in df.columns
