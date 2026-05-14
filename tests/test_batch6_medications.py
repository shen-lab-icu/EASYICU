"""Structural regression tests for batch-6 medications (2026-05-13).

Added: levetiracetam, dexamethasone, octreotide, neostigmine.

Coverage:
  - levetiracetam: 5/6 (no HiRID)
  - dexamethasone: 2/6 (eICU + SIC only — MIMIC dex is in chartevents/
                  prescriptions, not inputevents; AUMC drugitems regex no hit)
  - octreotide:    5/6 (no HiRID)
  - neostigmine:   4/6 (no HiRID, no MIMIC-III MV)
"""
from __future__ import annotations

import json
import pathlib

import pytest

DICT = pathlib.Path(__file__).resolve().parents[1] / "src" / "easyicu" / "data" / "concept-dict.json"
MAIN_DBS = {"miiv", "mimic", "eicu", "aumc", "hirid", "sic"}


BATCH6 = [
    ("levetiracetam", {"miiv", "mimic", "eicu", "aumc", "sic"}, 5),
    ("dexamethasone", {"eicu", "sic"},                          2),
    ("octreotide",    {"miiv", "mimic", "eicu", "aumc", "sic"}, 5),
    ("neostigmine",   {"miiv", "eicu", "aumc", "sic"},          4),
]


@pytest.fixture(scope="module")
def cdict():
    with DICT.open() as f:
        return json.load(f)


@pytest.mark.parametrize("name,_dbs,_cov", BATCH6, ids=[m[0] for m in BATCH6])
def test_is_lgl_cncpt(cdict, name, _dbs, _cov):
    entry = cdict[name]
    assert entry["class_name"] == "lgl_cncpt"
    assert entry["category"] == "medications"
    assert entry["aggregate"] == "max"


@pytest.mark.parametrize("name,required_dbs,_cov", BATCH6, ids=[m[0] for m in BATCH6])
def test_db_coverage(cdict, name, required_dbs, _cov):
    sources = set(cdict[name]["sources"].keys())
    main = sources & MAIN_DBS
    assert main == required_dbs, (
        f"{name} coverage drift: got {sorted(main)}, expected {sorted(required_dbs)}"
    )


@pytest.mark.parametrize("name,_dbs,_cov", BATCH6, ids=[m[0] for m in BATCH6])
def test_all_sources_use_set_val_true(cdict, name, _dbs, _cov):
    for db, sources in cdict[name]["sources"].items():
        if isinstance(sources, dict):
            sources = [sources]
        for src in sources:
            assert src.get("callback") == "transform_fun(set_val(TRUE))"


@pytest.mark.parametrize("name,_dbs,_cov", BATCH6, ids=[m[0] for m in BATCH6])
def test_registered_as_point_event(name, _dbs, _cov):
    from easyicu.compat import POINT_EVENT_CONCEPTS, WINDOW_CONCEPTS
    assert name in POINT_EVENT_CONCEPTS
    assert name not in WINDOW_CONCEPTS


@pytest.mark.parametrize("name,_dbs,expected_cov", BATCH6, ids=[m[0] for m in BATCH6])
def test_registered_in_webapp_catalog(name, _dbs, expected_cov):
    from easyicu.webapp.concept_catalog import (
        CONCEPT_DB_COVERAGE,
        CONCEPT_DICTIONARY,
    )
    assert name in CONCEPT_DICTIONARY
    assert CONCEPT_DB_COVERAGE[name] == expected_cov


# ── design contracts ──
def test_dexamethasone_excludes_mimic_inputevents(cdict):
    """MIMIC-III CV has dex eye/nasal drops (chartevents); MIMIC-IV's
    inputevents has no dex itemid — systemic dex is in prescriptions which
    is outside inputevents-based concept extraction."""
    sources = set(cdict["dexamethasone"]["sources"])
    assert "miiv" not in sources, (
        "dexamethasone has no MIIV inputevents itemid — adding one without "
        "verifying the systemic vs topical route would silently mix concepts"
    )
    assert "mimic" not in sources
    assert "aumc" not in sources, (
        "AUMC drugitems regex returned no matches; do not add without "
        "explicit itemid lookup"
    )


def test_neostigmine_no_mimic_iii_mv(cdict):
    """MIMIC-III MetaVision inputevents has no neostigmine itemid (only
    MIMIC-IV adds 229071). Guard rail."""
    mimic_sources = cdict["neostigmine"]["sources"].get("mimic", [])
    assert mimic_sources == [], "MIMIC-III neostigmine source must remain empty"


def test_levetiracetam_includes_miiv_and_mimic_mv(cdict):
    """itemid 228316 is shared between MIIV and MIMIC-III MV. Both must be
    populated."""
    assert cdict["levetiracetam"]["sources"]["miiv"][0]["ids"] == [228316]
    mv_src = [s for s in cdict["levetiracetam"]["sources"]["mimic"]
              if s["table"] == "inputevents_mv"][0]
    assert mv_src["ids"] == [228316]


def test_octreotide_consistent_across_miiv_mimic(cdict):
    """itemid 225155 is the canonical octreotide MetaVision id."""
    assert cdict["octreotide"]["sources"]["miiv"][0]["ids"] == [225155]
    mv_src = [s for s in cdict["octreotide"]["sources"]["mimic"]
              if s["table"] == "inputevents_mv"][0]
    assert mv_src["ids"] == [225155]


def test_batch6_all_hirid_absent(cdict):
    """HiRID pharma reference has no matches for these drugs."""
    for c in ["levetiracetam", "dexamethasone", "octreotide", "neostigmine"]:
        assert "hirid" not in cdict[c]["sources"]
