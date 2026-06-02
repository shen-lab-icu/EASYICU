"""Structural regression tests for batch-6 medications (2026-05-13).

Added: levetiracetam, dexamethasone, octreotide, neostigmine.

Coverage after the 2026-05-27 audit:
  - levetiracetam: 6/6 (HiRID added)
  - dexamethasone: 6/6 (prescriptions/AUMC/HiRID added; still no inputevents)
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
    ("levetiracetam", {"miiv", "mimic", "eicu", "aumc", "hirid", "sic"}, 6),
    ("dexamethasone", {"miiv", "mimic", "eicu", "aumc", "hirid", "sic"}, 6),
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
def test_dexamethasone_uses_prescriptions_not_inputevents(cdict):
    """MIMIC-III CV has dex eye/nasal drops (chartevents); MIMIC-IV's
    inputevents has no dex itemid — systemic dex is in prescriptions which
    is outside inputevents-based concept extraction."""
    sources = set(cdict["dexamethasone"]["sources"])
    assert sources >= {"miiv", "mimic", "aumc", "hirid"}
    assert cdict["dexamethasone"]["sources"]["miiv"][0]["table"] == "prescriptions"
    assert cdict["dexamethasone"]["sources"]["mimic"][0]["table"] == "prescriptions"
    assert cdict["dexamethasone"]["sources"]["aumc"][0]["ids"] == [6995]
    assert cdict["dexamethasone"]["sources"]["hirid"][0]["callback"] == "transform_fun(set_val(TRUE))"


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


def test_batch6_hirid_scope_matches_audit(cdict):
    for c in ["octreotide", "neostigmine"]:
        assert "hirid" not in cdict[c]["sources"]
    for c in ["levetiracetam", "dexamethasone"]:
        assert cdict[c]["sources"]["hirid"][0]["callback"] == "transform_fun(set_val(TRUE))"
