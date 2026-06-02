"""Structural regression tests for batch-7 medications (2026-05-14).

Added: phenytoin, labetalol, esmolol, diltiazem, nicardipine.

Coverage after the 2026-05-27 audit:
  - phenytoin:   5/6 (eICU admissionDrug + HiRID added; AUMC still absent)
  - labetalol:   6/6 (HiRID added)
  - esmolol:     6/6 (HiRID added; eICU is in infusionDrug not medication)
  - diltiazem:   6/6 (HiRID added)
  - nicardipine: 4/6 (no HiRID, no SIC)
"""
from __future__ import annotations

import json
import pathlib

import pytest

DICT = pathlib.Path(__file__).resolve().parents[1] / "src" / "easyicu" / "data" / "concept-dict.json"
MAIN_DBS = {"miiv", "mimic", "eicu", "aumc", "hirid", "sic"}


BATCH7 = [
    ("phenytoin",   {"miiv", "mimic", "eicu", "hirid", "sic"},        5),
    ("labetalol",   {"miiv", "mimic", "eicu", "aumc", "hirid", "sic"}, 6),
    ("esmolol",     {"miiv", "mimic", "eicu", "aumc", "hirid", "sic"}, 6),
    ("diltiazem",   {"miiv", "mimic", "eicu", "aumc", "hirid", "sic"}, 6),
    ("nicardipine", {"miiv", "mimic", "eicu", "aumc"},         4),
]


@pytest.fixture(scope="module")
def cdict():
    with DICT.open() as f:
        return json.load(f)


@pytest.mark.parametrize("name,_dbs,_cov", BATCH7, ids=[m[0] for m in BATCH7])
def test_is_lgl_cncpt(cdict, name, _dbs, _cov):
    entry = cdict[name]
    assert entry["class_name"] == "lgl_cncpt"
    assert entry["category"] == "medications"
    assert entry["aggregate"] == "max"


@pytest.mark.parametrize("name,required_dbs,_cov", BATCH7, ids=[m[0] for m in BATCH7])
def test_db_coverage(cdict, name, required_dbs, _cov):
    sources = set(cdict[name]["sources"].keys())
    main = sources & MAIN_DBS
    assert main == required_dbs, (
        f"{name} coverage drift: got {sorted(main)}, expected {sorted(required_dbs)}"
    )


@pytest.mark.parametrize("name,_dbs,_cov", BATCH7, ids=[m[0] for m in BATCH7])
def test_all_sources_use_set_val_true(cdict, name, _dbs, _cov):
    for db, sources in cdict[name]["sources"].items():
        if isinstance(sources, dict):
            sources = [sources]
        for src in sources:
            assert src.get("callback") == "transform_fun(set_val(TRUE))"


@pytest.mark.parametrize("name,_dbs,_cov", BATCH7, ids=[m[0] for m in BATCH7])
def test_registered_as_point_event(name, _dbs, _cov):
    from easyicu.compat import POINT_EVENT_CONCEPTS, WINDOW_CONCEPTS
    assert name in POINT_EVENT_CONCEPTS
    assert name not in WINDOW_CONCEPTS


@pytest.mark.parametrize("name,_dbs,expected_cov", BATCH7, ids=[m[0] for m in BATCH7])
def test_registered_in_webapp_catalog(name, _dbs, expected_cov):
    from easyicu.webapp.concept_catalog import (
        CONCEPT_DB_COVERAGE,
        CONCEPT_DICTIONARY,
    )
    assert name in CONCEPT_DICTIONARY
    assert CONCEPT_DB_COVERAGE[name] == expected_cov


def test_phenytoin_includes_fosphenytoin(cdict):
    """Both phenytoin (227689 Dilantin) and fosphenytoin (227690) MV ids
    must be present — they're clinically equivalent prodrug pair."""
    assert set(cdict["phenytoin"]["sources"]["miiv"][0]["ids"]) == {227689, 227690}


def test_esmolol_uses_eicu_infusiondrug_not_medication(cdict):
    """eICU charts esmolol as continuous infusion in infusionDrug,
    not in the medication table."""
    src = cdict["esmolol"]["sources"]["eicu"][0]
    assert src["table"] == "infusiondrug"


def test_nicardipine_excludes_sic(cdict):
    """SIC d_references has no nicardipine entry as of 2026-05-14."""
    assert "sic" not in cdict["nicardipine"]["sources"]


def test_phenytoin_uses_eicu_admissiondrug_but_excludes_aumc(cdict):
    sources = set(cdict["phenytoin"]["sources"])
    assert "aumc" not in sources
    assert cdict["phenytoin"]["sources"]["eicu"][0]["table"] == "admissiondrug"
    assert cdict["phenytoin"]["sources"]["hirid"][0]["callback"] == "transform_fun(set_val(TRUE))"


def test_batch7_hirid_scope_matches_audit(cdict):
    assert "hirid" not in cdict["nicardipine"]["sources"]
    for c in ["phenytoin", "labetalol", "esmolol", "diltiazem"]:
        assert cdict[c]["sources"]["hirid"][0]["callback"] == "transform_fun(set_val(TRUE))"
