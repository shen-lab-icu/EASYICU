"""Structural regression tests for batch-7 medications (2026-05-14).

Added: phenytoin, labetalol, esmolol, diltiazem, nicardipine.

Coverage:
  - phenytoin:   3/6 (MIIV+MIMIC+SIC; AUMC has no entry, eICU only PO)
  - labetalol:   5/6 (no HiRID)
  - esmolol:     5/6 (no HiRID; eICU is in infusionDrug not medication)
  - diltiazem:   5/6 (no HiRID)
  - nicardipine: 4/6 (no HiRID, no SIC)
"""
from __future__ import annotations

import json
import pathlib

import pytest

DICT = pathlib.Path(__file__).resolve().parents[1] / "src" / "easyicu" / "data" / "concept-dict.json"
MAIN_DBS = {"miiv", "mimic", "eicu", "aumc", "hirid", "sic"}


BATCH7 = [
    ("phenytoin",   {"miiv", "mimic", "sic"},                  3),
    ("labetalol",   {"miiv", "mimic", "eicu", "aumc", "sic"},  5),
    ("esmolol",     {"miiv", "mimic", "eicu", "aumc", "sic"},  5),
    ("diltiazem",   {"miiv", "mimic", "eicu", "aumc", "sic"},  5),
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


def test_phenytoin_excludes_aumc_eicu(cdict):
    """AUMC drugitems has no phenytoin entry; eICU has only PO tabs
    (which we deliberately exclude from systemic-administration concepts)."""
    sources = set(cdict["phenytoin"]["sources"])
    assert "aumc" not in sources
    assert "eicu" not in sources


def test_batch7_all_hirid_absent(cdict):
    """HiRID has no pharma reference for any of these drugs."""
    for c in ["phenytoin", "labetalol", "esmolol", "diltiazem", "nicardipine"]:
        assert "hirid" not in cdict[c]["sources"]
