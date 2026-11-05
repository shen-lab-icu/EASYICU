"""Structural regression tests for batch-7 medications (2026-05-14).

Added: phenytoin, labetalol, esmolol, diltiazem, nicardipine.

Coverage after the 2026-05-27 audit:
  - phenytoin:   6/6 (eICU admissionDrug + HiRID added; AUMC 7222 added 2026-11-03)
  - labetalol:   6/6 (HiRID added)
  - esmolol:     6/6 (HiRID added; eICU is in infusionDrug not medication)
  - diltiazem:   6/6 (HiRID added)
  - nicardipine: 4/6 (no HiRID, no SIC)
"""
from __future__ import annotations

import json
import pathlib

import pytest

DICT = pathlib.Path(__file__).resolve().parents[2] / "src" / "easyicu" / "data" / "concept-dict.json"

# Boolean medication sources mark an administration TRUE; MIMIC-III
# prescriptions are placed at noon of their start day, and a HiRID zero
# givendose (pump start/stop marker) is not an administration (2026-11-04
# audit round 3, F-17/F-23).
BOOLEAN_DRUG_CALLBACKS = {
    "transform_fun(set_val(TRUE))",
    "mimic_prescription_noon",
    "transform_fun(comp_na(`>`, 0))",
    # HiRID doses above numerical noise (2026-11-04 audit round 4, R4-088).
    "transform_fun(comp_na(`>`, 0.000001))",
}
MAIN_DBS = {"miiv", "mimic", "eicu", "aumc", "hirid", "sic"}


BATCH7 = [
    ("phenytoin",   {"miiv", "mimic", "eicu", "aumc", "hirid", "sic"}, 6),
    ("labetalol",   {"miiv", "mimic", "eicu", "aumc", "hirid", "sic"}, 6),
    ("esmolol",     {"miiv", "mimic", "eicu", "aumc", "hirid", "sic"}, 6),
    # Round-11 ruling: diltiazem is IV exposure; HiRID has only tablets
    # (1001071/1000331), so R5C-HIRID-012 is withdrawn for them.
    ("diltiazem",   {"miiv", "mimic", "eicu", "aumc", "sic"},          5),
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
def test_sources_use_explicit_boolean_evidence(cdict, name, _dbs, _cov):
    for db, sources in cdict[name]["sources"].items():
        if isinstance(sources, dict):
            sources = [sources]
        for src in sources:
            # RB-05/07: numeric exposure is interpreted by its table owner.
            evidence = {
                "inputevents": "mimic_positive_dose", "inputevents_mv": "mimic_positive_dose",
                "inputevents_cv": "mimic_positive_dose", "drugitems": "aumc_positive_dose",
                "medication": "sicdb_exposure", "intakeoutput": "eicu_positive_value",
            }
            table = src.get("table")
            if table in evidence and (table != "medication" or db == "sic"):
                assert src.get("callback") == evidence[table]
                continue
            # R5-EICU-023/037: IV routes and positive infusion rates are explicit owners.
            if db in {"eicu", "eicu_demo"} and src.get("callback") in {"eicu_iv_exposure", "eicu_positive_infusion", "eicu_positive_value"}:
                expected_table = "infusiondrug" if src["callback"] == "eicu_positive_infusion" else "medication"
                assert src["table"] == expected_table
                if src["callback"] == "eicu_positive_value":
                    assert src["val_var"] == "dosage"
                continue
            if db == "sic" and name in ('esmolol', 'phenytoin'):
                # R5-SIC-026: observed Amount, preserving zero and missing.
                assert src.get("callback") == "sicdb_exposure"
                assert src.get("val_var") == "Amount"
                continue
            assert src.get("callback") in BOOLEAN_DRUG_CALLBACKS


@pytest.mark.parametrize("name,_dbs,_cov", BATCH7, ids=[m[0] for m in BATCH7])
def test_registered_as_point_event(name, _dbs, _cov):
    from easyicu.utils.compat import POINT_EVENT_CONCEPTS, WINDOW_CONCEPTS
    assert name in POINT_EVENT_CONCEPTS
    assert name not in WINDOW_CONCEPTS


@pytest.mark.parametrize("name,_dbs,expected_cov", BATCH7, ids=[m[0] for m in BATCH7])
def test_registered_in_webapp_catalog(name, _dbs, expected_cov):
    from easyicu.concept.catalog import (
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


def test_phenytoin_uses_eicu_in_icu_medication_and_aumc_drugitems(cdict):
    # 2026-11-03 audit 2: admissionDrug is the pre-admission list; in-ICU orders are medication HICLs.
    assert [s["table"] for s in cdict["phenytoin"]["sources"]["eicu"]] == ["medication"]
    # AUMC drugitems 7222 'Fenytoine (Diphantoine)'.
    assert cdict["phenytoin"]["sources"]["aumc"][0]["ids"] == [7222]
    assert cdict["phenytoin"]["sources"]["hirid"][0]["callback"] == "transform_fun(comp_na(`>`, 0.000001))"


def test_batch7_hirid_scope_matches_audit(cdict):
    assert "hirid" not in cdict["nicardipine"]["sources"]
    # Round-11 ruling: the IV-only concept leaves HiRID's tablets unmapped.
    assert "hirid" not in cdict["diltiazem"]["sources"]
    for c in ["phenytoin", "labetalol", "esmolol"]:
        assert cdict[c]["sources"]["hirid"][0]["callback"] == "transform_fun(comp_na(`>`, 0.000001))"
