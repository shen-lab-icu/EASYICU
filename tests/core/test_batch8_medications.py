"""Batch 8: anticoagulation/antiplatelet medications (2026-05-14).

Added: warfarin, apixaban, enoxaparin, aspirin.

Coverage notes:
  - apixaban in SIC and MIMIC-IV (2/6): MIMIC-IV reads administered eMAR
    events (2026-11-03 extraction audit 2); MIMIC-III and eICU 2014-15 record
    none, AUMC/HiRID have no entry.
  - aspirin 6/6 after the 2026-05-27 prescriptions/HiRID audit; MIIV/MIMIC
    still use prescriptions, not inputevents.
"""
from __future__ import annotations

import ast
import json
import pathlib

import pytest

DICT = pathlib.Path(__file__).resolve().parents[2] / "src" / "easyicu" / "data" / "concept-dict.json"
MAIN_DBS = {"miiv", "mimic", "eicu", "aumc", "hirid", "sic"}


# 2026-07-17: warfarin, enoxaparin and milrinone were redefined as pharmacological
# CLASSES (VKA / LMWH / PDE3-inhibitor), not single molecules, because the Dutch and
# Austrian hospitals stock a different class member than the Anglo drug we had mapped.
# That gained each an AUMC source (acenocoumarol / nadroparin / enoximone), so all three
# go 5/6 -> 6/6. See test_*_is_pharmacological_class below.
BATCH8 = [
    ("warfarin",   {"miiv", "mimic", "eicu", "aumc", "hirid", "sic"}, 6),
    ("apixaban",   {"sic", "miiv"},                  2),
    ("enoxaparin", {"miiv", "mimic", "eicu", "aumc", "hirid", "sic"}, 6),
    ("aspirin",    {"miiv", "mimic", "eicu", "aumc", "hirid", "sic"}, 6),
]


@pytest.fixture(scope="module")
def cdict():
    with DICT.open() as f:
        return json.load(f)


@pytest.mark.parametrize("name,_dbs,_cov", BATCH8, ids=[m[0] for m in BATCH8])
def test_is_lgl_cncpt(cdict, name, _dbs, _cov):
    entry = cdict[name]
    assert entry["class_name"] == "lgl_cncpt"
    assert entry["category"] == "medications"


@pytest.mark.parametrize("name,required_dbs,_cov", BATCH8, ids=[m[0] for m in BATCH8])
def test_db_coverage(cdict, name, required_dbs, _cov):
    sources = set(cdict[name]["sources"].keys())
    main = sources & MAIN_DBS
    assert main == required_dbs, (
        f"{name} drift: got {sorted(main)}, expected {sorted(required_dbs)}"
    )


@pytest.mark.parametrize("name,_dbs,_cov", BATCH8, ids=[m[0] for m in BATCH8])
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
            if db == "sic" and name in ('aspirin', 'enoxaparin', 'warfarin'):
                # R5-SIC-026: observed Amount, preserving zero and missing.
                assert src.get("callback") == "sicdb_exposure"
                assert src.get("val_var") == "Amount"
                continue
            # An eMAR source keeps given doses first, then marks them TRUE.
            # MIMIC-III prescriptions without a time sit at noon (audit round 2, D5).
            if src.get("table") == "emar":
                expected = "emar_administered"
            elif src.get("table") == "prescriptions" and db in ("mimic", "mimic_demo"):
                expected = "mimic_prescription_noon"
            elif src.get("table") == "prescriptions" and db == "miiv" and name == "aspirin":
                expected = "mimic_prescription_exposure"
                assert {"dose_val_rx", "stoptime"} <= set(src["extra_vars"])
            elif db == "hirid":
                # A zero givendose is a pump marker (audit round 3).
                expected = "transform_fun(comp_na(`>`, 0.000001))"
            elif db == "aumc" and name == "warfarin":
                # R5C-AUMC-027: a zero-dose order is not VKA administration.
                expected = "aumc_positive_dose"
            elif db in {"eicu", "eicu_demo"}:
                # RR-04: medication orders require positive dose text.
                expected = "eicu_positive_value"
                assert src["table"] == "medication" and src["val_var"] == "dosage"
            else:
                expected = "transform_fun(set_val(TRUE))"
            assert src.get("callback") == expected


@pytest.mark.parametrize("name,_dbs,_cov", BATCH8, ids=[m[0] for m in BATCH8])
def test_registered_as_point_event(name, _dbs, _cov):
    from easyicu.utils.compat import POINT_EVENT_CONCEPTS
    assert name in POINT_EVENT_CONCEPTS


@pytest.mark.parametrize("name,_dbs,expected_cov", BATCH8, ids=[m[0] for m in BATCH8])
def test_registered_in_webapp_catalog(name, _dbs, expected_cov):
    from easyicu.concept.catalog import (
        CONCEPT_DB_COVERAGE,
        CONCEPT_DICTIONARY,
    )
    assert name in CONCEPT_DICTIONARY
    assert CONCEPT_DB_COVERAGE[name] == expected_cov


def test_warfarin_sic_uses_vka_class(cdict):
    """2026-07-17: `warfarin` is the vitamin-K-antagonist CLASS, not the molecule.
    SIC stocks no true warfarin; it has phenprocoumon (1657, 36 pts) and acenocoumarol
    (1892, 23 pts) -- both coumarin VKAs, both now included."""
    sic = cdict["warfarin"]["sources"]["sic"]
    ids = set(sic[0]["ids"] if isinstance(sic, list) else sic["ids"])
    assert ids == {1657, 1892}


def test_apixaban_sic_and_miiv_emar(cdict):
    """apixaban: SIC d_references DrugID and MIMIC-IV administered eMAR events;
    downstream usage must be aware of the two-database scope."""
    sources = set(cdict["apixaban"]["sources"])
    assert sources == {"sic", "miiv"}
    assert cdict["apixaban"]["sources"]["miiv"][0]["table"] == "emar"


def test_enoxaparin_sic_includes_lmwh_class(cdict):
    """2026-07-17: `enoxaparin` is the LMWH CLASS. SIC has prophylactic (1536) and
    therapeutic (1923) enoxaparin, plus dalteparin (1534, 5,966 pts) -- the LMWH the
    site actually stocks -- now added."""
    sic = cdict["enoxaparin"]["sources"]["sic"]
    ids = set(sic[0]["ids"] if isinstance(sic, list) else sic["ids"])
    assert ids == {1536, 1923, 1534}


def test_aspirin_mimic_and_miiv_use_prescriptions(cdict):
    assert cdict["aspirin"]["sources"]["miiv"][0]["table"] == "prescriptions"
    assert cdict["aspirin"]["sources"]["mimic"][0]["table"] == "prescriptions"


def test_batch8_hirid_scope_matches_audit(cdict):
    assert "hirid" not in cdict["apixaban"]["sources"]
    for c in ["warfarin", "enoxaparin", "aspirin"]:
        assert cdict[c]["sources"]["hirid"][0]["callback"] == "transform_fun(comp_na(`>`, 0.000001))"


def test_anticoagulation_group_includes_batch8(cdict):
    """Verify load_medications anticoagulation group references all batch-8
    anticoagulants, not just heparin."""
    from easyicu.api import medications

    # Inspect string literals rather than depending on Black's choice of quote
    # style in the implementation.
    literals = {
        node.value
        for node in ast.walk(
            ast.parse(pathlib.Path(medications.__file__).read_text())
        )
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    assert {"warfarin", "apixaban", "enoxaparin"} <= literals
