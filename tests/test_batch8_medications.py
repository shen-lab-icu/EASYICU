"""Batch 8: anticoagulation/antiplatelet medications (2026-05-14).

Added: warfarin, apixaban, enoxaparin, aspirin.

Coverage notes:
  - apixaban only in SIC (1/6) — MIMIC/AUMC/eICU all use prescriptions/PO charting
    that's outside inputevents/medication scope; HiRID has no entry.
  - aspirin 6/6 after the 2026-05-27 prescriptions/HiRID audit; MIIV/MIMIC
    still use prescriptions, not inputevents.
"""
from __future__ import annotations

import ast
import json
import pathlib

import pytest

DICT = pathlib.Path(__file__).resolve().parents[1] / "src" / "easyicu" / "data" / "concept-dict.json"
MAIN_DBS = {"miiv", "mimic", "eicu", "aumc", "hirid", "sic"}


# 2026-07-17: warfarin, enoxaparin and milrinone were redefined as pharmacological
# CLASSES (VKA / LMWH / PDE3-inhibitor), not single molecules, because the Dutch and
# Austrian hospitals stock a different class member than the Anglo drug we had mapped.
# That gained each an AUMC source (acenocoumarol / nadroparin / enoximone), so all three
# go 5/6 -> 6/6. See test_*_is_pharmacological_class below.
BATCH8 = [
    ("warfarin",   {"miiv", "mimic", "eicu", "aumc", "hirid", "sic"}, 6),
    ("apixaban",   {"sic"},                          1),
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
def test_all_sources_use_set_val_true(cdict, name, _dbs, _cov):
    for db, sources in cdict[name]["sources"].items():
        if isinstance(sources, dict):
            sources = [sources]
        for src in sources:
            assert src.get("callback") == "transform_fun(set_val(TRUE))"


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


def test_apixaban_sic_only(cdict):
    """apixaban only available in SIC d_references; downstream usage must
    be aware of single-DB scope."""
    sources = set(cdict["apixaban"]["sources"])
    assert sources == {"sic"}


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
        assert cdict[c]["sources"]["hirid"][0]["callback"] == "transform_fun(set_val(TRUE))"


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
