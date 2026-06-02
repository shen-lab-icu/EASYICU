"""Structural regression tests for batch-4 medication concepts (2026-05-13).

Added: potassium_iv, magnesium_iv, albumin_iv, packed_rbc.

All are lgl_cncpt with set_val(TRUE). Coverage after the 2026-05-27 audit:
  - potassium_iv: 5/6 (no HiRID)
  - magnesium_iv: 6/6 (HiRID added)
  - albumin_iv:   4/6 (no HiRID, no AUMC)
  - packed_rbc:   5/6 (HiRID added; no eICU; eICU intake charted outside medication)

Specific design decisions enforced by tests:
  - packed_rbc has NO eICU source (intake charting is in intakeOutput, not
    medication; adding medication regex would produce false negatives).
  - albumin_iv has NO AUMC source (no matching drugitem found 2026-05-13).
  - potassium_iv includes K-phosphate (225925) because it IS IV potassium
    administration, not just KCl.
"""
from __future__ import annotations

import json
import pathlib

import pytest

DICT = pathlib.Path(__file__).resolve().parents[1] / "src" / "easyicu" / "data" / "concept-dict.json"
MAIN_DBS = {"miiv", "mimic", "eicu", "aumc", "hirid", "sic"}


BATCH4 = [
    ("potassium_iv", {"miiv", "mimic", "eicu", "aumc", "sic"}, 5),
    ("magnesium_iv", {"miiv", "mimic", "eicu", "aumc", "hirid", "sic"}, 6),
    ("albumin_iv",   {"miiv", "mimic", "eicu", "sic"},         4),
    ("packed_rbc",   {"miiv", "mimic", "aumc", "hirid", "sic"}, 5),
]


@pytest.fixture(scope="module")
def cdict():
    with DICT.open() as f:
        return json.load(f)


@pytest.mark.parametrize("name,_dbs,_cov", BATCH4, ids=[m[0] for m in BATCH4])
def test_is_lgl_cncpt(cdict, name, _dbs, _cov):
    entry = cdict[name]
    assert entry["class_name"] == "lgl_cncpt"
    assert entry["category"] == "medications"
    assert entry["aggregate"] == "max"


@pytest.mark.parametrize("name,required_dbs,_cov", BATCH4, ids=[m[0] for m in BATCH4])
def test_db_coverage(cdict, name, required_dbs, _cov):
    sources = set(cdict[name]["sources"].keys())
    main = sources & MAIN_DBS
    assert main == required_dbs, (
        f"{name} coverage drift: got {sorted(main)}, expected {sorted(required_dbs)}"
    )


@pytest.mark.parametrize("name,_dbs,_cov", BATCH4, ids=[m[0] for m in BATCH4])
def test_all_sources_use_set_val_true(cdict, name, _dbs, _cov):
    for db, sources in cdict[name]["sources"].items():
        if isinstance(sources, dict):
            sources = [sources]
        for src in sources:
            assert src.get("callback") == "transform_fun(set_val(TRUE))"


@pytest.mark.parametrize("name,_dbs,_cov", BATCH4, ids=[m[0] for m in BATCH4])
def test_registered_as_point_event(name, _dbs, _cov):
    from easyicu.compat import POINT_EVENT_CONCEPTS, WINDOW_CONCEPTS
    assert name in POINT_EVENT_CONCEPTS
    assert name not in WINDOW_CONCEPTS


@pytest.mark.parametrize("name,_dbs,expected_cov", BATCH4, ids=[m[0] for m in BATCH4])
def test_registered_in_webapp_catalog(name, _dbs, expected_cov):
    from easyicu.webapp.concept_catalog import (
        CONCEPT_DB_COVERAGE,
        CONCEPT_DICTIONARY,
    )
    assert name in CONCEPT_DICTIONARY
    assert CONCEPT_DB_COVERAGE[name] == expected_cov


# ── design contract tests ──
def test_potassium_iv_includes_both_kcl_and_kphos(cdict):
    """K-phosphate 225925 is IV potassium and must be included, alongside KCl
    225166 / 227522 / 227536."""
    miiv = cdict["potassium_iv"]["sources"]["miiv"]
    ids = set(miiv[0]["ids"] if isinstance(miiv, list) else miiv["ids"])
    assert {225166, 225925, 227522, 227536} <= ids, (
        f"potassium_iv must cover KCl + K-phos + bolus + CRRT, got {sorted(ids)}"
    )


def test_magnesium_iv_includes_obgyn_variant(cdict):
    """227524 (Magnesium Sulfate OB-GYN) is still IV magnesium — included."""
    miiv = cdict["magnesium_iv"]["sources"]["miiv"]
    ids = set(miiv[0]["ids"] if isinstance(miiv, list) else miiv["ids"])
    assert 227524 in ids


def test_albumin_iv_covers_all_concentrations(cdict):
    """All four albumin concentrations (4%, 5%, 20%, 25%) must be present."""
    miiv = cdict["albumin_iv"]["sources"]["miiv"]
    ids = set(miiv[0]["ids"] if isinstance(miiv, list) else miiv["ids"])
    assert ids == {220861, 220862, 220863, 220864}


def test_albumin_iv_has_no_aumc(cdict):
    """AUMC drugitems has no human albumin entry (as of 2026-05-13).
    Guard: someone adding AUMC without re-audit should trip this."""
    assert "aumc" not in cdict["albumin_iv"]["sources"]


def test_packed_rbc_has_no_eicu(cdict):
    """eICU medication table doesn't reliably capture blood product transfusion —
    those are in intakeOutput / hospital.infusionDrug. Adding a medication
    regex would yield false negatives. Documented-absent by design."""
    assert "eicu" not in cdict["packed_rbc"]["sources"]


def test_packed_rbc_includes_or_pacu_intake(cdict):
    """MIMIC-IV splits RBC into 225168 (primary), 226368 (OR intake),
    227070 (PACU intake). All three are RBC administration events."""
    miiv = cdict["packed_rbc"]["sources"]["miiv"]
    ids = set(miiv[0]["ids"] if isinstance(miiv, list) else miiv["ids"])
    assert {225168, 226368, 227070} <= ids


def test_batch4_hirid_scope_matches_audit(cdict):
    for c in ["potassium_iv", "albumin_iv"]:
        assert "hirid" not in cdict[c]["sources"], (
            f"{c}: HiRID source added without callback validation"
        )
    assert cdict["magnesium_iv"]["sources"]["hirid"][0]["ids"] == [1000421]
    assert cdict["packed_rbc"]["sources"]["hirid"][0]["ids"] == [1000100, 1000743]
    for c in ["magnesium_iv", "packed_rbc"]:
        assert cdict[c]["sources"]["hirid"][0]["callback"] == "transform_fun(set_val(TRUE))"
