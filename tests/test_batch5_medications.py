"""Structural regression tests for batch-5 medication concepts (2026-05-13).

Added: bicarbonate, dextrose50, ffp, platelets.

Coverage after the 2026-05-27 audit:
  - bicarbonate: 6/6 (HiRID/SIC added)
  - dextrose50:  5/6 (HiRID added; no AUMC; distinct from the dex rec_cncpt)
  - ffp:         5/6 (HiRID/AUMC added; no SIC)
  - platelets:   5/6 (HiRID added; no eICU)

Design decisions enforced by tests:
  - ``dextrose50`` is separate from existing ``dex`` (rec_cncpt for D10);
    this concept targets the D50 hypoglycemia-rescue push specifically.
  - ``bicarbonate`` AUMC whitelist excludes the capsule form (9104) and
    K-bicarbonate 12769 (not Na).
  - ``platelets`` has NO eICU (consistent with packed_rbc).
"""
from __future__ import annotations

import json
import pathlib

import pytest

DICT = pathlib.Path(__file__).resolve().parents[1] / "src" / "easyicu" / "data" / "concept-dict.json"
MAIN_DBS = {"miiv", "mimic", "eicu", "aumc", "hirid", "sic"}


BATCH5 = [
    ("bicarbonate", {"miiv", "mimic", "eicu", "aumc", "hirid", "sic"}, 6),
    ("dextrose50",  {"miiv", "mimic", "eicu", "hirid", "sic"},         5),
    ("ffp",         {"miiv", "mimic", "eicu", "aumc", "hirid"},        5),
    ("platelets",   {"miiv", "mimic", "aumc", "hirid", "sic"},         5),
]


@pytest.fixture(scope="module")
def cdict():
    with DICT.open() as f:
        return json.load(f)


@pytest.mark.parametrize("name,_dbs,_cov", BATCH5, ids=[m[0] for m in BATCH5])
def test_is_lgl_cncpt(cdict, name, _dbs, _cov):
    entry = cdict[name]
    assert entry["class_name"] == "lgl_cncpt"
    assert entry["category"] == "medications"
    assert entry["aggregate"] == "max"


@pytest.mark.parametrize("name,required_dbs,_cov", BATCH5, ids=[m[0] for m in BATCH5])
def test_db_coverage(cdict, name, required_dbs, _cov):
    sources = set(cdict[name]["sources"].keys())
    main = sources & MAIN_DBS
    assert main == required_dbs, (
        f"{name} coverage drift: got {sorted(main)}, expected {sorted(required_dbs)}"
    )


@pytest.mark.parametrize("name,_dbs,_cov", BATCH5, ids=[m[0] for m in BATCH5])
def test_all_sources_use_set_val_true(cdict, name, _dbs, _cov):
    for db, sources in cdict[name]["sources"].items():
        if isinstance(sources, dict):
            sources = [sources]
        for src in sources:
            assert src.get("callback") == "transform_fun(set_val(TRUE))"


@pytest.mark.parametrize("name,_dbs,_cov", BATCH5, ids=[m[0] for m in BATCH5])
def test_registered_as_point_event(name, _dbs, _cov):
    from easyicu.compat import POINT_EVENT_CONCEPTS, WINDOW_CONCEPTS
    assert name in POINT_EVENT_CONCEPTS
    assert name not in WINDOW_CONCEPTS


@pytest.mark.parametrize("name,_dbs,expected_cov", BATCH5, ids=[m[0] for m in BATCH5])
def test_registered_in_webapp_catalog(name, _dbs, expected_cov):
    from easyicu.concept_catalog import (
        CONCEPT_DB_COVERAGE,
        CONCEPT_DICTIONARY,
    )
    assert name in CONCEPT_DICTIONARY
    assert CONCEPT_DB_COVERAGE[name] == expected_cov


# ── design contracts ──
def test_bicarbonate_aumc_excludes_capsule_and_k(cdict):
    """AUMC drugitems has oral capsule (9104) and K-bicarbonate 5% (12769).
    Neither is IV sodium bicarbonate — both must stay out."""
    aumc = cdict["bicarbonate"]["sources"]["aumc"]
    ids = set(aumc[0]["ids"] if isinstance(aumc, list) else aumc["ids"])
    assert 9104 not in ids, "oral capsule must stay excluded"
    assert 12769 not in ids, "K-bicarbonate must stay excluded"
    # Core Na-bicarbonate IV forms present
    assert {7295, 8998, 19932, 8936} <= ids


def test_dextrose50_distinct_from_dex_rec(cdict):
    """Ensure dextrose50 (D50 rescue push) coexists with the existing
    ``dex`` concept (D10 drip tracked as a numeric rate) — the two must
    stay separate, since they serve different clinical questions."""
    assert "dextrose50" in cdict
    assert "dex" in cdict
    # dex is a D10-rate concept (unt_cncpt/num_cncpt), NOT the same shape
    # as the boolean lgl_cncpt dextrose50.
    assert cdict["dex"].get("class_name") != "lgl_cncpt"
    assert cdict["dextrose50"]["class_name"] == "lgl_cncpt"
    # Their DB source shapes must not collide either.
    assert "unit" in cdict["dex"]  # dex carries a unit declaration
    dex_unit = cdict["dex"]["unit"]
    if isinstance(dex_unit, list):
        assert "ml/hr" in dex_unit  # D10 is a rate concept
    else:
        assert dex_unit == "ml/hr"


def test_ffp_aumc_added_but_sic_still_absent(cdict):
    """AUMC FFP was added by the 2026-05-27 audit; SIC remains absent."""
    sources = set(cdict["ffp"]["sources"])
    assert cdict["ffp"]["sources"]["aumc"][0]["ids"] == [7367]
    assert "sic" not in sources


def test_platelets_has_no_eicu(cdict):
    """Consistent with packed_rbc — platelet transfusions are in blood bank
    records, not medication table."""
    assert "eicu" not in cdict["platelets"]["sources"]


def test_batch5_hirid_sources_use_boolean_callback(cdict):
    for c in ["bicarbonate", "dextrose50", "ffp", "platelets"]:
        hirid = cdict[c]["sources"]["hirid"][0]
        assert hirid["callback"] == "transform_fun(set_val(TRUE))"
