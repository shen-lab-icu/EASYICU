"""Structural regression tests for batch-2 medication concepts (2026-05-13).

Added: lorazepam, ketamine, vecuronium, cisatracurium, nitroglycerin.

All are lgl_cncpt with set_val(TRUE) callbacks (no rate extraction). The
2026-05-27 audit added verified HiRID entries for lorazepam, ketamine,
vecuronium, and nitroglycerin; vecuronium still lacks a SIC reference and
cisatracurium still lacks AUMC/HiRID entries.

Ground-truth IDs and regex patterns verified against local reference tables
on 2026-05-13.
"""
from __future__ import annotations

import json
import pathlib

import pytest

DICT = pathlib.Path(__file__).resolve().parents[1] / "src" / "easyicu" / "data" / "concept-dict.json"
MAIN_DBS = {"miiv", "mimic", "eicu", "aumc", "hirid", "sic"}


BATCH2 = [
    # (name, required_dbs, expected_coverage)
    ("lorazepam",     {"miiv", "mimic", "eicu", "aumc", "hirid", "sic"}, 6),
    ("ketamine",      {"miiv", "mimic", "eicu", "aumc", "hirid", "sic"}, 6),
    ("vecuronium",    {"miiv", "mimic", "eicu", "aumc", "hirid"},        5),
    ("cisatracurium", {"miiv", "mimic", "eicu", "sic"},              4),
    ("nitroglycerin", {"miiv", "mimic", "eicu", "aumc", "hirid", "sic"}, 6),
]


@pytest.fixture(scope="module")
def cdict():
    with DICT.open() as f:
        return json.load(f)


@pytest.mark.parametrize("name,_dbs,_cov", BATCH2, ids=[m[0] for m in BATCH2])
def test_is_lgl_cncpt(cdict, name, _dbs, _cov):
    assert name in cdict
    entry = cdict[name]
    assert entry["class_name"] == "lgl_cncpt"
    assert entry["category"] == "medications"
    assert entry["aggregate"] == "max"


@pytest.mark.parametrize("name,required_dbs,_cov", BATCH2, ids=[m[0] for m in BATCH2])
def test_db_coverage(cdict, name, required_dbs, _cov):
    sources = set(cdict[name]["sources"].keys())
    main = sources & MAIN_DBS
    assert main == required_dbs, (
        f"{name} main-DB coverage changed: got {sorted(main)}, expected {sorted(required_dbs)}"
    )


@pytest.mark.parametrize("name,_dbs,_cov", BATCH2, ids=[m[0] for m in BATCH2])
def test_all_sources_use_set_val_true(cdict, name, _dbs, _cov):
    for db, sources in cdict[name]["sources"].items():
        if isinstance(sources, dict):
            sources = [sources]
        for src in sources:
            assert src.get("callback") == "transform_fun(set_val(TRUE))", (
                f"{name}/{db}: callback must be set_val(TRUE), got {src.get('callback')}"
            )


@pytest.mark.parametrize("name,_dbs,_cov", BATCH2, ids=[m[0] for m in BATCH2])
def test_registered_as_point_event(name, _dbs, _cov):
    from easyicu.compat import POINT_EVENT_CONCEPTS, WINDOW_CONCEPTS
    assert name in POINT_EVENT_CONCEPTS
    assert name not in WINDOW_CONCEPTS


@pytest.mark.parametrize("name,_dbs,expected_cov", BATCH2, ids=[m[0] for m in BATCH2])
def test_registered_in_webapp_catalog(name, _dbs, expected_cov):
    from easyicu.webapp.concept_catalog import (
        CONCEPT_DB_COVERAGE,
        CONCEPT_DICTIONARY,
    )
    assert name in CONCEPT_DICTIONARY
    assert CONCEPT_DB_COVERAGE[name] == expected_cov


# ── per-drug spot checks ──
def test_lorazepam_aumc_itemid(cdict):
    src = cdict["lorazepam"]["sources"]["aumc"]
    ids = src[0]["ids"] if isinstance(src, list) else src["ids"]
    assert ids == 7165


def test_ketamine_includes_esketamine_in_aumc_and_sic(cdict):
    aumc_src = cdict["ketamine"]["sources"]["aumc"]
    aumc_ids = set((aumc_src[0]["ids"] if isinstance(aumc_src, list) else aumc_src["ids"]))
    assert {7479, 9018} <= aumc_ids, "Must include both ketamine (7479) and esketamine (9018)"

    sic_src = cdict["ketamine"]["sources"]["sic"]
    sic_ids = set((sic_src[0]["ids"] if isinstance(sic_src, list) else sic_src["ids"]))
    assert {1556, 1578} <= sic_ids, "Must include both ketamine (1556) and S-ketamine (1578)"


def test_cisatracurium_has_no_aumc_no_hirid(cdict):
    sources = set(cdict["cisatracurium"]["sources"])
    assert "aumc" not in sources
    assert "hirid" not in sources


def test_nitroglycerin_aumc_excludes_spray(cdict):
    """Spray form (AUMC 20169) is intentionally excluded to keep the concept
    focused on IV/SL infusion routes used in ICU."""
    src = cdict["nitroglycerin"]["sources"]["aumc"]
    ids = src[0]["ids"] if isinstance(src, list) else src["ids"]
    id_set = set(ids if isinstance(ids, list) else [ids])
    assert 20169 not in id_set, "Nitro spray itemid 20169 must be excluded"


def test_vecuronium_hirid_added_but_sic_still_absent(cdict):
    sources = set(cdict["vecuronium"]["sources"])
    assert "sic" not in sources
    hirid = cdict["vecuronium"]["sources"]["hirid"][0]
    assert hirid["ids"] == [198]
    assert hirid["callback"] == "transform_fun(set_val(TRUE))"
