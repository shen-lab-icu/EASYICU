"""Structural regression tests for 10 new medication concepts (2026-05-12).

Verifies:
  - Each concept exists in concept-dict.json as lgl_cncpt
  - Database source coverage matches expectations
  - All sources use set_val(TRUE) callback
  - All are registered in POINT_EVENT_CONCEPTS
  - All have webapp catalog entries
"""
import json
import pathlib
import pytest

DICT_PATH = pathlib.Path(__file__).resolve().parents[1] / "src" / "easyicu" / "data" / "concept-dict.json"

with open(DICT_PATH) as f:
    CDICT = json.load(f)

# Expected: (concept_name, expected_main_db_count, required_dbs)
MEDICATIONS = [
    ("propofol",         6, {"miiv", "mimic", "eicu", "aumc", "hirid", "sic"}),
    ("midazolam",        6, {"miiv", "mimic", "eicu", "aumc", "hirid", "sic"}),
    ("dexmedetomidine",  4, {"miiv", "mimic", "eicu", "hirid"}),
    ("fentanyl",         6, {"miiv", "mimic", "eicu", "aumc", "hirid", "sic"}),
    ("morphine",         6, {"miiv", "mimic", "eicu", "aumc", "hirid", "sic"}),
    ("heparin",          6, {"miiv", "mimic", "eicu", "aumc", "hirid", "sic"}),
    ("mannitol",         5, {"miiv", "mimic", "eicu", "aumc", "sic"}),
    ("amiodarone",       6, {"miiv", "mimic", "eicu", "aumc", "hirid", "sic"}),
    ("milrinone",        5, {"miiv", "mimic", "eicu", "hirid", "sic"}),
    ("rocuronium",       5, {"miiv", "eicu", "aumc", "hirid", "sic"}),
]

MAIN_DBS = {"miiv", "mimic", "eicu", "aumc", "hirid", "sic"}


# ── existence & class ──
@pytest.mark.parametrize("name,_n,_dbs", MEDICATIONS, ids=[m[0] for m in MEDICATIONS])
def test_is_logical_concept(name, _n, _dbs):
    assert name in CDICT, f"{name} not found in concept-dict.json"
    assert CDICT[name].get("class_name") == "lgl_cncpt"
    assert CDICT[name].get("category") == "medications"
    assert CDICT[name].get("aggregate") == "max"


# ── database coverage ──
@pytest.mark.parametrize("name,expected_count,required_dbs", MEDICATIONS, ids=[m[0] for m in MEDICATIONS])
def test_db_coverage(name, expected_count, required_dbs):
    sources = set(CDICT[name].get("sources", {}).keys())
    main_covered = sources & MAIN_DBS
    assert main_covered == required_dbs, (
        f"{name}: expected {required_dbs}, got {main_covered}"
    )
    assert len(main_covered) == expected_count


# ── all sources use set_val(TRUE) ──
@pytest.mark.parametrize("name,_n,_dbs", MEDICATIONS, ids=[m[0] for m in MEDICATIONS])
def test_all_sources_use_set_val_true(name, _n, _dbs):
    for db, items in CDICT[name]["sources"].items():
        if not isinstance(items, list):
            items = [items]
        for item in items:
            cb = item.get("callback", "")
            assert "set_val(TRUE)" in cb, (
                f"{name}/{db}: callback '{cb}' missing set_val(TRUE)"
            )


# ── POINT_EVENT_CONCEPTS registration ──
def test_all_registered_as_point_events():
    from easyicu.compat import POINT_EVENT_CONCEPTS
    for name, _, _ in MEDICATIONS:
        assert name in POINT_EVENT_CONCEPTS, f"{name} not in POINT_EVENT_CONCEPTS"


# ── webapp catalog ──
def test_all_in_webapp_catalog():
    from easyicu.webapp.concept_catalog import CONCEPT_DICTIONARY, CONCEPT_DB_COVERAGE
    for name, expected_count, _ in MEDICATIONS:
        assert name in CONCEPT_DICTIONARY, f"{name} not in CONCEPT_DICTIONARY"
        assert name in CONCEPT_DB_COVERAGE, f"{name} not in CONCEPT_DB_COVERAGE"
        assert CONCEPT_DB_COVERAGE[name] == expected_count, (
            f"{name}: coverage {CONCEPT_DB_COVERAGE[name]} != {expected_count}"
        )


# ── specific item ID spot-checks ──
def test_propofol_miiv_itemid():
    src = CDICT["propofol"]["sources"]["miiv"]
    ids = src[0]["ids"] if isinstance(src, list) else src["ids"]
    assert 222168 in ids


def test_midazolam_hirid_pharmaids():
    src = CDICT["midazolam"]["sources"]["hirid"]
    ids = src[0]["ids"] if isinstance(src, list) else src["ids"]
    assert 251 in ids and 252 in ids


def test_fentanyl_excludes_sufentanil_regex():
    """eICU regex should use negative lookbehind to exclude sufentanil."""
    src = CDICT["fentanyl"]["sources"]["eicu"]
    for item in src:
        if "regex" in item:
            assert "su" not in item["regex"].lower().split("fentanyl")[0] or "(?<!" in item["regex"]


def test_heparin_miiv_includes_prophylaxis():
    src = CDICT["heparin"]["sources"]["miiv"]
    ids = src[0]["ids"] if isinstance(src, list) else src["ids"]
    assert 225152 in ids  # therapeutic
    assert 225975 in ids  # prophylaxis


def test_amiodarone_aumc_three_forms():
    src = CDICT["amiodarone"]["sources"]["aumc"]
    ids = src[0]["ids"] if isinstance(src, list) else src["ids"]
    assert set(ids) == {6844, 9015, 16113}


def test_milrinone_hirid_corotrop():
    src = CDICT["milrinone"]["sources"]["hirid"]
    ids = src[0]["ids"] if isinstance(src, list) else src["ids"]
    assert 1000441 in ids
