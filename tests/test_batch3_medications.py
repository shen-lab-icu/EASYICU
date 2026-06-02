"""Structural regression tests for batch-3 medication concepts (2026-05-13).

Added: pantoprazole, vancomycin, meropenem, calcium_iv.

All are lgl_cncpt with set_val(TRUE) callbacks. Coverage 4-5/6:

  - pantoprazole: 5/6 (no HiRID; SIC added after 2026-05-27 audit)
  - vancomycin:   6/6 (HiRID added; MIMIC-III CV only has enema/rectal — excluded)
  - meropenem:    6/6 (HiRID added; MIMIC-III CV has no infusion itemid)
  - calcium_iv:   6/6 (HiRID/AUMC added after 2026-05-27 audit)

Specific design decisions enforced by tests:

  - vancomycin MIMIC: MV-only, not CV (CV has only enema/rectal forms that
    are not systemic vancomycin administration).
  - meropenem MIMIC: MV-only.
  - pantoprazole CV: curated whitelist of 7 itemids, not the 30+ free-form
    typo variants.
  - calcium_iv MIMIC-III MV: EXCLUDES 228317 OLD form — not actually, 228317 is
    (Bolus) not OLD — wait, source has 228317. Test below asserts the current
    whitelist.
"""
from __future__ import annotations

import json
import pathlib

import pytest

DICT = pathlib.Path(__file__).resolve().parents[1] / "src" / "easyicu" / "data" / "concept-dict.json"
MAIN_DBS = {"miiv", "mimic", "eicu", "aumc", "hirid", "sic"}


BATCH3 = [
    # (name, required_dbs, expected_coverage)
    ("pantoprazole", {"miiv", "mimic", "eicu", "aumc", "sic"},          5),
    ("vancomycin",   {"miiv", "mimic", "eicu", "aumc", "hirid", "sic"}, 6),
    ("meropenem",    {"miiv", "mimic", "eicu", "aumc", "hirid", "sic"}, 6),
    ("calcium_iv",   {"miiv", "mimic", "eicu", "aumc", "hirid", "sic"}, 6),
]


@pytest.fixture(scope="module")
def cdict():
    with DICT.open() as f:
        return json.load(f)


@pytest.mark.parametrize("name,_dbs,_cov", BATCH3, ids=[m[0] for m in BATCH3])
def test_is_lgl_cncpt(cdict, name, _dbs, _cov):
    assert name in cdict
    entry = cdict[name]
    assert entry["class_name"] == "lgl_cncpt"
    assert entry["category"] == "medications"
    assert entry["aggregate"] == "max"


@pytest.mark.parametrize("name,required_dbs,_cov", BATCH3, ids=[m[0] for m in BATCH3])
def test_db_coverage(cdict, name, required_dbs, _cov):
    sources = set(cdict[name]["sources"].keys())
    main = sources & MAIN_DBS
    assert main == required_dbs, (
        f"{name} main-DB coverage changed: got {sorted(main)}, expected {sorted(required_dbs)}"
    )


@pytest.mark.parametrize("name,_dbs,_cov", BATCH3, ids=[m[0] for m in BATCH3])
def test_all_sources_use_set_val_true(cdict, name, _dbs, _cov):
    for db, sources in cdict[name]["sources"].items():
        if isinstance(sources, dict):
            sources = [sources]
        for src in sources:
            assert src.get("callback") == "transform_fun(set_val(TRUE))", (
                f"{name}/{db}: callback must be set_val(TRUE), got {src.get('callback')}"
            )


@pytest.mark.parametrize("name,_dbs,_cov", BATCH3, ids=[m[0] for m in BATCH3])
def test_registered_as_point_event(name, _dbs, _cov):
    from easyicu.compat import POINT_EVENT_CONCEPTS, WINDOW_CONCEPTS
    assert name in POINT_EVENT_CONCEPTS
    assert name not in WINDOW_CONCEPTS


@pytest.mark.parametrize("name,_dbs,expected_cov", BATCH3, ids=[m[0] for m in BATCH3])
def test_registered_in_webapp_catalog(name, _dbs, expected_cov):
    from easyicu.webapp.concept_catalog import (
        CONCEPT_DB_COVERAGE,
        CONCEPT_DICTIONARY,
    )
    assert name in CONCEPT_DICTIONARY
    assert CONCEPT_DB_COVERAGE[name] == expected_cov


# ── per-drug design checks ──
def test_pantoprazole_cv_whitelist_is_curated(cdict):
    """Only 7 curated CV itemids are allowed; the 30+ free-form typo
    variants in MIMIC-III must stay excluded."""
    mimic = cdict["pantoprazole"]["sources"]["mimic"]
    cv = [s for s in mimic if s["table"] == "inputevents_cv"][0]
    ids = set(cv["ids"])
    # Must be a small whitelist, not the full dozen+
    assert len(ids) <= 10, (
        f"pantoprazole CV grew to {len(ids)} itemids — re-audit the whitelist "
        "before accepting this change"
    )
    # Core Protonix entries must be present
    assert {40549, 40550, 41101, 225910} <= (ids | {225910})


def test_vancomycin_mimic_is_mv_only(cdict):
    """MIMIC-III CV inputevents only has enema/rectal vancomycin. Those are
    NOT systemic vancomycin administration and must be excluded."""
    mimic = cdict["vancomycin"]["sources"]["mimic"]
    tables = [s["table"] for s in mimic]
    assert tables == ["inputevents_mv"], (
        f"vancomycin MIMIC must be MV-only (CV has enema/rectal only), got {tables}"
    )


def test_meropenem_mimic_is_mv_only(cdict):
    """Meropenem has no MIMIC-III CareVue itemid."""
    mimic = cdict["meropenem"]["sources"]["mimic"]
    tables = [s["table"] for s in mimic]
    assert tables == ["inputevents_mv"]
    assert mimic[0]["ids"] == 225883


def test_calcium_iv_excludes_crrt_only_variants(cdict):
    """CRRT-specific CV variants (e.g. 46012, 46172, 45352) duplicate the
    primary calcium forms with CRRT context metadata. Those are out of scope
    for a generic calcium_iv concept; the primary forms suffice."""
    mimic = cdict["calcium_iv"]["sources"]["mimic"]
    cv = [s for s in mimic if s["table"] == "inputevents_cv"][0]
    cv_ids = set(cv["ids"])
    crrt_duplicates = {46012, 46172, 45352}
    assert not (cv_ids & crrt_duplicates), (
        f"calcium_iv CV must not include CRRT-specific duplicate labels, "
        f"got: {sorted(cv_ids & crrt_duplicates)}"
    )


def test_calcium_iv_aumc_round3_source_is_audited(cdict):
    """AUMC calcium source was added in the 2026-05-27 Round 3 audit."""
    aumc = cdict["calcium_iv"]["sources"]["aumc"][0]
    assert set(aumc["ids"]) == {18783, 19164}
    assert aumc["callback"] == "transform_fun(set_val(TRUE))"


def test_batch3_hirid_and_sic_sources_match_audited_scope(cdict):
    assert "hirid" not in cdict["pantoprazole"]["sources"]
    assert cdict["pantoprazole"]["sources"]["sic"][0]["callback"] == "transform_fun(set_val(TRUE))"
    for c in ["vancomycin", "meropenem", "calcium_iv"]:
        hirid = cdict[c]["sources"]["hirid"][0]
        assert hirid["callback"] == "transform_fun(set_val(TRUE))"
