"""Structural regression tests for batch-4 medication concepts (2026-05-13).

Added: potassium_iv, magnesium_iv, albumin_iv, packed_rbc.

All are lgl_cncpt with set_val(TRUE). Coverage after the 2026-07-17 audit
(previous 2026-05-27 figures in brackets):
  - potassium_iv: 6/6 [was 5/6 "no HiRID"] -- HiRID wired; the 2026-05-27 gap left it
    constant-FALSE across HiRID while 6,541 patients had IV potassium recorded.
  - magnesium_iv: 6/6
  - albumin_iv:   5/6 [was 4/6 "no HiRID, no AUMC"] -- AUMC wired; the "no matching
    drugitem" note of 2026-05-13 was wrong (8933 'Albumine 20%', 1,238 admissions).
  - packed_rbc:   6/6 [was 5/6 "no eICU"] -- eICU wired via intakeOutput, which is
    where the previous docstring itself said the data lived (17,575 stays).

Specific design decisions enforced by tests:
  - packed_rbc's eICU source is intakeOutput.celllabel with EXPLICIT ids -- never a
    medication regex (false negatives: the table has no blood products at all), and
    never a regex on intakeoutput (rgx_itm leaves charttime in minutes there and
    silently yields 0 rows).
  - potassium_iv includes K-phosphate (225925) because it IS IV potassium
    administration, not just KCl; its HiRID source excludes the oral syrup and
    effervescent tablets for the same reason.

Standing lesson from this batch: a "documented-absent by design" claim is a claim
about the data and decays like any other. Two of the four here were simply false when
re-checked against the raw tables, and because these are booleans, being wrong is
invisible -- an always-FALSE column is indistinguishable from a drug never given.
"""
from __future__ import annotations

import json
import pathlib

import pytest

DICT = pathlib.Path(__file__).resolve().parents[1] / "src" / "easyicu" / "data" / "concept-dict.json"
MAIN_DBS = {"miiv", "mimic", "eicu", "aumc", "hirid", "sic"}


BATCH4 = [
    ("potassium_iv", {"miiv", "mimic", "eicu", "aumc", "hirid", "sic"}, 6),
    ("magnesium_iv", {"miiv", "mimic", "eicu", "aumc", "hirid", "sic"}, 6),
    ("albumin_iv",   {"miiv", "mimic", "eicu", "aumc", "sic"},          5),
    ("packed_rbc",   {"miiv", "mimic", "eicu", "aumc", "hirid", "sic"}, 6),
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
    from easyicu.utils.compat import POINT_EVENT_CONCEPTS, WINDOW_CONCEPTS
    assert name in POINT_EVENT_CONCEPTS
    assert name not in WINDOW_CONCEPTS


@pytest.mark.parametrize("name,_dbs,expected_cov", BATCH4, ids=[m[0] for m in BATCH4])
def test_registered_in_webapp_catalog(name, _dbs, expected_cov):
    from easyicu.concept.catalog import (
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


def test_albumin_iv_aumc_source_is_the_dutch_product(cdict):
    """2026-07-17 REVERSED. This asserted AUMC had no albumin, on a 2026-05-13 note,
    and told anyone adding it to re-audit first. The re-audit disproves the note:
    AUMC drugitems 8933 'Albumine 20%' exists and carries 3,238 rows / 1,238
    admissions. So albumin_iv was a constant FALSE across AUMC for a drug given to
    1,238 patients -- and albumin is a colloid, i.e. it sits directly on the
    fluid -> venous-congestion path. The guard worked exactly as intended.
    """
    aumc = cdict["albumin_iv"]["sources"]["aumc"][0]
    assert set(aumc["ids"]) == {8933}
    assert aumc["callback"] == "transform_fun(set_val(TRUE))"


def test_packed_rbc_eicu_uses_intakeoutput_not_medication(cdict):
    """2026-07-17. The previous test asserted eICU absence outright; its docstring
    already said the data lives in intakeOutput and that a *medication regex* would
    give false negatives. Both halves of that are right -- so this now pins the
    positive form: source it from intakeOutput, never from `medication`.

    Verified: eICU `medication` holds 0 blood products; intakeOutput.celllabel has
    pRBCs (54,774 rows / 17,575 stays), 'Volume-Transfuse red blood cells' (5,841 /
    2,098), 'PRBC' (256 / 118) and the leukoreduced variant (1,835 / 415).
    Explicit celllabel ids, NOT a regex: rgx_itm on intakeoutput hits the
    charttime-stays-in-minutes bug and silently returns 0 rows.
    """
    eicu = cdict["packed_rbc"]["sources"]["eicu"]
    assert all(src["table"] == "intakeoutput" for src in eicu)
    assert all(src["sub_var"] == "celllabel" for src in eicu)
    assert all("regex" not in src for src in eicu), "intakeoutput must use explicit ids"
    ids = {i for src in eicu for i in src["ids"]}
    assert "pRBCs" in ids


def test_packed_rbc_includes_or_pacu_intake(cdict):
    """MIMIC-IV splits RBC into 225168 (primary), 226368 (OR intake),
    227070 (PACU intake). All three are RBC administration events."""
    miiv = cdict["packed_rbc"]["sources"]["miiv"]
    ids = set(miiv[0]["ids"] if isinstance(miiv, list) else miiv["ids"])
    assert {225168, 226368, 227070} <= ids


def test_batch4_hirid_scope_matches_audit(cdict):
    # 2026-07-17: potassium_iv graduated out of this list. The guard asked for callback
    # validation before a HiRID source was added; that was done. HiRID pharma has
    # 1000396 'Kalium Chlorid 15% 10 ml' (230,871 rows / 6,541 patients), 1000395
    # 'Kalium Phosphat' (66,656 / 2,135) and 1000836 'Kalium Phosphat fuer Perfusor'
    # (76,434 / 1,982) -- so potassium_iv was a constant FALSE across all of HiRID.
    # The oral products (1000612 'Kalium-Chlorid Sirup', 4,928 pts; 1000393
    # 'Kalium Effervetten', 1,260 pts) stay excluded: this concept is IV.
    # Post-fix integration check: 90 of 400 sampled patients = 22.5%, against a
    # whole-database expectation of ~20%.
    hirid_k = cdict["potassium_iv"]["sources"]["hirid"][0]
    # 2026-07-17 recall: added the IV concentrates 1000080 'K-Cl conc' (3,421 pts) and
    # 1000082 'K-Pi conc' (487 pts) that are added to infusions, on top of the bagged products.
    assert {1000396, 1000395, 1000836, 1000080, 1000082}.issubset(set(hirid_k["ids"]))
    assert hirid_k["callback"] == "transform_fun(set_val(TRUE))"
    assert {1000612, 1000393}.isdisjoint(set(hirid_k["ids"])), "oral potassium must stay out"

    for c in ["albumin_iv"]:
        assert "hirid" not in cdict[c]["sources"], (
            f"{c}: HiRID source added without callback validation"
        )
    assert cdict["magnesium_iv"]["sources"]["hirid"][0]["ids"] == [1000421]
    assert cdict["packed_rbc"]["sources"]["hirid"][0]["ids"] == [1000100, 1000743]
    for c in ["magnesium_iv", "packed_rbc"]:
        assert cdict[c]["sources"]["hirid"][0]["callback"] == "transform_fun(set_val(TRUE))"
