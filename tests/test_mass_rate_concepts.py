"""Structural regression tests for mass-rate medication concepts.

Covers ``fentanyl_rate`` (mcg/hour) and ``midazolam_rate`` (mg/hour) — both are
non-kg-normalized mass rates.

Coverage matrix (2026-05-13):

  =================  ============================================
  Database           Source                      Callback
  =================  ============================================
  MIIV               inputevents 221744/221668   mimic_rate_mv
  MIMIC-III MV       inputevents_mv (same)       mimic_rate_mv
  MIMIC-III CV       inputevents_cv 30118/30149  mimic_rate_cv
                     + 30124 (midazolam)
  eICU               infusiondrug regex          eicu_rate_mass
  AUMC               drugitems 7219/7194         aumc_rate_mass
  HiRID              pharma 1000741/252+1001051  hirid_rate_mass
  =================  ============================================

SIC still needs a dedicated mass-rate callback (AmountPerMinute unit
semantics differ by DrugID and haven't been cross-validated against ricu).
Explicitly absent.

Callback behavior:

  - ``eicu_rate_mass`` drops rows with ``(ml/hr)`` suffix (no concentration),
    ``/kg/`` suffix, or unknown units.
  - ``aumc_rate_mass`` drops rows where ``doserateperkg == 1`` (kg-norm)
    or rate unit is missing (bolus dose).
  - ``hirid_rate_mass`` converts based on per-row ``doseunit`` with
    µg/mg/g → target-mass conversion; sums per (patient, hour, infusionid);
    rows with non-mass units are dropped.
"""
from __future__ import annotations

import json
import pathlib

import pytest

DICT = pathlib.Path(__file__).resolve().parents[1] / "src" / "easyicu" / "data" / "concept-dict.json"


RATE_CONCEPTS = [
    ("fentanyl_rate",  221744, "mcg/hour",   "mcghr", 1000),
    ("midazolam_rate", 221668, "mg/hour",    "mghr",  200),
]


@pytest.fixture(scope="module")
def cdict():
    with DICT.open() as f:
        return json.load(f)


@pytest.mark.parametrize("name,itemid,unit_primary,unit_alt,rate_max",
                         RATE_CONCEPTS, ids=[c[0] for c in RATE_CONCEPTS])
def test_entry_shape(cdict, name, itemid, unit_primary, unit_alt, rate_max):
    assert name in cdict, f"{name} missing from concept-dict.json"
    entry = cdict[name]
    assert entry["unit"] == [unit_primary, unit_alt]
    assert entry["min"] == 0
    assert entry["max"] == rate_max
    assert entry["category"] == "medications"


@pytest.mark.parametrize("name,itemid,_u1,_u2,_max",
                         RATE_CONCEPTS, ids=[c[0] for c in RATE_CONCEPTS])
def test_miiv_source_uses_mimic_rate_mv(cdict, name, itemid, _u1, _u2, _max):
    miiv = cdict[name]["sources"]["miiv"]
    assert len(miiv) == 1
    src = miiv[0]
    assert src["table"] == "inputevents"
    assert src["sub_var"] == "itemid"
    assert src["ids"] == itemid
    assert src["callback"] == "mimic_rate_mv"
    assert src["stop_var"] == "endtime"


@pytest.mark.parametrize("name,itemid,_u1,_u2,_max",
                         RATE_CONCEPTS, ids=[c[0] for c in RATE_CONCEPTS])
def test_mimic_source_covers_cv_and_mv(cdict, name, itemid, _u1, _u2, _max):
    """MIMIC-III now covers CareVue + MetaVision. CV itemids chosen so every
    row has ``rateuom`` matching the target unit (no per-row unit branching)."""
    mimic = cdict[name]["sources"]["mimic"]
    tables = {src["table"]: src for src in mimic}
    assert set(tables) == {"inputevents_cv", "inputevents_mv"}, (
        f"{name} mimic tables changed: {sorted(tables)}"
    )
    mv = tables["inputevents_mv"]
    assert mv["ids"] == itemid
    assert mv["callback"] == "mimic_rate_mv"

    cv = tables["inputevents_cv"]
    assert cv["callback"] == "mimic_rate_cv"
    assert cv["grp_var"] == "linkorderid"


@pytest.mark.parametrize("name,expected_cv_ids", [
    ("fentanyl_rate",  [30118, 30149]),
    ("midazolam_rate", 30124),
], ids=["fentanyl_rate", "midazolam_rate"])
def test_mimic_cv_uses_pure_unit_itemids(cdict, name, expected_cv_ids):
    """Only itemids where 100 % of rows share the target rateuom are allowed.
    30308 (mcgkghr) for fentanyl is intentionally excluded because it would
    need kg renormalization."""
    cv = [s for s in cdict[name]["sources"]["mimic"]
          if s["table"] == "inputevents_cv"][0]
    assert cv["ids"] == expected_cv_ids
    # Guard: make sure no mixed-unit itemid snuck in
    excluded = {30308}  # fentanyl mcgkghr
    cv_id_set = set(cv["ids"] if isinstance(cv["ids"], list) else [cv["ids"]])
    assert not (cv_id_set & excluded), (
        f"{name} must not use kg-normalized itemids in mass-rate concept"
    )


@pytest.mark.parametrize("name,_iid,_u1,_u2,_max",
                         RATE_CONCEPTS, ids=[c[0] for c in RATE_CONCEPTS])
def test_other_dbs_intentionally_absent(cdict, name, _iid, _u1, _u2, _max):
    """Explicit contract: SIC NOT yet wired. SIC uses AmountPerMinute whose
    unit semantics are DrugID-dependent (vasopressors use grams/min, but
    sedatives follow a different convention that hasn't been cross-validated
    against ricu)."""
    sources = set(cdict[name]["sources"].keys())
    assert sources == {"miiv", "mimic", "mimic_demo", "eicu", "eicu_demo", "aumc", "hirid"}, (
        f"{name} sources changed: {sorted(sources)}. If adding a new DB, "
        "confirm the callback preserves the native unit before relaxing this "
        "assertion."
    )


@pytest.mark.parametrize("name,target_unit,hirid_ids", [
    ("fentanyl_rate",  "mcg/hour", [1000741]),
    ("midazolam_rate", "mg/hour",  [252, 1001051]),
], ids=["fentanyl_rate", "midazolam_rate"])
def test_hirid_uses_hirid_rate_mass(cdict, name, target_unit, hirid_ids):
    """HiRID source must use the new non-kg mass-rate callback."""
    hirid = cdict[name]["sources"]["hirid"]
    assert len(hirid) == 1
    src = hirid[0]
    assert src["table"] == "pharma"
    assert src["sub_var"] == "pharmaid"
    assert src["ids"] == hirid_ids
    assert src["grp_var"] == "infusionid"
    expected_cb = f'hirid_rate_mass(target_unit = "{target_unit}")'
    assert src["callback"] == expected_cb


@pytest.mark.parametrize("name,target_unit,aumc_itemid", [
    ("fentanyl_rate",  "mcg/hour", 7219),
    ("midazolam_rate", "mg/hour",  7194),
], ids=["fentanyl_rate", "midazolam_rate"])
def test_aumc_uses_aumc_rate_mass(cdict, name, target_unit, aumc_itemid):
    """AUMC source must use the new non-kg mass-rate callback."""
    aumc = cdict[name]["sources"]["aumc"]
    assert len(aumc) == 1
    src = aumc[0]
    assert src["table"] == "drugitems"
    assert src["sub_var"] == "itemid"
    assert src["ids"] == aumc_itemid
    expected_cb = f'aumc_rate_mass(target_unit = "{target_unit}")'
    assert src["callback"] == expected_cb, (
        f"{name}: AUMC callback drifted: {src['callback']!r}"
    )
    # Must carry the metadata needed by the callback
    assert src["rel_weight"] == "doserateperkg", (
        f"{name}: AUMC source lost doserateperkg flag — would incorrectly "
        "include kg-normalized rows"
    )
    assert src["rate_uom"] == "doserateunit"
    assert src["stop_var"] == "stop"


@pytest.mark.parametrize("name,target_unit", [
    ("fentanyl_rate",  "mcg/hour"),
    ("midazolam_rate", "mg/hour"),
], ids=["fentanyl_rate", "midazolam_rate"])
def test_eicu_uses_eicu_rate_mass(cdict, name, target_unit):
    """eICU source must use the dedicated non-kg mass-rate callback with the
    correct target unit string. Any drift here (e.g. someone swapping in
    eicu_rate_kg) would silently divide by weight and produce bogus data."""
    eicu = cdict[name]["sources"]["eicu"]
    assert len(eicu) == 1
    src = eicu[0]
    assert src["table"] == "infusiondrug"
    assert src["sub_var"] == "drugname"
    assert src["class_name"] == "rgx_itm"
    # Exact callback string — any change requires explicit test update
    expected_cb = f'eicu_rate_mass(target_unit = "{target_unit}")'
    assert src["callback"] == expected_cb, (
        f"eICU callback drifted for {name}: {src['callback']!r}"
    )
    # eicu_demo must mirror eicu
    demo = cdict[name]["sources"]["eicu_demo"]
    assert demo == eicu, f"{name} eicu_demo must mirror eicu"


@pytest.mark.parametrize("name,_iid,_u1,_u2,_max",
                         RATE_CONCEPTS, ids=[c[0] for c in RATE_CONCEPTS])
def test_registered_as_window_concept(name, _iid, _u1, _u2, _max):
    from easyicu.compat import (
        DURATION_CONCEPTS,
        POINT_EVENT_CONCEPTS,
        WINDOW_CONCEPTS,
    )
    assert name in WINDOW_CONCEPTS
    assert name not in POINT_EVENT_CONCEPTS
    assert name not in DURATION_CONCEPTS


@pytest.mark.parametrize("name,_iid,unit_primary,_u2,_max",
                         RATE_CONCEPTS, ids=[c[0] for c in RATE_CONCEPTS])
def test_registered_in_webapp_catalog(name, _iid, unit_primary, _u2, _max):
    from easyicu.webapp.concept_catalog import (
        CONCEPT_DB_COVERAGE,
        CONCEPT_DICTIONARY,
    )
    assert name in CONCEPT_DICTIONARY
    en, zh, unit = CONCEPT_DICTIONARY[name]
    assert "Rate" in en
    assert "速率" in zh
    assert unit == unit_primary
    assert CONCEPT_DB_COVERAGE[name] == 5
