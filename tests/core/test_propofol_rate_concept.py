"""Structural regression tests for the ``propofol_rate`` concept.

propofol_rate is a ts_tbl rate concept modeled after ``dobu_rate`` / ``norepi_rate``,
emitting continuous propofol infusion rate in mcg/kg/min.

Ground-truth IDs verified against each database's local reference dictionary on
2026-05-13 (see lookup scripts in /tmp):

  - AUMC drugitems itemid 7480 (Propofol (Diprivan))
  - MIMIC-IV inputevents itemid 222168
  - MIMIC-III inputevents_cv itemid 30131, inputevents_mv itemid 222168
  - SIC medication DrugID 1499 (Propofol 1%) + 1549 (Propofol 2%)
  - eICU infusiondrug regex (?i)^propofol.*\\(.+\\)$ (ml_to_mcg=10000, standard 10 mg/mL)
  - HiRID: intentionally absent — pharma reference has no propofol entry

MIIV rate-column study (pre-implementation): 402,985 of 498,811 matching rows
carry ``rateuom = mcg/kg/min`` natively, so mimic_rate_mv yields the target unit
without conversion. MIMIC-III CareVue uses mimic_rate_cv over linkorderid groups.
"""
from __future__ import annotations

import json
import pathlib

import pytest

DICT = pathlib.Path(__file__).resolve().parents[1] / "src" / "easyicu" / "data" / "concept-dict.json"
MAIN_DBS = {"miiv", "mimic", "eicu", "aumc", "hirid", "sic"}


@pytest.fixture(scope="module")
def propofol_rate_entry():
    with DICT.open() as f:
        cdict = json.load(f)
    assert "propofol_rate" in cdict, "propofol_rate missing from concept-dict.json"
    return cdict["propofol_rate"]


def test_propofol_rate_is_mcg_kg_min(propofol_rate_entry):
    assert propofol_rate_entry["unit"] == ["mcg/kg/min", "mcgkgmin"]
    assert propofol_rate_entry["min"] == 0
    assert propofol_rate_entry["max"] == 200
    assert propofol_rate_entry["category"] == "medications"


def test_propofol_rate_covers_four_main_databases(propofol_rate_entry):
    """HiRID pharma reference has no propofol entry. SIC was removed pending
    an audit of per-DrugID ``AmountPerMinute`` unit semantics — DrugID 1499
    (Propofol 1%) stores bolus totals in that column while DrugID 1549
    (Propofol 2%) stores genuine per-minute rates. Applying ``sic_rate_kg``
    uniformly produced accidentally-plausible but unit-mixed output; see
    ``audit_reports/sic_amount_per_minute_unit_audit_20260513.md``."""
    sources = set(propofol_rate_entry["sources"].keys())
    main = sources & MAIN_DBS
    assert main == {"miiv", "mimic", "eicu", "aumc"}, (
        f"Expected 4-DB main coverage (miiv, mimic, eicu, aumc), got: {sorted(main)}"
    )
    assert "hirid" not in sources
    assert "sic" not in sources, (
        "SIC source removed pending AmountPerMinute unit audit — do not re-add "
        "without first validating DrugID 1499/1549 unit convention."
    )


def test_aumc_uses_itemid_7480_and_aumc_rate_kg(propofol_rate_entry):
    aumc = propofol_rate_entry["sources"]["aumc"]
    assert len(aumc) == 1
    src = aumc[0]
    assert src["table"] == "drugitems"
    assert src["sub_var"] == "itemid"
    assert src["ids"] == 7480
    assert src["callback"] == "aumc_rate_kg"
    assert src["rel_weight"] == "doserateperkg"
    assert src["rate_uom"] == "doserateunit"
    assert src["stop_var"] == "stop"


def test_miiv_uses_itemid_222168_and_mimic_rate_mv(propofol_rate_entry):
    miiv = propofol_rate_entry["sources"]["miiv"]
    assert len(miiv) == 1
    src = miiv[0]
    assert src["table"] == "inputevents"
    assert src["sub_var"] == "itemid"
    assert src["ids"] == 222168
    assert src["callback"] == "mimic_rate_mv"
    assert src["stop_var"] == "endtime"


def test_mimic_covers_both_cv_and_mv(propofol_rate_entry):
    """MIMIC-III requires both CareVue (itemid 30131) and MetaVision (222168) sources."""
    mimic = propofol_rate_entry["sources"]["mimic"]
    tables = {src["table"]: src for src in mimic}
    assert "inputevents_cv" in tables
    assert "inputevents_mv" in tables

    cv = tables["inputevents_cv"]
    assert cv["ids"] == 30131
    assert cv["callback"] == "mimic_rate_cv"
    assert cv["grp_var"] == "linkorderid"

    mv = tables["inputevents_mv"]
    assert mv["ids"] == 222168
    assert mv["callback"] == "mimic_rate_mv"
    assert mv["stop_var"] == "endtime"


def test_eicu_regex_and_ml_to_mcg(propofol_rate_entry):
    """Standard propofol concentration 10 mg/mL → ml_to_mcg = 10000."""
    eicu = propofol_rate_entry["sources"]["eicu"]
    assert len(eicu) == 1
    src = eicu[0]
    assert src["table"] == "infusiondrug"
    assert src["sub_var"] == "drugname"
    assert src["class_name"] == "rgx_itm"
    assert "propofol" in src["regex"].lower()
    assert src["callback"] == "eicu_rate_kg(ml_to_mcg = 10000)"
    assert src["weight_var"] == "patientweight"


def test_sic_uses_both_concentration_drugids(propofol_rate_entry):
    """SIC source was REMOVED pending an audit of AmountPerMinute unit
    semantics (see SIC audit report). Guard rail: do not silently re-add."""
    sources = propofol_rate_entry["sources"]
    assert "sic" not in sources


def test_demo_mirrors_full_eicu_and_mimic(propofol_rate_entry):
    srcs = propofol_rate_entry["sources"]
    assert srcs["eicu_demo"] == srcs["eicu"]
    assert srcs["mimic_demo"] == srcs["mimic"]


def test_registered_in_window_concepts():
    """Rate concepts must expand into time-series windows just like norepi_rate."""
    from easyicu.utils.compat import WINDOW_CONCEPTS

    assert "propofol_rate" in WINDOW_CONCEPTS, (
        "propofol_rate must be in WINDOW_CONCEPTS so the loader expands rates "
        "across their administration intervals."
    )
    # Must NOT be in point/duration sets.
    from easyicu.utils.compat import DURATION_CONCEPTS, POINT_EVENT_CONCEPTS
    assert "propofol_rate" not in POINT_EVENT_CONCEPTS
    assert "propofol_rate" not in DURATION_CONCEPTS


def test_registered_in_webapp_catalog():
    from easyicu.concept.catalog import (
        CONCEPT_DB_COVERAGE,
        CONCEPT_DICTIONARY,
    )
    assert "propofol_rate" in CONCEPT_DICTIONARY
    en, zh, unit = CONCEPT_DICTIONARY["propofol_rate"]
    assert "Propofol" in en
    assert "丙泊酚" in zh
    assert unit == "mcg/kg/min"
    assert CONCEPT_DB_COVERAGE.get("propofol_rate") == 4
