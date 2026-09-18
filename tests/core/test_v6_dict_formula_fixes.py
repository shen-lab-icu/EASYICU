"""v6 dict/formula regression tests (Step1).

Covers the four code changes in this round plus verify-only guards for the
three items already closed on main:
  - crea 15->25 already in 170b5818 (guard against regression)
  - sic_death ICUOffset subtraction already in 170b5818 (guard)
  - mech_vent eicu/mimic sources present (v5 sealed shows non-empty; guard
    the pipeline mapping/regex from regressing to empty)
Changed in this round:
  - po2 min 40->20 (+ doc, per archive/concept-clinical-bounds-ee33539)
  - norepi_equiv median->sum (additive potency)
  - adh_rate adds 0-0.15, phn_rate adds 0-15 (per ee33539)
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from easyicu.concept.callbacks import (
    CALLBACK_REGISTRY,
    ConceptCallbackContext,
    _callback_norepi_equiv,
)
from easyicu.table import ICUTable

DICT_PATH = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "easyicu"
    / "data"
    / "concept-dict.json"
)


def _load_dict():
    with open(DICT_PATH, encoding="utf-8") as fh:
        return json.load(fh)


def _ctx(name: str) -> ConceptCallbackContext:
    return ConceptCallbackContext(
        concept_name=name,
        target=None,
        interval=pd.Timedelta(hours=1),
        resolver=None,
        data_source=None,
        patient_ids=None,
    )


def _tbl(name, rows):
    df = pd.DataFrame(rows)
    return ICUTable(
        df, id_columns=["stay_id"], index_column="charttime", value_column=name
    )


# --- dict bounds ---


def test_v6_crea_ceiling_stays_25():
    d = _load_dict()
    assert d["crea"]["max"] == 25
    assert d["crea"]["min"] == 0


def test_v6_po2_lower_bound_20_with_doc():
    d = _load_dict()
    assert d["po2"]["min"] == 20
    assert d["po2"]["max"] == 600
    comment = d["po2"].get("_comment", "")
    assert "ee33539" in comment
    assert "40" in comment  # documents the 40->20 change rationale


def test_v6_adh_rate_bounds():
    d = _load_dict()
    assert d["adh_rate"]["min"] == 0
    assert d["adh_rate"]["max"] == 0.15
    assert "ee33539" in d["adh_rate"].get("_comment", "")


def test_v6_phn_rate_bounds():
    d = _load_dict()
    assert d["phn_rate"]["min"] == 0
    assert d["phn_rate"]["max"] == 15
    assert "ee33539" in d["phn_rate"].get("_comment", "")


# --- norepi_equiv sum ---


def test_v6_norepi_equiv_sums_multi_drug():
    """norepi 0.2 + epi 0.2 + adh 0.04 -> 0.2+0.2+0.1=0.5 (median would be 0.2)."""
    t = pd.Timestamp("2026-01-01 08:00")
    tables = {
        "norepi_rate": _tbl("norepi_rate", [{"stay_id": 1, "charttime": t, "norepi_rate": 0.2}]),
        "epi_rate": _tbl("epi_rate", [{"stay_id": 1, "charttime": t, "epi_rate": 0.2}]),
        "adh_rate": _tbl("adh_rate", [{"stay_id": 1, "charttime": t, "adh_rate": 0.04}]),
    }
    result = _callback_norepi_equiv(tables, _ctx("norepi_equiv"))
    assert result.value_column == "norepi_equiv"
    assert len(result.data) == 1
    assert result.data["norepi_equiv"].iloc[0] == abs(0.5)  # 0.2+0.2+0.04/0.4
    # median regression guard: must NOT be 0.2
    assert result.data["norepi_equiv"].iloc[0] != 0.2


def test_v6_norepi_equiv_single_drug_unchanged():
    t = pd.Timestamp("2026-01-01 08:00")
    tables = {
        "norepi_rate": _tbl("norepi_rate", [{"stay_id": 1, "charttime": t, "norepi_rate": 0.3}]),
    }
    result = _callback_norepi_equiv(tables, _ctx("norepi_equiv"))
    assert result.data["norepi_equiv"].iloc[0] == abs(0.3)


def test_v6_norepi_equiv_registered():
    assert CALLBACK_REGISTRY["norepi_equiv"] is _callback_norepi_equiv


def test_v6_sofa_cardio_does_not_consume_norepi_equiv():
    """One-change-must-not-collapse-all: SOFA cardio uses 60-min rates, not equiv."""
    d = _load_dict()
    concepts = d["sofa_cardio"]["concepts"]
    assert "norepi_equiv" not in concepts
    assert "norepi60" in concepts


# --- verify-only guards (already closed on main) ---


def test_v6_mech_vent_eicu_mimic_sources_present():
    d = _load_dict()
    src = d["mech_vent"]["sources"]
    assert "eicu" in src and len(src["eicu"]) >= 3
    assert "mimic" in src and len(src["mimic"]) >= 1
    assert "miiv" in src and len(src["miiv"]) >= 1
    # eicu must keep the three evidence routes (airway/device/treatment)
    callbacks = json.dumps(src["eicu"])
    assert "eicu_invasive_airway_evidence" in callbacks
    assert "eicu_respiratory_device_ventilation_evidence" in callbacks
    assert "eicu_treatment_ventilation_evidence" in callbacks


def test_v6_sic_death_uses_icu_offset():
    src = (
        Path(__file__).resolve().parents[2]
        / "src"
        / "easyicu"
        / "concept"
        / "callback_apply.py"
    ).read_text(encoding="utf-8")
    assert "cases.ICUOffset" in src
    assert "sic_death requires cases.ICUOffset" in src
