"""MIMIC-IV eMAR loop-diuretic doses (2026-11-03 extraction audit round 2, D1/D3)."""

from __future__ import annotations

import json
from importlib import resources

import pandas as pd
import pytest

from easyicu.concept.emar_diuretics import (
    classify_route,
    drop_inputevent_duplicates,
    event_doses,
)

DICTIONARY = json.loads(
    resources.files("easyicu.data").joinpath("concept-dict.json").read_text(encoding="utf-8")
)


def test_event_dose_sums_detail_rows_and_takes_the_parent_route():
    details = pd.DataFrame({
        "emar_id": ["a", "a", "a", "b", "b"],
        "parent_field_ordinal": [None, 1.1, 1.2, None, 1.1],
        "dose_given": [None, "20", "20", None, "40"],
        "dose_given_unit": [None, "mg", "mg", None, "mg"],
        "route": ["PO", None, None, None, None],
    })
    out = event_doses(details)
    assert out.loc["a", ["_dose", "_unit", "_emar_route"]].tolist() == [40.0, "mg", "PO"]
    assert out.loc["b", "_dose"] == 40.0 and pd.isna(out.loc["b", "_emar_route"])


def test_route_classes():
    routes = pd.Series(["PO/NG", "IV", "IV DRIP", "NG", None, "TP", "IV/PO", "PO/IV", "IVPB"])
    assert classify_route(routes).tolist() == ["oral", "iv", "iv", "oral", pd.NA, pd.NA, pd.NA, pd.NA, "iv"]


def test_an_emar_dose_charted_in_inputevents_within_an_hour_is_dropped():
    t = pd.Timestamp("2150-01-01 10:00")
    # R6 RR-07: real inputevents has no eMAR ID; match drug, IV mass and time.
    doses = pd.DataFrame({"hadm_id": [1, 1, 2], "emar_id": ["a", "b", "c"],
                          "_route": "iv", "_dose": 40., "_unit": "mg",
                          "charttime": [t, t + pd.Timedelta(hours=3), t]})
    inputs = pd.DataFrame({"hadm_id": [1, 2], "itemid": 221794, "amount": 40., "amountuom": "mg",
                          "starttime": [t + pd.Timedelta(minutes=45),
                                                           t + pd.Timedelta(minutes=61)]})
    kept = drop_inputevent_duplicates(doses, inputs, key="hadm_id", time_col="charttime", drug="furosemide")
    assert kept.index.tolist() == [1, 2]


def test_d1_miiv_dose_reads_emar_for_each_loop_diuretic():
    sources = [s for s in DICTIONARY["loop_diuretic_furosemide_eq"]["sources"]["miiv"]
               if s["table"] == "emar"]
    assert {s["fe_drug"] for s in sources} == {"furosemide", "bumetanide", "torasemide"}
    for source in sources:
        assert source["callback"] == "emar_furosemide_equivalent"
        assert {"event_txt", "emar_id", "pharmacy_id"} <= set(source["extra_vars"])


@pytest.mark.parametrize(
    "concept",
    ["loop_diuretic", "thiazide_diuretic", "potassium_sparing_diuretic", "acetazolamide"],
)
def test_d3_miiv_diuretic_exposure_is_an_administered_emar_event(concept):
    sources = DICTIONARY[concept]["sources"]["miiv"]
    assert not [s for s in sources if s["table"] == "prescriptions"]
    [emar] = [s for s in sources if s["table"] == "emar"]
    assert emar["callback"] == "emar_administered"
