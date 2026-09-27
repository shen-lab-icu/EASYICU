"""D10 rates must not include undelivered orders or incompatible volumes."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from easyicu.concept.callback_apply import _apply_callback
from easyicu.concept.schema import ConceptSource
from easyicu.utils.callback_utils import mimv_rate


def dictionary_sources():
    path = Path(__file__).resolve().parents[2] / "src/easyicu/data/concept-dict.json"
    return json.loads(path.read_text())["dex"]["sources"]


def mv_frame():
    return pd.DataFrame({"stay_id": range(1, 7), "itemid": [220950, 220952, 220952, 220950, 228140, 220950],
                         "rate": [50., np.nan, np.nan, 20., 0., 10.], "rateuom": ["mL/hour"]*6,
                         "amount": [50., 25., 25., 20., 10., 10.], "amountuom": ["ml"]*6,
                         "dur_var": [60., 1., 1., 60., 60., 60.],
                         "statusdescription": ["Rewritten", " rewritten ", "FinishedRunning", "Paused", "Stopped", None]})


def test_rewritten_native_and_fallback_rows_are_removed_before_calculation():
    frame = mv_frame()
    before = frame.copy(deep=True)
    result = mimv_rate(frame, val_col="rate", unit_col="rateuom", dur_var="dur_var")
    assert result.stay_id.tolist() == [3, 4, 5, 6]
    assert result.rate.tolist() == [1500., 20., 10., 10.]
    pd.testing.assert_frame_equal(frame, before)


def test_missing_status_column_cannot_silently_certify_a_rate():
    with pytest.raises(ValueError, match="delivery-status column"):
        mimv_rate(mv_frame().drop(columns="statusdescription"), val_col="rate", dur_var="dur_var")


@pytest.mark.parametrize("database", ["miiv", "mimic", "mimic_demo"])
def test_dictionary_declares_status_projection_and_full_chain_applies_concentration(database):
    mapping = dictionary_sources()[database][0]
    assert "statusdescription" in mapping["extra_vars"]
    assert mapping["status_var"] == "statusdescription"
    source = ConceptSource.from_mapping({**mapping, "val_var": "rate", "unit_var": "rateuom"})
    result = _apply_callback(mv_frame(), source, concept_name="dex", unit_column="rateuom")
    assert result.stay_id.tolist() == [3, 4, 5, 6]
    assert result.rate.tolist() == [7500., 20., 20., 10.]


@pytest.mark.parametrize("missing", ["dur_var", "amount", "statusdescription"])
def test_dispatcher_does_not_return_uncorrected_rows_when_required_source_is_missing(missing):
    source = ConceptSource.from_mapping({**dictionary_sources()["miiv"][0], "val_var": "rate"})
    with pytest.raises(ValueError, match="requires"):
        _apply_callback(mv_frame().drop(columns=missing), source, concept_name="dex", unit_column="rateuom")


def test_aumc_drop_units_do_not_become_ml_and_minute_conversion_remains():
    source = ConceptSource.from_mapping({**dictionary_sources()["aumc"][0], "val_var": "dose", "unit_var": "doseunit"})
    assert source.params["required_volume_unit"] == "ml"
    assert {"doseunit", "doserateunit"}.issubset(source.params["extra_vars"])
    frame = pd.DataFrame({"admissionid": [1, 2, 3, 4], "itemid": [8940]*4, "dose": [10.]*4,
                          "doseunit": ["druppel", " ML ", "ml", None], "doserateunit": ["uur", "uur", "min", "uur"]})
    result = _apply_callback(frame, source, concept_name="dex", unit_column="doseunit")
    assert result.admissionid.tolist() == [2, 3]
    assert result.dose.tolist() == [40., 2400.]


def test_aumc_missing_declared_volume_unit_is_an_error():
    source = ConceptSource.from_mapping({**dictionary_sources()["aumc"][0], "val_var": "dose", "unit_var": "doseunit"})
    frame = pd.DataFrame({"itemid": [8940], "dose": [30.], "doserateunit": ["uur"]})
    with pytest.raises(ValueError, match="volume-unit column"):
        _apply_callback(frame, source, concept_name="dex", unit_column="doseunit")


def test_carevue_recipe_is_not_changed_to_drop_unitless_observed_zero():
    mapping = dictionary_sources()["mimic"][1]
    assert mapping["table"] == "inputevents_cv"
    assert "required_volume_unit" not in mapping
    assert "mimv_rate" not in mapping["callback"]
