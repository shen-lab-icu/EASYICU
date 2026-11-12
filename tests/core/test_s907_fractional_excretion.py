"""S907: FENa / FEUrea from a spot urine pair and the latest earlier plasma values."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from easyicu.concept import callbacks as concept_callbacks
from easyicu.concept.fractional_excretion import (
    FENA,
    FEUREA,
    RECEIPT_ATTR,
    fractional_excretion,
)
from easyicu.resources import load_dictionary

ROOT = Path(__file__).resolve().parents[2]


def _frame(rows):
    columns = ["stay_id", "charttime", "urine_na", "urine_crea", "na", "crea"]
    return pd.DataFrame(rows, columns=columns)


def test_fena_uses_same_bucket_urine_and_latest_earlier_plasma():
    frame = _frame([
        (1, 0.0, None, None, 140.0, 1.0),
        (1, 3.0, None, None, 130.0, None),
        (1, 5.0, 20.0, 100.0, None, None),
        # Plasma charted after the urine sample never feeds it.
        (1, 6.0, None, None, 100.0, 5.0),
        # Separate stay: its plasma must not cross the stay boundary.
        (2, 1.0, 20.0, 100.0, None, None),
    ])
    out = fractional_excretion(frame, spec=FENA, id_columns=["stay_id"], index_column="charttime")
    assert out[["stay_id", "charttime"]].values.tolist() == [[1, 5.0]]
    # 100 * (20 * 1.0) / (130 * 100)
    assert out.fena.iloc[0] == pytest.approx(100 * 20 * 1.0 / (130 * 100))
    receipt = out.attrs[RECEIPT_ATTR]
    assert receipt["urine_pairs"] == 2
    assert receipt["pairs_without_plasma_in_lookback"] == 1
    assert receipt["values"] == 1


def test_plasma_older_than_lookback_and_split_urine_pairs_give_no_value():
    frame = _frame([
        (1, 0.0, None, None, 140.0, 1.0),
        (1, 24.0, 20.0, 100.0, None, None),  # exactly 24 h: inside (inclusive)
        (1, 24.5, 20.0, 100.0, None, None),  # older than 24 h: no value
        (1, 30.0, 20.0, None, 140.0, 1.0),   # urine creatinine missing in the bucket
    ])
    out = fractional_excretion(frame, spec=FENA, id_columns=["stay_id"], index_column="charttime")
    assert out.charttime.tolist() == [24.0]
    shorter = fractional_excretion(
        frame, spec=FENA, id_columns=["stay_id"], index_column="charttime",
        lookback=pd.Timedelta(hours=6),
    )
    assert shorter.empty
    assert shorter.attrs[RECEIPT_ATTR]["pairs_without_plasma_in_lookback"] == 2


def test_bounds_are_inclusive_and_nonpositive_denominators_are_dropped():
    frame = _frame([
        (1, 0.0, 0.0, 100.0, 140.0, 1.0),     # FENa 0 %: kept
        (1, 1.0, 140.0, 1.0, 140.0, 1.0),     # FENa 100 %: kept
        (1, 2.0, 200.0, 1.0, 140.0, 1.0),     # > 100 %: dropped
        (1, 3.0, 20.0, 0.0, 140.0, 1.0),      # urine creatinine 0: undefined
    ])
    out = fractional_excretion(frame, spec=FENA, id_columns=["stay_id"], index_column="charttime")
    assert out.fena.tolist() == [0.0, 100.0]
    receipt = out.attrs[RECEIPT_ATTR]
    assert receipt["values_outside_bounds"] == 1
    assert receipt["pairs_with_nonpositive_denominator"] == 1


def test_timedelta_index_and_feurea_spec():
    frame = pd.DataFrame({
        "stay_id": [1, 1],
        "charttime": pd.to_timedelta([0, 2], unit="h"),
        "urine_urea": [np.nan, 500.0],
        "urine_crea": [np.nan, 100.0],
        "bun": [20.0, np.nan],
        "crea": [1.0, np.nan],
    })
    out = fractional_excretion(frame, spec=FEUREA, id_columns=["stay_id"], index_column="charttime")
    assert out.feurea.tolist() == pytest.approx([100 * 500 * 1.0 / (20 * 100)])


def test_missing_inputs_yield_an_empty_typed_frame():
    frame = _frame([]).drop(columns=["na"])
    out = fractional_excretion(frame, spec=FENA, id_columns=["stay_id"], index_column="charttime")
    assert out.empty and list(out) == ["stay_id", "charttime", "fena"]
    assert out.attrs[RECEIPT_ATTR]["missing_inputs"] == ["na"]


@pytest.mark.parametrize("name,spec", [("fena", FENA), ("feurea", FEUREA)])
def test_dictionary_declares_the_owner_inputs_and_registry_entry(name, spec):
    definition = load_dictionary()[name]
    assert definition.class_name == "rec_cncpt" and definition.callback == name
    assert definition.sub_concepts == list(spec.inputs)
    assert definition.minimum == spec.lower and definition.maximum == spec.upper
    assert name in concept_callbacks.CALLBACK_REGISTRY
    payload = json.loads((ROOT / "src/easyicu/data/concept-dict.json").read_text())
    assert "24 hours" in payload[name]["_comment"]

