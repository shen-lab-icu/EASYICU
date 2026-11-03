"""Intake coverage contracts: CareVue intake, AUMC IV categories, balance evidence."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from easyicu.concept import ConceptSource, _apply_callback
from easyicu.concept.callbacks import _callback_fluid_balance_hourly
from easyicu.table import ICUTable

DICTIONARY = Path(__file__).resolve().parents[2] / "src/easyicu/data/concept-dict.json"


def _sources(concept, database):
    dictionary = json.loads(DICTIONARY.read_text(encoding="utf-8"))
    return dictionary[concept]["sources"][database]


def _hourly(frame, value):
    return ICUTable(
        data=frame,
        id_columns=["icustay_id"],
        index_column="charttime",
        value_column=value,
    )


def test_a_stay_without_intake_evidence_gets_no_balance():
    intake = _hourly(
        pd.DataFrame({"icustay_id": [1], "charttime": [0.0], "total_input_ml": [250.0]}),
        "total_input_ml",
    )
    urine = _hourly(
        pd.DataFrame(
            {
                "icustay_id": [1, 1, 2, 2],
                "charttime": [0.0, 1.0, 0.0, 1.0],
                "urine": [100.0, 40.0, 80.0, 60.0],
            }
        ),
        "urine",
    )

    balance = _callback_fluid_balance_hourly(
        {"total_input_ml": intake, "urine": urine}, SimpleNamespace()
    ).data

    # Stay 2's intake was never captured: 0 - urine is not a balance.
    assert balance["icustay_id"].unique().tolist() == [1]
    # Inside a stay with intake evidence an hour without input is zero input.
    assert balance["fluid_balance"].tolist() == pytest.approx([150.0, -40.0])


@pytest.mark.parametrize("database", ["mimic", "mimic_demo"])
def test_carevue_intake_is_mapped_without_pre_icu_or_non_intake_items(database):
    metavision, carevue = _sources("total_input_ml", database)
    assert metavision["table"] == "inputevents_mv"
    assert carevue["table"] == "inputevents_cv"
    assert carevue["unit_var"] == "amountuom"
    assert carevue["index_var"] == "charttime"
    assert carevue["callback"] == "distribute_volume_hourly"

    ids = set(carevue["ids"])
    assert {30013, 30018, 30021, 30001, 30005, 30056, 30032} <= ids
    pre_icu = {30054, 30055, 30101, 30102, 30103, 30104, 30105, 30106, 30107, 30108}
    irrigant_residual_unlabeled = {30168, 30169, 30170, 30171, 30095, 30140}
    assert not ids & (pre_icu | irrigant_residual_unlabeled)


class _MimicSource:
    config = SimpleNamespace(name="mimic")

    def load_table(self, table, columns=None, verbose=False):
        assert table == "icustays"
        return pd.DataFrame(
            {"icustay_id": [7], "intime": [pd.Timestamp("2130-01-01 10:20")]}
        )


def test_carevue_amounts_are_volume_units_booked_in_their_icu_hour():
    source = ConceptSource.from_mapping(_sources("total_input_ml", "mimic")[1])
    frame = pd.DataFrame(
        {
            "icustay_id": [7, 7, 7],
            "itemid": [30018, 30013, 30018],
            "charttime": pd.to_datetime(
                ["2130-01-01 11:00", "2130-01-01 11:00", "2130-01-01 12:00"]
            ),
            "total_input_ml": ["100", "40", "2"],
            "amountuom": ["ml", "cc", "mg"],
        }
    )

    result = _apply_callback(
        frame,
        source,
        concept_name="total_input_ml",
        unit_column="amountuom",
        data_source=_MimicSource(),
    )

    assert result["charttime"].tolist() == [0.0]
    assert result["total_input_ml"].tolist() == pytest.approx([140.0])


def test_aumc_intake_is_every_intravenous_order_category():
    (source,) = _sources("total_input_ml", "aumc")
    assert source["sub_var"] == "ordercategoryid"
    assert source["val_var"] == "fluidin"
    assert source["value_unit"] == "mL"

    ids = set(source["ids"])
    intravenous = {55, 17, 61, 65, 39, 15, 23, 24, 25, 26, 27, 67}
    enteral_oral_or_not_iv = {114, 63, 42, 31, 119, 21, 71, 70, 32, 69, 29}
    assert ids == intravenous
    assert not ids & enteral_oral_or_not_iv
