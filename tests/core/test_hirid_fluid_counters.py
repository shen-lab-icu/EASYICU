"""HiRID daily fluid counters (30005010 input, 30005110 output) become hourly volumes.

Both variables restart once a day at 12:00. Their level is never a balance;
only the within-day rise is fluid that moved in that interval.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from easyicu.concept import ConceptSource, _apply_callback
from easyicu.concept.hirid_fluid_counters import (
    INPUT_COUNTER_ID,
    OUTPUT_COUNTER_ID,
    counter_increments,
    hirid_counter_frame,
)

DICTIONARY = Path(__file__).resolve().parents[2] / "src/easyicu/data/concept-dict.json"
ADMISSION = pd.Timestamp("2020-01-01 09:30")
ORIGINS = pd.Series([ADMISSION, ADMISSION], index=[1, 2])


def _readings(patient, variable, times, values):
    return pd.DataFrame(
        {
            "patientid": patient,
            "variableid": variable,
            "datetime": pd.to_datetime(times),
            "value": values,
        }
    )


def _relative(frame):
    frame = frame.copy()
    frame["datetime"] = (frame["datetime"] - ADMISSION).dt.total_seconds() / 3600.0
    return frame


def test_increments_follow_the_noon_restart_and_ignore_corrections_and_sentinels():
    readings = _readings(
        1,
        INPUT_COUNTER_ID,
        [
            "2020-01-01 10:00",
            "2020-01-01 11:00",
            "2020-01-01 11:30",  # downward correction inside the day
            "2020-01-01 11:45",
            "2020-01-01 13:00",  # first reading after the 12:00 restart
            "2020-01-01 14:00",  # int64 overflow sentinel
        ],
        [100.0, 500.0, 300.0, 520.0, 80.0, 9.2e18],
    )

    increments = counter_increments(
        readings, id_col="patientid", time_col="datetime", value_col="value"
    )

    assert increments["volume"].tolist() == pytest.approx([100.0, 400.0, 0.0, 20.0, 80.0])
    # The day's total is its highest reading, not the sum of every rise.
    assert increments["volume"].iloc[:4].sum() == pytest.approx(520.0)
    restart = increments.iloc[4]
    assert restart["start"] == pd.Timestamp("2020-01-01 12:00")
    assert restart["end"] == pd.Timestamp("2020-01-01 13:00")


def test_hourly_input_is_admission_anchored_and_conserves_volume():
    readings = _readings(
        1,
        INPUT_COUNTER_ID,
        ["2020-01-01 10:00", "2020-01-01 11:00", "2020-01-01 13:00"],
        [100.0, 500.0, 50.0],
    )

    hourly = hirid_counter_frame(
        _relative(readings),
        concept_name="total_input_ml",
        origins=ORIGINS,
        value_col="value",
        time_col="datetime",
    )

    # The first day's restart preceded admission: no volume before hour 0.
    assert hourly["datetime"].tolist() == [0.0, 1.0, 2.0, 3.0]
    assert hourly["total_input_ml"].tolist() == pytest.approx([300.0, 200.0, 25.0, 25.0])
    assert hourly["total_input_ml"].sum() == pytest.approx(550.0)


def test_net_balance_needs_both_counters_for_a_patient():
    readings = pd.concat(
        [
            _readings(1, INPUT_COUNTER_ID, ["2020-01-01 10:30"], [300.0]),
            _readings(1, OUTPUT_COUNTER_ID, ["2020-01-01 10:30"], [120.0]),
            # Patient 2 has intake only: input - 0 would be a fabricated balance.
            _readings(2, INPUT_COUNTER_ID, ["2020-01-01 10:30"], [700.0]),
        ],
        ignore_index=True,
    )

    balance = hirid_counter_frame(
        _relative(readings),
        concept_name="fluid_balance",
        origins=ORIGINS,
        value_col="value",
        time_col="datetime",
        net_balance=True,
    )

    assert balance["patientid"].unique().tolist() == [1]
    assert balance["fluid_balance"].sum() == pytest.approx(180.0)


def test_net_balance_without_any_output_counter_is_empty():
    readings = _readings(2, INPUT_COUNTER_ID, ["2020-01-01 10:30"], [700.0])

    balance = hirid_counter_frame(
        _relative(readings),
        concept_name="fluid_balance",
        origins=ORIGINS,
        value_col="value",
        time_col="datetime",
        net_balance=True,
    )

    assert balance.empty
    assert list(balance.columns) == ["patientid", "datetime", "fluid_balance"]


def test_absolute_datetimes_keep_their_representation():
    readings = _readings(1, INPUT_COUNTER_ID, ["2020-01-01 10:30"], [60.0])

    hourly = hirid_counter_frame(
        readings,
        concept_name="total_input_ml",
        origins=ORIGINS,
        value_col="value",
        time_col="datetime",
    )

    assert hourly["datetime"].tolist() == [ADMISSION]
    assert hourly["total_input_ml"].tolist() == pytest.approx([60.0])


class _HiridSource:
    config = SimpleNamespace(name="hirid")

    def load_table(self, table, columns=None, verbose=False):
        assert table == "general"
        return pd.DataFrame({"patientid": [1, 2], "admissiontime": [ADMISSION, ADMISSION]})


def _dictionary():
    return json.loads(DICTIONARY.read_text(encoding="utf-8"))


def test_dictionary_reads_the_counters_through_the_owner_callbacks():
    dictionary = _dictionary()
    total_input = dictionary["total_input_ml"]["sources"]["hirid"]
    balance = dictionary["fluid_balance"]["sources"]["hirid"]

    assert [(s["ids"], s["callback"]) for s in total_input] == [
        ([INPUT_COUNTER_ID], "hirid_fluid_counter_hourly")
    ]
    assert [(sorted(s["ids"]), s["callback"]) for s in balance] == [
        ([INPUT_COUNTER_ID, OUTPUT_COUNTER_ID], "hirid_fluid_balance_counters")
    ]
    # The input counter's level is not a cumulative net balance; the
    # cumulative concept must accumulate the hourly balance instead.
    assert "sources" not in dictionary["fluid_balance_cumulative"]

    source = ConceptSource.from_mapping(total_input[0])
    frame = _relative(
        _readings(1, INPUT_COUNTER_ID, ["2020-01-01 10:00", "2020-01-01 10:20"], [10.0, 40.0])
    )
    result = _apply_callback(
        frame, source, concept_name="total_input_ml", data_source=_HiridSource()
    )
    assert result["datetime"].tolist() == [0.0]
    assert result["total_input_ml"].tolist() == pytest.approx([40.0])


def test_a_drop_below_half_the_running_maximum_is_an_off_schedule_restart():
    readings = _readings(
        1,
        OUTPUT_COUNTER_ID,
        ["2020-01-01 13:00", "2020-01-01 15:00", "2020-01-01 16:00", "2020-01-01 18:00", "2020-01-01 19:00"],
        # 800 -> 650 is a correction; 650 -> 50 restarts the counter.
        [400.0, 800.0, 650.0, 50.0, 250.0],
    )
    out = counter_increments(readings, id_col="patientid", time_col="datetime", value_col="value")
    assert out["volume"].tolist() == [400.0, 400.0, 0.0, 50.0, 200.0]


def test_net_balance_is_missing_on_a_day_with_only_the_input_counter():
    readings = pd.concat(
        [
            _readings(1, INPUT_COUNTER_ID, ["2020-01-01 10:00", "2020-01-01 14:00"], [100.0, 200.0]),
            _readings(1, OUTPUT_COUNTER_ID, ["2020-01-01 10:00"], [50.0]),
        ],
        ignore_index=True,
    )
    out = hirid_counter_frame(
        _relative(readings), concept_name="fluid_balance", origins=ORIGINS,
        value_col="value", time_col="datetime", net_balance=True,
    )
    # The 12:00-13:00 day before admission has both counters; the next day has input only.
    hours = out["datetime"].tolist()
    assert hours and max(hours) < 2.5
