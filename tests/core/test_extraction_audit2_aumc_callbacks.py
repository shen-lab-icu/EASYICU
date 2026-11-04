"""Callback contracts from the 2026-11-03 extraction audit (batch 2, AUMCdb/HiRID).

* AUMC ``death`` (ricu's 72h-after-ICU-discharge proxy) marks survivors False,
  as ``hospital_expire_flag`` does on MIMIC/eICU, instead of leaving them NA.
* ``supp_o2`` counts an oxygen device flow > 0 L/min as supplemental oxygen
  where a database declares the ``o2_flow`` input.
"""

from __future__ import annotations

import types

import pandas as pd

from easyicu.concept.callback_apply import _apply_callback
from easyicu.concept.callbacks import ConceptCallbackContext, _callback_supp_o2
from easyicu.table import ICUTable


def _death_source():
    return types.SimpleNamespace(
        callback="aumc_death", index_var="dateofdeath", value_var="dischargedat",
        sub_var=None, unit_var=None, table=None, ids=None,
    )


def test_aumc_death_marks_survivors_false_and_unknown_discharge_na():
    frame = pd.DataFrame(
        {
            "admissionid": [1, 2, 3, 4, 5],
            # dischargedat renamed to the concept name by the loader (minutes)
            "death": [1000, 1000, 1000, 1000, None],
            "dateofdeath": [900, 1000 + 71 * 60, 1000 + 200 * 24 * 60, None, None],
        }
    )
    result = _apply_callback(frame, _death_source(), "death").set_index("admissionid")
    out = result["death"]
    assert out.loc[1] is True and out.loc[2] is True
    assert out.loc[3] is False  # died long after ICU discharge: not this endpoint
    assert out.loc[4] is False  # no recorded death
    assert pd.isna(out.loc[5])  # no discharge time: unknown
    # Deaths keep their date of death, but no earlier than ICU discharge: a
    # registry date (midnight) on the day of discharge is the discharge (audit
    # round 3, R3-13).  Survivors are timed at ICU discharge so the hourly grid
    # does not drop them as untimed rows.
    assert result.loc[[1, 2], "dateofdeath"].tolist() == [1000, 1000 + 71 * 60]
    assert result.loc[[3, 4], "dateofdeath"].tolist() == [1000, 1000]


def _ctx():
    return ConceptCallbackContext(
        concept_name="supp_o2", target=None, interval=None,
        resolver=None, data_source=None, patient_ids=None,
    )


def _tbl(frame, value):
    return ICUTable(frame, id_columns=["admissionid"], index_column="measuredat", value_column=value)


def test_supp_o2_counts_positive_oxygen_flow():
    fio2 = _tbl(pd.DataFrame({"admissionid": [1], "measuredat": [0.0], "fio2": [21.0]}), "fio2")
    vent = _tbl(pd.DataFrame({"admissionid": [1], "measuredat": [5.0], "vent_ind": [False]}), "vent_ind")
    flow = _tbl(
        pd.DataFrame({"admissionid": [1, 1, 2], "measuredat": [1.0, 2.0, 3.0], "o2_flow": [2.0, 0.0, 15.0]}),
        "o2_flow",
    )
    out = _callback_supp_o2({"vent_ind": vent, "fio2": fio2, "o2_flow": flow}, _ctx()).data
    got = set(map(tuple, out[["admissionid", "measuredat"]].to_numpy().tolist()))
    assert got == {(1, 1.0), (2, 3.0)}
    assert out["supp_o2"].eq(True).all()


def test_supp_o2_without_flow_input_is_unchanged():
    fio2 = _tbl(pd.DataFrame({"admissionid": [1, 1], "measuredat": [0.0, 1.0], "fio2": [21.0, 40.0]}), "fio2")
    vent = _tbl(pd.DataFrame({"admissionid": [1], "measuredat": [2.0], "vent_ind": [True]}), "vent_ind")
    out = _callback_supp_o2({"vent_ind": vent, "fio2": fio2}, _ctx()).data
    assert sorted(out["measuredat"].tolist()) == [1.0, 2.0]
