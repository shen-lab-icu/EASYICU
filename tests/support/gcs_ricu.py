"""Synthetic GCS inputs for ricu-alignment tests (invented values, one stay).

The reference outputs in the tests were produced by ricu 0.6.3 itself on the
same tables, built in R as ``as_ts_tbl`` components on a 1-hour interval and an
``as_win_tbl`` ``ett_gcs`` whose ``dur_var`` is in minutes, then passed to
``ricu:::gcs(egcs=, mgcs=, vgcs=, tgcs=, ett_gcs=, interval = hours(1L))``.
"""

from __future__ import annotations

from typing import Dict, Iterable, Optional

import pandas as pd

from easyicu.concept.callbacks import ConceptCallbackContext, _callback_gcs
from easyicu.table import ICUTable, WinTbl
from easyicu.table.duration import UNIT_MINUTES, set_dur_var_unit

ID = "patientunitstayid"
TIME = "charttime"


def component(name: str, hours: Iterable[float], values: Iterable[float]) -> ICUTable:
    frame = pd.DataFrame({ID: 1, TIME: [float(h) for h in hours], name: [float(v) for v in values]})
    frame[ID] = frame[ID].astype("int64")
    return ICUTable(data=frame, id_columns=[ID], index_column=TIME, value_column=name)


def airway_windows(hours: Iterable[float], minutes: Iterable[float], flags: Iterable[bool]) -> WinTbl:
    frame = pd.DataFrame({
        ID: 1, TIME: [float(h) for h in hours],
        "dur_var": [float(m) for m in minutes], "ett_gcs": list(flags),
    })
    frame[ID] = frame[ID].astype("int64")
    set_dur_var_unit(frame, UNIT_MINUTES)
    return WinTbl(data=frame, id_vars=[ID], index_var=TIME, dur_var="dur_var")


def airway_points(hours: Iterable[float], flags: Iterable[bool]) -> ICUTable:
    """A point ``ett_gcs`` table (no duration), as SICdb and AUMC publish it."""
    frame = pd.DataFrame({ID: 1, TIME: [float(h) for h in hours], "ett_gcs": list(flags)})
    frame[ID] = frame[ID].astype("int64")
    return ICUTable(data=frame, id_columns=[ID], index_column=TIME, value_column="ett_gcs")


def run_gcs(
    egcs: ICUTable, mgcs: ICUTable, vgcs: ICUTable, ett_gcs: ICUTable,
    tgcs: Optional[ICUTable] = None,
) -> Dict[float, float]:
    tables = {
        "egcs": egcs, "mgcs": mgcs, "vgcs": vgcs,
        "tgcs": tgcs if tgcs is not None else component("tgcs", [], []),
        "ett_gcs": ett_gcs,
    }
    ctx = ConceptCallbackContext(
        concept_name="gcs", target=None, interval=pd.Timedelta(hours=1),
        resolver=None, data_source=None, patient_ids=None,
    )
    frame = _callback_gcs(tables, ctx).data
    return {float(h): float(v) for h, v in zip(frame[TIME], frame["gcs"])}
