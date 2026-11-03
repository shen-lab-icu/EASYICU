"""SICdb invasive ventilation attributed to a documented airway.

SICdb records two kinds of ventilation evidence:

* ``data_range`` airway intervals (d_references ``Airway``): 720
  ``EndotrachealTube`` and 3041 ``TrachealCannula``, each with ``Offset`` and
  ``OffsetEnd`` (seconds from ICU admission);
* hourly ventilator settings in ``data_float_h`` (``RespiratorSetting``):
  2278 PEEP, 2282 RespRate, 2284 PressureSupport, 3040 PressureOverPeep,
  2285 IERatio and 3121 Trigger.

SICdb has no mask/helmet (interface) item, so a ventilator setting by itself
cannot tell invasive from non-invasive ventilation.  The rule here is the
conservative one: a setting hour is invasive ventilation only when it falls in
(or within one hour of) a documented airway interval.  An endotracheal tube
interval is invasive ventilation as a whole (as before); a tracheal cannula
counts only in the hours a ventilator setting is charted, because a
tracheostomy without the ventilator is not ventilation.  Setting hours away
from any airway are not attributed; their count is kept on the frame's
``attrs`` so the extraction can report it.  ``FIO2`` (2283) alone does not
establish positive-pressure ventilation and is not read.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

AIRWAY_ITEMS = (720, 3041)
SETTING_ITEMS = (2278, 2282, 2284, 3040, 2285, 3121)
#: Charting tolerance around an airway interval, in seconds.
AIRWAY_TOLERANCE_SECONDS = 3600
HOUR = 3600
UNATTRIBUTED_ATTR = "sicdb_unattributed_setting_hours"
CALLBACK = "sicdb_airway_attributed_ventilation"


def attributed_ventilation_windows(
    settings: pd.DataFrame,
    airway: pd.DataFrame,
    *,
    concept_name: str,
    id_col: str = "CaseID",
    time_col: str = "Offset",
) -> pd.DataFrame:
    """Merge airway-attributed setting hours into invasive windows.

    ``settings`` carries ``id_col`` and ``time_col`` (seconds); ``airway``
    carries ``id_col``, ``Offset`` and ``OffsetEnd`` (seconds).  The result has
    one row per run of consecutive attributed hours: ``time_col`` (run start,
    seconds), ``dur_var`` (seconds) and ``concept_name`` = ``'invasive'``.
    """
    from ..table.duration import UNIT_SECONDS, set_dur_var_unit

    columns = [id_col, time_col, "dur_var", concept_name]
    hours = pd.DataFrame(
        {
            id_col: settings[id_col].to_numpy(),
            "_h": np.floor(pd.to_numeric(settings[time_col], errors="coerce") / HOUR),
        }
    ).dropna().drop_duplicates()
    if hours.empty:
        out = pd.DataFrame(columns=columns)
        out.attrs[UNATTRIBUTED_ATTR] = 0
        return set_dur_var_unit(out, UNIT_SECONDS)
    spans = airway[[id_col, "Offset", "OffsetEnd"]].copy()
    spans["_a0"] = pd.to_numeric(spans["Offset"], errors="coerce") - AIRWAY_TOLERANCE_SECONDS
    spans["_a1"] = pd.to_numeric(spans["OffsetEnd"], errors="coerce") + AIRWAY_TOLERANCE_SECONDS
    pairs = hours.merge(spans[[id_col, "_a0", "_a1"]].dropna(), on=id_col, how="inner")
    start = pairs["_h"] * HOUR
    overlap = (start < pairs["_a1"]) & (start + HOUR > pairs["_a0"])
    attributed = pairs.loc[overlap, [id_col, "_h"]].drop_duplicates()
    unattributed = len(hours) - len(attributed)
    attributed = attributed.sort_values([id_col, "_h"], kind="mergesort")
    new_run = attributed.groupby(id_col)["_h"].diff().ne(1)
    attributed["_run"] = new_run.astype("int64").groupby(attributed[id_col]).cumsum()
    runs = attributed.groupby([id_col, "_run"], sort=False)["_h"].agg(["min", "max"]).reset_index()
    out = pd.DataFrame(
        {
            id_col: runs[id_col].to_numpy(),
            time_col: (runs["min"] * HOUR).to_numpy(dtype="float64"),
            "dur_var": ((runs["max"] - runs["min"] + 1) * HOUR).to_numpy(dtype="float64"),
            concept_name: "invasive",
        }
    )
    out.attrs[UNATTRIBUTED_ATTR] = int(unattributed)
    return set_dur_var_unit(out, UNIT_SECONDS)


def apply_sicdb_attributed_ventilation(
    frame: pd.DataFrame,
    *,
    concept_name: str,
    data_source: Any,
    id_col: str = "CaseID",
    time_col: str = "Offset",
) -> pd.DataFrame:
    """Source callback: load the cases' airway intervals and attribute."""
    from ..datasource import FilterOp, FilterSpec

    if frame.empty:
        return attributed_ventilation_windows(
            frame, pd.DataFrame(columns=[id_col, "Offset", "OffsetEnd"]),
            concept_name=concept_name, id_col=id_col, time_col=time_col,
        )
    if data_source is None:
        raise ValueError("sicdb_airway_attribution_requires_data_source")
    cases = pd.unique(frame[id_col].dropna()).tolist()
    table = data_source.load_table(
        "data_range",
        columns=[id_col, "DataID", "Offset", "OffsetEnd"],
        filters=[
            FilterSpec("DataID", FilterOp.IN, list(AIRWAY_ITEMS)),
            FilterSpec(id_col, FilterOp.IN, cases),
        ],
        verbose=False,
    )
    airway = table.data if hasattr(table, "data") else table
    missing = {id_col, "DataID", "Offset", "OffsetEnd"} - set(airway.columns)
    if missing:
        raise ValueError(f"sicdb_airway_columns_missing:{sorted(missing)}")
    airway = airway.loc[airway["DataID"].isin(AIRWAY_ITEMS)]
    return attributed_ventilation_windows(
        frame, airway, concept_name=concept_name, id_col=id_col, time_col=time_col
    )


__all__ = [
    "AIRWAY_ITEMS",
    "CALLBACK",
    "SETTING_ITEMS",
    "apply_sicdb_attributed_ventilation",
    "attributed_ventilation_windows",
]
