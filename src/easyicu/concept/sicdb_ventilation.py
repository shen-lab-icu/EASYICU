"""SICdb invasive ventilation attributed to a documented airway.

SICdb records two kinds of ventilation evidence:

* ``data_range`` airway intervals (d_references ``Airway``): 720
  ``EndotrachealTube`` and 3041 ``TrachealCannula``, each with ``Offset`` and
  ``OffsetEnd`` (primary-admission seconds, normalized using cases.ICUOffset);
* hourly ventilator settings in ``data_float_h`` (``RespiratorSetting``):
  2278 PEEP, 2282 RespRate, 2284 PressureSupport, 3040 PressureOverPeep,
  2285 IERatio and 3121 Trigger.

SICdb has no mask/helmet (interface) item, so a ventilator setting by itself
cannot tell invasive from non-invasive ventilation.  The rule here is the
conservative one: a setting hour is invasive ventilation only when it falls in
(or within one hour of) a documented airway interval.  An endotracheal tube
interval is invasive ventilation as a whole (as before); a tracheal cannula
counts only in the hours a ventilator setting is charted, because a
tracheostomy without the ventilator is not ventilation.  ``FIO2`` (2283) alone
does not establish positive-pressure ventilation and is not read.

Many intubated patients (mostly after cardiac surgery) have settings but no
airway interval (2026-11-03 extraction audit round 2, P1: 11,769 cases).  In
the intensive care units 2-4 (``cases.HospitalUnit``; unit 5 is the mixed ward
that also runs NIV), a setting hour with a set rate (2282) or a
PressureOverPeep (3040) while propofol is running (3057 > 0) is therefore
invasive ventilation under the stronger primary heuristic. Its attribution
label remains distinct from a documented airway interval.
An opioid-only setting hour is retained as a sensitivity candidate, not in
the primary definition (R5-SIC-015). A measured end-tidal CO2 (716 > 10 mmHg)
beside a controlled setting supports the primary rule, like propofol
(2026-11-04 audit round 3, M P1: 72,870 setting hours in units 2-4 had
neither an airway interval nor propofol).  Dexmedetomidine and clonidine are
also given for NIV tolerance and weaning and do not count.
``attributed_ventilation_windows`` without ``sedated_hours`` is the strict
airway-only rule, kept for sensitivity analyses.  Setting hours attributed by
neither rule are counted on the frame's ``attrs`` so the extraction can report
them.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd

from .immutable_receipts import freeze_sic_attrs

AIRWAY_ITEMS = (720, 3041)
SETTING_ITEMS = (2278, 2282, 2284, 3040, 2285, 3121)
#: Settings that, under propofol in an ICU unit, mean controlled ventilation.
CONTROLLED_SETTING_ITEMS = (2282, 3040)
PROPOFOL_ITEM = 3057
#: Primary hypnotic support; opioids are sensitivity evidence only.
SEDATION_ITEMS = (PROPOFOL_ITEM, 3059, 3063)
#: End-tidal CO2 (mmHg); above ``ETCO2_MIN_MMHG`` it is measured on an airway.
ETCO2_ITEM = 716
ETCO2_MIN_MMHG = 10.0
#: cases.HospitalUnit codes of the intensive care units (5 is the NIV ward).
ICU_UNITS = (2, 3, 4)
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
    sedated_hours: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Merge airway-attributed setting hours into invasive windows.

    ``settings`` carries ``id_col`` and ``time_col`` (seconds); ``airway``
    carries ``id_col``, ``Offset`` and ``OffsetEnd`` (seconds).
    ``sedated_hours`` (``id_col`` and hour index ``_h``) are setting hours
    attributed without an airway (see the module docstring); ``None`` is the
    strict airway-only rule.  The result has one row per run of consecutive
    attributed hours: ``time_col`` (run start, seconds), ``dur_var`` (seconds)
    and ``concept_name`` = ``'invasive'``.
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
    start, stop = pd.to_numeric(spans.Offset, errors="coerce"), pd.to_numeric(spans.OffsetEnd, errors="coerce")
    spans = spans.loc[np.isfinite(start) & np.isfinite(stop) & stop.gt(start)].copy()
    spans["_a0"] = pd.to_numeric(spans["Offset"], errors="coerce") - AIRWAY_TOLERANCE_SECONDS
    spans["_a1"] = pd.to_numeric(spans["OffsetEnd"], errors="coerce") + AIRWAY_TOLERANCE_SECONDS
    pairs = hours.merge(spans[[id_col, "_a0", "_a1"]].dropna(), on=id_col, how="inner")
    start = pairs["_h"] * HOUR
    overlap = (start < pairs["_a1"]) & (start + HOUR > pairs["_a0"])
    attributed = pairs.loc[overlap, [id_col, "_h"]]
    airway_keys = pd.MultiIndex.from_frame(attributed[[id_col, "_h"]])
    documented = pairs.loc[(start < pairs._a1 - HOUR) & (start + HOUR > pairs._a0 + HOUR)]
    documented_keys = pd.MultiIndex.from_frame(documented[[id_col, "_h"]])
    if sedated_hours is not None and not sedated_hours.empty:
        sedated = hours.merge(sedated_hours[[id_col, "_h"]].drop_duplicates(), on=[id_col, "_h"])
        attributed = pd.concat([attributed, sedated], ignore_index=True)
    attributed = attributed.drop_duplicates()
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
    if sedated_hours is not None and not sedated_hours.empty and "sicdb_ventilation_rule" in sedated_hours:
        labelled = attributed.merge(sedated_hours[[id_col, "_h", "sicdb_ventilation_rule"]], on=[id_col, "_h"], how="left")
    else:
        labelled = attributed.assign(sicdb_ventilation_rule="airway_setting_tolerance")
    labelled["sicdb_ventilation_rule"] = labelled.sicdb_ventilation_rule.fillna("airway_setting_tolerance")
    keys = pd.MultiIndex.from_frame(labelled[[id_col, "_h"]])
    labelled.loc[keys.isin(airway_keys), "sicdb_ventilation_rule"] = "airway_setting_tolerance"
    labelled.loc[keys.isin(documented_keys), "sicdb_ventilation_rule"] = "airway_setting_overlap"
    out.attrs["sicdb_ventilation_attribution"] = labelled[[id_col, "_h", "sicdb_ventilation_rule"]].to_dict("records")
    candidates = [] if sedated_hours is None else sedated_hours.attrs.get("sicdb_opioid_only_hours", [])
    if candidates:
        candidates = pd.DataFrame(candidates)
        covered = pd.MultiIndex.from_frame(attributed[[id_col, "_h"]])
        candidates = candidates.loc[~pd.MultiIndex.from_frame(candidates[[id_col, "_h"]]).isin(covered)].to_dict("records")
    out.attrs["sicdb_opioid_only_hours"] = candidates
    out.attrs["sicdb_ventilation_coverage_delta"] = {"main_setting_hours": len(attributed), "additional_opioid_only_sensitivity_hours": len(candidates)}
    return freeze_sic_attrs(set_dur_var_unit(out, UNIT_SECONDS))


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
    from .sicdb_sources import icu_seconds, materialize_intervals

    if frame.empty:
        windows = attributed_ventilation_windows(
            frame, pd.DataFrame(columns=[id_col, "Offset", "OffsetEnd"]),
            concept_name=concept_name, id_col=id_col, time_col=time_col,
        )
        # Match the nonempty hourly schema. A phantom dur_var from this
        # empty source makes the resolver discard the other source's points.
        windows["OffsetEnd"] = windows[time_col]
        return materialize_intervals(windows, start=time_col, stop="OffsetEnd", relative=True)
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
    sedated = _sedated_controlled_hours(frame, data_source, cases, id_col, time_col)
    windows = attributed_ventilation_windows(
        icu_seconds(frame, data_source, [time_col]),
        icu_seconds(airway, data_source, ["Offset", "OffsetEnd"]),
        concept_name=concept_name, id_col=id_col, time_col=time_col,
        sedated_hours=sedated,
    )
    receipt = dict(windows.attrs)
    windows["OffsetEnd"] = windows[time_col] + windows.dur_var
    out = materialize_intervals(windows, start=time_col, stop="OffsetEnd", relative=True)
    out.attrs.update({k: v for k, v in receipt.items() if k != "easyicu_duration_units"})
    return freeze_sic_attrs(out)


def _sedated_controlled_hours(
    frame: pd.DataFrame, data_source: Any, cases: list, id_col: str, time_col: str
) -> pd.DataFrame:
    """Primary controlled hours under propofol or ETCO2; opioids are sensitivity."""
    from ..datasource import FilterOp, FilterSpec

    empty = pd.DataFrame(columns=[id_col, "_h"])
    if "DataID" not in frame.columns:
        raise ValueError("sicdb_setting_item_column_missing:DataID")
    controlled = frame.loc[frame["DataID"].isin(CONTROLLED_SETTING_ITEMS), [id_col, time_col]]
    if controlled.empty:
        return empty
    units = data_source.load_table(
        "cases",
        columns=[id_col, "HospitalUnit", "ICUOffset"],
        filters=[FilterSpec(id_col, FilterOp.IN, cases)],
        verbose=False,
    )
    units = units.data if hasattr(units, "data") else units
    icu_cases = units.loc[units["HospitalUnit"].isin(ICU_UNITS), id_col]
    controlled = controlled.loc[controlled[id_col].isin(icu_cases)]
    if controlled.empty:
        return empty
    evidence = data_source.load_table(
        "data_float_h",
        columns=[id_col, "DataID", "Offset", "Val"],
        filters=[
            FilterSpec("DataID", FilterOp.IN, [*SEDATION_ITEMS, ETCO2_ITEM]),
            FilterSpec(id_col, FilterOp.IN, pd.unique(controlled[id_col]).tolist()),
        ],
        verbose=False,
    )
    evidence = evidence.data if hasattr(evidence, "data") else evidence
    value = pd.to_numeric(evidence["Val"], errors="coerce")
    evidence = evidence.loc[
        (evidence["DataID"].isin(SEDATION_ITEMS) & (value > 0))
        | (evidence["DataID"].eq(ETCO2_ITEM) & (value > ETCO2_MIN_MMHG))
    ].copy()
    evidence["sicdb_ventilation_rule"] = np.where(
        evidence.DataID.eq(PROPOFOL_ITEM), "propofol_controlled",
        np.where(evidence.DataID.eq(ETCO2_ITEM), "etco2_controlled", "opioid_only_sensitivity"),
    )
    origins = units.set_index(id_col).ICUOffset
    for rows, time in [(controlled, time_col), (evidence, "Offset")]:
        origin = pd.to_numeric(rows[id_col].map(origins), errors="coerce")
        if not np.isfinite(origin).all():
            raise ValueError("SICdb ventilation requires cases.ICUOffset")
        rows[time] = pd.to_numeric(rows[time], errors="coerce") - origin
    indexed = _hour_index(evidence, id_col, "Offset")
    # Preserve rule labels before deduplicating individual evidence hours.
    labelled = evidence[[id_col, "sicdb_ventilation_rule"]].copy()
    labelled["_h"] = np.floor(evidence.Offset / HOUR)
    hours = _hour_index(controlled, id_col, time_col).merge(indexed, on=[id_col, "_h"])
    labelled = hours.merge(labelled.drop_duplicates(), on=[id_col, "_h"])
    primary = labelled.loc[labelled.sicdb_ventilation_rule.ne("opioid_only_sensitivity")].drop_duplicates([id_col, "_h"]).copy()
    keys = pd.MultiIndex.from_frame(primary[[id_col, "_h"]])
    opioid = labelled.loc[labelled.sicdb_ventilation_rule.eq("opioid_only_sensitivity")]
    primary.attrs["sicdb_opioid_only_hours"] = opioid.loc[~pd.MultiIndex.from_frame(opioid[[id_col, "_h"]]).isin(keys)].to_dict("records")
    return freeze_sic_attrs(primary)


def _hour_index(frame: pd.DataFrame, id_col: str, time_col: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            id_col: frame[id_col].to_numpy(),
            "_h": np.floor(pd.to_numeric(frame[time_col], errors="coerce") / HOUR),
        }
    ).dropna().drop_duplicates()


__all__ = [
    "AIRWAY_ITEMS",
    "CALLBACK",
    "CONTROLLED_SETTING_ITEMS",
    "ICU_UNITS",
    "PROPOFOL_ITEM",
    "SETTING_ITEMS",
    "apply_sicdb_attributed_ventilation",
    "attributed_ventilation_windows",
]
