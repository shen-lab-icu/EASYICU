"""Time placement of MIMIC-III ``prescriptions`` rows (``mimic_abx_presc``).

MIMIC-III prescriptions carry only a calendar date (``startdate`` is always
00:00).  ricu's ``mimic_abx_presc`` therefore moves each order to the middle of
its day, ``startdate + mins(720L)``, before the row is placed on the ICU clock.

The shift is twelve hours whatever representation the index has when the
callback runs.  The loader hands this callback absolute ``datetime64`` values;
adding the bare number 720 to their integer form moved the order by 720
microseconds, kept it at midnight, and turned the column into an epoch integer
that only a later guard converted back.  ICU-relative numeric indexes in
EasyICU are hours.

Every MIMIC-III prescription concept uses this placement (2026-11-03 audit
round 2, D5): at midnight an order written on the day of ICU admission sat
about ten hours before the drug was given and read as pre-ICU exposure.
Non-antibiotic sources declare ``mimic_prescription_noon``, the same rule.
The time is still a calendar day, so these concepts are day-level exposure
and do not support an hourly landmark; a multi-day order marks its first day
only.
"""

from __future__ import annotations

import pandas as pd

PRESCRIPTION_DAY_OFFSET_MINUTES = 720
PRESCRIPTION_NOON_CALLBACK = "mimic_prescription_noon"


def shift_index_minutes(values: pd.Series, minutes: int = PRESCRIPTION_DAY_OFFSET_MINUTES) -> pd.Series:
    """Return ``values`` moved forward by ``minutes``, keeping their type.

    * ``datetime64`` (naive or tz-aware) and ``timedelta64`` move by a
      :class:`pandas.Timedelta`;
    * numeric values are ICU-relative hours and move by ``minutes / 60``;
    * object columns are parsed as datetimes; values that do not parse raise
      instead of being silently reinterpreted as numbers.
    """
    offset = pd.Timedelta(minutes=minutes)
    if pd.api.types.is_datetime64_any_dtype(values) or pd.api.types.is_timedelta64_dtype(values):
        return values + offset
    if pd.api.types.is_numeric_dtype(values) and not pd.api.types.is_bool_dtype(values):
        return values + minutes / 60.0
    parsed = pd.to_datetime(values, errors="coerce")
    unparsed = values.notna() & parsed.isna()
    if bool(unparsed.any()):
        raise ValueError(
            "mimic_abx_presc: prescription index values are neither datetimes nor "
            f"ICU-relative hours ({int(unparsed.sum())} unparseable values)"
        )
    return parsed + offset


def prescription_exposure(frame, source, concept_name, *, icu_stays=None):
    """MIIV positive-dose exposure on a valid, half-open prescription interval.

    Prescriptions have no execution-status column. If a caller supplies an
    execution status, explicit non-administration wins; an explicit given
    status can establish exposure when the dose is missing, never when it is
    zero/negative. Drug strength text is not dosage evidence.
    """
    from ..io.ts_utils import expand
    from .eicu_medication_evidence import _positive_dose
    from .mimic_support import source_values

    if frame.empty:
        return frame.assign(**{concept_name: pd.Series(dtype=bool)})
    time_col = source.index_var or "starttime"
    end_col = source.params.get("stop_var", "stoptime")
    id_col = next(c for c in ("stay_id", "icustay_id") if c in frame)
    dose = source_values(frame, source, "dose_val_rx", concept_name)
    if "dose_val_rx" not in frame and source.value_var in (None, "dose_val_rx") and concept_name in frame:
        dose = frame[concept_name]
    positive = _positive_dose(dose).fillna(False)
    status = frame.get("statusdescription", pd.Series(pd.NA, index=frame.index, dtype="string"))
    status = status.astype("string").str.strip().str.casefold()
    given = status.isin(["given", "administered", "complete", "completed"])
    denied = status.isin(["not given", "not administered", "held", "refused", "cancelled", "canceled", "rewritten", "discontinued"])
    # Only genuinely missing dose can use the given-status exception. An
    # unparseable or zero/negative dose cannot be laundered through a status.
    missing = dose.isna() | dose.astype("string").str.strip().eq("").fillna(False)
    valid = (positive | (given & missing)) & ~denied
    valid &= frame[end_col].notna() & frame[time_col].notna() & frame[end_col].gt(frame[time_col])
    spans = frame.loc[valid, [id_col, time_col, end_col]].assign(**{concept_name: True})
    result = expand(spans, start_var=time_col, end_var=end_col, id_cols=[id_col],
                    step_size=pd.Timedelta(hours=1), keep_vars=[concept_name],
                    admission_times=icu_stays, end_inclusive=False)
    result.attrs["medication_exposure"] = {"kind": "positive_dose_prescription", "interval": "[starttime,stoptime)"}
    return result


def ordered_iv_proxy(frame, *, source, concept_name, data_source=None):
    """Day-level IV order evidence, suppressed by same-day administered evidence.

    An order's end date never becomes an administration interval. Source
    semantic parameters retain ordered/administered, day precision and IV
    route in the exported metadata lineage.
    """
    from ..datasource import FilterOp, FilterSpec

    route = frame["route"].astype("string").str.strip().str.upper()
    # Recognize explicit IV forms, including IVPB; a mixed IV/PO order does
    # not establish a uniquely intravenous route.
    out = frame.loc[route.str.fullmatch(r"IV(?:\s*(?:DRIP|PUSH|BOLUS|INFUSION|PB|P|B))?", na=False)].copy()
    time_col = source.index_var or "startdate"
    unavailable = []
    if data_source is not None and not out.empty:
        id_col = "icustay_id" if "icustay_id" in out else "stay_id"
        days = pd.to_datetime(out[time_col]).dt.normalize()
        remove = pd.Series(False, index=out.index)
        for administered in source.params.get("administered_sources", []):
            table, items = administered["table"], administered["ids"]
            clock = "charttime" if table == "inputevents_cv" else "starttime"
            columns = [id_col, clock, "amount", "rate"]
            if table == "inputevents_mv":
                columns.append("statusdescription")
            try:
                given = data_source.load_table(
                    table, columns=columns,
                    filters=[FilterSpec("itemid", FilterOp.IN, items),
                             FilterSpec(id_col, FilterOp.IN, out[id_col].dropna().unique().tolist())],
                    verbose=False,
                )
            except (KeyError, FileNotFoundError):
                # A CareVue-only source can lack the MetaVision table. This
                # does not erase the observable order or certify a dose.
                unavailable.append(table)
                continue
            given = given.data if hasattr(given, "data") else given
            if given.empty:
                continue
            valid = pd.to_numeric(given["amount"], errors="coerce").gt(0) | pd.to_numeric(given["rate"], errors="coerce").gt(0)
            if "statusdescription" in given:
                valid &= ~given.statusdescription.astype("string").str.casefold().eq("rewritten").fillna(False)
            evidence = given.loc[valid, [id_col, clock]].copy()
            evidence["__day"] = pd.to_datetime(evidence[clock]).dt.normalize()
            keys = pd.MultiIndex.from_frame(evidence[[id_col, "__day"]])
            candidates = pd.MultiIndex.from_arrays([out[id_col], days])
            remove |= candidates.isin(keys)
        out = out.loc[~remove]
    out[time_col] = shift_index_minutes(out[time_col])
    out[concept_name] = True
    out = out.drop(columns=["enddate", "endtime", "dur_var"], errors="ignore")
    out.attrs["medication_exposure"] = {"kind": "ordered", "time_precision": "day", "route": "iv",
                                        "unavailable_administered_sources": unavailable}
    return out
