"""Source-specific rules for the S732 opt-in batch."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..utils.callback_utils import mimic_rate_mv


def angiotensin_rate(frame, source, concept_name, data_source):
    """Expand a continuous already-weight-normalized rate in ng/kg/min."""
    if frame.empty:
        return frame.copy()
    value_col = "rate" if "rate" in frame else concept_name
    start_col = source.index_var or "starttime"
    stop_col = source.params.get("stop_var", "endtime")
    required = {"stay_id", start_col, stop_col, value_col, "rateuom", "ordercategorydescription", "statusdescription"}
    missing = required - set(frame.columns)
    if missing:
        raise KeyError(f"angiotensin rate requires columns {sorted(missing)}")
    unit = frame.rateuom.astype("string").str.strip().str.lower()
    rate = pd.to_numeric(frame[value_col], errors="coerce")
    if pd.api.types.is_numeric_dtype(frame[start_col]):
        start = pd.to_numeric(frame[start_col], errors="coerce")
        stop = pd.to_numeric(frame[stop_col], errors="coerce")
    else:
        start = pd.to_datetime(frame[start_col], errors="coerce", utc=True)
        stop = pd.to_datetime(frame[stop_col], errors="coerce", utc=True)
    valid = (
        np.isfinite(rate) & rate.gt(0) & stop.gt(start)
        & unit.isin(["mcg/kg/min", "ng/kg/min"])
        & frame.ordercategorydescription.eq("Continuous Med")
        & ~frame.statusdescription.astype("string").str.strip().str.lower().isin(
            ["rewritten", "cancelled", "canceled", "bolus"]
        )
    ).fillna(False)
    out = frame.loc[valid].copy()
    out[value_col] = rate.loc[valid] * np.where(unit.loc[valid].eq("mcg/kg/min"), 1000., 1.)
    out["rateuom"] = "ng/kg/min"
    if out.empty:
        return out
    admission_times = None
    if not pd.api.types.is_numeric_dtype(out[start_col]):
        if data_source is None:
            raise ValueError("angiotensin rate requires ICU admission times")
        result = data_source.load_table("icustays", columns=["stay_id", "intime"], verbose=False)
        admission_times = result.data if hasattr(result, "data") else result
        if not set(out.stay_id).issubset(set(admission_times.stay_id)):
            raise ValueError("angiotensin rate requires an ICU admission time for each stay")
    out = mimic_rate_mv(out, val_col=value_col, unit_col="rateuom", stop_var=stop_col,
                        id_cols=["stay_id"], admission_times=admission_times)
    if value_col != concept_name:
        out = out.rename(columns={value_col: concept_name})
    return out


def miiv_gas_site(frame, source, concept_name, data_source):
    """Keep only the unanimous explicit site for the same subject and specimen."""
    from .blood_gas_specimen import SpecimenRecordRule, _restrict_by_records

    required = {"subject_id", "specimen_id", "charttime", concept_name}
    missing = required - set(frame.columns)
    if missing:
        raise KeyError(f"venous specimen restriction requires columns {sorted(missing)}")
    if frame.empty:
        return frame.copy()
    token = source.params.get("specimen_token")
    if token not in {"VEN.", "MIX."}:
        raise ValueError("venous specimen restriction requires an exact site token")
    rule = SpecimenRecordRule(
        table="labevents", item_column="itemid", item_id=52033,
        keys=("subject_id", "specimen_id"), patient_column="subject_id",
        value_column="value", arterial_tokens=frozenset({token}), keep_unrecorded=False,
    )
    result, _ = _restrict_by_records(frame, rule, data_source, frozenset({token}))
    return result
