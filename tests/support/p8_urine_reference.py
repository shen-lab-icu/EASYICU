"""Frozen pre-P8 assignment paths; synthetic equivalence oracle only."""

import logging

import numpy as np
import pandas as pd

from easyicu.scores.kdigo_aki import (
    _detect_id_col, _resolve_time_unit, _strict_numeric_component,
    _strict_time_component, _UO_MAX_GAP_HOURS,
    KDIGOComponentCalculationError, KDIGOComponentSchemaError,
)
from easyicu.scores.urine_output import collection_windows, time_hours
from easyicu.urine_weight_linkage import (
    ConflictingKeyedWeightError, resolve_keyed_unique_weights,
    resolve_unkeyed_single_entity_weight,
)

logger = logging.getLogger(__name__)

def descriptive_windows(
    urine: pd.DataFrame,
    weight: pd.DataFrame,
    windows,
    *,
    interval: pd.Timedelta | None = None,
    source_is_rate: bool = False,
) -> dict[str, pd.DataFrame]:
    """Descriptive rates use apportioned volume / actual overlap / weight."""
    ids = [c for c in urine if c.endswith("_id") or c in {
        "icustay_id", "patientunitstayid", "admissionid", "patientid", "CaseID", "patient_SN", "INP_NO",
    }]
    time_col = next((c for c in (
        "charttime", "measuredat", "measuredat_minutes", "nursingchartoffset",
        "observationoffset", "intakeoutputoffset", "registeredat", "datetime", "Offset", "offset",
    ) if c in urine), None)
    if not ids or time_col is None:
        raise ValueError("Urine windows require patient and time columns")
    data = urine[ids + [time_col, "urine"]].copy().dropna(subset=ids + [time_col])
    data["urine"] = pd.to_numeric(data.urine, errors="coerce")
    data.loc[~np.isfinite(data.urine) | (data.urine < 0), "urine"] = np.nan
    groups = data.groupby(ids + [time_col], sort=True, dropna=False).urine
    grouped = groups.agg(["sum", "mean", "count", "size"]).reset_index()
    grouped["urine"] = grouped["mean" if source_is_rate else "sum"].where(grouped["count"] == grouped["size"])
    data = grouped[ids + [time_col, "urine"]]
    if '_urine_available_at' in urine:
        receipt_cols = [c for c in ('_urine_available_at', '_urine_gap_h') if c in urine]
        receipts = urine.groupby(ids + [time_col], as_index=False)[receipt_cols].max()
        data = data.merge(receipts, on=ids + [time_col], how='left', validate='one_to_one')
    common_ids = [c for c in ids if c in weight]
    if common_ids and "weight" in weight:
        keyed = resolve_keyed_unique_weights(weight, id_columns=common_ids)
        data = data.merge(keyed, on=common_ids, how="left", validate="many_to_one")
    else:
        resolution = resolve_unkeyed_single_entity_weight(data, weight, urine_id_columns=ids)
        data["weight"] = resolution.weight if resolution.weight is not None else np.nan
    # Preserve the descriptive API's existing omission of unweighted stays
    # whenever at least one stay has a usable weight.
    weight_ok = pd.to_numeric(data.weight, errors="coerce").gt(0)
    if weight_ok.any():
        data = data.loc[weight_ok].reset_index(drop=True)
    width = pd.Timedelta(interval or pd.Timedelta(hours=1)).total_seconds() / 3600
    times = time_hours(data[time_col])
    result = {}
    for window, minimum in windows:
        name = f"uo_{window}h"
        frame = data[ids + [time_col]].copy()
        frame[name] = np.nan
        frame[f"{name}_covered_h"] = 0.
        frame[f"{name}_partial_window"] = True
        frame[f"{name}_available_at"] = np.nan
        result[name] = frame
    for positions in data.groupby(ids, sort=False).indices.values():
        amounts = data.urine.to_numpy(dtype=float)[positions]
        # HiRID source callbacks express a rate as volume per extraction bin.
        if source_is_rate:
            amounts = amounts / width
        integrated = collection_windows(times[positions], amounts, [w for w, _ in windows],
                                        source_is_rate=source_is_rate, first_hours=width,
                                        source_times=data['_urine_available_at'].to_numpy()[positions]
                                        if '_urine_available_at' in data else None,
                                        record_gaps=data['_urine_gap_h'].to_numpy()[positions]
                                        if '_urine_gap_h' in data else None)
        weights = pd.to_numeric(data.weight.iloc[positions], errors="coerce").to_numpy(dtype=float)
        for window, minimum in windows:
            volume, covered, available = integrated[window]
            valid = (covered >= minimum - 1e-9) & np.isfinite(weights) & (weights > 0)
            values = np.divide(volume, covered * weights, out=np.full(len(volume), np.nan), where=valid)
            name = f"uo_{window}h"
            result[name].loc[positions, name] = values
            result[name].loc[positions, f"{name}_covered_h"] = covered
            result[name].loc[positions, f"{name}_partial_window"] = covered < window - 1e-9
            result[name].loc[positions, f"{name}_available_at"] = available
    return result

def _calculate_uo_rates_simple(
    urine_df: pd.DataFrame,
    weight_df: pd.DataFrame,
    id_col: str,
    time_col: str,
    urine_col: str = 'urine',
    weight_col: str = 'weight',
    source_is_rate: bool = False,
    interval: pd.Timedelta | None = None,
    time_unit: str | None = None,
) -> pd.DataFrame:
    """Calculate urine output rates using simplified time-windowed averages.

    This function handles datetime, minutes (MIIV, AUMC, eICU), and
    seconds (SICdb) time columns automatically.

    Args:
        urine_df: DataFrame with urine output values (mL), or hourly-equivalent
            rate values when ``source_is_rate`` is true
        weight_df: DataFrame with patient weight (kg)
        id_col: Column name for patient ID
        time_col: Column name for time
        urine_col: Column name for urine output values
        weight_col: Column name for weight values
        source_is_rate: Use time-weighted observed-interval rate coverage
            instead of event-volume summation.
        interval: Extraction-bin width represented by each rate-source value.

    Returns:
        DataFrame with columns: id_col, time_col, uo_rt_6hr, uo_rt_12hr, uo_rt_24hr
    """
    if urine_df.empty or weight_df.empty:
        return pd.DataFrame()

    urine = urine_df.copy()
    weight = weight_df.copy()
    if urine_col not in urine.columns:
        urine_col = next(
            (column for column in ("urine", "value", "valuenum") if column in urine),
            "",
        )
    if not urine_col:
        raise KDIGOComponentSchemaError(
            component="urine_output",
            reason_code="kdigo_urine_output_value_column_missing",
            message="Urine-output source has no detectable value column",
        )
    urine[urine_col] = _strict_numeric_component(
        urine[urine_col],
        component="urine_output",
        reason_code="kdigo_urine_output_numeric_encoding_invalid",
        field_name="urine output",
    )
    negative_output = urine[urine_col].notna() & (urine[urine_col] < 0)
    if negative_output.any():
        # Ignore impossible volumes but retain their collection endpoints.
        # Removing the row would bridge a hole with a false valid interval.
        logger.warning(
            "Ignoring %d negative urine-output volume(s); retaining their "
            "collection breakpoints for KDIGO staging",
            int(negative_output.sum()),
        )
        urine.loc[negative_output, urine_col] = np.nan
    urine[time_col] = _strict_time_component(
        urine[time_col],
        component="urine_output",
        reason_code="kdigo_urine_output_time_encoding_invalid",
    )
    urine = urine.dropna(subset=[id_col, time_col])
    if urine.empty:
        return pd.DataFrame()

    # Multiple channels at one collection end share one duration.
    aggregation = {urine_col: "mean" if source_is_rate else "sum"}
    if '_urine_available_at' in urine:
        aggregation['_urine_available_at'] = 'max'
    if '_urine_gap_h' in urine:
        aggregation['_urine_gap_h'] = 'max'
    groups = urine.groupby([id_col, time_col], sort=True)
    values = groups.agg(aggregation)
    # A valid zero contributes coverage. An invalid same-time channel makes
    # the collection incomplete, matching the shared assessment owner.
    values[urine_col] = values[urine_col].where(groups[urine_col].count() == groups.size())
    urine = values.reset_index()

    if weight_col not in weight.columns:
        weight_col = next(
            (column for column in ("weight", "value", "valuenum") if column in weight),
            "",
        )
    if not weight_col:
        raise KDIGOComponentSchemaError(
            component="weight",
            reason_code="kdigo_weight_value_column_missing",
            message="Weight source has no detectable value column",
        )
    weight[weight_col] = _strict_numeric_component(
        weight[weight_col],
        component="weight",
        reason_code="kdigo_weight_numeric_encoding_invalid",
        field_name="weight",
    )
    weight_id_col = _detect_id_col(weight)

    keyed_weight = None
    if weight_id_col and weight_id_col in weight.columns:
        try:
            keyed_weight = resolve_keyed_unique_weights(
                weight,
                id_columns=[weight_id_col],
                weight_column=weight_col,
            )
        except ConflictingKeyedWeightError as exc:
            raise KDIGOComponentCalculationError(
                component="weight",
                reason_code="kdigo_weight_values_conflict",
                message=(
                    "KDIGO urine-output staging cannot choose among "
                    "different valid keyed weights without a clinical selector"
                ),
            ) from exc

    # Resolve weight only through a patient key.  The exceptional unkeyed
    # one-entity path is explicitly proved below; selecting ``iloc[0]`` from a
    # multi-patient table would silently apply one patient's weight to another.
    global_weight = np.nan
    if keyed_weight is not None:
        weight_per_patient = keyed_weight.set_index(weight_id_col)[
            weight_col
        ].to_dict()
    else:
        resolution = resolve_unkeyed_single_entity_weight(
            urine,
            weight,
            urine_id_columns=[id_col],
            weight_column=weight_col,
        )
        global_weight = resolution.weight if resolution.weight is not None else np.nan
        if resolution.diagnostic_code:
            logger.info(
                "Leaving urine-output rates unassessable: %s",
                resolution.diagnostic_code,
            )
        weight_per_patient = {}

    # Sort urine by patient and time
    urine = urine.sort_values([id_col, time_col]).reset_index(drop=True)

    # Determine time unit (datetime, hours, minutes, or seconds)
    time_unit = _resolve_time_unit(urine[time_col], time_col, time_unit)
    logger.debug(f"Detected time unit for UO calculation: {time_unit}")

    # Vectorized UO rate calculation using cumsum + searchsorted (O(N log N))
    # Convert time to minutes for uniform window computation
    urine = urine.copy()
    if time_unit == 'datetime':
        _ref = urine[time_col].min()
        urine['_min'] = (urine[time_col] - _ref) / pd.Timedelta(minutes=1)
    elif time_unit == 'timedelta':
        urine['_min'] = urine[time_col] / pd.Timedelta(minutes=1)
    elif time_unit == 'seconds':
        urine['_min'] = urine[time_col].astype(np.float64) / 60.0
    elif time_unit == 'hours':
        urine['_min'] = urine[time_col].astype(np.float64) * 60.0
    else:
        urine['_min'] = urine[time_col].astype(np.float64)

    # Map weights to each row
    if weight_per_patient:
        urine['_wt'] = urine[id_col].map(weight_per_patient)
    else:
        urine['_wt'] = global_weight
    urine['_wt'] = pd.to_numeric(urine['_wt'], errors='coerce')
    urine.loc[urine['_wt'] <= 0, '_wt'] = np.nan

    urine[urine_col] = urine[urine_col].astype(np.float64)
    urine = urine.sort_values([id_col, '_min']).reset_index(drop=True)

    from easyicu.scores.urine_output import collection_windows, KDIGO_EVIDENCE_COLUMNS

    output = [f"uo_rt_{w}hr" for w in (6, 12, 24)]
    for name in [*output, *KDIGO_EVIDENCE_COLUMNS]:
        urine[name] = np.nan
    width = pd.Timedelta(interval or pd.Timedelta(hours=1)).total_seconds() / 3600
    for positions in urine.groupby(id_col, sort=False).indices.values():
        ends = urine['_min'].to_numpy()[positions] / 60.0
        amounts = urine[urine_col].to_numpy()[positions]
        if source_is_rate:
            amounts = amounts / width
        integrated = collection_windows(
            ends, amounts, (6, 12, 24), source_is_rate=source_is_rate,
            max_gap=_UO_MAX_GAP_HOURS,
            source_times=urine['_urine_available_at'].to_numpy()[positions]
            if '_urine_available_at' in urine else None,
            record_gaps=urine['_urine_gap_h'].to_numpy()[positions]
            if '_urine_gap_h' in urine else None,
        )
        stage_available = np.full(len(ends), np.nan)
        for window in (6, 12, 24):
            volume, covered, available = integrated[window]
            wt = urine['_wt'].to_numpy()[positions]
            valid = (covered >= window - 1e-9) & np.isfinite(wt) & (wt > 0)
            name = f"uo_rt_{window}hr"
            urine.loc[positions, name] = np.divide(
                volume, wt * covered, out=np.full(len(ends), np.nan), where=valid,
            )
            urine.loc[positions, f"{name}_covered_h"] = covered
            stage_available = np.fmax(stage_available, np.where(valid, available, np.nan))
        urine.loc[positions, "aki_stage_uo_available_at"] = stage_available
    return urine[[id_col, time_col, *output, *KDIGO_EVIDENCE_COLUMNS]]

