"""Construct a source-preserving counting-process panel for early exposures.

This owner transforms one verified hospital-mortality follow-up table and a
long, timestamped trajectory into local analysis input.  It deliberately does
not fit a model: missing-measurement handling, covariate encoding, and
clustered inference are separate scientific/runtime contracts.  The panel
therefore makes the pre-first-measurement state explicit instead of silently
dropping it or treating it as a clinical value.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import pandas as pd


_FOLLOWUP_COLUMNS = (
    "stay_id",
    "hospital_death",
    "death_time_hours",
    "hospital_followup_time_hours",
)
_TRAJECTORY_COLUMNS = ("stay_id", "charttime", "concept", "value_num", "evidence_state")
_PANEL_COLUMNS = (
    "stay_id",
    "interval_start_hours",
    "interval_stop_hours",
    "hospital_death",
    "source_stop_hours",
    "source_event_time_hours",
    "zero_time_event_epsilon_applied",
    "exposure_state",
    "exposure_running_max",
    "exposure_measurements_seen",
    "exposure_last_measurement_time_hours",
)
_EXCLUSION_COLUMNS = ("stay_id", "reason_code")
_ZERO_TIME_EVENT_EPSILON_HOURS = 1e-6
_EARLY_EXPOSURE_WINDOW_HOURS = 24.0


class TimeVaryingExposureError(ValueError):
    """The source inputs cannot support a source-faithful exposure panel."""


@dataclass(frozen=True)
class TimeVaryingExposurePanel:
    """Counting-process intervals plus path-free construction receipts."""

    panel: pd.DataFrame
    exclusions: pd.DataFrame
    receipt: dict[str, Any]


def _require_unique_identifiers(frame: pd.DataFrame, *, label: str) -> None:
    if frame["stay_id"].isna().any() or frame["stay_id"].duplicated().any():
        raise TimeVaryingExposureError(f"{label} must contain one non-null row per stay")


def _validated_followup(followup: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(followup, pd.DataFrame):
        raise TimeVaryingExposureError("hospital follow-up must be a dataframe")
    missing = sorted(set(_FOLLOWUP_COLUMNS) - set(followup.columns))
    if missing:
        raise TimeVaryingExposureError(
            "hospital follow-up lacks required columns: " + ", ".join(missing)
        )
    result = followup.loc[:, _FOLLOWUP_COLUMNS].copy()
    _require_unique_identifiers(result, label="hospital follow-up")
    death = pd.to_numeric(result["hospital_death"], errors="coerce")
    if death.isna().any() or not death.isin([0, 1]).all():
        raise TimeVaryingExposureError("hospital follow-up death indicator must be binary")
    result["hospital_death"] = death.astype("int8")
    duration = pd.to_numeric(result["hospital_followup_time_hours"], errors="coerce")
    if duration.isna().any() or not np.isfinite(duration.to_numpy(dtype=float)).all():
        raise TimeVaryingExposureError("hospital follow-up duration must be finite")
    if (duration < 0).any():
        raise TimeVaryingExposureError("hospital follow-up duration cannot be negative")
    result["hospital_followup_time_hours"] = duration.astype(float)
    event_time = pd.to_numeric(result["death_time_hours"], errors="coerce")
    events = result["hospital_death"].eq(1)
    if event_time.loc[events].isna().any() or not np.isfinite(
        event_time.loc[events].to_numpy(dtype=float)
    ).all():
        raise TimeVaryingExposureError("recorded hospital deaths require finite event times")
    if (event_time.loc[events] < 0).any():
        raise TimeVaryingExposureError("hospital death times cannot be negative")
    if event_time.loc[~events].notna().any():
        raise TimeVaryingExposureError(
            "hospital survivors cannot carry a death-event time"
        )
    if bool((event_time.loc[events] > duration.loc[events]).any()):
        raise TimeVaryingExposureError(
            "hospital death time cannot exceed recorded hospital follow-up"
        )
    result["death_time_hours"] = event_time.astype(float)
    return result


def _validated_direct_measurements(
    trajectory: pd.DataFrame, *, exposure_concept: str
) -> tuple[pd.DataFrame, dict[str, int]]:
    if not isinstance(trajectory, pd.DataFrame):
        raise TimeVaryingExposureError("trajectory must be a dataframe")
    missing = sorted(set(_TRAJECTORY_COLUMNS) - set(trajectory.columns))
    if missing:
        raise TimeVaryingExposureError(
            "trajectory lacks required columns: " + ", ".join(missing)
        )
    concept = str(exposure_concept or "").strip()
    if not concept:
        raise TimeVaryingExposureError("time-varying exposure concept is required")
    source = trajectory.loc[
        trajectory["concept"].astype("string").eq(concept),
        _TRAJECTORY_COLUMNS,
    ].copy()
    direct = source.loc[source["evidence_state"].eq("direct_observed")].copy()
    times = pd.to_numeric(direct["charttime"], errors="coerce")
    values = pd.to_numeric(direct["value_num"], errors="coerce")
    finite = times.notna() & values.notna()
    if finite.any():
        finite &= np.isfinite(times.to_numpy(dtype=float)) & np.isfinite(
            values.to_numpy(dtype=float)
        )
    in_window = finite & times.ge(0.0) & times.le(_EARLY_EXPOSURE_WINDOW_HOURS)
    retained = direct.loc[in_window, ["stay_id"]].copy()
    retained["charttime"] = times.loc[in_window].astype(float)
    retained["value_num"] = values.loc[in_window].astype(float)
    if retained["stay_id"].isna().any():
        raise TimeVaryingExposureError("trajectory exposure measurements require stay ids")
    direct_early_rows = int(len(retained))
    retained = (
        retained.groupby(["stay_id", "charttime"], as_index=False, sort=True)
        .agg(value_num=("value_num", "max"), measurement_rows=("value_num", "size"))
        .sort_values(["stay_id", "charttime"])
        .reset_index(drop=True)
    )
    return retained, {
        "exposure_trajectory_rows": int(len(source)),
        "direct_observed_rows": int(len(direct)),
        "nonfinite_value_or_time_rows_excluded": int((~finite).sum()),
        "outside_early_window_rows_excluded": int((finite & ~in_window).sum()),
        "direct_early_measurement_rows": direct_early_rows,
    }


def _panel_for_stay(
    followup: Mapping[str, Any], measurements: pd.DataFrame | None
) -> tuple[list[dict[str, Any]], str | None]:
    event = int(followup["hospital_death"])
    source_event_time = (
        float(followup["death_time_hours"])
        if event
        else math.nan
    )
    source_stop = (
        source_event_time
        if event
        else float(followup["hospital_followup_time_hours"])
    )
    if source_stop == 0.0 and not event:
        return [], "zero_hospital_followup_without_event"
    analysis_stop = (
        _ZERO_TIME_EVENT_EPSILON_HOURS if event and source_stop == 0.0 else source_stop
    )
    # The source event time, rather than epsilon-expanded computation time,
    # determines predictability: a measurement at an event time never updates
    # the covariate for that event.
    valid = (
        measurements.loc[measurements["charttime"] < source_stop]
        if measurements is not None
        else None
    )
    measurement_times = (
        valid["charttime"].to_numpy(dtype=float) if valid is not None else np.array([])
    )
    measurement_values = (
        valid["value_num"].to_numpy(dtype=float) if valid is not None else np.array([])
    )
    measurement_counts = (
        valid["measurement_rows"].to_numpy(dtype=int)
        if valid is not None
        else np.array([], dtype=int)
    )
    boundaries = [0.0, *measurement_times.tolist(), float(analysis_stop)]
    boundaries = sorted(set(float(value) for value in boundaries))
    running_max = math.nan
    last_measurement = math.nan
    measurements_seen = 0
    update_index = 0
    rows: list[dict[str, Any]] = []
    for start, stop in zip(boundaries, boundaries[1:]):
        if stop <= start:
            continue
        while (
            update_index < len(measurement_times)
            and float(measurement_times[update_index]) <= start
        ):
            value = float(measurement_values[update_index])
            running_max = value if math.isnan(running_max) else max(running_max, value)
            last_measurement = float(measurement_times[update_index])
            measurements_seen += int(measurement_counts[update_index])
            update_index += 1
        is_final = stop == analysis_stop
        rows.append(
            {
                "stay_id": followup["stay_id"],
                "interval_start_hours": float(start),
                "interval_stop_hours": float(stop),
                "hospital_death": int(event and is_final),
                "source_stop_hours": float(source_stop),
                "source_event_time_hours": source_event_time,
                "zero_time_event_epsilon_applied": bool(event and source_stop == 0.0),
                "exposure_state": (
                    "observed_running_max"
                    if measurements_seen
                    else "unmeasured"
                ),
                "exposure_running_max": running_max,
                "exposure_measurements_seen": int(measurements_seen),
                "exposure_last_measurement_time_hours": last_measurement,
            }
        )
    if event and sum(int(row["hospital_death"]) for row in rows) != 1:
        raise TimeVaryingExposureError("time-varying panel lost a recorded death event")
    return rows, None


def build_early_running_max_exposure_panel(
    trajectory: pd.DataFrame,
    hospital_followup: pd.DataFrame,
    *,
    exposure_concept: str,
) -> TimeVaryingExposurePanel:
    """Build a 0--24 h direct-measurement running-maximum exposure panel.

    Each interval carries the running maximum of direct measurements observed
    at or before its start.  An event at time ``t`` uses only measurements
    strictly before ``t``.  The pre-first-measurement interval remains
    ``unmeasured``; this function never chooses an imputation or exclusion
    policy for it.  The only computational adjustment is documented epsilon
    expansion of an exact zero-time death so that it remains in a positive
    start/stop interval for a future Cox runtime.
    """

    followup = _validated_followup(hospital_followup)
    measurements, measurement_counts = _validated_direct_measurements(
        trajectory, exposure_concept=exposure_concept
    )
    measurements_by_stay = {
        stay_id: group.drop(columns=["stay_id"]).reset_index(drop=True)
        for stay_id, group in measurements.groupby("stay_id", sort=False)
    }
    rows: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    for followup_values in followup.itertuples(index=False, name=None):
        followup_row = dict(zip(_FOLLOWUP_COLUMNS, followup_values, strict=True))
        stay_rows, exclusion = _panel_for_stay(
            followup_row,
            measurements_by_stay.get(followup_row["stay_id"]),
        )
        if exclusion is not None:
            exclusions.append({"stay_id": followup_row["stay_id"], "reason_code": exclusion})
        else:
            rows.extend(stay_rows)
    panel = pd.DataFrame(rows, columns=_PANEL_COLUMNS)
    exclusion_frame = pd.DataFrame(exclusions, columns=_EXCLUSION_COLUMNS)
    if not panel.empty:
        panel = panel.sort_values(
            ["stay_id", "interval_start_hours", "interval_stop_hours"]
        ).reset_index(drop=True)
    event_input = int(followup["hospital_death"].sum())
    event_output = int(panel["hospital_death"].sum()) if not panel.empty else 0
    if event_input != event_output:
        raise TimeVaryingExposureError("time-varying panel changed hospital death count")
    panel_stays = set(panel["stay_id"]) if not panel.empty else set()
    unmeasured_stays = (
        int(
            panel.groupby("stay_id", sort=False)["exposure_state"]
            .apply(lambda states: bool(states.eq("unmeasured").all()))
            .sum()
        )
        if not panel.empty
        else 0
    )
    receipt = {
        "schema_version": "easyicu.time_varying_early_exposure_panel/1",
        "exposure_concept": str(exposure_concept),
        "exposure_definition": "running_max_of_direct_measurements",
        "early_exposure_window": {
            "start_hours": 0.0,
            "end_hours": _EARLY_EXPOSURE_WINDOW_HOURS,
            "inclusive": True,
        },
        "event_predictability": "measurements_at_or_after_source_event_time_excluded",
        "pre_measurement_state": "unmeasured",
        "post_window_state": "last_observed_running_max_persists_to_followup",
        "zero_time_event_convention": {
            "source_event_time_preserved": True,
            "computational_stop_epsilon_hours": _ZERO_TIME_EVENT_EPSILON_HOURS,
        },
        "counts": {
            "followup_stays": int(len(followup)),
            "panel_stays": int(len(panel_stays)),
            "panel_rows": int(len(panel)),
            "input_hospital_deaths": event_input,
            "panel_hospital_deaths": event_output,
            "zero_time_hospital_deaths": int(
                ((followup["hospital_death"] == 1) & (followup["death_time_hours"] == 0)).sum()
            ),
            "fully_unmeasured_stays": unmeasured_stays,
            "excluded_stays": int(len(exclusion_frame)),
            **measurement_counts,
        },
        "exclusion_reason_counts": {
            str(key): int(value)
            for key, value in exclusion_frame["reason_code"].value_counts().items()
        },
        "privacy": {
            "patient_rows_returned": False,
            "identifier_values_returned": False,
            "source_paths_returned": False,
        },
    }
    return TimeVaryingExposurePanel(
        panel=panel,
        exclusions=exclusion_frame,
        receipt=receipt,
    )


__all__ = [
    "TimeVaryingExposureError",
    "TimeVaryingExposurePanel",
    "build_early_running_max_exposure_panel",
]
