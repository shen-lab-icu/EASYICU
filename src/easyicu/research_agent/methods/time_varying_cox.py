"""Piecewise time-varying coefficients for a Cox survival model.

The caller owns the interval cut points and model columns.  This module only
expands one-row-per-subject survival data into start/stop form, fits the sealed
extended Cox model, and returns interval-specific linear contrasts.  It does
not choose cut points, covariates, or a headline estimand.
"""

from __future__ import annotations

import math
from typing import Any, Sequence


class TimeVaryingCoxError(ValueError):
    """The sealed piecewise-Cox design could not be estimated."""


def fit_piecewise_time_varying_cox(
    frame: Any,
    *,
    duration_col: str,
    event_col: str,
    covariates: Sequence[str],
    interval_cutpoints: Sequence[float],
    exposure_col: str,
) -> Any:
    """Fit an extended Cox model with one coefficient per time interval.

    The first interval uses each covariate's base coefficient.  Later
    intervals add one covariate-by-interval interaction.  The returned table
    contains the corresponding linear contrast and Wald 95% confidence
    interval for every covariate and interval.
    """

    import numpy as np
    import pandas as pd
    from lifelines import CoxTimeVaryingFitter
    from scipy.stats import norm

    columns = [str(value) for value in covariates]
    if not columns or len(columns) != len(set(columns)):
        raise TimeVaryingCoxError("time-varying Cox covariates must be unique")
    if exposure_col not in columns:
        raise TimeVaryingCoxError("time-varying Cox exposure is not a covariate")
    required = {duration_col, event_col, *columns}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise TimeVaryingCoxError(
            "time-varying Cox input lacks columns: " + ", ".join(missing)
        )
    cutpoints = tuple(float(value) for value in interval_cutpoints)
    if (
        not cutpoints
        or any(not math.isfinite(value) or value <= 0 for value in cutpoints)
        or tuple(sorted(set(cutpoints))) != cutpoints
    ):
        raise TimeVaryingCoxError(
            "time-varying Cox cut points must be finite, unique, and increasing"
        )

    source = frame[[duration_col, event_col, *columns]].copy()
    for column in source.columns:
        source[column] = pd.to_numeric(source[column], errors="coerce")
    if source.isna().any().any() or not np.isfinite(source.to_numpy(dtype=float)).all():
        raise TimeVaryingCoxError(
            "time-varying Cox input must be finite complete cases"
        )
    duration = source[duration_col].to_numpy(dtype=float)
    event = source[event_col].to_numpy(dtype=int)
    if (duration <= 0).any() or not np.isin(event, [0, 1]).all():
        raise TimeVaryingCoxError(
            "time-varying Cox durations must be positive and events binary"
        )
    if cutpoints[-1] >= float(duration.max()):
        raise TimeVaryingCoxError(
            "time-varying Cox final cut point must precede observed follow-up"
        )

    interval_starts = (0.0, *cutpoints)
    interval_ends = (*cutpoints, float(duration.max()))
    records: list[dict[str, Any]] = []
    values = source[columns].to_numpy(dtype=float)
    for subject_id, (subject_duration, subject_event, row) in enumerate(
        zip(duration, event, values, strict=True)
    ):
        for interval_index, start in enumerate(interval_starts):
            if start >= subject_duration:
                break
            stop = min(subject_duration, interval_ends[interval_index])
            record: dict[str, Any] = {
                "__subject_id": subject_id,
                "__start": start,
                "__stop": stop,
                "__event": int(subject_event and stop == subject_duration),
            }
            for column, value in zip(columns, row, strict=True):
                record[column] = value
                for later_index in range(1, len(interval_starts)):
                    record[f"{column}__interval_{later_index + 1}"] = (
                        value if interval_index == later_index else 0.0
                    )
            records.append(record)
    long_frame = pd.DataFrame.from_records(records)
    if int(long_frame["__event"].sum()) != int(event.sum()):
        raise TimeVaryingCoxError("time-varying Cox expansion changed event count")

    fitter = CoxTimeVaryingFitter()
    fitter.fit(
        long_frame,
        id_col="__subject_id",
        start_col="__start",
        stop_col="__stop",
        event_col="__event",
        show_progress=False,
    )
    terms = [str(value) for value in fitter.params_.index]
    term_index = {term: index for index, term in enumerate(terms)}
    covariance = fitter.variance_matrix_.to_numpy(dtype=float)
    coefficients = fitter.params_.to_numpy(dtype=float)
    rows: list[dict[str, Any]] = []
    for column in columns:
        if column not in term_index:
            raise TimeVaryingCoxError(
                f"time-varying Cox fit omitted covariate {column!r}"
            )
        for interval_index, (start, stop) in enumerate(
            zip(interval_starts, interval_ends, strict=True)
        ):
            contrast = np.zeros(len(terms), dtype=float)
            contrast[term_index[column]] = 1.0
            interaction = None
            if interval_index:
                interaction = f"{column}__interval_{interval_index + 1}"
                if interaction not in term_index:
                    raise TimeVaryingCoxError(
                        f"time-varying Cox fit omitted interaction {interaction!r}"
                    )
                contrast[term_index[interaction]] = 1.0
            coefficient = float(contrast @ coefficients)
            variance = float(contrast @ covariance @ contrast)
            if not math.isfinite(variance) or variance <= 0:
                raise TimeVaryingCoxError(
                    f"time-varying Cox contrast variance is invalid for {column!r}"
                )
            standard_error = math.sqrt(variance)
            z_value = coefficient / standard_error
            rows.append(
                {
                    "term": column,
                    "is_exposure": column == exposure_col,
                    "interval_index": interval_index + 1,
                    "interval_start_days": start,
                    "interval_end_days": stop,
                    "coefficient": coefficient,
                    "standard_error": standard_error,
                    "hazard_ratio": math.exp(coefficient),
                    "ci_low": math.exp(coefficient - 1.96 * standard_error),
                    "ci_high": math.exp(coefficient + 1.96 * standard_error),
                    "p_value": 2.0 * float(norm.sf(abs(z_value))),
                    "base_term": column,
                    "interaction_term": interaction,
                    "method": "piecewise_time_varying_cox",
                }
            )
    result = pd.DataFrame(rows)
    numeric = result[
        [
            "coefficient",
            "standard_error",
            "hazard_ratio",
            "ci_low",
            "ci_high",
            "p_value",
        ]
    ].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise TimeVaryingCoxError("time-varying Cox produced non-finite estimates")
    return result


__all__ = ["TimeVaryingCoxError", "fit_piecewise_time_varying_cox"]
