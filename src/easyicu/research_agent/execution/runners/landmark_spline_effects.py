"""Effect products of an already fitted landmark model, without another fit."""

from __future__ import annotations

import hashlib
import json
import math

import numpy as np
import pandas as pd
import patsy


def model_rows_digest(frame: pd.DataFrame) -> str:
    """Seal row order, model values and declared grouping source without IDs."""

    digest = hashlib.sha256()
    digest.update(str(tuple(frame.columns)).encode())
    digest.update(pd.util.hash_pandas_object(frame, index=True).to_numpy().tobytes())
    return digest.hexdigest()


def fitted_exposure_curve(*, fit, spline, knots, grid, exposure_column):
    lower, reference, upper = knots
    reference_basis = np.asarray(patsy.build_design_matrices(
        [spline.design_info], {"x": [reference], "middle": reference, "lower": lower, "upper": upper},
    )[0])[0]
    bases = np.asarray(patsy.build_design_matrices(
        [spline.design_info], {"x": grid, "middle": reference, "lower": lower, "upper": upper},
    )[0])
    names = list(spline.columns)
    beta = fit.params.loc[names].to_numpy(dtype=float)
    covariance = fit.cov_params().loc[names, names].to_numpy(dtype=float)
    rows = []
    for value, basis in zip(grid, bases, strict=True):
        delta = basis - reference_basis
        estimate = float(delta @ beta)
        variance = float(delta @ covariance @ delta)
        if not np.isfinite([estimate, variance]).all() or variance < -1e-10:
            raise ValueError("functional-form contrast variance is invalid")
        se = math.sqrt(max(variance, 0))
        interval = np.exp([estimate, estimate - 1.96 * se, estimate + 1.96 * se])
        if not np.isfinite(interval).all() or (interval <= 0).any():
            raise ValueError("functional-form exposure effect is non-finite")
        rows.append({
            "exposure": exposure_column, "exposure_value": float(value),
            "reference_exposure_value": float(reference),
            "adjusted_odds_ratio": float(interval[0]),
            "ci_low": float(interval[1]), "ci_high": float(interval[2]),
            "contrast_vector": json.dumps([
                float(delta[names.index(name)]) if name in names else 0.0
                for name in fit.params.index
            ], allow_nan=False),
        })
    return rows
