"""Interpretation-free facts derived from one sealed cohort snapshot.

This leaf owns facts that can be recomputed from the exact staged table bytes.
It deliberately does not assign analysis roles, exposures, outcomes, methods,
covariates, or estimands.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import pandas as pd


def observed_domain_for_series(series: pd.Series) -> Optional[Dict[str, Any]]:
    """Return the canonical domain actually observed in one physical column."""

    nonnull = series.dropna()
    if len(nonnull) == 0:
        return None
    n_unique = int(nonnull.nunique())
    domain: Dict[str, Any] = {
        "n_unique": n_unique,
        "is_constant": n_unique <= 1,
        # A binary fact is numeric {0, 1}; two-level categorical labels remain
        # categorical and are surfaced below without reinterpretation.
        "is_binary": False,
    }
    if pd.api.types.is_numeric_dtype(nonnull):
        try:
            domain["min"] = float(nonnull.min())
            domain["max"] = float(nonnull.max())
        except (TypeError, ValueError):
            pass
        if n_unique <= 2:
            try:
                values = {
                    int(value)
                    for value in nonnull.unique()
                    if float(value).is_integer()
                }
                domain["is_binary"] = values.issubset({0, 1}) and bool(values)
            except (TypeError, ValueError):
                domain["is_binary"] = False
    elif n_unique <= 8:
        try:
            domain["levels"] = sorted(str(value) for value in nonnull.unique())
        except (TypeError, ValueError):
            pass
    return domain


__all__ = ["observed_domain_for_series"]
