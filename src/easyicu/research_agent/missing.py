"""[Layer 3: Safe Analytical Runtime] Deterministic missing-data strategies."""

from __future__ import annotations

from typing import Literal


MissingStrategy = Literal["complete_case", "mean_imputation", "median_imputation"]


def apply_missing_strategy(df, strategy: MissingStrategy | str):
    """Apply a narrow, deterministic missing-data strategy to a dataframe."""

    strategy = str(strategy or "complete_case")
    if strategy == "complete_case":
        return df.dropna().copy()
    if strategy not in {"mean_imputation", "median_imputation"}:
        raise ValueError(f"unsupported missing strategy: {strategy}")
    out = df.copy()
    numeric_columns = list(out.select_dtypes(include="number").columns)
    for column in numeric_columns:
        if strategy == "mean_imputation":
            fill_value = out[column].mean()
        else:
            fill_value = out[column].median()
        out[column] = out[column].fillna(fill_value)
    return out.dropna().copy()


__all__ = ["MissingStrategy", "apply_missing_strategy"]
