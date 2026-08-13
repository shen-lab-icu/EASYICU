"""Deterministic outcome-by-stratum data contract for publication figures."""

from __future__ import annotations

from typing import Any, Mapping, Optional

import pandas as pd

from .base import first_exact_column
from .display_labels import display_label, label_lookup

_GROUP_SUFFIXES = (
    "_group",
    "_stratum",
    "_strata",
    "_bin",
    "_band",
    "_category",
    "_class",
    "_quartile",
    "_quintile",
    "_decile",
    "_tertile",
    "_level",
)


def normalise_strata_frame(
    frame: pd.DataFrame,
    *,
    display_labels: Optional[Mapping[str, str]] = None,
) -> pd.DataFrame:
    """Project a registered grouped-outcome table into score/rate panel data."""

    if frame.empty:
        return pd.DataFrame(columns=["score", "rate"])
    columns = {str(column).lower(): column for column in frame.columns}
    score_column = first_exact_column(
        columns,
        [
            "score",
            "stratum",
            "severity_score",
            "risk_score",
            "sofa2",
            "sofa_2",
            "gcs",
            "gcs_score",
            "kdigo",
            "kdigo_stage",
            "exposure",
            "exposure_label",
            "exposure_group",
            "group",
            "group_label",
            "status",
            "category",
            "level",
            "sepsis3",
            "sepsis_3",
            "exposure_status",
        ],
    )
    rate_column = first_exact_column(
        columns,
        [
            "death_rate",
            "mortality_rate",
            "outcome_rate",
            "outcome_risk",
            "event_rate",
            "outcome_rate_pct",
            "death_pct",
            "mortality_pct",
            "event_pct",
            "outcome_pct",
            "incidence_proportion",
            "incidence_pct",
            "risk",
            "rate",
            "death_risk",
            "mortality_risk",
        ],
    )
    if score_column is None:
        for name, column in columns.items():
            if name.endswith(_GROUP_SUFFIXES) and not name.endswith("_order"):
                score_column = column
                break
    count_column = first_exact_column(
        columns,
        ["n", "count", "n_total", "n_rows", "outcome_denominator"],
    )
    if score_column is None or rate_column is None:
        return pd.DataFrame(columns=["score", "rate"])

    working = frame.copy()
    row_role_column = columns.get("row_role")
    if row_role_column is not None:
        row_roles = working[row_role_column].astype(str).str.strip().str.casefold()
        working = working.loc[
            ~row_roles.isin({"overall", "total", "summary"})
        ].copy()

    raw_score = working[score_column]
    numeric_score = pd.to_numeric(raw_score, errors="coerce")
    semantic_category = score_column_is_semantic_category(score_column)
    score_is_numeric = bool(numeric_score.notna().all()) and not semantic_category
    score_values = (
        numeric_score
        if score_is_numeric
        else raw_score.map(
            lambda value: _score_category_label(
                score_column,
                value,
                display_labels=display_labels,
            )
        )
    )
    score_order = (
        numeric_score
        if numeric_score.notna().any()
        else pd.Series(range(len(working)), index=working.index)
    )
    result = pd.DataFrame(
        {
            "score": score_values,
            "rate": pd.to_numeric(working[rate_column], errors="coerce"),
            "_score_order": score_order,
        }
    ).dropna(subset=["score", "rate"])
    if count_column is not None:
        result["n"] = pd.to_numeric(
            working.loc[result.index, count_column], errors="coerce"
        )
    if not result.empty and result["rate"].max() > 1.0:
        result["rate"] = result["rate"] / 100.0
    result = (
        result.sort_values("score")
        .drop(columns=["_score_order"])
        .reset_index(drop=True)
        if score_is_numeric
        else result.sort_values("_score_order")
        .drop(columns=["_score_order"])
        .reset_index(drop=True)
    )
    result.attrs["score_label"] = score_axis_label(
        score_column, display_labels=display_labels
    )
    result.attrs["score_is_numeric"] = score_is_numeric
    return result


def score_column_is_semantic_category(column: Any) -> bool:
    normalized = str(column or "").strip().lower().replace("-", "_").replace(" ", "_")
    return normalized in {
        "exposure",
        "exposure_level",
        "exposure_label",
        "group",
        "group_label",
        "status",
        "category",
        "level",
        "sepsis3",
        "sepsis_3",
        "exposure_status",
    }


def _score_category_label(
    column: Any,
    value: Any,
    *,
    display_labels: Optional[Mapping[str, str]] = None,
) -> str:
    normalized_column = (
        str(column or "").strip().lower().replace("-", "_").replace(" ", "_")
    )
    state = _binary_state_label(value)
    column_label = display_label(column, display_labels)
    value_label = display_label(value, display_labels)
    if state is not None and label_lookup(column, display_labels) is not None:
        return f"{column_label} {state}"
    if normalized_column in {
        "exposure",
        "exposure_level",
        "exposure_status",
    } and state is not None:
        return "Exposed" if state == "positive" else "Unexposed"
    if normalized_column == "status" and state is not None:
        return state.capitalize()
    if normalized_column in {"group", "group_label"}:
        return f"Group {value_label}"
    return value_label


def _binary_state_label(value: Any) -> Optional[str]:
    token = str(value).strip().lower()
    if token in {"1", "1.0", "true", "yes", "y", "positive", "present"}:
        return "positive"
    if token in {"0", "0.0", "false", "no", "n", "negative", "absent"}:
        return "negative"
    return None


def strata_score_label(frame: pd.DataFrame) -> str:
    label = frame.attrs.get("score_label")
    return str(label) if label else "Score"


def score_axis_label(
    column: Any, *, display_labels: Optional[Mapping[str, str]] = None
) -> str:
    raw = str(column or "").strip()
    normalized = raw.lower().replace("-", "_").replace(" ", "_")
    declared = label_lookup(raw, display_labels)
    if declared is not None:
        if score_column_is_semantic_category(column) and not any(
            word in declared.casefold()
            for word in ("status", "category", "group", "stratum")
        ):
            return f"{declared} status"
        return declared
    mapping = {
        "score": "Score",
        "stratum": "Stratum",
        "severity_score": "Severity score",
        "risk_score": "Risk score",
        "exposure": "Exposure group",
        "exposure_level": "Exposure group",
        "exposure_label": "Exposure group",
        "group": "Group",
        "group_label": "Group",
        "status": "Status",
        "category": "Category",
        "level": "Level",
        "exposure_status": "Exposure status",
    }
    if normalized in mapping:
        return mapping[normalized]
    pretty = display_label(raw)
    if any(
        word in pretty.lower()
        for word in ("score", "stratum", "stage", "group", "status", "category")
    ):
        return pretty
    return f"{pretty} score"


__all__ = [
    "normalise_strata_frame",
    "score_axis_label",
    "score_column_is_semantic_category",
    "strata_score_label",
]
