"""Display authority for robustness and sensitivity-analysis evidence.

Robustness summary tables are audit products: they record how many locked
specifications were registered, fitted, and independently estimated.  Their
``range_low``/``range_high`` columns are envelopes across source rows and must
not be presented as confidence intervals or as mutually comparable effects.

Only a richer row-level table with an explicit shared estimand, contrast, unit,
effect scale, convergence status, and independence status may authorize a
quantitative sensitivity forest plot.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Callable

import numpy as np
import pandas as pd


ROBUSTNESS_EFFECT_COMPARABILITY_UNRESOLVED = (
    "ROBUSTNESS_EFFECT_COMPARABILITY_UNRESOLVED"
)
ROBUSTNESS_EFFECT_COMPARABLE = "ROBUSTNESS_EFFECT_COMPARABLE"


@dataclass(frozen=True)
class RobustnessEffectComparability:
    """Typed decision about whether one quantitative effect axis is valid."""

    authorized: bool
    reason_code: str
    message: str
    missing_columns: tuple[str, ...] = ()


def _token(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().casefold())


def _boolean_series(frame: pd.DataFrame, column: str) -> pd.Series:
    values = frame[column]
    if pd.api.types.is_bool_dtype(values):
        return values.astype(bool)
    normalized = values.map(_token)
    if not normalized.isin({"true", "false"}).all():
        raise ValueError(f"{column!r} must contain only booleans")
    return normalized.eq("true")


def assess_robustness_effect_comparability(
    frame: pd.DataFrame,
) -> RobustnessEffectComparability:
    """Authorize a common effect axis only for an explicit shared estimand.

    The three identity fields deliberately remain strict.  Inferring a common
    estimand from a shared ``effect_scale`` such as OR is unsafe: a per-unit OR,
    a high-vs-reference OR, and a duplicated complete-case documentation row
    can all carry the same scale while answering different questions.
    """

    required = {
        "point_estimate",
        "ci_low",
        "ci_high",
        "effect_scale",
        "estimand_id",
        "contrast_id",
        "effect_unit",
        "converged",
        "independent_variant",
    }
    missing = tuple(sorted(required - set(frame.columns)))
    if missing:
        return RobustnessEffectComparability(
            authorized=False,
            reason_code=ROBUSTNESS_EFFECT_COMPARABILITY_UNRESOLVED,
            message=(
                "A common robustness effect axis is not authorized because the "
                "source does not explicitly identify a shared estimand, contrast, "
                "unit, effect scale, convergence, and independence status."
            ),
            missing_columns=missing,
        )
    if frame.empty:
        return RobustnessEffectComparability(
            authorized=False,
            reason_code=ROBUSTNESS_EFFECT_COMPARABILITY_UNRESOLVED,
            message="A common robustness effect axis is not authorized for an empty table.",
        )

    for column in ("point_estimate", "ci_low", "ci_high"):
        values = pd.to_numeric(frame[column], errors="coerce")
        if values.isna().any() or not np.isfinite(values.to_numpy(dtype=float)).all():
            return RobustnessEffectComparability(
                authorized=False,
                reason_code=ROBUSTNESS_EFFECT_COMPARABILITY_UNRESOLVED,
                message=f"A common robustness effect axis is not authorized: {column} is not finite.",
            )
    estimate = pd.to_numeric(frame["point_estimate"], errors="coerce")
    low = pd.to_numeric(frame["ci_low"], errors="coerce")
    high = pd.to_numeric(frame["ci_high"], errors="coerce")
    if (low > estimate).any() or (estimate > high).any():
        return RobustnessEffectComparability(
            authorized=False,
            reason_code=ROBUSTNESS_EFFECT_COMPARABILITY_UNRESOLVED,
            message="A common robustness effect axis is not authorized because an interval does not contain its estimate.",
        )

    if not _boolean_series(frame, "converged").all():
        return RobustnessEffectComparability(
            authorized=False,
            reason_code=ROBUSTNESS_EFFECT_COMPARABILITY_UNRESOLVED,
            message="A common robustness effect axis is not authorized because at least one specification did not converge.",
        )
    if not _boolean_series(frame, "independent_variant").all():
        return RobustnessEffectComparability(
            authorized=False,
            reason_code=ROBUSTNESS_EFFECT_COMPARABILITY_UNRESOLVED,
            message="A common robustness effect axis is not authorized because at least one row is not an independent estimate.",
        )

    for column in ("effect_scale", "estimand_id", "contrast_id", "effect_unit"):
        identities = {_token(value) for value in frame[column]}
        if "" in identities or len(identities) != 1:
            return RobustnessEffectComparability(
                authorized=False,
                reason_code=ROBUSTNESS_EFFECT_COMPARABILITY_UNRESOLVED,
                message=(
                    "A common robustness effect axis is not authorized because "
                    f"{column} is missing or differs across specifications."
                ),
            )

    return RobustnessEffectComparability(
        authorized=True,
        reason_code=ROBUSTNESS_EFFECT_COMPARABLE,
        message="All rows explicitly share one estimand, contrast, unit, and effect scale.",
    )


def prepare_robustness_coverage(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate an audit summary and return count/fraction display columns."""

    required = {"axis", "total_specs", "converged_specs"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"robustness coverage is missing columns: {missing!r}")
    if frame.empty:
        raise ValueError("robustness coverage cannot be empty")
    result = frame.copy()
    result["axis"] = result["axis"].map(lambda value: str(value or "").strip())
    if result["axis"].eq("").any() or result["axis"].duplicated().any():
        raise ValueError("robustness coverage axes must be non-empty and unique")

    count_columns = ["total_specs", "converged_specs"]
    if "non_independent_specs" in result.columns:
        count_columns.append("non_independent_specs")
    for column in count_columns:
        numeric = pd.to_numeric(result[column], errors="coerce")
        if numeric.isna().any() or not np.isfinite(numeric.to_numpy(dtype=float)).all():
            raise ValueError(f"{column!r} must contain finite counts")
        if not np.isclose(numeric, np.rint(numeric), rtol=0.0, atol=1e-9).all():
            raise ValueError(f"{column!r} must contain integer-like counts")
        result[column] = numeric.astype("int64")

    if (result["total_specs"] <= 0).any():
        raise ValueError("robustness total_specs must be positive")
    if (result["converged_specs"] < 0).any() or (
        result["converged_specs"] > result["total_specs"]
    ).any():
        raise ValueError("converged robustness counts do not nest within total_specs")
    if "non_independent_specs" in result.columns and (
        (result["non_independent_specs"] < 0).any()
        or (result["non_independent_specs"] > result["total_specs"]).any()
    ):
        raise ValueError(
            "non-independent robustness counts do not nest within total_specs"
        )

    result["registered_specs"] = result["total_specs"]
    if "non_independent_specs" in result.columns:
        result["independent_specs"] = (
            result["total_specs"] - result["non_independent_specs"]
        )
    return result


def robustness_matrix_to_coverage(frame: pd.DataFrame) -> pd.DataFrame:
    """Project row-level specifications to status-only display evidence."""

    if frame.empty:
        raise ValueError("robustness matrix cannot be empty")
    label_column = "spec_id" if "spec_id" in frame.columns else "axis"
    required = {label_column, "converged"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"robustness matrix is missing columns: {missing!r}")
    labels = frame[label_column].map(lambda value: str(value or "").strip())
    if labels.eq("").any() or labels.duplicated().any():
        raise ValueError("robustness specification labels must be non-empty and unique")
    converged = _boolean_series(frame, "converged")
    data: dict[str, Any] = {
        "axis": labels,
        "total_specs": np.ones(len(frame), dtype="int64"),
        "converged_specs": converged.astype("int64"),
    }
    if "independent_variant" in frame.columns:
        independent = _boolean_series(frame, "independent_variant")
        data["non_independent_specs"] = (~independent).astype("int64")
    return pd.DataFrame(data)


def draw_robustness_coverage(
    ax: Any,
    frame: pd.DataFrame,
    *,
    color: str,
    title: str = "Sensitivity-analysis coverage",
    label_formatter: Callable[[Any], str] | None = None,
) -> dict[str, Any]:
    """Draw a compact audit strip without implying effect comparability.

    A fully saturated heatmap made a tiny ``1/1`` audit look like a scientific
    result panel.  The journal-facing display now uses quiet status markers and
    keeps the exact numerator/denominator text.  No rows, counts, or authority
    semantics are changed.
    """

    result = prepare_robustness_coverage(frame)
    columns = [
        ("registered_specs", "Registered"),
        ("converged_specs", "Converged"),
    ]
    if "independent_specs" in result.columns:
        columns.append(("independent_specs", "Independent"))
    counts = result[[name for name, _ in columns]].to_numpy(dtype=float)
    totals = result["total_specs"].to_numpy(dtype=float)[:, None]
    fractions = counts / totals
    for row_index in range(len(result)):
        for column_index in range(len(columns)):
            fraction = float(fractions[row_index, column_index])
            complete = bool(np.isclose(fraction, 1.0))
            marker_color = color if complete else "#D9DEE3"
            ax.scatter(
                column_index,
                row_index,
                s=190,
                facecolor=marker_color,
                edgecolor="white",
                linewidth=1.2,
                alpha=0.16 if complete else 0.55,
                zorder=1,
            )
            ax.text(
                column_index,
                row_index,
                f"{int(counts[row_index, column_index])}/{int(totals[row_index, 0])}",
                ha="center",
                va="center",
                fontsize=5.8,
                fontweight="semibold" if complete else "normal",
                color=color if complete else "#555B61",
                zorder=2,
            )
    formatter = label_formatter or (lambda value: str(value).replace("_", " "))
    ax.set_xticks(
        np.arange(len(columns)), [label for _, label in columns], fontsize=5.8
    )
    ax.set_yticks(
        np.arange(len(result)),
        [formatter(value) for value in result["axis"]],
        fontsize=5.8,
    )
    ax.tick_params(length=0)
    ax.set_xlim(-0.55, len(columns) - 0.45)
    ax.set_ylim(len(result) - 0.45, -0.55)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for boundary in np.arange(0.5, len(result) - 0.5, 1.0):
        ax.axhline(boundary, color="#ECEFF1", linewidth=0.6, zorder=0)
    ax.set_xlabel("Registered specifications meeting each condition")
    ax.set_title(title, loc="left", pad=7)
    return {
        "chart_type": "sensitivity_coverage_matrix",
        "effect_comparison_authorized": False,
        "reason_code": ROBUSTNESS_EFFECT_COMPARABILITY_UNRESOLVED,
        "display_authority": "audit_only",
    }


__all__ = [
    "ROBUSTNESS_EFFECT_COMPARABILITY_UNRESOLVED",
    "ROBUSTNESS_EFFECT_COMPARABLE",
    "RobustnessEffectComparability",
    "assess_robustness_effect_comparability",
    "draw_robustness_coverage",
    "prepare_robustness_coverage",
    "robustness_matrix_to_coverage",
]
