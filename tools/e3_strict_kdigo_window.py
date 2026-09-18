"""Study-local, deterministic KDIGO source recipe for the E3 development run.

The public-reference AKI profile intentionally mirrors the published source
algorithm. That profile can emit stage 0 when one measured component is
negative and another component is unavailable. The E3 question uses the
existing strict EasyICU ascertainment contract: a positive component
establishes stage 1--3, while stage 0 requires complete negative creatinine,
urine-output, and RRT evidence. Everything else remains unknown.

This recipe records how the developer-prepared E3 source was built. It is not
a Research Agent acquisition capability or evidence of autonomous extraction.
Loading source tables, choosing a cohort, and joining outcomes remain separate
host responsibilities.
"""

from __future__ import annotations

from typing import Any

import pandas as pd


STRICT_KDIGO_WINDOW_SCHEMA_VERSION = "easyicu.e3_strict_kdigo_window/1"

_COMPONENTS = (
    ("creat", "aki_stage_creat", "creatinine_ascertainment"),
    ("uo", "aki_stage_uo", "urine_ascertainment"),
    ("rrt", "aki_stage_rrt", "rrt_ascertainment"),
)
_STRICT_REQUIRED_COLUMNS = {
    "stay_id",
    "charttime",
    "aki_stage",
    "aki_ascertainment",
    *(column for _, column, _ in _COMPONENTS),
    *(column for _, _, column in _COMPONENTS),
}
_STRICT_STATES = frozenset(
    {
        "positive",
        "negative_complete",
        "partial_no_observed_positive",
        "indeterminate",
    }
)
_COMPONENT_STATES = frozenset({"positive", "negative", "indeterminate"})


class StrictKdigoWindowError(ValueError):
    """The renal profile cannot support the strict window contract."""


def _numeric_stage(frame: pd.DataFrame, column: str) -> pd.Series:
    numeric = pd.to_numeric(frame[column], errors="coerce")
    conversion_loss = frame[column].notna() & numeric.isna()
    if bool(conversion_loss.any()) or bool(
        (numeric.dropna().lt(0) | numeric.dropna().gt(3)).any()
    ):
        raise StrictKdigoWindowError(
            f"{column} must contain only KDIGO stages 0-3 or missing"
        )
    return numeric.astype("Int64")


def _states(frame: pd.DataFrame, column: str, allowed: frozenset[str]) -> pd.Series:
    values = frame[column].astype("string").fillna("indeterminate")
    unknown = sorted(set(values.dropna().astype(str)) - allowed)
    if unknown:
        raise StrictKdigoWindowError(
            f"{column} contains unsupported evidence states: {unknown!r}"
        )
    return values


def _strict_stage_from_authority(
    frame: pd.DataFrame,
    *,
    stage_column: str,
    state_column: str,
    complete_negative_state: str,
    allowed_states: frozenset[str],
) -> pd.Series:
    stage = _numeric_stage(frame, stage_column)
    state = _states(frame, state_column, allowed_states)
    positive = state.eq("positive")
    negative = state.eq(complete_negative_state)
    if bool((positive & (stage.isna() | stage.le(0))).any()):
        raise StrictKdigoWindowError(
            f"{state_column}=positive requires a positive {stage_column}"
        )
    result = pd.Series(pd.NA, index=frame.index, dtype="Int64")
    result.loc[negative] = 0
    result.loc[positive] = stage.loc[positive]
    return result


def _window(frame: pd.DataFrame, start: float, end: float, *, label: str) -> pd.DataFrame:
    time = pd.to_numeric(frame["charttime"], errors="coerce")
    if bool((frame["charttime"].notna() & time.isna()).any()):
        raise StrictKdigoWindowError(f"{label} charttime contains non-numeric values")
    result = frame.loc[time.between(start, end, inclusive="both")].copy()
    result["charttime"] = time.loc[result.index]
    if result["stay_id"].isna().any():
        raise StrictKdigoWindowError(f"{label} window contains a missing stay_id")
    return result


def derive_strict_kdigo_window(
    strict_profile: pd.DataFrame,
    *,
    reference_profile: pd.DataFrame | None = None,
    window_start_hours: float = 0.0,
    window_end_hours: float = 24.0,
) -> pd.DataFrame:
    """Reduce strict and optional public-reference profiles to one row per stay.

    ``aki_ascertainment`` is the authority for the primary stage. Positive
    rows retain their observed stage; only ``negative_complete`` rows become
    stage 0. Partial or indeterminate rows remain missing. When supplied, the
    public-reference profile is reduced independently as a sensitivity
    exposure and never supplies missing primary values.
    """

    if not isinstance(strict_profile, pd.DataFrame):
        raise TypeError("strict KDIGO materialization requires a DataFrame")
    missing = sorted(_STRICT_REQUIRED_COLUMNS - set(strict_profile.columns))
    if missing:
        raise StrictKdigoWindowError(
            "strict profile lacks KDIGO columns: " + ", ".join(missing)
        )
    if not window_start_hours < window_end_hours:
        raise StrictKdigoWindowError("KDIGO window start must precede its end")
    window = _window(
        strict_profile,
        window_start_hours,
        window_end_hours,
        label="strict KDIGO",
    )
    if window.empty:
        return pd.DataFrame(
            {
                "stay_id": pd.Series(dtype="int64"),
                "aki_stage_strict": pd.Series(dtype="Int64"),
                "aki_ascertainment": pd.Series(dtype="string"),
                "aki_stage_reference": pd.Series(dtype="Int64"),
                "aki_stage_creat_strict": pd.Series(dtype="Int64"),
                "aki_stage_uo_strict": pd.Series(dtype="Int64"),
                "aki_stage_rrt_strict": pd.Series(dtype="Int64"),
                "kidney_complete_negative_observed": pd.Series(dtype="boolean"),
                "kidney_window_row_count": pd.Series(dtype="int64"),
            }
        )

    window["__aki_stage_strict"] = _strict_stage_from_authority(
        window,
        stage_column="aki_stage",
        state_column="aki_ascertainment",
        complete_negative_state="negative_complete",
        allowed_states=_STRICT_STATES,
    )
    for name, stage_column, state_column in _COMPONENTS:
        window[f"__{name}_strict"] = _strict_stage_from_authority(
            window,
            stage_column=stage_column,
            state_column=state_column,
            complete_negative_state="negative",
            allowed_states=_COMPONENT_STATES,
        )
    window["__complete_negative"] = window["aki_ascertainment"].astype(
        "string"
    ).eq("negative_complete")
    grouped = window.groupby("stay_id", sort=False, observed=True)
    result = grouped[
        [
            "__aki_stage_strict",
            "__creat_strict",
            "__uo_strict",
            "__rrt_strict",
        ]
    ].max().reset_index()
    result = result.rename(
        columns={
            "__aki_stage_strict": "aki_stage_strict",
            "__creat_strict": "aki_stage_creat_strict",
            "__uo_strict": "aki_stage_uo_strict",
            "__rrt_strict": "aki_stage_rrt_strict",
        }
    )
    complete_by_stay = grouped["__complete_negative"].any()
    row_count = grouped.size()
    result["kidney_complete_negative_observed"] = result["stay_id"].map(
        complete_by_stay
    ).astype("boolean")
    result["kidney_window_row_count"] = result["stay_id"].map(row_count).astype(
        "int64"
    )
    result["aki_ascertainment"] = pd.Series(
        "indeterminate", index=result.index, dtype="string"
    )
    result.loc[result["aki_stage_strict"].gt(0).fillna(False), "aki_ascertainment"] = (
        "positive"
    )
    result.loc[result["aki_stage_strict"].eq(0).fillna(False), "aki_ascertainment"] = (
        "negative_complete"
    )

    reference_by_stay = pd.Series(pd.NA, index=result["stay_id"], dtype="Int64")
    if reference_profile is not None:
        if not isinstance(reference_profile, pd.DataFrame):
            raise TypeError("public-reference KDIGO profile must be a DataFrame")
        required = {"stay_id", "charttime", "aki_stage_reference"}
        missing_reference = sorted(required - set(reference_profile.columns))
        if missing_reference:
            raise StrictKdigoWindowError(
                "reference profile lacks columns: " + ", ".join(missing_reference)
            )
        reference_window = _window(
            reference_profile,
            window_start_hours,
            window_end_hours,
            label="public-reference KDIGO",
        )
        reference_window["__reference"] = _numeric_stage(
            reference_window, "aki_stage_reference"
        )
        reference_by_stay = reference_window.groupby(
            "stay_id", sort=False, observed=True
        )["__reference"].max()
    result["aki_stage_reference"] = result["stay_id"].map(reference_by_stay).astype(
        "Int64"
    )

    ordered = [
        "stay_id",
        "aki_stage_strict",
        "aki_ascertainment",
        "aki_stage_reference",
        "aki_stage_creat_strict",
        "aki_stage_uo_strict",
        "aki_stage_rrt_strict",
        "kidney_complete_negative_observed",
        "kidney_window_row_count",
    ]
    return result.loc[:, ordered].sort_values("stay_id", kind="stable").reset_index(
        drop=True
    )


def strict_kdigo_summary(frame: pd.DataFrame) -> dict[str, Any]:
    """Return aggregate-only counts suitable for a public materialization receipt."""

    required = {"aki_stage_strict", "aki_ascertainment", "aki_stage_reference"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise StrictKdigoWindowError(
            "strict KDIGO summary lacks columns: " + ", ".join(missing)
        )

    def counts(column: str) -> dict[str, int]:
        return {
            ("missing" if pd.isna(key) else str(key)): int(value)
            for key, value in frame[column].value_counts(dropna=False).items()
        }

    return {
        "schema_version": STRICT_KDIGO_WINDOW_SCHEMA_VERSION,
        "rows": int(len(frame)),
        "strict_stage_counts": counts("aki_stage_strict"),
        "strict_ascertainment_counts": counts("aki_ascertainment"),
        "public_reference_stage_counts": counts("aki_stage_reference"),
        "strict_missing_not_recoded_to_zero": True,
    }


__all__ = [
    "STRICT_KDIGO_WINDOW_SCHEMA_VERSION",
    "StrictKdigoWindowError",
    "derive_strict_kdigo_window",
    "strict_kdigo_summary",
]
