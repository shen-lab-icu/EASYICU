"""Observability-preserving KDIGO staging derived from evidence receipts.

Owner
-----
The packaged renal contract (:mod:`easyicu.scores.aki_profiles`) publishes the
public-reference KDIGO stage together with per-component evidence receipts, and
deliberately withholds a pre-combined "strict" disease label.  The reference
stage follows the upstream MIT-LCP semantics, where a component with no usable
evidence contributes zero; a stage of ``0`` therefore means "no positive
evidence was found", not "kidney injury was ruled out".

This module owns the other reading of the same receipts: a positive component
establishes stage 1-3, a stage of ``0`` requires complete negative evidence
from every component, and anything else stays unknown.  It derives that from
whatever the contract emits rather than reviving a retired profile, and it
never reads a stage column to decide whether evidence existed -- the component
evidence status is the authority, the stage column only supplies the magnitude
of a positive component.

Use it when a study treats KDIGO stage as an exposure or endpoint with an
explicit unknown category.  Use the reference stage when reproducing the
published cross-database phenotype.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

import pandas as pd


STRICT_KDIGO_SCHEMA_VERSION = "easyicu.strict_kdigo_stage/1"

#: ``component -> (stage column candidates, evidence status column candidates)``.
#: Current exports use the ``*_reference`` / ``*_evidence_status`` spelling; the
#: retired strict profile used the bare / ``*_ascertainment`` spelling.  Both are
#: read here so a historical export remains analysable without re-extraction.
_COMPONENTS: tuple[tuple[str, tuple[str, ...], tuple[str, ...]], ...] = (
    (
        "creat",
        ("aki_stage_creat_reference", "aki_stage_creat"),
        ("creatinine_evidence_status", "creatinine_ascertainment"),
    ),
    (
        "uo",
        ("aki_stage_uo_reference", "aki_stage_uo"),
        ("urine_evidence_status", "urine_ascertainment"),
    ),
    (
        "rrt",
        ("aki_stage_rrt_reference", "aki_stage_rrt", "aki_stage_crrt_source_native"),
        ("rrt_evidence_status", "rrt_ascertainment"),
    ),
)
_COMPONENT_STATES = frozenset({"positive", "negative", "indeterminate"})
#: Combined readings.  ``partial_no_observed_positive`` is deliberately distinct
#: from ``indeterminate``: the first had usable negative evidence from at least
#: one component, the second had none at all.
STRICT_ASCERTAINMENT_STATES = (
    "positive",
    "positive_stage_unresolved",
    "negative_complete",
    "partial_no_observed_positive",
    "indeterminate",
)
#: Readings that establish kidney injury; they can never become stage 0.
POSITIVE_ASCERTAINMENT_STATES = ("positive", "positive_stage_unresolved")


class StrictKdigoError(ValueError):
    """The renal receipts cannot support an observability-preserving stage."""


def strict_kdigo_receipt_columns(
    columns: Sequence[str],
) -> Optional[dict[str, tuple[str, str]]]:
    """Resolve the receipt columns a strict derivation needs, or ``None``.

    This is the contract check callers use before promising an unknown-aware
    KDIGO exposure: it answers whether the artifact carries, for every KDIGO
    component, both an evidence status and a stage magnitude.
    """

    available = set(columns)
    resolved: dict[str, tuple[str, str]] = {}
    for component, stage_names, status_names in _COMPONENTS:
        stage = next((name for name in stage_names if name in available), None)
        status = next((name for name in status_names if name in available), None)
        if stage is None or status is None:
            return None
        resolved[component] = (stage, status)
    return resolved


def _component_states(frame: pd.DataFrame, column: str) -> pd.Series:
    states = frame[column].astype("string").str.strip().str.lower()
    unknown = sorted(set(states.dropna().unique()) - _COMPONENT_STATES)
    if unknown:
        raise StrictKdigoError(
            f"{column} contains unsupported evidence states: " + ", ".join(unknown)
        )
    return states


def _component_stage(frame: pd.DataFrame, column: str) -> pd.Series:
    numeric = pd.to_numeric(frame[column], errors="coerce")
    lost = frame[column].notna() & numeric.isna()
    if bool(lost.any()):
        raise StrictKdigoError(f"{column} contains non-numeric KDIGO stages")
    observed = numeric.dropna()
    if bool((observed.lt(0) | observed.gt(3)).any()):
        raise StrictKdigoError(f"{column} contains stages outside 0-3")
    return numeric


def derive_strict_kdigo_stage(frame: pd.DataFrame) -> pd.DataFrame:
    """Add ``aki_stage_strict`` / ``aki_ascertainment`` to a renal frame.

    ``aki_stage_strict`` is ``<NA>`` for every row whose components cannot rule
    injury in or out.  It is never filled from the reference stage, because that
    column cannot distinguish an observed negative from an absent measurement.
    """

    if not isinstance(frame, pd.DataFrame):
        raise TypeError("renal receipts must be a DataFrame")
    resolved = strict_kdigo_receipt_columns(frame.columns)
    if resolved is None:
        raise StrictKdigoError(
            "the renal frame carries no per-component KDIGO evidence receipts"
        )
    result = frame.copy()
    contributions: list[pd.Series] = []
    any_positive = pd.Series(False, index=frame.index)
    all_negative = pd.Series(True, index=frame.index)
    any_negative = pd.Series(False, index=frame.index)
    for stage_column, status_column in resolved.values():
        states = _component_states(frame, status_column)
        magnitudes = _component_stage(frame, stage_column)
        positive = states.eq("positive").fillna(False)
        negative = states.eq("negative").fillna(False)
        if bool((negative & magnitudes.gt(0)).any()):
            raise StrictKdigoError(
                f"{status_column}=negative contradicts a positive {stage_column}"
            )
        # A positive component whose stage column carries no magnitude is kept
        # as "positive, magnitude unresolved" rather than being given a stage
        # this module would have to invent.  The published renal bundle pairs
        # one implementation's evidence receipts with another's stage columns,
        # and their component coverage differs (for example an eICU RRT
        # positive whose reference component stage is not populated).
        contribution = pd.Series(pd.NA, index=frame.index, dtype="Int64")
        resolvable = positive & magnitudes.gt(0)
        contribution.loc[resolvable] = (
            magnitudes.loc[resolvable].round().astype("Int64")
        )
        contributions.append(contribution)
        any_positive |= positive
        any_negative |= negative
        all_negative &= negative
    # ``max`` skips NA, so a row keeps the highest resolved component and stays
    # NA when no component supplied a magnitude.
    stage = (
        pd.concat(contributions, axis=1).max(axis=1).astype("Int64")
        if contributions
        else pd.Series(pd.NA, index=frame.index, dtype="Int64")
    )
    # A positive row is never stage 0, even when its magnitude is unresolved.
    stage = stage.where(any_positive | (all_negative & ~any_positive), pd.NA)
    stage.loc[all_negative & ~any_positive] = 0
    ascertainment = pd.Series("indeterminate", index=frame.index, dtype="string")
    ascertainment.loc[any_negative & ~any_positive] = "partial_no_observed_positive"
    ascertainment.loc[all_negative & ~any_positive] = "negative_complete"
    ascertainment.loc[any_positive] = "positive"
    ascertainment.loc[any_positive & stage.isna()] = "positive_stage_unresolved"
    result["aki_stage_strict"] = stage
    result["aki_ascertainment"] = ascertainment
    return result


def summarize_strict_kdigo_window(
    frame: pd.DataFrame,
    *,
    id_column: str,
    time_column: str = "charttime",
    window_start_hours: float = 0.0,
    window_end_hours: float = 24.0,
) -> pd.DataFrame:
    """Reduce a strict renal frame to one unknown-preserving row per stay.

    A single positive row in the window establishes the stay's stage; a stay is
    stage ``0`` only when at least one row proved complete negative evidence and
    none was positive.  Every other stay stays ``<NA>``: a window that merely
    contains no positive reading is not a ruled-out stay.  The summary reports
    the same ascertainment vocabulary as the row-level reading, so "partly
    assessed, never positive" stays distinguishable from "never assessed".
    """

    if id_column not in frame.columns or time_column not in frame.columns:
        raise StrictKdigoError(
            f"the renal frame needs {id_column!r} and {time_column!r} columns"
        )
    strict = (
        frame
        if {"aki_stage_strict", "aki_ascertainment"}.issubset(frame.columns)
        else derive_strict_kdigo_stage(frame)
    )
    times = pd.to_numeric(strict[time_column], errors="coerce")
    lost = strict[time_column].notna() & times.isna()
    if bool(lost.any()):
        raise StrictKdigoError(f"{time_column} contains non-numeric values")
    window = strict.loc[
        times.between(window_start_hours, window_end_hours, inclusive="both")
    ]
    if window[id_column].isna().any():
        raise StrictKdigoError(f"{id_column} contains missing identities")
    identities = window[id_column]
    states = window["aki_ascertainment"].astype("string")
    injured = states.isin(POSITIVE_ASCERTAINMENT_STATES)
    max_positive = (
        window["aki_stage_strict"].where(injured).groupby(identities, sort=True).max()
    )
    any_positive = injured.groupby(identities, sort=True).any()
    complete_negative = (
        states.eq("negative_complete").groupby(identities, sort=True).any()
    )
    partial_negative = (
        states.eq("partial_no_observed_positive").groupby(identities, sort=True).any()
    )
    rows = identities.groupby(identities, sort=True).size()
    summary = pd.DataFrame({id_column: list(rows.index)})
    summary["aki_stage_strict"] = summary[id_column].map(max_positive).astype("Int64")
    observed_positive = summary[id_column].map(any_positive).astype("boolean")
    observed_negative = summary[id_column].map(complete_negative).astype("boolean")
    # A stay with any positive row is never stage 0, even when no row resolved a
    # magnitude; a stay with neither a positive row nor a complete negative row
    # stays unknown rather than joining the ruled-out reference group.
    summary.loc[
        ~observed_positive.fillna(False) & observed_negative.fillna(False),
        "aki_stage_strict",
    ] = 0
    observed_partial = summary[id_column].map(partial_negative).astype("boolean")
    summary["aki_ascertainment"] = pd.Series(
        "indeterminate", index=summary.index, dtype="string"
    )
    # A stay that was partly assessed and never positive is still unknown for
    # the stage, but it is not the same as a stay nothing was measured on --
    # the window summary keeps the row-level vocabulary rather than collapsing
    # two different kinds of missing into one.
    summary.loc[
        ~observed_positive.fillna(False)
        & ~observed_negative.fillna(False)
        & observed_partial.fillna(False),
        "aki_ascertainment",
    ] = "partial_no_observed_positive"
    summary.loc[
        ~observed_positive.fillna(False) & observed_negative.fillna(False),
        "aki_ascertainment",
    ] = "negative_complete"
    summary.loc[observed_positive.fillna(False), "aki_ascertainment"] = "positive"
    summary.loc[
        observed_positive.fillna(False) & summary["aki_stage_strict"].isna(),
        "aki_ascertainment",
    ] = "positive_stage_unresolved"
    summary["kidney_complete_negative_observed"] = observed_negative
    summary["kidney_window_row_count"] = (
        summary[id_column].map(rows).fillna(0).astype("int64")
    )
    return summary.reset_index(drop=True)


def strict_kdigo_summary(frame: pd.DataFrame) -> dict[str, Any]:
    """Aggregate-only counts for a public materialization receipt."""

    required = {"aki_stage_strict", "aki_ascertainment"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise StrictKdigoError(
            "strict KDIGO summary needs columns: " + ", ".join(missing)
        )
    stage = frame["aki_stage_strict"]
    states = frame["aki_ascertainment"].astype("string")
    return {
        "schema_version": STRICT_KDIGO_SCHEMA_VERSION,
        "rows": int(len(frame)),
        "unknown_rows": int(stage.isna().sum()),
        "stage_counts": {
            str(int(value)): int((stage == value).sum())
            for value in sorted(pd.unique(stage.dropna()))
        },
        "ascertainment_counts": {
            state: int((states == state).sum())
            for state in STRICT_ASCERTAINMENT_STATES
            if int((states == state).sum())
        },
    }


__all__ = [
    "POSITIVE_ASCERTAINMENT_STATES",
    "STRICT_ASCERTAINMENT_STATES",
    "STRICT_KDIGO_SCHEMA_VERSION",
    "StrictKdigoError",
    "derive_strict_kdigo_stage",
    "strict_kdigo_receipt_columns",
    "strict_kdigo_summary",
    "summarize_strict_kdigo_window",
]
