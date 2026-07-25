"""Deterministic temporal-feature primitives over the long trajectory.

The research agent's wide per-stay universe (``cohort.materializer``) is a
baseline-summary lens: it cannot express *when* a value crossed a clinical
threshold, an *incident-after-exposure* endpoint, or a *landmark* design. The
long trajectory (``cohort.materializer.build_trajectory_long`` →
``TRAJECTORY_PARQUET``: ``stay_id, charttime, concept, value_num, value_str``)
carries the per-timepoint detail, but turning it into analysis inputs is the
same handful of operations every timing/causal question needs.

Before this module, each generated analysis step re-implemented those
operations in LLM-authored code — and got them wrong (e.g. counting a KDIGO
stage-0 record, "measured but event-absent", as an AKI onset). This module is
the shared, tested, deterministic home for them, so any step composes validated
building blocks instead of reinventing them:

* :func:`onset_times` — first time a concept crosses a clinical threshold
  (first MAP<65 = hypotension onset; first lactate>2; first AKI stage>=1).
* :func:`incident_outcome_cohort` — classify each stay as prevalent /
  incident / event-free of an outcome **relative to an index event**, the
  exposure-before-outcome construction that defends against reverse-order and
  prevalent-disease mixing.
* :func:`landmark_cohort` — at-risk set + follow-up clock from a landmark time,
  the standard immortal-time-bias guard for early-vs-delayed designs.

All functions are pure (DataFrame in, DataFrame out), concept/threshold-agnostic,
and never call an LLM. ``window`` (hours from ICU admission) optionally bounds
the trajectory; ``None`` uses the full series.
"""
from __future__ import annotations

import operator
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import pandas as pd

ID_COL = "stay_id"
TIME_COL = "charttime"

_OPS: Dict[str, Callable[[pd.Series, float], pd.Series]] = {
    ">=": operator.ge,
    ">": operator.gt,
    "<=": operator.le,
    "<": operator.lt,
    "==": operator.eq,
    "!=": operator.ne,
}

# A categorical/boolean concept (e.g. mech_vent stored as "invasive"/"noninvasive",
# or a True/None event flag) has an all-NaN value_num; its presence is the event.
# These value_str tokens read as the NEGATIVE level, so they are not an onset.
_NEGATIVE_TOKENS = {"", "0", "0.0", "false", "f", "no", "n", "none", "nan", "na", "null", "off"}


def _check_ops(op: str) -> Callable[[pd.Series, float], pd.Series]:
    if op not in _OPS:
        raise ValueError(
            f"unsupported comparison op {op!r}; use one of {sorted(_OPS)} or "
            "'present' (categorical/boolean presence)"
        )
    return _OPS[op]


def _concept_slice(
    trajectory: pd.DataFrame, concept: str, window: Optional[Tuple[float, float]]
) -> pd.DataFrame:
    """Rows of one concept (numeric + raw value), optionally windowed, time-sorted.

    Keeps ``value_str`` and does NOT drop value_num-NaN rows, so categorical /
    boolean concepts (all-NaN ``value_num``) survive for presence-mode onsets.
    """
    for col in (ID_COL, TIME_COL, "concept", "value_num", "value_str"):
        if col not in trajectory.columns:
            raise KeyError(
                f"trajectory is missing required column {col!r}; expected the "
                "long format (stay_id, charttime, concept, value_num, value_str)"
            )
    sub = trajectory[trajectory["concept"] == concept]
    sub = sub[[ID_COL, TIME_COL, "value_num", "value_str"]]
    if window is not None:
        lo, hi = window
        sub = sub[(sub[TIME_COL] >= lo) & (sub[TIME_COL] <= hi)]
    return sub.sort_values([ID_COL, TIME_COL])


def _positive_mask(
    sub: pd.DataFrame,
    *,
    op: str,
    threshold: float,
    positive_values: Optional[set],
) -> pd.Series:
    """Boolean mask of rows that count as the (positive) event.

    Three modes, covering both numeric and categorical concepts generally:
    ``positive_values`` (value_str membership) > ``op='present'`` (any recorded
    non-negative value_str) > numeric ``value_num`` comparison.
    """
    if positive_values is not None:
        wanted = {str(v).strip().lower() for v in positive_values}
        return sub["value_str"].astype("string").str.strip().str.lower().isin(wanted)
    if op == "present":
        norm = sub["value_str"].astype("string").str.strip().str.lower()
        recorded = sub["value_str"].notna() & norm.notna()
        # numeric-zero recorded as value_num also reads as negative
        numeric_zero = sub["value_num"].notna() & (sub["value_num"] == 0)
        return recorded & ~norm.isin(_NEGATIVE_TOKENS) & ~numeric_zero
    cmp = _check_ops(op)
    valid = sub["value_num"].notna()
    return valid & cmp(sub["value_num"], threshold)


def onset_times(
    trajectory: pd.DataFrame,
    concept: str,
    *,
    op: str = ">=",
    threshold: float = 1.0,
    positive_values: Optional[set] = None,
    window: Optional[Tuple[float, float]] = None,
    onset_col: Optional[str] = None,
) -> pd.DataFrame:
    """Per-stay first ``charttime`` where ``concept`` reaches the positive level.

    The clinical onset of a state, NOT the first recorded measurement. Handles
    both numeric and categorical concepts generally:

    * numeric: ``op`` / ``threshold`` on ``value_num`` (e.g. first MAP ``<`` 65,
      first AKI stage ``>=`` 1 — a value-0 stage record is correctly skipped);
    * boolean/categorical (all-NaN ``value_num``): pass ``op="present"`` for any
      recorded non-negative value (e.g. mech_vent="invasive"/"noninvasive"), or
      ``positive_values={...}`` to require specific value_str levels (e.g.
      ``{"invasive"}``).

    Returns ``[stay_id, <onset_col>]`` (default ``<concept>_onset_time``); a stay
    that never reaches the positive level is absent (NaN after a left-merge),
    never 0.
    """
    col = onset_col or f"{concept}_onset_time"
    sub = _concept_slice(trajectory, concept, window)
    if sub.empty:
        return pd.DataFrame(columns=[ID_COL, col])
    mask = _positive_mask(
        sub, op=op, threshold=threshold, positive_values=positive_values
    )
    crossed = sub[mask]
    if crossed.empty:
        return pd.DataFrame(columns=[ID_COL, col])
    out = crossed.groupby(ID_COL)[TIME_COL].first().reset_index()
    out.columns = [ID_COL, col]
    return out


def incident_outcome_cohort(
    trajectory: pd.DataFrame,
    *,
    outcome_concept: str,
    index_concept: str,
    outcome_op: str = ">=",
    outcome_threshold: float = 1.0,
    outcome_positive_values: Optional[set] = None,
    index_op: str = ">=",
    index_threshold: float = 1.0,
    index_positive_values: Optional[set] = None,
    window: Optional[Tuple[float, float]] = None,
) -> pd.DataFrame:
    """Classify each stay's outcome as prevalent / incident / event-free
    relative to an index event, the exposure-before-outcome construction.

    * ``index_time`` = onset of the index event (e.g. first mechanical
      ventilation, or first PEEP>=X — the "time zero" the outcome is judged
      against).
    * ``outcome_onset_time`` = onset of the outcome (first threshold crossing).
    * ``classification`` per stay with an index event:
        - ``prevalent``  — outcome onset at or before ``index_time`` (cannot be
          caused by / attributed after the index; must be excluded),
        - ``incident``   — outcome onset strictly after ``index_time``,
        - ``event_free`` — index present, outcome never crosses.
      Stays with no index event are ``no_index`` (outside the at-risk set).

    Returns one row per stay (all stays that have an index event OR an outcome
    onset) with columns: ``stay_id, index_time, outcome_onset_time,
    classification, at_risk`` (1 for incident/event_free, 0 otherwise),
    ``incident`` (1 incident, 0 event_free, NaN when not at risk),
    ``time_to_event`` (outcome_onset - index_time for incident, else NaN).

    Join the exposure of interest (e.g. PEEP level from the wide cohort) to the
    ``at_risk==1`` subset and model ``incident`` (logistic) or
    ``(incident, time_to_event)`` (survival). This is the generic
    reverse-order / prevalent-mixing guard — not specific to any concept.
    """
    idx = onset_times(
        trajectory, index_concept, op=index_op, threshold=index_threshold,
        positive_values=index_positive_values, window=window, onset_col="index_time",
    )
    out = onset_times(
        trajectory, outcome_concept, op=outcome_op, threshold=outcome_threshold,
        positive_values=outcome_positive_values, window=window,
        onset_col="outcome_onset_time",
    )
    merged = idx.merge(out, on=ID_COL, how="outer")
    has_index = merged["index_time"].notna()
    has_outcome = merged["outcome_onset_time"].notna()

    classification = np.where(
        ~has_index,
        "no_index",
        np.where(
            ~has_outcome,
            "event_free",
            np.where(
                merged["outcome_onset_time"] <= merged["index_time"],
                "prevalent",
                "incident",
            ),
        ),
    )
    merged["classification"] = classification
    merged["at_risk"] = (
        merged["classification"].isin(["incident", "event_free"]).astype(int)
    )
    merged["incident"] = np.where(
        merged["classification"] == "incident",
        1.0,
        np.where(merged["classification"] == "event_free", 0.0, np.nan),
    )
    merged["time_to_event"] = np.where(
        merged["classification"] == "incident",
        merged["outcome_onset_time"] - merged["index_time"],
        np.nan,
    )
    return merged[
        [
            ID_COL,
            "index_time",
            "outcome_onset_time",
            "classification",
            "at_risk",
            "incident",
            "time_to_event",
        ]
    ].reset_index(drop=True)


def landmark_cohort(
    trajectory: pd.DataFrame,
    *,
    outcome_concept: str,
    landmark_hours: float,
    exposure_onset: Optional[pd.DataFrame] = None,
    outcome_op: str = ">=",
    outcome_threshold: float = 1.0,
    window: Optional[Tuple[float, float]] = None,
) -> pd.DataFrame:
    """Landmark at-risk set + follow-up clock from a fixed landmark time.

    The standard immortal-time-bias guard: only stays still event-free and
    alive/observed at ``landmark_hours`` enter the at-risk set, and exposure is
    classified by what happened BEFORE the landmark. Returns per stay:
    ``stay_id, outcome_onset_time, eligible_at_landmark`` (outcome did not occur
    at or before the landmark), ``event_after_landmark`` (1/0 among eligible),
    ``time_from_landmark`` (outcome_onset - landmark for events).

    ``exposure_onset`` (optional ``[stay_id, <onset_col>]`` from
    :func:`onset_times`) adds ``exposed_by_landmark`` = onset at or before the
    landmark, the early-vs-not contrast assessed at the landmark — never using
    post-landmark exposure, which is what causes immortal-time bias.
    """
    out = onset_times(
        trajectory, outcome_concept, op=outcome_op, threshold=outcome_threshold,
        window=window, onset_col="outcome_onset_time",
    )
    # base = every stay seen in the (windowed) trajectory, so event-free stays
    # are retained rather than dropped.
    base_slice = _concept_slice(trajectory, outcome_concept, window)
    stays = pd.Series(
        pd.unique(
            pd.concat([base_slice[ID_COL], out[ID_COL]], ignore_index=True)
        ),
        name=ID_COL,
    )
    df = stays.to_frame().merge(out, on=ID_COL, how="left")
    onset = df["outcome_onset_time"]
    df["eligible_at_landmark"] = (onset.isna() | (onset > landmark_hours)).astype(int)
    df["event_after_landmark"] = np.where(
        df["eligible_at_landmark"] == 1,
        (onset > landmark_hours).fillna(False).astype(int),
        np.nan,
    )
    df["time_from_landmark"] = np.where(
        (df["eligible_at_landmark"] == 1) & onset.notna(),
        onset - landmark_hours,
        np.nan,
    )
    if exposure_onset is not None:
        exp_col = [c for c in exposure_onset.columns if c != ID_COL]
        if len(exp_col) != 1:
            raise ValueError(
                "exposure_onset must have exactly one non-stay_id column "
                "(an onset time from onset_times)"
            )
        df = df.merge(exposure_onset, on=ID_COL, how="left")
        eo = df[exp_col[0]]
        df["exposed_by_landmark"] = (eo.notna() & (eo <= landmark_hours)).astype(int)
    return df.reset_index(drop=True)
