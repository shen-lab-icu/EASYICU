"""Shared, dependency-light cohort-viability + self-block detection.

Single source of truth for the "is this cohort task-viable yet the run blocked
its primary deliverable?" judgement used in two places:

* post-hoc in :mod:`evaluation_scorecard` (``detect_self_inflicted_block``),
  which reads a finished run directory, and
* at runtime in :mod:`execution.phase`, which decides whether to fire a
  *directed* full replan when a prediction/estimation model step emits a
  non-execution stub on a populated cohort.

The thresholds live here so the two callers cannot drift apart. The module is
intentionally light (no pandas at import time, no research-agent imports) to
keep it cycle-free and cheap on the hot execute path.

Impartiality note: viability is a *floor* — enough rows, a non-trivial outcome
minority (when the outcome is known), and a handful of well-populated
predictors. It never asserts which model should be fit. Blocking on genuinely
non-viable data (too few rows, no outcome variation, no usable predictors)
stays legitimate and is reported as ``viable=False``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

__all__ = [
    "CohortViability",
    "MIN_VIABLE_ROWS",
    "MIN_MINORITY_EVENTS",
    "MIN_WELL_POPULATED_PREDICTORS",
    "WELL_POPULATED_FRACTION",
    "assess_cohort_viability",
    "step_summary_block_signal",
    "step_requires_model_performance",
]

# Thresholds — the single source of truth. Conservative on purpose: below any
# of these a block is plausibly legitimate, so the detectors stay silent.
MIN_VIABLE_ROWS = 100
MIN_MINORITY_EVENTS = 10
MIN_WELL_POPULATED_PREDICTORS = 5
WELL_POPULATED_FRACTION = 0.5


@dataclass(frozen=True)
class CohortViability:
    """Verdict on whether a cohort is populated enough to model.

    ``viable`` is True only when every applicable floor is cleared. ``note`` is a
    short human-readable fact list (rows / events / predictors) suitable for an
    audit finding or a replan directive; it is empty when not viable.
    """

    viable: bool
    n_rows: int
    well_populated_predictors: int
    minority_events: Optional[int]
    note: str


def assess_cohort_viability(
    df: Any,
    *,
    outcome: Optional[str] = None,
) -> CohortViability:
    """Assess whether ``df`` (a pandas DataFrame) is task-viable.

    Floors: ``MIN_VIABLE_ROWS`` rows; when ``outcome`` names a column, at least
    two outcome classes with the minority class ``>= MIN_MINORITY_EVENTS``; and
    ``>= MIN_WELL_POPULATED_PREDICTORS`` non-outcome columns each at least
    ``WELL_POPULATED_FRACTION`` non-missing. Returns ``viable=False`` (rather than
    raising) whenever a floor is not cleared or the outcome has no variation.
    """
    import pandas as pd  # lazy: keep module import light

    n_rows = int(len(df))
    if n_rows < MIN_VIABLE_ROWS:
        return CohortViability(False, n_rows, 0, None, "")

    minority_events: Optional[int] = None
    outcome = outcome or ""
    if outcome and outcome in getattr(df, "columns", []):
        col = pd.to_numeric(df[outcome], errors="coerce").dropna()
        if col.empty:
            return CohortViability(False, n_rows, 0, None, "")
        counts = col.value_counts()
        if len(counts) < 2:
            # No outcome variation -> genuinely not modellable.
            return CohortViability(False, n_rows, 0, None, "")
        minority_events = int(counts.min())
        if minority_events < MIN_MINORITY_EVENTS:
            return CohortViability(False, n_rows, 0, minority_events, "")

    feature_cols = [c for c in df.columns if c != outcome]
    well_populated = sum(
        1 for c in feature_cols if df[c].notna().mean() >= WELL_POPULATED_FRACTION
    )
    if well_populated < MIN_WELL_POPULATED_PREDICTORS:
        return CohortViability(False, n_rows, well_populated, minority_events, "")

    bits = [f"{n_rows} rows", f"{well_populated} well-populated predictor columns"]
    if minority_events is not None:
        bits.append(f"{minority_events} minority-class outcome events")
    return CohortViability(
        True, n_rows, well_populated, minority_events, ", ".join(bits)
    )


def step_summary_block_signal(summary: Mapping[str, Any]) -> Optional[str]:
    """A short reason if a step summary *deliberately* recorded a non-execution /
    blocked-modeling status (the agent chose not to run), else ``None``.

    Distinguishes a chosen block from a hard crash: a crash is a code failure
    (already reflected elsewhere); a deliberate block on viable data is the
    self-paralysis failure mode this module exists to surface.
    """
    if not isinstance(summary, Mapping):
        return None
    status = str(summary.get("execution_status") or "").lower()
    if "non_execution" in status or "blocked" in status:
        return str(summary.get("modeling_block_reason") or status)[:160]
    if summary.get("modeling_blocked") is True:
        return str(summary.get("modeling_block_reason") or "modeling_blocked")[:160]
    return None


def step_requires_model_performance(expected_outputs: Sequence[str]) -> bool:
    """True when a step's contract requires model-performance statistics.

    Structural and case-neutral: keys on the canonical ``statistic:auroc`` /
    ``statistic:brier_score`` outputs the advanced-plan contract assigns to a
    prediction/estimation modeling step, never on the step's name or the
    benchmark case. A step that must emit these yet recorded a block is the
    signature this guard targets.
    """
    wanted = {"statistic:auroc", "statistic:brier_score"}
    return any(str(item).strip().lower() in wanted for item in (expected_outputs or []))
