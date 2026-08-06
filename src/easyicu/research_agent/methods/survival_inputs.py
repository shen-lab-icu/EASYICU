"""Host-owned reconciliation of a declared event flag against its event time.

The caller owns every scientific choice: which columns carry the event and the
time, what censors follow-up, and what the administrative horizon is. This
module makes one mechanical question explicit and fails closed on it: for each
row, do the event flag and the event time agree about whether the event is
placeable in time?

MEASURED on the never-passing canonical tasks, three separate runs, three
separate silent recodings of the same unreconciled pair:

* "For death=1 rows with missing death_time, the outcome construction silently
  converts death_time > 24h to 0, treating an unavailable post-landmark time as
  a negative outcome." -- a death moved into the survivor arm by a comparison
  against NaN.
* "Outcome and censoring times are not reconciled before risk-set construction.
  Deaths with death=1 but missing death_time, or survivors with missing
  los_hosp, receive neither event nor censoring and are silently removed through
  duration/event missingness."
* "Deaths with missing or unusable death_time are silently treated as non-events
  and censored; missing los_hosp is silently replaced by administrative end
  follow-up."

And its mirror, which the same absence of a rule produced in the opposite
direction:

* "complete_case requires death_time for death-negative rows, although
  death_time is structurally not applicable when death=0; this silently excludes
  non-events from the survival denominator."

Those four are not four judgement calls. Under any protocol, an event whose time
is unknown cannot be placed on the follow-up axis, and recoding it to "no event"
moves a death into the survivor arm without changing any number's appearance.
Equally, an absent event time on a row whose flag says no event is the expected
shape, not a missing value to exclude on. ``EndpointSpec.absence_semantics``
declares what an absent ROW means; it has never had anything to say about a
present row whose event time is missing, which is why every run answered it
again.

The receipt carries counts only -- no values, no row mask, no filtered frame --
so it cannot change the caller's analysis population. Rows that cannot be
reconciled raise, with the counts in the audit, rather than being dropped here:
choosing between excluding them and censoring them at last contact is the
caller's protocol decision, and it has to be a visible one.

NOT the same boundary as ``source_status.reconcile_binary_event_presence``,
whose name is close enough to invite exactly that confusion in both directions.
That one reconciles a sparse-event TRIAD -- count, measured flag, positive-only
representative -- and answers "was this concept observed for this row at all".
This one reconciles an event CODE against its event TIME and answers "can the
event this row records be placed on the follow-up axis". Different columns,
different question; neither substitutes for the other, and a step doing survival
work on a sparse concept needs both.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd

from .descriptive_inputs import DescriptiveInputError, _semantic_numeric

__all__ = [
    "SurvivalInputError",
    "event_time_reconciliation_receipt",
]


class SurvivalInputError(DescriptiveInputError):
    """Raised when a declared event/time pair cannot be reconciled.

    A subclass of ``DescriptiveInputError`` so an existing ``except`` around the
    host's descriptive boundaries keeps catching it, and so the runtime failure
    classifier that already recognises that type does not need a second entry.
    """


def _placeable_event_times(
    values: pd.Series,
    *,
    index: pd.Index,
) -> pd.Series:
    """Rows whose event time is a finite number, by the host's own conversion.

    Reuses ``_semantic_numeric`` rather than ``pd.to_numeric``: a time column
    arriving as a string, a Decimal or an extension dtype has to be judged the
    same way every other host boundary judges it, or "unusable" would mean
    something different here than one module away.
    """

    numeric, semantic_valid = _semantic_numeric(values, allow_boolean=False)
    present = numeric.notna()
    finite = pd.Series(False, index=index, dtype=bool)
    if bool(present.any()):
        finite.loc[present] = np.isfinite(numeric.loc[present].to_numpy())
    return semantic_valid & present & finite


def event_time_reconciliation_receipt(
    frame: pd.DataFrame,
    *,
    event_column: str,
    time_column: str,
    event_levels: Sequence[Any],
    censored_level: Any,
) -> dict[str, Any]:
    """Reconcile a declared event flag against its event time, or fail closed.

    ``event_levels`` is the closed level set of ``event_column`` -- the same set
    the endpoint declares -- and ``censored_level`` is which of those codes means
    "no event observed". Both are the caller's declaration: this function does
    not decide which value means death, and a level set it had to infer would be
    the guess the endpoint declaration exists to remove.

    Raises when any row is unreconcilable:

    * an event code outside the declared closed set (an undeclared value must
      stop the step, not be counted as a non-event);
    * an event row whose event time is missing or non-finite -- unplaceable on
      the follow-up axis;

    and returns a receipt otherwise. A censored row with no event time is NOT a
    defect: that is the expected shape, and counting it as one is how non-events
    were excluded from a survival denominator.
    """

    if not isinstance(frame, pd.DataFrame):
        raise TypeError("event_time_reconciliation_receipt requires a DataFrame")

    declared_levels = list(event_levels)
    if len(declared_levels) < 2:
        raise SurvivalInputError(
            "an event column needs at least two declared levels: the censored "
            "code plus one code per event type",
            audit={"row_n": int(len(frame)), "undeclared_event_code_n": 0},
        )
    if not any(bool(level == censored_level) for level in declared_levels):
        raise SurvivalInputError(
            f"the censored level {censored_level!r} is not one of the declared "
            f"event levels {declared_levels!r}",
            audit={"row_n": int(len(frame)), "undeclared_event_code_n": 0},
        )
    if event_column == time_column:
        raise SurvivalInputError(
            "the event flag and the event time must be two distinct columns",
            audit={"row_n": int(len(frame)), "undeclared_event_code_n": 0},
        )
    missing_columns = [
        column for column in (event_column, time_column) if column not in frame.columns
    ]
    if missing_columns:
        raise SurvivalInputError(
            f"declared event/time columns missing: {missing_columns}",
            audit={"row_n": int(len(frame)), "undeclared_event_code_n": 0},
        )

    codes = frame[event_column]
    declared = pd.Series(False, index=frame.index, dtype=bool)
    for level in declared_levels:
        try:
            declared = declared | codes.eq(level).fillna(False)
        except (TypeError, ValueError):
            # An exotic comparison must become a typed fail-closed audit, never
            # leak a raw pandas exception into the coder's repair loop.
            continue
    undeclared_n = int((~declared).sum())

    try:
        is_censored = codes.eq(censored_level).fillna(False)
    except (TypeError, ValueError):
        is_censored = pd.Series(False, index=frame.index, dtype=bool)
    is_event = declared & ~is_censored

    placeable = _placeable_event_times(frame[time_column], index=frame.index)
    unplaceable_event = is_event & ~placeable

    audit = {
        "row_n": int(len(frame)),
        "event_n": int(is_event.sum()),
        "censored_n": int(is_censored.sum()),
        "undeclared_event_code_n": undeclared_n,
        "event_without_placeable_time_n": int(unplaceable_event.sum()),
        # Reported, never treated as a defect: the expected shape for a row that
        # had no event. Present so a caller writing a complete-case filter can
        # see the size of what excluding on this column would remove.
        "censored_without_time_n": int((is_censored & ~placeable).sum()),
    }

    if undeclared_n:
        raise SurvivalInputError(
            f"{undeclared_n} row(s) carry an event code outside the declared "
            f"closed set {declared_levels!r}; an undeclared code must stop the "
            "step, not be counted as a non-event",
            audit=audit,
        )
    if audit["event_without_placeable_time_n"]:
        raise SurvivalInputError(
            f"{audit['event_without_placeable_time_n']} row(s) record an event "
            f"in {event_column!r} with no usable time in {time_column!r}. The "
            "event cannot be placed on the follow-up axis. Recoding these to "
            "'no event' moves them into the survivor arm; dropping them through "
            "duration/event missingness removes them from both arms. Exclude "
            "them explicitly with this count reported, or censor them at a "
            "declared last-contact time -- either way it must appear in the "
            "step's own output.",
            audit=audit,
        )

    return {
        "event_column": event_column,
        "time_column": time_column,
        "declared_event_levels": declared_levels,
        "censored_level": censored_level,
        "status": "reconciled",
        "role": "audit_only",
        **audit,
    }
