"""An emitted table that contains a total row must say which row that is.

The host already owns this vocabulary.  ``exposure_outcome_distribution_executor``
emits a ``row_role`` column whose values are ``exposure_level`` for the per-level
rows and ``overall`` for the whole-cohort row, and
``CrossStepReconciliationTraceValidator`` documents what goes wrong when a
consumer matches rows by label instead: it binds a prevalence row and reports
its denominator as the stratum N.

What was never required is that a *generated* table do the same.  So a step
emits two exposure-level rows plus an unlabelled "Overall" row, and the next
step -- which has only the bytes to go on -- sums the denominator column to
recover the cohort size and gets twice the real number.

Measured over the recorded corpus (494 emitted tables of 3-60 rows): 57 carry a
row whose value equals the sum of the other rows in two or more independent
count columns.  **35 of those already declare a role column** -- that agreement
is what says this detector is finding total rows rather than arithmetic
coincidences.  The other 22 are ambiguous to every consumer, and one of them is
``absolute_risk_context.csv`` from the 2026-08-01 E1 run, whose figure step
recomputed ``100 * n / cohort_n.sum()`` over 660 + 340 + 1000 = 2000, failed its
own reconciliation on every row, spent both repairs, and died.

Two or more agreeing columns is what separates a total row from a coincidence.
One column agreeing means little -- a two-group table of 1 and 1 with a third
row of 2 satisfies it by accident.  Four independent counts agreeing at once
(as in the real table: cohort_n, deaths, non-events, denominator) does not
happen by accident.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Mapping, Sequence

import pandas as pd

from ..schema import ValidationFinding

#: The column spellings elsewhere in the host already accept as a row's declared
#: role.  Collected from ``multiple_testing.py`` (``record_role`` / ``record_type``
#: / ``result_role`` / ``row_role``), ``validators.py`` (``row_role`` / ``row_type``
#: / ``group_type`` / ``estimate_type``), ``pipeline.py`` (``row_type`` /
#: ``summary_type`` / ``variable_class``) and ``declared_product.py`` (``status`` /
#: ``partition_status`` / ``row_role`` / ``role``).  A table declaring ANY of them
#: has told its consumers that its rows are not interchangeable, which is the
#: whole obligation here -- this module does not also police which spelling.
AGGREGATE_ROW_ROLE_COLUMNS: tuple[str, ...] = (
    "estimate_type",
    "group_type",
    "partition_status",
    "record_role",
    "record_type",
    "result_role",
    "role",
    "row_role",
    "row_type",
    "status",
    "summary_type",
    "variable_class",
)

#: The values the host's own distribution executor writes, offered in the
#: refusal so the Coder is told a spelling that already exists rather than
#: inventing one.
LEVEL_ROW_ROLE = "exposure_level"
OVERALL_ROW_ROLE = "overall"

#: A table smaller than this cannot distinguish "one row is the total" from
#: "two groups happen to be equal": with two rows, row A equals the sum of the
#: others exactly when A == B.
MIN_ROWS_FOR_AGGREGATE_ROW = 3

#: Above this, the shape is a long-form record dump rather than a partition
#: with a total, and the pairwise scan stops being meaningful.
MAX_ROWS_FOR_AGGREGATE_ROW = 60

#: How many independent count columns must agree that the same row is the total.
MIN_AGREEING_COUNT_COLUMNS = 2

#: How many of the OTHER rows must actually contribute to that sum.
#:
#: ``MIN_ROWS_FOR_AGGREGATE_ROW`` states the same argument in terms of rows: with
#: two rows, "A equals the sum of the others" is just ``A == B``, which is a
#: coincidence and not a total.  But the degeneracy is not about how many rows
#: exist -- it is about how many of them carry any of the quantity.  A five-row
#: table whose other four read ``[52707, 0, 0, 0]`` says exactly ``A == B``
#: again, and the row floor does not see it.
#:
#: MEASURED on m1_hepatobiliary_missingness, 2026-08-03 (``..._7e98a59_verify05``).
#: Step 03 emitted five audit tables over the concepts bili, sofa2_liver, death,
#: age and sex.  SOFA-2 liver stage is COMPUTED FROM bilirubin, so the two share
#: a missingness count to the row (52,707 of 94,458); death, age and sex are
#: never missing, so they contribute 0.  Row 0 therefore "equalled the sum of the
#: others" in several independent columns at once, and all five tables were
#: refused as unlabelled totals.  None of them has a total row.  The step's own
#: correct output went to the Coder for repair, the repair mutated it, the
#: concept audit then blocked the mutated code, and 08_missingness_audit_panel
#: died as collateral -- three of the task's failures from one arithmetic
#: coincidence with a clinical cause.
#:
#: MEASURED over the recorded corpus: 1,374 emitted tables of 3-60 rows, 143
#: flagged, 111 of which already declare a role column.  137 have two or more
#: contributing rows and stay flagged -- including ``absolute_risk_context.csv``
#: (13 runs), the table this module was written for, whose 660 + 340 = 1000 is
#: exactly the shape this rule preserves.  Six are degenerate: the five above,
#: and one that already declares ``status`` and so was never refused.
MIN_CONTRIBUTING_ROWS_FOR_AGGREGATE_ROW = 2


def _role_column(columns: Sequence[Any]) -> str | None:
    for column in columns:
        if str(column).strip().lower() in AGGREGATE_ROW_ROLE_COLUMNS:
            return str(column)
    return None


def _count_columns(frame: pd.DataFrame) -> dict[str, pd.Series]:
    """Columns of non-negative whole numbers: the ones that can partition.

    A negative or fractional column is an estimate, a rate or a difference, and
    one row there equalling the sum of the others carries no claim about a
    partition -- three effect sizes of 0.25, 0.5 and 0.75 are not a total row.

    A whole-valued PERCENTAGE column is deliberately NOT excluded.  In the real
    table that motivated this module, ``cohort_pct`` reads 66 / 34 / 100, and
    "the other rows sum to this one" is the same evidence there as in a count
    column -- units do not change what additivity means.  The corpus split that
    validates the rule was measured with such columns included.

    Two guards that were here have been removed as unreachable rather than left
    to look protective.  ``nunique() <= 1`` only ever mattered for a two-row
    table (in three or more rows a constant column c never satisfies
    ``sum(others) == c``, since that would need 2c == c), and the row-count
    floor below already excludes those.  ``sum() <= 0`` only ever mattered for
    an all-zero column, which the ``sum(others) > 0`` test in the candidate
    scan already refuses.  Each was a second spelling of a rule stated
    elsewhere, and a guard no input can reach is a guard no test can prove.
    """

    counts: dict[str, pd.Series] = {}
    for column in frame.columns:
        values = pd.to_numeric(frame[column], errors="coerce")
        if values.isna().any():
            continue
        if not (values.mod(1) == 0).all() or (values < 0).any():
            continue
        counts[str(column)] = values
    return counts


def aggregate_row_candidates(frame: pd.DataFrame) -> dict[int, list[str]]:
    """Rows equal to the sum of every other row, per agreeing count column."""

    if not (MIN_ROWS_FOR_AGGREGATE_ROW <= len(frame) <= MAX_ROWS_FOR_AGGREGATE_ROW):
        return {}
    counts = _count_columns(frame)
    if len(counts) < MIN_AGREEING_COUNT_COLUMNS:
        return {}
    agreeing: dict[int, list[str]] = {}
    for position in range(len(frame)):
        columns = []
        for column, values in counts.items():
            others = values.drop(values.index[position])
            if others.sum() <= 0 or others.sum() != values.iloc[position]:
                continue
            # A sum only one row contributes to is that row, and "equal to the
            # one other nonzero row" is the coincidence MIN_ROWS_FOR_AGGREGATE_ROW
            # already refuses to read as a total.
            if int((others > 0).sum()) < MIN_CONTRIBUTING_ROWS_FOR_AGGREGATE_ROW:
                continue
            columns.append(column)
        if len(columns) >= MIN_AGREEING_COUNT_COLUMNS:
            agreeing[position] = sorted(columns)
    return agreeing


def unlabelled_aggregate_row_findings(
    *,
    step_id: str,
    out_dir: Path,
) -> List[ValidationFinding]:
    """Refuse an emitted table whose total row no consumer can identify.

    Returned as an ``error`` from the shared pre-registration contract sequence,
    so the step spends one repair on labelling the row instead of handing the
    next step a table it will aggregate wrongly.
    """

    findings: List[ValidationFinding] = []
    try:
        paths = sorted(Path(out_dir).glob("*.csv"))
    except OSError:
        return findings
    for path in paths:
        try:
            frame = pd.read_csv(path)
        except Exception:
            continue
        if _role_column(frame.columns) is not None:
            continue
        candidates = aggregate_row_candidates(frame)
        if not candidates:
            continue
        position = min(candidates)
        columns = candidates[position]
        findings.append(
            ValidationFinding(
                validator="emitted_table_aggregate_row",
                severity="error",
                message=(
                    f"Table '{path.name}' emitted by step '{step_id}' contains a "
                    f"row (row {position}, 0-based, excluding the header) whose "
                    f"value equals the sum of every other row in {len(columns)} "
                    f"independent count columns ({', '.join(columns)}). That is a "
                    "total row, and nothing in the table says so, so a consumer "
                    "reading only these bytes cannot tell it apart from another "
                    f"group and will double every denominator it sums. Add a "
                    f"'{OVERALL_ROW_ROLE}'/'{LEVEL_ROW_ROLE}' column named "
                    f"'row_role' -- the spelling the host's own distribution "
                    "output already uses -- marking that row "
                    f"'{OVERALL_ROW_ROLE}' and each partition row "
                    f"'{LEVEL_ROW_ROLE}'. If the row is NOT a total, the "
                    "agreement across those columns is what needs explaining: "
                    "say what it is in the same column."
                ),
                detail={
                    "step_id": step_id,
                    "reason": "unlabelled_aggregate_row",
                    "table": path.name,
                    "row_position": position,
                    "agreeing_count_columns": columns,
                    "accepted_role_columns": list(AGGREGATE_ROW_ROLE_COLUMNS),
                    "row_count": int(len(frame)),
                },
            )
        )
    return findings


def aggregate_row_summary(frame: pd.DataFrame) -> Mapping[str, Any]:
    """Diagnostic projection used by tests and by corpus measurement."""

    return {
        "role_column": _role_column(frame.columns),
        "candidates": aggregate_row_candidates(frame),
    }


__all__ = [
    "AGGREGATE_ROW_ROLE_COLUMNS",
    "LEVEL_ROW_ROLE",
    "MAX_ROWS_FOR_AGGREGATE_ROW",
    "MIN_AGREEING_COUNT_COLUMNS",
    "MIN_ROWS_FOR_AGGREGATE_ROW",
    "OVERALL_ROW_ROLE",
    "aggregate_row_candidates",
    "aggregate_row_summary",
    "unlabelled_aggregate_row_findings",
]
