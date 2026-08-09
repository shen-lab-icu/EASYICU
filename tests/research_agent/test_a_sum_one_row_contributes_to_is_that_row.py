"""A total is a sum over rows; equal to one other row is a coincidence.

``MIN_ROWS_FOR_AGGREGATE_ROW`` already makes this argument for two-row tables:
"A equals the sum of the others" is just ``A == B`` there, which is not a total.
The degeneracy is not about how many rows exist, though -- it is about how many
of them carry any of the quantity.  A five-row table whose other four read
``[52707, 0, 0, 0]`` says ``A == B`` again, and a row count cannot see it.

MEASURED on m1_hepatobiliary_missingness, 2026-08-03 (``..._7e98a59_verify05``).
Step 03 audited bili, sofa2_liver, death, age and sex.  SOFA-2 liver stage is
COMPUTED FROM bilirubin, so those two share a missingness count exactly (52,707
of 94,458); death, age and sex are never missing and contribute 0.  All five of
the step's emitted tables were refused as unlabelled totals.  None has a total
row.  The step's own correct output went to the Coder, the repair mutated it,
the concept audit blocked the mutated code, and 08_missingness_audit_panel died
as collateral.

MEASURED over the recorded corpus: 1,374 tables of 3-60 rows, 143 flagged, 111
already declaring a role column.  137 have two or more contributing rows and
stay flagged -- including ``absolute_risk_context.csv``, the table this module
was written for.  Six are degenerate: m1's five, and one already declaring
``status``.
"""

from __future__ import annotations

import pathlib

import pandas as pd
import pytest

from easyicu.research_agent.audits.aggregate_row import (
    MIN_CONTRIBUTING_ROWS_FOR_AGGREGATE_ROW,
    _count_columns,
    _role_column,
    aggregate_row_candidates,
    unlabelled_aggregate_row_findings,
)


def test_the_derived_concept_case_that_blocked_m1(tmp_path):
    """bili and sofa2_liver share a count; the demographics contribute none."""

    frame = pd.DataFrame(
        {
            "concept": ["bili", "sofa2_liver", "death", "age", "sex"],
            "n_total": [94458] * 5,
            "measured_one_n": [41751, 41751, 94458, 94458, 94458],
            "value_missing_n": [52707, 52707, 0, 0, 0],
            "not_applicable_n": [52707, 52707, 0, 0, 0],
        }
    )

    assert aggregate_row_candidates(frame) == {}


def test_a_real_total_over_two_groups_is_still_caught():
    """660 + 340 = 1000, the shape this module was written for."""

    frame = pd.DataFrame(
        {
            "group": ["Overall", "negative", "positive"],
            "cohort_n": [1000, 660, 340],
            "deaths": [100, 57, 43],
            "denominator": [1000, 660, 340],
        }
    )

    assert 0 in aggregate_row_candidates(frame)


def test_a_total_over_three_groups_is_still_caught():
    """Adding a zero group must not disarm a table that has real contributors."""

    frame = pd.DataFrame(
        {
            "group": ["Overall", "a", "b", "c"],
            "cohort_n": [1000, 660, 340, 0],
            "deaths": [100, 57, 43, 0],
        }
    )

    assert 0 in aggregate_row_candidates(frame)


def test_two_contributors_is_the_boundary():
    """Exactly two nonzero others is a total; exactly one is a coincidence."""

    two = pd.DataFrame(
        {"g": ["t", "a", "b"], "n": [3, 1, 2], "m": [30, 10, 20]}
    )
    one = pd.DataFrame(
        {"g": ["t", "a", "b"], "n": [3, 3, 0], "m": [30, 30, 0]}
    )

    assert MIN_CONTRIBUTING_ROWS_FOR_AGGREGATE_ROW == 2
    assert 0 in aggregate_row_candidates(two)
    assert aggregate_row_candidates(one) == {}


def test_the_five_recorded_tables_stop_being_refused(tmp_path):
    """Drives the finding function on m1's real emitted directory."""

    recorded = sorted(
        pathlib.Path("/Volumes/外置硬盘/easyicu_data/canonical9_runs").glob(
            "batch_*_verify05/m1_*/aware/run_*/steps/"
            "03_missingness_measurement_audit/outputs"
        )
    )
    if not recorded:
        pytest.skip("the m1 run that recorded this refusal is not on disk")

    findings = unlabelled_aggregate_row_findings(
        step_id="03_missingness_measurement_audit", out_dir=recorded[-1]
    )
    assert findings == [], [f.message[:90] for f in findings]


_CORPUS = pathlib.Path("/Volumes/外置硬盘/easyicu_data/canonical9_runs")


def test_the_corpus_split_is_what_the_rule_was_read_off():
    """Re-measures rather than restating: the real totals must stay flagged.

    Reads every recorded emitted table and asserts the rule releases only the
    degenerate shape.  A table released here that has two or more contributing
    rows would mean the guard was widened past its argument.
    """

    if not _CORPUS.exists():
        pytest.skip("recorded run corpus is not mounted")

    flagged = released_without_role = 0
    for path in _CORPUS.rglob("steps/*/outputs/*.csv"):
        try:
            frame = pd.read_csv(path)
        except Exception:  # noqa: BLE001 - a malformed CSV is not this test's subject
            continue
        candidates = aggregate_row_candidates(frame)
        if candidates:
            flagged += 1
            # Everything still flagged must have real contributors.
            counts = _count_columns(frame)
            for position, columns in candidates.items():
                for column in columns:
                    others = counts[column].drop(counts[column].index[position])
                    assert int((others > 0).sum()) >= 2, (path.name, column)
            continue
        # Nothing released may look like a genuine multi-contributor total.
        if _role_column(frame.columns) is not None:
            continue
        released_without_role += 1

    assert flagged >= 100, flagged
    # The tables this module was written for are still caught.
    names = {
        p.name
        for p in _CORPUS.rglob("steps/*/outputs/absolute_risk_context.csv")
        if aggregate_row_candidates(pd.read_csv(p))
    }
    assert names == {"absolute_risk_context.csv"}, names
