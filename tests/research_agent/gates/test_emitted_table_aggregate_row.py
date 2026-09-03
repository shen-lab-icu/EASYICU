"""A table with a total row nobody can identify is a table read wrongly.

The 2026-08-01 E1 run died here.  Step 04 emitted ``absolute_risk_context.csv``
with two exposure-level rows and an ``Overall`` row, and no column saying which
was which.  Step 07 -- the figure -- recomputed the prevalence percentages from
the counts to verify them, recovering the cohort denominator as
``cohort_n.sum()``: 660 + 340 + 1000 = 2000.  Every percentage then disagreed
with the table by a factor of two, the step raised its own reconciliation
guard, spent both repairs and both rewrites, and died.  The numbers in the
table were all correct.

The host already owns the fix and applies it in exactly one place:
``exposure_outcome_distribution_executor`` writes ``row_role`` with values
``exposure_level`` and ``overall``.  Nothing required a generated table to do
the same.

The measurement that decided the detector's shape, over the recorded corpus
(494 emitted tables of 3-60 rows): 57 contain a row equal to the sum of the
others in two or more independent count columns, and **35 of those already
declare a role column**.  A detector that fires on the tables whose own authors
labelled them is detecting total rows, not coincidences.  The remaining 22 are
the silent ones.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pandas as pd
import pytest

from easyicu.research_agent.audits.aggregate_row import (
    AGGREGATE_ROW_ROLE_COLUMNS,
    LEVEL_ROW_ROLE,
    MIN_AGREEING_COUNT_COLUMNS,
    OVERALL_ROW_ROLE,
    aggregate_row_candidates,
    unlabelled_aggregate_row_findings,
)

# The exact bytes step 04 emitted on 2026-08-01, header and all.
REAL_ABSOLUTE_RISK_CONTEXT = """\
exposure_group,cohort_n,cohort_pct,sep3_prevalence_n,sep3_prevalence_pct_full_cohort,\
death_event_n,death_non_event_n,death_missing_n,mortality_denominator_n,\
mortality_pct_group_denominator
0.0,660,66.0,660,66.0,57,603,0,660,8.636363636363637
1.0,340,34.0,340,34.0,45,295,0,340,13.235294117647058
Overall,1000,100.0,340,34.0,102,898,0,1000,10.2
"""


def _write(tmp_path: Path, name: str, text: str) -> Path:
    out = tmp_path / "outputs"
    out.mkdir(exist_ok=True)
    (out / name).write_text(text)
    return out


# --- the real table -----------------------------------------------------------


def test_the_real_table_that_killed_the_figure_step_is_refused(tmp_path: Path) -> None:
    out = _write(tmp_path, "absolute_risk_context.csv", REAL_ABSOLUTE_RISK_CONTEXT)

    findings = unlabelled_aggregate_row_findings(step_id="04_x", out_dir=out)

    assert len(findings) == 1
    detail = findings[0].detail
    assert detail["reason"] == "unlabelled_aggregate_row"
    assert detail["row_position"] == 2
    # Five independent additive columns agree that row 2 is the total. That is
    # what makes this a total row rather than an arithmetic accident.
    #
    # ``cohort_pct`` is in the set because 66 + 34 == 100 and all three values
    # are whole: the rule keys on additivity, not on a column's name or units.
    # A percentage column whose rows sum to the row in question is evidence of
    # the same thing a count column is, so it is not excluded -- and the corpus
    # split that validates this rule was measured with it included.
    assert set(detail["agreeing_count_columns"]) == {
        "cohort_n",
        "cohort_pct",
        "death_event_n",
        "death_non_event_n",
        "mortality_denominator_n",
    }
    assert findings[0].severity == "error"


def test_the_same_table_with_the_row_labelled_is_accepted(tmp_path: Path) -> None:
    """The fix the refusal asks for must actually satisfy it."""

    frame = pd.read_csv(pd.io.common.StringIO(REAL_ABSOLUTE_RISK_CONTEXT))
    frame["row_role"] = [LEVEL_ROW_ROLE, LEVEL_ROW_ROLE, OVERALL_ROW_ROLE]
    out = tmp_path / "outputs"
    out.mkdir()
    frame.to_csv(out / "absolute_risk_context.csv", index=False)

    assert unlabelled_aggregate_row_findings(step_id="04_x", out_dir=out) == []


def test_the_refusal_names_a_spelling_the_host_already_writes() -> None:
    """The Coder is told an existing name, not asked to invent one."""

    from easyicu.research_agent.execution.runners import (
        exposure_outcome_distribution_executor as producer,
    )

    source = Path(producer.__file__).read_text()
    assert f'_OVERALL_ROLE = "{OVERALL_ROW_ROLE}"' in source
    assert f'_LEVEL_ROLE = "{LEVEL_ROW_ROLE}"' in source
    assert "row_role" in AGGREGATE_ROW_ROLE_COLUMNS


@pytest.mark.parametrize("role_column", AGGREGATE_ROW_ROLE_COLUMNS)
def test_any_role_spelling_the_host_reads_elsewhere_is_accepted(
    tmp_path: Path, role_column: str
) -> None:
    """This module states the obligation; it does not also police the spelling.

    Every name here is one some other host reader already accepts, so a table
    that satisfies that reader must not be refused by this one.
    """

    frame = pd.read_csv(pd.io.common.StringIO(REAL_ABSOLUTE_RISK_CONTEXT))
    frame[role_column] = ["level", "level", "total"]
    out = tmp_path / "outputs"
    out.mkdir()
    frame.to_csv(out / "t.csv", index=False)

    assert unlabelled_aggregate_row_findings(step_id="s", out_dir=out) == []


# --- what must NOT be refused -------------------------------------------------


def test_two_equal_groups_are_not_a_total_row(tmp_path: Path) -> None:
    """With two rows, "A equals the sum of the others" is just A == B.

    This is the whole job of the row-count floor: a balanced two-arm table
    satisfies the additivity test in every column at once and means nothing by
    it.  Both rows would be reported as the total.
    """

    out = _write(tmp_path, "t.csv", "g,n,events\na,500,50\nb,500,50\n")
    assert unlabelled_aggregate_row_findings(step_id="s", out_dir=out) == []


def test_one_agreeing_column_is_not_enough(tmp_path: Path) -> None:
    """A single column agreeing is the coincidence the threshold exists for."""

    # n: 1 + 1 == 2 agrees on row 2; events disagrees (7 + 9 != 3).
    out = _write(tmp_path, "t.csv", "g,n,events\na,1,7\nb,1,9\nc,2,3\n")
    assert aggregate_row_candidates(pd.read_csv(out / "t.csv")) == {}
    assert unlabelled_aggregate_row_findings(step_id="s", out_dir=out) == []


def test_a_plain_partition_with_no_total_row_is_accepted(tmp_path: Path) -> None:
    out = _write(tmp_path, "t.csv", "g,n,events\na,660,57\nb,340,45\nc,120,11\n")
    assert unlabelled_aggregate_row_findings(step_id="s", out_dir=out) == []


def test_fractional_columns_are_not_a_partition(tmp_path: Path) -> None:
    """Three effect sizes that happen to add up are not a total row.

    The values are chosen to sum EXACTLY in binary floating point
    (0.25 + 0.5 == 0.75, 0.125 + 0.25 == 0.375), so this test turns on the
    whole-number rule rather than on float comparison luck -- an earlier
    version of it used 8.6 + 13.2 != 21.8 and proved nothing.
    """

    out = _write(
        tmp_path,
        "t.csv",
        "g,hazard_ratio,risk_difference\na,0.25,0.125\nb,0.5,0.25\nc,0.75,0.375\n",
    )
    assert aggregate_row_candidates(pd.read_csv(out / "t.csv")) == {}
    assert unlabelled_aggregate_row_findings(step_id="s", out_dir=out) == []


def test_negative_columns_are_not_a_partition(tmp_path: Path) -> None:
    """A signed difference column adds up without partitioning anything."""

    out = _write(
        tmp_path,
        "t.csv",
        "g,delta,shift\na,-5,-2\nb,8,6\nc,3,4\n",
    )
    assert aggregate_row_candidates(pd.read_csv(out / "t.csv")) == {}


def test_an_all_zero_column_cannot_mark_every_row_as_the_total(
    tmp_path: Path,
) -> None:
    """0 == sum of zeros holds for every row; it is not evidence of anything.

    Without the positive-sum requirement in the candidate scan, a table with
    two all-zero columns would report EVERY row as the total row.
    """

    out = _write(
        tmp_path,
        "t.csv",
        "g,missing_n,excluded_n\na,0,0\nb,0,0\nc,0,0\n",
    )
    assert aggregate_row_candidates(pd.read_csv(out / "t.csv")) == {}
    assert unlabelled_aggregate_row_findings(step_id="s", out_dir=out) == []


def test_an_unreadable_or_absent_directory_is_silent(tmp_path: Path) -> None:
    assert (
        unlabelled_aggregate_row_findings(step_id="s", out_dir=tmp_path / "nope") == []
    )


def test_a_malformed_csv_is_skipped_not_guessed(tmp_path: Path) -> None:
    out = _write(tmp_path, "t.csv", '\x00\x01 not, a ,csv\n"unterminated\n')
    unlabelled_aggregate_row_findings(step_id="s", out_dir=out)  # must not raise


# --- the wiring, proved from the source ---------------------------------------


def test_the_shared_pre_registration_sequence_calls_it() -> None:
    """It must run in the gate that feeds the repair loop, not only at the end.

    The early gate and the final gate evaluate this one sequence, so a call
    here is a call in both; a finding raised here costs one repair, while the
    same defect found after execution costs the step.
    """

    from easyicu.research_agent.gates import contract

    tree = ast.parse(Path(contract.__file__).read_text())
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        if node.name != "_step_deterministic_contract_findings":
            continue
        called = {
            ast.unparse(child.func)
            for child in ast.walk(node)
            if isinstance(child, ast.Call)
        }
        assert "unlabelled_aggregate_row_findings" in called
        return
    raise AssertionError("_step_deterministic_contract_findings not found")


# --- the recorded corpus ------------------------------------------------------

_CORPUS = Path("/Volumes/外置硬盘/easyicu_data/canonical9_runs")


@pytest.mark.skipif(
    not _CORPUS.exists(), reason="recorded runs are not on this machine"
)
def test_most_flagged_tables_already_label_the_row_themselves() -> None:
    """The property that makes this a detector rather than a heuristic.

    If the rule fired mostly on tables whose authors saw no total row, it would
    be finding arithmetic accidents.  It fires on a majority that DID label one
    -- so the shape it keys on is the shape producers themselves recognise.
    """

    labelled = unlabelled = 0
    for path in sorted(_CORPUS.glob("batch_*/*/aware/run_*/steps/*/outputs/*.csv")):
        try:
            frame = pd.read_csv(path)
        except Exception:
            continue
        if not aggregate_row_candidates(frame):
            continue
        if any(
            str(c).strip().lower() in AGGREGATE_ROW_ROLE_COLUMNS for c in frame.columns
        ):
            labelled += 1
        else:
            unlabelled += 1

    if labelled + unlabelled == 0:
        pytest.skip("no recorded table carries an aggregate row")
    assert labelled > unlabelled, (
        f"{labelled} flagged tables declare a role column and {unlabelled} do not; "
        "if that inverts, the rule has started firing on coincidences"
    )
    assert MIN_AGREEING_COUNT_COLUMNS >= 2
