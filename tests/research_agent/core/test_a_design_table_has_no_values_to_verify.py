"""The lineage check demanded values from a table that has none.

``figure_source_data`` requires every typed parent a figure binds to be
independently value-verified, so a figure cannot claim a number no parent
supports.  A parent carrying no value column supports no number, and the rule
then becomes a demand nothing can meet: an exact row-for-row copy of it is
rejected with ``no_verifiable_values``.

MEASURED on e2, 2026-08-03 (``..._c8a1263_verify04``).  The robustness figure
renderer claimed its step for the first time -- the audit log records "Using
planner-scoped robustness figure executor" -- rendered the figure, wrote source
data for every bound parent, and was then failed closed on two findings: its
specification-grid companion "is not a traceable subset", and the bundle "does
not cover every bound result source", naming
``sensitivity_specification_grid.csv``.  The grid is the plan's own description
of what each specification CHANGES (``spec_id``/``axis``/``description`` plus
override columns that are empty).  None of it is a result.

The filter that exists for exactly this -- ``result_families`` deciding a bound
table is not a value source -- is skipped whenever the step declares no result
family, and a rendering-only figure step never declares one.  Measured over
every recorded plan, deduplicated per (run, step): 912 of 1052 visualization
steps (87%).  The guard is disabled in the case it is for.

HOW NARROW THE ADMISSION IS, measured rather than argued.  Of the 90 distinct
producer tables on disk, an exact self-copy is admitted by this branch for
exactly 3 -- ``sensitivity_specification_grid.csv``,
``sensitivity_specification_matrix.csv`` and ``specification_grid.csv``, which
are one artifact under the three names its producer writes it to.  The other 87
are still judged on their values.

TWO THINGS IT MUST NOT DO, both checked below.  A source that DROPS the values
of a value-bearing parent must still fail.  And with no value to verify,
faithful reproduction is the only verification left, so the copy must be exact
on every shared column -- not only on the fixed 22 in ``_TEXT_COLUMNS``, which
does not include the grid's own ``description``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pd = pytest.importorskip("pandas")

from easyicu.research_agent.audits.validators import (  # noqa: E402
    FigureSourceDataValidator as V,
)

#: One real grid row, from the e2 run that recorded the failure.  Every column
#: is text or empty; ``cohort_override``/``outcome_override`` read back as
#: all-null float columns, which is why "has a numeric dtype" is not the test.
_GRID = {
    "spec_id": ["complete_case_required_variables"],
    "axis": ["missing"],
    "description": [
        "Repeat the locked primary estimand using complete cases for the "
        "primary exposure, outcome, and prespecified covariates without "
        "imputing the exposure or outcome."
    ],
    "cohort_override": [None],
    "outcome_override": [None],
    "missing_override.strategy": ["complete_case"],
}

#: The sibling the same producer writes, which does carry results.
_MATRIX = {
    "spec_id": ["primary", "complete_case_required_variables"],
    "effect_scale": ["OR", "OR"],
    "point_estimate": [1.341455, 1.341455],
    "ci_low": [1.328695, 1.328695],
    "ci_high": [1.354338, 1.354338],
    "axis": ["primary", "missing"],
}


def _write(path: Path, payload: dict) -> Path:
    pd.DataFrame(payload).to_csv(path, index=False)
    return path


def _compare(source: Path, upstream: Path) -> dict:
    return V._compare_source_to_upstream(
        source_df=V._read_tabular(source),
        source_path=source,
        upstream_path=upstream,
    )


def test_a_faithful_copy_of_a_design_table_is_verified(tmp_path: Path) -> None:
    """The property that was false, and that killed the step."""

    grid = _write(tmp_path / "sensitivity_specification_grid.csv", _GRID)
    copy = _write(tmp_path / "figure_specification_grid_source_data.csv", _GRID)
    result = _compare(copy, grid)
    assert result["ok"] is True
    assert result["reason"] == "valueless_parent_reproduced"
    # Nothing was claimed as verified, because nothing was.
    assert result["verified_value_mappings"] == {}


def test_a_source_that_drops_the_parents_values_still_fails(tmp_path: Path) -> None:
    """Why this is not keyed on the source alone.

    Keeping only the key columns of a result table would make the SOURCE
    value-less.  If that were enough, a figure could shed the numbers it is
    supposed to be answerable for and call the parent covered.
    """

    matrix = _write(tmp_path / "robustness_matrix.csv", _MATRIX)
    stripped = _write(
        tmp_path / "figure_source_data.csv",
        {"spec_id": _MATRIX["spec_id"], "axis": _MATRIX["axis"]},
    )
    result = _compare(stripped, matrix)
    assert result["ok"] is False
    assert result["reason"] == "no_verifiable_values"


def test_the_copy_must_be_exact_on_a_column_outside_the_text_list(
    tmp_path: Path,
) -> None:
    """``description`` is not one of the 22 names ``_TEXT_COLUMNS`` compares.

    With no value to verify, an unchecked text column is an unchecked table.
    The sentence the plan registered for a specification is exactly the kind of
    thing a reader would take on trust.
    """

    assert "description" not in V._TEXT_COLUMNS  # the reason this test exists

    grid = _write(tmp_path / "sensitivity_specification_grid.csv", _GRID)
    tampered = dict(_GRID)
    tampered["description"] = ["a sentence the plan never registered"]
    altered = _write(tmp_path / "figure_grid_source_data.csv", tampered)
    result = _compare(altered, grid)
    assert result["ok"] is False
    assert result["reason"] == "source_values_disagree"
    assert result["mismatches"][0]["column"] == "description"


@pytest.mark.parametrize("column", ["missing_override.strategy", "axis"])
def test_every_shared_column_is_compared_not_just_the_first(
    tmp_path: Path, column: str
) -> None:
    """The exactness check covers the whole row, not one column of it."""

    grid = _write(tmp_path / "sensitivity_specification_grid.csv", _GRID)
    tampered = dict(_GRID)
    tampered[column] = ["something-else"]
    altered = _write(tmp_path / "figure_grid_source_data.csv", tampered)
    result = _compare(altered, grid)
    assert result["ok"] is False
    assert result["mismatches"][0]["column"] == column
