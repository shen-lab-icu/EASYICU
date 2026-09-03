"""The specification-grid companion is verified against the grid, not the matrix.

The robustness figure binds its producer's specification grid and writes it out
verbatim as a source-data companion.  The figure source-data validator then tries
every candidate upstream table and accepts the companion if ANY of them matches.

Two of those candidates share ``spec_id`` with the companion:

* ``sensitivity_specification_grid.csv`` -- the companion's real parent, whose
  columns (``spec_id``, ``axis``, ``description``) are all text, and
* ``robustness_matrix.csv`` -- the estimates, which carry real values.

Before ``7e98a59`` a comparison whose two sides both lacked a value column was
refused as ``no_verifiable_values``, so BOTH candidates failed and the figure
step was blocked.  The refusal named the matrix, which reads as "the companion
disagrees with the estimates" -- it does not; it describes the specifications.

MEASURED on e2_lactate_mortality, 2026-08-03 (``..._c8a1263_verify04``, which
predates the fix).  Against the recorded artifacts of that run:

    before 7e98a59:  grid FAIL(no_verifiable_values)  matrix FAIL
    at HEAD:         grid OK(valueless_parent_reproduced)  matrix FAIL

The matrix is expected to keep failing: it has values, so it is not this
companion's parent.  One passing candidate is what the validator needs.
"""

from __future__ import annotations

import pathlib

import pandas as pd
import pytest

from easyicu.research_agent.audits.validators import FigureSourceDataValidator

_CORPUS = pathlib.Path("/Volumes/外置硬盘/easyicu_data/canonical9_runs")
_COMPANION = "robustness_plot_specification_grid_source_data.csv"


def _recorded_case():
    """The e2 run whose figure step this blocked, with its real artifacts."""

    companions = sorted(
        _CORPUS.glob(f"batch_*/e2_*/aware/run_*/steps/*/outputs/{_COMPANION}")
    )
    if not companions:
        pytest.skip("no recorded run emitted a specification-grid companion")
    companion = companions[-1]
    run = companion.parent.parent.parent.parent
    grids = sorted(run.glob("steps/*/outputs/sensitivity_specification_grid.csv"))
    matrices = sorted(run.glob("steps/*/outputs/robustness_matrix.csv"))
    if not grids or not matrices:
        pytest.skip("the recorded run has no producer grid beside its matrix")
    return companion, grids[-1], matrices[-1]


def _compare(source: pathlib.Path, upstream: pathlib.Path) -> dict:
    return FigureSourceDataValidator._compare_source_to_upstream(
        source_df=pd.read_csv(source),
        source_path=source,
        upstream_path=upstream,
    )


def test_the_companion_verifies_against_the_grid_that_produced_it():
    companion, grid, _matrix = _recorded_case()

    result = _compare(companion, grid)

    assert result.get("ok") is True, result.get("reason")
    assert result.get("reason") == "valueless_parent_reproduced"


def test_the_estimates_table_is_not_this_companion_s_parent():
    """It must keep failing -- the matrix has values the grid never describes.

    Accepting it would mean a companion could be authenticated against any
    table it merely shares a key with.
    """

    companion, _grid, matrix = _recorded_case()

    result = _compare(companion, matrix)

    assert result.get("ok") is False
    assert result.get("reason") == "no_verifiable_values"


def test_a_tampered_specification_is_still_refused():
    """The valueless branch compares every shared column exactly.

    ``description`` is the whole content of a specification, so a companion
    that reworded it must not pass as a faithful copy of its parent.
    """

    companion, grid, _matrix = _recorded_case()
    frame = pd.read_csv(companion)
    if "description" not in frame.columns or frame.empty:
        pytest.skip("the recorded companion carries no description column")
    frame.loc[frame.index[0], "description"] = "a specification nobody locked"

    result = FigureSourceDataValidator._compare_source_to_upstream(
        source_df=frame,
        source_path=companion,
        upstream_path=grid,
    )

    assert result.get("ok") is False
    assert result.get("reason") == "source_values_disagree"
