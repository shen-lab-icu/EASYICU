"""The window grid is a case decision, declared once, by the task that needs it.

The engine's ``FixedWindowGrid`` chooses no family, width, horizon or aggregate;
its parser only requires that a grid be uniform.  Which grid a study uses is a
clinical choice about time resolution, so it lives in the task's own
materialization spec beside the note that asks for it -- "Build fixed-anchor
ICU-hour trajectories over hours 0-72 ... Use a common time grid, make
missingness explicit."

12 h over 0-72 gives six points per family.  MEASURED on the sealed long table:
sofa2 total is present in 100.0 / 97.7 / 92.8 / 87.7 / 82.1 / 76.7 % of stays
across those six windows, and 97.7 % of stays have at least two.  The decline is
discharge and death rather than measurement failure.

The other eight tasks must declare nothing.  Their cohorts are sealed benchmark
authorities, and a window column none of them asked for would be a digest change
nobody requested.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys

import pytest

_REPO = pathlib.Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from benchmarks.figure2_canonical9.materialization_plan import (  # noqa: E402
    CANONICAL9_MIMIC_IV_PLAN,
)


def _materializer_module():
    path = _REPO / "tools" / "materialize_canonical9_miiv.py"
    if not path.is_file():
        pytest.skip("the canonical9 materializer tool is not present")
    spec = importlib.util.spec_from_file_location("_canonical9_materializer", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_exactly_the_trajectory_task_compiles_a_grid():
    module = _materializer_module()

    declared = {
        spec.task_id
        for spec in CANONICAL9_MIMIC_IV_PLAN
        if module._panel_grid(spec) is not None
    }

    assert declared == {"h3_trajectory_clustering"}, declared


def test_the_declared_grid_is_the_one_the_protocol_describes():
    module = _materializer_module()
    spec = next(
        item
        for item in CANONICAL9_MIMIC_IV_PLAN
        if item.task_id == "h3_trajectory_clustering"
    )

    grid = module._panel_grid(spec)

    assert grid.width_hours == 12.0
    assert grid.horizon_hours == 72.0
    assert grid.aggregate == "max"
    assert len(grid.edges) == 6
    assert grid.edges[0] == (0.0, 12.0)
    assert grid.edges[-1] == (60.0, 72.0)


def test_the_horizon_comes_from_the_trajectory_window_not_a_second_declaration():
    """A grid must never describe hours the long table does not contain."""

    module = _materializer_module()
    spec = next(
        item
        for item in CANONICAL9_MIMIC_IV_PLAN
        if item.task_id == "h3_trajectory_clustering"
    )

    assert spec.trajectory_window is not None
    assert module._panel_grid(spec).horizon_hours == spec.trajectory_window[1]


def test_a_grid_without_a_trajectory_window_is_refused():
    import dataclasses

    module = _materializer_module()
    spec = next(
        item
        for item in CANONICAL9_MIMIC_IV_PLAN
        if item.task_id == "h3_trajectory_clustering"
    )
    orphaned = dataclasses.replace(spec, trajectory_window=None)

    with pytest.raises(ValueError, match="trajectory window"):
        module._panel_grid(orphaned)


def test_a_window_that_does_not_start_at_the_anchor_is_refused():
    """The grid's first window is the anchored one; a shifted start is not gridded."""

    import dataclasses

    module = _materializer_module()
    spec = next(
        item
        for item in CANONICAL9_MIMIC_IV_PLAN
        if item.task_id == "h3_trajectory_clustering"
    )
    shifted = dataclasses.replace(spec, trajectory_window=(6.0, 72.0))

    with pytest.raises(ValueError, match="trajectory anchor"):
        module._panel_grid(shifted)


def test_declaring_a_width_without_emitting_a_trajectory_declares_nothing():
    """There is no long table to summarize, so there is no panel to build."""

    import dataclasses

    module = _materializer_module()
    spec = next(
        item
        for item in CANONICAL9_MIMIC_IV_PLAN
        if item.task_id == "e1_sepsis3_prevalence_mortality"
    )
    assert not spec.emit_trajectory

    widened = dataclasses.replace(spec, trajectory_panel_width_hours=12.0)
    assert module._panel_grid(widened) is None
