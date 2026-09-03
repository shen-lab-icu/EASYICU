"""A declared window grid has to be in the cohort before the cohort is sealed.

``trajectory/contract.py`` parses ``<family>_h<start>_<end>`` off the sealed
cohort's columns, and the trajectory plan contract's primary path needs two of
them from one family.  The wide cohort carried whole-stay summaries while the
per-timepoint values lived only in the long table beside it -- which the
materializer built AFTER the cohort was already written, so no window column
could ever reach the seal.  Over the recorded corpus the parser answered
``None`` for every column of all 234 sealed research contexts.

Two properties pull against each other here:

* declaring a grid must actually put the columns in the sealed cohort, and
* NOT declaring one must leave every existing caller's output unchanged --
  this function seals a benchmark authority, so a stray column is a digest
  change nobody asked for.

Driven through the real materialization path against the synthetic typed export
already used by ``test_materialized_trajectory_authority``, rather than a stub:
the provenance, sealing and authority layers are exactly what a window column
has to survive.
"""

from __future__ import annotations

import inspect
import json

import pandas as pd
import pytest

from easyicu.research_agent.cohort import materializer as cohort_materializer
from easyicu.research_agent.trajectory.contract import (
    infer_fixed_window_trajectory_metadata,
)
from easyicu.research_agent.trajectory.panel import FixedWindowGrid

from tests.research_agent.figures.test_materialized_trajectory_authority import _typed_export


def _materialize(tmp_path, **kwargs):
    tmp_path.mkdir(parents=True, exist_ok=True)
    return cohort_materializer.materialize_to_parquet(
        tmp_path / "materialized",
        stem="universe",
        data_path=_typed_export(tmp_path / "export"),
        database="miiv",
        static_concepts=("age",),
        feature_concepts=("lact",),
        outcome_concepts=("death",),
        emit_trajectory=True,
        trajectory_concepts=("lact",),
        trajectory_window=(0.0, 24.0),
        **kwargs,
    )


def test_not_declaring_a_grid_leaves_the_cohort_exactly_as_before(tmp_path):
    """The default must not change a byte: this seals a benchmark authority."""

    paths = _materialize(tmp_path)
    sealed = pd.read_parquet(paths["parquet"])

    assert not [c for c in sealed.columns if "_h0_" in c or "_h12_" in c]


def test_a_declared_grid_puts_readable_window_columns_in_the_sealed_cohort(tmp_path):
    paths = _materialize(
        tmp_path,
        trajectory_panel_grid=FixedWindowGrid(width_hours=1.0, horizon_hours=3.0),
    )
    sealed = pd.read_parquet(paths["parquet"])

    windows = [column for column in sealed.columns if column.startswith("lact_h")]
    assert windows, list(sealed.columns)
    for column in windows:
        metadata = infer_fixed_window_trajectory_metadata(
            column_name=column, values=sealed[column], source_scale="continuous"
        )
        assert metadata is not None, column
        assert metadata.family == "lact"
    # The contract's primary path needs two windows from one family.
    assert len(windows) >= 2, windows


def test_the_window_values_are_the_ones_the_long_table_holds(tmp_path):
    """The wiring must not reduce differently from the panel it calls.

    The export records lact 1.0 at hour 1 and 2.0 at hour 2 for stay 1, and
    3.0 at hour 1 for stay 2.
    """

    paths = _materialize(
        tmp_path,
        trajectory_panel_grid=FixedWindowGrid(width_hours=1.0, horizon_hours=3.0),
    )
    sealed = pd.read_parquet(paths["parquet"]).set_index("stay_id")

    assert sealed.loc[1, "lact_h1_2"] == 1.0
    assert sealed.loc[1, "lact_h2_3"] == 2.0
    assert sealed.loc[2, "lact_h1_2"] == 3.0
    # Stay 2 has nothing in the later window; that cell stays missing rather
    # than reading as a lactate of zero.
    assert pd.isna(sealed.loc[2, "lact_h2_3"])


def test_the_existing_summary_columns_survive_untouched(tmp_path):
    plain = pd.read_parquet(_materialize(tmp_path / "a")["parquet"])
    windowed = pd.read_parquet(
        _materialize(
            tmp_path / "b",
            trajectory_panel_grid=FixedWindowGrid(width_hours=1.0, horizon_hours=3.0),
        )["parquet"]
    )

    for column in plain.columns:
        assert column in windowed.columns, column
        pd.testing.assert_series_equal(
            plain[column], windowed[column], check_names=False
        )


def test_the_grid_is_recorded_so_a_reader_can_replay_it(tmp_path):
    grid = FixedWindowGrid(width_hours=1.0, horizon_hours=3.0)
    paths = _materialize(tmp_path, trajectory_panel_grid=grid)

    text = json.dumps(
        json.loads(paths["provenance"].read_text(encoding="utf-8")), ensure_ascii=False
    )
    if "trajectory_panel_grid" not in text:
        authority = paths.get("cohort_authority")
        assert authority is not None, "the grid is recorded nowhere a reader can find"
        text = authority.read_text(encoding="utf-8")
    assert "trajectory_panel_grid" in text
    assert '"width_hours": 1.0' in text or '"width_hours":1.0' in text


def test_the_long_table_is_built_once_even_when_a_grid_is_declared(
    tmp_path, monkeypatch
):
    """Declaring a grid must not cost a second pass over the raw streams."""

    calls = {"n": 0}
    inner = cohort_materializer.build_trajectory_long

    def _counted(**kwargs):
        calls["n"] += 1
        return inner(**kwargs)

    monkeypatch.setattr(cohort_materializer, "build_trajectory_long", _counted)
    _materialize(
        tmp_path,
        trajectory_panel_grid=FixedWindowGrid(width_hours=1.0, horizon_hours=3.0),
    )

    assert calls["n"] == 1, calls


def test_a_window_column_may_not_overwrite_a_materialized_feature(
    tmp_path, monkeypatch
):
    """Silently replacing a feature column would be worse than refusing."""

    inner = cohort_materializer._materialize_cohort_with_metadata

    def _with_clash(**kwargs):
        cohort, provenance, collector = inner(**kwargs)
        cohort = cohort.assign(lact_h1_2=1.0)
        return cohort, provenance, collector

    monkeypatch.setattr(
        cohort_materializer, "_materialize_cohort_with_metadata", _with_clash
    )

    with pytest.raises(ValueError, match="already exist in the cohort"):
        _materialize(
            tmp_path,
            trajectory_panel_grid=FixedWindowGrid(width_hours=1.0, horizon_hours=3.0),
        )


def test_the_default_keeps_the_public_signature_backward_compatible():
    parameter = inspect.signature(
        cohort_materializer.materialize_to_parquet
    ).parameters["trajectory_panel_grid"]

    assert parameter.default is None
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
