"""A study that groups a measured value keeps the trajectory its run staged.

A run that reads a concept over a window stages a trajectory bound to the
run's copy of its cohort.  A grouping restages that cohort in place, and the
context rebuilt on it could no longer verify the trajectory, so on the
hosted pilot every grouped study whose data carried a trajectory stopped at
data preparation.  The trajectory is now staged again from its source on the
grouped cohort, and the context, the checkpoint and the execution read that
binding.  A signed trajectory suite, or a trajectory the run cannot stage
again, still takes no grouping.

Synthetic export only.
"""
from __future__ import annotations

import functools
from pathlib import Path

import pytest

from easyicu.research_agent.cohort import materializer
from easyicu.research_agent.intake.materialized_metadata import (
    EXPOSURE_GROUP_STAGE_PRODUCER,
    load_verified_materialized_cohort_authority,
    stage_materialized_cohort_authority,
)
from easyicu.research_agent.intake.materialized_trajectory import (
    StagedTrajectoryBinding,
    load_verified_materialized_trajectory_authority,
    stage_materialized_trajectory_authority,
)
from easyicu.research_agent.orchestration.exposure_grouping_phase import (
    RunTrajectory,
    run_exposure_grouping_phase,
    trajectory_takes_no_grouping,
)
from easyicu.research_agent.providers.mocks import ScriptedMockLLMClient
from tests.support.grouped_resume import ANSWER, context
from tests.support.typed_export import typed_export


def _stage(*, source_path: Path, source_cohort_path: Path, target_path: Path, target_cohort_path: Path):
    """The run's staging: an exact copy bound to a cohort of the same source."""

    source = load_verified_materialized_trajectory_authority(source_path)
    target = load_verified_materialized_cohort_authority(target_cohort_path)
    staged = stage_materialized_trajectory_authority(
        source_path, target_path,
        source_universe_path=source_cohort_path, target_universe_path=target_cohort_path,
        expected_source_authority=source.reference,
        expected_target_universe_authority=target.reference,
        producer_implementation_sha256="b" * 64,
    )
    return StagedTrajectoryBinding(
        path=target_path, sha256=staged.authority.trajectory_sha256,
        size=staged.authority.trajectory_size, authority_ref=staged.reference,
    )


@pytest.fixture
def run(tmp_path):
    paths = materializer.materialize_to_parquet(
        tmp_path / "source", stem="universe", data_path=typed_export(tmp_path / "export"),
        database="miiv", static_concepts=("age",), feature_concepts=("lact",),
        outcome_concepts=("death",), emit_trajectory=True, trajectory_concepts=("lact",),
        trajectory_window=(0.0, 24.0),
    )
    cohort = tmp_path / "run/cohort.parquet"
    stage_materialized_cohort_authority(paths["parquet"], cohort, producer_implementation_sha256="a" * 64)
    trajectory = RunTrajectory(
        binding=None,
        restage=functools.partial(
            _stage, source_path=paths["trajectory"], source_cohort_path=paths["parquet"]
        ),
    )
    trajectory.binding = trajectory.restage(
        target_path=cohort.with_name("cohort_trajectory.parquet"), target_cohort_path=cohort
    )
    return cohort, trajectory


def test_the_grouped_cohort_is_planned_with_the_trajectory_staged_on_it(run) -> None:
    cohort, trajectory = run
    staged_on_the_copy = trajectory.binding

    # As the plan phase rebuilds it: bound as first built, with the trajectory
    # the run staged unless the grouping staged it again.
    rebuild = functools.partial(context, trajectory_binding=staged_on_the_copy)

    phase = run_exposure_grouping_phase(
        context=rebuild(cohort), cohort_path=cohort,
        run_dir=cohort.parent, planner=ScriptedMockLLMClient([ANSWER]),
        rebuild_context=rebuild, capability_review_pending=False,
        trajectory_staged=trajectory_takes_no_grouping(trajectory, suite_authority=None),
        emit_progress=lambda *_args, **_kwargs: None,
        restage_trajectory=trajectory.on_grouped_cohort,
    )

    assert phase.context.variable("lact_group_x1") is not None
    grouped = load_verified_materialized_cohort_authority(cohort)
    assert grouped.authority.producer == EXPOSURE_GROUP_STAGE_PRODUCER
    # The checkpoint and the execution read the binding staged on the grouped
    # cohort: the same bytes, a new authority naming that cohort.
    restaged = trajectory.binding
    assert restaged.path == staged_on_the_copy.path
    assert (restaged.sha256, restaged.size) == (staged_on_the_copy.sha256, staged_on_the_copy.size)
    assert restaged.authority_ref != staged_on_the_copy.authority_ref
    assert load_verified_materialized_trajectory_authority(
        restaged.path,
        expected_authority=restaged.authority_ref,
        expected_universe_authority=grouped.reference,
    ) is not None


def test_a_suite_or_an_unrestageable_trajectory_takes_no_grouping(run) -> None:
    _cohort, trajectory = run

    assert trajectory_takes_no_grouping(trajectory, suite_authority=None) is False
    assert trajectory_takes_no_grouping(trajectory, suite_authority=object()) is True
    assert trajectory_takes_no_grouping(
        RunTrajectory(binding=trajectory.binding), suite_authority=None
    ) is True
    assert trajectory_takes_no_grouping(RunTrajectory(binding=None), suite_authority=object()) is False
