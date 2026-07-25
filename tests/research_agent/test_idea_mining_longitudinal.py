"""Contracts for design-first longitudinal Idea Mining."""

from __future__ import annotations

import importlib
import inspect

import pyarrow as pa
import pyarrow.parquet as pq

from easyicu.research_agent.discovery.idea_mining_longitudinal import (
    LongitudinalArtifactProfile,
    generate_longitudinal_transportability_candidates,
    profile_longitudinal_table,
)


def _write_table(path, *, repeated: bool = True, value_name: str = "signal"):
    if repeated:
        unit = [1, 1, 1, 2, 2, 2, 3, 3, 3]
        time = [0, 1, 2] * 3
    else:
        unit = [1, 2, 3, 4]
        time = [0, 0, 0, 0]
    pq.write_table(
        pa.table(
            {
                "unit_id": unit,
                "charttime": time,
                value_name: [float(index) for index in range(len(unit))],
            }
        ),
        path,
    )


def _profile(database: str, *, concept: str = "generic_signal", ready: bool = True):
    return LongitudinalArtifactProfile(
        concept=concept,
        database=database,
        artifact_path=f"/{database}.parquet",
        artifact_sha256=(database * 64)[:64],
        row_count=100 if ready else 4,
        id_column="unit_id",
        time_column="charttime",
        value_column="signal",
        sample_row_count=100 if ready else 4,
        sample_unit_count=10 if ready else 4,
        sample_distinct_time_count=10 if ready else 1,
        sample_units_with_repeats=10 if ready else 0,
        sample_repeated_unit_fraction=1.0 if ready else 0.0,
        sample_median_observations_per_unit=10.0 if ready else 1.0,
        sample_value_nonnull_fraction=1.0,
    )


def test_profiles_explicit_repeated_measure_coordinates(tmp_path):
    path = tmp_path / "generic.parquet"
    _write_table(path)

    profiles = profile_longitudinal_table(
        path=path,
        database="db1",
        id_column="unit_id",
        time_column="charttime",
        value_columns=["signal"],
        concept_by_value_column={"signal": "Generic Signal"},
    )

    assert len(profiles) == 1
    profile = profiles[0]
    assert profile.concept == "generic_signal"
    assert profile.row_count == 9
    assert profile.sample_unit_count == 3
    assert profile.sample_distinct_time_count == 3
    assert profile.sample_repeated_unit_fraction == 1.0
    assert profile.sample_median_observations_per_unit == 3.0
    assert len(profile.artifact_sha256) == 64


def test_file_presence_without_repeated_measurements_is_not_ready(tmp_path):
    profiles = []
    for database in ("db1", "db2", "db3", "db4"):
        path = tmp_path / f"{database}.parquet"
        _write_table(path, repeated=False)
        profiles.extend(
            profile_longitudinal_table(
                path=path,
                database=database,
                id_column="unit_id",
                time_column="charttime",
                value_columns=["signal"],
            )
        )

    assert (
        generate_longitudinal_transportability_candidates(
            profiles=profiles, min_ready_databases=4
        )
        == []
    )


def test_empty_artifact_profiles_but_never_becomes_ready(tmp_path):
    path = tmp_path / "empty.parquet"
    pq.write_table(
        pa.table(
            {
                "unit_id": pa.array([], type=pa.int64()),
                "charttime": pa.array([], type=pa.int64()),
                "signal": pa.array([], type=pa.float64()),
            }
        ),
        path,
    )

    profiles = profile_longitudinal_table(
        path=path,
        database="db1",
        id_column="unit_id",
        time_column="charttime",
        value_columns=["signal"],
    )

    assert len(profiles) == 1
    assert profiles[0].sample_row_count == 0
    assert (
        generate_longitudinal_transportability_candidates(
            profiles=profiles, min_ready_databases=1
        )
        == []
    )


def test_emits_case_neutral_cross_database_trajectory_candidate():
    candidates = generate_longitudinal_transportability_candidates(
        profiles=[_profile(f"db{index}") for index in range(1, 7)],
        min_ready_databases=4,
    )

    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate.concept == "generic_signal"
    assert candidate.ready_database_count == 6
    assert candidate.analysis_family == "trajectory_clustering"
    assert candidate.design_archetype == "cross_database_trajectory_transportability"
    assert candidate.requires_human_confirmation is True
    assert candidate.novelty_claimed is False
    assert candidate.scientific_result_claimed is False
    assert candidate.paper_authorized is False
    assert "not a novelty or scientific-result claim" in candidate.differentiator_note


def test_requires_the_declared_number_of_ready_databases():
    profiles = [
        _profile("db1"),
        _profile("db2"),
        _profile("db3"),
        _profile("db4", ready=False),
    ]
    assert (
        generate_longitudinal_transportability_candidates(
            profiles=profiles, min_ready_databases=4
        )
        == []
    )


def test_duplicate_database_authority_fails_closed():
    profiles = [_profile("db1"), _profile("db1"), _profile("db2"), _profile("db3")]
    assert (
        generate_longitudinal_transportability_candidates(
            profiles=profiles, min_ready_databases=3
        )
        == []
    )


def test_module_is_leaf_and_does_not_import_main_idea_mining():
    source = inspect.getsource(
        importlib.import_module(
            "easyicu.research_agent.discovery.idea_mining_longitudinal"
        )
    )
    assert "import idea_mining" not in source
    assert "from .idea_mining import" not in source
