"""Typed long-format trajectory publication and staging authority."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from easyicu.concept.metadata_projection import (
    ColumnProjectionSpec,
    ConceptColumnRole,
    project_concept_column_metadata,
)
from easyicu.concept.metadata_sidecar import (
    EXPORT_PHYSICAL_SCOPE,
    ColumnMetadataBinding,
    ColumnMetadataFileBinding,
    ColumnMetadataSidecar,
    TimeCoordinate,
    write_content_addressed_sidecar,
)
from easyicu.resources import load_dictionary
from easyicu.research_agent.cohort import materializer as cohort_materializer
from easyicu.research_agent.authority.filesystem import AnchoredDirectory
from easyicu.research_agent.cohort.schema import (
    CohortDefinition,
    ConceptPredicate,
    TimeWindow,
)
from easyicu.research_agent.execution.development_sample import (
    materialize_development_execution_sample,
)
from easyicu.research_agent.intake import export_package as intake
from easyicu.research_agent.intake.materialized_metadata import (
    MaterializedCohortAuthorityRef,
    canonical_parameters_sha256,
    implementation_bundle_sha256,
    load_verified_materialized_cohort_authority,
    stage_materialized_cohort_authority,
)
from easyicu.research_agent.intake.materialized_trajectory import (
    MaterializedTrajectoryAuthority,
    MaterializedTrajectoryAuthorityRef,
    MaterializedTrajectoryError,
    StagedTrajectoryBinding,
    TRAJECTORY_COLUMNS,
    load_verified_materialized_trajectory_authority,
    publish_materialized_trajectory_authority,
    stage_materialized_trajectory_authority,
)


def _resign_selected_trajectory_authority(
    trajectory_path: Path,
    mutate,
) -> MaterializedTrajectoryAuthorityRef:
    selector_path = trajectory_path.with_name(f"{trajectory_path.stem}_provenance.json")
    selector = json.loads(selector_path.read_text(encoding="utf-8"))
    old_ref = MaterializedTrajectoryAuthorityRef.from_dict(
        selector["trajectory_authority"]["authority"]
    )
    payload = json.loads(
        (trajectory_path.parent / old_ref.file).read_text(encoding="utf-8")
    )
    mutate(payload)
    authority = MaterializedTrajectoryAuthority.from_dict(payload)
    raw = json.dumps(
        authority.to_dict(),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    digest = hashlib.sha256(raw).hexdigest()
    name = f"trajectory_authority.sha256-{digest}.json"
    (trajectory_path.parent / name).write_bytes(raw)
    reference = MaterializedTrajectoryAuthorityRef(
        file=name,
        sha256=digest,
        size=len(raw),
    )
    selector["trajectory_authority"]["authority"] = reference.to_dict()
    selector_path.write_text(
        json.dumps(selector, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return reference


def _binding(concept: str, column: str, role: ConceptColumnRole):
    definition = load_dictionary(include_sofa2=True).get(concept)
    assert definition is not None
    return ColumnMetadataBinding(
        metadata=project_concept_column_metadata(
            definition,
            spec=ColumnProjectionSpec(
                column_name=column,
                source_concept=concept,
                role=role,
            ),
            source_database="miiv",
        )
    )


def _typed_export(root: Path) -> Path:
    root.mkdir()
    labs = pd.DataFrame(
        {
            "stay_id": [1, 1, 2],
            "charttime": [1.0, 2.0, 1.0],
            "age": [50, 50, 60],
            "lact": [1.0, 2.0, 3.0],
        }
    )
    outcomes = pd.DataFrame({"stay_id": [1, 2], "death": [False, True]})
    labs.to_parquet(root / "labs.parquet", index=False)
    outcomes.to_parquet(root / "outcomes.parquet", index=False)
    lab_binding = ColumnMetadataFileBinding(
        relative_path="labs.parquet",
        module="labs",
        identity_column="stay_id",
        time_coordinates=(
            TimeCoordinate(column="charttime", origin="icu_admission", unit="h"),
        ),
        columns={
            "age": _binding("age", "age", ConceptColumnRole.VALUE),
            "lact": _binding("lact", "lact", ConceptColumnRole.VALUE),
        },
    )
    outcome_binding = ColumnMetadataFileBinding(
        relative_path="outcomes.parquet",
        module="outcomes",
        identity_column="stay_id",
        time_coordinates=(),
        columns={"death": _binding("death", "death", ConceptColumnRole.EVENT_STATUS)},
    )
    sidecar = ColumnMetadataSidecar(
        source_database="miiv",
        source_database_class_prefixes=(),
        scope=EXPORT_PHYSICAL_SCOPE,
        files=(lab_binding, outcome_binding),
    )
    reference = write_content_addressed_sidecar(root, sidecar)
    (root / intake.NATIVE_MANIFEST).write_text(
        json.dumps(
            {
                "schema_version": intake.NATIVE_MANIFEST_SCHEMA_V2,
                "database": "miiv",
                "format": "parquet",
                "concept_selection": {
                    "mode": "explicit",
                    "modules": {
                        "labs": ["age", "lact"],
                        "outcomes": ["death"],
                    },
                },
                "files": [
                    {
                        "file": "labs.parquet",
                        "module": "labs",
                        "concepts": 2,
                        "concept_ids": ["age", "lact"],
                        "rows": len(labs),
                        "column_metadata_columns": ["age", "lact"],
                    },
                    {
                        "file": "outcomes.parquet",
                        "module": "outcomes",
                        "concepts": 1,
                        "concept_ids": ["death"],
                        "rows": len(outcomes),
                        "column_metadata_columns": ["death"],
                    },
                ],
                "feature_definitions": {"included": False},
                "column_metadata": reference.to_dict(),
            }
        ),
        encoding="utf-8",
    )
    return root


def _bundle(tmp_path: Path):
    source = _typed_export(tmp_path / "export")
    paths = cohort_materializer.materialize_to_parquet(
        tmp_path / "materialized",
        stem="universe",
        data_path=source,
        database="miiv",
        static_concepts=("age",),
        feature_concepts=("lact",),
        outcome_concepts=("death",),
        emit_trajectory=True,
        trajectory_concepts=("lact",),
        trajectory_window=(0.0, 24.0),
    )
    cohort = load_verified_materialized_cohort_authority(paths["parquet"])
    trajectory = load_verified_materialized_trajectory_authority(paths["trajectory"])
    assert cohort is not None and trajectory is not None
    return paths, cohort, trajectory


def _implementation_sha() -> str:
    import easyicu.research_agent.intake.materialized_trajectory as module

    return implementation_bundle_sha256((Path(module.__file__),))


def test_materializer_publishes_verified_long_trajectory_authority(tmp_path):
    paths, cohort, trajectory = _bundle(tmp_path)

    assert tuple(pd.read_parquet(paths["trajectory"]).columns) == TRAJECTORY_COLUMNS
    assert trajectory.authority.bound_universe_authority == cohort.reference
    assert trajectory.authority.bound_universe_row_identity_sha256 == (
        cohort.authority.row_identity_sha256
    )
    assert trajectory.authority.time_origin == "icu_admission"
    assert trajectory.authority.time_unit == "h"
    assert trajectory.authority.window.to_dict() == {
        "origin": "icu_admission",
        "unit": "h",
        "start_hours": 0.0,
        "end_hours": 24.0,
        "inclusive": True,
    }
    assert trajectory.authority.requested_concepts == ("lact",)
    assert trajectory.authority.materialized_concepts == ("lact",)
    assert trajectory.authority.concept_bindings[0].binding.metadata.source_concept == (
        "lact"
    )
    assert paths["trajectory_authority"].name == trajectory.reference.file


def test_source_bound_concept_without_window_rows_is_available_unobserved(tmp_path):
    source = _typed_export(tmp_path / "export")
    paths = cohort_materializer.materialize_to_parquet(
        tmp_path / "materialized",
        stem="universe",
        data_path=source,
        database="miiv",
        static_concepts=("age",),
        feature_concepts=("lact",),
        outcome_concepts=("death",),
        emit_trajectory=True,
        trajectory_concepts=("lact",),
        trajectory_window=(10.0, 24.0),
    )

    trajectory = load_verified_materialized_trajectory_authority(paths["trajectory"])
    assert trajectory is not None
    frame = pd.read_parquet(paths["trajectory"])
    assert frame.empty
    assert tuple(frame.columns) == TRAJECTORY_COLUMNS
    assert trajectory.authority.requested_concepts == ("lact",)
    assert trajectory.authority.materialized_concepts == ()
    assert trajectory.authority.available_unobserved_concepts == ("lact",)
    assert trajectory.authority.unavailable_concepts == ()
    assert tuple(item.concept_id for item in trajectory.authority.concept_bindings) == (
        "lact",
    )
    assert trajectory.authority.trajectory_rows == 0
    assert trajectory.authority.trajectory_stays == 0
    payload = trajectory.authority.to_dict()
    assert payload["producer_parameters"]["available_unobserved_concepts"] == ["lact"]
    assert payload["semantic_provenance"]["available_unobserved_concepts"] == ["lact"]


@pytest.mark.parametrize(
    ("window", "materialized", "available_unobserved"),
    [
        ((0.0, 24.0), ("lact",), ()),
        ((10.0, 24.0), (), ("lact",)),
    ],
)
def test_trajectory_only_concept_binds_directly_to_sealed_source_metadata(
    tmp_path,
    window,
    materialized,
    available_unobserved,
):
    source = _typed_export(tmp_path / "export")
    paths = cohort_materializer.materialize_to_parquet(
        tmp_path / "materialized",
        stem="universe",
        data_path=source,
        database="miiv",
        static_concepts=("age",),
        feature_concepts=(),
        outcome_concepts=("death",),
        emit_trajectory=True,
        trajectory_concepts=("lact",),
        trajectory_window=window,
    )

    cohort = pd.read_parquet(paths["parquet"])
    assert "lact" not in cohort.columns
    trajectory = load_verified_materialized_trajectory_authority(paths["trajectory"])
    assert trajectory is not None
    assert trajectory.authority.materialized_concepts == materialized
    assert trajectory.authority.available_unobserved_concepts == available_unobserved
    assert trajectory.authority.unavailable_concepts == ()
    assert tuple(item.concept_id for item in trajectory.authority.concept_bindings) == (
        "lact",
    )
    binding = trajectory.authority.concept_bindings[0]
    assert binding.source.file == "labs.parquet"
    assert binding.source.column == "lact"
    assert binding.binding.metadata.source_concept == "lact"
    assert binding.binding.metadata.role is ConceptColumnRole.VALUE


def test_final_cohort_filter_reclassifies_materialized_concept_as_unobserved(
    tmp_path,
):
    source = _typed_export(tmp_path / "export")
    raw, raw_provenance = cohort_materializer.build_trajectory_long(
        data_path=source,
        concepts=("lact",),
        database="miiv",
        window=(1.5, 2.5),
    )
    assert raw["stay_id"].tolist() == [1]
    assert raw_provenance["trajectory_concepts_materialized"] == ["lact"]

    definition = CohortDefinition(
        name="older_adults",
        inclusion=(
            ConceptPredicate(
                concept_id="age",
                time_window=TimeWindow(
                    anchor="icu_admit",
                    start_offset_hours=0,
                    end_offset_hours=24,
                ),
                aggregation="first",
                op=">=",
                value=55,
            ),
        ),
    )
    paths = cohort_materializer.materialize_to_parquet(
        tmp_path / "materialized",
        stem="universe",
        data_path=source,
        database="miiv",
        static_concepts=("age",),
        feature_concepts=("lact",),
        outcome_concepts=("death",),
        cohort_definition=definition,
        emit_trajectory=True,
        trajectory_concepts=("lact",),
        trajectory_window=(1.5, 2.5),
    )

    assert pd.read_parquet(paths["parquet"])["stay_id"].tolist() == [2]
    trajectory = load_verified_materialized_trajectory_authority(paths["trajectory"])
    assert trajectory is not None
    assert pd.read_parquet(paths["trajectory"]).empty
    assert trajectory.authority.materialized_concepts == ()
    assert trajectory.authority.available_unobserved_concepts == ("lact",)
    assert trajectory.authority.unavailable_concepts == ()
    assert tuple(item.concept_id for item in trajectory.authority.concept_bindings) == (
        "lact",
    )
    payload = trajectory.authority.to_dict()
    assert payload["producer_parameters"]["materialized_concepts"] == []
    assert payload["producer_parameters"]["available_unobserved_concepts"] == ["lact"]
    assert payload["semantic_provenance"]["trajectory_concepts_materialized"] == []
    assert payload["semantic_provenance"]["available_unobserved_concepts"] == ["lact"]


@pytest.mark.parametrize(
    "mutator",
    [
        lambda frame: frame.drop(columns=["value_str"]),
        lambda frame: frame.assign(extra=1),
        lambda frame: frame.assign(charttime=[float("inf")]),
        lambda frame: frame.assign(concept=[""]),
        lambda frame: frame.assign(stay_id=[None]),
    ],
)
def test_invalid_trajectory_frame_is_rejected_before_publication(tmp_path, mutator):
    paths, cohort, _trajectory = _bundle(tmp_path)
    target = paths["parquet"].parent / "invalid_trajectory.parquet"
    base = pd.DataFrame(
        {
            "stay_id": [1],
            "charttime": [1.0],
            "concept": ["lact"],
            "value_num": [1.0],
            "value_str": ["1.0"],
        }
    )

    with pytest.raises(MaterializedTrajectoryError):
        publish_materialized_trajectory_authority(
            mutator(base),
            target,
            bound_universe_path=paths["parquet"],
            bound_universe=cohort,
            requested_concepts=("lact",),
            materialized_concepts=("lact",),
            available_unobserved_concepts=(),
            unavailable_concepts=(),
            window=(0.0, 24.0),
            semantic_provenance={},
            producer_implementation_sha256=_implementation_sha(),
            producer_parameters={},
        )
    assert not target.exists()
    assert not target.with_name("invalid_trajectory_provenance.json").exists()


def test_trajectory_rejects_identity_outside_bound_universe(tmp_path):
    paths, cohort, _trajectory = _bundle(tmp_path)
    target = paths["parquet"].parent / "foreign_trajectory.parquet"
    frame = pd.DataFrame(
        {
            "stay_id": [999],
            "charttime": [1.0],
            "concept": ["lact"],
            "value_num": [1.0],
            "value_str": ["1.0"],
        }
    )

    with pytest.raises(MaterializedTrajectoryError, match="outside"):
        publish_materialized_trajectory_authority(
            frame,
            target,
            bound_universe_path=paths["parquet"],
            bound_universe=cohort,
            requested_concepts=("lact",),
            materialized_concepts=("lact",),
            available_unobserved_concepts=(),
            unavailable_concepts=(),
            window=(0.0, 24.0),
            semantic_provenance={},
            producer_implementation_sha256=_implementation_sha(),
            producer_parameters={},
        )
    assert not target.exists()
    assert not target.with_name("foreign_trajectory_provenance.json").exists()


def test_trajectory_authority_rejects_artifact_mutation(tmp_path):
    paths, _cohort, trajectory = _bundle(tmp_path)
    paths["trajectory"].write_bytes(paths["trajectory"].read_bytes() + b"tamper")

    with pytest.raises(MaterializedTrajectoryError):
        load_verified_materialized_trajectory_authority(
            paths["trajectory"], expected_authority=trajectory.reference
        )


def test_trajectory_target_symlink_never_touches_victim(tmp_path):
    paths, cohort, _trajectory = _bundle(tmp_path)
    victim = tmp_path / "victim"
    victim.write_bytes(b"sentinel")
    target = paths["parquet"].parent / "linked_trajectory.parquet"
    target.symlink_to(victim)
    frame = pd.DataFrame(
        {
            "stay_id": [1],
            "charttime": [1.0],
            "concept": ["lact"],
            "value_num": [1.0],
            "value_str": ["1.0"],
        }
    )

    with pytest.raises(MaterializedTrajectoryError):
        publish_materialized_trajectory_authority(
            frame,
            target,
            bound_universe_path=paths["parquet"],
            bound_universe=cohort,
            requested_concepts=("lact",),
            materialized_concepts=("lact",),
            available_unobserved_concepts=(),
            unavailable_concepts=(),
            window=(0.0, 24.0),
            semantic_provenance={},
            producer_implementation_sha256=_implementation_sha(),
            producer_parameters={},
        )
    assert victim.read_bytes() == b"sentinel"


def test_trajectory_stage_is_exact_copy_and_rebinds_universe(tmp_path):
    paths, cohort, trajectory = _bundle(tmp_path)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    staged_cohort = stage_materialized_cohort_authority(
        paths["parquet"],
        run_dir / "cohort.parquet",
        expected_source_authority=cohort.reference,
        producer_implementation_sha256=_implementation_sha(),
    )
    assert staged_cohort is not None

    staged = stage_materialized_trajectory_authority(
        paths["trajectory"],
        run_dir / "cohort_trajectory.parquet",
        source_universe_path=paths["parquet"],
        target_universe_path=run_dir / "cohort.parquet",
        expected_source_authority=trajectory.reference,
        expected_target_universe_authority=staged_cohort.reference,
        producer_implementation_sha256=_implementation_sha(),
    )

    assert (run_dir / "cohort_trajectory.parquet").read_bytes() == paths[
        "trajectory"
    ].read_bytes()
    assert staged.authority.parent_trajectory_authority == trajectory.reference
    assert staged.authority.bound_universe_authority == staged_cohort.reference
    assert staged.authority.bound_universe_authority != cohort.reference


def test_development_sample_republishes_typed_trajectory_against_sampled_cohort(
    tmp_path,
):
    paths, cohort, trajectory = _bundle(tmp_path)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    staged_cohort_path = run_dir / "cohort_analysis.parquet"
    staged_cohort = stage_materialized_cohort_authority(
        paths["parquet"],
        staged_cohort_path,
        expected_source_authority=cohort.reference,
        producer_implementation_sha256=_implementation_sha(),
    )
    assert staged_cohort is not None
    staged_trajectory_path = run_dir / "cohort_trajectory.parquet"
    staged_trajectory = stage_materialized_trajectory_authority(
        paths["trajectory"],
        staged_trajectory_path,
        source_universe_path=paths["parquet"],
        target_universe_path=staged_cohort_path,
        expected_source_authority=trajectory.reference,
        expected_target_universe_authority=staged_cohort.reference,
        producer_implementation_sha256=_implementation_sha(),
    )

    sampled = materialize_development_execution_sample(
        run_dir=run_dir,
        target_rows=1,
        seed=17,
        declared_id_columns=("stay_id",),
        trajectory_binding=StagedTrajectoryBinding(
            path=staged_trajectory_path,
            sha256=staged_trajectory.authority.trajectory_sha256,
            size=staged_trajectory.authority.trajectory_size,
            authority_ref=staged_trajectory.reference,
        ),
    )

    assert sampled.cohort_authority_ref is not None
    assert sampled.trajectory_binding is not None
    assert sampled.trajectory_binding.authority_ref is not None
    assert sampled.trajectory_bound_cohort_authority_ref == sampled.cohort_authority_ref
    verified = load_verified_materialized_trajectory_authority(
        sampled.trajectory_binding.path,
        expected_authority=sampled.trajectory_binding.authority_ref,
        expected_universe_authority=sampled.cohort_authority_ref,
    )
    assert verified is not None
    sample_ids = set(pd.read_parquet(sampled.cohort_path)["stay_id"])
    trajectory_ids = set(pd.read_parquet(sampled.trajectory_binding.path)["stay_id"])
    assert trajectory_ids == sample_ids
    assert verified.authority.semantic_provenance["paper_authority"] is False
    assert (
        materialize_development_execution_sample(
            run_dir=run_dir,
            target_rows=1,
            seed=17,
            declared_id_columns=("stay_id",),
            trajectory_binding=StagedTrajectoryBinding(
                path=staged_trajectory_path,
                sha256=staged_trajectory.authority.trajectory_sha256,
                size=staged_trajectory.authority.trajectory_size,
                authority_ref=staged_trajectory.reference,
            ),
        )
        == sampled
    )


def test_trajectory_publication_root_swap_cannot_write_victim(tmp_path, monkeypatch):
    paths, cohort, _trajectory = _bundle(tmp_path)
    original_root = paths["parquet"].parent
    held_root = tmp_path / "held"
    victim_root = tmp_path / "victim-root"
    victim_root.mkdir()
    (victim_root / "sentinel").write_text("safe", encoding="utf-8")
    target = original_root / "swapped_trajectory.parquet"
    frame = pd.DataFrame(
        {
            "stay_id": [1],
            "charttime": [1.0],
            "concept": ["lact"],
            "value_num": [1.0],
            "value_str": ["1.0"],
        }
    )
    original = AnchoredDirectory.require_absent
    swapped = False

    def swap_then_check(directory, *names):
        nonlocal swapped
        if not swapped and directory.path == original_root:
            original_root.rename(held_root)
            original_root.symlink_to(victim_root, target_is_directory=True)
            swapped = True
        return original(directory, *names)

    monkeypatch.setattr(AnchoredDirectory, "require_absent", swap_then_check)
    with pytest.raises(MaterializedTrajectoryError):
        publish_materialized_trajectory_authority(
            frame,
            target,
            bound_universe_path=paths["parquet"],
            bound_universe=cohort,
            requested_concepts=("lact",),
            materialized_concepts=("lact",),
            available_unobserved_concepts=(),
            unavailable_concepts=(),
            window=(0.0, 24.0),
            semantic_provenance={},
            producer_implementation_sha256=_implementation_sha(),
            producer_parameters={},
        )
    assert {path.name for path in victim_root.iterdir()} == {"sentinel"}
    assert (victim_root / "sentinel").read_text(encoding="utf-8") == "safe"


@pytest.mark.parametrize(
    "mutate",
    [
        lambda payload: payload["window"].update({"end_hours": 25.0}),
        lambda payload: payload["semantic_provenance"].update(
            {"forged_scientific_claim": True}
        ),
        lambda payload: (
            payload["requested_concepts"].append("age"),
            payload["unavailable_concepts"].append("age"),
        ),
    ],
    ids=("window", "semantic-provenance", "concept-availability"),
)
def test_resigned_staged_trajectory_cannot_rewrite_parent_science(
    tmp_path,
    mutate,
):
    paths, cohort, trajectory = _bundle(tmp_path)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    staged_cohort = stage_materialized_cohort_authority(
        paths["parquet"],
        run_dir / "cohort.parquet",
        expected_source_authority=cohort.reference,
        producer_implementation_sha256=_implementation_sha(),
    )
    assert staged_cohort is not None
    staged_path = run_dir / "cohort_trajectory.parquet"
    stage_materialized_trajectory_authority(
        paths["trajectory"],
        staged_path,
        source_universe_path=paths["parquet"],
        target_universe_path=run_dir / "cohort.parquet",
        expected_source_authority=trajectory.reference,
        expected_target_universe_authority=staged_cohort.reference,
        producer_implementation_sha256=_implementation_sha(),
    )
    _resign_selected_trajectory_authority(staged_path, mutate)

    with pytest.raises(MaterializedTrajectoryError, match="exact deterministic"):
        load_verified_materialized_trajectory_authority(
            staged_path,
            expected_universe_authority=staged_cohort.reference,
            expected_parent_universe_authority=cohort.reference,
        )


def test_resigned_initial_trajectory_cannot_rewrite_producer_receipts(tmp_path):
    paths, cohort, _trajectory = _bundle(tmp_path)

    def mutate(payload):
        payload["producer_parameters"]["database"] = "eicu"
        payload["producer_parameters_sha256"] = canonical_parameters_sha256(
            payload["producer_parameters"]
        )

    _resign_selected_trajectory_authority(paths["trajectory"], mutate)

    with pytest.raises(MaterializedTrajectoryError, match="producer receipts"):
        load_verified_materialized_trajectory_authority(
            paths["trajectory"],
            expected_universe_authority=cohort.reference,
        )


def test_staged_trajectory_rejects_wrong_source_cohort_join(tmp_path):
    paths, cohort, trajectory = _bundle(tmp_path)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    staged_cohort = stage_materialized_cohort_authority(
        paths["parquet"],
        run_dir / "cohort.parquet",
        expected_source_authority=cohort.reference,
        producer_implementation_sha256=_implementation_sha(),
    )
    assert staged_cohort is not None
    staged_path = run_dir / "cohort_trajectory.parquet"
    staged = stage_materialized_trajectory_authority(
        paths["trajectory"],
        staged_path,
        source_universe_path=paths["parquet"],
        target_universe_path=run_dir / "cohort.parquet",
        expected_source_authority=trajectory.reference,
        expected_target_universe_authority=staged_cohort.reference,
        producer_implementation_sha256=_implementation_sha(),
    )
    wrong_source = MaterializedCohortAuthorityRef(
        file=f"cohort_authority.sha256-{'f' * 64}.json",
        sha256="f" * 64,
        size=1,
    )

    with pytest.raises(MaterializedTrajectoryError, match="different source universe"):
        load_verified_materialized_trajectory_authority(
            staged_path,
            expected_authority=staged.reference,
            expected_universe_authority=staged_cohort.reference,
            expected_parent_universe_authority=wrong_source,
        )
