"""Materialized cohorts preserve typed export metadata without name guessing."""

from __future__ import annotations

import json
from pathlib import Path
import ast
from dataclasses import replace
from types import SimpleNamespace

import pandas as pd
import pytest

from easyicu.concept.metadata_projection import (
    ColumnProjectionSpec,
    ConceptColumnRole,
    NumericBounds,
    project_concept_column_metadata,
)
from easyicu.concept.metadata_sidecar import (
    EXPORT_PHYSICAL_SCOPE,
    MATERIALIZED_COHORT_SCOPE,
    ColumnMetadataBinding,
    ColumnMetadataFileBinding,
    ColumnMetadataSidecar,
    SidecarRef,
    TimeCoordinate,
    binding_payload_sha256,
    read_content_addressed_sidecar,
    write_content_addressed_sidecar,
)
from easyicu.resources import load_dictionary
from easyicu.research_agent.cohort import materializer as cohort_materializer
from easyicu.research_agent.authority.filesystem import AnchoredDirectory
from easyicu.research_agent.authority.analysis_cohort import (
    bind_execution_cohort_authority,
)
from easyicu.research_agent.execution.development_sample import (
    materialize_development_execution_sample,
)
from easyicu.research_agent.research_context.builder import (
    build_research_context,
    build_retrieved_research_context,
)
from easyicu.research_agent.cohort.schema import (
    CohortDefinition,
    ConceptPredicate,
    TimeWindow,
    materialize_locked_analysis_cohort,
)
from easyicu.research_agent.intake import export_package as intake
from easyicu.research_agent.intake import materialized_metadata as materialized
from easyicu.research_agent.intake.materialized_metadata import (
    MaterializedMetadataError,
    implementation_bundle_sha256,
    load_verified_materialized_cohort_authority,
    publish_ordered_subset_materialized_cohort,
    stage_materialized_cohort_authority,
)
from easyicu.research_agent.authority.evidence_store import (
    EvidenceStore,
    sha256_of_file,
)
from easyicu.research_agent.authority.typed_binding import (
    _write_resolved_inputs_manifest,
)
from easyicu.research_agent.providers.mocks import MockLLMClient
from easyicu.research_agent import pipeline as pipeline_module
from easyicu.research_agent.pipeline import ResearchAgentPipeline
from easyicu.research_agent.authority.run_input import (
    RUN_INPUT_CAPSULE_SCHEMA_VERSION_V2,
    RunInputIdentityError,
    load_verified_run_input_capsule,
    seal_run_input_capsule,
)
from easyicu.research_agent.research_context.typed import (
    CanonicalColumnBinding,
    RESEARCH_CONTEXT_V2_SCHEMA_VERSION,
    ResearchContextV2,
    ResearchContextV3,
    binding_preserves_analysis_range,
    canonical_column_binding,
    descriptor_physical_updates,
    effective_analysis_plausibility_range,
    materialized_input_prompt_attachment,
    migrate_research_context_v2,
    parse_research_context,
    resolved_raw_input_contracts,
)


def _binding(
    concept: str, column: str, role: ConceptColumnRole
) -> ColumnMetadataBinding:
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


def _typed_export(
    root: Path,
    *,
    roles: dict[str, ConceptColumnRole] | None = None,
    labs: pd.DataFrame | None = None,
    outcomes: pd.DataFrame | None = None,
    binding_overrides: dict[str, ColumnMetadataBinding] | None = None,
) -> Path:
    root.mkdir()
    roles = roles or {}
    binding_overrides = binding_overrides or {}
    labs = (
        labs
        if labs is not None
        else pd.DataFrame(
            {
                "stay_id": [1, 1, 2],
                "charttime": [1.0, 2.0, 1.0],
                "age": [50, 50, 60],
                "lact": [1.0, 2.0, 3.0],
                "mech_vent": [False, True, False],
            }
        )
    )
    outcomes = (
        outcomes
        if outcomes is not None
        else pd.DataFrame({"stay_id": [1, 2], "death": [False, True]})
    )
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
            "age": binding_overrides.get("age")
            or _binding("age", "age", roles.get("age", ConceptColumnRole.VALUE)),
            "lact": binding_overrides.get("lact")
            or _binding("lact", "lact", roles.get("lact", ConceptColumnRole.VALUE)),
            "mech_vent": binding_overrides.get("mech_vent")
            or _binding(
                "mech_vent",
                "mech_vent",
                roles.get("mech_vent", ConceptColumnRole.EVENT_STATUS),
            ),
        },
    )
    outcome_binding = ColumnMetadataFileBinding(
        relative_path="outcomes.parquet",
        module="outcomes",
        identity_column="stay_id",
        time_coordinates=(),
        columns={
            "death": binding_overrides.get("death")
            or _binding(
                "death", "death", roles.get("death", ConceptColumnRole.EVENT_STATUS)
            )
        },
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
                        "labs": ["age", "lact", "mech_vent"],
                        "outcomes": ["death"],
                    },
                },
                "files": [
                    {
                        "file": "labs.parquet",
                        "module": "labs",
                        "concepts": 3,
                        "concept_ids": ["age", "lact", "mech_vent"],
                        "rows": len(labs),
                        "column_metadata_columns": list(lab_binding.columns),
                    },
                    {
                        "file": "outcomes.parquet",
                        "module": "outcomes",
                        "concepts": 1,
                        "concept_ids": ["death"],
                        "rows": len(outcomes),
                        "column_metadata_columns": list(outcome_binding.columns),
                    },
                ],
                "feature_definitions": {"included": False},
                "column_metadata": reference.to_dict(),
            }
        ),
        encoding="utf-8",
    )
    return root


def _resign_selected_authority(
    cohort_path: Path,
    authority,
    *,
    sidecar: ColumnMetadataSidecar,
) -> None:
    authority_ref = materialized._write_authority(cohort_path.parent, authority)
    sidecar_ref = authority.column_metadata
    file_binding = sidecar.files[0]
    provenance_path = materialized.materialized_provenance_path(cohort_path)
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["column_metadata"] = materialized._descriptor(
        authority=authority_ref,
        sidecar=sidecar_ref,
        file_binding=file_binding,
    )
    materialized._atomic_write_json(provenance_path, provenance)


def _build_v2_context(
    tmp_path: Path,
    *,
    binding_overrides: dict[str, ColumnMetadataBinding] | None = None,
    id_columns: tuple[str, ...] = ("stay_id",),
) -> ResearchContextV2:
    source = _typed_export(
        tmp_path / "export",
        binding_overrides=binding_overrides,
    )
    paths = cohort_materializer.materialize_to_parquet(
        tmp_path / "materialized",
        data_path=source,
        database="miiv",
        static_concepts=("age",),
        feature_concepts=("lact", "mech_vent"),
        outcome_concepts=("death",),
    )
    context = build_research_context(
        research_question="Describe age while retaining the declared effect model.",
        cohort=paths["parquet"],
        cohort_name="typed_context",
        database="miiv",
        target_outcome="death",
        primary_exposure="lact_max",
        id_columns=id_columns,
        outcome_columns=("death",),
    )
    assert isinstance(context, ResearchContextV2)
    return context


def _as_archived_v2_payload(context: ResearchContextV3) -> dict[str, object]:
    payload = context.model_dump(mode="json")
    payload["schema_version"] = RESEARCH_CONTEXT_V2_SCHEMA_VERSION
    for variable in payload["variables"]:
        if variable.get("analysis_window_role") == "outer_observation_window":
            variable["analysis_window_role"] = None
    return payload


def test_archived_v2_context_reads_exactly_and_upgrades_explicitly(
    tmp_path: Path,
) -> None:
    current = _build_v2_context(tmp_path)
    assert isinstance(current, ResearchContextV3)
    archived_payload = _as_archived_v2_payload(current)
    original_payload = json.loads(json.dumps(archived_payload))

    archived = parse_research_context(archived_payload)

    assert type(archived) is ResearchContextV2
    assert archived.model_dump(mode="json") == original_payload
    assert archived_payload == original_payload

    upgraded = migrate_research_context_v2(archived)
    assert type(upgraded) is ResearchContextV3
    assert upgraded.schema_version == "easyicu.research_context/3"
    assert any(
        variable.analysis_window_role == "outer_observation_window"
        for variable in upgraded.variables
    )


def test_v3_rejects_a_missing_or_conflicting_bound_window_role(
    tmp_path: Path,
) -> None:
    current = _build_v2_context(tmp_path)
    assert isinstance(current, ResearchContextV3)
    payload = current.model_dump(mode="python")
    variable = next(
        item
        for item in payload["variables"]
        if item.get("analysis_window_role") == "outer_observation_window"
    )
    variable["analysis_window_role"] = None

    with pytest.raises(ValueError, match="analysis_window_role"):
        ResearchContextV3.model_validate(payload)

    archived_payload = _as_archived_v2_payload(current)
    archived_variable = next(
        item
        for item in archived_payload["variables"]
        if item.get("analysis_window") is not None
    )
    archived_variable["analysis_window_role"] = "exposure_definition"
    archived = ResearchContextV2.model_validate(archived_payload)
    with pytest.raises(ValueError, match="conflicts with typed column binding"):
        migrate_research_context_v2(archived)


def test_resolved_raw_input_contracts_bind_domain_and_range_policy(
    tmp_path: Path,
) -> None:
    context = _build_v2_context(tmp_path)

    contracts = resolved_raw_input_contracts(
        context,
        ["age", "death"],
    )

    assert contracts is not None
    assert set(contracts["contracts"]) == {"age", "death"}
    age = contracts["contracts"]["age"]
    assert age["analysis_plausibility_range"] == {
        "minimum": 0.0,
        "maximum": 120.0,
    }
    assert age["plausibility_policy"] == {
        "range_policy": "flag_only",
        "out_of_range_action": "retain_and_flag",
    }
    assert contracts["contracts"]["death"]["allowed_values"] == [0, 1]

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    context_path = run_dir / "research_context.json"
    context_path.write_text(context.model_dump_json(indent=2), encoding="utf-8")
    manifest_path = _write_resolved_inputs_manifest(
        run_dir=run_dir,
        step_id="01_define_cohort",
        planner_declared_inputs=["age", "death"],
        bindings={},
        context_path=context_path,
        raw_input_contracts=contracts,
    )
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["raw_input_contracts"] == contracts


def test_resolved_raw_input_contracts_bind_materialized_identity_column(
    tmp_path: Path,
) -> None:
    context = _build_v2_context(tmp_path)

    contracts = resolved_raw_input_contracts(context, ["stay_id"])

    identity = contracts["contracts"]["stay_id"]
    assert identity == {
        "column": "stay_id",
        "dtype": "int64",
        "physical_role": "identity",
        "representation_transform": "row_identity",
        "source_database_actual": "miiv",
        "authority_kind": "materialized_cohort_identity",
        "row_identity_sha256": (
            context.materialized_inputs.cohort.row_identity_sha256
        ),
    }


def test_resolved_raw_input_contracts_bind_sealed_observed_levels(
    tmp_path: Path,
) -> None:
    context = _build_v2_context(tmp_path)
    variables = [
        (
            variable.model_copy(
                update={
                    "observed_domain": {
                        "n_unique": 2,
                        "is_constant": False,
                        "is_binary": False,
                        "levels": ["Female", "Male"],
                    }
                }
            )
            if variable.name == "age"
            else variable
        )
        for variable in context.variables
    ]
    context = type(context).model_validate(
        context.model_dump(mode="python") | {"variables": variables}
    )

    contracts = resolved_raw_input_contracts(context, ["age"])

    assert contracts is not None
    assert contracts["contracts"]["age"]["allowed_values"] == [
        "Female",
        "Male",
    ]
    assert contracts["contracts"]["age"]["allowed_values_basis"] == (
        "sealed_research_context_observed_domain"
    )


def test_resolved_raw_input_contracts_reject_unbounded_or_malformed_levels(
    tmp_path: Path,
) -> None:
    context = _build_v2_context(tmp_path)

    def with_domain(domain: dict[str, object]) -> ResearchContextV2:
        variables = [
            (
                variable.model_copy(update={"observed_domain": domain})
                if variable.name == "age"
                else variable
            )
            for variable in context.variables
        ]
        return type(context).model_validate(
            context.model_dump(mode="python") | {"variables": variables}
        )

    for domain in (
        {
            "n_unique": 2,
            "is_constant": False,
            "is_binary": False,
            "levels": ["Female", "Female"],
        },
        {
            "n_unique": 9,
            "is_constant": False,
            "is_binary": False,
            "levels": [f"level_{index}" for index in range(9)],
        },
    ):
        contracts = resolved_raw_input_contracts(with_domain(domain), ["age"])
        assert contracts is not None
        assert "allowed_values" not in contracts["contracts"]["age"]


def test_materialized_cohort_publishes_exact_typed_sidecar(tmp_path: Path) -> None:
    source = _typed_export(tmp_path / "export")
    output = tmp_path / "materialized"

    paths = cohort_materializer.materialize_to_parquet(
        output,
        stem="universe",
        data_path=source,
        database="miiv",
        static_concepts=("age",),
        feature_concepts=("lact", "mech_vent"),
        outcome_concepts=("death",),
    )

    provenance = json.loads(paths["provenance"].read_text(encoding="utf-8"))
    assert provenance["cohort_file_sha256"] == sha256_of_file(paths["parquet"])
    assert provenance["cohort_file_size"] == paths["parquet"].stat().st_size
    descriptor = provenance["column_metadata"]
    reference = SidecarRef.from_dict(descriptor["sidecar"])
    sidecar = read_content_addressed_sidecar(
        paths["column_metadata"],
        expected_sha256=reference.sha256,
        expected_size=reference.size,
    )
    assert sidecar.scope == MATERIALIZED_COHORT_SCOPE
    assert paths["column_metadata"].name == reference.file
    assert len(sidecar.files) == 1
    file_binding = sidecar.files[0]
    assert file_binding.relative_path == "universe.parquet"
    assert file_binding.identity_column == "stay_id"
    cohort = pd.read_parquet(paths["parquet"])
    assert set(file_binding.columns) == set(cohort.columns) - {"stay_id"}
    assert file_binding.columns["age"].metadata.role is ConceptColumnRole.VALUE
    assert (
        file_binding.columns["lact_max"].metadata.role
        is ConceptColumnRole.NUMERIC_AGGREGATE
    )
    assert file_binding.columns["lact_n"].metadata.role is ConceptColumnRole.COUNT
    assert (
        file_binding.columns["lact_measured"].metadata.role
        is ConceptColumnRole.MEASUREMENT_STATUS
    )
    assert (
        file_binding.columns["lact_first_time"].metadata.role
        is ConceptColumnRole.FIRST_OBSERVATION_TIME
    )
    assert (
        file_binding.columns["mech_vent_max"].metadata.role
        is ConceptColumnRole.EVENT_STATUS
    )
    assert (
        file_binding.columns["mech_vent_mean"].metadata.role
        is ConceptColumnRole.EVENT_FRACTION
    )
    assert file_binding.columns["death"].metadata.role is ConceptColumnRole.EVENT_STATUS
    assert file_binding.columns["lact_max"].derivation_window.to_dict() == {
        "origin": "icu_admission",
        "start_hours": 0.0,
        "end_hours": 24.0,
    }
    verified = load_verified_materialized_cohort_authority(paths["parquet"])
    assert verified is not None
    assert verified.reference.to_dict() == descriptor["authority"]
    assert verified.authority.cohort_rows == len(cohort)
    assert verified.authority.cohort_columns == tuple(cohort.columns)
    assert verified.authority.cohort_sha256
    assert verified.authority.cohort_schema_sha256
    assert verified.authority.row_identity_sha256
    assert verified.authority.source_column_metadata_sha256
    assert verified.authority.source_export_authority_sha256
    assert set(
        item.output_column for item in verified.authority.output_derivations
    ) == (set(cohort.columns) - {"stay_id"})


def test_in_memory_materialization_does_not_publish_orphan_sidecar(
    tmp_path: Path,
) -> None:
    source = _typed_export(tmp_path / "export")

    _cohort, provenance = cohort_materializer.materialize_cohort(
        data_path=source,
        database="miiv",
        static_concepts=("age",),
        feature_concepts=("lact",),
        outcome_concepts=("death",),
    )

    assert "column_metadata" not in provenance
    assert not list(source.glob("cohort_column_metadata.sha256-*.json"))


def test_materialized_authority_rejects_artifact_mutation(tmp_path: Path) -> None:
    source = _typed_export(tmp_path / "export")
    paths = cohort_materializer.materialize_to_parquet(
        tmp_path / "materialized",
        stem="universe",
        data_path=source,
        database="miiv",
        static_concepts=("age",),
        feature_concepts=("lact",),
        outcome_concepts=("death",),
    )
    frame = pd.read_parquet(paths["parquet"])
    frame.loc[0, "lact_max"] = 999.0
    frame.to_parquet(paths["parquet"], index=False)

    with pytest.raises(MaterializedMetadataError, match="no longer matches"):
        load_verified_materialized_cohort_authority(paths["parquet"])


def test_materialized_authority_rejects_sidecar_mutation(tmp_path: Path) -> None:
    source = _typed_export(tmp_path / "export")
    paths = cohort_materializer.materialize_to_parquet(
        tmp_path / "materialized",
        stem="universe",
        data_path=source,
        database="miiv",
        static_concepts=("age",),
        feature_concepts=("lact",),
        outcome_concepts=("death",),
    )
    paths["column_metadata"].write_text("{}", encoding="utf-8")

    with pytest.raises(MaterializedMetadataError, match="size/type mismatch"):
        load_verified_materialized_cohort_authority(paths["parquet"])


def test_required_authority_cannot_downgrade_when_descriptor_is_removed(
    tmp_path: Path,
) -> None:
    source = _typed_export(tmp_path / "export")
    paths = cohort_materializer.materialize_to_parquet(
        tmp_path / "materialized",
        data_path=source,
        database="miiv",
        static_concepts=("age",),
        feature_concepts=("lact",),
        outcome_concepts=("death",),
    )
    provenance = json.loads(paths["provenance"].read_text(encoding="utf-8"))
    provenance.pop("column_metadata")
    paths["provenance"].write_text(json.dumps(provenance), encoding="utf-8")

    with pytest.raises(MaterializedMetadataError, match="descriptor is missing"):
        load_verified_materialized_cohort_authority(paths["parquet"])


def test_staging_is_exact_copy_with_parent_bound_authority(tmp_path: Path) -> None:
    source = _typed_export(tmp_path / "export")
    paths = cohort_materializer.materialize_to_parquet(
        tmp_path / "materialized",
        stem="universe",
        data_path=source,
        database="miiv",
        static_concepts=("age",),
        feature_concepts=("lact",),
        outcome_concepts=("death",),
    )
    parent = load_verified_materialized_cohort_authority(paths["parquet"])
    assert parent is not None
    target = tmp_path / "run" / "cohort.parquet"

    staged = stage_materialized_cohort_authority(
        paths["parquet"],
        target,
        producer_implementation_sha256=implementation_bundle_sha256(
            (Path(cohort_materializer.__file__),)
        ),
    )

    assert staged is not None
    assert target.read_bytes() == paths["parquet"].read_bytes()
    assert staged.authority.parent_authority_sha256 == parent.reference.sha256
    assert staged.authority.cohort_file == "cohort.parquet"
    assert staged.sidecar.files[0].relative_path == "cohort.parquet"
    assert {item.transform_id for item in staged.authority.output_derivations} == {
        "identity_stage_copy"
    }


def test_staging_rejects_same_source_and_target_without_mutation(
    tmp_path: Path,
) -> None:
    source = _typed_export(tmp_path / "export")
    paths = cohort_materializer.materialize_to_parquet(
        tmp_path / "materialized",
        data_path=source,
        database="miiv",
        static_concepts=("age",),
        feature_concepts=("lact",),
        outcome_concepts=("death",),
    )
    artifact_before = paths["parquet"].read_bytes()
    selector_before = paths["provenance"].read_bytes()

    with pytest.raises(MaterializedMetadataError, match="different artifacts"):
        stage_materialized_cohort_authority(
            paths["parquet"],
            paths["parquet"],
            producer_implementation_sha256="a" * 64,
        )

    assert paths["parquet"].read_bytes() == artifact_before
    assert paths["provenance"].read_bytes() == selector_before
    assert load_verified_materialized_cohort_authority(paths["parquet"]) is not None


def test_staging_directory_swap_cannot_redirect_writes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _typed_export(tmp_path / "export")
    paths = cohort_materializer.materialize_to_parquet(
        tmp_path / "materialized",
        data_path=source,
        database="miiv",
        static_concepts=("age",),
        feature_concepts=("lact",),
        outcome_concepts=("death",),
    )
    run_root = tmp_path / "run"
    victim = tmp_path / "victim"
    victim.mkdir()
    sentinel = victim / "sentinel.txt"
    sentinel.write_text("unchanged", encoding="utf-8")
    displaced = tmp_path / "displaced_run"
    original = AnchoredDirectory.require_absent
    swapped = False

    def swap_after_anchor(self: AnchoredDirectory, *names: str) -> None:
        nonlocal swapped
        if not swapped and self.path == run_root.absolute():
            run_root.rename(displaced)
            run_root.symlink_to(victim, target_is_directory=True)
            swapped = True
        original(self, *names)

    monkeypatch.setattr(AnchoredDirectory, "require_absent", swap_after_anchor)

    with pytest.raises(
        MaterializedMetadataError, match="cannot stage materialized cohort"
    ):
        stage_materialized_cohort_authority(
            paths["parquet"],
            run_root / "cohort.parquet",
            producer_implementation_sha256="a" * 64,
        )

    assert swapped is True
    assert sentinel.read_text(encoding="utf-8") == "unchanged"
    assert {item.name for item in victim.iterdir()} == {"sentinel.txt"}


def test_staged_authority_requires_parent_snapshot_closure(tmp_path: Path) -> None:
    source = _typed_export(tmp_path / "export")
    paths = cohort_materializer.materialize_to_parquet(
        tmp_path / "materialized",
        data_path=source,
        database="miiv",
        static_concepts=("age",),
        feature_concepts=("lact",),
        outcome_concepts=("death",),
    )
    parent = load_verified_materialized_cohort_authority(paths["parquet"])
    assert parent is not None
    target = tmp_path / "run" / "cohort.parquet"
    staged = stage_materialized_cohort_authority(
        paths["parquet"],
        target,
        producer_implementation_sha256="a" * 64,
    )
    assert staged is not None
    (target.parent / parent.reference.file).unlink()

    with pytest.raises(MaterializedMetadataError, match="parent authority is missing"):
        load_verified_materialized_cohort_authority(target)


def test_staged_authority_rejects_resigned_metadata_projection(tmp_path: Path) -> None:
    source = _typed_export(tmp_path / "export")
    paths = cohort_materializer.materialize_to_parquet(
        tmp_path / "materialized",
        data_path=source,
        database="miiv",
        static_concepts=("age",),
        feature_concepts=("lact",),
        outcome_concepts=("death",),
    )
    target = tmp_path / "run" / "cohort.parquet"
    staged = stage_materialized_cohort_authority(
        paths["parquet"],
        target,
        producer_implementation_sha256="a" * 64,
    )
    assert staged is not None
    forged_binding = replace(staged.sidecar.files[0], module="forged_stage")
    forged_sidecar = replace(staged.sidecar, files=(forged_binding,))
    forged_ref = write_content_addressed_sidecar(
        target.parent,
        forged_sidecar,
        stem="cohort_column_metadata",
    )
    forged_authority = replace(
        staged.authority,
        column_metadata=forged_ref,
        file_metadata_payload_sha256=forged_binding.metadata_payload_sha256,
    )
    _resign_selected_authority(
        target,
        forged_authority,
        sidecar=forged_sidecar,
    )

    with pytest.raises(MaterializedMetadataError, match="parent projection"):
        load_verified_materialized_cohort_authority(target)


def test_pipeline_stages_typed_cohort_without_parquet_reserialization(
    tmp_path: Path,
) -> None:
    source = _typed_export(tmp_path / "export")
    paths = cohort_materializer.materialize_to_parquet(
        tmp_path / "materialized",
        data_path=source,
        database="miiv",
        static_concepts=("age",),
        feature_concepts=("lact",),
        outcome_concepts=("death",),
    )
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    pipeline = object.__new__(ResearchAgentPipeline)

    target = pipeline._materialise_cohort(paths["parquet"], run_dir)

    assert target.read_bytes() == paths["parquet"].read_bytes()
    assert load_verified_materialized_cohort_authority(target) is not None


def test_pipeline_does_not_fallback_when_typed_authority_is_tampered(
    tmp_path: Path,
) -> None:
    source = _typed_export(tmp_path / "export")
    paths = cohort_materializer.materialize_to_parquet(
        tmp_path / "materialized",
        data_path=source,
        database="miiv",
        static_concepts=("age",),
        feature_concepts=("lact",),
        outcome_concepts=("death",),
    )
    provenance = json.loads(paths["provenance"].read_text(encoding="utf-8"))
    provenance["column_metadata"]["authority"]["sha256"] = "0" * 64
    paths["provenance"].write_text(json.dumps(provenance), encoding="utf-8")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    pipeline = object.__new__(ResearchAgentPipeline)

    with pytest.raises(MaterializedMetadataError):
        pipeline._materialise_cohort(paths["parquet"], run_dir)

    assert not (run_dir / "cohort.parquet").exists()


def test_pipeline_rejects_database_label_mismatching_typed_authority(
    tmp_path: Path,
) -> None:
    source = _typed_export(tmp_path / "export")
    paths = cohort_materializer.materialize_to_parquet(
        tmp_path / "materialized",
        data_path=source,
        database="miiv",
        static_concepts=("age",),
        feature_concepts=("lact",),
        outcome_concepts=("death",),
    )
    workdir = tmp_path / "agent"
    pipeline = ResearchAgentPipeline(
        workdir=workdir,
        llm=MockLLMClient(),
        enable_literature=False,
        enable_memory=False,
        enable_latex=False,
    )

    with pytest.raises(MaterializedMetadataError, match="database does not match"):
        pipeline.run(
            question="Is lactate associated with hospital mortality?",
            cohort=paths["parquet"],
            cohort_name="typed_db_mismatch",
            database="eicu",
            target_outcome="death",
            primary_exposure="lact_max",
            stop_after_analysis=True,
        )

    assert not list(workdir.glob("run_*"))


def test_pipeline_rejects_typed_authority_in_historical_naive_ablation(
    tmp_path: Path,
) -> None:
    source = _typed_export(tmp_path / "export")
    paths = cohort_materializer.materialize_to_parquet(
        tmp_path / "materialized",
        data_path=source,
        database="miiv",
        static_concepts=("age",),
        feature_concepts=("lact",),
        outcome_concepts=("death",),
    )
    workdir = tmp_path / "agent"
    pipeline = ResearchAgentPipeline(
        workdir=workdir,
        llm=MockLLMClient(),
        disable_icu_context=True,
        enable_literature=False,
        enable_memory=False,
        enable_latex=False,
    )

    with pytest.raises(
        MaterializedMetadataError,
        match="require ICU-aware ResearchContext v2",
    ):
        pipeline.run(
            question="Is the declared exposure associated with the outcome?",
            cohort=paths["parquet"],
            cohort_name="typed_naive_boundary",
            database="miiv",
            target_outcome="death",
            primary_exposure="lact_max",
            stop_after_analysis=True,
        )

    assert not list(workdir.glob("run_*"))


def test_typed_pipeline_normalizes_database_alias_before_context_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _typed_export(tmp_path / "export")
    paths = cohort_materializer.materialize_to_parquet(
        tmp_path / "materialized",
        data_path=source,
        database="miiv",
        static_concepts=("age",),
        feature_concepts=("lact",),
        outcome_concepts=("death",),
    )
    workdir = tmp_path / "agent"
    pipeline = ResearchAgentPipeline(
        workdir=workdir,
        llm=MockLLMClient(),
        enable_literature=False,
        enable_memory=False,
        enable_latex=False,
    )

    class _IdentityProbeComplete(Exception):
        pass

    observed: dict[str, str] = {}

    def capture_scientific_identity(**kwargs):
        observed["database"] = kwargs["database"]
        raise _IdentityProbeComplete

    monkeypatch.setattr(
        pipeline_module,
        "build_scientific_identity",
        capture_scientific_identity,
    )
    with pytest.raises(_IdentityProbeComplete):
        pipeline.run(
            question="Summarize the typed cohort.",
            cohort=paths["parquet"],
            cohort_name="typed_database_alias",
            database="mimiciv",
            target_outcome="death",
            primary_exposure="lact_max",
        )

    assert observed == {"database": "miiv"}
    assert not list(workdir.glob("run_*"))


def test_typed_context_builder_normalizes_database_alias(
    tmp_path: Path,
) -> None:
    source = _typed_export(tmp_path / "export")
    paths = cohort_materializer.materialize_to_parquet(
        tmp_path / "materialized",
        data_path=source,
        database="miiv",
        static_concepts=("age",),
        feature_concepts=("lact",),
        outcome_concepts=("death",),
    )

    context = build_research_context(
        research_question="Summarize the typed cohort.",
        cohort=paths["parquet"],
        cohort_name="typed_database_alias",
        database="mimiciv",
        target_outcome="death",
        primary_exposure="lact_max",
    )

    assert isinstance(context, ResearchContextV2)
    assert context.cohort.database == "miiv"
    assert context.materialized_inputs.cohort.source_database == "miiv"


def test_typed_run_input_capsule_binds_exact_staged_authority(
    tmp_path: Path,
) -> None:
    source = _typed_export(tmp_path / "export")
    paths = cohort_materializer.materialize_to_parquet(
        tmp_path / "materialized",
        data_path=source,
        database="miiv",
        static_concepts=("age",),
        feature_concepts=("lact",),
        outcome_concepts=("death",),
    )
    source_authority = load_verified_materialized_cohort_authority(paths["parquet"])
    assert source_authority is not None
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    cohort_path = run_dir / "cohort.parquet"
    staged = stage_materialized_cohort_authority(
        paths["parquet"],
        cohort_path,
        producer_implementation_sha256=implementation_bundle_sha256(
            (Path(cohort_materializer.__file__),)
        ),
    )
    assert staged is not None
    context = build_research_context(
        research_question="Is lactate associated with hospital mortality?",
        cohort=cohort_path,
        cohort_name="typed_capsule",
        database="miiv",
        target_outcome="death",
        primary_exposure="lact_max",
        id_columns=("stay_id",),
        outcome_columns=("death",),
    )
    context_path = run_dir / "research_context.json"
    context_path.write_text(context.model_dump_json(indent=2), encoding="utf-8")
    evidence = EvidenceStore(run_dir)
    evidence.register_file(
        kind="log",
        description="Frozen typed research context.",
        source_path=context_path,
        evidence_id="research_context",
        producer="pipeline",
        generation_mode="system",
    )
    scientific_identity = {
        "materialized_cohort_authority_ref": source_authority.reference.to_dict(),
        "question": "Is lactate associated with hospital mortality?",
    }
    capsule = seal_run_input_capsule(
        run_dir=run_dir,
        evidence=evidence,
        scientific_identity=scientific_identity,
        initial_environment={"llm_signature": "test"},
        context_path=context_path,
        cohort_path=cohort_path,
        experiment_spec_path=None,
    )
    assert capsule.schema_version == RUN_INPUT_CAPSULE_SCHEMA_VERSION_V2
    assert capsule.materialized_cohort_authority_ref == staged.reference.to_dict()
    assert (
        load_verified_run_input_capsule(
            run_dir=run_dir,
            scientific_identity=scientific_identity,
        ).capsule
        == capsule
    )

    selector_path = materialized.materialized_provenance_path(cohort_path)
    selector = json.loads(selector_path.read_text(encoding="utf-8"))
    selector["column_metadata"]["authority"] = {
        "file": f"cohort_authority.sha256-{'f' * 64}.json",
        "sha256": "f" * 64,
        "size": 2,
    }
    selector_path.write_text(json.dumps(selector), encoding="utf-8")

    with pytest.raises(
        RunInputIdentityError,
        match="staged cohort authority is missing or changed",
    ):
        load_verified_run_input_capsule(
            run_dir=run_dir,
            scientific_identity=scientific_identity,
        )


def test_verified_materialized_cohort_builds_scopable_v2_context(
    tmp_path: Path,
) -> None:
    source = _typed_export(tmp_path / "export")
    paths = cohort_materializer.materialize_to_parquet(
        tmp_path / "materialized",
        data_path=source,
        database="miiv",
        static_concepts=("age",),
        feature_concepts=("lact", "mech_vent"),
        outcome_concepts=("death",),
    )
    verified = load_verified_materialized_cohort_authority(paths["parquet"])
    assert verified is not None
    context = build_research_context(
        research_question="Describe age while retaining the declared effect model.",
        cohort=paths["parquet"],
        cohort_name="typed_context",
        database="miiv",
        target_outcome="death",
        primary_exposure="lact_max",
        id_columns=("stay_id",),
        outcome_columns=("death",),
    )
    assert isinstance(context, ResearchContextV2)
    typed = context.materialized_inputs.cohort
    assert typed.authority_ref == verified.reference.to_dict()
    assert typed.row_identity_sha256 == verified.authority.row_identity_sha256
    assert typed.source_database == "miiv"
    assert typed.projection_scope == "full"

    lact = typed.column_bindings["lact_max"]
    metadata = lact.binding["metadata"]
    assert metadata["role"] == "numeric_aggregate"
    assert metadata["source_database"] == "miiv"
    assert "eicu" in metadata["available_databases"]
    assert metadata["extraction_bounds"] == {"minimum": 0.0, "maximum": 50.0}
    assert metadata["analysis_plausibility_range"] is None
    assert lact.analysis_plausibility_range == {
        "minimum": 0.0,
        "maximum": 30.0,
    }
    assert context.variable("lact_max").valid_range == [0.0, 30.0]
    assert context.variable("lact_n").unit is None
    assert context.variable("lact_n").valid_range is None

    scoped = build_retrieved_research_context(context, query="age", top_k=1)
    assert isinstance(scoped, ResearchContextV2)
    assert scoped.materialized_inputs.cohort.projection_scope == "scoped"
    selected = {item.name for item in scoped.variables}
    assert {"stay_id", "age", "death", "lact_max"}.issubset(selected)
    scoped_bindings = scoped.materialized_inputs.cohort.column_bindings
    assert "mech_vent" not in scoped_bindings
    assert "mech_vent_n" not in scoped_bindings
    assert {"lact_max", "lact_n", "lact_measured"}.issubset(scoped_bindings)
    assert context.materialized_inputs.cohort.projection_scope == "full"


def test_v2_scoped_context_cannot_drop_selected_typed_bindings(
    tmp_path: Path,
) -> None:
    context = _build_v2_context(tmp_path)
    payload = context.model_dump(mode="python")
    cohort = payload["materialized_inputs"]["cohort"]
    cohort["projection_scope"] = "scoped"
    cohort["column_bindings"] = {}
    cohort["column_binding_payload_sha256"] = binding_payload_sha256({})

    with pytest.raises(ValueError, match="lack typed cohort bindings"):
        type(context).model_validate(payload)


def test_v2_rejects_coerced_numeric_authority_fields(tmp_path: Path) -> None:
    context = _build_v2_context(tmp_path)
    payload = context.model_dump(mode="python")
    payload["materialized_inputs"]["cohort"]["cohort_rows"] = "2"

    with pytest.raises(ValueError):
        type(context).model_validate(payload)

    payload = context.model_dump(mode="python")
    payload["materialized_inputs"]["cohort"]["cohort_size"] = True
    with pytest.raises(ValueError):
        type(context).model_validate(payload)


def test_v2_prompt_revalidates_nested_authority_and_is_bounded(
    tmp_path: Path,
) -> None:
    context = _build_v2_context(tmp_path)
    attachment = materialized_input_prompt_attachment(context)
    assert len(attachment.encode("utf-8")) <= 4 * 1024
    payload = json.loads(attachment.split("\n", 1)[1])
    assert payload["schema_version"] == "easyicu.materialized_input_prompt_facts/2"
    columns = payload["cohort"]["column_bindings"]
    assert any(item["column"] == "lact_max" for item in columns)
    assert all(
        key not in payload
        for key in ("primary_exposure", "target_outcome", "method", "estimand")
    )
    context.materialized_inputs.cohort.column_bindings.clear()
    with pytest.raises(ValueError):
        materialized_input_prompt_attachment(context)


def test_typed_allowed_values_do_not_promote_event_status_to_ordinal(
    tmp_path: Path,
) -> None:
    context = _build_v2_context(tmp_path)

    assert context.variable("death").is_ordinal is False
    assert context.variable("death").ordinal_levels is None
    assert context.materialized_inputs.cohort.column_bindings["death"].binding[
        "metadata"
    ]["allowed_values"] == [0, 1]


@pytest.mark.parametrize(
    "role",
    [
        ConceptColumnRole.FIRST_OBSERVATION_TIME,
        ConceptColumnRole.LAST_OBSERVATION_TIME,
        ConceptColumnRole.EVENT_TIME,
    ],
)
def test_typed_time_representations_use_their_sealed_time_unit(role) -> None:
    definition = load_dictionary(include_sofa2=True).get("lact")
    assert definition is not None
    binding = ColumnMetadataBinding(
        metadata=project_concept_column_metadata(
            definition,
            spec=ColumnProjectionSpec(
                column_name=f"lact_{role.value}",
                source_concept="lact",
                role=role,
                time_origin="icu_admission",
                time_unit="h",
            ),
            source_database="miiv",
        )
    )
    canonical = canonical_column_binding(binding.metadata.column_name, binding)

    updates = descriptor_physical_updates(canonical)

    assert binding.metadata.canonical_unit is None
    assert binding.metadata.time_unit == "h"
    assert updates["unit"] == "h"
    assert updates["temporal_resolution"] == "relative to icu_admission in h"


def test_one_sided_analysis_plausibility_range_is_preserved(
    tmp_path: Path,
) -> None:
    base = _binding("age", "age", ConceptColumnRole.VALUE)
    one_sided = ColumnMetadataBinding(
        metadata=replace(
            base.metadata,
            analysis_plausibility_range=NumericBounds(minimum=0.0),
        )
    )
    context = _build_v2_context(
        tmp_path,
        binding_overrides={"age": one_sided},
    )

    assert context.materialized_inputs.cohort.column_bindings[
        "age"
    ].analysis_plausibility_range == {"minimum": 0.0, "maximum": None}
    assert context.variable("age").valid_range is None


def test_typed_range_fallback_uses_sealed_source_concept() -> None:
    definition = load_dictionary(include_sofa2=True).get("lact")
    assert definition is not None
    binding = ColumnMetadataBinding(
        metadata=project_concept_column_metadata(
            definition,
            spec=ColumnProjectionSpec(
                column_name="opaque_signal",
                source_concept="lact",
                role=ConceptColumnRole.NUMERIC_AGGREGATE,
                aggregation="max",
            ),
            source_database="miiv",
        )
    )

    assert effective_analysis_plausibility_range(binding) == {
        "minimum": 0.0,
        "maximum": 30.0,
    }


def test_non_range_preserving_sum_does_not_inherit_icu_range() -> None:
    definition = load_dictionary(include_sofa2=True).get("lact")
    assert definition is not None
    binding = ColumnMetadataBinding(
        metadata=project_concept_column_metadata(
            definition,
            spec=ColumnProjectionSpec(
                column_name="lact_total",
                source_concept="lact",
                role=ConceptColumnRole.NUMERIC_AGGREGATE,
                aggregation="sum",
            ),
            source_database="miiv",
        )
    )
    payload = binding.to_dict()

    assert binding_preserves_analysis_range(binding) is False
    assert effective_analysis_plausibility_range(binding) is None
    CanonicalColumnBinding(
        binding=payload,
        binding_sha256=binding_payload_sha256({"lact_total": binding}),
        analysis_plausibility_range=None,
    )
    with pytest.raises(ValueError, match="range-preserving binding"):
        CanonicalColumnBinding(
            binding=payload,
            binding_sha256=binding_payload_sha256({"lact_total": binding}),
            analysis_plausibility_range={"minimum": 0.0, "maximum": 30.0},
        )


def test_v2_rejects_context_identity_that_disagrees_with_typed_cohort(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="typed cohort identity"):
        _build_v2_context(tmp_path, id_columns=("age",))


def test_v2_builder_reads_through_verified_snapshot_not_plain_parquet(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _typed_export(tmp_path / "export")
    paths = cohort_materializer.materialize_to_parquet(
        tmp_path / "materialized",
        data_path=source,
        database="miiv",
        static_concepts=("age",),
        feature_concepts=("lact",),
        outcome_concepts=("death",),
    )
    from easyicu.research_agent.research_context import builder as context_module

    monkeypatch.setattr(
        context_module.pd,
        "read_parquet",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("plain parquet reopen is forbidden for typed context")
        ),
    )
    context = build_research_context(
        research_question="Use verified typed bytes.",
        cohort=paths["parquet"],
        cohort_name="typed_context",
        database="miiv",
        id_columns=("stay_id",),
        outcome_columns=("death",),
    )
    assert isinstance(context, ResearchContextV2)


def test_initial_authority_binds_exact_export_source_coordinates(
    tmp_path: Path,
) -> None:
    source = _typed_export(tmp_path / "export")
    paths = cohort_materializer.materialize_to_parquet(
        tmp_path / "materialized",
        data_path=source,
        database="miiv",
        static_concepts=("age",),
        feature_concepts=("lact",),
        outcome_concepts=("death",),
    )
    verified = load_verified_materialized_cohort_authority(paths["parquet"])
    assert verified is not None
    derivations = {
        item.output_column: item for item in verified.authority.output_derivations
    }

    source_ref = derivations["lact_max"].sources[0]
    assert source_ref.authority_sha256 == (
        verified.authority.source_export_authority_sha256
    )
    assert source_ref.file == "labs.parquet"
    assert source_ref.column == "lact"
    assert source_ref.binding_sha256
    assert derivations["lact_max"].transform_id == "window_numeric_max"


def test_initial_derivation_cannot_name_an_unrelated_source_authority(
    tmp_path: Path,
) -> None:
    source = _typed_export(tmp_path / "export")
    paths = cohort_materializer.materialize_to_parquet(
        tmp_path / "materialized",
        data_path=source,
        database="miiv",
        static_concepts=("age",),
        feature_concepts=("lact",),
        outcome_concepts=("death",),
    )
    verified = load_verified_materialized_cohort_authority(paths["parquet"])
    assert verified is not None
    derivations = list(verified.authority.output_derivations)
    target = next(
        index
        for index, derivation in enumerate(derivations)
        if derivation.output_column == "lact_max"
    )
    original = derivations[target]
    derivations[target] = replace(
        original,
        sources=(replace(original.sources[0], authority_sha256="f" * 64),),
    )
    forged = replace(verified.authority, output_derivations=tuple(derivations))

    with pytest.raises(MaterializedMetadataError, match="source authority mismatch"):
        materialized._validate_derivation_contract(
            forged,
            file_binding=verified.sidecar.files[0],
            source_sidecar=materialized._read_source_column_metadata(
                paths["parquet"].parent,
                authority=verified.authority,
            ),
        )


def test_initial_authority_requires_source_sidecar_snapshot(tmp_path: Path) -> None:
    source = _typed_export(tmp_path / "export")
    paths = cohort_materializer.materialize_to_parquet(
        tmp_path / "materialized",
        data_path=source,
        database="miiv",
        static_concepts=("age",),
        feature_concepts=("lact",),
        outcome_concepts=("death",),
    )
    verified = load_verified_materialized_cohort_authority(paths["parquet"])
    assert verified is not None
    (paths["parquet"].parent / verified.authority.source_column_metadata.file).unlink()

    with pytest.raises(MaterializedMetadataError):
        load_verified_materialized_cohort_authority(paths["parquet"])


def test_initial_authority_rejects_coordinated_fake_export_digest(
    tmp_path: Path,
) -> None:
    source = _typed_export(tmp_path / "export")
    paths = cohort_materializer.materialize_to_parquet(
        tmp_path / "materialized",
        data_path=source,
        database="miiv",
        static_concepts=("age",),
        feature_concepts=("lact",),
        outcome_concepts=("death",),
    )
    verified = load_verified_materialized_cohort_authority(paths["parquet"])
    assert verified is not None
    fake_digest = "f" * 64
    forged = replace(
        verified.authority,
        source_export_authority_sha256=fake_digest,
        output_derivations=tuple(
            replace(
                derivation,
                sources=tuple(
                    replace(source_ref, authority_sha256=fake_digest)
                    for source_ref in derivation.sources
                ),
            )
            for derivation in verified.authority.output_derivations
        ),
    )
    _resign_selected_authority(
        paths["parquet"],
        forged,
        sidecar=verified.sidecar,
    )

    with pytest.raises(MaterializedMetadataError, match="provenance"):
        load_verified_materialized_cohort_authority(paths["parquet"])


@pytest.mark.parametrize("coordinate", ["file", "column", "binding"])
def test_initial_authority_rejects_resigned_fake_source_coordinate(
    tmp_path: Path,
    coordinate: str,
) -> None:
    source = _typed_export(tmp_path / "export")
    paths = cohort_materializer.materialize_to_parquet(
        tmp_path / "materialized",
        data_path=source,
        database="miiv",
        static_concepts=("age",),
        feature_concepts=("lact",),
        outcome_concepts=("death",),
    )
    verified = load_verified_materialized_cohort_authority(paths["parquet"])
    assert verified is not None
    derivations = list(verified.authority.output_derivations)
    index = next(
        i for i, item in enumerate(derivations) if item.output_column == "lact_max"
    )
    source_ref = derivations[index].sources[0]
    changes = {
        "file": {"file": "outcomes.parquet"},
        "column": {"column": "death"},
        "binding": {"binding_sha256": "f" * 64},
    }[coordinate]
    derivations[index] = replace(
        derivations[index],
        sources=(replace(source_ref, **changes),),
    )
    forged = replace(verified.authority, output_derivations=tuple(derivations))
    _resign_selected_authority(
        paths["parquet"],
        forged,
        sidecar=verified.sidecar,
    )

    with pytest.raises(MaterializedMetadataError, match="source metadata"):
        load_verified_materialized_cohort_authority(paths["parquet"])


def _publish_typed_analysis_subset(tmp_path: Path):
    source = _typed_export(tmp_path / "export")
    paths = cohort_materializer.materialize_to_parquet(
        tmp_path / "materialized",
        data_path=source,
        database="miiv",
        static_concepts=("age",),
        feature_concepts=("lact",),
        outcome_concepts=("death",),
    )
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    parent_path = run_dir / "cohort.parquet"
    parent = stage_materialized_cohort_authority(
        paths["parquet"],
        parent_path,
        producer_implementation_sha256=implementation_bundle_sha256(
            (Path(cohort_materializer.__file__),)
        ),
    )
    assert parent is not None
    child_path = run_dir / "cohort_analysis.parquet"
    definition = {"name": "test", "inclusion": [], "exclusion": []}
    child = publish_ordered_subset_materialized_cohort(
        parent_path,
        child_path,
        selected_row_positions=(1,),
        semantic_provenance={
            "cohort_definition": definition,
            "cohort_sha256": "e" * 64,
            "n_universe": 2,
            "n_analysis_cohort": 1,
            "predicate_column_bindings": [],
        },
        producer_implementation_sha256=implementation_bundle_sha256(
            (Path(cohort_materializer.__file__),)
        ),
        producer_parameters={
            "cohort_definition": definition,
            "cohort_definition_sha256": "e" * 64,
            "predicate_column_bindings": [],
            "stem": "cohort_analysis",
        },
    )
    assert child is not None
    return parent_path, parent, child_path, child


def test_ordered_analysis_subset_seals_exact_parent_bound_child(
    tmp_path: Path,
) -> None:
    parent_path, parent, child_path, child = _publish_typed_analysis_subset(tmp_path)

    assert parent_path.exists()
    assert pd.read_parquet(child_path)["stay_id"].tolist() == [2]
    assert child.authority.parent_authority_sha256 == parent.reference.sha256
    assert child.authority.cohort_columns == parent.authority.cohort_columns
    assert {
        derivation.transform_id for derivation in child.authority.output_derivations
    } == {"ordered_row_subset"}
    assert load_verified_materialized_cohort_authority(child_path) is not None


def test_typed_development_sample_is_child_of_locked_analysis_authority(
    tmp_path: Path,
) -> None:
    _parent_path, _parent, child_path, child = _publish_typed_analysis_subset(tmp_path)

    binding = materialize_development_execution_sample(
        run_dir=child_path.parent,
        target_rows=1,
        seed=20260719,
        declared_id_columns=("stay_id",),
        trajectory_binding=None,
    )

    sampled = load_verified_materialized_cohort_authority(binding.cohort_path)
    assert sampled is not None
    assert binding.cohort_authority_ref == sampled.reference
    assert sampled.authority.parent_authority_sha256 == child.reference.sha256
    assert sampled.provenance["paper_authority"] is False
    assert (
        materialize_development_execution_sample(
            run_dir=child_path.parent,
            target_rows=1,
            seed=20260719,
            declared_id_columns=("stay_id",),
            trajectory_binding=None,
        )
        == binding
    )


def test_ordered_analysis_subset_rejects_same_parent_and_target_without_mutation(
    tmp_path: Path,
) -> None:
    parent_path, parent, _child_path, _child = _publish_typed_analysis_subset(tmp_path)
    artifact_before = parent_path.read_bytes()
    selector_path = materialized.materialized_provenance_path(parent_path)
    selector_before = selector_path.read_bytes()

    with pytest.raises(MaterializedMetadataError, match="different artifacts"):
        publish_ordered_subset_materialized_cohort(
            parent_path,
            parent_path,
            selected_row_positions=(1,),
            semantic_provenance={"cohort_definition_sha256": "e" * 64},
            producer_implementation_sha256="a" * 64,
            producer_parameters={"predicate_bindings": []},
            expected_parent_authority=parent.reference,
        )

    assert parent_path.read_bytes() == artifact_before
    assert selector_path.read_bytes() == selector_before
    assert (
        load_verified_materialized_cohort_authority(
            parent_path,
            expected_authority=parent.reference,
        )
        is not None
    )


def test_analysis_directory_swap_cannot_redirect_writes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent_path, parent, _child_path, _child = _publish_typed_analysis_subset(tmp_path)
    run_root = parent_path.parent
    victim = tmp_path / "victim"
    victim.mkdir()
    sentinel = victim / "sentinel.txt"
    sentinel.write_text("unchanged", encoding="utf-8")
    displaced = tmp_path / "displaced_run"
    original = AnchoredDirectory.require_absent
    swapped = False

    def swap_after_anchor(self: AnchoredDirectory, *names: str) -> None:
        nonlocal swapped
        if not swapped and "cohort_analysis2.parquet" in names:
            run_root.rename(displaced)
            run_root.symlink_to(victim, target_is_directory=True)
            swapped = True
        original(self, *names)

    monkeypatch.setattr(AnchoredDirectory, "require_absent", swap_after_anchor)

    with pytest.raises(
        MaterializedMetadataError,
        match="cannot publish ordered analysis cohort",
    ):
        publish_ordered_subset_materialized_cohort(
            parent_path,
            run_root / "cohort_analysis2.parquet",
            selected_row_positions=(0,),
            semantic_provenance={
                "cohort_definition": {
                    "name": "test",
                    "inclusion": [],
                    "exclusion": [],
                },
                "cohort_sha256": "e" * 64,
                "n_universe": 2,
                "n_analysis_cohort": 1,
                "predicate_column_bindings": [],
            },
            producer_implementation_sha256="a" * 64,
            producer_parameters={
                "cohort_definition": {
                    "name": "test",
                    "inclusion": [],
                    "exclusion": [],
                },
                "cohort_definition_sha256": "e" * 64,
                "predicate_column_bindings": [],
                "stem": "cohort_analysis2",
            },
            expected_parent_authority=parent.reference,
        )

    assert swapped is True
    assert sentinel.read_text(encoding="utf-8") == "unchanged"
    assert {item.name for item in victim.iterdir()} == {"sentinel.txt"}


def test_ordered_analysis_subset_cannot_overwrite_an_ancestor(
    tmp_path: Path,
) -> None:
    parent_path, parent, child_path, child = _publish_typed_analysis_subset(tmp_path)
    parent_bytes = parent_path.read_bytes()
    parent_selector = materialized.materialized_provenance_path(parent_path)
    parent_selector_bytes = parent_selector.read_bytes()

    with pytest.raises(MaterializedMetadataError, match="cannot be overwritten"):
        publish_ordered_subset_materialized_cohort(
            child_path,
            parent_path,
            selected_row_positions=(0,),
            semantic_provenance={"cohort_definition_sha256": "d" * 64},
            producer_implementation_sha256="a" * 64,
            producer_parameters={"predicate_bindings": []},
            expected_parent_authority=child.reference,
        )

    assert parent_path.read_bytes() == parent_bytes
    assert parent_selector.read_bytes() == parent_selector_bytes
    assert (
        load_verified_materialized_cohort_authority(
            parent_path,
            expected_authority=parent.reference,
        )
        is not None
    )
    assert (
        load_verified_materialized_cohort_authority(
            child_path,
            expected_authority=child.reference,
        )
        is not None
    )


def test_analysis_subset_rejects_resigned_metadata_projection(tmp_path: Path) -> None:
    _parent_path, _parent, child_path, child = _publish_typed_analysis_subset(tmp_path)
    forged_binding = replace(child.sidecar.files[0], module="forged_analysis")
    forged_sidecar = replace(child.sidecar, files=(forged_binding,))
    forged_ref = write_content_addressed_sidecar(
        child_path.parent,
        forged_sidecar,
        stem="cohort_column_metadata",
    )
    forged_authority = replace(
        child.authority,
        column_metadata=forged_ref,
        file_metadata_payload_sha256=forged_binding.metadata_payload_sha256,
    )
    _resign_selected_authority(
        child_path,
        forged_authority,
        sidecar=forged_sidecar,
    )

    with pytest.raises(MaterializedMetadataError, match="parent binding mismatch"):
        load_verified_materialized_cohort_authority(child_path)


def test_analysis_subset_rejects_resigned_false_row_count(tmp_path: Path) -> None:
    _parent_path, _parent, child_path, child = _publish_typed_analysis_subset(tmp_path)
    forged_provenance = materialized._thaw_json(child.authority.semantic_provenance)
    forged_provenance["n_analysis_cohort"] = 999
    forged = replace(child.authority, semantic_provenance=forged_provenance)
    _resign_selected_authority(
        child_path,
        forged,
        sidecar=child.sidecar,
    )
    selector_path = materialized.materialized_provenance_path(child_path)
    selector = json.loads(selector_path.read_text(encoding="utf-8"))
    selector["n_analysis_cohort"] = 999
    selector_path.write_text(json.dumps(selector), encoding="utf-8")

    with pytest.raises(MaterializedMetadataError, match="position receipt"):
        load_verified_materialized_cohort_authority(child_path)


def test_analysis_subset_fails_when_local_parent_authority_is_deleted(
    tmp_path: Path,
) -> None:
    _parent_path, parent, child_path, _child = _publish_typed_analysis_subset(tmp_path)
    (child_path.parent / parent.reference.file).unlink()

    with pytest.raises(MaterializedMetadataError, match="parent authority is missing"):
        load_verified_materialized_cohort_authority(child_path)


def test_analysis_subset_fails_when_local_parent_artifact_is_tampered(
    tmp_path: Path,
) -> None:
    parent_path, _parent, child_path, _child = _publish_typed_analysis_subset(tmp_path)
    parent_frame = pd.read_parquet(parent_path)
    parent_frame.loc[0, "lact_max"] = 999.0
    parent_frame.to_parquet(parent_path, index=False)

    with pytest.raises(MaterializedMetadataError, match="no longer matches"):
        load_verified_materialized_cohort_authority(child_path)


def _adult_over_55_definition() -> CohortDefinition:
    return CohortDefinition(
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


def test_locked_typed_analysis_cohort_uses_parent_bound_arrow_subset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _typed_export(tmp_path / "export")
    paths = cohort_materializer.materialize_to_parquet(
        tmp_path / "materialized",
        data_path=source,
        database="miiv",
        static_concepts=("age",),
        feature_concepts=("lact",),
        outcome_concepts=("death",),
    )
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    parent_path = run_dir / "cohort.parquet"
    parent = stage_materialized_cohort_authority(
        paths["parquet"],
        parent_path,
        producer_implementation_sha256=implementation_bundle_sha256(
            (Path(cohort_materializer.__file__),)
        ),
    )
    assert parent is not None

    result = materialize_locked_analysis_cohort(
        run_dir=run_dir,
        plan=SimpleNamespace(cohort=_adult_over_55_definition(), steps=[]),
        universe_path=parent_path,
    )

    assert result["status"] == "applied"
    assert result["authority_ref"] is not None
    assert Path(result["authority_path"]).exists()
    child_path = Path(result["path"])
    assert pd.read_parquet(child_path)["stay_id"].tolist() == [2]
    child = load_verified_materialized_cohort_authority(child_path)
    assert child is not None
    assert child.authority.parent_authority_sha256 == parent.reference.sha256
    assert child.authority.cohort_schema_sha256 == parent.authority.cohort_schema_sha256
    context = build_research_context(
        research_question="Which older adults enter the analysis cohort?",
        cohort=parent_path,
        cohort_name="typed_analysis_child",
        database="miiv",
        target_outcome="death",
        primary_exposure="lact_max",
        id_columns=("stay_id",),
        outcome_columns=("death",),
    )
    selected = bind_execution_cohort_authority(
        universe_path=parent_path,
        analysis_path=child_path,
        plan=SimpleNamespace(cohort=_adult_over_55_definition(), steps=[]),
        context=context,
    )
    assert selected.selected_path == child_path
    assert selected.analysis_authority is not None
    from easyicu.research_agent.authority import analysis_cohort as authority_module

    monkeypatch.setattr(
        authority_module,
        "implementation_bundle_sha256",
        lambda _paths: "f" * 64,
    )
    with pytest.raises(
        MaterializedMetadataError,
        match="does not match the locked cohort authority",
    ):
        bind_execution_cohort_authority(
            universe_path=parent_path,
            analysis_path=child_path,
            plan=SimpleNamespace(cohort=_adult_over_55_definition(), steps=[]),
            context=context,
        )


def test_locked_typed_analysis_cohort_rejects_parent_selector_swap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_source = _typed_export(tmp_path / "export_a")
    first_paths = cohort_materializer.materialize_to_parquet(
        tmp_path / "materialized_a",
        data_path=first_source,
        database="miiv",
        static_concepts=("age",),
        feature_concepts=("lact",),
        outcome_concepts=("death",),
    )
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    parent_path = run_dir / "cohort.parquet"
    first_parent = stage_materialized_cohort_authority(
        first_paths["parquet"],
        parent_path,
        producer_implementation_sha256=implementation_bundle_sha256(
            (Path(cohort_materializer.__file__),)
        ),
    )
    assert first_parent is not None
    real_publish = materialized.publish_ordered_subset_materialized_cohort

    def swap_then_publish(*args, **kwargs):
        selector_path = materialized.materialized_provenance_path(parent_path)
        selector = json.loads(selector_path.read_text(encoding="utf-8"))
        selector["column_metadata"]["authority"] = {
            "schema_version": "easyicu.materialized_cohort_authority_ref/1",
            "file": f"cohort_authority.sha256-{'f' * 64}.json",
            "sha256": "f" * 64,
            "size": 2,
        }
        selector_path.write_text(json.dumps(selector), encoding="utf-8")
        return real_publish(*args, **kwargs)

    monkeypatch.setattr(
        materialized,
        "publish_ordered_subset_materialized_cohort",
        swap_then_publish,
    )

    with pytest.raises(
        MaterializedMetadataError,
        match="caller-selected reference",
    ):
        materialize_locked_analysis_cohort(
            run_dir=run_dir,
            plan=SimpleNamespace(cohort=_adult_over_55_definition(), steps=[]),
            universe_path=parent_path,
        )

    assert not (run_dir / "cohort_analysis.parquet").exists()


def test_locked_typed_analysis_cohort_never_falls_back_on_authority_failure(
    tmp_path: Path,
) -> None:
    source = _typed_export(tmp_path / "export")
    paths = cohort_materializer.materialize_to_parquet(
        tmp_path / "materialized",
        data_path=source,
        database="miiv",
        static_concepts=("age",),
        feature_concepts=("lact",),
        outcome_concepts=("death",),
    )
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    parent_path = run_dir / "cohort.parquet"
    staged = stage_materialized_cohort_authority(
        paths["parquet"],
        parent_path,
        producer_implementation_sha256=implementation_bundle_sha256(
            (Path(cohort_materializer.__file__),)
        ),
    )
    assert staged is not None
    (run_dir / staged.reference.file).write_text("{}", encoding="utf-8")

    with pytest.raises(MaterializedMetadataError):
        materialize_locked_analysis_cohort(
            run_dir=run_dir,
            plan=SimpleNamespace(cohort=_adult_over_55_definition(), steps=[]),
            universe_path=parent_path,
        )

    assert not (run_dir / "cohort_analysis.parquet").exists()


def test_locked_typed_analysis_cohort_never_falls_back_on_predicate_failure(
    tmp_path: Path,
) -> None:
    source = _typed_export(tmp_path / "export")
    paths = cohort_materializer.materialize_to_parquet(
        tmp_path / "materialized",
        data_path=source,
        database="miiv",
        static_concepts=("age",),
        feature_concepts=("lact",),
        outcome_concepts=("death",),
    )
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    parent_path = run_dir / "cohort.parquet"
    staged = stage_materialized_cohort_authority(
        paths["parquet"],
        parent_path,
        producer_implementation_sha256=implementation_bundle_sha256(
            (Path(cohort_materializer.__file__),)
        ),
    )
    assert staged is not None
    missing_concept = CohortDefinition(
        name="missing_typed_column",
        inclusion=(
            ConceptPredicate(
                concept_id="crea",
                time_window=TimeWindow(
                    anchor="icu_admit",
                    start_offset_hours=0,
                    end_offset_hours=24,
                ),
                aggregation="max",
                op=">=",
                value=0,
            ),
        ),
    )

    with pytest.raises(
        MaterializedMetadataError,
        match="could not be applied to its sealed universe",
    ):
        materialize_locked_analysis_cohort(
            run_dir=run_dir,
            plan=SimpleNamespace(cohort=missing_concept, steps=[]),
            universe_path=parent_path,
        )

    assert not (run_dir / "cohort_analysis.parquet").exists()


def test_typed_value_rejects_lossy_numeric_summary(tmp_path: Path) -> None:
    labs = pd.DataFrame(
        {
            "stay_id": [1, 1, 2],
            "charttime": [1.0, 2.0, 1.0],
            "age": [50, 50, 60],
            "lact": ["bad", "1.2", "3.0"],
            "mech_vent": [False, True, False],
        }
    )
    source = _typed_export(tmp_path / "export", labs=labs)

    with pytest.raises(MaterializedMetadataError, match="lossy numeric coercion"):
        cohort_materializer.materialize_to_parquet(
            tmp_path / "materialized",
            data_path=source,
            database="miiv",
            static_concepts=("age",),
            feature_concepts=("lact",),
            outcome_concepts=("death",),
        )


@pytest.mark.parametrize(
    ("age_values", "lact_values", "expected"),
    [
        ([999, 999, 60], [1.0, 2.0, 3.0], "age"),
        ([50, 50, 60], [999.0, 2.0, 3.0], "lact"),
    ],
)
def test_typed_range_preserving_values_enforce_sealed_bounds(
    tmp_path: Path,
    age_values: list[float],
    lact_values: list[float],
    expected: str,
) -> None:
    labs = pd.DataFrame(
        {
            "stay_id": [1, 1, 2],
            "charttime": [1.0, 2.0, 1.0],
            "age": age_values,
            "lact": lact_values,
            "mech_vent": [False, True, False],
        }
    )
    source = _typed_export(tmp_path / "export", labs=labs)

    with pytest.raises(MaterializedMetadataError, match=expected):
        cohort_materializer.materialize_to_parquet(
            tmp_path / "materialized",
            data_path=source,
            database="miiv",
            static_concepts=("age",),
            feature_concepts=("lact",),
            outcome_concepts=("death",),
        )


def test_typed_value_cannot_be_promoted_to_event_outcome(tmp_path: Path) -> None:
    source = _typed_export(
        tmp_path / "export", roles={"death": ConceptColumnRole.VALUE}
    )

    with pytest.raises(MaterializedMetadataError, match="not authorized"):
        cohort_materializer.materialize_to_parquet(
            tmp_path / "materialized",
            data_path=source,
            database="miiv",
            static_concepts=("age",),
            feature_concepts=("lact",),
            outcome_concepts=("death",),
        )


def test_typed_boolean_value_cannot_be_promoted_to_event_summary(
    tmp_path: Path,
) -> None:
    source = _typed_export(
        tmp_path / "export", roles={"mech_vent": ConceptColumnRole.VALUE}
    )

    with pytest.raises(MaterializedMetadataError, match="as a boolean"):
        cohort_materializer.materialize_to_parquet(
            tmp_path / "materialized",
            data_path=source,
            database="miiv",
            static_concepts=("age",),
            feature_concepts=("mech_vent",),
            outcome_concepts=("death",),
        )


def test_typed_static_conflicts_fail_closed(tmp_path: Path) -> None:
    labs = pd.DataFrame(
        {
            "stay_id": [1, 1, 2],
            "charttime": [1.0, 2.0, 1.0],
            "age": [50, 80, 60],
            "lact": [1.0, 2.0, 3.0],
            "mech_vent": [False, True, False],
        }
    )
    source = _typed_export(tmp_path / "export", labs=labs)

    with pytest.raises(MaterializedMetadataError, match="conflicting stay-level"):
        cohort_materializer.materialize_to_parquet(
            tmp_path / "materialized",
            data_path=source,
            database="miiv",
            static_concepts=("age",),
            feature_concepts=("lact",),
            outcome_concepts=("death",),
        )


def test_typed_event_all_null_preserves_zero_measurement_semantics(
    tmp_path: Path,
) -> None:
    labs = pd.DataFrame(
        {
            "stay_id": [1, 1, 2],
            "charttime": [1.0, 2.0, 1.0],
            "age": [50, 50, 60],
            "lact": [1.0, 2.0, 3.0],
            "mech_vent": pd.Series([None, None, None], dtype="boolean"),
        }
    )
    source = _typed_export(tmp_path / "export", labs=labs)
    paths = cohort_materializer.materialize_to_parquet(
        tmp_path / "materialized",
        data_path=source,
        database="miiv",
        static_concepts=("age",),
        feature_concepts=("mech_vent",),
        outcome_concepts=("death",),
    )
    cohort = pd.read_parquet(paths["parquet"])

    assert cohort["mech_vent_n"].tolist() == [0, 0]
    assert cohort["mech_vent_measured"].tolist() == [0, 0]
    assert cohort["mech_vent_max"].tolist() == [0, 0]
    assert cohort["mech_vent_mean"].tolist() == [0.0, 0.0]


def test_typed_predicate_normalizes_icu_admission_anchor(tmp_path: Path) -> None:
    source = _typed_export(tmp_path / "export")
    predicate = ConceptPredicate(
        concept_id="lact",
        time_window=TimeWindow(
            anchor="icu_admit",
            start_offset_hours=0,
            end_offset_hours=24,
        ),
        aggregation="max",
        op=">",
        value=0,
    )
    paths = cohort_materializer.materialize_to_parquet(
        tmp_path / "materialized",
        data_path=source,
        database="miiv",
        static_concepts=("age",),
        feature_concepts=(),
        outcome_concepts=("death",),
        cohort_definition=CohortDefinition(name="test", inclusion=(predicate,)),
    )
    verified = load_verified_materialized_cohort_authority(paths["parquet"])
    assert verified is not None
    binding = verified.sidecar.files[0].columns["lact"]
    assert binding.derivation_window is not None
    assert binding.derivation_window.origin == "icu_admission"


def test_typed_predicate_rejects_unconverted_time_anchor(tmp_path: Path) -> None:
    source = _typed_export(tmp_path / "export")
    predicate = ConceptPredicate(
        concept_id="lact",
        time_window=TimeWindow(
            anchor="hospital_admit",
            start_offset_hours=0,
            end_offset_hours=24,
        ),
        aggregation="max",
        op=">",
        value=0,
    )

    with pytest.raises(MaterializedMetadataError, match="unsupported time anchor"):
        cohort_materializer.materialize_to_parquet(
            tmp_path / "materialized",
            data_path=source,
            database="miiv",
            static_concepts=("age",),
            feature_concepts=(),
            outcome_concepts=("death",),
            cohort_definition=CohortDefinition(name="test", inclusion=(predicate,)),
        )


def test_typed_predicate_rejects_first_write_wins_across_windows(
    tmp_path: Path,
) -> None:
    source = _typed_export(tmp_path / "export")
    predicates = (
        ConceptPredicate(
            concept_id="lact",
            time_window=TimeWindow(
                anchor="icu_admit",
                start_offset_hours=0,
                end_offset_hours=12,
            ),
            aggregation="max",
            op=">",
            value=0,
        ),
        ConceptPredicate(
            concept_id="lact",
            time_window=TimeWindow(
                anchor="icu_admit",
                start_offset_hours=0,
                end_offset_hours=24,
            ),
            aggregation="mean",
            op=">",
            value=0,
        ),
    )

    with pytest.raises(MaterializedMetadataError, match="multiple incompatible"):
        cohort_materializer.materialize_to_parquet(
            tmp_path / "materialized",
            data_path=source,
            database="miiv",
            static_concepts=("age",),
            feature_concepts=(),
            outcome_concepts=("death",),
            cohort_definition=CohortDefinition(name="test", inclusion=predicates),
        )


def test_typed_value_predicate_cannot_use_boolean_aggregation(
    tmp_path: Path,
) -> None:
    source = _typed_export(tmp_path / "export")
    predicate = ConceptPredicate(
        concept_id="lact",
        time_window=TimeWindow(
            anchor="icu_admit",
            start_offset_hours=0,
            end_offset_hours=24,
        ),
        aggregation="any",
        op="==",
        value=True,
    )

    with pytest.raises(MaterializedMetadataError, match="cannot use 'any'"):
        cohort_materializer.materialize_to_parquet(
            tmp_path / "materialized",
            data_path=source,
            database="miiv",
            static_concepts=("age",),
            feature_concepts=(),
            outcome_concepts=("death",),
            cohort_definition=CohortDefinition(name="test", inclusion=(predicate,)),
        )


def test_typed_event_rejects_unknown_or_nonbinary_values(tmp_path: Path) -> None:
    labs = pd.DataFrame(
        {
            "stay_id": [1, 1, 2],
            "charttime": [1.0, 2.0, 1.0],
            "age": [50, 50, 60],
            "lact": [1.0, 2.0, 3.0],
            "mech_vent": ["maybe", "2", "off"],
        }
    )
    source = _typed_export(tmp_path / "export", labs=labs)

    with pytest.raises(MaterializedMetadataError, match="event concept"):
        cohort_materializer.materialize_to_parquet(
            tmp_path / "materialized",
            data_path=source,
            database="miiv",
            static_concepts=("age",),
            feature_concepts=("mech_vent",),
            outcome_concepts=("death",),
        )


def test_typed_numeric_output_rejects_infinity(tmp_path: Path) -> None:
    labs = pd.DataFrame(
        {
            "stay_id": [1, 1, 2],
            "charttime": [1.0, 2.0, 1.0],
            "age": [50, 50, 60],
            "lact": [1.0, float("inf"), 3.0],
            "mech_vent": [False, True, False],
        }
    )
    source = _typed_export(tmp_path / "export", labs=labs)

    with pytest.raises(MaterializedMetadataError, match="non-finite"):
        cohort_materializer.materialize_to_parquet(
            tmp_path / "materialized",
            data_path=source,
            database="miiv",
            static_concepts=("age",),
            feature_concepts=("lact",),
            outcome_concepts=("death",),
        )


def test_typed_static_event_domain_is_validated_before_integer_cast(
    tmp_path: Path,
) -> None:
    labs = pd.DataFrame(
        {
            "stay_id": [1, 2],
            "charttime": [1.0, 1.0],
            "age": [50, 60],
            "lact": [1.0, 3.0],
            "mech_vent": [0.5, 2.0],
        }
    )
    source = _typed_export(tmp_path / "export", labs=labs)

    with pytest.raises(MaterializedMetadataError, match="non-binary numeric"):
        cohort_materializer.materialize_to_parquet(
            tmp_path / "materialized",
            data_path=source,
            database="miiv",
            static_concepts=("age", "mech_vent"),
            feature_concepts=(),
            outcome_concepts=("death",),
        )


def test_typed_timed_predicate_cannot_reuse_whole_stay_outcome(
    tmp_path: Path,
) -> None:
    source = _typed_export(tmp_path / "export")
    predicate = ConceptPredicate(
        concept_id="death",
        time_window=TimeWindow(
            anchor="icu_admit", start_offset_hours=0, end_offset_hours=24
        ),
        aggregation="any",
        op="==",
        value=True,
    )
    with pytest.raises(MaterializedMetadataError, match="whole-stay outcome"):
        cohort_materializer.materialize_to_parquet(
            tmp_path / "materialized",
            data_path=source,
            database="miiv",
            static_concepts=("age",),
            feature_concepts=(),
            outcome_concepts=("death",),
            cohort_definition=CohortDefinition(name="test", inclusion=(predicate,)),
        )


def test_materialize_to_parquet_preserves_public_kwarg_validation(
    tmp_path: Path,
) -> None:
    with pytest.raises(TypeError):
        cohort_materializer.materialize_to_parquet(
            tmp_path / "materialized", typo_option="ignored"
        )


def test_materialization_stem_must_be_one_path_component(tmp_path: Path) -> None:
    with pytest.raises(MaterializedMetadataError, match="one path component"):
        cohort_materializer.materialize_to_parquet(
            tmp_path / "materialized",
            stem="../escaped",
            data_path=tmp_path / "unused",
            feature_concepts=(),
        )
    assert not (tmp_path / "escaped.parquet").exists()


def test_typed_authority_cannot_downgrade_when_selector_is_deleted(
    tmp_path: Path,
) -> None:
    source = _typed_export(tmp_path / "export")
    paths = cohort_materializer.materialize_to_parquet(
        tmp_path / "materialized",
        data_path=source,
        database="miiv",
        static_concepts=("age",),
        feature_concepts=("lact",),
        outcome_concepts=("death",),
    )
    paths["provenance"].unlink()

    with pytest.raises(MaterializedMetadataError, match="selector is missing"):
        load_verified_materialized_cohort_authority(paths["parquet"])


def test_typed_authority_binds_semantic_provenance(tmp_path: Path) -> None:
    source = _typed_export(tmp_path / "export")
    paths = cohort_materializer.materialize_to_parquet(
        tmp_path / "materialized",
        data_path=source,
        database="miiv",
        static_concepts=("age",),
        feature_concepts=("lact",),
        outcome_concepts=("death",),
    )
    provenance = json.loads(paths["provenance"].read_text(encoding="utf-8"))
    provenance["database"] = "fabricated"
    paths["provenance"].write_text(json.dumps(provenance), encoding="utf-8")

    with pytest.raises(MaterializedMetadataError, match="semantic provenance"):
        load_verified_materialized_cohort_authority(paths["parquet"])


def test_verified_authority_payload_is_deeply_immutable(tmp_path: Path) -> None:
    source = _typed_export(tmp_path / "export")
    paths = cohort_materializer.materialize_to_parquet(
        tmp_path / "materialized",
        data_path=source,
        database="miiv",
        static_concepts=("age",),
        feature_concepts=("lact",),
        outcome_concepts=("death",),
    )
    verified = load_verified_materialized_cohort_authority(paths["parquet"])
    assert verified is not None

    with pytest.raises(TypeError):
        verified.authority.semantic_provenance["database"] = "fabricated"  # type: ignore[index]


def test_materialization_rejects_symlinked_output_root_before_artifact_write(
    tmp_path: Path,
) -> None:
    source = _typed_export(tmp_path / "export")
    victim = tmp_path / "victim"
    victim.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(victim, target_is_directory=True)

    with pytest.raises(MaterializedMetadataError, match="real directory"):
        cohort_materializer.materialize_to_parquet(
            alias,
            data_path=source,
            database="miiv",
            static_concepts=("age",),
            feature_concepts=("lact",),
            outcome_concepts=("death",),
        )
    assert not (victim / "cohort.parquet").exists()


def test_materialized_metadata_lifecycle_has_no_orchestration_imports() -> None:
    module_path = (
        Path(__file__).resolve().parents[2]
        / "src"
        / "easyicu"
        / "research_agent"
        / "intake"
        / "materialized_metadata.py"
    )
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    imports = {
        node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
    }

    assert not any(
        name.endswith(
            ("pipeline", "cohort_materializer", "data_foundation", "cohort_schema")
        )
        for name in imports
    )
