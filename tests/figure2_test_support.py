"""Shared typed-input fixtures for repository-local Figure 2 tests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from benchmarks.figure2_canonical9.evaluator import input_binding_v2
from benchmarks.figure2_canonical9.evaluator.rubric_v1 import FIGURE2_TASK_IDS
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
from easyicu.research_agent.research_context.builder import build_research_context
from easyicu.research_agent.authority.evidence_store import EvidenceStore
from easyicu.research_agent.intake import export_package as intake
from easyicu.research_agent.intake.materialized_metadata import (
    implementation_bundle_sha256,
    load_verified_materialized_cohort_authority,
    stage_materialized_cohort_authority,
)
from easyicu.research_agent.authority.run_input import (
    RunInputCapsuleV2,
    build_environment_identity,
    seal_run_input_capsule,
)

_DEFAULT_BINDING_PATH = (
    Path(input_binding_v2.__file__).resolve().parents[1]
    / "canonical_run_input_bindings_v2.json"
)


def _binding(
    concept: str,
    column: str,
    role: ConceptColumnRole,
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
        columns={
            "death": _binding(
                "death",
                "death",
                ConceptColumnRole.EVENT_STATUS,
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


def seal_test_run_input_capsule(
    *,
    run_dir: Path,
    evidence: EvidenceStore,
    research_question: str,
    primary_exposure: str | None,
    target_outcome: str,
) -> RunInputCapsuleV2:
    """Seal a replay-verifiable typed V2 capsule for a small ICU export."""

    source = _typed_export(run_dir.parent / f"{run_dir.name}_typed_export")
    materialized = cohort_materializer.materialize_to_parquet(
        run_dir.parent / f"{run_dir.name}_typed_materialized",
        data_path=source,
        database="miiv",
        static_concepts=("age",),
        feature_concepts=("lact",),
        outcome_concepts=("death",),
    )
    source_authority = load_verified_materialized_cohort_authority(
        materialized["parquet"]
    )
    assert source_authority is not None
    cohort_path = run_dir / "cohort.parquet"
    staged = stage_materialized_cohort_authority(
        materialized["parquet"],
        cohort_path,
        expected_source_authority=source_authority.reference,
        producer_implementation_sha256=implementation_bundle_sha256(
            (Path(cohort_materializer.__file__),)
        ),
    )
    assert staged is not None
    context = build_research_context(
        research_question=research_question,
        cohort=cohort_path,
        cohort_name="figure2_typed_fixture",
        database="miiv",
        target_outcome=target_outcome,
        primary_exposure=primary_exposure,
        inclusion_criteria=[],
        exclusion_criteria=[],
        id_columns=["stay_id"],
        outcome_columns=[target_outcome],
    )
    context_path = run_dir / "research_context.json"
    context_path.write_text(context.model_dump_json(indent=2), encoding="utf-8")
    evidence.register_file(
        kind="log",
        description="Frozen typed Figure 2 fixture research context.",
        source_path=context_path,
        evidence_id="research_context",
        producer="pipeline",
        generation_mode="system",
    )
    scientific_identity: dict[str, Any] = {
        "question": research_question,
        "database": "miiv",
        "primary_exposure": primary_exposure,
        "target_outcome": target_outcome,
        "materialized_cohort_authority_ref": (source_authority.reference.to_dict()),
    }
    capsule = seal_run_input_capsule(
        run_dir=run_dir,
        evidence=evidence,
        scientific_identity=scientific_identity,
        initial_environment=build_environment_identity(llm_signature="fixture"),
        context_path=context_path,
        cohort_path=cohort_path,
        experiment_spec_path=None,
    )
    assert isinstance(capsule, RunInputCapsuleV2)
    return capsule


def install_ready_input_binding(
    *,
    selector: Path,
    task_id: str,
    research_question: str,
    capsule: RunInputCapsuleV2,
) -> Path:
    """Install one test-only ready selector without weakening production data."""

    default_payload = json.loads(_DEFAULT_BINDING_PATH.read_text(encoding="utf-8"))
    source_ref = capsule.scientific_identity["materialized_cohort_authority_ref"]
    ready = {
        "task_id": task_id,
        "state": "ready",
        "research_question_sha256": hashlib.sha256(
            research_question.encode("utf-8")
        ).hexdigest(),
        "database": capsule.scientific_identity["database"],
        "operational_exposure": capsule.scientific_identity["primary_exposure"],
        "target_outcome": capsule.scientific_identity["target_outcome"],
        "expected_run_input_capsule_schema_version": capsule.schema_version,
        "scientific_identity_sha256": capsule.scientific_identity_sha256,
        "source_materialized_cohort_authority_ref": source_ref,
        "source_materialized_trajectory_authority_ref": None,
    }
    default_payload["tasks"] = tuple(
        (
            ready
            if current_task_id == task_id
            else {
                "task_id": current_task_id,
                "state": "blocked",
                "blockers": ("TEST_NOT_SELECTED",),
            }
        )
        for current_task_id in FIGURE2_TASK_IDS
    )
    manifest = input_binding_v2.CanonicalRunInputBindingManifest.model_validate(
        default_payload,
        strict=True,
    )
    selector.parent.mkdir(parents=True, exist_ok=True)
    selector.write_bytes(
        input_binding_v2._canonical_json_bytes(manifest.model_dump(mode="json")) + b"\n"
    )
    return selector


def ready_submission_manifest_fields() -> dict[str, Any]:
    """Return the exact profile/dictionary coordinates frozen by the selector."""

    payload = json.loads(_DEFAULT_BINDING_PATH.read_text(encoding="utf-8"))
    profile = payload["submission_profile"]
    name, version = str(profile["ref"]).split("/", 1)
    concept_sha = str(profile["concept_dict_sha256"])
    sofa2_sha = str(profile["sofa2_dict_sha256"])
    return {
        "submission_profile_name": name,
        "submission_profile_version": version,
        "concept_dict_sha": concept_sha,
        "sofa2_dict_sha": sofa2_sha,
        "concept_dict_fingerprint": {
            "concept_dict_sha": concept_sha,
            "sofa2_dict_sha": sofa2_sha,
        },
    }


__all__ = [
    "install_ready_input_binding",
    "ready_submission_manifest_fields",
    "seal_test_run_input_capsule",
]
