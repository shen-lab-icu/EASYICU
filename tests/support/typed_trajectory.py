"""A typed universe with its long-format trajectory, shared by trajectory tests.

Test modules may not import one another (``tests/governance/test_test_organization.py``);
these helpers moved here from ``test_materialized_trajectory_authority`` and
``test_run_input_trajectory_authority`` when the Web execution retry needed a
sealed V3 run input.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from easyicu.concept.metadata_projection import ConceptColumnRole
from easyicu.concept.metadata_sidecar import (
    EXPORT_PHYSICAL_SCOPE,
    ColumnMetadataFileBinding,
    ColumnMetadataSidecar,
    TimeCoordinate,
    write_content_addressed_sidecar,
)
from easyicu.research_agent.authority.evidence_store import EvidenceStore
from easyicu.research_agent.authority.run_input import build_scientific_identity
from easyicu.research_agent.cohort import materializer as cohort_materializer
from easyicu.research_agent.intake import export_package as intake
from easyicu.research_agent.intake.materialized_metadata import (
    implementation_bundle_sha256,
    load_verified_materialized_cohort_authority,
)
from easyicu.research_agent.intake.materialized_trajectory import (
    MaterializedTrajectoryAuthorityRef,
    StagedTrajectoryBinding,
    load_verified_materialized_trajectory_authority,
)
from easyicu.research_agent.research_context.builder import build_research_context
from tests.support.typed_export import metadata_binding

TRAJECTORY_QUESTION = "Is lactate associated with hospital mortality?"


def typed_trajectory_export(root: Path) -> Path:
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
            "age": metadata_binding("age", "age", ConceptColumnRole.VALUE),
            "lact": metadata_binding("lact", "lact", ConceptColumnRole.VALUE),
        },
    )
    outcome_binding = ColumnMetadataFileBinding(
        relative_path="outcomes.parquet",
        module="outcomes",
        identity_column="stay_id",
        time_coordinates=(),
        columns={
            "death": metadata_binding("death", "death", ConceptColumnRole.EVENT_STATUS)
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


def typed_trajectory_bundle(root: Path):
    source = typed_trajectory_export(root / "export")
    paths = cohort_materializer.materialize_to_parquet(
        root / "materialized",
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


def trajectory_implementation_sha() -> str:
    import easyicu.research_agent.intake.materialized_trajectory as module

    return implementation_bundle_sha256((Path(module.__file__),))


def trajectory_scientific_identity(
    *,
    cohort_path: Path,
    cohort_ref=None,
    trajectory_path: Path | None = None,
    trajectory_ref: MaterializedTrajectoryAuthorityRef | None = None,
):
    return build_scientific_identity(
        cohort=cohort_path,
        question=TRAJECTORY_QUESTION,
        cohort_name="typed_trajectory_capsule",
        database="miiv",
        target_outcome="death",
        primary_exposure="lact_max",
        cross_database_validation=None,
        inclusion_criteria=None,
        exclusion_criteria=None,
        id_columns=("stay_id",),
        time_columns=None,
        outcome_columns=("death",),
        time_windows=None,
        concept_descriptions=None,
        user_preferences=None,
        notes=None,
        skill_key=None,
        experiment_spec=None,
        source_files=None,
        disable_icu_context=False,
        materialized_cohort_authority_ref=(
            cohort_ref.to_dict() if cohort_ref is not None else None
        ),
        trajectory_path=trajectory_path,
        materialized_trajectory_authority_ref=(
            trajectory_ref.to_dict() if trajectory_ref is not None else None
        ),
    )


def trajectory_context_and_evidence(
    run_dir: Path,
    cohort_path: Path,
    *,
    trajectory_binding: StagedTrajectoryBinding | None = None,
):
    context = build_research_context(
        research_question=TRAJECTORY_QUESTION,
        cohort=cohort_path,
        cohort_name="typed_trajectory_capsule",
        database="miiv",
        target_outcome="death",
        primary_exposure="lact_max",
        id_columns=("stay_id",),
        outcome_columns=("death",),
        trajectory_binding=trajectory_binding,
    )
    context_path = run_dir / "research_context.json"
    context_path.write_text(context.model_dump_json(indent=2), encoding="utf-8")
    evidence = EvidenceStore(run_dir)
    evidence.register_file(
        kind="log",
        description="Frozen trajectory research context.",
        source_path=context_path,
        evidence_id="research_context",
        producer="pipeline",
        generation_mode="system",
    )
    return context_path, evidence
