"""A minimal typed native export shared by materialized-metadata tests.

Test modules may not import one another (``tests/governance/test_test_organization.py``);
these helpers moved here from ``test_materialized_column_metadata`` when a second
module needed a typed export.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

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
from easyicu.research_agent.intake import export_package as intake
from easyicu.resources import load_dictionary


def metadata_binding(
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


def typed_export(
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
            or metadata_binding("age", "age", roles.get("age", ConceptColumnRole.VALUE)),
            "lact": binding_overrides.get("lact")
            or metadata_binding("lact", "lact", roles.get("lact", ConceptColumnRole.VALUE)),
            "mech_vent": binding_overrides.get("mech_vent")
            or metadata_binding(
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
            or metadata_binding(
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
