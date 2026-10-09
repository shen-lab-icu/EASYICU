"""A small typed native export with a stay-level outcome module.

A native export publishes its outcome module as stay-level rows at one
coordinate, 0 h from ICU admission, and types each file with the module
exporter's own binder.  Test modules may not import one another
(``tests/governance/test_test_organization.py``); these helpers moved here from
``test_an_outcome_is_timed_by_the_time_its_export_issues`` when a second module
needed them.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from easyicu.concept.export_metadata import build_export_file_metadata_binding
from easyicu.concept.metadata_sidecar import (
    EXPORT_PHYSICAL_SCOPE,
    ColumnMetadataFileBinding,
    ColumnMetadataSidecar,
    write_content_addressed_sidecar,
)
from easyicu.research_agent.intake import export_package as intake
from easyicu.resources import load_dictionary


def file_binding(
    *, relative_path: str, module: str, frame: pd.DataFrame, concepts: list[str]
) -> ColumnMetadataFileBinding:
    """Type one physical file with the module exporter's own binder."""

    return build_export_file_metadata_binding(
        relative_path=relative_path,
        module=module,
        frame=frame,
        concept_ids=concepts,
        database="miiv",
        database_class_prefixes=(),
        dictionary=load_dictionary(include_sofa2=True),
    )


def typed_native_export(
    root: Path,
    *,
    outcome: pd.DataFrame,
    outcome_concepts: list[str],
    longitudinal: pd.DataFrame | None = None,
    longitudinal_concepts: list[str] | None = None,
    statics: pd.DataFrame | None = None,
) -> Path:
    """Write the export; ``statics`` holds each stay's age, else one per index."""

    root.mkdir()
    if statics is None:
        stays = sorted(int(value) for value in outcome["stay_id"].unique())
        statics = pd.DataFrame(
            {"stay_id": stays, "age": [50 + 5 * index for index in range(len(stays))]}
        )
    members: list[tuple[str, str, pd.DataFrame, list[str]]] = [
        ("demographics.parquet", "demographics", statics, ["age"]),
        ("outcome.parquet", "outcome", outcome, list(outcome_concepts)),
    ]
    if longitudinal is not None:
        members.append(
            (
                "medications.parquet",
                "medications",
                longitudinal,
                list(longitudinal_concepts or ()),
            )
        )
    bindings = []
    files = []
    for relative_path, module, frame, concepts in members:
        frame.to_parquet(root / relative_path, index=False)
        binding = file_binding(
            relative_path=relative_path, module=module, frame=frame, concepts=concepts
        )
        bindings.append(binding)
        files.append(
            {
                "file": relative_path,
                "module": module,
                "concepts": len(concepts),
                "concept_ids": concepts,
                "rows": len(frame),
                "column_metadata_columns": list(binding.columns),
            }
        )
    sidecar = ColumnMetadataSidecar(
        source_database="miiv",
        source_database_class_prefixes=(),
        scope=EXPORT_PHYSICAL_SCOPE,
        files=tuple(bindings),
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
                    "modules": {item["module"]: item["concept_ids"] for item in files},
                },
                "files": files,
                "feature_definitions": {"included": False},
                "column_metadata": reference.to_dict(),
            }
        ),
        encoding="utf-8",
    )
    return root


def untyped_native_export(
    root: Path,
    *,
    outcome: pd.DataFrame,
    outcome_concepts: list[str],
    unrecorded_stays: tuple[int, ...] = (),
    longitudinal: pd.DataFrame | None = None,
    longitudinal_concepts: list[str] | None = None,
) -> Path:
    """The same export as an older native export writes it: no column types.

    ``unrecorded_stays`` are stays the outcome module has no row for.
    """

    root.mkdir()
    stays = sorted(
        {*(int(value) for value in outcome["stay_id"].unique()), *unrecorded_stays}
    )
    statics = pd.DataFrame(
        {"stay_id": stays, "age": [50 + 5 * index for index in range(len(stays))]}
    )
    members: list[tuple[str, str, pd.DataFrame, list[str]]] = [
        ("demographics.parquet", "demographics", statics, ["age"]),
        ("outcome.parquet", "outcome", outcome, list(outcome_concepts)),
    ]
    if longitudinal is not None:
        members.append(
            (
                "medications.parquet",
                "medications",
                longitudinal,
                list(longitudinal_concepts or ()),
            )
        )
    files = []
    for relative_path, module, frame, concepts in members:
        frame.to_parquet(root / relative_path, index=False)
        files.append(
            {
                "file": relative_path,
                "module": module,
                "concepts": len(concepts),
                "concept_ids": concepts,
                "rows": len(frame),
            }
        )
    (root / intake.NATIVE_MANIFEST).write_text(
        json.dumps(
            {
                "database": "miiv",
                "format": "parquet",
                "concept_selection": {
                    "mode": "explicit",
                    "modules": {item["module"]: item["concept_ids"] for item in files},
                },
                "files": files,
            }
        ),
        encoding="utf-8",
    )
    return root


def native_outcome(**columns: list) -> pd.DataFrame:
    """Stay-level outcome rows exactly as the native publisher writes them."""

    stays = [1, 2, 3, 4]
    frame = pd.DataFrame({"stay_id": stays, "charttime": [0.0] * len(stays)})
    for name, values in columns.items():
        frame[name] = values
    return frame
