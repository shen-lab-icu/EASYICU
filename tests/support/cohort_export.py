"""Disk-streamed, deterministic native-v2 exports for cohort parity/profiling.

All values are synthetic. Even the full 108 M-row fixture holds only one
bounded Arrow batch at a time; it never builds a full module in pandas.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

# Hospital-sized shape: 19 modules, 108 M rows, largest module 16 M.
MODULES = {
    "demographics": (200_859, ("age", "sex")),
    "outcome": (
        401_718,
        ("death", "icu_death", "mort_28d", "los_icu", "los_hosp", "followup_days_28d"),
    ),
    "sepsis3_sofa2": (401_718, ("sep3_sofa2",)),
    "sofa1_score": (8_000_000, ("sofa1",)),
    "sofa2_score": (8_000_000, ("sofa2",)),
    "vitals": (16_000_000, ("hr", "map")),
    "blood_gas": (9_000_000, ("lact",)),
    "respiratory": (8_000_000, ("pafi",)),
    "chemistry": (9_000_000, ("crea", "bun")),
    "hematology": (8_000_000, ("wbc",)),
    "renal": (6_000_000, ("aki_stage",)),
    "vasopressors": (5_000_000, ("norepi_equiv",)),
    "ventilator": (6_000_000, ("peep",)),
    "other_medications": (5_000_000, ("abx",)),
    "diagnoses": (3_000_000, ("diagnosis",)),
    "neurological": (4_000_000, ("gcs",)),
    "liver": (4_000_000, ("bili",)),
    "coagulation": (4_000_000, ("inr",)),
    "urine_output": (3_995_705, ("urine",)),
}


def write_cohort_export(
    root: Path,
    *,
    stays: int = 24,
    scale: float | None = None,
    batch_rows: int = 200_000,
    case: str = "regular",
    entity_column: str = "stay_id",
) -> dict:
    """Write Parquet batches (or one small CSV edge case) and return a descriptor.

    ``scale=1`` reproduces the full shape; the default scales row counts with
    ``stays``. Nulls, duplicate chart times, per-stay variation, absent stays,
    all-null and empty modules exercise the actual cohort reductions.
    """
    root.mkdir(parents=True, exist_ok=True)
    factor = stays / 200_859 if scale is None else scale
    files = []
    for module, (full_rows, columns) in MODULES.items():
        rows = max(stays, round(full_rows * factor))
        if module == "demographics":
            rows = stays
        if case == "empty" and module in {"sofa1_score", "blood_gas"}:
            rows = 0
        suffix = ".csv" if case == "csv" else ".parquet"
        target = root / f"{module}{suffix}"
        writer = None
        for offset in range(0, max(rows, 1), batch_rows):
            index = np.arange(offset, min(offset + batch_rows, rows), dtype=np.int64)
            sid = index % stays + 1
            if case == "missing" and module != "demographics":
                sid = np.where(sid % 4 == 0, sid + stays, sid)
            if case == "duplicates":
                # Whole duplicate rows, not just repeated stays.
                index = index // 2
                sid = index % stays + 1
            ids = pa.array(sid)
            if case == "string_ids":
                ids = pa.array(
                    [f"{value}.0" if value % 2 else f"s_{value}" for value in sid]
                )
            elif case == "float_ids":
                ids = pa.array(sid.astype(float))
            arrays = {
                entity_column: ids,
                "charttime": pa.array((index // stays // 2).astype(float)),
            }
            for column in columns:
                nulls = (index + len(column)) % 7 == 0
                if case == "all_null" and module in {
                    "blood_gas",
                    "outcome",
                    "sofa2_score",
                }:
                    nulls = np.ones(len(index), dtype=bool)
                if column == "sex":
                    values = np.where(sid % 2, "F", "M")
                elif column == "diagnosis":
                    values = np.where(sid % 2, "synthetic_alpha", "synthetic_beta")
                elif column in {"death", "icu_death", "mort_28d", "sep3_sofa2", "abx"}:
                    values = (sid + index // stays) % 5 == 0
                elif column == "age":
                    values = (18 + sid % 73).astype(float)
                elif column == "followup_days_28d":
                    values = np.where(sid % 5 == 0, 7.0, 28.0)
                else:
                    values = ((sid * 13 + index // stays * 7) % 97 / 4).astype(float)
                arrays[column] = pa.array(values, mask=nulls)
            table = pa.table(arrays)
            if suffix == ".csv":
                # CSV is only used by small parity cases.
                table.to_pandas().to_csv(
                    target,
                    mode="w" if offset == 0 else "a",
                    header=offset == 0,
                    index=False,
                )
            else:
                if writer is None:
                    writer = pq.ParquetWriter(
                        target, table.schema, compression="snappy"
                    )
                writer.write_table(table)
        if writer is not None:
            writer.close()
        files.append(
            {
                "module": module,
                "file": target.name,
                "rows": rows,
                "columns": list(arrays),
                "concept_ids": list(columns),
                "concepts": len(columns),
            }
        )
    manifest = {
        "schema_version": "easyicu_native_export_v2",
        "database": "eicu",
        "generated": "2026-01-01T00:00:00",
        "format": "csv" if case == "csv" else "parquet",
        "stays": stays,
        "files": files,
        "feature_definitions": {"included": False},
        "concept_selection": {
            "mode": "explicit",
            "modules": {name: list(spec[1]) for name, spec in MODULES.items()},
        },
    }
    _write_column_metadata(root, manifest, entity_column)
    (root / "_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return {
        "ok": True,
        "path": str(root),
        "database": "eicu",
        "generated": manifest["generated"],
        "files": files,
        "summary": {
            "stays": stays,
            "modules": len(files),
            "file_count": len(files),
            "total_rows": sum(item["rows"] for item in files),
        },
    }


def _write_column_metadata(root, manifest, entity_column):
    # Test-owned definitions explicitly carry no clinical source assertions.
    # This makes the fixture a verifiable native-v2 package, rather than just
    # labelling a legacy manifest as v2.
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
    from easyicu.concept.schema import ConceptDefinition

    bindings = []
    for item in manifest["files"]:
        columns = item["concept_ids"]
        item["column_metadata_columns"] = sorted(columns)
        bindings.append(
            ColumnMetadataFileBinding(
                relative_path=item["file"],
                module=item["module"],
                identity_column=entity_column,
                time_coordinates=(
                    TimeCoordinate(
                        column="charttime", origin="icu_admission", unit="h"
                    ),
                ),
                columns={
                    column: ColumnMetadataBinding(
                        metadata=project_concept_column_metadata(
                            ConceptDefinition(
                                name=column,
                                sources={},
                                description="Synthetic fixture only",
                            ),
                            spec=ColumnProjectionSpec(
                                column_name=column,
                                source_concept=column,
                                role=ConceptColumnRole.VALUE,
                            ),
                            source_database="eicu",
                        )
                    )
                    for column in columns
                },
            )
        )
    reference = write_content_addressed_sidecar(
        root,
        ColumnMetadataSidecar(
            source_database="eicu",
            source_database_class_prefixes=(),
            scope=EXPORT_PHYSICAL_SCOPE,
            files=tuple(bindings),
        ),
    )
    manifest["column_metadata"] = reference.to_dict()
