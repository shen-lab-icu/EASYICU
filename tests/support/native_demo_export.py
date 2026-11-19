"""A minimal validated native eICU-demo export (synthetic, two stays)."""

from __future__ import annotations

import json

import pandas as pd

from easyicu.concept.export_metadata import build_export_file_metadata_binding
from easyicu.concept.metadata_sidecar import (
    EXPORT_PHYSICAL_SCOPE,
    ColumnMetadataSidecar,
    write_content_addressed_sidecar,
)
from easyicu.resources import load_dictionary
from tests.support.published_export import record_member_digests


def _native_demo(root):
    root.mkdir()
    frame = pd.DataFrame({"patientunitstayid": [1, 2], "age": [40, 60]})
    frame.to_parquet(root / "demographics.parquet", index=False)
    binding = build_export_file_metadata_binding(
        relative_path="demographics.parquet", module="demographics", frame=frame,
        concept_ids=["age"], database="eicu_demo", database_class_prefixes=("eicu_demo", "eicu"),
        dictionary=load_dictionary(include_sofa2=True),
    )
    reference = write_content_addressed_sidecar(root, ColumnMetadataSidecar(
        source_database="eicu_demo", source_database_class_prefixes=("eicu_demo", "eicu"),
        scope=EXPORT_PHYSICAL_SCOPE, files=(binding,),
    ))
    manifest = {
        "schema_version": "easyicu_native_export_v2", "database": "eicu_demo", "format": "parquet",
        "concept_selection": {"mode": "explicit", "modules": {"demographics": ["age"]}},
        "files": [{"file": "demographics.parquet", "module": "demographics", "rows": 2,
                   "concepts": 1, "concept_ids": ["age"], "column_metadata_columns": list(binding.columns)}],
        "column_metadata": reference.to_dict(),
    }
    (root / "_manifest.json").write_text(json.dumps(manifest))
    manifest = record_member_digests(root)  # the writer records every member's digest
    # Shared eICU column names cannot distinguish a demo from the full data.
    (root / "patient.csv").write_text("patientunitstayid,uniquepid\n")
    return root, manifest
