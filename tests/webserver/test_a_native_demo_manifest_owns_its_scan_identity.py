"""Folder recognition reads a validated export identity before raw heuristics."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from easyicu.concept.export_metadata import build_export_file_metadata_binding
from easyicu.concept.metadata_sidecar import (
    EXPORT_PHYSICAL_SCOPE,
    ColumnMetadataSidecar,
    write_content_addressed_sidecar,
)
from easyicu.resources import load_dictionary
from easyicu.webserver import dataio


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
    # Shared eICU column names cannot distinguish a demo from the full data.
    (root / "patient.csv").write_text("patientunitstayid,uniquepid\n")
    return root, manifest


def test_a_valid_native_v2_demo_is_detected_and_named_as_the_demo(tmp_path):
    root, _ = _native_demo(tmp_path / "neutral_folder")
    result = dataio.scan_path(str(root))
    assert result["ok"] is True
    assert result["db_key"] == "eicu_demo"
    assert result["db"] == "eICU Demo v2.0.1"
    assert result["source"] == "module"
    assert result["privacy"]["raw_rows_read"] is False


def test_a_native_manifest_takes_precedence_over_an_old_legacy_marker(tmp_path):
    root, _ = _native_demo(tmp_path / "neutral_folder")
    (root / "easyicu_export_manifest.json").write_text(json.dumps({"database": "eicu"}))
    assert dataio.scan_path(str(root))["db_key"] == "eicu_demo"


def test_a_raw_eicu_folder_still_uses_schema_detection(tmp_path):
    root = tmp_path / "raw"
    root.mkdir()
    (root / "patient.csv").write_text("patientunitstayid,uniquepid\n")
    result = dataio.scan_path(str(root))
    assert result["ok"] is True
    assert result["db_key"] == "eicu"
    assert result["source"] == "raw"


@pytest.mark.parametrize(("field", "value", "code"), [
    ("format", "invalid", "manifest_format_invalid"),
    ("database", "unsupported_database", "database_detection_manifest_invalid"),
    ("column_metadata", None, "column_metadata_inventory_invalid"),
])
def test_an_invalid_manifest_cannot_assert_a_demo_identity(tmp_path, field, value, code):
    root, manifest = _native_demo(tmp_path / "invalid")
    manifest[field] = value
    (root / "_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError) as caught:
        dataio._detect_database(root)
    assert caught.value.code == code
    result = dataio.scan_path(str(root))
    assert result["ok"] is False
    assert result["error"] == code


def test_an_unreadable_manifest_member_is_a_typed_scan_failure(tmp_path):
    root, _ = _native_demo(tmp_path / "unreadable")
    (root / "demographics.parquet").write_bytes(b"not a parquet file")
    result = dataio.scan_path(str(root))
    assert result["ok"] is False
    assert result["error"] == "export_layout_invalid"
