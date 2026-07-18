"""Typed-v2 column metadata is exact, digest-bound, and v1 compatible."""

from __future__ import annotations

import json
from dataclasses import replace
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
    DerivationWindow,
    TimeCoordinate,
    parse_column_metadata_sidecar,
    write_content_addressed_sidecar,
)
from easyicu.resources import load_dictionary
from easyicu.research_agent.intake import export_package as intake


def _write_frame(path: Path, frame: pd.DataFrame, export_format: str) -> None:
    if export_format == "parquet":
        frame.to_parquet(path, index=False)
    elif export_format == "csv":
        frame.to_csv(path, index=False)
    else:
        frame.to_excel(path, index=False, sheet_name="EasyICU export")


def _binding(
    concept: str,
    column: str,
    role: ConceptColumnRole,
    *,
    aggregation: str | None = None,
    window: DerivationWindow | None = None,
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
                aggregation=aggregation,
            ),
            source_database="miiv",
        ),
        derivation_window=window,
    )


def _v2_package(root: Path, *, export_format: str = "parquet") -> Path:
    root.mkdir(parents=True)
    extension = {"parquet": "parquet", "csv": "csv", "excel": "xlsx"}[export_format]
    labs_name = f"labs.{extension}"
    outcomes_name = f"outcomes.{extension}"
    labs = pd.DataFrame(
        {
            "stay_id": [1, 2],
            "charttime": [1.0, 2.0],
            "lact": [1.2, 3.4],
            "lact_n": [1, 1],
            "lact_measured": [True, True],
            "lact_mean_6h": [1.2, 3.4],
            "lact_noise": [999.0, 999.0],
        }
    )
    outcomes = pd.DataFrame({"stay_id": [1, 2], "death": [False, True]})
    _write_frame(root / labs_name, labs, export_format)
    _write_frame(root / outcomes_name, outcomes, export_format)
    six_hours = DerivationWindow(origin="icu_admission", start_hours=0.0, end_hours=6.0)
    labs_binding = ColumnMetadataFileBinding(
        relative_path=labs_name,
        module="labs",
        identity_column="stay_id",
        time_coordinates=(
            TimeCoordinate(column="charttime", origin="icu_admission", unit="h"),
        ),
        columns={
            "lact": _binding("lact", "lact", ConceptColumnRole.VALUE),
            "lact_n": _binding("lact", "lact_n", ConceptColumnRole.COUNT),
            "lact_measured": _binding(
                "lact", "lact_measured", ConceptColumnRole.MEASUREMENT_STATUS
            ),
            "lact_mean_6h": _binding(
                "lact",
                "lact_mean_6h",
                ConceptColumnRole.NUMERIC_AGGREGATE,
                aggregation="mean",
                window=six_hours,
            ),
        },
    )
    outcomes_binding = ColumnMetadataFileBinding(
        relative_path=outcomes_name,
        module="outcomes",
        identity_column="stay_id",
        time_coordinates=(),
        columns={"death": _binding("death", "death", ConceptColumnRole.EVENT_STATUS)},
    )
    sidecar = ColumnMetadataSidecar(
        source_database="miiv",
        source_database_class_prefixes=(),
        scope=EXPORT_PHYSICAL_SCOPE,
        files=(labs_binding, outcomes_binding),
    )
    reference = write_content_addressed_sidecar(root, sidecar)
    files = [
        {
            "file": labs_name,
            "module": "labs",
            "concepts": 1,
            "concept_ids": ["lact"],
            "rows": 2,
            "column_metadata_columns": list(labs_binding.columns),
        },
        {
            "file": outcomes_name,
            "module": "outcomes",
            "concepts": 1,
            "concept_ids": ["death"],
            "rows": 2,
            "column_metadata_columns": list(outcomes_binding.columns),
        },
    ]
    manifest = {
        "schema_version": intake.NATIVE_MANIFEST_SCHEMA_V2,
        "database": "miiv",
        "format": export_format,
        "concept_selection": {
            "mode": "explicit",
            "modules": {"labs": ["lact"], "outcomes": ["death"]},
        },
        "files": files,
        "feature_definitions": {"included": False},
        "column_metadata": reference.to_dict(),
    }
    (root / intake.NATIVE_MANIFEST).write_text(json.dumps(manifest), encoding="utf-8")
    return root


@pytest.mark.parametrize("export_format", ["parquet", "csv", "excel"])
def test_v2_sidecar_drives_exact_mapping_across_formats(
    tmp_path: Path, export_format: str
) -> None:
    root = _v2_package(tmp_path / export_format, export_format=export_format)

    with intake.open_export_package(root) as package:
        intake.require_column_metadata(package)
        assert package.column_metadata_sha256
        assert package.column_metadata_scope == EXPORT_PHYSICAL_SCOPE
        assert set(package.concept_index) == {
            "lact",
            "lact_n",
            "lact_measured",
            "lact_mean_6h",
            "death",
        }
        assert "lact_noise" not in package.concept_index
        assert package.concept_index["lact_n"]["source_concept"] == "lact"
        assert package.concept_index["lact_n"]["column_metadata_role"] == "count"
        assert intake.resolve_exported_concept(package.concept_index, "lact") == "lact"
        frame = intake.read_exported_concept(
            package, "lact", extra_columns=["lact_n", "lact_noise"]
        )

    assert frame.columns.tolist() == ["stay_id", "charttime", "lact", "lact_n"]
    assert frame["lact"].tolist() == pytest.approx([1.2, 3.4])


def test_v2_sidecar_tamper_never_downgrades_to_v1(tmp_path: Path) -> None:
    root = _v2_package(tmp_path / "tamper", export_format="csv")
    manifest = json.loads((root / intake.NATIVE_MANIFEST).read_text())
    sidecar_path = root / manifest["column_metadata"]["file"]
    raw = sidecar_path.read_bytes()
    sidecar_path.write_bytes(raw.replace(b'"lact"', b'"fact"', 1))

    with pytest.raises(intake.ExportPackageError) as exc_info:
        intake.open_export_package(root)
    assert exc_info.value.code == "column_metadata_digest_mismatch"


@pytest.mark.parametrize("mutation", ["null", "missing", "partial_v1"])
def test_native_schema_prevents_partial_sidecar_downgrade(
    tmp_path: Path, mutation: str
) -> None:
    root = _v2_package(tmp_path / mutation, export_format="csv")
    manifest_path = root / intake.NATIVE_MANIFEST
    manifest = json.loads(manifest_path.read_text())
    if mutation == "null":
        manifest["column_metadata"] = None
    elif mutation == "missing":
        del manifest["column_metadata"]
    else:
        del manifest["schema_version"]
        del manifest["column_metadata"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(intake.ExportPackageError) as exc_info:
        intake.open_export_package(root)
    assert exc_info.value.code == "column_metadata_inventory_invalid"


def test_native_v2_markers_cannot_be_fully_stripped_around_orphan_sidecar(
    tmp_path: Path,
) -> None:
    root = _v2_package(tmp_path / "full_strip", export_format="csv")
    manifest_path = root / intake.NATIVE_MANIFEST
    manifest = json.loads(manifest_path.read_text())
    del manifest["schema_version"]
    del manifest["column_metadata"]
    for entry in manifest["files"]:
        del entry["column_metadata_columns"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(intake.ExportPackageError) as exc_info:
        intake.open_export_package(root)
    assert exc_info.value.code == "column_metadata_inventory_invalid"


def test_v2_manifest_coverage_must_equal_sidecar(tmp_path: Path) -> None:
    root = _v2_package(tmp_path / "coverage", export_format="csv")
    manifest_path = root / intake.NATIVE_MANIFEST
    manifest = json.loads(manifest_path.read_text())
    manifest["files"][0]["column_metadata_columns"].remove("lact_n")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(intake.ExportPackageError) as exc_info:
        intake.open_export_package(root)
    assert exc_info.value.code == "column_metadata_column_mismatch"


@pytest.mark.parametrize(
    ("mutation", "expected_code"),
    [
        ("module", "column_metadata_column_mismatch"),
        ("identity", "column_metadata_column_mismatch"),
        ("time", "column_metadata_column_mismatch"),
        ("source", "column_metadata_source_mismatch"),
    ],
)
def test_v2_sidecar_coordinates_must_match_manifest_and_selection(
    tmp_path: Path,
    mutation: str,
    expected_code: str,
) -> None:
    root = _v2_package(tmp_path / mutation, export_format="csv")
    manifest_path = root / intake.NATIVE_MANIFEST
    manifest = json.loads(manifest_path.read_text())
    sidecar_path = root / manifest["column_metadata"]["file"]
    sidecar = parse_column_metadata_sidecar(sidecar_path.read_bytes())
    labs = sidecar.files[0]
    if mutation == "module":
        labs = replace(labs, module="outcomes")
    elif mutation == "identity":
        labs = replace(labs, identity_column="patient_id")
    elif mutation == "time":
        labs = replace(labs, time_coordinates=())
    else:
        labs = replace(
            labs,
            columns={
                **dict(labs.columns),
                "lact": _binding("death", "lact", ConceptColumnRole.EVENT_STATUS),
            },
        )
    updated = replace(sidecar, files=(labs, sidecar.files[1]))
    reference = write_content_addressed_sidecar(root, updated)
    manifest["column_metadata"] = reference.to_dict()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(intake.ExportPackageError) as exc_info:
        intake.open_export_package(root)
    assert exc_info.value.code == expected_code


@pytest.mark.parametrize("descriptor_mutation", ["extra", "bad_filename"])
def test_v2_sidecar_descriptor_is_exact_and_digest_named(
    tmp_path: Path, descriptor_mutation: str
) -> None:
    root = _v2_package(tmp_path / descriptor_mutation, export_format="csv")
    manifest_path = root / intake.NATIVE_MANIFEST
    manifest = json.loads(manifest_path.read_text())
    if descriptor_mutation == "extra":
        manifest["column_metadata"]["unexpected"] = True
        expected_code = "column_metadata_schema_invalid"
    else:
        manifest["column_metadata"]["file"] = "column_metadata.json"
        expected_code = "column_metadata_digest_mismatch"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(intake.ExportPackageError) as exc_info:
        intake.open_export_package(root)
    assert exc_info.value.code == expected_code


@pytest.mark.parametrize(
    "role", ["count", "measurement_status", "first_observation_time", "event_fraction"]
)
def test_typed_resolver_never_promotes_a_nonprimary_companion(role: str) -> None:
    index = {
        "lact_companion": {
            "column_metadata_v2": True,
            "source_concept": "lact",
            "column_metadata_role": role,
        }
    }

    assert intake.resolve_exported_concept(index, "lact") is None


def test_require_column_metadata_rejects_selected_concept_without_primary_binding(
    tmp_path: Path,
) -> None:
    root = _v2_package(tmp_path / "missing_primary", export_format="csv")
    manifest_path = root / intake.NATIVE_MANIFEST
    manifest = json.loads(manifest_path.read_text())
    sidecar_path = root / manifest["column_metadata"]["file"]
    sidecar = parse_column_metadata_sidecar(sidecar_path.read_bytes())
    labs = sidecar.files[0]
    companions_only = replace(
        labs,
        columns={
            name: binding for name, binding in labs.columns.items() if name != "lact"
        },
    )
    updated = replace(sidecar, files=(companions_only, sidecar.files[1]))
    reference = write_content_addressed_sidecar(root, updated)
    manifest["column_metadata"] = reference.to_dict()
    manifest["files"][0]["column_metadata_columns"] = list(companions_only.columns)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with intake.open_export_package(root) as package:
        assert package.missing_selected_concepts == ("lact",)
        with pytest.raises(intake.ExportPackageError) as exc_info:
            intake.require_column_metadata(package)
    assert exc_info.value.code == "column_metadata_required"


def test_v2_rejects_auxiliary_identifier_bound_as_analysis_value(
    tmp_path: Path,
) -> None:
    root = _v2_package(tmp_path / "aux_id", export_format="csv")
    manifest_path = root / intake.NATIVE_MANIFEST
    manifest = json.loads(manifest_path.read_text())
    sidecar_path = root / manifest["column_metadata"]["file"]
    sidecar = parse_column_metadata_sidecar(sidecar_path.read_bytes())
    labs = sidecar.files[0]
    frame = pd.read_csv(root / labs.relative_path)
    frame["subject_id"] = [101, 102]
    frame.to_csv(root / labs.relative_path, index=False)
    forged = replace(
        labs,
        columns={
            **dict(labs.columns),
            "subject_id": _binding("lact", "subject_id", ConceptColumnRole.VALUE),
        },
    )
    updated = replace(sidecar, files=(forged, sidecar.files[1]))
    reference = write_content_addressed_sidecar(root, updated)
    manifest["column_metadata"] = reference.to_dict()
    manifest["files"][0]["column_metadata_columns"] = list(forged.columns)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(intake.ExportPackageError) as exc_info:
        intake.open_export_package(root)
    assert exc_info.value.code == "column_metadata_column_mismatch"


def test_canonical_time_consumer_rejects_non_icu_hour_coordinates(
    tmp_path: Path,
) -> None:
    root = _v2_package(tmp_path / "time_semantics", export_format="csv")
    manifest_path = root / intake.NATIVE_MANIFEST
    manifest = json.loads(manifest_path.read_text())
    sidecar_path = root / manifest["column_metadata"]["file"]
    sidecar = parse_column_metadata_sidecar(sidecar_path.read_bytes())
    labs = replace(
        sidecar.files[0],
        time_coordinates=(
            TimeCoordinate("charttime", "database_native_absolute", "timestamp"),
        ),
    )
    updated = replace(sidecar, files=(labs, sidecar.files[1]))
    reference = write_content_addressed_sidecar(root, updated)
    manifest["column_metadata"] = reference.to_dict()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with intake.open_export_package(root) as package:
        with pytest.raises(intake.ExportPackageError) as exc_info:
            intake.require_canonical_time_projection(package, "lact")
    assert exc_info.value.code == "export_time_projection_required"


def test_v2_explicit_alias_uses_source_owner_not_prefix_guess(tmp_path: Path) -> None:
    root = _v2_package(tmp_path / "alias", export_format="csv")
    manifest_path = root / intake.NATIVE_MANIFEST
    manifest = json.loads(manifest_path.read_text())
    sidecar_path = root / manifest["column_metadata"]["file"]
    sidecar = parse_column_metadata_sidecar(sidecar_path.read_bytes())
    labs = sidecar.files[0]
    alias_frame = pd.read_csv(root / labs.relative_path)
    alias_frame["sep3_sofa2"] = [0, 1]
    alias_frame.to_csv(root / labs.relative_path, index=False)
    alias_binding = replace(
        labs,
        columns={
            **dict(labs.columns),
            "sep3_sofa2": _binding("sep3", "sep3_sofa2", ConceptColumnRole.VALUE),
        },
    )
    updated = replace(sidecar, files=(alias_binding, sidecar.files[1]))
    reference = write_content_addressed_sidecar(root, updated)
    manifest["column_metadata"] = reference.to_dict()
    manifest["files"][0]["concept_ids"].append("sep3")
    manifest["files"][0]["concepts"] = 2
    manifest["files"][0]["column_metadata_columns"] = list(alias_binding.columns)
    manifest["concept_selection"]["modules"]["labs"].append("sep3")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with intake.open_export_package(root) as package:
        assert (
            intake.resolve_exported_concept(package.concept_index, "sep3")
            == "sep3_sofa2"
        )
        assert "lact_noise" not in package.concept_index


def test_v1_package_keeps_legacy_authority_shape(tmp_path: Path) -> None:
    root = tmp_path / "v1"
    root.mkdir()
    pd.DataFrame(
        {"stay_id": [1, 2], "charttime": [1.0, 2.0], "lact": [1.2, 3.4]}
    ).to_csv(root / "labs.csv", index=False)
    manifest = {
        "database": "miiv",
        "format": "csv",
        "concept_selection": {"mode": "explicit", "modules": {"labs": ["lact"]}},
        "files": [
            {
                "file": "labs.csv",
                "module": "labs",
                "concepts": 1,
                "concept_ids": ["lact"],
                "rows": 2,
            }
        ],
        "feature_definitions": {"included": False},
    }
    (root / intake.NATIVE_MANIFEST).write_text(json.dumps(manifest), encoding="utf-8")

    with intake.open_export_package(root) as package:
        assert package.column_metadata_sha256 is None
        assert dict(package.column_metadata_by_file) == {}
        assert package.authority_sha256 == (
            "3bb5a8405efef01ad2365b1f36a3bdf10b467d60e07915460440ffe5b2ea2f5b"
        )
