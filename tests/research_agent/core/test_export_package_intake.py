"""Characterize the manifest-authoritative EasyICU export intake boundary."""

from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from easyicu.research_agent.intake import export_package as intake


def _write_frame(path: Path, frame: pd.DataFrame, export_format: str) -> None:
    if export_format == "parquet":
        frame.to_parquet(path, index=False)
    elif export_format == "csv":
        frame.to_csv(path, index=False)
    elif export_format == "excel":
        frame.to_excel(path, index=False, sheet_name="EasyICU export")
    else:  # pragma: no cover - test helper contract
        raise AssertionError(export_format)


def _native_package(
    root: Path,
    *,
    export_format: str = "parquet",
    include_feature_definitions: bool = False,
    database: str = "miiv",
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    extension = {"parquet": "parquet", "csv": "csv", "excel": "xlsx"}[export_format]
    files: list[dict[str, Any]] = []
    selection: dict[str, list[str]] = {}
    frames = {
        "labs": pd.DataFrame(
            {
                "stay_id": [1, 2],
                "charttime": [1.0, 2.0],
                "lact": [1.2, 3.4],
            }
        ),
        "outcomes": pd.DataFrame({"stay_id": [1, 2], "death": [0, 1]}),
    }
    concepts_by_module = {"labs": ["lact"], "outcomes": ["death"]}
    for module, frame in frames.items():
        file_name = f"{module}.{extension}"
        _write_frame(root / file_name, frame, export_format)
        concepts = concepts_by_module[module]
        files.append(
            {
                "file": file_name,
                "module": module,
                "concepts": len(concepts),
                "concept_ids": concepts,
                "rows": len(frame),
            }
        )
        selection[module] = concepts

    if include_feature_definitions:
        records = [
            {
                "database": database,
                "module": module,
                "concept_id": concept,
                "name_en": concept,
                "unit": "mmol/L" if concept == "lact" else "1",
                "source": {
                    "export_files": [f"{module}.{extension}"],
                    "raw_tables": [],
                    "raw_columns": [],
                },
                "callback": {"import_path": "easyicu.api.load_concepts"},
            }
            for module, concepts in selection.items()
            for concept in concepts
        ]
        definitions = {
            "schema_version": intake.FEATURE_DEFINITION_SCHEMA,
            "database": database,
            "record_count": len(records),
            "records": records,
        }
        (root / "feature_definitions.json").write_text(
            json.dumps(definitions), encoding="utf-8"
        )
        pd.DataFrame(
            [
                {
                    "database": row["database"],
                    "module": row["module"],
                    "concept_id": row["concept_id"],
                }
                for row in records
            ]
        ).to_csv(root / "feature_definitions.csv", index=False)
        feature_descriptor: dict[str, Any] = {
            "included": True,
            "schema_version": intake.FEATURE_DEFINITION_SCHEMA,
            "record_count": len(records),
            "files": [
                {
                    "file": "feature_definitions.json",
                    "kind": "feature_definitions",
                    "records": len(records),
                },
                {
                    "file": "feature_definitions.csv",
                    "kind": "feature_definitions_csv",
                    "records": len(records),
                },
            ],
        }
    else:
        feature_descriptor = {"included": False}

    manifest = {
        "database": database,
        "format": export_format,
        "concept_selection": {"mode": "explicit", "modules": selection},
        "files": files,
        "feature_definitions": feature_descriptor,
    }
    (root / intake.NATIVE_MANIFEST).write_text(json.dumps(manifest), encoding="utf-8")
    return root


@pytest.mark.parametrize("export_format", ["parquet", "csv", "excel"])
def test_native_export_round_trip_across_supported_formats(
    tmp_path: Path, export_format: str
) -> None:
    root = _native_package(tmp_path / export_format, export_format=export_format)

    package = intake.open_export_package(root)
    assert package.manifest_kind == "native"
    assert package.export_format == export_format
    assert set(package.concept_index) == {"lact", "death"}

    frame = intake.read_exported_concept(root, "lact")
    assert frame.columns.tolist() == ["stay_id", "charttime", "lact"]
    assert frame["lact"].tolist() == pytest.approx([1.2, 3.4])


def test_projected_read_uses_verified_immutable_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _native_package(tmp_path / "snapshot", export_format="csv")
    package = intake.open_export_package(root)
    path = root / "labs.csv"
    original_bytes = path.read_bytes()
    forged_bytes = original_bytes.replace(b"1.2", b"9.2")
    assert len(forged_bytes) == len(original_bytes)
    original_stat = path.stat()
    original_read_csv = intake.pd.read_csv

    def mutate_live_source_during_parse(source, *args, **kwargs):
        path.write_bytes(forged_bytes)
        os.utime(path, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
        try:
            return original_read_csv(source, *args, **kwargs)
        finally:
            path.write_bytes(original_bytes)
            os.utime(path, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))

    monkeypatch.setattr(intake.pd, "read_csv", mutate_live_source_during_parse)
    frame = intake.read_exported_concept(package, "lact")

    assert frame["lact"].tolist() == pytest.approx([1.2, 3.4])


def test_package_reuses_one_physical_snapshot_across_concept_reads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    temporary_file_calls = 0
    original_temporary_file = intake.tempfile.TemporaryFile

    def counted_temporary_file(*args, **kwargs):
        nonlocal temporary_file_calls
        temporary_file_calls += 1
        return original_temporary_file(*args, **kwargs)

    monkeypatch.setattr(intake.tempfile, "TemporaryFile", counted_temporary_file)
    root = _native_package(tmp_path / "snapshot-reuse", export_format="csv")
    package = intake.open_export_package(root)
    calls_after_open = temporary_file_calls

    for _ in range(4):
        frame = intake.read_exported_concept(package, "lact")
        assert frame["lact"].tolist() == pytest.approx([1.2, 3.4])

    assert temporary_file_calls == calls_after_open
    package.close()


@pytest.mark.parametrize("export_format", ["csv", "excel"])
def test_row_oriented_snapshot_is_parsed_once_per_package_session(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    export_format: str,
) -> None:
    root = _native_package(
        tmp_path / f"parsed-cache-{export_format}",
        export_format=export_format,
    )
    suffix = "csv" if export_format == "csv" else "xlsx"
    labs_path = root / f"labs.{suffix}"
    labs = (
        pd.read_csv(labs_path)
        if export_format == "csv"
        else pd.read_excel(labs_path, sheet_name="EasyICU export")
    )
    labs["bun"] = [12.0, 18.0]
    _write_frame(labs_path, labs, export_format)
    manifest_path = root / intake.NATIVE_MANIFEST
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["concept_selection"]["modules"]["labs"].append("bun")
    labs_entry = next(item for item in manifest["files"] if item["module"] == "labs")
    labs_entry["concepts"] = 2
    labs_entry["concept_ids"].append("bun")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    package = intake.open_export_package(root)
    parser_name = "read_csv" if export_format == "csv" else "read_excel"
    original_parser = getattr(intake.pd, parser_name)
    parse_calls = 0

    def counted_parser(*args, **kwargs):
        nonlocal parse_calls
        parse_calls += 1
        return original_parser(*args, **kwargs)

    monkeypatch.setattr(intake.pd, parser_name, counted_parser)
    try:
        lact = intake.read_exported_concept(package, "lact")
        bun = intake.read_exported_concept(package, "bun")
        lact_again = intake.read_exported_concept(package, "lact")
        lact.iloc[0, lact.columns.get_loc("lact")] = 999.0

        assert parse_calls == 1
        assert bun["bun"].tolist() == pytest.approx([12.0, 18.0])
        assert lact_again["lact"].tolist() == pytest.approx([1.2, 3.4])
    finally:
        package.close()


def test_concurrent_reads_serialize_the_retained_snapshot_cursor(
    tmp_path: Path,
) -> None:
    root = _native_package(tmp_path / "snapshot-concurrent", export_format="csv")
    with intake.open_export_package(root) as package:
        with ThreadPoolExecutor(max_workers=4) as executor:
            results = list(
                executor.map(
                    lambda _index: intake.read_exported_concept(package, "lact"),
                    range(8),
                )
            )

    assert all(frame["lact"].tolist() == pytest.approx([1.2, 3.4]) for frame in results)


def test_package_close_is_idempotent_and_reads_fail_closed(tmp_path: Path) -> None:
    root = _native_package(tmp_path / "snapshot-close")
    package = intake.open_export_package(root)

    package.close()
    package.close()

    with pytest.raises(intake.ExportPackageError) as exc_info:
        intake.read_exported_concept(package, "lact")
    assert exc_info.value.code == "export_snapshot_closed"


def test_retained_snapshot_reader_cannot_poison_authority_bytes(tmp_path: Path) -> None:
    root = _native_package(tmp_path / "snapshot-read-only", export_format="csv")
    with intake.open_export_package(root) as package:
        physical = next(item for item in package.files if "lact" in item.columns)
        with physical._snapshot.reader() as reader:
            assert not reader.writable()
            with pytest.raises((AttributeError, OSError)):
                reader.write(b"forged")
            with pytest.raises(OSError):
                reader.fileno()

        frame = intake.read_exported_concept(package, "lact")

    assert frame["lact"].tolist() == pytest.approx([1.2, 3.4])


def test_failed_package_build_closes_already_retained_snapshots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _native_package(tmp_path / "snapshot-build-failure")
    (root / "outcomes.parquet").write_bytes(b"not parquet")
    closed_members: list[str] = []
    original_close = intake._RetainedVerifiedSnapshot.close

    def recording_close(snapshot) -> None:
        closed_members.append(snapshot.member)
        original_close(snapshot)

    monkeypatch.setattr(intake._RetainedVerifiedSnapshot, "close", recording_close)

    with pytest.raises(intake.ExportPackageError):
        intake.open_export_package(root)

    assert "labs.parquet" in closed_members


def test_path_form_concept_read_closes_its_temporary_package(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _native_package(tmp_path / "snapshot-path-lifecycle")
    closed_members: list[str] = []
    original_close = intake._RetainedVerifiedSnapshot.close

    def recording_close(snapshot) -> None:
        if snapshot.member not in closed_members:
            closed_members.append(snapshot.member)
        original_close(snapshot)

    monkeypatch.setattr(intake._RetainedVerifiedSnapshot, "close", recording_close)

    frame = intake.read_exported_concept(root, "lact")

    assert frame["lact"].tolist() == pytest.approx([1.2, 3.4])
    assert set(closed_members) == {"labs.parquet", "outcomes.parquet"}


def test_manifest_inventory_is_the_only_data_authority(tmp_path: Path) -> None:
    root = _native_package(tmp_path / "export")
    pd.DataFrame({"stay_id": [1], "forged": [999]}).to_parquet(
        root / "stale_unlisted.parquet", index=False
    )

    index = intake.index_export_package(root)
    assert set(index) == {"lact", "death"}
    with pytest.raises(KeyError):
        intake.read_exported_concept(root, "forged")


def test_native_concept_selection_does_not_authorize_extra_physical_columns(
    tmp_path: Path,
) -> None:
    root = _native_package(tmp_path / "column_authority")
    labs_path = root / "labs.parquet"
    labs = pd.read_parquet(labs_path)
    labs["forged_outcome"] = [1, 1]
    labs.to_parquet(labs_path, index=False)

    package = intake.open_export_package(root)
    frame = intake.read_exported_concept(
        package, "lact", extra_columns=["forged_outcome"]
    )

    assert "forged_outcome" not in package.concept_index
    assert "forged_outcome" not in frame.columns


def test_native_requested_but_unproduced_concept_is_recorded_not_authorized(
    tmp_path: Path,
) -> None:
    root = _native_package(tmp_path / "partially-produced")
    manifest_path = root / intake.NATIVE_MANIFEST
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["concept_selection"]["modules"]["labs"].append("troponin")
    labs_entry = next(entry for entry in manifest["files"] if entry["module"] == "labs")
    labs_entry["concept_ids"].append("troponin")
    labs_entry["concepts"] += 1
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    package = intake.open_export_package(root)

    assert package.missing_selected_concepts == ("troponin",)
    assert "troponin" not in package.concept_index
    with pytest.raises(KeyError):
        intake.read_exported_concept(package, "troponin")


def test_manifest_parser_ignores_a_forged_separate_read_window(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _native_package(tmp_path / "manifest_snapshot")
    forged_file = root / "forged.parquet"
    pd.DataFrame({"stay_id": [1, 2], "forged": [9, 9]}).to_parquet(
        forged_file, index=False
    )
    manifest_path = root / intake.NATIVE_MANIFEST
    forged_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    forged_manifest["files"].append(
        {
            "file": forged_file.name,
            "module": "labs",
            "concepts": 1,
            "concept_ids": ["forged"],
            "rows": 2,
        }
    )
    forged_manifest["concept_selection"]["modules"]["labs"].append("forged")
    forged_bytes = json.dumps(forged_manifest).encode("utf-8")
    original_read_bytes = Path.read_bytes

    def forged_separate_read(path: Path) -> bytes:
        if path == manifest_path:
            return forged_bytes
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", forged_separate_read)
    package = intake.open_export_package(root)

    assert set(package.concept_index) == {"lact", "death"}


def test_directory_without_manifest_is_not_an_export_package(tmp_path: Path) -> None:
    pd.DataFrame({"stay_id": [1], "lact": [2.0]}).to_parquet(
        tmp_path / "labs.parquet", index=False
    )
    assert not intake.is_export_package(tmp_path)
    with pytest.raises(intake.ExportPackageError) as exc_info:
        intake.open_export_package(tmp_path)
    assert exc_info.value.code == "manifest_marker_missing"


def test_native_marker_has_priority_and_never_falls_back_to_valid_legacy(
    tmp_path: Path,
) -> None:
    root = tmp_path / "mixed"
    root.mkdir()
    pd.DataFrame({"stay_id": [1], "lact": [2.0]}).to_parquet(
        root / "labs.parquet", index=False
    )
    (root / intake.LEGACY_MANIFEST).write_text(
        json.dumps(
            {
                "database": "miiv",
                "export_format": "parquet",
                "modules": [{"file": "labs.parquet", "module": "labs"}],
            }
        ),
        encoding="utf-8",
    )
    (root / intake.NATIVE_MANIFEST).write_text("{not-json", encoding="utf-8")

    with pytest.raises(intake.ExportPackageError) as exc_info:
        intake.open_export_package(root)
    assert exc_info.value.code == "export_manifest_json_invalid"


def test_legacy_modules_and_exported_files_must_agree(tmp_path: Path) -> None:
    root = tmp_path / "legacy"
    root.mkdir()
    pd.DataFrame({"stay_id": [1], "lact": [2.0]}).to_parquet(
        root / "labs.parquet", index=False
    )
    (root / intake.LEGACY_MANIFEST).write_text(
        json.dumps(
            {
                "database": "miiv",
                "export_format": "parquet",
                "modules": [{"file": "labs.parquet", "module": "labs"}],
                "exported_files": ["labs.parquet"],
            }
        ),
        encoding="utf-8",
    )
    assert set(intake.index_export_package(root)) == {"lact"}

    manifest = json.loads((root / intake.LEGACY_MANIFEST).read_text())
    manifest["exported_files"] = ["other.parquet"]
    (root / intake.LEGACY_MANIFEST).write_text(json.dumps(manifest))
    with pytest.raises(intake.ExportPackageError) as exc_info:
        intake.open_export_package(root)
    assert exc_info.value.code == "manifest_inventory_mismatch"


def test_feature_definition_authority_is_loaded_and_digest_bound(
    tmp_path: Path,
) -> None:
    root = _native_package(tmp_path / "export", include_feature_definitions=True)

    package = intake.open_export_package(root)
    assert set(package.feature_definitions) == {"lact", "death"}
    assert package.feature_definitions["lact"]["unit"] == "mmol/L"
    assert package.feature_definitions_sha256


@pytest.mark.parametrize(
    ("mutation", "expected_code"),
    [
        ("count", "feature_definition_count_mismatch"),
        ("source", "feature_definition_source_mismatch"),
        ("missing_csv", "manifest_file_missing"),
    ],
)
def test_feature_definition_contract_fails_closed(
    tmp_path: Path, mutation: str, expected_code: str
) -> None:
    root = _native_package(tmp_path / mutation, include_feature_definitions=True)
    if mutation == "count":
        manifest = json.loads((root / intake.NATIVE_MANIFEST).read_text())
        manifest["feature_definitions"]["record_count"] = 99
        (root / intake.NATIVE_MANIFEST).write_text(json.dumps(manifest))
    elif mutation == "source":
        payload = json.loads((root / "feature_definitions.json").read_text())
        payload["records"][0]["source"]["export_files"] = ["unlisted.csv"]
        (root / "feature_definitions.json").write_text(json.dumps(payload))
    else:
        (root / "feature_definitions.csv").unlink()

    with pytest.raises(intake.ExportPackageError) as exc_info:
        intake.open_export_package(root)
    assert exc_info.value.code == expected_code


def test_feature_definitions_marked_not_included_are_not_discovered(
    tmp_path: Path,
) -> None:
    root = _native_package(tmp_path / "export")
    (root / "feature_definitions.json").write_text("{not-json")
    package = intake.open_export_package(root)
    assert dict(package.feature_definitions) == {}
    assert package.feature_definitions_sha256 is None


@pytest.mark.parametrize(
    "member", ["../outside.parquet", "/tmp/outside.parquet", "nested\\outside.parquet"]
)
def test_manifest_rejects_path_escape(tmp_path: Path, member: str) -> None:
    root = _native_package(tmp_path / "export")
    manifest = json.loads((root / intake.NATIVE_MANIFEST).read_text())
    manifest["files"][0]["file"] = member
    (root / intake.NATIVE_MANIFEST).write_text(json.dumps(manifest))

    with pytest.raises(intake.ExportPackageError) as exc_info:
        intake.open_export_package(root)
    assert exc_info.value.code == "manifest_path_escape"


def test_manifest_rejects_symlink_member(tmp_path: Path) -> None:
    root = _native_package(tmp_path / "export")
    outside = tmp_path / "outside.parquet"
    pd.DataFrame({"stay_id": [1], "lact": [7.0]}).to_parquet(outside, index=False)
    (root / "linked.parquet").symlink_to(outside)
    manifest = json.loads((root / intake.NATIVE_MANIFEST).read_text())
    manifest["files"][0]["file"] = "linked.parquet"
    (root / intake.NATIVE_MANIFEST).write_text(json.dumps(manifest))

    with pytest.raises(intake.ExportPackageError) as exc_info:
        intake.open_export_package(root)
    assert exc_info.value.code == "manifest_file_symlink"


def test_duplicate_physical_concept_is_rejected(tmp_path: Path) -> None:
    root = _native_package(tmp_path / "export")
    duplicate = pd.DataFrame({"stay_id": [1, 2], "lact": [5.0, 6.0]})
    duplicate.to_parquet(root / "duplicate.parquet", index=False)
    manifest = json.loads((root / intake.NATIVE_MANIFEST).read_text())
    manifest["files"].append(
        {
            "file": "duplicate.parquet",
            "module": "duplicate",
            "concepts": 1,
            "concept_ids": ["lact"],
            "rows": 2,
        }
    )
    manifest["concept_selection"]["modules"]["duplicate"] = ["lact"]
    (root / intake.NATIVE_MANIFEST).write_text(json.dumps(manifest))

    with pytest.raises(intake.ExportPackageError):
        intake.open_export_package(root)


def test_retained_snapshot_never_reads_post_open_mutation(tmp_path: Path) -> None:
    root = _native_package(tmp_path / "export")
    package = intake.open_export_package(root)
    physical = next(item for item in package.files if "lact" in item.columns)
    pd.DataFrame(
        {"stay_id": [1, 2], "charttime": [1.0, 2.0], "lact": [9.0, 9.0]}
    ).to_parquet(physical.path, index=False)

    frame = intake._read_projected(physical, ["stay_id", "lact"])
    assert frame["lact"].tolist() == pytest.approx([1.2, 3.4])
    with pytest.raises(intake.ExportPackageError) as exc_info:
        intake.verify_export_package(package)
    assert exc_info.value.code == "export_package_authority_changed"
    package.close()


def test_retained_snapshot_ignores_same_size_same_mtime_csv_mutation(
    tmp_path: Path,
) -> None:
    root = _native_package(tmp_path / "digest", export_format="csv")
    package = intake.open_export_package(root)
    physical = next(item for item in package.files if "lact" in item.columns)
    original_stat = physical.path.stat()
    original = physical.path.read_bytes()
    forged = original.replace(b"1.2", b"9.2", 1)
    assert len(forged) == len(original)
    physical.path.write_bytes(forged)
    os.utime(
        physical.path,
        ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns),
    )

    frame = intake._read_projected(physical, [physical.id_column, "lact"])
    assert frame["lact"].tolist() == pytest.approx([1.2, 3.4])
    with pytest.raises(intake.ExportPackageError) as exc_info:
        intake.verify_export_package(package)
    assert exc_info.value.code == "export_package_authority_changed"
    package.close()


@pytest.mark.parametrize(
    ("database", "id_column"),
    [
        ("eicu", "patientunitstayid"),
        ("aumc", "admissionid"),
        ("hirid", "patientid"),
        ("sic", "CaseID"),
    ],
)
def test_native_cross_database_ids_are_structural_and_normalized(
    tmp_path: Path, database: str, id_column: str
) -> None:
    root = tmp_path / database
    root.mkdir()
    pd.DataFrame({id_column: [1], "age": [60]}).to_parquet(
        root / "demographics.parquet", index=False
    )
    pd.DataFrame({id_column: [1], "death": [0]}).to_parquet(
        root / "outcomes.parquet", index=False
    )
    (root / intake.NATIVE_MANIFEST).write_text(
        json.dumps(
            {
                "database": database,
                "format": "parquet",
                "concept_selection": {
                    "modules": {"demographics": ["age"], "outcomes": ["death"]}
                },
                "files": [
                    {
                        "file": "demographics.parquet",
                        "module": "demographics",
                        "concepts": 1,
                        "concept_ids": ["age"],
                        "rows": 1,
                    },
                    {
                        "file": "outcomes.parquet",
                        "module": "outcomes",
                        "concepts": 1,
                        "concept_ids": ["death"],
                        "rows": 1,
                    },
                ],
                "feature_definitions": {"included": False},
            }
        )
    )

    package = intake.open_export_package(root)
    assert set(package.concept_index) == {"age", "death"}
    assert {item.id_column for item in package.files} == {id_column}
    frame = intake.read_exported_concept(package, "age")
    assert frame.columns.tolist() == ["stay_id", "age"]
    assert frame["stay_id"].tolist() == [1]


def test_raw_cross_database_time_is_preserved_until_typed_metadata_projection(
    tmp_path: Path,
) -> None:
    root = tmp_path / "eicu-time"
    root.mkdir()
    pd.DataFrame(
        {"patientunitstayid": [1], "observationoffset": [120], "lact": [2.5]}
    ).to_parquet(root / "labs.parquet", index=False)
    (root / intake.NATIVE_MANIFEST).write_text(
        json.dumps(
            {
                "database": "eicu",
                "format": "parquet",
                "concept_selection": {"modules": {"labs": ["lact"]}},
                "files": [
                    {
                        "file": "labs.parquet",
                        "module": "labs",
                        "concepts": 1,
                        "concept_ids": ["lact"],
                        "rows": 1,
                    }
                ],
                "feature_definitions": {"included": False},
            }
        )
    )

    package = intake.open_export_package(root)
    assert package.files[0].time_column == "observationoffset"
    frame = intake.read_exported_concept(package, "lact")
    assert frame.columns.tolist() == ["stay_id", "observationoffset", "lact"]
    assert "charttime" not in frame


def test_interval_time_columns_are_structural_and_require_typed_projection(
    tmp_path: Path,
) -> None:
    root = tmp_path / "eicu-interval-time"
    root.mkdir()
    pd.DataFrame(
        {
            "patientunitstayid": [1],
            "drugstartoffset": [60],
            "drugstopoffset": [180],
            "vaso_ind": [1],
        }
    ).to_parquet(root / "vasopressors.parquet", index=False)
    (root / intake.NATIVE_MANIFEST).write_text(
        json.dumps(
            {
                "database": "eicu",
                "format": "parquet",
                "concept_selection": {"modules": {"vasopressors": ["vaso_ind"]}},
                "files": [
                    {
                        "file": "vasopressors.parquet",
                        "module": "vasopressors",
                        "concepts": 1,
                        "concept_ids": ["vaso_ind"],
                        "rows": 1,
                    }
                ],
                "feature_definitions": {"included": False},
            }
        )
    )

    package = intake.open_export_package(root)
    assert package.files[0].time_columns == (
        "drugstartoffset",
        "drugstopoffset",
    )
    assert "drugstartoffset" not in package.concept_index
    assert "drugstopoffset" not in package.concept_index
    with pytest.raises(intake.ExportPackageError) as exc_info:
        intake.require_canonical_time_projection(package, "vaso_ind")
    assert exc_info.value.code == "export_time_projection_required"


def test_composite_concept_id_need_not_equal_physical_output_column(
    tmp_path: Path,
) -> None:
    root = tmp_path / "composite"
    root.mkdir()
    pd.DataFrame({"stay_id": [1], "sep3_sofa2": [True]}).to_parquet(
        root / "scores.parquet", index=False
    )
    record = {
        "database": "miiv",
        "module": "scores",
        "concept_id": "sep3",
        "source": {"export_files": ["scores.parquet"]},
    }
    (root / "feature_definitions.json").write_text(
        json.dumps(
            {
                "schema_version": intake.FEATURE_DEFINITION_SCHEMA,
                "database": "miiv",
                "record_count": 1,
                "records": [record],
            }
        )
    )
    (root / "feature_definitions.csv").write_text("concept_id\nsep3\n")
    (root / intake.NATIVE_MANIFEST).write_text(
        json.dumps(
            {
                "database": "miiv",
                "format": "parquet",
                "concept_selection": {"modules": {"scores": ["sep3"]}},
                "files": [
                    {
                        "file": "scores.parquet",
                        "module": "scores",
                        "concepts": 1,
                        "concept_ids": ["sep3"],
                        "rows": 1,
                    }
                ],
                "feature_definitions": {
                    "included": True,
                    "schema_version": intake.FEATURE_DEFINITION_SCHEMA,
                    "record_count": 1,
                    "files": [
                        {
                            "file": "feature_definitions.json",
                            "kind": "feature_definitions",
                            "records": 1,
                        },
                        {
                            "file": "feature_definitions.csv",
                            "kind": "feature_definitions_csv",
                            "records": 1,
                        },
                    ],
                },
            }
        )
    )

    package = intake.open_export_package(root)
    assert set(package.concept_index) == {"sep3_sofa2"}
    assert set(package.feature_definitions) == {"sep3"}
    assert (
        intake.resolve_exported_concept(package.concept_index, "sep3") == "sep3_sofa2"
    )


@pytest.mark.parametrize("marker_kind", ["directory", "symlink"])
def test_present_invalid_native_marker_never_falls_back_to_legacy(
    tmp_path: Path, marker_kind: str
) -> None:
    root = tmp_path / marker_kind
    root.mkdir()
    pd.DataFrame({"stay_id": [1], "lact": [2.0]}).to_parquet(
        root / "labs.parquet", index=False
    )
    (root / intake.LEGACY_MANIFEST).write_text(
        json.dumps(
            {
                "database": "miiv",
                "export_format": "parquet",
                "exported_files": ["labs.parquet"],
            }
        )
    )
    native = root / intake.NATIVE_MANIFEST
    if marker_kind == "directory":
        native.mkdir()
    else:
        native.symlink_to(tmp_path / "missing-native-manifest.json")

    with pytest.raises(intake.ExportPackageError) as exc_info:
        intake.open_export_package(root)
    assert exc_info.value.code == "manifest_marker_invalid"


@pytest.mark.parametrize(
    ("mutation", "expected_code"),
    [
        ("format", "manifest_format_mismatch"),
        ("rows", "manifest_row_count_mismatch"),
        ("rows_bool", "manifest_row_count_invalid"),
        ("concept_count", "manifest_concept_count_mismatch"),
        ("concept_float", "manifest_concept_count_mismatch"),
        ("selection", "manifest_concept_selection_mismatch"),
    ],
)
def test_native_manifest_internal_mismatch_fails_closed(
    tmp_path: Path, mutation: str, expected_code: str
) -> None:
    root = _native_package(tmp_path / mutation)
    manifest = json.loads((root / intake.NATIVE_MANIFEST).read_text())
    if mutation == "format":
        manifest["format"] = "csv"
    elif mutation == "rows":
        manifest["files"][0]["rows"] = 999
    elif mutation == "rows_bool":
        manifest["files"][0]["rows"] = True
    elif mutation == "concept_count":
        manifest["files"][0]["concepts"] = 2
    elif mutation == "concept_float":
        manifest["files"][0]["concepts"] = 1.9
    else:
        manifest["concept_selection"]["modules"]["labs"] = ["lact", "map"]
    (root / intake.NATIVE_MANIFEST).write_text(json.dumps(manifest))

    with pytest.raises(intake.ExportPackageError) as exc_info:
        intake.open_export_package(root)
    assert exc_info.value.code == expected_code


def test_csv_duplicate_header_fails_before_pandas_mangles_it(tmp_path: Path) -> None:
    root = tmp_path / "duplicate-header"
    root.mkdir()
    (root / "labs.csv").write_text("stay_id,lact,lact\n1,1.0,2.0\n")
    (root / intake.NATIVE_MANIFEST).write_text(
        json.dumps(
            {
                "database": "miiv",
                "format": "csv",
                "concept_selection": {"modules": {"labs": ["lact"]}},
                "files": [
                    {
                        "file": "labs.csv",
                        "module": "labs",
                        "concepts": 1,
                        "concept_ids": ["lact"],
                        "rows": 1,
                    }
                ],
                "feature_definitions": {"included": False},
            }
        )
    )
    with pytest.raises(intake.ExportPackageError) as exc_info:
        intake.open_export_package(root)
    assert exc_info.value.code == "file_schema_invalid"


def test_excel_reads_the_only_nonempty_sheet_not_the_first_sheet(
    tmp_path: Path,
) -> None:
    root = tmp_path / "excel-sheet"
    root.mkdir()
    path = root / "labs.xlsx"
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        pd.DataFrame().to_excel(writer, sheet_name="empty", index=False)
        pd.DataFrame({"stay_id": [1], "lact": [2.5]}).to_excel(
            writer, sheet_name="data", index=False
        )
    (root / intake.NATIVE_MANIFEST).write_text(
        json.dumps(
            {
                "database": "miiv",
                "format": "excel",
                "concept_selection": {"modules": {"labs": ["lact"]}},
                "files": [
                    {
                        "file": "labs.xlsx",
                        "module": "labs",
                        "concepts": 1,
                        "concept_ids": ["lact"],
                        "rows": 1,
                    }
                ],
                "feature_definitions": {"included": False},
            }
        )
    )

    package = intake.open_export_package(root)
    assert package.files[0].excel_sheet == "data"
    assert intake.read_exported_concept(root, "lact")["lact"].tolist() == [2.5]


@pytest.mark.parametrize("mutation", ["database", "module", "double_json"])
def test_feature_definition_identity_binding_fails_closed(
    tmp_path: Path, mutation: str
) -> None:
    root = _native_package(tmp_path / mutation, include_feature_definitions=True)
    if mutation == "double_json":
        (root / "other.json").write_text("{}")
        manifest = json.loads((root / intake.NATIVE_MANIFEST).read_text())
        manifest["feature_definitions"]["files"].append(
            {
                "file": "other.json",
                "kind": "feature_definitions",
                "records": 2,
            }
        )
        (root / intake.NATIVE_MANIFEST).write_text(json.dumps(manifest))
    else:
        payload = json.loads((root / "feature_definitions.json").read_text())
        if mutation == "database":
            payload["records"][0]["database"] = "eicu"
        else:
            payload["records"][0]["module"] = "outcomes"
        (root / "feature_definitions.json").write_text(json.dumps(payload))

    with pytest.raises(intake.ExportPackageError):
        intake.open_export_package(root)
