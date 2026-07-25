"""Synthetic tests for the protected full0717 schema-inventory builder."""

from __future__ import annotations

import json
import stat
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import pytest

from benchmarks.figure2_canonical9 import schema_inventory_builder as builder


def _fixture_export(tmp_path: Path) -> Path:
    root = tmp_path / "full6_20260717"
    root.mkdir()
    (root / "run_manifest.json").write_text('{"run":"synthetic"}\n', encoding="utf-8")
    for _source_id, directory in builder._SOURCE_DIRECTORIES:
        source = root / directory
        source.mkdir()
        frame = pd.DataFrame({"stay_id": [1, 2], "value": [3.0, 4.0]})
        frame.to_parquet(source / "demographics.parquet", index=False)
        (source / "demographics.manifest.json").write_text(
            json.dumps(
                {
                    "module": "demographics",
                    "saved": {
                        "demographics": {
                            "rows": 2,
                            "concepts": ["age"],
                        }
                    },
                    "errors": [],
                    "warnings": [],
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    return root


def _patch_snapshot_pin(monkeypatch: pytest.MonkeyPatch, root: Path) -> None:
    digest = builder.full_export_content_sha256(root)
    run_manifest_digest = next(
        item["sha256"]
        for item in builder.full_export_file_identities(root)
        if item["relative_path"] == "run_manifest.json"
    )
    monkeypatch.setattr(builder, "FULL0717_EXPORT_CONTENT_SHA256", digest)
    monkeypatch.setattr(builder, "FULL0717_RUN_MANIFEST_SHA256", run_manifest_digest)


def test_build_schema_inventory_is_metadata_only_and_private(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _fixture_export(tmp_path)
    _patch_snapshot_pin(monkeypatch, root)
    monkeypatch.setattr(
        pq.ParquetFile,
        "read",
        lambda *_args, **_kwargs: pytest.fail("schema inventory must not read rows"),
    )

    result = builder.build_schema_inventory(
        full_export_root=root, output_root=tmp_path / "protected-inventory"
    )

    payload = json.loads(result.inventory_path.read_text(encoding="utf-8"))
    assert payload["metadata_only"] is True
    assert payload["source_attested"] is False
    assert payload["real_run_authorized"] is False
    assert payload["requires_unlisted_member_disposition"] is False
    assert len(payload["sources"]) == 6
    assert payload["sources"][0]["modules"][0]["data_file"]["row_count"] == 2
    assert stat.S_IMODE(result.output_root.stat().st_mode) & 0o077 == 0
    assert stat.S_IMODE(result.inventory_path.stat().st_mode) & 0o077 == 0


def test_schema_inventory_records_unlisted_member_for_review(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _fixture_export(tmp_path)
    spill = root / "eicu/.easyicu_spill/intermediate.parquet"
    spill.parent.mkdir()
    pd.DataFrame({"stay_id": [1]}).to_parquet(spill, index=False)
    _patch_snapshot_pin(monkeypatch, root)

    result = builder.build_schema_inventory(
        full_export_root=root, output_root=tmp_path / "protected-inventory"
    )
    payload = json.loads(result.inventory_path.read_text(encoding="utf-8"))

    assert result.unlisted_member_count == 1
    assert payload["requires_unlisted_member_disposition"] is True
    assert payload["unlisted_members"][0]["relative_path"].endswith(
        ".easyicu_spill/intermediate.parquet"
    )


def test_schema_inventory_records_partial_manifest_without_inferring_semantics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _fixture_export(tmp_path)
    manifest_path = root / "miiv/demographics.manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["saved"]["demographics"].pop("rows")
    manifest["saved"]["demographics"].pop("concepts")
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8"
    )
    _patch_snapshot_pin(monkeypatch, root)

    result = builder.build_schema_inventory(
        full_export_root=root, output_root=tmp_path / "protected-inventory"
    )
    payload = json.loads(result.inventory_path.read_text(encoding="utf-8"))
    module = payload["sources"][0]["modules"][0]

    assert module["manifest_completeness"] == "partial_or_invalid"
    assert module["manifest_missing_fields"] == ["rows", "concepts"]
    assert module["manifest_declared_row_count"] is None
    assert module["declared_concepts"] is None


def test_schema_inventory_rejects_missing_declared_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _fixture_export(tmp_path)
    (root / "miiv/demographics.parquet").unlink()
    _patch_snapshot_pin(monkeypatch, root)

    with pytest.raises(builder.SchemaInventoryBuildError, match="declares an output"):
        builder.build_schema_inventory(
            full_export_root=root, output_root=tmp_path / "must-not-exist"
        )


def test_schema_inventory_rejects_existing_output_and_wrong_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _fixture_export(tmp_path)
    _patch_snapshot_pin(monkeypatch, root)
    existing = tmp_path / "existing"
    existing.mkdir()

    with pytest.raises(
        builder.SchemaInventoryBuildError, match="must not already exist"
    ):
        builder.build_schema_inventory(full_export_root=root, output_root=existing)

    monkeypatch.setattr(builder, "FULL0717_EXPORT_CONTENT_SHA256", "f" * 64)
    with pytest.raises(builder.SchemaInventoryBuildError, match="differs from"):
        builder.build_schema_inventory(
            full_export_root=root, output_root=tmp_path / "must-not-exist"
        )
