from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from easyicu.webserver import export_download
from easyicu.webserver.app import app


def _export(root: Path) -> dict:
    root.mkdir()
    (root / "demographics.csv").write_text("stay_id,age\na,65\n", encoding="utf-8")
    (root / "README.md").write_text("EasyICU export\n", encoding="utf-8")
    (root / "feature_definitions.json").write_text("[]\n", encoding="utf-8")
    (root / "feature_definitions.csv").write_text(
        "concept_id,module\nage,demographics\n", encoding="utf-8"
    )
    manifest = {
        "files": [{"file": "demographics.csv", "module": "demographics"}],
        "feature_definitions": {
            "included": True,
            "files": [
                {"file": "feature_definitions.json"},
                {"file": "feature_definitions.csv"},
            ],
        },
    }
    (root / "_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    return {
        "sources": [
            {
                "id": "src_0123456789ab",
                "path": str(root),
                "label": "Registered export",
                "ok": True,
            }
        ]
    }


def test_registered_export_bundle_contains_only_manifest_allowlist(tmp_path: Path) -> None:
    root = tmp_path / "export"
    registry = _export(root)
    (root / "unlisted-secret.txt").write_text("not bundled", encoding="utf-8")

    bundle = export_download.prepare_registered_export_bundle(
        "src_0123456789ab", registry=registry, temp_dir=tmp_path / "downloads"
    )
    payload = b"".join(export_download.iter_bundle_and_cleanup(bundle))

    assert not bundle.path.exists()
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        assert set(archive.namelist()) == {
            "_manifest.json",
            "README.md",
            "demographics.csv",
            "feature_definitions.json",
            "feature_definitions.csv",
        }


def test_registered_export_bundle_rejects_manifest_traversal(tmp_path: Path) -> None:
    root = tmp_path / "export"
    registry = _export(root)
    manifest = json.loads((root / "_manifest.json").read_text(encoding="utf-8"))
    manifest["files"].append({"file": "../outside.csv"})
    (root / "_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(export_download.ExportDownloadError) as blocked:
        export_download.prepare_registered_export_bundle(
            "src_0123456789ab", registry=registry, temp_dir=tmp_path / "downloads"
        )

    assert blocked.value.code == "registered_export_manifest_file_invalid"


def test_registered_export_bundle_fails_closed_above_size_limit(tmp_path: Path) -> None:
    root = tmp_path / "export"
    registry = _export(root)

    with pytest.raises(export_download.ExportDownloadError) as blocked:
        export_download.prepare_registered_export_bundle(
            "src_0123456789ab",
            registry=registry,
            temp_dir=tmp_path / "downloads",
            max_source_bytes=1,
        )

    assert blocked.value.code == "registered_export_download_too_large"


def test_registered_export_download_route_accepts_only_source_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry = _export(tmp_path / "export")
    monkeypatch.setattr(export_download.sources, "load_registry", lambda: registry)
    client = TestClient(app)

    rejected = client.post(
        "/api/workspaces/download",
        json={"source_id": "src_0123456789ab", "path": "/private/export"},
    )
    assert rejected.status_code == 400
    assert rejected.json()["detail"]["error"] == (
        "registered_export_download_arguments_invalid"
    )

    response = client.post(
        "/api/workspaces/download", json={"source_id": "src_0123456789ab"}
    )
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/zip"
    assert response.headers["x-easyicu-source-id"] == "src_0123456789ab"
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        assert "demographics.csv" in archive.namelist()
