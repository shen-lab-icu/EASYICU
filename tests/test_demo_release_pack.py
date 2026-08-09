from __future__ import annotations

import json
from pathlib import Path
import zipfile

from easyicu import demo_release_pack
from easyicu.webserver import demo_source_storage
from easyicu.webserver.demo_source_contracts import DemoSourcePaths, DemoSourceSpec


def _source(size_bytes: int) -> DemoSourceSpec:
    return DemoSourceSpec(
        id="fixture_demo",
        title="Fixture ICU Demo",
        version="1.0",
        database="miiv",
        description="fixture",
        scope_summary="one stay",
        patients=1,
        icu_stays="1",
        size_label=f"{size_bytes} bytes",
        size_bytes=size_bytes,
        download_url="https://physionet.org/static/published-projects/fixture/demo.zip",
        landing_page="https://physionet.org/content/fixture/1.0/",
        archive_filename="demo.zip",
        attribution="Fixture authors. Fixture ICU Demo. PhysioNet.",
        citation_url="https://doi.org/10.0000/fixture",
        license_url="https://physionet.org/content/fixture/view-license/1.0/",
        max_download_bytes=1024 * 1024,
        max_uncompressed_bytes=1024 * 1024,
    )


def _paths(root: Path) -> DemoSourcePaths:
    raw = root / "raw"
    export = root / "export"
    return DemoSourcePaths(
        root=root,
        archive=root / "demo.zip",
        raw=raw,
        export=export,
        extracted_marker=raw / ".easyicu-demo-extracted.json",
        converted_marker=raw / ".easyicu-demo-converted.json",
        prepared_marker=export / ".easyicu-demo-prepared.json",
    )


def test_release_pack_sanitizes_paths_and_preserves_license(
    tmp_path: Path, monkeypatch
) -> None:
    paths = _paths(tmp_path / "cache")
    paths.root.mkdir(parents=True)
    with zipfile.ZipFile(paths.archive, "w") as archive:
        archive.writestr("LICENSE.txt", "ODbL")
    source = _source(paths.archive.stat().st_size)
    paths.raw.mkdir()
    (paths.raw / "LICENSE.txt").write_text(
        "Open Database License v1.0", encoding="utf-8"
    )
    paths.export.mkdir()
    manifest = {
        "database": "miiv",
        "data_path": "/Users/fixture/private/raw",
        "export_folder": {"path": "/Users/fixture/private/export"},
        "files": [],
        "concept_availability": {"structurally_unavailable": []},
    }
    (paths.export / "_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (paths.export / "README.md").write_text(
        "# Export\n- Source path: `/Users/fixture/private/raw`\n",
        encoding="utf-8",
    )
    demo_source_storage.write_marker(
        paths.prepared_marker,
        source,
        archive_sha256="a" * 64,
        export={"file_count": 0, "total_rows": 0},
    )
    monkeypatch.setattr(
        demo_release_pack,
        "_release_export_description",
        lambda _root: {"database": "miiv", "files": []},
    )

    receipt = demo_release_pack.build_release_pack(source, paths, tmp_path / "dist")

    pack = Path(receipt.prepared_pack_path)
    with zipfile.ZipFile(pack) as archive:
        names = archive.namelist()
        assert any(name.endswith("/LICENSE.txt") for name in names)
        assert any(name.endswith("/SHA256SUMS") for name in names)
        manifest_name = next(
            name for name in names if name.endswith("/export/_manifest.json")
        )
        packed_manifest = archive.read(manifest_name).decode()
        assert "/Users/" not in packed_manifest
        assert "<official-demo-cache>/raw" in packed_manifest
    assert receipt.feature_summary["definitions"] == 288
