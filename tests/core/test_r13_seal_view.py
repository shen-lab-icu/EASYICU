"""R13-SEAL-VIEW: a sealed root that mirrors a linked installation stays readable.

The production SIC/eICU/AUMC roots link their installation's conversion manifest; the seal writes an
owned copy, which moves the data root to the sealed root. Its mirrored links into the installation
must then be admitted through the declared view the copy names, never refused as reads outside the
data root, and only at the installation's declared table and size.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from easyicu.storage_layout import TableScanError, admit_consumed_realpath
from tools.seal_buckets import seal_rebuild as sr


def _installation(tmp_path: Path) -> tuple[Path, Path]:
    raw = tmp_path / "databases" / "raw_db"
    raw.mkdir(parents=True)
    (raw / "cases.parquet").write_bytes(b"PAR1-cases-PAR1")
    (raw / "other.parquet").write_bytes(b"PAR1-other-longer-PAR1")
    manifest = {
        "database": "raw_db",
        "data_path": str(raw),
        "tables": [
            {"table": "cases", "outputs": [{"relative_path": "cases.parquet", "size_bytes": 15}]},
            {"table": "other", "outputs": [{"relative_path": "other.parquet", "size_bytes": 22}]},
        ],
    }
    (raw / sr.CONVERSION_MANIFEST).write_text(json.dumps(manifest))
    current = tmp_path / "b6" / "raw_cur"
    current.mkdir(parents=True)
    for name in ("cases.parquet", "other.parquet", sr.CONVERSION_MANIFEST):
        os.symlink(raw / name, current / name)
    return raw, current


def _sealed(tmp_path: Path, current: Path) -> Path:
    sealed = tmp_path / "b6" / "raw_cur_sabc"
    sealed.mkdir()
    pinned = sr.manifest_source(current / sr.CONVERSION_MANIFEST)
    for name in ("cases.parquet", "other.parquet"):
        os.symlink(os.readlink(current / name), sealed / name)
    sr.write_conversion_manifest(
        current / sr.CONVERSION_MANIFEST, sealed / sr.CONVERSION_MANIFEST,
        current, sealed, pinned,
    )
    return sealed


def test_the_current_linked_root_is_read_through_its_installation(tmp_path):
    raw, current = _installation(tmp_path)
    got = admit_consumed_realpath(current, "cases", current / "cases.parquet")
    assert got["admitted_by"] == "data_root"
    assert got["target_realpath"] == os.path.realpath(raw / "cases.parquet")


def test_a_sealed_mirror_of_a_linked_root_admits_its_links_through_the_declared_view(tmp_path):
    raw, current = _installation(tmp_path)
    sealed = _sealed(tmp_path, current)
    manifest = sealed / sr.CONVERSION_MANIFEST
    assert manifest.is_file() and not manifest.is_symlink()
    derived = json.loads(manifest.read_text())["derived_from"]
    source = os.path.realpath(raw / sr.CONVERSION_MANIFEST)
    assert derived == {
        "path": source, "sha256": hashlib.sha256(Path(source).read_bytes()).hexdigest(),
    }
    for table in ("cases", "other"):
        got = admit_consumed_realpath(sealed, table, sealed / f"{table}.parquet")
        assert got["admitted_by"] == "declared_view"
        # the realpath stays in the installation: nothing is copied into the sealed root
        assert got["target_realpath"] == os.path.realpath(raw / f"{table}.parquet")


def test_the_view_admits_a_file_only_for_its_own_table_and_size(tmp_path):
    raw, current = _installation(tmp_path)
    sealed = _sealed(tmp_path, current)
    with pytest.raises(TableScanError, match="storage_read_outside_data_root"):
        admit_consumed_realpath(sealed, "other", sealed / "cases.parquet")
    stray = tmp_path / "databases" / "stray.parquet"
    stray.write_bytes(b"PAR1-stray-PAR1")
    os.symlink(stray, sealed / "stray.parquet")
    with pytest.raises(TableScanError, match="storage_read_outside_data_root"):
        admit_consumed_realpath(sealed, "cases", sealed / "stray.parquet")


def test_a_changed_installation_manifest_drops_the_view(tmp_path):
    raw, current = _installation(tmp_path)
    sealed = _sealed(tmp_path, current)
    content = json.loads((raw / sr.CONVERSION_MANIFEST).read_text())
    content["note"] = "rebuilt"
    target = raw / sr.CONVERSION_MANIFEST
    target.chmod(0o644)
    target.write_text(json.dumps(content))
    with pytest.raises(TableScanError, match="storage_read_outside_data_root"):
        admit_consumed_realpath(sealed, "cases", sealed / "cases.parquet")


def test_an_owned_manifest_with_its_own_binding_is_not_rebound(tmp_path):
    raw, current = _installation(tmp_path)
    (current / sr.CONVERSION_MANIFEST).unlink()
    source = raw / sr.CONVERSION_MANIFEST
    own = json.loads(source.read_text())
    own["data_path"] = str(current)
    own["derived_from"] = {
        "path": str(source), "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    }
    (current / sr.CONVERSION_MANIFEST).write_text(json.dumps(own))
    sealed = _sealed(tmp_path, current)
    got = json.loads((sealed / sr.CONVERSION_MANIFEST).read_text())
    assert got["derived_from"] == own["derived_from"]
    assert got["data_path"] == str(sealed)
