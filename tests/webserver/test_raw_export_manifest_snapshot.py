"""Raw source binding hashes the exact export manifest it parsed."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pandas as pd
import pytest

from easyicu.research_agent.authority.filesystem import AnchoredDirectory
from easyicu.webserver import raw_source_authority as owner


def _export(tmp_path: Path) -> tuple[Path, Path]:
    raw = tmp_path / "raw"
    raw.mkdir()
    pd.DataFrame({"patientunitstayid": [1, 2], "hospitaldischargestatus": ["Alive", "Expired"], "hospitaldischargeoffset": [100, 200]}).to_parquet(raw / "patient.parquet", index=False)
    export = tmp_path / "export"
    export.mkdir()
    manifest = export / "_manifest.json"
    manifest.write_text(json.dumps({"schema_version": "easyicu_native_export_v2", "database": "eicu_demo", "data_path": str(raw)}))
    return export, manifest


def test_source_binding_digest_is_computed_from_the_parsed_manifest(tmp_path: Path):
    export, manifest = _export(tmp_path)
    binding = owner.resolve_manifest_raw_source_binding(export_path=export, database="eicu_demo")
    assert binding is not None
    assert binding.export_manifest_sha256 == hashlib.sha256(manifest.read_bytes()).hexdigest()
    assert binding.source_root == (tmp_path / "raw").resolve()


def test_manifest_replaced_while_source_tables_are_hashed_is_refused(tmp_path: Path, monkeypatch):
    export, manifest = _export(tmp_path)
    original_binding = owner._table_schema_and_digest
    replacement = json.loads(manifest.read_text())
    replacement["database"] = "miiv"
    replaced = False

    def competing_write(path, *, table):
        nonlocal replaced
        if path.name == "patient.parquet" and not replaced:
            replaced = True
            candidate = export / "replacement.json"
            candidate.write_text(json.dumps(replacement))
            os.replace(candidate, manifest)
        return original_binding(path, table=table)

    monkeypatch.setattr(owner, "_table_schema_and_digest", competing_write)
    with pytest.raises(owner.RawSourceAuthorityError, match="manifest changed") as exc:
        owner.resolve_manifest_raw_source_binding(export_path=export, database="eicu_demo")
    assert replaced
    assert exc.value.code == "raw_source_authority_export_manifest_changed"


@pytest.mark.parametrize("change", ["in_place", "replace"])
def test_mutation_during_manifest_read_is_refused(tmp_path: Path, monkeypatch, change):
    export, manifest = _export(tmp_path)
    original_open = AnchoredDirectory.open_regular
    mutated = False

    class ConcurrentReader:
        def __init__(self, wrapped):
            self.wrapped = wrapped
        def __enter__(self):
            return self
        def __exit__(self, *args):
            self.wrapped.close()
        def fileno(self):
            return self.wrapped.fileno()
        def seek(self, *args):
            return self.wrapped.seek(*args)
        def read(self, *args):
            nonlocal mutated
            content = self.wrapped.read(*args)
            if not mutated:
                mutated = True
                changed = content.replace(b'eicu_demo', b'miiv_demo')
                assert changed != content and len(changed) == len(content)
                if change == "in_place":
                    manifest.write_bytes(changed)
                else:
                    new = export / "replacement.json"
                    new.write_bytes(changed)
                    os.replace(new, manifest)
            return content

    def open_with_competitor(directory, name):
        handle = original_open(directory, name)
        return ConcurrentReader(handle) if name == "_manifest.json" else handle

    monkeypatch.setattr(AnchoredDirectory, "open_regular", open_with_competitor)
    with pytest.raises(owner.RawSourceAuthorityError) as exc:
        owner._read_export_manifest(export)
    assert mutated
    assert exc.value.code == "raw_source_authority_export_manifest_unreadable"


def test_schema_and_digest_are_bound_to_same_unchanged_table(tmp_path: Path):
    export, _ = _export(tmp_path)
    patient = tmp_path / "raw" / "patient.parquet"
    binding = owner.resolve_manifest_raw_source_binding(export_path=export, database="eicu_demo")
    assert binding is not None
    assert binding.table_sha256["patient"] == hashlib.sha256(patient.read_bytes()).hexdigest()
    assert "patientunitstayid" in binding.table_columns["patient"]


@pytest.mark.parametrize("change", ["in_place", "replace"])
def test_same_size_table_change_during_binding_is_refused(tmp_path: Path, monkeypatch, change):
    export, _ = _export(tmp_path)
    patient = tmp_path / "raw" / "patient.parquet"
    original_read_schema = owner.pq.read_schema
    mutated = False

    def competing_schema_read(source, *args, **kwargs):
        nonlocal mutated
        result = original_read_schema(source, *args, **kwargs)
        if not mutated:
            mutated = True
            content = patient.read_bytes()
            if change == "in_place":
                # Preserve byte length, but change a byte after schema reading.
                with patient.open("r+b") as target:
                    target.seek(5)
                    target.write(bytes([content[5] ^ 1]))
                    target.flush()
                    os.fsync(target.fileno())
            else:
                # Even replacement with identical bytes changes the named entry.
                replacement = patient.with_name("replacement.parquet")
                replacement.write_bytes(content)
                os.replace(replacement, patient)
        return result

    monkeypatch.setattr(owner.pq, "read_schema", competing_schema_read)
    with pytest.raises(owner.RawSourceAuthorityError) as exc:
        owner.resolve_manifest_raw_source_binding(export_path=export, database="eicu_demo")
    assert mutated
    assert exc.value.code == "raw_source_authority_table_changed"


def test_legacy_authority_preserves_digest_mismatch_before_schema_error(tmp_path: Path):
    from tests.webserver.test_webserver_raw_source_authority import _authority_environment

    environ, export = _authority_environment(tmp_path)
    (tmp_path / "raw" / "icu" / "icustays.parquet").write_bytes(b"not a parquet file")
    with pytest.raises(owner.RawSourceAuthorityError) as exc:
        owner.resolve_raw_mimic_iv_source_binding(export_path=export, database="miiv", environ=environ)
    assert exc.value.code == "raw_source_authority_table_digest_mismatch"
    assert exc.value.details == {"object": "icustays", "database": "mimic_iv"}
