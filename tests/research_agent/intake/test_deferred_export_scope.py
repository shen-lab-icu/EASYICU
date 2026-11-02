"""Synthetic package reads cannot exceed a host-bound deferred scope."""
from __future__ import annotations

import pytest

from easyicu.research_agent.intake import export_package as owner
from tests.support.typed_export import typed_export


def test_deferred_scope_checks_manifest_and_actual_projection(tmp_path, monkeypatch):
    export = typed_export(tmp_path / "export", database="eicu")
    with owner.open_export_package(export) as package:
        manifest = package.manifest_sha256
        columns = {item.relative_path: tuple(item.columns) for item in package.files}
    seen = []
    opened = []
    original = owner._RetainedVerifiedSnapshot.projected_frame
    original_schema = owner._schema_for_file

    def observe(self, **kwargs):
        seen.append(tuple(kwargs["columns"]))
        return original(self, **kwargs)

    monkeypatch.setattr(owner._RetainedVerifiedSnapshot, "projected_frame", observe)
    def observe_file(path, *args, **kwargs):
        opened.append(path.name)
        return original_schema(path, *args, **kwargs)

    monkeypatch.setattr(owner, "_schema_for_file", observe_file)
    with owner.host_deferred_export_read_scope(manifest_sha256=manifest, columns_by_file=columns):
        with pytest.raises(ValueError, match="scope_nested"):
            with owner.host_deferred_export_read_scope(manifest_sha256=manifest, columns_by_file=columns):
                pass
        with owner.open_export_package(export) as package:
            owner.read_exported_concept(package, "death")
    assert seen and all(set(row).issubset(set(columns["outcomes.parquet"])) for row in seen)
    assert set(opened) == set(columns)
    seen.clear()
    narrowed = {**columns, "outcomes.parquet": ("stay_id",)}
    with owner.host_deferred_export_read_scope(manifest_sha256=manifest, columns_by_file=narrowed):
        with owner.open_export_package(export) as package:
            with pytest.raises(owner.ExportPackageError, match="column scope changed"):
                owner.read_exported_concept(package, "death")
    assert not seen
    opened.clear()
    with owner.host_deferred_export_read_scope(manifest_sha256="0" * 64, columns_by_file=columns):
        with pytest.raises(owner.ExportPackageError, match="manifest changed"):
            owner.open_export_package(export)
    assert not opened
    with owner.open_export_package(export) as package:
        with owner.host_deferred_export_read_scope(manifest_sha256="0" * 64, columns_by_file=columns):
            with pytest.raises(owner.ExportPackageError, match="manifest changed"):
                owner.read_exported_concept(package, "death")
        reduced_roster = {"outcomes.parquet": columns["outcomes.parquet"]}
        with owner.host_deferred_export_read_scope(manifest_sha256=manifest, columns_by_file=reduced_roster):
            with pytest.raises(owner.ExportPackageError, match="manifest changed"):
                owner.read_exported_concept(package, "death")
