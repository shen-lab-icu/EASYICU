"""Delta60s: bind resume to files/edges and share provenance resolution.

Synthetic sentinel files only; no Parquet decoder or extraction is needed.
Each regression includes its healthy control before exercising the defect.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from unittest.mock import Mock

import pytest

from easyicu import storage_layout as layout
from easyicu import storage_seal_publication as publication


def _json(path, **fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "schema_version": layout.CONVERSION_MANIFEST_SCHEMA, "tables": [], **fields,
    }))
    return path


def _binding(path):
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def _driver():
    path = Path(__file__).resolve().parents[2] / "tools/reextract_native_export_v2.py"
    spec = importlib.util.spec_from_file_location("closure_identity_driver", path)
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)
    return driver


@pytest.mark.parametrize("change", [
    "retarget", "swap", "touch", "replace_inode", "resize", "intermediate_hop",
])
def test_completed_export_resume_binds_files_and_each_link_hop(tmp_path, monkeypatch, change):
    driver = _driver()
    install = tmp_path / "install"
    install.mkdir()
    target = tmp_path / "target"
    target.mkdir()
    first, second = target / "a.parquet", target / "b.parquet"
    first.write_bytes(b"synthetic-a")
    second.write_bytes(b"synthetic-b")
    middle = target / "middle.parquet"
    middle.symlink_to(first.name)
    link = install / "sample.parquet"
    link.symlink_to(middle)
    other = install / "other.parquet"
    other.symlink_to(second)
    out = tmp_path / "output"
    out.mkdir()
    args = driver._parse_args([
        "--output-root", str(out), "--databases", "miiv",
        "--data-path", f"miiv={install}", "--resume",
    ])
    paths = {"miiv": str(install)}
    identity = {"repository_root": str(driver.REPOSITORY_ROOT), "commit": "a" * 40}
    manifest = driver._new_run_manifest(
        databases=["miiv"], data_paths=paths, identity=identity,
        monitoring={}, memory={}, args=args,
    )
    manifest["sources"] = {"miiv": {"status": "complete"}}
    manifest["runtime_provenance"] = {
        "seal_publication_closure": driver._source_publication_closure(paths),
    }
    ledger = out / "run_manifest.json"
    ledger.write_text(json.dumps(manifest))
    assert driver._load_resume_manifest(
        run_root=out, databases=["miiv"], data_paths=paths,
        identity=identity, resource_policy=args.resource_policy,
    ) == manifest
    before = ledger.read_bytes()

    if change in {"retarget", "swap", "intermediate_hop"}:
        link.unlink()
        link.symlink_to(first if change == "intermediate_hop" else second)
        if change == "swap":
            other.unlink()
            other.symlink_to(middle)
    elif change == "touch":
        info = first.stat()
        os.utime(first, ns=(info.st_atime_ns, info.st_mtime_ns + 1_000_000_000))
    elif change == "replace_inode":
        # Preserve size and mtime; only the final inode changes.
        replacement = target / "replacement"
        replacement.write_bytes(first.read_bytes())
        info = first.stat()
        os.utime(replacement, ns=(info.st_atime_ns, info.st_mtime_ns))
        replacement.replace(first)
        assert first.stat().st_ino != info.st_ino
    else:
        # Also cover a regular file reached by scanning, without a file link.
        info = second.stat()
        second.write_bytes(b"synthetic-size-changed")
        os.utime(second, ns=(info.st_atime_ns, info.st_mtime_ns))

    monkeypatch.setattr(driver, "_git_identity", lambda: identity)
    monkeypatch.setattr(driver, "_require_clean_identity", lambda _: None)
    monkeypatch.setattr(driver, "_load_psutil", lambda _: (None, {}))
    monkeypatch.setattr(driver, "_detect_effective_memory", lambda _: {})
    reuse = Mock(side_effect=AssertionError("completed exports must not be reused"))
    monkeypatch.setattr(driver, "_pending_databases", reuse)
    with pytest.raises(publication.SealPublicationError) as caught:
        driver.run(args)
    assert caught.value.reason == publication.UNACCEPTED
    reuse.assert_not_called()
    assert ledger.read_bytes() == before


def test_configured_roots_are_part_of_the_identity_even_with_the_same_reachable_set(tmp_path):
    table = tmp_path / "sample.parquet"
    table.write_bytes(b"synthetic")
    root = publication.assert_roots_published([tmp_path])
    explicit = publication.assert_roots_published([tmp_path, table])
    assert root.install_count == explicit.install_count
    assert root.sha256 != explicit.sha256
    assert explicit == publication.assert_roots_published([table, tmp_path, table])


def test_relative_binding_does_not_visit_the_unrelated_metadata_relative_guess(
    tmp_path, monkeypatch,
):
    source = _json(tmp_path / "source/conversion_manifest.json")
    work = tmp_path / "work"
    work.mkdir()
    install = tmp_path / "inst/install"
    derived = {**_binding(source), "path": "../source/conversion_manifest.json"}
    _json(install / layout.CONVERSION_MANIFEST, derived_from=derived)
    guess = tmp_path / "inst/source"
    _json(guess / layout.CONVERSION_MANIFEST)
    marker = guess.parent / f".{guess.name}{publication.PENDING_SUFFIX}"
    marker.write_text("{}")
    monkeypatch.chdir(work)
    layout.validate_conversion_manifest(install)
    closure = publication.assert_roots_published([install])
    marker.unlink()
    assert closure == publication.assert_roots_published([install])


@pytest.mark.parametrize("kind", ["relative", "absolute", "bucket_files"])
def test_reader_and_gate_use_storage_layout_source_resolver(tmp_path, monkeypatch, kind):
    install = tmp_path / "install"
    calls = []
    if kind == "bucket_files":
        source = install / "sample.parquet"
        source.parent.mkdir()
        source.write_bytes(b"synthetic")
        bucket = install / "sample_bucket"
        info = source.stat()
        payload = {"derived_from": {"files": [{
            "path": source.name, "size": info.st_size, "mtime_ns": info.st_mtime_ns,
        }]}}
        _json(bucket / layout.MANIFEST, **payload)
        expected_raw, expected_bucket = source.name, bucket

        def read():
            layout._verify_bucket_sources(bucket, payload)
    else:
        source = _json(tmp_path / "source/conversion_manifest.json")
        monkeypatch.chdir(tmp_path)
        raw = str(source.relative_to(tmp_path)) if kind == "relative" else str(source)
        derived = {**_binding(source), "path": raw}
        _json(install / layout.CONVERSION_MANIFEST, derived_from=derived)
        expected_raw, expected_bucket = raw, None

        def read():
            layout.validate_conversion_manifest(install)

    def resolve(raw, *, bucket_root=None):
        calls.append((raw, bucket_root))
        return source

    # raising=False lets BASE_TIP reach the behavioral assertion instead of
    # failing collection solely because the new shared function is absent.
    monkeypatch.setattr(layout, "resolve_derived_from_source", resolve, raising=False)
    read()
    assert calls and set(calls) == {(expected_raw, expected_bucket)}
    calls.clear()
    publication.assert_roots_published([install])
    assert calls and set(calls) == {(expected_raw, expected_bucket)}


@pytest.mark.parametrize("basename", ["provenance.json", "arbitrary.parquet"])
def test_arbitrary_provenance_role_follows_chain_without_opening_derived_files(
    tmp_path, basename,
):
    end = _json(tmp_path / "end/conversion_manifest.json")
    metadata = _json(tmp_path / "middle" / basename, derived_from=_binding(end))
    # Real-file identity dedup must terminate a hard-linked metadata cycle.
    alias = metadata.with_name("alias.json")
    alias.hardlink_to(metadata)
    end = _json(end, derived_from=_binding(alias))
    install = tmp_path / "install"
    _json(install / layout.CONVERSION_MANIFEST, derived_from=_binding(metadata))
    poison = _json(tmp_path / "poison/conversion_manifest.json")
    (poison.parent.parent / f".poison{publication.PENDING_SUFFIX}").write_text("{}")
    table = install / "sample.parquet"
    # If the gate parses this table as metadata it will reach the poison root.
    _json(table, derived_from=_binding(poison))
    _json(install / "sample_bucket" / layout.MANIFEST,
          derived_from={"files": [{"path": table.name}]})

    active = True
    opened = []
    metadata_identity = metadata.stat().st_ino

    def audit(event, args):
        if active and event == "open" and isinstance(args[0], (str, bytes, os.PathLike)):
            path = Path(os.fsdecode(args[0]))
            assert path != table, "derived_from.files must never open table content"
            if path in {metadata, alias}:
                opened.append(path.stat().st_ino)

    sys.addaudithook(audit)
    try:
        publication.assert_roots_published([install])  # accepted chain
        accepted_opens = list(opened)
        opened.clear()
        (end.parent.parent / f".end{publication.PENDING_SUFFIX}").write_text("{}")
        with pytest.raises(publication.SealPublicationError) as caught:
            publication.assert_roots_published([install])
        assert caught.value.reason == publication.UNPUBLISHED
        assert caught.value.install == end.parent
        assert accepted_opens == [metadata_identity]
    finally:
        active = False
