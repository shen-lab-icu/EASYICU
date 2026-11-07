"""Delta60s/61s: bind resume to source files/edges in every anchor context.

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


@pytest.mark.parametrize("kind", ["relative", "absolute", "bucket_files", "bucket_source_root"])
def test_reader_and_gate_use_storage_layout_source_resolver(tmp_path, monkeypatch, kind):
    install = tmp_path / "install"
    calls = []
    expected_source_root = None
    if kind in {"bucket_files", "bucket_source_root"}:
        source = (install if kind == "bucket_files" else tmp_path / "source") / "sample.parquet"
        source.parent.mkdir()
        source.write_bytes(b"synthetic")
        bucket = install / "sample_bucket"
        info = source.stat()
        payload = {"derived_from": {"table": "sample", "files": [{
            "path": source.name, "size": info.st_size, "mtime_ns": info.st_mtime_ns,
        }]}}
        if kind == "bucket_source_root":
            expected_source_root = str(source.parent)
            payload["derived_from"]["source_root"] = expected_source_root
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

    original_resolver = layout.resolve_derived_from_source

    def resolve(raw, *, bucket_root=None, source_root=None):
        calls.append((raw, bucket_root, source_root))
        return original_resolver(raw, bucket_root=bucket_root, source_root=source_root)

    # raising=False lets BASE_TIP reach the behavioral assertion instead of
    # failing collection solely because the new shared function is absent.
    monkeypatch.setattr(layout, "resolve_derived_from_source", resolve, raising=False)
    read()
    expected_calls = {(expected_raw, expected_bucket, expected_source_root)}
    if kind in {"bucket_files", "bucket_source_root"}:
        expected_calls.add((".", expected_bucket, expected_source_root))
    assert calls and set(calls) == expected_calls
    calls.clear()
    publication.assert_roots_published([install])
    assert calls and set(calls) == expected_calls


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


def test_hardlinked_bucket_metadata_walks_each_anchor_but_parses_once(tmp_path, monkeypatch):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from types import SimpleNamespace

    import easyicu.resources as resources
    from tests.support.bucket_storage import write_contract

    buckets = [tmp_path / name / "sample_bucket" for name in ("a", "b")]
    sources = []
    for bucket in buckets:
        bucket.mkdir(parents=True)
        source = bucket.parent / "sample.parquet"
        source.write_bytes(b"synthetic-source")
        sources.append(source)
        part = bucket / "bucket_id=0/part.parquet"
        part.parent.mkdir()
        pq.write_table(pa.table({"synthetic_key": pa.array(range(16), type=pa.int64())}), part)
        write_contract(bucket, key="synthetic_key")
    info = sources[0].stat()
    remote = tmp_path / "remote"
    remote.mkdir()
    target = remote / "source.parquet"
    target.write_bytes(sources[1].read_bytes())
    os.utime(target, ns=(info.st_atime_ns, info.st_mtime_ns))
    sources[1].unlink()
    sources[1].symlink_to(target)
    first, second = [bucket / layout.MANIFEST for bucket in buckets]
    payload = json.loads(first.read_text())
    payload["derived_from"] = {"table": "sample", "files": [{
        "path": "sample.parquet", "size": info.st_size, "mtime_ns": info.st_mtime_ns,
        "sha256": hashlib.sha256(sources[0].read_bytes()).hexdigest(),
    }]}
    first.write_text(json.dumps(payload))
    second.unlink()
    second.hardlink_to(first)
    for bucket in buckets:
        layout.read_bucket_contract(bucket)  # Complete, healthy storage contracts.

    view = tmp_path / "view"
    view.mkdir()
    for name, bucket in zip(("admissions", "labevents"), buckets):
        (view / f"{name}.parquet").symlink_to(bucket / "bucket_id=0/part.parquet")
    driver = _driver()
    monkeypatch.setattr(resources, "load_data_sources", lambda: {"miiv": SimpleNamespace(tables={})})
    read_text = Path.read_text
    reads = []

    def spy(path, *args, **kwargs):
        if path.stat().st_ino == first.stat().st_ino:
            reads.append(path)
        return read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", spy)
    parser = Mock(wraps=publication.json.loads)
    monkeypatch.setattr(publication.json, "loads", parser)
    resolver = Mock(wraps=layout.resolve_derived_from_source)
    monkeypatch.setattr(layout, "resolve_derived_from_source", resolver)
    driver._source_publication_closure({"miiv": str(view)})
    assert len(reads) == 1
    assert parser.call_count == 1
    anchors = {call.kwargs["bucket_root"] for call in resolver.call_args_list}
    reads.clear()
    parser.reset_mock()
    (tmp_path / f".remote{publication.PENDING_SUFFIX}").write_text("{}")
    with pytest.raises(publication.SealPublicationError) as caught:
        driver._source_publication_closure({"miiv": str(view)})
    assert caught.value.reason == publication.UNPUBLISHED
    assert caught.value.install == remote
    assert len(reads) == 1
    assert parser.call_count == 1
    assert anchors == set(buckets)


def _shared_provenance(tmp_path):
    shared = tmp_path / "shared"
    shared.mkdir()
    bound = shared / "bound.parquet"
    bound.write_bytes(b"synthetic-bound-source")
    metadata = _json(shared / "provenance.json", derived_from={
        "files": [{"path": "shared/bound.parquet"}],
    })
    root = tmp_path / "install"
    _json(root / layout.CONVERSION_MANIFEST, derived_from=_binding(metadata))
    (root / "sample.parquet").write_bytes(b"synthetic-input")
    layout.validate_conversion_manifest(root)
    return root, shared, bound


@pytest.mark.parametrize("suffix,reason", [
    (publication.PENDING_SUFFIX, publication.UNPUBLISHED),
    (publication.REFUSAL_SUFFIX, publication.REFUSED),
])
def test_metadata_parent_is_marker_only_and_bound_files_remain_in_digest(tmp_path, suffix, reason):
    root, shared, bound = _shared_provenance(tmp_path)
    before = publication.assert_roots_published([root])
    (shared / "unrelated.txt").write_text("unrelated new file")
    assert publication.assert_roots_published([root]) == before
    poison = _json(tmp_path / "poison/conversion_manifest.json")
    (tmp_path / f".poison{publication.PENDING_SUFFIX}").write_text("{}")
    _json(shared / layout.CONVERSION_MANIFEST, derived_from=_binding(poison))
    assert publication.assert_roots_published([root]) == before
    info = bound.stat()
    replacement = shared / "replacement"
    replacement.write_bytes(bound.read_bytes())
    os.utime(replacement, ns=(info.st_atime_ns, info.st_mtime_ns))
    replacement.replace(bound)
    assert bound.stat().st_ino != info.st_ino
    assert publication.assert_roots_published([root]).sha256 != before.sha256
    (shared.parent / f".{shared.name}{suffix}").write_text("{}")
    with pytest.raises(publication.SealPublicationError) as caught:
        publication.assert_roots_published([root])
    assert caught.value.reason == reason
    assert caught.value.install == shared


@pytest.mark.parametrize("placement", ["external", "under_input"])
@pytest.mark.parametrize("change", ["normal_resume", "bound_inode", "pending", "refused"])
def test_stream_resume_excludes_configured_outputs_but_binds_sources(
    tmp_path, monkeypatch, placement, change,
):
    import logging

    import easyicu
    import pandas as pd
    from easyicu.api import extraction as api

    root, shared, bound = _shared_provenance(tmp_path)
    container = shared if placement == "external" else root
    out = container / "configured_destination"
    out.mkdir()
    # A configured log outside output_dir also needs exclusion when a root
    # scan reaches it. No filename convention is used by the digest owner.
    handler = logging.FileHandler(container / "configured_trace.txt")
    api.logger.addHandler(handler)
    calls = []
    interrupt = True

    def load(**kwargs):
        ids = list(kwargs["patient_ids"]["stay_id"])
        calls.append(ids)
        if interrupt and len(calls) == 2:
            raise MemoryError("synthetic interruption")
        return pd.DataFrame({"stay_id": ids, "charttime": [0.0] * len(ids),
                             "test_signal": [1.5] * len(ids)})

    monkeypatch.setattr(easyicu, "load_concepts", load)
    kwargs = {"database": "miiv", "data_path": str(root), "merge": True,
              "concepts": ["test_signal"]}
    cohort = {"stay_id": list(range(1, 11))}

    def run():
        return api._stream_module_batches_to_parquet(
            "test_module", ["test_signal"], kwargs, cohort, 4, str(out),
        )

    try:
        with pytest.raises(MemoryError):
            run()
        ledger_path = out / ".test_module.stream_ledger.json"
        ledger = json.loads(ledger_path.read_text())
        handler.emit(logging.LogRecord(api.__name__, logging.INFO, "", 0,
                                       "synthetic run progress", (), None))
        # All normal order/part/ledger and log writes leave both source
        # identities unchanged before any negative control is introduced.
        current = api._stream_ledger_identity(
            "test_module", ["test_signal"], kwargs, "stay_id",
            api._stream_ids_digest("stay_id", cohort["stay_id"]),
            **({"output_paths": api._extraction_output_paths(out)}
               if hasattr(api, "_extraction_output_paths") else {}),
        )
        assert current == {key: ledger[key] for key in current}
        if change == "bound_inode":
            info = bound.stat()
            replacement = shared / "replacement"
            replacement.write_bytes(bound.read_bytes())
            os.utime(replacement, ns=(info.st_atime_ns, info.st_mtime_ns))
            replacement.replace(bound)
            assert bound.stat().st_ino != info.st_ino
        elif change in {"pending", "refused"}:
            suffix = (publication.PENDING_SUFFIX if change == "pending"
                      else publication.REFUSAL_SUFFIX)
            (shared.parent / f".{shared.name}{suffix}").write_text("{}")
        calls.clear()
        interrupt = False
        if change == "normal_resume":
            result = run()
            assert [value for batch in calls for value in batch] == cohort["stay_id"][4:]
            assert result["stream_resume_events"][0]["kind"] == "resume_after_worker_retry"
            assert pd.read_parquet(result["path"])["stay_id"].tolist() == cohort["stay_id"]
        else:
            before = {path.name: path.read_bytes() for path in out.iterdir()}
            with pytest.raises(publication.SealPublicationError) as caught:
                run()
            assert caught.value.reason == {
                "bound_inode": publication.UNACCEPTED, "pending": publication.UNPUBLISHED,
                "refused": publication.REFUSED,
            }[change]
            assert not calls
            assert {path.name: path.read_bytes() for path in out.iterdir()} == before
    finally:
        api.logger.removeHandler(handler)
        handler.close()


@pytest.mark.parametrize("placement", ["external", "under_input"])
def test_launcher_resume_ignores_its_configured_run_directory(tmp_path, monkeypatch, placement):
    driver = _driver()
    root, shared, _ = _shared_provenance(tmp_path)
    out = (shared if placement == "external" else root) / "configured_run"
    args = driver._parse_args([
        "--output-root", str(out), "--databases", "miiv", "--data-path", f"miiv={root}",
    ])
    identity = {"repository_root": str(driver.REPOSITORY_ROOT), "commit": "a" * 40}
    monkeypatch.setattr(driver, "_git_identity", lambda: identity)
    monkeypatch.setattr(driver, "_require_clean_identity", lambda _: None)
    monkeypatch.setattr(driver, "_load_psutil", lambda _: (None, {"release_sealable": True}))
    monkeypatch.setattr(driver, "_detect_effective_memory", lambda _: {})

    def completed(**kwargs):
        kwargs["manifest"]["sources"]["miiv"] = {"status": "complete"}
        return []

    monkeypatch.setattr(driver, "_pending_databases", completed)
    original = driver.run(args)
    (out / "configured_trace.txt").write_text("normal run progress")
    resumed = driver._load_resume_manifest(
        run_root=out, databases=["miiv"], data_paths={"miiv": str(root)},
        identity=identity, resource_policy=args.resource_policy,
    )
    assert resumed["runtime_provenance"] == original["runtime_provenance"]


def test_stream_refuses_an_explicit_source_binding_inside_its_output(tmp_path, monkeypatch):
    import easyicu
    from easyicu.api import extraction as api

    out = tmp_path / "configured_output"
    source = _json(out / "source.json")
    root = tmp_path / "install"
    _json(root / layout.CONVERSION_MANIFEST, derived_from=_binding(source))
    loader = Mock(side_effect=AssertionError("input/output overlap reached a data read"))
    monkeypatch.setattr(easyicu, "load_concepts", loader)
    with pytest.raises(publication.SealPublicationError) as caught:
        api._stream_module_batches_to_parquet(
            "test_module", ["test_signal"],
            {"database": "miiv", "data_path": str(root), "concepts": ["test_signal"]},
            {"stay_id": list(range(1, 11))}, 4, str(out),
        )
    assert caught.value.reason == publication.UNACCEPTED
    assert "input overlaps run output" in str(caught.value)
    loader.assert_not_called()
