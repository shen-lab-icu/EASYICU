"""One source binding per run covers every storage read, not only stay tables (Codex delta8c).

Within a unit, the storage owner binds what it resolves, at first sight:
a flat table's whole file set, a bucket read's selected part files, and the
installation's manifest and build receipt (each with a sha256). Any later
resolve that meets another version is ``storage_source_changed``, a
``StorageIntegrityError``. It is never a bucket fallback, and never the new
bytes. A new run binds afresh. ``binding_summary`` digests the sorted
entries for the run receipt. Rows are synthetic.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from easyicu.storage_binding import (
    StorageSourceChanged,
    binding_summary,
    binding_unit,
    child_binding,
)
from easyicu.storage_layout import BUILD_RECEIPT, MANIFEST, StorageIntegrityError
from tests.support.bucket_storage import b6_source, install_layout

TABLE, KEY, ITEMS = "chartevents", "itemid", [220045]


def _touch(path: Path) -> None:
    stat = path.stat()
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))


def _rewrite(path: Path) -> None:
    payload = path.read_bytes()
    path.unlink()
    path.write_bytes(payload)  # same bytes, a new file
    _touch(path)


@pytest.fixture
def bucket(tmp_path):
    source = b6_source(tmp_path / "miiv", "miiv")
    install_layout(source, "hash")
    return source, tmp_path / "miiv" / f"{TABLE}_bucket"


def _read(source):
    return source.resolve_target_files(TABLE, KEY, ITEMS)


def test_a_rewritten_bucket_manifest_between_two_reads_is_refused(bucket):
    source, root = bucket
    with binding_unit():
        first = _read(source)
        assert first and all(Path(p).is_relative_to(root) for p in first)
        _rewrite(root / MANIFEST)
        with pytest.raises(StorageSourceChanged) as error:
            _read(source)
    assert error.value.reason == "storage_source_changed"
    assert isinstance(error.value, StorageIntegrityError)
    assert error.value.path == str(root / MANIFEST)


def test_a_rewritten_bucket_part_between_two_reads_is_refused(bucket):
    source, root = bucket
    with binding_unit():
        (part, *_) = _read(source)
        _rewrite(Path(part))
        with pytest.raises(StorageSourceChanged) as error:
            _read(source)
    assert error.value.reason == "storage_source_changed"
    assert error.value.path == str(part)


def test_a_rewritten_flat_shard_between_two_reads_is_refused(tmp_path):
    source = b6_source(tmp_path / "miiv", "miiv")
    with binding_unit():
        (shard,) = source.resolve_target_files(TABLE)
        _rewrite(Path(shard))
        with pytest.raises(StorageSourceChanged):
            source.resolve_target_files(TABLE)


def test_a_new_run_binds_the_new_bytes(bucket):
    source, root = bucket
    with binding_unit():
        _read(source)
    _rewrite(root / MANIFEST)
    with binding_unit():
        assert _read(source)


def test_the_binding_digest_is_stable_and_names_manifest_and_receipt(bucket):
    source, root = bucket
    assert binding_summary() is None  # no run, no binding
    with binding_unit():
        _read(source)
        first = binding_summary()
        snapshot = child_binding(source)
    with binding_unit():
        _read(source)
        again = binding_summary()
    assert first == again and len(first.digest) == 64 and first.files >= 3
    assert {str(root / MANIFEST), str(root / BUILD_RECEIPT)} <= set(snapshot["sha256"])
    _touch(root / MANIFEST)
    with binding_unit():
        _read(source)
        assert binding_summary().digest != first.digest


def test_a_child_verifies_the_parents_bucket_binding(bucket):
    source, root = bucket
    with binding_unit():
        _read(source)
        snapshot = child_binding(source)
    _rewrite(root / MANIFEST)
    with binding_unit(snapshot), pytest.raises(StorageSourceChanged):
        _read(source)  # the child's first read meets the parent's binding
