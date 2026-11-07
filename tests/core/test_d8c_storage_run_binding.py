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


def test_a_bound_bucket_whose_source_changes_never_falls_back_to_flat(bucket):
    """D9B-LAYOUT-DOWNGRADE: the bound layout refuses; no silent flat read."""
    source, root = bucket
    with binding_unit():
        _read(source)
        (shard,) = (root.parent / TABLE).glob("*.parquet")
        _rewrite(shard)  # the layout's derived_from source
        with pytest.raises(StorageSourceChanged) as error:
            _read(source)
    assert error.value.path == str(root / MANIFEST)


def test_a_table_read_flat_never_switches_to_a_bucket_mid_run(tmp_path):
    source = b6_source(tmp_path / "miiv", "miiv")
    with binding_unit():
        source.resolve_target_files(TABLE)
        install_layout(source, "hash")
        with pytest.raises(StorageSourceChanged) as error:
            _read(source)
    assert error.value.reason == "storage_source_changed"


def test_a_bound_table_that_vanishes_is_refused_not_read_as_empty(tmp_path):
    """D9B-EMPTY-RESOLVE: an installation without a manifest loses the table."""
    source = b6_source(tmp_path / "miiv", "miiv")
    (tmp_path / "miiv" / "conversion_manifest.json").unlink(missing_ok=True)
    with binding_unit():
        (shard,) = source.resolve_target_files(TABLE)
        Path(shard).unlink()
        with pytest.raises(StorageSourceChanged) as error:
            source.resolve_target_files(TABLE)
    assert error.value.reason == "storage_source_missing"


def test_a_deleted_bound_build_receipt_is_missing(bucket):
    source, root = bucket
    with binding_unit():
        _read(source)
        (root / BUILD_RECEIPT).unlink()
        with pytest.raises(StorageSourceChanged) as error:
            _read(source)
    assert error.value.reason == "storage_source_missing"


def test_binding_refusals_cross_process_boundaries():
    import pickle

    from easyicu.concept.stay_tables import StayTableSourceChanged, StayTableUnbound

    for error in (StorageSourceChanged("storage_source_changed", "t", "/x/p.parquet"),
                  StayTableSourceChanged("stay_table_source_missing", "patient", "/x/q")):
        again = pickle.loads(pickle.dumps(error))
        assert (type(again), again.reason, again.table, again.path, str(again)) == (
            type(error), error.reason, error.table, error.path, str(error))
    again = pickle.loads(pickle.dumps(StayTableUnbound("patient")))
    assert (again.table, again.reason) == ("patient", "stay_table_unbound")


# Codex delta10d: the table's layout choice is an explicit record, checked
# before any existence branch; legitimate coexisting reads are each verified.


def test_a_vanished_bound_bucket_is_missing_not_a_flat_fallback(bucket, tmp_path):
    source, root = bucket
    with binding_unit():
        assert all(Path(p).is_relative_to(root) for p in _read(source))
        root.rename(tmp_path / "moved_bucket")  # the flat table is still beside it
        with pytest.raises(StorageSourceChanged) as error:
            _read(source)
    assert error.value.reason == "storage_source_missing"


def test_a_vanished_bound_bucket_without_flat_is_missing_not_empty(bucket, tmp_path):
    from easyicu.storage_layout import flat_table_files

    source, root = bucket
    with binding_unit():
        _read(source)
        for flat in flat_table_files(source, TABLE):
            Path(flat).unlink()
        root.rename(tmp_path / "moved_bucket")
        with pytest.raises(StorageSourceChanged) as error:
            source.resolve_target_files(TABLE)
    assert error.value.reason == "storage_source_missing"


def test_a_bound_bucket_swapped_for_another_copy_is_refused(bucket, tmp_path):
    import shutil

    source, root = bucket
    with binding_unit():
        _read(source)
        copy = tmp_path / "other_bucket"
        shutil.copytree(root, copy)
        root.rename(tmp_path / "moved_bucket")
        root.symlink_to(copy, target_is_directory=True)  # same path, another installation
        with pytest.raises(StorageSourceChanged) as error:
            _read(source)
    assert error.value.reason == "storage_source_changed"


def test_a_flat_read_beside_the_bucket_and_the_bucket_read_coexist(bucket):
    # HiRID-shaped: a deliberate flat read of the table, then the resolver's
    # bucket read, in one unit. Each is bound and verified on its own.
    from easyicu.storage_layout import flat_table_files, flat_table_scan

    source, root = bucket
    (flat,) = flat_table_files(source, TABLE)
    with binding_unit():
        assert flat_table_scan(source, TABLE, Path(flat)) == (Path(flat),)
        assert all(Path(p).is_relative_to(root) for p in _read(source))
        assert flat_table_scan(source, TABLE, Path(flat)) == (Path(flat),)
        _rewrite(Path(flat))  # the flat read stays bound on its own
        with pytest.raises(StorageSourceChanged):
            flat_table_scan(source, TABLE, Path(flat))


def test_a_bucketed_stay_binding_and_a_flat_read_of_the_table_coexist(bucket):
    from easyicu.concept.stay_tables import _bound
    from easyicu.storage_layout import flat_table_files, flat_table_scan

    source, _root = bucket
    (flat,) = flat_table_files(source, TABLE)
    with binding_unit():
        _bound(source, TABLE)
        assert flat_table_scan(source, TABLE, Path(flat)) == (Path(flat),)
        _bound(source, TABLE)
        assert _read(source)
