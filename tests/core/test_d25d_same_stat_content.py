"""A Parquet rewritten in place with ordinary values is never served stale (delta25d).

B-D17D-CONTENT: a real Parquet file rewritten with normal finite values,
keeping its inode and size and restoring its mtime, used to pass every warm
check, so the cache returned the frame read before the rewrite. The storage
binding now records (size, mtime_ns, ctime_ns, ino, dev): ``utime`` cannot
restore ctime, so the rewrite is a new version and a bound run refuses it.
A file that changed within ``RACY_WINDOW_NS`` of its read is racily clean:
no payload is cached from it, and within the unit it is bound by content.
Rows are synthetic.
"""

from __future__ import annotations

import os
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from easyicu import storage_binding
from easyicu.config import load_src_cfg
from easyicu.datasource import ICUDataSource
from easyicu.storage_binding import StorageSourceChanged, binding_unit

TABLE = "chartevents"


def _write(path: Path, value: float) -> None:
    pd.DataFrame({"value": np.repeat(value, 20)}).to_parquet(
        path, index=False, compression=None, use_dictionary=False
    )


def _source(tmp_path: Path):
    """The reviewer's REAL_PARQUET_SAME_STAT_CONTENT_BYPASS installation."""
    root = tmp_path / "miiv"
    (root / TABLE).mkdir(parents=True)
    path = root / TABLE / "first.parquet"
    _write(path, 60.0)
    source = ICUDataSource(load_src_cfg("miiv"), base_path=root, enable_cache=True)
    calls = []

    def cold(self, table, columns, pf, items, wide, enable, key):
        calls.append(table)
        files = self.resolve_target_files(table)
        frame = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
        if enable:
            self._cache_raw_frame(key, frame)
        return frame

    source._load_raw_frame_uncached = types.MethodType(cold, source)
    return source, path, calls


def _rewrite_same_stat(path: Path, value: float) -> None:
    """Ordinary finite values, same inode and size, mtime restored."""
    before = path.stat()
    _write(path, value)
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    after = path.stat()
    assert (after.st_ino, after.st_size, after.st_mtime_ns) == (
        before.st_ino, before.st_size, before.st_mtime_ns
    )


def test_a_same_stat_rewrite_of_a_warm_parquet_is_refused(tmp_path, monkeypatch):
    # A settled installation: the payload is cached, as for files written long ago.
    monkeypatch.setattr(storage_binding, "RACY_WINDOW_NS", 0)
    source, path, calls = _source(tmp_path)
    with binding_unit():
        old = source._load_raw_frame(TABLE, None)
        assert source._load_raw_frame(TABLE, None).equals(old) and len(calls) == 1
        _rewrite_same_stat(path, 61.0)
        with pytest.raises(StorageSourceChanged) as error:
            source._load_raw_frame(TABLE, None)
    assert error.value.reason == "storage_source_changed"


def test_the_reviewer_probe_never_returns_the_old_frame(tmp_path):
    # The probe as written: the file is read right after it was written.
    source, path, _calls = _source(tmp_path)
    with binding_unit():
        old = source._load_raw_frame(TABLE, None)
        source._load_raw_frame(TABLE, None)
        _rewrite_same_stat(path, 61.0)
        try:
            warm = source._load_raw_frame(TABLE, None)
        except StorageSourceChanged:
            return  # refused
    assert not warm.equals(old) and warm.equals(pd.read_parquet(path))  # or read again


def test_a_payload_from_a_racily_clean_file_is_not_cached(tmp_path):
    source, _path, calls = _source(tmp_path)
    with binding_unit():
        first = source._load_raw_frame(TABLE, None)
        second = source._load_raw_frame(TABLE, None)
    assert second.equals(first) and len(calls) == 2
    assert not source._table_cache


def test_a_racily_clean_file_is_bound_by_content(tmp_path, monkeypatch):
    # A filesystem whose stat cannot see the rewrite (coarse timestamps):
    # every stat field stays the same, only the bytes differ.
    source, path, _calls = _source(tmp_path)
    source.enable_cache = False
    frozen = storage_binding._version(TABLE, str(path))
    monkeypatch.setattr(storage_binding, "_version", lambda table, p: frozen)
    with binding_unit():
        source._load_raw_frame(TABLE, None)
        _write(path, 61.0)
        with pytest.raises(StorageSourceChanged) as error:
            source._load_raw_frame(TABLE, None)
    assert error.value.reason == "storage_source_changed"
