"""Setting a cache directory is not consent to deserialize from it.

``ConceptResolver``'s cross-call disk cache can only serialize with pickle:
it stores ``ICUTable`` subclasses whose type and metadata do not survive a
plain parquet round trip. ``pickle.load`` executes the payload before anything
can inspect it, so ``easyicu.api``'s cache states the rule explicitly —
"``use_pickle=True`` is a trusted-local compatibility opt-in and must not be
used with cache files supplied by another user or process boundary".

This cache ignored that rule and gated on ``cache_dir`` alone. A cache
directory that is shared, synced, or writable by another process was therefore
enough to get arbitrary code executed inside the extraction process.
"""

from __future__ import annotations

import pickle

import pandas as pd
import pytest

from easyicu.concept import ConceptResolver
from easyicu.table import IdTbl


def _resolver(tmp_path, **kwargs) -> ConceptResolver:
    return ConceptResolver({}, cache_dir=tmp_path, **kwargs)


def _entry() -> IdTbl:
    return IdTbl(pd.DataFrame({"stay_id": [1], "map": [70.0]}), id_vars="stay_id")


class _Detonator:
    """Stands in for a hostile cache file; records that it was executed."""

    fired = False

    def __reduce__(self):
        return (_Detonator._fire, ())

    @staticmethod
    def _fire():
        _Detonator.fired = True
        return _Detonator()


def test_cache_dir_alone_does_not_enable_the_pickle_cache(tmp_path) -> None:
    resolver = _resolver(tmp_path)
    assert resolver.use_pickle is False
    resolver._store_in_disk_cache("map", None, "k", _entry())
    assert list(tmp_path.glob("*.pkl")) == []
    assert resolver._load_from_disk_cache("map", None, "k") is None


def test_a_planted_cache_file_is_not_executed_without_the_opt_in(tmp_path) -> None:
    _Detonator.fired = False
    (tmp_path / "k.trusted.pkl").write_bytes(pickle.dumps(_Detonator()))
    (tmp_path / "k.pkl").write_bytes(pickle.dumps(_Detonator()))

    assert _resolver(tmp_path)._load_from_disk_cache("map", None, "k") is None
    assert _Detonator.fired is False, (
        "the planted cache entry was deserialized without use_pickle=True"
    )


def test_the_opt_in_round_trips_a_concept(tmp_path) -> None:
    resolver = _resolver(tmp_path, use_pickle=True)
    resolver._store_in_disk_cache("map", None, "k", _entry())
    assert (tmp_path / "k.trusted.pkl").is_file(), (
        "executable cache entries must carry the same .trusted.pkl marker as "
        "easyicu.api's cache"
    )
    restored = resolver._load_from_disk_cache("map", None, "k")
    assert restored is not None
    pd.testing.assert_frame_equal(restored.data, _entry().data)


def test_a_cache_entry_of_the_wrong_shape_is_refused(tmp_path) -> None:
    (tmp_path / "k.trusted.pkl").write_bytes(pickle.dumps("not a table"))
    resolver = _resolver(tmp_path, use_pickle=True)
    assert resolver._load_from_disk_cache("map", None, "k") is None


def test_an_unreadable_cache_entry_is_reported_not_swallowed(tmp_path, caplog) -> None:
    (tmp_path / "k.trusted.pkl").write_bytes(b"not a pickle at all")
    resolver = _resolver(tmp_path, use_pickle=True)
    with caplog.at_level("WARNING"):
        assert resolver._load_from_disk_cache("map", None, "k") is None
    assert "Disk cache read failed" in caplog.text


def test_setting_cache_dir_without_the_opt_in_says_so(tmp_path, caplog) -> None:
    with caplog.at_level("WARNING"):
        _resolver(tmp_path)
    assert "use_pickle=True" in caplog.text
