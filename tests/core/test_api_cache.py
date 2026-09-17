from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd
import pytest

from easyicu import api
from easyicu.api import concepts as concept_api


def test_clear_global_loader_clears_datasource_cache() -> None:
    class Resolver:
        def __init__(self) -> None:
            self.cleared = False

        def clear(self) -> None:
            self.cleared = True

    class DataSource:
        def __init__(self) -> None:
            self.cleared = False

        def clear(self) -> None:
            self.cleared = True

    class Loader:
        def __init__(self) -> None:
            self.concept_resolver = Resolver()
            self.datasource = DataSource()

    loader = Loader()
    concept_api._global_loader = loader
    concept_api._loader_config = ("miiv", "/tmp/example", None, frozenset())

    api.clear_global_loader()

    assert loader.concept_resolver.cleared
    assert loader.datasource.cleared
    assert concept_api._global_loader is None
    assert concept_api._loader_config is None


def test_compress_dtypes_handles_table_wrappers() -> None:
    class Table:
        def __init__(self) -> None:
            self.data = pd.DataFrame({"value": [1.0, 2.0]})

    table = Table()

    compressed = api._compress_dtypes(table)

    assert compressed is table
    assert str(table.data["value"].dtype) == "Int8"


def test_pickle_cache_without_hmac_key_is_refused(monkeypatch, tmp_path) -> None:
    data_path = tmp_path / "data"
    cache_path = tmp_path / "cache"
    data_path.mkdir()
    monkeypatch.delenv("EASYICU_CACHE_HMAC_KEY", raising=False)
    monkeypatch.setattr(
        api, "load_concepts", lambda **_kwargs: pd.DataFrame({"stay_id": [1]})
    )

    with pytest.raises(RuntimeError, match="EASYICU_CACHE_HMAC_KEY"):
        api.load_concept_cached(
            "hr",
            "miiv",
            data_path,
            cache_dir=cache_path,
            use_pickle=True,
            verbose=False,
        )


def test_tampered_pickle_cache_falls_back_with_warning(monkeypatch, tmp_path) -> None:
    data_path = tmp_path / "data"
    cache_path = tmp_path / "cache"
    data_path.mkdir()
    monkeypatch.setenv("EASYICU_CACHE_HMAC_KEY", "test-hmac-key")
    calls: list[dict] = []

    def fake_load_concepts(**kwargs):
        calls.append(kwargs)
        return pd.DataFrame({"stay_id": [len(calls)]})

    monkeypatch.setattr(api, "load_concepts", fake_load_concepts)

    first = api.load_concept_cached(
        "hr",
        "miiv",
        data_path,
        cache_dir=cache_path,
        use_pickle=True,
        verbose=False,
    )
    assert len(calls) == 1
    payloads = list(cache_path.glob("*.trusted.pkl"))
    sidecars = list(cache_path.glob("*.trusted.pkl.hmac"))
    assert len(payloads) == 1 and len(sidecars) == 1

    raw = payloads[0].read_bytes()
    tampered = bytes([raw[0] ^ 0xFF]) + raw[1:] if raw else b"\x00"
    payloads[0].write_bytes(tampered)

    with pytest.warns(UserWarning, match="HMAC"):
        second = api.load_concept_cached(
            "hr",
            "miiv",
            data_path,
            cache_dir=cache_path,
            use_pickle=True,
            verbose=False,
        )
    # Tampered entry was deleted and recomputed (not served from cache).
    assert len(calls) == 2
    assert second["stay_id"].tolist() == [2]
    assert first["stay_id"].tolist() != second["stay_id"].tolist()

    # The recomputed entry is signed again, so the next load is a cache hit.
    third = api.load_concept_cached(
        "hr",
        "miiv",
        data_path,
        cache_dir=cache_path,
        use_pickle=True,
        verbose=False,
    )
    assert len(calls) == 2
    assert third.equals(second)


def test_signed_pickle_cache_cannot_be_replayed_under_another_key(
    monkeypatch, tmp_path
) -> None:
    data_path = tmp_path / "data"
    cache_path = tmp_path / "cache"
    data_path.mkdir()
    monkeypatch.setenv("EASYICU_CACHE_HMAC_KEY", "test-hmac-key")
    calls = 0

    def fake_load_concepts(**_kwargs):
        nonlocal calls
        calls += 1
        return pd.DataFrame({"stay_id": [calls]})

    monkeypatch.setattr(api, "load_concepts", fake_load_concepts)
    api.load_concept_cached(
        "hr", "miiv", data_path, cache_dir=cache_path, use_pickle=True, verbose=False
    )
    first_path = next(cache_path.glob("*.trusted.pkl"))
    api.load_concept_cached(
        "sbp", "miiv", data_path, cache_dir=cache_path, use_pickle=True, verbose=False
    )
    second_path = next(
        path for path in cache_path.glob("*.trusted.pkl") if path != first_path
    )

    first_path.write_bytes(second_path.read_bytes())
    Path(str(first_path) + ".hmac").write_bytes(
        Path(str(second_path) + ".hmac").read_bytes()
    )
    with pytest.warns(UserWarning, match="HMAC"):
        reloaded = api.load_concept_cached(
            "hr",
            "miiv",
            data_path,
            cache_dir=cache_path,
            use_pickle=True,
            verbose=False,
        )
    assert calls == 3
    assert reloaded["stay_id"].tolist() == [3]


def test_signed_pickle_cache_does_not_sign_disk_tampering_during_write(
    monkeypatch, tmp_path
) -> None:
    data_path = tmp_path / "data"
    cache_path = tmp_path / "cache"
    data_path.mkdir()
    monkeypatch.setenv("EASYICU_CACHE_HMAC_KEY", "test-hmac-key")
    calls = 0
    tampered = False
    original_read_bytes = Path.read_bytes

    def fake_load_concepts(**_kwargs):
        nonlocal calls
        calls += 1
        return pd.DataFrame({"stay_id": [calls]})

    def tamper_before_read(path: Path) -> bytes:
        nonlocal tampered
        if (
            path.parent == cache_path
            and path.name.endswith(".trusted.pkl")
            and not tampered
        ):
            path.write_bytes(pickle.dumps(pd.DataFrame({"stay_id": [999]})))
            tampered = True
        return original_read_bytes(path)

    monkeypatch.setattr(api, "load_concepts", fake_load_concepts)
    monkeypatch.setattr(Path, "read_bytes", tamper_before_read)
    api.load_concept_cached(
        "hr", "miiv", data_path, cache_dir=cache_path, use_pickle=True, verbose=False
    )
    with pytest.warns(UserWarning, match="HMAC"):
        reloaded = api.load_concept_cached(
            "hr",
            "miiv",
            data_path,
            cache_dir=cache_path,
            use_pickle=True,
            verbose=False,
        )
    assert tampered is True
    assert calls == 2
    assert reloaded["stay_id"].tolist() == [2]
