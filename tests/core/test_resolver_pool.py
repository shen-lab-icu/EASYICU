"""A shared loader lends each call one resolver from a bounded pool (B18).

The ``api.concepts`` global loader is shared across threads. Each
``api.load_concepts`` call checks out ONE resolver for all its loads and
returns it on every exit path; with all N out, a further call waits on the
pool (not on a resolver lock) and then succeeds. Reentry into a call's own
resolver from another context is still refused (``ResolverBusyError``), and a
chunked load's workers use their own resolvers, never a pool member. Rows are
synthetic.
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout

import pandas as pd
import pytest

from easyicu.api import concepts as api
from easyicu.concept import ConceptResolver, stay_tables
from easyicu.resolver_errors import ResolverBusyError
from easyicu.resolver_pool import (
    DEFAULT_POOL_SIZE,
    POOL_SIZE_ENV,
    ResolverPool,
    call_resolver,
    resolver_call_scope,
    resolver_pool_size,
)
from tests.support.stay_call_sources import (
    SYNTHETIC_EICU_IDS as _IDS,
    gate_first_read as _gate_first_read,
    run_threads as _run_threads,
    synthetic_eicu as _eicu,
    wait_for as _wait_for,
)

pytestmark = pytest.mark.filterwarnings("ignore")


@pytest.fixture(autouse=True)
def _fresh_global_loader():
    api.clear_global_loader()
    yield
    api.clear_global_loader()


def _api_load(root, stays=_IDS, **extra):
    return api.load_concepts(
        ["pap_mean"], database="eicu", data_path=str(root), interval=pd.Timedelta(hours=1),
        patient_ids={"patientunitstayid": [int(s) for s in stays]}, verbose=False, **extra,
    )


def _pool(root):
    return api._get_global_loader(database="eicu", data_path=str(root), use_sofa2=False)._resolver_pool_for_calls()


def test_two_overlapping_api_calls_on_one_database_both_succeed(tmp_path, monkeypatch):
    root = tmp_path / "eicu"
    _eicu(root)
    serial = _api_load(root)
    inside, release = _gate_first_read(monkeypatch)
    threads, results = _run_threads(lambda _i: _api_load(root), 2)
    assert _wait_for(lambda: len(inside) == 2), "the two calls never overlapped"
    release.set()
    for thread in threads:
        thread.join(timeout=120)
    assert not any(thread.is_alive() for thread in threads)
    for index in range(2):
        assert isinstance(results[index], pd.DataFrame), results[index]
        pd.testing.assert_frame_equal(results[index], serial, check_exact=True)
    assert _pool(root).stats() == {"size": DEFAULT_POOL_SIZE, "members": 2, "idle": 2}


def test_a_call_beyond_n_waits_on_the_pool_then_succeeds(tmp_path, monkeypatch):
    monkeypatch.setenv(POOL_SIZE_ENV, "2")
    root = tmp_path / "eicu"
    _eicu(root)
    serial = _api_load(root)
    pool = _pool(root)
    assert pool.size == 2
    inside, release = _gate_first_read(monkeypatch)
    threads, results = _run_threads(lambda _i: _api_load(root), 3)
    try:
        assert _wait_for(lambda: len(inside) == 2), "two calls never held both resolvers"
        time.sleep(0.3)
        # The third call waits on the pool: it never reached a read, nothing idle.
        assert len(inside) == 2
        assert pool.stats() == {"size": 2, "members": 2, "idle": 0}
        assert sum(thread.is_alive() for thread in threads) == 3
        assert not results
    finally:
        release.set()
    for thread in threads:
        thread.join(timeout=120)
    assert not any(thread.is_alive() for thread in threads), "a waiting call never returned"
    for index in range(3):
        assert isinstance(results[index], pd.DataFrame), results[index]
        pd.testing.assert_frame_equal(results[index], serial, check_exact=True)
    assert pool.stats() == {"size": 2, "members": 2, "idle": 2}


def test_an_exception_inside_a_call_returns_its_resolver(tmp_path, monkeypatch):
    root = tmp_path / "eicu"
    _eicu(root)
    expected = _api_load(root)
    pool = _pool(root)
    before = pool.stats()
    real = stay_tables.stay_table_frame

    def failing(data_source, table, stays, read, **kwargs):
        raise RuntimeError("boom")

    with monkeypatch.context() as patched:
        patched.setattr(stay_tables, "stay_table_frame", failing)
        with pytest.raises(Exception):
            _api_load(root)
    assert stay_tables.stay_table_frame is real
    assert pool.stats() == before  # same members, every one idle again
    pd.testing.assert_frame_equal(_api_load(root), expected, check_exact=True)
    assert pool.stats() == before


def test_reentry_into_a_calls_resolver_is_refused_and_a_nested_call_joins_it(tmp_path, monkeypatch):
    root = tmp_path / "eicu"
    _eicu(root)
    expected = _api_load(root)
    nested_expected = _api_load(root, stays=[1])
    pool = _pool(root)
    real = stay_tables.stay_table_frame
    seen: dict = {}
    plain = ThreadPoolExecutor(max_workers=1)

    def probe(data_source, table, stays, read, **kwargs):
        if "busy" not in seen:
            seen["busy"] = None
            own = call_resolver(pool)  # the resolver this call holds
            future = plain.submit(own.load_concepts, ["pap_mean"], data_source,
                                  interval=pd.Timedelta(hours=1), verbose=False,
                                  patient_ids={"patientunitstayid": [1]})
            try:
                future.result(timeout=30)
                seen["busy"] = "returned"
            except FutureTimeout:
                seen["busy"] = "hung"
            except ResolverBusyError as error:
                seen["busy"] = error
            seen["members"] = pool.stats()["members"]
            seen["nested"] = _api_load(root, stays=[1])  # same context: joins this call
            seen["members_after_nested"] = pool.stats()["members"]
        return real(data_source, table, stays, read, **kwargs)

    monkeypatch.setattr(stay_tables, "stay_table_frame", probe)
    try:
        result = _api_load(root)
    finally:
        plain.shutdown(wait=False, cancel_futures=True)
    assert isinstance(seen["busy"], ResolverBusyError), seen["busy"]
    assert seen["members"] == seen["members_after_nested"]  # the nested call took no resolver
    pd.testing.assert_frame_equal(seen["nested"], nested_expected, check_exact=True)
    pd.testing.assert_frame_equal(result, expected, check_exact=True)
    assert pool.stats()["idle"] == pool.stats()["members"]


def test_chunk_workers_never_use_a_pool_resolver(tmp_path, monkeypatch):
    root = tmp_path / "eicu"
    _eicu(root)
    used: list = []
    lock = threading.Lock()
    real = ConceptResolver.load_concepts

    def recording(self, *args, **kwargs):
        with lock:
            used.append((self, threading.get_ident()))
        return real(self, *args, **kwargs)

    monkeypatch.setattr(ConceptResolver, "load_concepts", recording)
    loader = api._get_global_loader(database="eicu", data_path=str(root), use_sofa2=False)
    pool = loader._resolver_pool_for_calls()
    main = threading.get_ident()
    with resolver_call_scope():
        held = call_resolver(pool)
        loader.load_concepts(
            ["pap_mean"], patient_ids={"patientunitstayid": [int(s) for s in _IDS]},
            interval=pd.Timedelta(hours=1), chunk_size=1, parallel_workers=2,
            parallel_backend="thread", verbose=False,
        )
    chunk_resolvers = [resolver for resolver, ident in used if ident != main]
    assert chunk_resolvers, "the load did not run chunk workers"
    members = pool.members()
    assert not any(r is m for r in chunk_resolvers for m in members)
    assert not any(r is held for r in chunk_resolvers)


def test_pool_size_is_configured_validated_and_recorded(monkeypatch):
    monkeypatch.delenv(POOL_SIZE_ENV, raising=False)
    assert resolver_pool_size() == DEFAULT_POOL_SIZE == 4
    monkeypatch.setenv(POOL_SIZE_ENV, "3")
    assert resolver_pool_size() == 3
    for bad in ("0", "-1", "four"):
        monkeypatch.setenv(POOL_SIZE_ENV, bad)
        with pytest.raises(ValueError, match=POOL_SIZE_ENV):
            resolver_pool_size()
    monkeypatch.setenv(POOL_SIZE_ENV, "3")
    from easyicu.api.extraction import _native_export_runtime_provenance

    assert _native_export_runtime_provenance()["resolver_pool_size"] == 3


def test_pool_refuses_foreign_or_double_returns():
    primary = object()
    pool = ResolverPool(primary, object, size=1)
    with pytest.raises(RuntimeError, match="resolver_pool_double_return"):
        pool.checkin(primary)
    with pytest.raises(RuntimeError, match="resolver_pool_foreign"):
        pool.checkin(object())
    assert pool.checkout() is primary
    pool.checkin(primary)
