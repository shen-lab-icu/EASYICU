"""Resolver-call lifetime and interrupt safety (lead-r13 delta27s).

A call's resolver lease and its resolver guard end with the call AND every
worker that joined it: a copied context that outlives the call is refused
(``ResolverCallClosed``), a joined worker still running keeps its resolver
out of the pool and its guard held, and an interrupt right after the lease
or the guard is taken leaves nothing held. A pool wait is bounded
(``ResolverPoolExhausted``). The interrupt tests inject ``KeyboardInterrupt``
at the first traced line after the lease or guard is actually held, so the
same tests target the reviewer's probe points on any version. Rows are
synthetic.
"""

from __future__ import annotations

import contextvars
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout

import pandas as pd
import pytest

import easyicu.concept as concept_module
import easyicu.resolver_errors as resolver_errors
import easyicu.resolver_pool as pool_module
from easyicu.api import concepts as api
from easyicu.concept import ConceptResolver, stay_tables
from easyicu.resolver_errors import ResolverBusyError
from easyicu.resolver_pool import ResolverPool, call_resolver, resolver_call_scope
from easyicu.resources import load_dictionary
from tests.support.stay_call_sources import SYNTHETIC_EICU_IDS as _IDS
from tests.support.stay_call_sources import synthetic_eicu as _eicu

pytestmark = pytest.mark.filterwarnings("ignore")


class _Absent(Exception):
    """Stands in for a refusal type a version does not define (its test then fails)."""


ResolverCallClosed = getattr(resolver_errors, "ResolverCallClosed", _Absent)
ResolverPoolExhausted = getattr(resolver_errors, "ResolverPoolExhausted", _Absent)


@pytest.fixture(autouse=True)
def _fresh_global_loader():
    # Pools freeze the size/wait configuration per process; tests that set
    # the environment start from an unfrozen one.
    reset = getattr(pool_module, "_reset_resolver_pool_config", lambda: None)
    reset()
    api.clear_global_loader()
    yield
    api.clear_global_loader()
    reset()


def _resolver():
    return ConceptResolver(load_dictionary(include_sofa2=True))


def _load(resolver, source, stays=_IDS):
    return resolver.load_concepts(
        ["pap_mean"], source, merge=True, interval=pd.Timedelta(hours=1),
        r_compatible=False, verbose=False, concept_workers=1,
        patient_ids={"patientunitstayid": [int(s) for s in stays]},
    )


def _api_load(root, stays=_IDS):
    return api.load_concepts(
        ["pap_mean"], database="eicu", data_path=str(root), interval=pd.Timedelta(hours=1),
        patient_ids={"patientunitstayid": [int(s) for s in stays]}, verbose=False,
    )


def _loader_pool(root):
    loader = api._get_global_loader(database="eicu", data_path=str(root), use_sofa2=False)
    return loader, loader._resolver_pool_for_calls()


def _guard_held(resolver) -> bool:
    guard_of = getattr(concept_module, "_call_guard", None)
    if guard_of is not None:
        return guard_of(resolver).call is not None
    return concept_module._top_level_call_lock(resolver)._is_owned()  # 3802ddf70


def _interrupt_once(filename, held, frame_name=None):
    """A tracer raising KeyboardInterrupt at the first line in ``filename`` once ``held()``."""
    hit: list = []

    def tracer(frame, event, arg):
        if (
            not hit and event == "line" and frame.f_code.co_filename == filename
            and (frame_name is None or frame.f_code.co_name == frame_name) and held()
        ):
            hit.append((frame.f_code.co_name, frame.f_lineno))
            raise KeyboardInterrupt()
        return tracer

    return tracer, hit


# -- D27-SCOPE-LIFETIME -------------------------------------------------------

def test_a_copied_context_that_outlives_its_call_is_refused_and_leaks_nothing():
    """Reviewer probe B_scope_stale_copy_leaks_lease."""
    pool = ResolverPool(object(), object, size=1)
    with resolver_call_scope():
        call_resolver(pool)
        stale = contextvars.copy_context()
    assert pool.stats() == {"size": 1, "members": 1, "idle": 1}

    def late():
        with resolver_call_scope():
            return call_resolver(pool)

    with pytest.raises(ResolverCallClosed):
        stale.run(late)
    assert pool.stats() == {"size": 1, "members": 1, "idle": 1}


def test_a_copied_context_cannot_load_through_a_finished_resolver_call(tmp_path, monkeypatch):
    source = _eicu(tmp_path / "eicu")
    resolver = _resolver()
    real = stay_tables.stay_table_frame
    kept: dict = {}

    def keep_context(data_source, table, stays, read, **kwargs):
        kept.setdefault("context", contextvars.copy_context())
        return real(data_source, table, stays, read, **kwargs)

    monkeypatch.setattr(stay_tables, "stay_table_frame", keep_context)
    _load(resolver, source)
    monkeypatch.setattr(stay_tables, "stay_table_frame", real)
    with pytest.raises(ResolverCallClosed):
        kept["context"].run(_load, resolver, source, [1])
    pd.testing.assert_frame_equal(_load(resolver, source), _load(_resolver(), source), check_exact=True)


def test_a_lease_stays_out_until_its_joined_worker_leaves(tmp_path, monkeypatch):
    """Reviewer probe B_scope_returns_running_worker_lease (N=1, real loader)."""
    monkeypatch.setenv(pool_module.POOL_SIZE_ENV, "1")
    root = tmp_path / "eicu"
    _eicu(root)
    serial = _api_load(root)
    loader, pool = _loader_pool(root)
    started, release = threading.Event(), threading.Event()
    in_worker = threading.local()
    real = stay_tables.stay_table_frame

    def gated(data_source, table, stays, read, **kwargs):
        if getattr(in_worker, "on", False):
            started.set()
            release.wait(timeout=60)
        return real(data_source, table, stays, read, **kwargs)

    monkeypatch.setattr(stay_tables, "stay_table_frame", gated)
    executor = ThreadPoolExecutor(max_workers=1)
    results: dict = {}
    try:
        with resolver_call_scope():
            held = call_resolver(pool)
            context = contextvars.copy_context()

            def worker():
                with resolver_call_scope():
                    own = call_resolver(pool)
                    assert own is held
                    in_worker.on = True
                    return _load(own, loader.datasource)

            future = executor.submit(context.run, worker)
            assert started.wait(timeout=30)
        # The owner's scope has ended; its joined worker still uses the resolver.

        def unrelated():
            try:
                results["unrelated"] = _api_load(root)
            except BaseException as error:  # noqa: BLE001 - asserted below
                results["unrelated"] = error

        thread = threading.Thread(target=unrelated, daemon=True)
        thread.start()
        time.sleep(0.4)
        try:
            # It waits on the pool: neither refused nor handed the busy resolver.
            assert "unrelated" not in results, results.get("unrelated")
            assert pool.stats()["idle"] == 0
        finally:
            release.set()
        pd.testing.assert_frame_equal(future.result(timeout=60), serial, check_exact=True)
        thread.join(timeout=60)
    finally:
        release.set()
        executor.shutdown(wait=False, cancel_futures=True)
    assert isinstance(results["unrelated"], pd.DataFrame), results["unrelated"]
    pd.testing.assert_frame_equal(results["unrelated"], serial, check_exact=True)
    assert pool.stats() == {"size": 1, "members": 1, "idle": 1}


def test_a_call_ends_with_its_joined_worker_and_keeps_its_guard_until_then(tmp_path, monkeypatch):
    """The resolver-level twin: depth, unit and guard outlive no joined worker."""
    source = _eicu(tmp_path / "eicu")
    resolver = _resolver()
    expected = _load(_resolver(), source)
    nested_expected = _load(_resolver(), source, [1])
    real = stay_tables.stay_table_frame
    started, release = threading.Event(), threading.Event()
    in_worker = threading.local()
    box: dict = {}

    def spawn_unjoined(data_source, table, stays, read, **kwargs):
        if getattr(in_worker, "on", False):
            started.set()
            release.wait(timeout=60)
        elif "thread" not in box:
            def nested():
                in_worker.on = True
                try:
                    box["nested"] = _load(resolver, source, [1])
                except BaseException as error:  # noqa: BLE001 - asserted below
                    box["nested"] = error

            box["thread"] = threading.Thread(
                target=contextvars.copy_context().run, args=(nested,), daemon=True)
            box["thread"].start()
            assert started.wait(timeout=30)
        return real(data_source, table, stays, read, **kwargs)

    monkeypatch.setattr(stay_tables, "stay_table_frame", spawn_unjoined)

    def owner():
        try:
            box["owner"] = _load(resolver, source)
        except BaseException as error:  # noqa: BLE001 - asserted below
            box["owner"] = error

    owner_thread = threading.Thread(target=owner, daemon=True)
    try:
        owner_thread.start()
        assert started.wait(timeout=30)
        time.sleep(0.4)
        # The owner's body is done, but its call has not ended: the worker runs.
        assert "owner" not in box, box.get("owner")
        assert _guard_held(resolver)
        with ThreadPoolExecutor(max_workers=1) as plain:
            with pytest.raises(ResolverBusyError):
                plain.submit(_load, resolver, source).result(timeout=30)
    finally:
        release.set()
    owner_thread.join(timeout=60)
    box["thread"].join(timeout=60)
    pd.testing.assert_frame_equal(box["owner"], expected, check_exact=True)
    pd.testing.assert_frame_equal(box["nested"], nested_expected, check_exact=True)
    assert not _guard_held(resolver)
    monkeypatch.setattr(stay_tables, "stay_table_frame", real)
    pd.testing.assert_frame_equal(_load(resolver, source), expected, check_exact=True)


# -- D27-CHECKOUT-CANCEL ------------------------------------------------------

def test_an_interrupt_right_after_the_lease_is_taken_returns_it():
    """Reviewer probe B_interrupt_after_checkout_before_record."""
    pool = ResolverPool(object(), object, size=1)
    tracer, hit = _interrupt_once(pool_module.__file__, lambda: pool.stats()["idle"] == 0)
    sys.settrace(tracer)
    try:
        with pytest.raises(KeyboardInterrupt):
            with resolver_call_scope():
                call_resolver(pool)
    finally:
        sys.settrace(None)
    assert hit, "the lease was never taken"
    assert pool.stats() == {"size": 1, "members": 1, "idle": 1}
    with resolver_call_scope():  # the next call is served at once
        assert call_resolver(pool) is pool.members()[0]


# -- D27-LOCK-CANCEL ----------------------------------------------------------

@pytest.mark.parametrize("frame_name", [None, "wrapper"], ids=["first-line-held", "first-wrapper-line-held"])
def test_an_interrupt_right_after_the_resolver_guard_is_taken_releases_it(tmp_path, frame_name):
    """Reviewer probe B_interrupt_after_resolver_lock_before_try."""
    source = _eicu(tmp_path / "eicu")
    resolver = _resolver()
    tracer, hit = _interrupt_once(concept_module.__file__, lambda: _guard_held(resolver), frame_name)
    sys.settrace(tracer)
    try:
        with pytest.raises(KeyboardInterrupt):
            _load(resolver, source)
    finally:
        sys.settrace(None)
    assert hit, "the guard was never taken"
    assert not _guard_held(resolver)
    with ThreadPoolExecutor(max_workers=1) as other:  # another thread is served, not refused
        result = other.submit(_load, resolver, source).result(timeout=60)
    pd.testing.assert_frame_equal(result, _load(_resolver(), source), check_exact=True)


# -- pool wait bound ----------------------------------------------------------

def test_a_pool_wait_is_bounded_and_typed():
    pool = ResolverPool(object(), object, size=1, wait_seconds=0.3)
    first = pool.checkout()
    started = time.monotonic()
    with pytest.raises(ResolverPoolExhausted) as refused:
        pool.checkout()
    assert time.monotonic() - started < 10
    assert refused.value.reason == "resolver_pool_exhausted" and refused.value.size == 1
    pool.checkin(first)
    assert pool.checkout() is first


def test_a_plain_thread_nested_api_call_ends_typed_instead_of_waiting_forever(tmp_path, monkeypatch):
    """The reviewer's known plain-worker wait cycle, at N=1 with a short wait."""
    monkeypatch.setenv(pool_module.POOL_SIZE_ENV, "1")
    monkeypatch.setenv(getattr(pool_module, "POOL_WAIT_ENV", "EASYICU_RESOLVER_POOL_WAIT_SECONDS"), "0.5")
    root = tmp_path / "eicu"
    _eicu(root)
    expected = _api_load(root)
    real = stay_tables.stay_table_frame
    seen: dict = {}
    plain = ThreadPoolExecutor(max_workers=1)

    def nested_from_plain_thread(data_source, table, stays, read, **kwargs):
        if "outcome" not in seen:
            seen["outcome"] = None
            future = plain.submit(_api_load, root, [1])
            try:
                future.result(timeout=30)
                seen["outcome"] = "returned"
            except FutureTimeout:
                seen["outcome"] = "hung"
            except BaseException as error:  # noqa: BLE001 - asserted below
                seen["outcome"] = error
        return real(data_source, table, stays, read, **kwargs)

    monkeypatch.setattr(stay_tables, "stay_table_frame", nested_from_plain_thread)
    try:
        result = _api_load(root)
    finally:
        plain.shutdown(wait=False, cancel_futures=True)
    assert isinstance(seen["outcome"], ResolverPoolExhausted), seen["outcome"]
    pd.testing.assert_frame_equal(result, expected, check_exact=True)


def test_the_pool_wait_is_configured_validated_and_recorded(monkeypatch):
    env = getattr(pool_module, "POOL_WAIT_ENV", "EASYICU_RESOLVER_POOL_WAIT_SECONDS")
    monkeypatch.delenv(env, raising=False)
    assert pool_module.resolver_pool_wait_seconds() == pool_module.DEFAULT_POOL_WAIT_SECONDS == 600.0
    monkeypatch.setenv(env, "2.5")
    assert pool_module.resolver_pool_wait_seconds() == 2.5
    for bad in ("0", "-1", "nan", "inf", "soon"):
        monkeypatch.setenv(env, bad)
        with pytest.raises(ValueError, match=env):
            pool_module.resolver_pool_wait_seconds()
    monkeypatch.setenv(env, "2.5")
    from easyicu.api.extraction import _native_export_runtime_provenance

    assert _native_export_runtime_provenance()["resolver_pool_wait_seconds"] == 2.5


def test_run_metadata_records_the_frozen_pool_values_not_a_later_environment(monkeypatch):
    """Residual (3): the record names the values the pools use."""
    monkeypatch.setenv(pool_module.POOL_SIZE_ENV, "3")
    monkeypatch.setenv(pool_module.POOL_WAIT_ENV, "7")
    first = ResolverPool(object(), object)  # freezes this process's configuration
    assert (first.size, first.wait_seconds) == (3, 7.0)
    monkeypatch.setenv(pool_module.POOL_SIZE_ENV, "5")
    monkeypatch.setenv(pool_module.POOL_WAIT_ENV, "9")
    later = ResolverPool(object(), object)
    assert (later.size, later.wait_seconds) == (3, 7.0)
    from easyicu.api.extraction import _native_export_runtime_provenance

    record = _native_export_runtime_provenance()
    assert (record["resolver_pool_size"], record["resolver_pool_wait_seconds"]) == (3, 7.0)
