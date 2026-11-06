"""A stay-table binding and its payload belong to one call or one run (B-STAY-SCOPE).

The token is the binding unit (``storage_binding.binding_unit``): a
top-level ``load_concepts`` opens one, or joins the run's. It is released on
every exit path (return, exception, ``KeyboardInterrupt``), so a rerun after
a refused batch reads the stable installation again. Concurrent top-level
calls never share or evict each other's payload. On one shared resolver a
second context's call waits for the first instead of being taken for a
nested call. A streamed run binds once, drops each batch's payload at the
batch clear, and leaves nothing held after a memory retry or an isolated
batch. Rows are synthetic.
"""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from easyicu.api import extraction as api
from easyicu.concept import ConceptResolver
from easyicu.concept import stay_tables
from easyicu.concept.stay_tables import (
    StayTableSourceChanged,
    held_stay_tables,
    stay_table_reads,
)
from easyicu.resources import load_dictionary
from easyicu.storage_binding import active_unit, binding_unit, live_units
from tests.support.inprocess_extraction import inprocess_mp_context
from tests.support.urine_raw_source import _source

pytestmark = pytest.mark.filterwarnings("ignore")

_IDS = np.arange(1, 4)


def _eicu(root: Path):
    return _source(
        root, "eicu",
        patient=pd.DataFrame({
            "patientunitstayid": _IDS, "patienthealthsystemstayid": _IDS,
            "unitdischargeoffset": 6000, "hospitaladmitoffset": 0, "admissionweight": 70.0,
        }),
        vitalperiodic=pd.DataFrame({
            "vitalperiodicid": _IDS, "patientunitstayid": _IDS, "observationoffset": 150,
            "pasystolic": 25.0, "padiastolic": 15.0, "pamean": [30.0, 31.0, 32.0],
            "heartrate": 80.0,
        }),
    )


def _load(resolver, source, stays):
    return resolver.load_concepts(
        ["pap_mean"], source, merge=True, interval=pd.Timedelta(hours=1),
        r_compatible=False, verbose=False, concept_workers=1,
        patient_ids={"patientunitstayid": [int(s) for s in stays]},
    )


def _resolver():
    return ConceptResolver(load_dictionary(include_sofa2=True))


def _touch(path: Path) -> None:
    stat = path.stat()
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))


def _released(resolver=None) -> None:
    from easyicu.concept import _RESOLVER_CALLS

    assert live_units() == ()
    assert held_stay_tables() == 0
    assert active_unit() is None
    assert _RESOLVER_CALLS.get() == frozenset()  # no call marked live in this context
    if resolver is not None:
        assert resolver._load_depth == 0
        # The call lock is free for another thread right away.
        free: list = []
        thread = threading.Thread(
            target=lambda: free.append(_lock_of(resolver).acquire(timeout=5) and (_lock_of(resolver).release() or True))
        )
        thread.start()
        thread.join(timeout=10)
        assert free == [True]


def _lock_of(resolver):
    from easyicu.concept import _top_level_call_lock

    return _top_level_call_lock(resolver)


def _fail_on_read(monkeypatch, error):
    real = stay_tables.stay_table_frame

    def failing(data_source, table, stays, read, **kwargs):
        def broken():
            read()
            raise error
        return real(data_source, table, stays, broken, **kwargs)

    monkeypatch.setattr(stay_tables, "stay_table_frame", failing)


@pytest.mark.parametrize("error", [RuntimeError("boom"), KeyboardInterrupt()])
def test_a_failed_or_interrupted_call_releases_its_unit_and_payload(tmp_path, monkeypatch, error):
    source = _eicu(tmp_path / "eicu")
    resolver = _resolver()
    expected = _load(_resolver(), source, _IDS)
    with monkeypatch.context() as patched:
        _fail_on_read(patched, error)
        with pytest.raises(type(error)):
            _load(resolver, source, _IDS)
    _released(resolver)
    # The next call on the same resolver is top level, bound and complete.
    pd.testing.assert_frame_equal(_load(resolver, source, _IDS), expected, check_exact=True)
    _released(resolver)


def test_a_touch_between_calls_is_read_again_and_a_rerun_after_a_refusal_succeeds(tmp_path):
    source = _eicu(tmp_path / "eicu")
    patient = tmp_path / "eicu" / "patient.parquet"
    resolver = _resolver()
    expected = _load(resolver, source, _IDS)
    _touch(patient)  # (i) same bytes, new version, between two calls
    pd.testing.assert_frame_equal(_load(resolver, source, _IDS), expected, check_exact=True)
    with binding_unit():  # (ii) a change inside one batch is refused
        _load(resolver, source, _IDS)
        _touch(patient)
        with pytest.raises(StayTableSourceChanged) as refused:
            _load(_resolver(), source, _IDS)
    assert refused.value.reason == "stay_table_source_changed"
    _released(resolver)
    # (iii) the refusal says to rerun on a stable installation: that works.
    pd.testing.assert_frame_equal(_load(resolver, source, _IDS), expected, check_exact=True)
    _released(resolver)


@pytest.mark.parametrize("shared", [False, True], ids=["own-resolvers", "shared-resolver"])
def test_concurrent_top_level_calls_neither_share_nor_refuse(tmp_path, monkeypatch, shared):
    source = _eicu(tmp_path / "eicu")
    stay_sets = ([1, 2], [1, 2], [2, 3])
    serial = {tuple(s): _load(_resolver(), source, s) for s in ({1, 2}, {2, 3})}
    common = _resolver()
    first_inside = threading.Event()
    real = stay_tables.stay_table_frame

    def slow_first(data_source, table, stays, read, **kwargs):
        if not first_inside.is_set():
            first_inside.set()
            time.sleep(0.5)  # the other calls start while this one is live
        return real(data_source, table, stays, read, **kwargs)

    monkeypatch.setattr(stay_tables, "stay_table_frame", slow_first)
    results: dict = {}

    def call(index, stays):
        resolver = common if shared else _resolver()
        if index:
            first_inside.wait(timeout=60)
        try:
            results[index] = _load(resolver, source, stays)
        except Exception as error:  # noqa: BLE001 - reported by the assertion below
            results[index] = error

    before = stay_table_reads()
    threads = [threading.Thread(target=call, args=(i, s)) for i, s in enumerate(stay_sets)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=120)
    assert first_inside.is_set()
    for index, stays in enumerate(stay_sets):
        assert isinstance(results[index], pd.DataFrame), results[index]
        pd.testing.assert_frame_equal(results[index], serial[tuple(stays)], check_exact=True)
    # Each call is its own unit: no payload is shared across calls.
    assert stay_table_reads() - before == len(stay_sets)
    _released(common)


def test_concurrent_calls_with_concept_worker_pools_do_not_deadlock(tmp_path):
    """Lock ordering: a call's concept workers (copied context) never take the call lock."""
    source = _eicu(tmp_path / "eicu")
    names = ["pap_mean", "pap_sys", "pap_dia", "hr"]
    shared = _resolver()

    def load(resolver):
        return resolver.load_concepts(
            names, source, merge=True, interval=pd.Timedelta(hours=1), r_compatible=False,
            verbose=False, concept_workers=4, patient_ids={"patientunitstayid": [1, 2, 3]},
        )

    expected = load(_resolver())
    results: dict = {}

    def call(index):
        try:
            results[index] = load(shared)
        except BaseException as error:  # noqa: BLE001 - reported below
            results[index] = error

    threads = [threading.Thread(target=call, args=(i,), daemon=True) for i in range(3)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=180)
    assert not any(thread.is_alive() for thread in threads), "deadlock: a call never returned"
    for index in range(3):
        assert isinstance(results[index], pd.DataFrame), results[index]
        pd.testing.assert_frame_equal(results[index], expected, check_exact=True)
    _released(shared)


def test_a_nested_call_from_a_copied_context_thread_joins_the_outer_call(tmp_path, monkeypatch):
    """ContextVar propagation: a copy_context thread inside a call is nested, not blocked."""
    import contextvars

    source = _eicu(tmp_path / "eicu")
    resolver = _resolver()
    real = stay_tables.stay_table_frame
    seen: dict = {}

    def nested_from_thread(data_source, table, stays, read, **kwargs):
        frame = real(data_source, table, stays, read, **kwargs)
        if "nested" not in seen:
            seen["nested"] = None
            outer = active_unit()

            def inner():
                seen["same_unit"] = active_unit() is outer
                seen["nested"] = _load(resolver, source, [1])

            context = contextvars.copy_context()
            worker = threading.Thread(target=context.run, args=(inner,), daemon=True)
            worker.start()
            worker.join(timeout=60)
            seen["alive"] = worker.is_alive()
        return frame

    monkeypatch.setattr(stay_tables, "stay_table_frame", nested_from_thread)
    _load(resolver, source, _IDS)
    assert seen["alive"] is False  # it neither waited for the call lock nor deadlocked
    assert seen["same_unit"] is True
    assert isinstance(seen["nested"], pd.DataFrame)
    _released(resolver)


def _aumc_run(tmp_path, monkeypatch, *, oom_at=None, isolated=()):
    from easyicu.api.concepts import clear_global_loader
    from easyicu.synthetic_sources import generate_source

    root = tmp_path / "aumc"
    generate_source(root, "aumc", stays=6)
    events: list = []
    units: set = set()
    real_frame = stay_tables.stay_table_frame
    real_drop = stay_tables.drop_stay_table_frames

    def frame(data_source, table, stays, read, **kwargs):
        units.add(id(active_unit()))

        def counted():
            events.append(("read", table, stays))
            if oom_at is not None and sum(e[0] == "read" for e in events) == oom_at:
                raise MemoryError("simulated")
            return read()

        return real_frame(data_source, table, stays, counted, **kwargs)

    def drop():
        real_drop()
        events.append(("clear", held_stay_tables()))

    monkeypatch.setattr(stay_tables, "stay_table_frame", frame)
    monkeypatch.setattr(stay_tables, "drop_stay_table_frames", drop)
    monkeypatch.setattr(api, "_get_extraction_mp_context", inprocess_mp_context)
    monkeypatch.setattr(api, "_ISOLATED_STREAM_BATCH_TARGETS", frozenset(isolated))
    try:
        api.extract_database(
            "aumc", data_path=str(root), output_dir=str(tmp_path / "export"),
            modules=["vitals", "outcome"], native_export_v2=True,
            stream_output_batches=True, batch_size=4, verbose=False,
        )
    finally:
        clear_global_loader()
    return events, units


def test_a_streamed_run_binds_once_and_drops_each_batch_at_its_clear(tmp_path, monkeypatch):
    events, units = _aumc_run(tmp_path, monkeypatch)
    assert len(units) == 1  # one run unit across both module groups and every batch
    clears = [e for e in events if e[0] == "clear"]
    reads = [e for e in events if e[0] == "read"]
    assert reads and len(clears) >= 2
    assert {held for _, held in clears} == {0}  # every clear drops the payload
    window: list = []
    for event in events:  # between two clears, a stay set is read at most once
        if event[0] == "clear":
            window = []
            continue
        assert event[1:] not in window
        window.append(event[1:])
    _released()


@pytest.mark.parametrize(
    "oom_at, isolated",
    [(2, ()), (None, {("aumc", "vitals"), ("aumc", "outcome")})],
    ids=["memory-retry", "isolated-batches"],
)
def test_a_memory_retry_or_an_isolated_batch_leaves_nothing_held(tmp_path, monkeypatch, oom_at, isolated):
    events, _units = _aumc_run(tmp_path, monkeypatch, oom_at=oom_at, isolated=isolated)
    assert any(e[0] == "read" for e in events)
    assert {e[1] for e in events if e[0] == "clear"} == {0}
    assert (tmp_path / "export").exists()
    _released()
