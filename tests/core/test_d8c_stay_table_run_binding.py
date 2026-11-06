"""Stay-table reads verify against one run binding; the cache holds payload only (Codex delta7c, delta8c).

The source binding (file set plus each file's size, mtime_ns and inode) is
owned by the outermost unit: an extraction run, or one public
``load_concepts`` call. Every read, whether from the cache or from disk,
verifies against it. Dropping the payload (a batch boundary, or between
base and derived vitals) never drops the binding. Child processes verify
against the parent's binding. A preload failure cannot leave the next call
looking nested and unbound. Rows are synthetic.
"""

from __future__ import annotations

import contextvars
import multiprocessing
import os
import queue
import threading
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow
import pytest

from easyicu.api import extraction as api
from easyicu.concept import ConceptResolver
from easyicu.concept import stay_tables
from easyicu.concept.stay_tables import (
    StayTableSourceChanged,
    StayTableUnbound,
    child_stay_binding,
    drop_stay_table_frames,
    held_stay_tables,
    stay_table_reads,
    stay_table_unit,
)
from easyicu.resources import load_dictionary
from tests.support.stay_table_child import load_twice_in_own_unit, load_under_binding
from tests.support.urine_raw_source import _load, _source

pytestmark = pytest.mark.filterwarnings("ignore")

_IDS = np.arange(1, 4)


def _eicu(root: Path):
    return _source(
        root, "eicu",
        patient=pd.DataFrame({
            "patientunitstayid": _IDS, "patienthealthsystemstayid": _IDS,
            "unitdischargeoffset": 6000, "hospitaladmitoffset": 0, "admissionweight": 70.0,
        }),
        # pap_mean is a single-source wide concept (vitalperiodic preload);
        # crea a long-table one (lab).
        vitalperiodic=pd.DataFrame({
            "vitalperiodicid": _IDS, "patientunitstayid": _IDS, "observationoffset": 150,
            "pasystolic": 25.0, "padiastolic": 15.0, "pamean": 30.0, "heartrate": 80.0,
        }),
        lab=pd.DataFrame({
            "labid": _IDS, "patientunitstayid": _IDS, "labresultoffset": 200,
            "labname": "creatinine", "labresult": 1.0, "labresultrevisedoffset": 200,
            "labmeasurenamesystem": "mg/dL",
        }),
    )


def _touch(path: Path) -> None:
    stat = path.stat()
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))


def _ids(stays=_IDS):
    return {"patientunitstayid": [int(s) for s in stays]}


def _rewrite_after_first_read(monkeypatch, patient: Path) -> list:
    real = stay_tables.stay_table_frame
    calls: list = []

    def wrapped(data_source, table, stays, read, **kwargs):
        calls.append(table)
        frame = real(data_source, table, stays, read, **kwargs)
        if len(calls) == 1:
            _touch(patient)  # the installation changes inside the unit
        return frame

    monkeypatch.setattr(stay_tables, "stay_table_frame", wrapped)
    return calls


def test_a_call_with_a_preload_holds_nothing_afterwards(tmp_path):
    _load(_eicu(tmp_path / "eicu"), ["pap_mean"], patient_ids=_ids())
    assert held_stay_tables() == 0


def test_a_rewrite_between_the_preload_and_a_later_read_is_refused(tmp_path, monkeypatch):
    source = _eicu(tmp_path / "eicu")
    calls = _rewrite_after_first_read(monkeypatch, tmp_path / "eicu" / "patient.parquet")
    with pytest.raises(StayTableSourceChanged) as error:
        _load(source, ["pap_mean", "crea"], patient_ids=_ids())
    assert error.value.reason == "stay_table_source_changed"
    assert len(calls) >= 2
    assert held_stay_tables() == 0


def test_a_preload_failure_leaves_the_next_call_top_level_and_bound(tmp_path, monkeypatch):
    """D8C-RECOVERY-OWNER: a failure before load_concepts' own try."""
    source = _eicu(tmp_path / "eicu")
    resolver = ConceptResolver(load_dictionary(include_sofa2=True))
    real = stay_tables.stay_table_frame

    def fail_once(*args, **kwargs):
        monkeypatch.setattr(stay_tables, "stay_table_frame", real)
        raise OSError("synthetic preload read failure")

    monkeypatch.setattr(stay_tables, "stay_table_frame", fail_once)

    def load(names):
        return resolver.load_concepts(
            names, source, merge=False, interval=pd.Timedelta(hours=1),
            r_compatible=False, verbose=False, concept_workers=1, patient_ids=_ids(),
        )

    with pytest.raises(Exception):
        load(["pap_mean", "crea"])
    assert resolver._load_depth == 0
    # The next call is top-level with its own binding: a rewrite between its
    # reads is refused, not read through unbound.
    _rewrite_after_first_read(monkeypatch, tmp_path / "eicu" / "patient.parquet")
    with pytest.raises(StayTableSourceChanged):
        load(["pap_mean", "crea"])
    assert resolver._load_depth == 0


def test_base_then_dropped_payload_then_derived_still_verifies(tmp_path):
    """D8C-LIVE-BINDING on the streamed vitals path (base, drop, derived)."""
    source = _eicu(tmp_path / "eicu")
    patient = tmp_path / "eicu" / "patient.parquet"
    calls: list = []

    def load_concepts_fn(*, concepts, patient_ids, **_):
        frame = _load(source, ["pap_mean"], patient_ids=patient_ids)
        calls.append(tuple(concepts))
        if len(calls) == 1:
            _touch(patient)  # between the base and the derived concept
        return frame

    with stay_table_unit():
        with pytest.raises(StayTableSourceChanged) as error:
            api._load_stream_module_batch(
                load_concepts_fn, module_name="vitals",
                concepts=["pap_mean", "pulse_pressure"], load_kwargs={},
                patient_ids=_ids(), loader=None, pyarrow_module=pyarrow,
            )
        assert held_stay_tables() == 0  # the boundary dropped the payload
    assert error.value.reason == "stay_table_source_changed"
    assert calls == [("pap_mean",)]


def test_a_dropped_payload_is_reread_and_verified(tmp_path):
    source = _eicu(tmp_path / "eicu")
    before = stay_table_reads()
    with stay_table_unit():
        _load(source, ["pap_mean"], patient_ids=_ids())
        drop_stay_table_frames()
        _load(source, ["pap_mean"], patient_ids=_ids())
        _touch(tmp_path / "eicu" / "patient.parquet")
        drop_stay_table_frames()
        with pytest.raises(StayTableSourceChanged):
            _load(source, ["pap_mean"], patient_ids=_ids())
    assert stay_table_reads() - before == 2


def test_concurrent_units_neither_share_nor_evict_and_each_refuses(tmp_path):
    source = _eicu(tmp_path / "eicu")
    loaded, touched = threading.Barrier(3), threading.Event()
    outcome: dict = {}

    def unit(name, stays):
        try:
            with stay_table_unit():
                _load(source, ["pap_mean"], patient_ids=_ids(stays))
                loaded.wait(timeout=60)
                touched.wait(timeout=60)
                _load(source, ["pap_mean"], patient_ids=_ids(stays))
            outcome[name] = "ok"
        except StayTableSourceChanged as error:
            outcome[name] = error.reason

    before = stay_table_reads()
    threads = [
        threading.Thread(target=unit, args=("a", [1, 2])),
        threading.Thread(target=unit, args=("b", [2, 3])),
    ]
    for thread in threads:
        thread.start()
    loaded.wait(timeout=60)
    assert stay_table_reads() - before == 2  # no shared payload
    assert held_stay_tables() == 2
    _touch(tmp_path / "eicu" / "patient.parquet")
    touched.set()
    for thread in threads:
        thread.join(timeout=60)
    assert outcome == {"a": "stay_table_source_changed", "b": "stay_table_source_changed"}
    assert held_stay_tables() == 0


def test_chunk_threads_of_one_unit_keep_their_own_stay_sets(tmp_path):
    """A loader's chunk threads share the unit's binding without evicting each other."""
    source = _eicu(tmp_path / "eicu")
    both = threading.Barrier(2)
    before = stay_table_reads()

    def chunk(stays):
        _load(source, ["pap_mean"], patient_ids=_ids(stays))
        both.wait(timeout=60)  # both chunks hold their stay sets now
        _load(source, ["pap_mean"], patient_ids=_ids(stays))

    with stay_table_unit():
        threads = [
            threading.Thread(target=contextvars.copy_context().run, args=(chunk, stays))
            for stays in ([1, 2], [2, 3])
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=60)
        assert held_stay_tables() == 2
    assert stay_table_reads() - before == 2  # no chunk re-read its stay set
    assert held_stay_tables() == 0


def _bound_before_and_after_a_rewrite(start, patient: Path, events):
    """Unit a binds, the stay table is rewritten, unit b binds (and drops its payload)."""
    a_loaded, b_loaded, go = events
    start("a", a_loaded, False)
    assert a_loaded.wait(timeout=300)
    _touch(patient)
    start("b", b_loaded, True)
    assert b_loaded.wait(timeout=300)
    return go


def test_threads_bound_before_and_after_a_rewrite_keep_their_own_bindings(tmp_path):
    root = tmp_path / "eicu"
    _eicu(root)
    events = (threading.Event(), threading.Event(), threading.Event())
    results: queue.Queue = queue.Queue()
    threads = []

    def start(tag, loaded, drop):
        thread = threading.Thread(target=load_twice_in_own_unit, args=(
            str(root), ["pap_mean"], [1, 2], loaded, events[2], results, tag, drop))
        thread.start()
        threads.append(thread)

    go = _bound_before_and_after_a_rewrite(start, root / "patient.parquet", events)
    # b dropped its own payload: a's frame is still held.
    assert held_stay_tables() == 1
    go.set()
    for thread in threads:
        thread.join(timeout=300)
    outcome = dict(results.get(timeout=5) for _ in threads)
    # a verifies against its own (older) binding and refuses; b reads its
    # own snapshot, never a's binding.
    assert outcome == {"a": "stay_table_source_changed", "b": "ok"}
    assert held_stay_tables() == 0


def test_processes_bound_before_and_after_a_rewrite_keep_their_own_bindings(tmp_path):
    root = tmp_path / "eicu"
    _eicu(root)
    spawn = multiprocessing.get_context("spawn")
    events = (spawn.Event(), spawn.Event(), spawn.Event())
    results = spawn.Queue()
    processes = []

    def start(tag, loaded, drop):
        process = spawn.Process(target=load_twice_in_own_unit, args=(
            str(root), ["pap_mean"], [1, 2], loaded, events[2], results, tag, drop))
        process.start()
        processes.append(process)

    go = _bound_before_and_after_a_rewrite(start, root / "patient.parquet", events)
    go.set()
    outcome = dict(results.get(timeout=300) for _ in processes)
    for process in processes:
        process.join(timeout=60)
        assert process.exitcode == 0
    assert outcome == {"a": "stay_table_source_changed", "b": "ok"}


def _in_child(snapshot, root, results):
    process = multiprocessing.get_context("spawn").Process(
        target=load_under_binding, args=(snapshot, str(root), ["pap_mean"], [1, 2], results),
    )
    process.start()
    process.join(timeout=300)
    assert process.exitcode == 0
    return results.get(timeout=5)


def test_a_child_process_verifies_against_the_parent_binding(tmp_path):
    root = tmp_path / "eicu"
    source = _eicu(root)
    results = multiprocessing.get_context("spawn").Queue()
    with stay_table_unit():
        _load(source, ["pap_mean"], patient_ids=_ids())
        snapshot = child_stay_binding(source)
    assert str(root / "patient.parquet") in snapshot["versions"]
    assert _in_child(snapshot, root, results) == "ok"
    _touch(root / "patient.parquet")
    assert _in_child(snapshot, root, results) == "stay_table_source_changed"
    assert _in_child(None, root, results) == "ok"  # a new run binds the new bytes


def test_isolated_batch_children_receive_the_run_binding(tmp_path, monkeypatch):
    from easyicu.api.concepts import clear_global_loader
    from easyicu.synthetic_sources import generate_source

    root = tmp_path / "hirid"
    generate_source(root, "hirid", stays=6)
    bindings: dict = {"group": [], "batch": []}

    class _Run:
        def __init__(self, *, target, args, daemon):
            self.target, self.args, self.exitcode = target, args, None

        def start(self):
            self.target(*self.args)
            self.exitcode = 0

        def join(self):
            return None

    class Context:
        @staticmethod
        def Process(*, target, args, daemon):
            kind = "batch" if target is api._write_isolated_stream_batch else "group"
            bindings[kind].append(args[-1])
            return _Run(target=target, args=args, daemon=daemon)

    monkeypatch.setattr(api, "_get_extraction_mp_context", lambda _mp: Context())
    monkeypatch.setattr(api, "_ISOLATED_STREAM_BATCH_TARGETS", frozenset({("hirid", "outcome")}))
    try:
        api.extract_database(
            "hirid", data_path=str(root), output_dir=str(tmp_path / "export"),
            modules=["outcome"], native_export_v2=True, stream_output_batches=True,
            batch_size=4, verbose=False,
        )
    finally:
        clear_global_loader()
    [run] = bindings["group"]
    # HiRID's stay-origin table (``general``) is pinned once for the run.
    assert any(Path(p).name.startswith("general") for p in run["versions"])
    assert len(bindings["batch"]) >= 2
    for child in bindings["batch"]:
        assert {p: child["versions"][p] for p in run["versions"]} == run["versions"]


def test_a_read_or_a_nested_load_without_a_unit_is_refused(tmp_path):
    source = _eicu(tmp_path / "eicu")
    resolver = ConceptResolver(load_dictionary(include_sofa2=True))
    with pytest.raises(StayTableUnbound):
        resolver._stay_table(source, "patient")
    resolver._load_depth = 1  # a nested call whose outer unit is gone
    with pytest.raises(StayTableUnbound):
        resolver.load_concepts(["pap_mean"], source, verbose=False, patient_ids=_ids())
