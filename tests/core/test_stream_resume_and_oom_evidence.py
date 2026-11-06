"""Memory failures are decided from evidence; a retry resumes (lead-r13 #12, #13).

- An exit code (9, 137, a signal) or text in a traceback (``bad_alloc``) is
  not evidence of an out-of-memory failure. Evidence is a cgroup
  ``oom_kill`` event while the worker ran, or a memory exception by type
  (``MemoryError``, DuckDB ``OutOfMemoryException``) through the cause chain.
- A memory-failed streamed module keeps its committed batches; the retry
  continues after the last one at the new batch size, never from batch 1.
- The module file depends only on the batch boundaries: a resumed run equals
  a run replaying the same boundaries, byte for byte.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import sys

import duckdb
import pandas as pd
import pytest

import easyicu
from easyicu.api import extraction as api

STAYS = list(range(1, 11))


# ── classification ─────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("exit_code", "failures", "delta"),
    [
        (9, [], None),
        (137, [], None),
        (-9, [], None),
        (-9, [], 0),
        (1, [{"exception_type": "ValueError", "traceback": "std::bad_alloc"}], None),
        (1, [{"exception_type": "RuntimeError", "traceback": "Out of Memory Error"}], None),
        (3221226505, [], None),
    ],
)
def test_exit_codes_and_text_are_not_memory_evidence(exit_code, failures, delta):
    assert api._worker_memory_evidence(exit_code, failures, delta) is None
    assert api._worker_failure_class(exit_code, failures, delta) != "out_of_memory"


def test_an_unexplained_kill_is_its_own_class():
    assert api._worker_failure_class(-9, [], None) == "worker_killed_unknown"
    assert api._worker_failure_class(137, [], 0) == "worker_killed_unknown"
    assert api._worker_failure_class(1, [], None) == "worker_failed"


def test_memory_evidence():
    assert api._worker_memory_evidence(-9, [], 1) == "oom_kill_event"
    assert api._worker_memory_evidence(1, [{"memory_exhausted": True}], None) == "memory_exception"
    assert api._worker_failure_class(-9, [], 2) == "out_of_memory"


def test_memory_exceptions_by_type_and_cause_chain():
    assert api._is_memory_exception(MemoryError())
    assert api._is_memory_exception(duckdb.OutOfMemoryException("x"))
    try:
        try:
            raise MemoryError()
        except MemoryError as exc:
            raise RuntimeError("wrapped") from exc
    except RuntimeError as wrapped:
        assert api._is_memory_exception(wrapped)
    assert not api._is_memory_exception(ValueError("std::bad_alloc out of memory"))
    assert not api._is_memory_exception(None)


def test_module_memory_errors_read_the_flag_not_the_text(tmp_path):
    for module, manifest in {
        "text_only": {"errors": ["DuckDB: Out of Memory Error"], "memory_exhausted": False},
        "flagged": {"errors": ["streamed module export exhausted memory"], "memory_exhausted": True},
    }.items():
        (tmp_path / module).mkdir()
        (tmp_path / module / "_manifest.json").write_text(json.dumps(manifest))
    assert api._module_memory_errors(str(tmp_path), ["text_only", "flagged"]) == ["flagged"]


# ── resume ──────────────────────────────────────────────────────────────


def _frame(ids):
    return pd.DataFrame({
        "stay_id": ids,
        "charttime": [0.0] * len(ids),
        "test_signal": [float(v) * 1.5 for v in ids],
    })


def _run(tmp_path, monkeypatch, batch_size, *, fail_on_call=None, replay=None, calls=None):
    seen = calls if calls is not None else []

    def fake_load_concepts(**kwargs):
        ids = list(kwargs["patient_ids"]["stay_id"])
        seen.append(ids)
        if fail_on_call is not None and len(seen) == fail_on_call:
            raise MemoryError("simulated")
        return _frame(ids)

    monkeypatch.setattr(easyicu, "load_concepts", fake_load_concepts)
    return api._stream_module_batches_to_parquet(
        "test_module", ["test_signal"],
        {"database": "miiv", "concepts": ["test_signal"]},
        {"stay_id": STAYS}, batch_size, str(tmp_path),
        replay_boundaries=replay,
    )


def _sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def test_a_memory_failure_keeps_committed_batches_and_resumes(tmp_path, monkeypatch):
    out = tmp_path / "resumed"
    out.mkdir()
    calls: list = []
    with pytest.raises(MemoryError):
        _run(out, monkeypatch, 4, fail_on_call=2, calls=calls)
    # Batch 1 is committed and recorded; nothing is published yet.
    ledger = json.loads((out / ".test_module.stream_ledger.json").read_text())
    assert [b["stays"] for b in ledger["batches"]] == [4]
    assert ledger["pending_reason"] == "memory_exception"
    assert not (out / "test_module.parquet").exists()

    resumed_calls: list = []
    info = _run(out, monkeypatch, 2, calls=resumed_calls)
    # The retry never re-extracts batch 1.
    first = set(calls[0])
    assert all(not (set(ids) & first) for ids in resumed_calls)
    assert [b["stays"] for b in info["stream_batches"]] == [4, 2, 2, 2]
    assert info["stream_resume_events"][0]["kind"] == "resume_after_worker_retry"
    assert info["stream_resume_events"][0]["start_offset"] == 4
    # No parts, ledger or order file remain.
    assert sorted(p.name for p in out.iterdir()) == ["test_module.parquet"]

    # Same content as an uninterrupted run, and the same bytes as a run that
    # replays the recorded boundaries.
    fresh = tmp_path / "fresh"
    fresh.mkdir()
    plain = _run(fresh, monkeypatch, 4)
    pd.testing.assert_frame_equal(
        pd.read_parquet(info["path"]).sort_values("stay_id").reset_index(drop=True),
        pd.read_parquet(plain["path"]).sort_values("stay_id").reset_index(drop=True),
        check_exact=True,
    )
    replay = tmp_path / "replay"
    replay.mkdir()
    replayed = _run(replay, monkeypatch, 4, replay=[4, 2, 2, 2])
    assert _sha(replayed["path"]) == _sha(info["path"])


def test_a_ledger_for_another_request_is_refused(tmp_path, monkeypatch):
    with pytest.raises(MemoryError):
        _run(tmp_path, monkeypatch, 4, fail_on_call=2)
    ledger_path = tmp_path / ".test_module.stream_ledger.json"
    ledger = json.loads(ledger_path.read_text())
    ledger["concepts"] = ["another_concept"]
    ledger_path.write_text(json.dumps(ledger))
    with pytest.raises(ValueError, match="does not match"):
        _run(tmp_path, monkeypatch, 2)


def test_a_non_memory_failure_cleans_up(tmp_path, monkeypatch):
    def broken(**kwargs):
        raise ValueError("schema fault")

    monkeypatch.setattr(easyicu, "load_concepts", broken)
    with pytest.raises(ValueError):
        api._stream_module_batches_to_parquet(
            "test_module", ["test_signal"], {"database": "miiv", "concepts": ["test_signal"]},
            {"stay_id": STAYS}, 4, str(tmp_path),
        )
    assert list(tmp_path.iterdir()) == []


def test_parts_assemble_to_the_bytes_the_single_writer_produced(tmp_path, monkeypatch):
    """An in-process part is written back as one table (the former layout)."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    out = tmp_path / "parts"
    out.mkdir()
    seen = []
    info = _run(out, monkeypatch, 4, calls=seen)
    reference = tmp_path / "reference.parquet"
    writer = None
    schema = None
    # The reference writes the batches the planner actually produced, so the
    # check holds for any batch order (contiguous since r13varlay).
    for ids in seen:
        table = api._module_arrow_table(
            api._normalise_module_frame_for_parquet(_frame(ids), ["test_signal"], reorder=False),
            ["test_signal"], pa, module="test_module", schema=schema,
        )
        if writer is None:
            schema = table.schema
            writer = pq.ParquetWriter(reference, schema, compression="snappy")
        writer.write_table(table)
    writer.close()
    assert [b["stays"] for b in info["stream_batches"]] == [4, 4, 2]
    assert _sha(info["path"]) == _sha(reference)


# ── isolated batches ────────────────────────────────────────────────────


KILLS = [0]


class _Kill:
    """A batch child the OOM killer stops (no failure record, exit -9)."""

    def __init__(self, *, target, args, daemon):
        self.target, self.args = target, args
        self.exitcode = None

    def start(self):
        KILLS[0] += 1
        self.exitcode = -9

    def join(self):
        return None


class _Run:
    def __init__(self, *, target, args, daemon):
        self.target, self.args = target, args
        self.exitcode = None

    def start(self):
        try:
            self.target(*self.args)
            self.exitcode = 0
        except BaseException:
            self.exitcode = 1

    def join(self):
        return None


def _isolated(monkeypatch, tmp_path, kill_first, kills_counter):
    started = []

    class Context:
        @staticmethod
        def Process(*, target, args, daemon):
            started.append(len(args[3]["stay_id"]))
            cls = _Kill if len(started) <= kill_first else _Run
            return cls(target=target, args=args, daemon=daemon)

    monkeypatch.setattr(easyicu, "load_concepts", lambda **kw: _frame(list(kw["patient_ids"]["stay_id"])))
    monkeypatch.setattr(api, "_extract_worker_env_setup", lambda _path: None)
    monkeypatch.setattr(api, "_get_extraction_mp_context", lambda _mp: Context())
    monkeypatch.setattr(api, "_ISOLATED_STREAM_BATCH_TARGETS", frozenset({("eicu", "test_module")}))
    monkeypatch.setattr(api, "_cgroup_oom_kills", kills_counter)
    info = api._stream_module_batches_to_parquet(
        "test_module", ["test_signal"], {"database": "eicu", "concepts": ["test_signal"]},
        {"stay_id": STAYS}, 8, str(tmp_path),
    )
    return info, started


def test_an_oom_killed_isolated_batch_halves_from_the_same_offset(tmp_path, monkeypatch):
    KILLS[0] = 0
    info, started = _isolated(monkeypatch, tmp_path, 1, lambda: KILLS[0])
    assert started[:2] == [8, 4]
    event = info["stream_resume_events"][0]
    assert event["kind"] == "isolated_batch_memory_retry"
    assert (event["start_offset"], event["previous_batch_size"], event["batch_size"]) == (0, 8, 4)
    assert event["evidence"] == "oom_kill_event"
    exported = pd.read_parquet(info["path"])
    assert sorted(exported["stay_id"]) == STAYS


def test_an_isolated_kill_without_evidence_is_not_retried(tmp_path, monkeypatch):
    # A host with a cgroup counter (Linux) that recorded no OOM kill.
    with pytest.raises(api.StreamBatchWorkerKilled, match="worker_killed_unknown"):
        _isolated(monkeypatch, tmp_path, 1, lambda: 0)


def _isolated_run(monkeypatch, tmp_path, batch_size, *, replay=None, fail_after=None):
    """An isolated (eICU-style, append-after-each-child) streamed export."""
    extracted: list = []
    cleared = [0]
    in_child = [False]

    class Child(_Run):
        def start(self):
            in_child[0] = True
            try:
                super().start()
            finally:
                in_child[0] = False

    class Context:
        @staticmethod
        def Process(*, target, args, daemon):
            extracted.append(list(args[3]["stay_id"]))
            return Child(target=target, args=args, daemon=daemon)

    real_clear = api._clear_stream_loader_caches

    def clear(loader):
        if in_child[0]:
            return real_clear(loader)
        cleared[0] += 1
        if fail_after is not None and cleared[0] == fail_after:
            # The parent runs out of memory between batches.
            raise MemoryError("simulated parent")
        return real_clear(loader)

    monkeypatch.setattr(easyicu, "load_concepts", lambda **kw: _frame(list(kw["patient_ids"]["stay_id"])))
    monkeypatch.setattr(api, "_extract_worker_env_setup", lambda _path: None)
    monkeypatch.setattr(api, "_get_extraction_mp_context", lambda _mp: Context())
    monkeypatch.setattr(api, "_ISOLATED_STREAM_BATCH_TARGETS", frozenset({("eicu", "test_module")}))
    monkeypatch.setattr(api, "_clear_stream_loader_caches", clear)
    info = api._stream_module_batches_to_parquet(
        "test_module", ["test_signal"], {"database": "eicu", "concepts": ["test_signal"]},
        {"stay_id": STAYS}, batch_size, str(tmp_path), replay_boundaries=replay,
    )
    return info, extracted


def test_an_isolated_module_resumes_after_the_last_committed_batch(tmp_path, monkeypatch):
    out = tmp_path / "resumed"
    out.mkdir()
    with pytest.raises(MemoryError):
        _isolated_run(monkeypatch, out, 4, fail_after=1)
    ledger = json.loads((out / ".test_module.stream_ledger.json").read_text())
    assert [b["stays"] for b in ledger["batches"]] == [4]
    assert ledger["batches"][0]["batch_process_isolation"] is True
    assert (out / ".test_module.batch-00001.parquet").exists()

    info, extracted = _isolated_run(monkeypatch, out, 2)
    assert [len(ids) for ids in extracted] == [2, 2, 2]
    assert [b["stays"] for b in info["stream_batches"]] == [4, 2, 2, 2]
    assert info["batch_process_isolation"] is True
    assert sorted(p.name for p in out.iterdir()) == ["test_module.parquet"]

    replay = tmp_path / "replay"
    replay.mkdir()
    replayed, _ = _isolated_run(monkeypatch, replay, 4, replay=[4, 2, 2, 2])
    assert _sha(replayed["path"]) == _sha(info["path"])
    assert sorted(pd.read_parquet(info["path"])["stay_id"]) == STAYS


# ── hosts without an OOM counter (macOS, Windows; #13 ruling C) ─────────


def _no_counter(monkeypatch):
    monkeypatch.setattr(api, "_cgroup_oom_kills", lambda: None)
    monkeypatch.setattr(api, "_host_reports_oom_kills", lambda: False)


def test_the_host_reports_oom_kills_from_its_counter_or_linux(monkeypatch):
    monkeypatch.setattr(api, "_cgroup_oom_kills", lambda: 0)
    assert api._host_reports_oom_kills() is True
    monkeypatch.setattr(api, "_cgroup_oom_kills", lambda: None)
    # Linux stays evidence-only even without a readable counter.
    monkeypatch.setattr(sys, "platform", "linux")
    assert api._host_reports_oom_kills() is True
    for platform in ("darwin", "win32"):
        monkeypatch.setattr(sys, "platform", platform)
        assert api._host_reports_oom_kills() is False


@pytest.mark.parametrize("exit_code", [-9, 137, 0xC0000017])
def test_a_bare_kill_is_eligible_only_without_a_counter(monkeypatch, exit_code):
    _no_counter(monkeypatch)
    assert api._unverified_kill_retry_eligible(exit_code, [])
    # A worker that recorded its own exception is not a bare kill.
    assert not api._unverified_kill_retry_eligible(exit_code, [{"exception_type": "ValueError"}])
    # A plain failure exit is not a kill.
    assert not api._unverified_kill_retry_eligible(1, [])
    # Unexplained kills stay typed.
    assert api._worker_failure_class(exit_code, []) == "worker_killed_unknown"
    monkeypatch.setattr(api, "_host_reports_oom_kills", lambda: True)
    assert not api._unverified_kill_retry_eligible(exit_code, [])


@pytest.mark.parametrize(
    "exit_code",
    [
        0xC0000409,  # Windows fail-fast crash, not a kill
        -15,  # multiprocessing terminate() (TerminateProcess maps to -SIGTERM)
        1,  # an external TerminateProcess / taskkill /F exit code
    ],
)
def test_crashes_and_terminations_are_not_bare_kills(monkeypatch, exit_code):
    _no_counter(monkeypatch)
    assert not api._unverified_kill_retry_eligible(exit_code, [])


def _isolated_kills(monkeypatch, tmp_path, kill_at, *, replay=None, batch_size=8):
    """Isolated export whose children numbered in ``kill_at`` are bare-killed."""
    started = []

    class Context:
        @staticmethod
        def Process(*, target, args, daemon):
            started.append(len(args[3]["stay_id"]))
            cls = _Kill if len(started) in kill_at else _Run
            return cls(target=target, args=args, daemon=daemon)

    monkeypatch.setattr(easyicu, "load_concepts", lambda **kw: _frame(list(kw["patient_ids"]["stay_id"])))
    monkeypatch.setattr(api, "_extract_worker_env_setup", lambda _path: None)
    monkeypatch.setattr(api, "_get_extraction_mp_context", lambda _mp: Context())
    monkeypatch.setattr(api, "_ISOLATED_STREAM_BATCH_TARGETS", frozenset({("eicu", "test_module")}))
    info = api._stream_module_batches_to_parquet(
        "test_module", ["test_signal"], {"database": "eicu", "concepts": ["test_signal"]},
        {"stay_id": STAYS}, batch_size, str(tmp_path), replay_boundaries=replay,
    )
    return info, started


def test_a_bare_kill_without_a_counter_gets_one_half_size_retry(tmp_path, monkeypatch):
    _no_counter(monkeypatch)
    out = tmp_path / "retried"
    out.mkdir()
    info, started = _isolated_kills(monkeypatch, out, {1})
    assert started == [8, 4, 4, 2]
    [event] = info["stream_resume_events"]
    assert event["kind"] == "unverified_kill_retry"
    assert (event["start_offset"], event["previous_batch_size"], event["batch_size"]) == (0, 8, 4)
    assert event["worker_exit_code"] == -9
    assert event["oom_kill_counter"] == "unavailable"
    assert "process_tree_peak_rss_mb" in event and "host_total_memory_mb" in event
    assert [b["stays"] for b in info["stream_batches"]] == [4, 4, 2]

    # Byte-identical to a run with the same boundaries and no kill.
    replay = tmp_path / "replay"
    replay.mkdir()
    replayed, _ = _isolated_kills(monkeypatch, replay, set(), replay=[4, 4, 2])
    assert _sha(replayed["path"]) == _sha(info["path"])
    assert sorted(p.name for p in out.iterdir()) == ["test_module.parquet"]


def test_a_second_bare_kill_in_the_module_fails_typed(tmp_path, monkeypatch):
    _no_counter(monkeypatch)
    with pytest.raises(api.StreamBatchWorkerKilled) as caught:
        _isolated_kills(monkeypatch, tmp_path, {1, 3})
    assert caught.value.failure_class == "worker_killed_unknown"
    assert caught.value.retries_used == 1
    # The terminated module keeps its failure ledger (lead-r13 D3-EXCEPT);
    # nothing is published.
    ledger = json.loads((tmp_path / ".test_module.stream_ledger.json").read_text())
    assert ledger["pending_reason"] == "worker_killed_unknown"
    assert ledger["termination"]["failure_class"] == "worker_killed_unknown"
    assert [e["kind"] for e in ledger["resume_events"]] == ["unverified_kill_retry"]
    assert not (tmp_path / "test_module.parquet").exists()


def test_a_bare_kill_with_a_counter_is_not_retried(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "_cgroup_oom_kills", lambda: 0)
    monkeypatch.setattr(api, "_host_reports_oom_kills", lambda: True)
    with pytest.raises(api.StreamBatchWorkerKilled, match="worker_killed_unknown"):
        _isolated_kills(monkeypatch, tmp_path, {1})


def test_a_module_worker_kill_resumes_once_from_committed_batches(tmp_path, monkeypatch):
    """The module-worker level: held ledger carries the retry; resume is exact."""
    out = tmp_path / "resumed"
    out.mkdir()
    with pytest.raises(MemoryError):
        _run(out, monkeypatch, 4, fail_on_call=2)
    # A SIGKILL leaves committed parts and a ledger with no pending reason.
    ledger_path = out / ".test_module.stream_ledger.json"
    ledger = json.loads(ledger_path.read_text())
    ledger["pending_reason"] = None
    ledger_path.write_text(json.dumps(ledger))
    group_root = tmp_path / "group"
    group_root.mkdir()
    (out).rename(group_root / "test_module")
    assert not api._ledgers_record_unverified_kill(str(group_root), ["test_module"])
    held = api._hold_resumable_module_dirs(str(group_root), ["test_module"])
    api._record_unverified_kill_in_ledgers(held, {
        "kind": "unverified_kill_retry", "level": "module_worker",
        "previous_batch_size": 4, "batch_size": 2, "worker_exit_code": -9,
        "oom_kill_counter": "unavailable",
        "process_tree_peak_rss_mb": 1.0, "host_total_memory_mb": 2.0,
    })
    folder = held["test_module"]
    assert api._ledgers_record_unverified_kill(str(pathlib.Path(folder).parent), ["test_module"])

    info = _run(pathlib.Path(folder), monkeypatch, 2)
    kinds = [e["kind"] for e in info["stream_resume_events"]]
    assert kinds == ["unverified_kill_retry", "resume_after_worker_retry"]
    assert info["stream_resume_events"][0]["start_offset"] == 4
    assert info["stream_resume_events"][1]["reason"] == "unverified_kill"
    replay = tmp_path / "replay"
    replay.mkdir()
    replayed = _run(replay, monkeypatch, 4, replay=[4, 2, 2, 2])
    assert _sha(replayed["path"]) == _sha(info["path"])


class _BareKilled:
    calls: list = []

    def __init__(self, *, target, args, daemon):
        import inspect

        _BareKilled.calls.append(inspect.signature(target).bind(*args).arguments)
        self.exitcode = None

    def start(self):
        self.exitcode = -9

    def join(self):
        return None


def _extract_killed(monkeypatch, tmp_path):
    from types import SimpleNamespace

    _BareKilled.calls = []
    monkeypatch.setattr(
        api, "_get_extraction_mp_context", lambda _: SimpleNamespace(Process=_BareKilled)
    )
    with pytest.raises(api.NativeExportPublicationError):
        api.extract_database(
            "aumc", data_path=tmp_path, output_dir=tmp_path / "output",
            modules=["demographics"], patient_ids={"admissionid": list(range(23106))},
            resource_budget_mb=8192, stream_output_batches=True, verbose=False,
        )
    return [call["batch_size"] for call in _BareKilled.calls]


def test_a_module_worker_bare_kill_without_a_counter_retries_once(monkeypatch, tmp_path):
    _no_counter(monkeypatch)
    sizes = _extract_killed(monkeypatch, tmp_path)
    # One retry at half the batch; the second bare kill is final.
    assert len(sizes) == 2 and sizes[1] == max(1, sizes[0] // 2)


def test_a_module_worker_bare_kill_with_a_counter_is_not_retried(monkeypatch, tmp_path):
    monkeypatch.setattr(api, "_cgroup_oom_kills", lambda: 0)
    sizes = _extract_killed(monkeypatch, tmp_path)
    assert len(sizes) == 1
