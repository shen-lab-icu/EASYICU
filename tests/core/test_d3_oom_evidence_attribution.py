"""A recorded non-memory exception always wins over the cgroup counter.

lead-r13 D3-OOM-EVIDENCE (Codex delta3): the cgroup ``oom_kill`` counter is
shared by every process in the cgroup, so its increment while a worker ran
may be another process's kill. It may classify only a worker the OOM killer
could have stopped (SIGKILL) that left no exception record; a worker that
recorded a ``ValueError`` failed with a ``ValueError``. Synthetic rows only.
"""

from __future__ import annotations

import pytest

import easyicu
from easyicu.api import extraction as api

STAYS = list(range(1, 9))


@pytest.mark.parametrize("exit_code", [1, -9, 137])
def test_a_recorded_value_error_is_not_out_of_memory(exit_code):
    record = {"exception_type": "ValueError", "memory_exhausted": False}
    assert api._worker_memory_evidence(exit_code, [record], oom_kill_delta=1) is None
    assert api._worker_failure_class(exit_code, [record], 1) != "out_of_memory"
    assert not api.worker_failure_is_memory(
        {**record, "worker_exit_code": exit_code, "oom_kill_delta": 1}
    )


@pytest.mark.parametrize("exit_code", [1, -15, 143, -11, 0xC0000017])
def test_the_counter_explains_only_a_sigkill(exit_code):
    assert api._worker_memory_evidence(exit_code, [], oom_kill_delta=1) is None


@pytest.mark.parametrize("exit_code", [-9, 137])
def test_an_unrecorded_sigkill_with_a_rising_counter_is_out_of_memory(exit_code):
    parent_record = {"exception_type": None, "failure_kind": "worker_exit_without_failure_record"}
    assert api._worker_memory_evidence(exit_code, [parent_record], 1) == "oom_kill_event"
    assert api._worker_memory_evidence(exit_code, [], 1) == "oom_kill_event"
    assert api._worker_failure_class(exit_code, [], 1) == "out_of_memory"


def test_a_memory_exception_record_is_memory_without_the_counter():
    assert api._worker_memory_evidence(1, [{"exception_type": "MemoryError"}], 0) == "memory_exception"
    assert api._worker_memory_evidence(-9, [{"memory_exhausted": True}], None) == "memory_exception"


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


def test_an_isolated_batch_value_error_is_not_retried_as_memory(tmp_path, monkeypatch):
    """Another process's OOM kill during the child does not relabel its ValueError."""
    started: list = []
    counter = [0]

    class Context:
        @staticmethod
        def Process(*, target, args, daemon):
            started.append(len(args[3]["stay_id"]))
            return _Run(target=target, args=args, daemon=daemon)

    def shared_counter():
        counter[0] += 1  # rises across every child: a neighbour's kills
        return counter[0]

    def broken(**_kwargs):
        raise ValueError("schema fault")

    monkeypatch.setattr(easyicu, "load_concepts", broken)
    monkeypatch.setattr(api, "_extract_worker_env_setup", lambda _path: None)
    monkeypatch.setattr(api, "_get_extraction_mp_context", lambda _mp: Context())
    monkeypatch.setattr(api, "_ISOLATED_STREAM_BATCH_TARGETS", frozenset({("eicu", "test_module")}))
    monkeypatch.setattr(api, "_cgroup_oom_kills", shared_counter)
    with pytest.raises(api.StreamBatchWorkerKilled) as caught:
        api._stream_module_batches_to_parquet(
            "test_module", ["test_signal"], {"database": "eicu", "concepts": ["test_signal"]},
            {"stay_id": STAYS}, 8, str(tmp_path),
        )
    assert caught.value.failure_class == "worker_failed"
    assert started == [8]  # never halved as a memory failure
