"""Batch-level retry events reach the module manifest and the release receipt.

lead-r13 D3-LEDGER (Codex delta3): an isolated batch's retry is recorded in
the stream ledger's ``resume_events`` and the worker's module manifest, but
the collector rebuilt the module result and published manifest without it,
and the run's ``stream_retry_history`` held only module-worker retries. The
test runs the whole chain (planner, group worker, isolated batch children,
collector, native-v2 publication) on a synthetic source, with one isolated
child stopped by an attributed OOM kill. Synthetic rows only.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from easyicu.api import extraction as api

DATABASE = "hirid"
MODULE = "outcome"


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


class _OomKilled:
    def __init__(self, *, target, args, daemon):
        self.exitcode = None

    def start(self):
        KILLS[0] += 1
        self.exitcode = -9

    def join(self):
        return None


KILLS = [0]


@pytest.fixture
def chain(tmp_path, monkeypatch):
    from easyicu.api.concepts import clear_global_loader
    from easyicu.synthetic_sources import generate_source

    root = tmp_path / DATABASE
    generate_source(root, DATABASE, stays=6)
    batch_children: list = []

    class Context:
        @staticmethod
        def Process(*, target, args, daemon):
            if target is api._write_isolated_stream_batch:
                batch_children.append(len(next(iter(args[3].values()))))
                if len(batch_children) == 1:
                    return _OomKilled(target=target, args=args, daemon=daemon)
            return _Run(target=target, args=args, daemon=daemon)

    KILLS[0] = 0
    monkeypatch.setattr(api, "_get_extraction_mp_context", lambda _mp: Context())
    monkeypatch.setattr(api, "_ISOLATED_STREAM_BATCH_TARGETS", frozenset({(DATABASE, MODULE)}))
    monkeypatch.setattr(api, "_cgroup_oom_kills", lambda: KILLS[0])
    output = tmp_path / "export"
    try:
        result = api.extract_database(
            DATABASE, data_path=str(root), output_dir=str(output), modules=[MODULE],
            native_export_v2=True, stream_output_batches=True, batch_size=4, verbose=False,
        )
    finally:
        clear_global_loader()
    return result, output, batch_children


def test_an_isolated_batch_retry_reaches_every_layer(chain):
    result, output, batch_children = chain
    assert batch_children[:2] == [4, 2]

    events = result["modules"][MODULE]["stream_resume_events"]
    assert [e["kind"] for e in events] == ["isolated_batch_memory_retry"]
    assert events[0]["evidence"] == "oom_kill_event"

    module_manifest = json.loads((output / f"{MODULE}.manifest.json").read_text())
    assert module_manifest["stream_resume_events"] == events

    [entry] = result["stream_retry_history"]
    assert entry["level"] == "isolated_batch"
    assert entry["event_source"] == "stream_ledger.resume_events"
    assert (entry["policy"], entry["previous_batch_size"], entry["retry_batch_size"]) == (
        "isolated_batch_memory_retry", 4, 2,
    )
    assert entry["memory_evidence"] == "oom_kill_event"

    receipt = json.loads((output / "_manifest.json").read_text())
    assert receipt["stream_retry_history"] == result["stream_retry_history"]
    assert receipt["stream_resume_events"] == {MODULE: events}


def test_a_terminated_module_carries_its_typed_failure_and_is_not_published(tmp_path, monkeypatch):
    """No OOM counter (ruling C): the second bare kill ends the module, typed."""
    from easyicu.api.concepts import clear_global_loader
    from easyicu.synthetic_sources import generate_source

    root = tmp_path / DATABASE
    generate_source(root, DATABASE, stays=6)

    class Context:
        @staticmethod
        def Process(*, target, args, daemon):
            if target is api._write_isolated_stream_batch:
                return _OomKilled(target=target, args=args, daemon=daemon)
            return _Run(target=target, args=args, daemon=daemon)

    monkeypatch.setattr(api, "_get_extraction_mp_context", lambda _mp: Context())
    monkeypatch.setattr(api, "_ISOLATED_STREAM_BATCH_TARGETS", frozenset({(DATABASE, MODULE)}))
    monkeypatch.setattr(api, "_cgroup_oom_kills", lambda: None)
    monkeypatch.setattr(api, "_host_reports_oom_kills", lambda: False)
    output = tmp_path / "export"
    try:
        with pytest.raises(api.NativeExportPublicationError) as caught:
            api.extract_database(
                DATABASE, data_path=str(root), output_dir=str(output), modules=[MODULE],
                native_export_v2=True, stream_output_batches=True, batch_size=4, verbose=False,
            )
    finally:
        clear_global_loader()
    assert not (output / "_manifest.json").exists()
    [module_manifest] = list(Path(caught.value.quarantine).rglob(f"{MODULE}.manifest.json"))
    published = json.loads(module_manifest.read_text())
    failure = published["stream_failure"]
    assert failure["exception_type"] == "StreamBatchWorkerKilled"
    assert failure["failure_class"] == "worker_killed_unknown"
    assert failure["retries_used"] == 1
    assert [e["kind"] for e in published["stream_resume_events"]] == ["unverified_kill_retry"]
