"""A streamed module's terminal batch failure stays typed at the module boundary.

lead-r13 D3-EXCEPT (Codex delta3): ``StreamBatchMemoryExhausted`` and
``StreamBatchWorkerKilled`` are ``RuntimeError`` subclasses, and the module
boundary's ``except Exception`` flattened them to an empty result and a
string. The module manifest now records the failure class, the memory
evidence, the batch it stopped at, the retry budget used and the resume
events; the failure ledger is kept; publication still refuses the module.
Synthetic rows only.
"""

from __future__ import annotations

import json

import pandas as pd

import easyicu
from easyicu.api import extraction as api

STAYS = list(range(1, 11))


def _frame(ids):
    return pd.DataFrame({
        "stay_id": ids, "charttime": [0.0] * len(ids),
        "test_signal": [float(v) * 1.5 for v in ids],
    })


class _Kill:
    def __init__(self, *, target, args, daemon):
        self.exitcode = None

    def start(self):
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


def _module_run(monkeypatch, out, kill_at, *, counter):
    started: list = []

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
    monkeypatch.setattr(api, "_cgroup_oom_kills", counter)
    monkeypatch.setattr(api, "_host_reports_oom_kills", lambda: counter() is not None)
    api._run_module_extraction(
        "test_module", ["test_signal"], "eicu", str(out), {"stay_id": STAYS}, 8,
        str(out), stream_output_batches=True,
    )
    return json.loads((out / "_manifest.json").read_text()), started


def test_a_second_bare_kill_is_recorded_typed_in_the_module_manifest(tmp_path, monkeypatch):
    manifest, started = _module_run(monkeypatch, tmp_path, {1, 3}, counter=lambda: None)
    assert started == [8, 4, 4]
    assert manifest["errors"] and manifest["memory_exhausted"] is False
    failure = manifest["stream_failure"]
    assert failure["exception_type"] == "StreamBatchWorkerKilled"
    assert failure["failure_class"] == "worker_killed_unknown"
    assert failure["memory_evidence"] is None
    assert (failure["batch_index"], failure["start_offset"], failure["batch_size"]) == (2, 4, 4)
    assert failure["worker_exit_code"] == -9
    assert failure["retries_used"] == 1
    assert [e["kind"] for e in manifest["stream_resume_events"]] == ["unverified_kill_retry"]
    # The failure ledger is kept beside the committed part.
    ledger = json.loads((tmp_path / ".test_module.stream_ledger.json").read_text())
    assert ledger["termination"]["failure_class"] == "worker_killed_unknown"
    assert not (tmp_path / "test_module.parquet").exists()


def test_a_one_stay_memory_failure_is_recorded_with_its_evidence(tmp_path, monkeypatch):
    kills = [0]

    def counter():
        kills[0] += 1  # rises across each killed child
        return kills[0]

    manifest, started = _module_run(monkeypatch, tmp_path, set(range(1, 100)), counter=counter)
    assert started == [8, 4, 2, 1]
    failure = manifest["stream_failure"]
    assert failure["exception_type"] == "StreamBatchMemoryExhausted"
    assert failure["failure_class"] == "out_of_memory"
    assert failure["memory_evidence"] == "oom_kill_event"
    assert failure["retries_used"] == 3
    assert [e["kind"] for e in manifest["stream_resume_events"]] == ["isolated_batch_memory_retry"] * 3


def test_publication_refuses_a_module_with_a_stream_failure():
    result = {"modules": {"m": {"errors": ["streamed export(m): stopped"]}}}
    try:
        api._publish_native_export_v2(
            database="eicu", data_path="unused", output_dir="/nonexistent-d3", modules=["m"],
            max_patients=None, result=result,
        )
    except ValueError as exc:
        assert "requires every requested module to finish" in str(exc)
    else:  # pragma: no cover - the refusal is the contract
        raise AssertionError("published a failed module")
