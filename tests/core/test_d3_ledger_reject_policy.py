"""An unreadable stream ledger refuses the unverified-kill retry.

lead-r13 D3 ledger reject policy (Codex delta3): the one half-size retry a
bare kill earns on a host without an OOM counter (ruling C) is spent once a
module's stream ledger records it. A ledger that exists but cannot be read
(bad JSON, an I/O error, not a ledger) cannot prove the retry unspent, so it
refuses the retry; only a ledger that does not exist counts as unused.
Synthetic rows only.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from easyicu.api import extraction as api

LEDGER = ".m.stream_ledger.json"


def _ledger(root: Path, content) -> None:
    folder = root / "m"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / LEDGER).write_text(content if isinstance(content, str) else json.dumps(content))


def test_no_ledger_is_unused(tmp_path):
    assert api._unverified_kill_ledger_state(str(tmp_path), ["m"]) is None
    assert not api._ledgers_record_unverified_kill(str(tmp_path), ["m"])


def test_a_recorded_retry_is_used(tmp_path):
    _ledger(tmp_path, {"resume_events": [{"kind": "unverified_kill_retry"}]})
    assert api._unverified_kill_ledger_state(str(tmp_path), ["m"]) == "used"


@pytest.mark.parametrize(
    "content",
    ["{not json", json.dumps(["not", "a", "ledger"]), json.dumps({"resume_events": "x"}),
     json.dumps({"resume_events": [1]})],
)
def test_an_unreadable_ledger_refuses_the_retry(tmp_path, content):
    _ledger(tmp_path, content)
    assert api._unverified_kill_ledger_state(str(tmp_path), ["m"]) == "stream_ledger_unreadable"
    assert api._ledgers_record_unverified_kill(str(tmp_path), ["m"])


def test_a_ledger_path_that_cannot_be_opened_refuses_the_retry(tmp_path):
    (tmp_path / "m" / LEDGER).mkdir(parents=True)  # IsADirectoryError, an OSError
    assert api._unverified_kill_ledger_state(str(tmp_path), ["m"]) == "stream_ledger_unreadable"


class _BareKilled:
    calls: list = []
    corrupt = False

    def __init__(self, *, target, args, daemon):
        bound = inspect.signature(target).bind(*args).arguments
        _BareKilled.calls.append(bound)
        self.bound = bound
        self.exitcode = None

    def start(self):
        if _BareKilled.corrupt:
            for module, _concepts in self.bound["module_specs"]:
                folder = Path(self.bound["output_root"]) / module
                folder.mkdir(parents=True, exist_ok=True)
                (folder / f".{module}.stream_ledger.json").write_text("{not json")
        self.exitcode = -9

    def join(self):
        return None


def _killed_worker_batch_sizes(monkeypatch, tmp_path, *, corrupt):
    _BareKilled.calls = []
    _BareKilled.corrupt = corrupt
    monkeypatch.setattr(api, "_cgroup_oom_kills", lambda: None)
    monkeypatch.setattr(api, "_host_reports_oom_kills", lambda: False)
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


def test_a_corrupt_ledger_left_by_the_killed_worker_refuses_the_retry(monkeypatch, tmp_path):
    assert len(_killed_worker_batch_sizes(monkeypatch, tmp_path / "clean", corrupt=False)) == 2
    assert len(_killed_worker_batch_sizes(monkeypatch, tmp_path / "corrupt", corrupt=True)) == 1
