"""A recovery path cannot recreate an erased Provider accounting ledger."""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

import pytest

from easyicu.research_agent.authority import provider_hard_stop as owner
from tests.support.provider_recovery import _limits


def open_ledger(path, **kwargs):
    return owner.ProviderHardStopLedger(
        path=path, task_ids=("task-a",), limits=_limits(),
        batch_id="task-a", declaration_sha256="a" * 64, **kwargs,
    )


def test_required_missing_ledger_is_not_created(tmp_path):
    path = tmp_path / "ledger.json"
    with pytest.raises(owner.ProviderHardStopLedgerError, match="missing"):
        open_ledger(path, resume_existing=True, require_existing=True)
    assert not path.exists()
    # Ordinary new-run creation retains its existing explicit default.
    assert open_ledger(path).snapshot()["tasks"][0]["status"] == "pending"


def test_missing_check_occurs_inside_the_exclusive_ledger_lock(tmp_path, monkeypatch):
    path = tmp_path / "ledger.json"
    open_ledger(path)
    original_lock = owner._exclusive_ledger_file_lock
    entered = []
    @contextmanager
    def removed_while_acquiring_lock(candidate):
        with original_lock(candidate):
            entered.append(candidate)
            candidate.unlink()
            yield
    monkeypatch.setattr(owner, "_exclusive_ledger_file_lock", removed_while_acquiring_lock)
    with pytest.raises(owner.ProviderHardStopLedgerError, match="cannot reset accounting"):
        open_ledger(path, resume_existing=True, require_existing=True)
    assert entered == [path]
    assert not path.exists()


def test_required_existing_ledger_preserves_prior_accounting_and_pause(tmp_path):
    path = tmp_path / "ledger.json"
    original = open_ledger(path)
    original.start_task("task-a").pause()
    before = original.snapshot()
    raw = path.read_bytes()
    recovered = open_ledger(path, resume_existing=True, require_existing=True)
    after = recovered.snapshot()
    assert after["tasks"] == before["tasks"]
    assert after["totals"] == before["totals"]
    assert path.read_bytes() == raw


def test_web_review_recovery_requires_the_original_ledger(tmp_path, monkeypatch):
    from easyicu.webserver import agent_review_recovery
    from tests.support.provider_recovery import _recoverable_package_record
    record, _ = _recoverable_package_record(tmp_path)
    ledger_path = Path(record.hard_stop_ledger_path)
    assert not ledger_path.exists()
    monkeypatch.setattr(agent_review_recovery, "get_record", lambda run_id: record)
    monkeypatch.setattr(agent_review_recovery, "pending_from_record", lambda *args, **kwargs: object())
    with pytest.raises(owner.ProviderHardStopLedgerError, match="missing"):
        agent_review_recovery.recover_pending_review(
            record.run_id, provider_environment={}, reviewer_identity="synthetic-reviewer", rejection_only=True,
        )
    assert not ledger_path.exists()
