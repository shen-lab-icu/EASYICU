from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest

from easyicu.research_agent.authority.provider_hard_stop import (
    PROVIDER_HARD_STOP_SCHEMA,
)
from easyicu.research_agent.canonical_json import canonical_sha256
from easyicu.webserver import agent_pipeline_runs
from easyicu.webserver.research_run_usage import research_run_usage


@pytest.mark.parametrize("tamper", [None, "digest", "task", "declaration", "escape", "symlink"])
def test_automatic_continuation_accounts_shared_ledger_after_seed_cleanup(tmp_path, tamper):
    from types import SimpleNamespace
    from easyicu.webserver.research_run_usage import bind_shared_provider_task

    original = tmp_path / "run_original/.runtime/provider_hard_stop_ledger.json"
    _ledger(original, [_call(failed=True), _call("provider_recovery_probe")])
    ledger = json.loads(original.read_text())
    ledger["declaration_sha256"] = "d" * 64
    ledger["tasks"][0]["task_id"] = "task"
    ledger.pop("sha256")
    ledger["sha256"] = canonical_sha256(ledger, trailing_newline=True)
    original.write_text(json.dumps(ledger))
    continued = tmp_path / "run_continued"
    task = SimpleNamespace(task_id="task", ledger=SimpleNamespace(path=original, snapshot=lambda: ledger))
    bind_shared_provider_task(continued, task)
    reference = continued / ".runtime/shared_provider_task.json"
    if tamper == "symlink":
        saved = continued / "saved-binding.json"
        saved.write_bytes(reference.read_bytes())
        reference.unlink()
        reference.symlink_to(saved)
    elif tamper:
        binding = json.loads(reference.read_text())
        if tamper == "digest":
            binding["sha256"] = "e" * 64
        else:
            if tamper == "escape":
                outside = tmp_path.parent / f"outside-{tmp_path.name}.json"
                outside.write_bytes(original.read_bytes())
                binding["ledger"] = "../" + outside.name
            else:
                binding[{"task": "task_id", "declaration": "declaration_sha256"}[tamper]] = "changed"
            binding.pop("sha256")
            binding["sha256"] = canonical_sha256(binding)
        reference.write_text(json.dumps(binding))
    usage = research_run_usage(continued)
    if tamper:
        assert usage["accounting_complete"] is False
    else:
        assert usage["calls"] == 2 and usage["failed_calls"] == 1
        assert usage["usage_unknown_calls"] == 1 and usage["estimated_cost_usd"] == pytest.approx(0.05)


def _ledger(path, calls, status="completed"):
    payload = {
        "schema_version": PROVIDER_HARD_STOP_SCHEMA,
        "tasks": [{"status": status, "calls": calls}],
    }
    payload["sha256"] = canonical_sha256(payload, trailing_newline=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def _shared_task(path, calls):
    from types import SimpleNamespace
    _ledger(path, calls)
    payload = json.loads(path.read_text())
    payload["declaration_sha256"] = "d" * 64
    payload["tasks"][0]["task_id"] = "task"
    payload.pop("sha256")
    payload["sha256"] = canonical_sha256(payload, trailing_newline=True)
    path.write_text(json.dumps(payload))
    return SimpleNamespace(task_id="task", ledger=SimpleNamespace(path=path, snapshot=lambda: payload))


def test_later_report_task_cannot_replace_continuation_accounting_lineage(tmp_path):
    from easyicu.webserver.research_run_usage import bind_shared_provider_task
    original = _shared_task(tmp_path / "run_original/.runtime/provider_hard_stop_ledger.json", [_call(failed=True)])
    continued = tmp_path / "run_continued"
    bind_shared_provider_task(continued, original)
    reference = continued / ".runtime/shared_provider_task.json"
    before = reference.read_bytes()
    report = _shared_task(continued / "report_revisions/revision/runtime/provider_hard_stop.json", [_call("writer")])
    bind_shared_provider_task(continued, report)
    assert reference.read_bytes() == before
    assert research_run_usage(continued)["accounted_tokens"] == 300
    later = _shared_task(tmp_path / "run_later/.runtime/provider_hard_stop_ledger.json", [_call("writer")])
    bind_shared_provider_task(continued, later)
    assert reference.read_bytes() == before
    assert research_run_usage(continued)["accounted_tokens"] == 300


def test_shared_ledger_is_counted_once_when_also_a_revision_ledger(tmp_path):
    wrapper = tmp_path / "run_continued"
    report = _shared_task(wrapper / "report_revisions/revision/runtime/provider_hard_stop.json", [_call("writer")])
    # The reader counts a referenced ledger once even when another owned
    # path enumeration reaches it. This reference is synthetic input metadata.
    binding = {"schema_version": "easyicu.shared-provider-task/1", "task_id": "task",
               "declaration_sha256": "d" * 64, "ledger": report.ledger.path.relative_to(tmp_path).as_posix()}
    binding["sha256"] = canonical_sha256(binding)
    (wrapper / ".runtime").mkdir()
    (wrapper / ".runtime/shared_provider_task.json").write_text(json.dumps(binding))
    usage = research_run_usage(wrapper)
    assert usage["calls"] == 1 and usage["accounted_tokens"] == 100


def test_new_report_cannot_disguise_missing_pipeline_usage_as_complete(tmp_path):
    from easyicu.webserver.research_run_usage import bind_shared_provider_task
    wrapper = tmp_path / "run_incomplete"
    report = _shared_task(wrapper / "report_revisions/revision/runtime/provider_hard_stop.json", [_call("writer")])
    bind_shared_provider_task(wrapper, report)
    assert not (wrapper / ".runtime/shared_provider_task.json").exists()
    assert research_run_usage(wrapper)["accounting_complete"] is False


def _call(role="planner", *, failed=False):
    return {
        "role": role,
        "accounted_tokens": 200 if failed else 100,
        "accounted_estimated_cost_usd": 0.04 if failed else 0.01,
        "reported_total_tokens": None if failed else 100,
        "reported_prompt_tokens": None if failed else 80,
        "reported_completion_tokens": None if failed else 20,
        "error_type": "TimeoutError" if failed else None,
        "state": "failed" if failed else "completed",
        "started_at": "2026-09-16T01:00:00+00:00",
        "finished_at": "2026-09-16T01:00:02+00:00",
        "private_unused_payload": "/Users/private/credential",
    }


def test_all_attempts_and_roles_preserve_failed_and_zero_call_denominators(tmp_path):
    _ledger(
        tmp_path / ".runtime/provider_hard_stop_ledger.json",
        [_call(), _call(failed=True)],
    )
    _ledger(tmp_path / ".runtime/provider_hard_stop_retry_a.json", [])
    _ledger(tmp_path / ".runtime/provider_hard_stop_retry_b.json", [_call("writer")])
    _ledger(
        tmp_path / "report_revisions/c/runtime/provider_hard_stop.json", [_call(None)]
    )
    usage = agent_pipeline_runs._provider_usage_projection(tmp_path)
    assert usage["calls"] == 4
    assert usage["accounted_tokens"] == 500
    assert usage["estimated_cost_usd"] == pytest.approx(0.07)
    assert usage["provider_reported_tokens"] == 300
    assert usage["provider_reported_prompt_tokens"] == 240
    assert usage["provider_reported_completion_tokens"] == 60
    assert usage["usage_unknown_calls"] == usage["failed_calls"] == 1
    assert usage["provider_elapsed_seconds"] == 8
    assert len(usage["attempts"]) == 4
    assert usage["attempts"][1]["calls"] == 0
    assert sum(r["calls"] for r in usage["by_role"]) == usage["calls"]
    assert {r["role"] for r in usage["by_role"]} == {
        "planner",
        "writer",
        "unclassified",
    }
    assert "/Users/private" not in json.dumps(usage)


@pytest.mark.parametrize(
    "corruption",
    ["digest", "negative", "nonfinite", "shape", "symlink", "parent_symlink"],
)
def test_untrusted_ledger_never_appears_as_zero_or_partial_cost(tmp_path, corruption):
    _ledger(tmp_path / ".runtime/provider_hard_stop_ledger.json", [_call()])
    path = tmp_path / ".runtime/provider_hard_stop_retry_b.json"
    _ledger(path, [_call("writer")])
    if corruption == "symlink":
        path.unlink()
        path.symlink_to("provider_hard_stop_ledger.json")
    elif corruption == "parent_symlink":
        (tmp_path / ".runtime").rename(tmp_path / "outside")
        (tmp_path / ".runtime").symlink_to(
            tmp_path / "outside", target_is_directory=True
        )
    else:
        payload = json.loads(path.read_text())
        if corruption == "digest":
            payload["tasks"][0]["calls"][0]["accounted_tokens"] += 1
        else:
            call = _call()
            if corruption == "negative":
                call["accounted_tokens"] = -1
            elif corruption == "nonfinite":
                path.write_text('{"cost": NaN}')
            else:
                call = "invalid"
            if corruption != "nonfinite":
                _ledger(path, [call])
        if corruption == "digest":
            path.write_text(json.dumps(payload))
    usage = research_run_usage(tmp_path)
    assert usage["accounting_complete"] is False
    assert usage["calls"] is usage["estimated_cost_usd"] is None


def test_cli_deduplicates_selected_run_and_does_not_modify_ledgers(tmp_path):
    path = tmp_path / ".runtime/provider_hard_stop_ledger.json"
    _ledger(path, [_call()])
    before = path.read_bytes()
    result = subprocess.run(
        [
            sys.executable,
            "tools/summarize_research_run_costs.py",
            str(tmp_path),
            str(tmp_path / "."),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    summary = json.loads(result.stdout)
    assert summary["totals"]["calls"] == 1
    assert summary["copilot_shell_included"] is False
    assert path.read_bytes() == before


def test_missing_ledger_is_unknown_and_cli_fails(tmp_path):
    assert research_run_usage(tmp_path) is None
    result = subprocess.run(
        [sys.executable, "tools/summarize_research_run_costs.py", str(tmp_path)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert json.loads(result.stdout)["totals"]["estimated_cost_usd"] is None


def test_missing_initial_ledger_does_not_publish_only_retry_cost(tmp_path):
    _ledger(tmp_path / ".runtime/provider_hard_stop_retry_b.json", [_call("writer")])
    assert research_run_usage(tmp_path)["accounting_complete"] is False


def test_missing_token_components_remain_unknown(tmp_path):
    call = _call()
    del call["reported_prompt_tokens"]
    _ledger(tmp_path / ".runtime/provider_hard_stop_ledger.json", [call])
    summary = research_run_usage(tmp_path)
    assert summary["provider_reported_tokens"] == 100
    assert summary["provider_reported_prompt_tokens"] is None
    assert summary["provider_reported_completion_tokens"] == 20
