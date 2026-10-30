"""Read-only, digest-verified usage across one run and its continuations."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from itertools import islice
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping

from easyicu.model_request_metadata import project_request
from easyicu.research_agent.authority.provider_hard_stop import (
    ProviderHardStopLedgerError,
    is_pre_transport_capacity_rejection,
    load_provider_hard_stop_ledger,
)
from easyicu.research_agent.canonical_json import canonical_sha256

MAX_USAGE_LEDGERS = 64
MAX_USAGE_BYTES = 8 * 1024 * 1024
MAX_USAGE_CALLS = 10_000


def _call_state(call: Mapping[str, Any]) -> str:
    state = str(call.get("state") or "")
    if call.get("error_type") or state.startswith("failed"):
        return "failed"
    if state == "in_progress":
        return "in_flight"
    if state.startswith("completed"):
        return "completed"
    return "unknown"


def bind_shared_provider_task(wrapper_dir: Path, task: Any) -> None:
    """Retain accounting lineage after the temporary launch seed is pruned."""
    root = wrapper_dir.resolve()
    if any(path.exists() or path.is_symlink() for path in (
        root / ".runtime/provider_hard_stop_ledger.json",
        root / ".runtime/shared_provider_task.json",
    )):
        return
    # Only a continuation in another wrapper imports the original ledger.
    # A new report/retry ledger cannot stand in for missing pipeline usage.
    if task.ledger.path.resolve().is_relative_to(root):
        return
    ledger = task.ledger.snapshot()
    payload = {
        "schema_version": "easyicu.shared-provider-task/1",
        "ledger": task.ledger.path.resolve().relative_to(root.parent).as_posix(),
        "task_id": task.task_id,
        "declaration_sha256": ledger["declaration_sha256"],
    }
    payload["sha256"] = canonical_sha256(payload)
    runtime = root / ".runtime"
    runtime.mkdir(parents=True, exist_ok=True)
    (runtime / "shared_provider_task.json").write_text(json.dumps(payload) + "\n", encoding="utf-8")


def _number(value: Any, *, integer: bool = False) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("invalid usage number")
    if not math.isfinite(value) or value < 0 or (integer and int(value) != value):
        raise ValueError("invalid usage number")
    return int(value) if integer else float(value)


def _summarize(calls: list[Mapping[str, Any]], *, group_states: bool = True) -> dict[str, Any]:
    reported = [call for call in calls if call.get("reported_total_tokens") is not None]
    unsent = [call for call in calls if is_pre_transport_capacity_rejection(call)]
    durations = []
    for call in calls:
        if call.get("started_at") and call.get("finished_at"):
            duration = (
                datetime.fromisoformat(call["finished_at"])
                - datetime.fromisoformat(call["started_at"])
            ).total_seconds()
            durations.append(_number(duration))
    return {
        "calls": len(calls),
        "accounted_tokens": sum(
            _number(c["accounted_tokens"], integer=True) for c in calls
        ),
        "estimated_cost_usd": round(
            sum(_number(c["accounted_estimated_cost_usd"]) for c in calls), 8
        ) if all(c.get("accounted_estimated_cost_usd") is not None for c in calls) else None,
        "provider_reported_calls": len(reported),
        "provider_reported_tokens": sum(
            _number(c["reported_total_tokens"], integer=True) for c in reported
        ),
        **{
            f"provider_{key}": (
                sum(_number(c[key], integer=True) for c in reported)
                if all(c.get(key) is not None for c in reported)
                else None
            )
            for key in ("reported_prompt_tokens", "reported_completion_tokens")
        },
        "usage_unknown_calls": len(calls) - len(reported) - len(unsent),
        "pre_transport_capacity_rejections": len(unsent),
        "failed_calls": sum(
            _call_state(c) == "failed" for c in calls
        ),
        "in_flight_calls": sum(_call_state(c) == "in_flight" for c in calls),
        "provider_elapsed_seconds": round(sum(durations), 6),
        "timed_calls": len(durations),
        **({"by_state": {
            state: _summarize([c for c in calls if _call_state(c) == state], group_states=False)
            for state in ("completed", "failed", "in_flight", "unknown")
        }} if group_states else {}),
    }


def research_run_usage(wrapper_dir: Path, *, include_requests: bool = False) -> dict[str, Any] | None:
    """Include every owned ledger once, retaining failed and zero-call attempts.

    This accounts for this wrapper, its execution retries and report revisions.
    Earlier candidate wrappers and the Copilot shell have separate ownership.
    An unreadable ledger makes totals unavailable rather than silently dropping
    that attempt from the denominator. No request, response or patient data is
    read or returned.
    """

    root = Path(wrapper_dir).resolve()
    runtime = root / ".runtime"
    paths = [("pipeline", runtime / "provider_hard_stop_ledger.json")]
    paths.extend(
        ("execution_retry", p)
        for p in sorted(islice(runtime.glob("provider_hard_stop_retry_*.json"), MAX_USAGE_LEDGERS + 1))
    )
    paths.extend(
        ("report_revision", p)
        for p in sorted(islice(root.glob("report_revisions/*/runtime/provider_hard_stop.json"), MAX_USAGE_LEDGERS + 1))
    )
    attempts = []
    calls = []
    requests = []
    bindings = []
    statuses: set[str] = set()
    try:
        if len(paths) > MAX_USAGE_LEDGERS:
            raise ValueError("usage ledger count exceeds read budget")
        bytes_read = 0

        def load(path: Path) -> dict:
            nonlocal bytes_read
            bytes_read += path.stat().st_size
            if bytes_read > MAX_USAGE_BYTES:
                raise ValueError("usage exceeds read budget")
            return load_provider_hard_stop_ledger(path)

        primary = paths[0][1]
        reference = runtime / "shared_provider_task.json"
        if not primary.exists() and (reference.exists() or reference.is_symlink()):
            if reference.is_symlink() or runtime.is_symlink():
                raise ValueError("symlinked shared task")
            if reference.stat().st_size > 4096:
                raise ValueError("oversized shared task binding")
            binding = json.loads(reference.read_text(encoding="utf-8"))
            digest = binding.pop("sha256")
            if digest != canonical_sha256(binding) or binding.get("schema_version") != "easyicu.shared-provider-task/1":
                raise ValueError("invalid shared task binding")
            relative = Path(binding["ledger"])
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("shared ledger outside study")
            shared = root.parent / relative
            if shared.is_symlink() or any(p.is_symlink() for p in shared.parents if p != root.parent and root.parent in p.parents):
                raise ValueError("invalid shared recovery ledger")
            ledger = load(shared)
            if ledger.get("declaration_sha256") != binding["declaration_sha256"] or not any(
                row.get("task_id") == binding["task_id"] for row in ledger.get("tasks", [])
            ):
                raise ValueError("shared recovery task mismatch")
            paths[0] = ("pipeline", shared)
        seen: set[Path] = set()
        for stage, path in paths:
            if stage == "pipeline" and not path.exists() and not path.is_symlink():
                if len(paths) > 1:
                    raise ValueError("initial usage ledger missing")
                continue
            # Reject symlinked parents as well as the file itself. A local
            # report directory cannot import another run's private ledger.
            if any(
                p.is_symlink()
                for p in (path, *path.parents)
                if p != root and root in p.parents
            ):
                raise ValueError("symlinked usage ledger")
            if path.resolve() in seen:
                continue
            seen.add(path.resolve())
            ledger = load(path)
            rows = ledger["tasks"]
            if not isinstance(rows, list):
                raise ValueError("invalid usage tasks")
            stage_calls = []
            for row in rows:
                if not isinstance(row, Mapping) or not isinstance(
                    row.get("calls"), list
                ):
                    raise ValueError("invalid usage task")
                status = str(row.get("status") or "unknown")
                statuses.add(
                    status
                    if status
                    in {
                        "pending",
                        "running",
                        "paused",
                        "completed",
                        "failed",
                        "budget_exhausted",
                        "batch_canary_blocked",
                    }
                    else "unknown"
                )
                if any(not isinstance(c, Mapping) for c in row["calls"]):
                    raise ValueError("invalid usage call")
                stage_calls.extend(row["calls"])
                if len(calls) + len(stage_calls) > MAX_USAGE_CALLS:
                    raise ValueError("usage call count exceeds read budget")
            binding = {"stage": stage, "ledger_digest": ledger["sha256"]}
            bindings.append(binding)
            attempts.append({**binding, **_summarize(stage_calls)})
            calls.extend(stage_calls)
            if include_requests:
                for call in stage_calls:
                    requests.append({"stage": stage, **project_request(call, sequence=len(requests) + 1)})
        if not bindings:
            return None
        by_role: dict[str, list] = defaultdict(list)
        for call in calls:
            role = str(call.get("role") or "unclassified")
            if not re.fullmatch(r"[a-z][a-z0-9_]{0,79}", role):
                role = "unclassified"
            by_role[role].append(call)
        return {
            "schema_version": "easyicu.research-run-usage/1",
            "status": next(iter(statuses)) if len(statuses) == 1 else "mixed",
            "accounting_complete": True,
            "scope": "run_with_execution_retries_and_report_revisions",
            "cost_kind": "conservative_ledger_estimate",
            "ledger_sha256": canonical_sha256(bindings),
            **_summarize(calls),
            "attempts": attempts,
            "by_role": [
                {"role": role, **_summarize(rows)}
                for role, rows in sorted(by_role.items())
            ],
            **({"request_records": requests} if include_requests else {}),
        }
    except (
        OSError,
        ValueError,
        TypeError,
        KeyError,
        OverflowError,
        ProviderHardStopLedgerError,
    ):
        return {
            "schema_version": "easyicu.research-run-usage/1",
            "status": "unavailable",
            "accounting_complete": False,
            "reason_code": "provider_usage_ledger_invalid",
            "calls": None,
            "accounted_tokens": None,
            "estimated_cost_usd": None,
        }
