"""Read-only, digest-verified usage across one run and its continuations."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime
import math
from pathlib import Path
import re
from typing import Any, Mapping

from easyicu.research_agent.authority.provider_hard_stop import (
    ProviderHardStopLedgerError,
    load_provider_hard_stop_ledger,
)
from easyicu.research_agent.canonical_json import canonical_sha256


def _number(value: Any, *, integer: bool = False) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("invalid usage number")
    if not math.isfinite(value) or value < 0 or (integer and int(value) != value):
        raise ValueError("invalid usage number")
    return int(value) if integer else float(value)


def _summarize(calls: list[Mapping[str, Any]]) -> dict[str, Any]:
    reported = [call for call in calls if call.get("reported_total_tokens") is not None]
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
        ),
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
        "usage_unknown_calls": len(calls) - len(reported),
        "failed_calls": sum(
            bool(c.get("error_type")) or c.get("state") == "failed" for c in calls
        ),
        "provider_elapsed_seconds": round(sum(durations), 6),
        "timed_calls": len(durations),
    }


def research_run_usage(wrapper_dir: Path) -> dict[str, Any] | None:
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
        for p in sorted(runtime.glob("provider_hard_stop_retry_*.json"))
    )
    paths.extend(
        ("report_revision", p)
        for p in sorted(root.glob("report_revisions/*/runtime/provider_hard_stop.json"))
    )
    attempts = []
    calls = []
    bindings = []
    statuses: set[str] = set()
    try:
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
            ledger = load_provider_hard_stop_ledger(path)
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
            binding = {"stage": stage, "ledger_digest": ledger["sha256"]}
            bindings.append(binding)
            attempts.append({**binding, **_summarize(stage_calls)})
            calls.extend(stage_calls)
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
