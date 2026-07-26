"""Bounded real-Provider protocol probe before an E1 development canary.

The probe sends six small, synthetic, non-clinical requests through the same
factory-minted OpenAI-compatible client used by the benchmark.  It never opens
Canonical9 inputs and never stores prompts, responses, or credentials.  The
durable report contains only response digests, lengths, usage, finish reasons,
latency, and validation outcomes.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import subprocess
import tempfile
import time
from typing import Any, Callable, Mapping, Sequence
from urllib.parse import urlparse

from easyicu.research_agent.authority.provider_hard_stop import (
    PROVIDER_COMPLETION_TOKEN_RESERVATION_FLOOR,
    ProviderHardStopLedger,
    ProviderHardStopLimits,
)
from easyicu.research_agent.providers.hard_stop import HardStopClient
from easyicu.research_agent.providers.protocol import LLMMessage


PROVIDER_PROTOCOL_PROBE_SCHEMA = "easyicu.provider_protocol_probe/2"
PROVIDER_PROTOCOL_REPORT_FILENAME = "provider_protocol_probe.json"
PROVIDER_PROTOCOL_LEDGER_FILENAME = "provider_protocol_ledger.json"
PROVIDER_PROTOCOL_CALL_COUNT = 6
_ALLOWED_BASE_URL = "http://127.0.0.1:8317/v1"
_REASONING_LEAK = re.compile(
    r"<\s*/?\s*think\b|reasoning_content|chain[- ]of[- ]thought",
    flags=re.IGNORECASE,
)


class ProviderProtocolProbeError(RuntimeError):
    """A bounded Provider readiness probe did not satisfy its contract."""


@dataclass(frozen=True)
class ProbeSpec:
    name: str
    role: str
    effort: str
    messages: tuple[LLMMessage, ...]
    max_tokens: int
    validator: Callable[[str], Mapping[str, object]]


def _strip_fence(text: str) -> str:
    match = re.search(r"```[^\n`]*\n(.*?)\n```", text, flags=re.DOTALL)
    return match.group(1).strip() if match else text.strip()


def _strict_json(text: str) -> dict[str, Any]:
    def reject_pairs(pairs: Sequence[tuple[str, object]]) -> dict[str, object]:
        payload: dict[str, object] = {}
        for key, value in pairs:
            if key in payload:
                raise ValueError(f"duplicate JSON key {key!r}")
            payload[key] = value
        return payload

    payload = json.loads(
        _strip_fence(text),
        object_pairs_hook=reject_pairs,
        parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError(f"non-finite JSON constant {value!r}")
        ),
    )
    if not isinstance(payload, dict):
        raise ValueError("response is not a JSON object")
    return payload


def _validate_ready(text: str) -> Mapping[str, object]:
    if text.strip() != "READY":
        raise ValueError("transport probe did not return exact READY")
    return {"contract": "exact_text"}


def _validate_plan(text: str) -> Mapping[str, object]:
    payload = _strict_json(text)
    steps = payload.get("steps")
    if not isinstance(steps, list) or not steps:
        raise ValueError("planner response has no non-empty steps list")
    for step in steps:
        if (
            not isinstance(step, dict)
            or not isinstance(step.get("id"), str)
            or not isinstance(step.get("method"), str)
        ):
            raise ValueError("planner step is missing string id/method")
    return {"contract": "strict_json", "step_count": len(steps)}


def _validate_python(text: str) -> Mapping[str, object]:
    source = _strip_fence(text)
    tree = ast.parse(source)
    functions = [
        node.name for node in tree.body if isinstance(node, ast.FunctionDef)
    ]
    if not functions:
        raise ValueError("code response defines no function")
    return {
        "contract": "python_ast",
        "function_count": len(functions),
        "source_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
    }


def _validate_writer(text: str) -> Mapping[str, object]:
    payload = _strict_json(text)
    if not isinstance(payload.get("claim"), str):
        raise ValueError("writer response has no claim")
    if payload.get("evidence_ids") != ["toy-evidence-1"]:
        raise ValueError("writer response changed the evidence id")
    return {"contract": "evidence_json"}


def _validate_nonempty(text: str) -> Mapping[str, object]:
    if not text.strip():
        raise ValueError("bounded response is empty")
    return {"contract": "bounded_nonempty"}


def _probe_specs() -> tuple[ProbeSpec, ...]:
    return (
        ProbeSpec(
            name="transport_exact_text",
            role="analyzer",
            effort="low",
            messages=(
                LLMMessage(
                    role="system",
                    content="Return only the exact requested text.",
                ),
                LLMMessage(role="user", content="Return exactly READY"),
            ),
            max_tokens=32,
            validator=_validate_ready,
        ),
        ProbeSpec(
            name="planner_strict_json",
            role="planner",
            effort="medium",
            messages=(
                LLMMessage(
                    role="system",
                    content=(
                        "Return JSON only. This is a synthetic software protocol "
                        "probe with no clinical or patient data."
                    ),
                ),
                LLMMessage(
                    role="user",
                    content=(
                        'Create one toy analysis step. Required shape: '
                        '{"steps":[{"id":"step_1","method":"mean"}]}.'
                    ),
                ),
            ),
            max_tokens=128,
            validator=_validate_plan,
        ),
        ProbeSpec(
            name="coder_python",
            role="coder",
            effort="medium",
            messages=(
                LLMMessage(
                    role="system",
                    content="Return only valid Python source, without explanation.",
                ),
                LLMMessage(
                    role="user",
                    content=(
                        "Define summarize(values), ignoring None values and returning "
                        "a dict with n and mean. Do not read files or use the network."
                    ),
                ),
            ),
            max_tokens=192,
            validator=_validate_python,
        ),
        ProbeSpec(
            name="repair_python",
            role="repair",
            effort="medium",
            messages=(
                LLMMessage(
                    role="system",
                    content="Return only corrected Python source.",
                ),
                LLMMessage(
                    role="user",
                    content="Repair this syntax error:\ndef add(a, b)\n    return a + b",
                ),
            ),
            max_tokens=96,
            validator=_validate_python,
        ),
        ProbeSpec(
            name="writer_evidence_json",
            role="writer",
            effort="low",
            messages=(
                LLMMessage(role="system", content="Return JSON only."),
                LLMMessage(
                    role="user",
                    content=(
                        "Return a short toy claim and bind it to the exact id "
                        "toy-evidence-1. Required keys: claim, evidence_ids."
                    ),
                ),
            ),
            max_tokens=96,
            validator=_validate_writer,
        ),
        ProbeSpec(
            name="bounded_finish_reason",
            role="writer",
            effort="low",
            messages=(
                LLMMessage(
                    role="system",
                    content="Return only repetitions of the requested token.",
                ),
                LLMMessage(
                    role="user",
                    content="Repeat the token ALPHA exactly 300 times.",
                ),
            ),
            max_tokens=128,
            validator=_validate_nonempty,
        ),
    )


def _validate_base_url(base_url: str) -> str:
    normalized = str(base_url or "").rstrip("/")
    parsed = urlparse(normalized)
    if (
        normalized != _ALLOWED_BASE_URL
        or parsed.scheme != "http"
        or parsed.hostname != "127.0.0.1"
        or parsed.port != 8317
        or parsed.path != "/v1"
        or parsed.params
        or parsed.query
        or parsed.fragment
    ):
        raise ProviderProtocolProbeError(
            f"protocol probe is restricted to {_ALLOWED_BASE_URL}"
        )
    return normalized


def _git_identity(repo_root: Path) -> tuple[str, bool]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
        timeout=30.0,
    )
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
        timeout=30.0,
    )
    if commit.returncode != 0 or status.returncode != 0:
        raise ProviderProtocolProbeError("could not bind protocol probe to Git")
    return commit.stdout.strip(), not bool(status.stdout)


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    parent = path.parent.lstat()
    if not stat.S_ISDIR(parent.st_mode) or stat.S_ISLNK(parent.st_mode):
        raise ProviderProtocolProbeError("probe output parent must be a real directory")
    if path.exists() and (path.is_symlink() or not path.is_file()):
        raise ProviderProtocolProbeError("unsafe protocol report destination")
    raw = (
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
        ).encode("utf-8")
        + b"\n"
    )
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
        os.chmod(path, 0o600)
    finally:
        Path(temporary_name).unlink(missing_ok=True)


def _limits() -> ProviderHardStopLimits:
    return ProviderHardStopLimits(
        max_provider_attempts_per_run=PROVIDER_PROTOCOL_CALL_COUNT,
        max_provider_attempts_per_batch=PROVIDER_PROTOCOL_CALL_COUNT,
        max_total_tokens_per_run=500_000,
        max_total_tokens_per_batch=500_000,
        max_estimated_cost_usd_per_batch=1.0,
        max_wall_clock_seconds_per_task=900.0,
        input_cost_usd_per_million_tokens=0.0,
        output_cost_usd_per_million_tokens=0.0,
    )


def run_provider_protocol_probe(
    *,
    output_dir: Path,
    model: str,
    base_url: str,
    client_for_effort: Callable[[str], Any],
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """Execute exactly six bounded calls and persist no response content."""

    normalized_url = _validate_base_url(base_url)
    normalized_model = str(model or "").strip()
    if normalized_model != "gpt-5.6-luna":
        raise ProviderProtocolProbeError(
            "this frozen probe requires model gpt-5.6-luna"
        )
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(destination, 0o700)
    root = (
        Path(repo_root).resolve()
        if repo_root is not None
        else Path(__file__).resolve().parents[2]
    )
    commit, clean = _git_identity(root)
    ledger = ProviderHardStopLedger(
        path=destination / PROVIDER_PROTOCOL_LEDGER_FILENAME,
        task_ids=("provider_protocol_probe",),
        limits=_limits(),
        batch_id=f"provider-protocol-{commit[:12]}",
    )
    task = ledger.start_task("provider_protocol_probe")
    results: list[dict[str, object]] = []
    started_all = time.monotonic()
    try:
        for spec in _probe_specs():
            inner = client_for_effort(spec.effort)
            client = HardStopClient(inner, role=spec.role, task=task)
            started = time.monotonic()
            text, usage = client.complete_with_usage(
                spec.messages,
                max_tokens=spec.max_tokens,
                temperature=0.0,
            )
            latency = time.monotonic() - started
            if not isinstance(text, str) or not text.strip():
                raise ProviderProtocolProbeError(
                    f"{spec.name} returned no usable text"
                )
            if _REASONING_LEAK.search(text):
                raise ProviderProtocolProbeError(
                    f"{spec.name} exposed a reasoning marker"
                )
            if not isinstance(usage, dict) or int(usage.get("total_tokens") or 0) <= 0:
                raise ProviderProtocolProbeError(
                    f"{spec.name} returned no authoritative usage"
                )
            completion_tokens = int(usage.get("completion_tokens") or 0)
            finish_reason = getattr(client, "last_finish_reason", None)
            if not isinstance(finish_reason, str) or not finish_reason:
                raise ProviderProtocolProbeError(
                    f"{spec.name} returned no finish_reason"
                )
            validation = dict(spec.validator(text))
            results.append(
                {
                    "name": spec.name,
                    "role": spec.role,
                    "reasoning_effort": spec.effort,
                    "status": "passed",
                    "max_tokens": spec.max_tokens,
                    "response_chars": len(text),
                    "response_sha256": hashlib.sha256(
                        text.encode("utf-8")
                    ).hexdigest(),
                    "finish_reason": finish_reason,
                    "usage": {
                        key: int(usage.get(key) or 0)
                        for key in (
                            "prompt_tokens",
                            "completion_tokens",
                            "total_tokens",
                        )
                    },
                    "latency_seconds": latency,
                    "reasoning_marker_exposed": False,
                    "completion_cap_observed": (
                        completion_tokens <= spec.max_tokens
                    ),
                    "validation": validation,
                }
            )
        task.finish(score={"protocol_status": "passed"})
    except BaseException as exc:
        task.finish(error=type(exc).__name__)
        raise

    snapshot = ledger.snapshot()
    attempts = int(
        (snapshot.get("totals") or {}).get("provider_attempts") or 0  # type: ignore[union-attr]
    )
    if attempts != PROVIDER_PROTOCOL_CALL_COUNT:
        raise ProviderProtocolProbeError(
            f"expected {PROVIDER_PROTOCOL_CALL_COUNT} transport attempts, got {attempts}"
        )
    elapsed = time.monotonic() - started_all
    if not math.isfinite(elapsed) or elapsed <= 0:
        raise ProviderProtocolProbeError("protocol probe duration is invalid")
    raw_tasks = snapshot.get("tasks")
    if (
        not isinstance(raw_tasks, list)
        or len(raw_tasks) != 1
        or not isinstance(raw_tasks[0], Mapping)
    ):
        raise ProviderProtocolProbeError("protocol ledger task shape is invalid")
    raw_calls = raw_tasks[0].get("calls")
    if (
        not isinstance(raw_calls, list)
        or len(raw_calls) != PROVIDER_PROTOCOL_CALL_COUNT
        or any(
            not isinstance(call, Mapping)
            or call.get("completion_token_reservation")
            != PROVIDER_COMPLETION_TOKEN_RESERVATION_FLOOR
            for call in raw_calls
        )
    ):
        raise ProviderProtocolProbeError(
            "protocol calls did not reserve the full unbounded completion envelope"
        )
    truncation_probe = next(
        result
        for result in results
        if result["name"] == "bounded_finish_reason"
    )
    provider_completion_cap_enforced = bool(
        truncation_probe["completion_cap_observed"]
    )
    report: dict[str, Any] = {
        "schema_version": PROVIDER_PROTOCOL_PROBE_SCHEMA,
        "status": "passed",
        "development_only": True,
        "paper_authorized": False,
        "contains_clinical_data": False,
        "contains_patient_data": False,
        "stores_prompt_or_response_text": False,
        "provider": "openai-compatible-loopback",
        "base_url": normalized_url,
        "model": normalized_model,
        "git_commit": commit,
        "git_tree_clean": clean,
        "call_count": len(results),
        "transport_attempts": attempts,
        "transport_attempt_cap": PROVIDER_PROTOCOL_CALL_COUNT,
        "transport_retries": attempts - len(results),
        "provider_completion_cap_enforced": provider_completion_cap_enforced,
        "provider_completion_cap_mode": (
            "observed_enforced"
            if provider_completion_cap_enforced
            else "not_enforced_conservative_reservation_required"
        ),
        "hard_stop_completion_token_reservation_per_attempt": (
            PROVIDER_COMPLETION_TOKEN_RESERVATION_FLOOR
        ),
        "total_latency_seconds": elapsed,
        "usage": dict(snapshot.get("totals") or {}),
        "monetary_cost_authoritative": False,
        "monetary_cost_note": (
            "No locked provider price was supplied; token usage is authoritative "
            "but the zero-price ledger estimate is not a monetary claim."
        ),
        "truncation_finish_reason": truncation_probe["finish_reason"],
        "probes": results,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    _atomic_json(destination / PROVIDER_PROTOCOL_REPORT_FILENAME, report)
    return report


__all__ = [
    "PROVIDER_PROTOCOL_CALL_COUNT",
    "PROVIDER_PROTOCOL_LEDGER_FILENAME",
    "PROVIDER_PROTOCOL_PROBE_SCHEMA",
    "PROVIDER_PROTOCOL_REPORT_FILENAME",
    "ProviderProtocolProbeError",
    "run_provider_protocol_probe",
]
