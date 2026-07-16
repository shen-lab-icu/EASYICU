#!/usr/bin/env python3
"""Aggregate a research-agent run's EXISTING artifacts into a performance baseline.

Read-only. Does NOT re-run any experiment and does NOT import easyicu.
Purpose (P0-5a): establish the OLD baseline that Track-A improvements must beat.

Fail-CLOSED by design: a performance baseline that silently drops a corrupt
receipt or cost file would understate cost. Any unreadable / digest-invalid /
schema-invalid input raises BaselineError (non-zero exit).

Call accounting (reconciled):
  all_provider_calls   = step_scoped_calls + run_level_planner_calls
  step_scoped_calls    = sum of per-step provider-call-budget receipt entries
  run_level_planner_calls = planner-role calls in cost_records (outside step budgets)
  repair_calls         = repair/rewrite categories in receipts
  blocked_provider_requests = budget-denied requests (NOT real provider calls)

Wall time is segmented; sandbox_execution_seconds (pure in-sandbox compute) is
tiny and must not be reported as "the run time" -- the LLM round-trips dominate.

Usage:
    python tools/agent_perf_baseline.py <run_dir> [--json out.json]
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import re
import sys
from collections import defaultdict
from datetime import datetime

REPAIR_KEYS = ("repair", "rewrite")
AUDIT_KEYS = ("audit",)
INIT_KEYS = ("initial",)
RECEIPT_SCHEMA_VERSIONS = {1, 2, 3, 4, 5, 6}
REPAIR_AUTHORITY_BINDING_SCHEMA_V2 = "easyicu.repair_authority_binding/2"
REPAIR_TRANSPORT_PROVIDER_SUFFIXES = ("patch", "full_rewrite")
RUN_SESSION_START = "Research context built."
RUN_SESSION_END = "Research-agent run complete."
STEP_SESSION_START = re.compile(r"^Step \d+/\d+ started:")


class BaselineError(RuntimeError):
    """Fail-closed error: a required input could not be trusted."""


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _receipt_digest(payload: dict) -> str:
    # Must match provider_budget.py::_receipt_digest exactly.
    canonical = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _is_sha256_hex(value: object) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value)
    )


def _bound_repair_provider_category(binding: object, *, path: str) -> str | None:
    if not isinstance(binding, dict):
        return None
    schema_version = binding.get("schema_version")
    has_provider_category = "provider_category" in binding
    if schema_version != REPAIR_AUTHORITY_BINDING_SCHEMA_V2:
        if has_provider_category:
            raise BaselineError(
                f"legacy repair binding declares provider category: {path}"
            )
        return None
    provider_category = binding.get("provider_category")
    if (
        not isinstance(provider_category, str)
        or not provider_category.strip()
        or provider_category != provider_category.strip()
    ):
        raise BaselineError(f"repair binding provider category invalid: {path}")
    return provider_category


def _repair_owned_provider_calls(
    categories: list[str],
    *,
    reserved_history_len: int,
    history_len: int,
    provider_category: str | None,
) -> int:
    suffix = categories[reserved_history_len:history_len]
    if provider_category is None:
        return len(suffix)
    owned_categories = {
        f"{provider_category}_{transport_suffix}"
        for transport_suffix in REPAIR_TRANSPORT_PROVIDER_SUFFIXES
    }
    return sum(category in owned_categories for category in suffix)


def _verify_initial_generation(
    raw_entry: object,
    *,
    categories: list[str],
    path: str,
) -> None:
    """Mirror the schema-v6 initial candidate transport checks fail-closed."""

    if raw_entry is None:
        return
    required_entry_keys = {
        "binding",
        "binding_sha256",
        "provider_history_len",
        "provider_history_sha256",
        "provider_transport_id",
        "transport",
    }
    if not isinstance(raw_entry, dict) or set(raw_entry) != required_entry_keys:
        raise BaselineError(f"receipt initial-generation entry invalid: {path}")
    binding = raw_entry.get("binding")
    binding_sha256 = raw_entry.get("binding_sha256")
    reserved_len = raw_entry.get("provider_history_len")
    transport_id = raw_entry.get("provider_transport_id")
    if (
        not isinstance(binding, dict)
        or not _is_sha256_hex(binding_sha256)
        or binding_sha256 != _receipt_digest(binding)
        or isinstance(reserved_len, bool)
        or not isinstance(reserved_len, int)
        or not 0 <= reserved_len <= len(categories)
        or raw_entry.get("provider_history_sha256")
        != _receipt_digest({"categories": categories[:reserved_len]})
        or not isinstance(transport_id, str)
        or not transport_id.startswith("initial_generation:")
        or not transport_id.removeprefix("initial_generation:").isdigit()
    ):
        raise BaselineError(
            f"receipt initial-generation authority inconsistent: {path}"
        )
    transport = raw_entry.get("transport")
    if not isinstance(transport, dict):
        raise BaselineError(f"receipt initial-generation transport invalid: {path}")
    state = transport.get("state")
    allowed_keys = {"state"}
    if state in {"completed", "failed"}:
        allowed_keys.update(
            {
                "provider_history_len",
                "provider_history_sha256",
                "provider_calls",
            }
        )
    if state == "completed":
        allowed_keys.update({"after_code_sha256", "after_code_size_bytes"})
    elif state == "failed":
        allowed_keys.add("error_type")
    elif state != "pending":
        raise BaselineError(
            f"receipt initial-generation transport state invalid: {path}"
        )
    if set(transport) != allowed_keys:
        raise BaselineError(
            f"receipt initial-generation transport fields inconsistent: {path}"
        )
    if state == "pending":
        return
    history_len = transport.get("provider_history_len")
    provider_calls = transport.get("provider_calls")
    if (
        isinstance(history_len, bool)
        or not isinstance(history_len, int)
        or not reserved_len <= history_len <= len(categories)
        or transport.get("provider_history_sha256")
        != _receipt_digest({"categories": categories[:history_len]})
        or isinstance(provider_calls, bool)
        or not isinstance(provider_calls, int)
        or provider_calls
        != sum(
            category == "initial_generation"
            for category in categories[reserved_len:history_len]
        )
    ):
        raise BaselineError(f"receipt initial-generation history inconsistent: {path}")
    if state == "completed" and (
        provider_calls < 1
        or not _is_sha256_hex(transport.get("after_code_sha256"))
        or isinstance(transport.get("after_code_size_bytes"), bool)
        or not isinstance(transport.get("after_code_size_bytes"), int)
        or transport["after_code_size_bytes"] < 0
    ):
        raise BaselineError(f"receipt completed initial generation invalid: {path}")
    if state == "failed" and (
        not isinstance(transport.get("error_type"), str)
        or not transport["error_type"].strip()
        or transport["error_type"] != transport["error_type"].strip()
    ):
        raise BaselineError(f"receipt failed initial generation invalid: {path}")


def _categorize(cat: str) -> str:
    c = cat.lower()
    if any(k in c for k in AUDIT_KEYS):
        return "concept_audit"
    if any(k in c for k in REPAIR_KEYS):
        return "repair"
    if any(k in c for k in INIT_KEYS):
        return "initial_generation"
    return cat


def _load_json(path: str) -> object:
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except Exception as exc:  # fail-closed
        raise BaselineError(f"unreadable JSON input: {path}: {exc}") from exc


def read_receipts(run_dir: str, inputs: dict) -> list[dict]:
    """Authoritative append-only per-step ledger, with digest verification."""
    out = []
    for p in sorted(
        glob.glob(os.path.join(run_dir, ".runtime", "provider_call_budgets", "*.json"))
    ):
        d = _load_json(p)
        if not isinstance(d, dict):
            raise BaselineError(f"receipt is not an object: {p}")
        stored = d.get("sha256")
        payload = {k: v for k, v in d.items() if k != "sha256"}
        if not isinstance(stored, str) or stored != _receipt_digest(payload):
            raise BaselineError(f"receipt digest invalid (tampered/corrupt): {p}")
        if d.get("schema_version") not in RECEIPT_SCHEMA_VERSIONS:
            raise BaselineError(f"receipt schema_version unsupported: {p}")
        step = d.get("step_id")
        if not step:
            raise BaselineError(f"receipt missing step_id: {p}")
        expected = hashlib.sha256(str(step).encode("utf-8")).hexdigest()[:16]
        if os.path.basename(p) != f"{expected}.json":
            raise BaselineError(f"receipt filename does not match step_id: {p}")
        inputs[os.path.relpath(p, run_dir)] = _sha256_file(p)
        limit = d.get("limit")
        cats = d.get("categories")
        if (
            isinstance(limit, bool)
            or not isinstance(limit, int)
            or limit < 0
            or not isinstance(cats, list)
            or len(cats) > limit
            or any(not isinstance(cat, str) or not cat.strip() for cat in cats)
        ):
            raise BaselineError(f"receipt has invalid limit or categories: {p}")
        breakdown = defaultdict(int)
        for c in cats:
            breakdown[_categorize(c)] += 1
        logical_repairs = d.get("logical_repairs", [])
        if d.get("schema_version") in {3, 4, 5, 6}:
            if not isinstance(logical_repairs, list):
                raise BaselineError(f"receipt logical repair ledger invalid: {p}")
            for index, entry in enumerate(logical_repairs, start=1):
                history_len = (
                    entry.get("provider_history_len")
                    if isinstance(entry, dict)
                    else None
                )
                history_sha256 = (
                    entry.get("provider_history_sha256")
                    if isinstance(entry, dict)
                    else None
                )
                expected_history_sha256 = (
                    _receipt_digest({"categories": cats[:history_len]})
                    if isinstance(history_len, int)
                    and not isinstance(history_len, bool)
                    else None
                )
                binding = entry.get("binding") if isinstance(entry, dict) else None
                binding_sha256 = (
                    entry.get("binding_sha256") if isinstance(entry, dict) else None
                )
                if binding is None and binding_sha256 is None:
                    binding_valid = True
                elif isinstance(binding, dict) and isinstance(binding_sha256, str):
                    binding_valid = binding_sha256 == _receipt_digest(binding)
                else:
                    binding_valid = False
                if (
                    not isinstance(entry, dict)
                    or entry.get("attempt_id") != index
                    or not isinstance(entry.get("repair_class"), str)
                    or not entry["repair_class"].strip()
                    or isinstance(history_len, bool)
                    or not isinstance(history_len, int)
                    or not 0 <= history_len <= len(cats)
                    or history_sha256 != expected_history_sha256
                    or not binding_valid
                ):
                    raise BaselineError(
                        f"receipt logical repair history inconsistent: {p}"
                    )
                provider_category = _bound_repair_provider_category(binding, path=p)
                if d.get("schema_version") in {5, 6}:
                    transport = entry.get("transport")
                    if not isinstance(transport, dict):
                        raise BaselineError(
                            f"receipt logical repair transport invalid: {p}"
                        )
                    state = transport.get("state")
                    if state in {"pending", "legacy_untracked"}:
                        transport_valid = set(transport) == {"state"}
                    elif state in {"completed", "failed"}:
                        terminal_history_len = transport.get("provider_history_len")
                        terminal_calls = transport.get("provider_calls")
                        expected_terminal_calls = (
                            _repair_owned_provider_calls(
                                cats,
                                reserved_history_len=history_len,
                                history_len=terminal_history_len,
                                provider_category=provider_category,
                            )
                            if isinstance(terminal_history_len, int)
                            and not isinstance(terminal_history_len, bool)
                            else None
                        )
                        terminal_history_valid = bool(
                            isinstance(terminal_history_len, int)
                            and not isinstance(terminal_history_len, bool)
                            and history_len <= terminal_history_len <= len(cats)
                            and transport.get("provider_history_sha256")
                            == _receipt_digest(
                                {"categories": cats[:terminal_history_len]}
                            )
                            and isinstance(terminal_calls, int)
                            and not isinstance(terminal_calls, bool)
                            and terminal_calls == expected_terminal_calls
                            and (state == "failed" or terminal_calls > 0)
                        )
                        if state == "completed":
                            completed_keys = {
                                "state",
                                "mode",
                                "after_code_sha256",
                                "provider_history_len",
                                "provider_history_sha256",
                                "provider_calls",
                            }
                            result_persistence = transport.get("result_persistence")
                            if d.get("schema_version") == 6:
                                completed_keys.add("result_persistence")
                            if result_persistence == "content_addressed":
                                completed_keys.add("after_code_size_bytes")
                            transport_valid = bool(
                                terminal_history_valid
                                and set(transport) == completed_keys
                                and isinstance(transport.get("mode"), str)
                                and transport["mode"].strip() == transport["mode"]
                                and transport["mode"]
                                and _is_sha256_hex(transport.get("after_code_sha256"))
                                and (
                                    d.get("schema_version") != 6
                                    or result_persistence
                                    in {
                                        "content_addressed",
                                        "legacy_untracked",
                                        "untracked",
                                    }
                                )
                                and (
                                    result_persistence != "content_addressed"
                                    or (
                                        isinstance(
                                            transport.get("after_code_size_bytes"),
                                            int,
                                        )
                                        and not isinstance(
                                            transport.get("after_code_size_bytes"),
                                            bool,
                                        )
                                        and transport["after_code_size_bytes"] >= 0
                                    )
                                )
                            )
                        else:
                            transport_valid = bool(
                                terminal_history_valid
                                and set(transport)
                                == {
                                    "state",
                                    "error_type",
                                    "provider_history_len",
                                    "provider_history_sha256",
                                    "provider_calls",
                                }
                                and isinstance(transport.get("error_type"), str)
                                and transport["error_type"].strip()
                                == transport["error_type"]
                                and transport["error_type"]
                            )
                    else:
                        transport_valid = False
                    if not transport_valid:
                        raise BaselineError(
                            f"receipt logical repair transport inconsistent: {p}"
                        )
                    if state == "pending" and index != len(logical_repairs):
                        raise BaselineError(
                            f"receipt pending logical repair is not final: {p}"
                        )
        elif logical_repairs:
            raise BaselineError(
                f"legacy receipt unexpectedly declares logical repairs: {p}"
            )
        if d.get("schema_version") == 6:
            _verify_initial_generation(
                d.get("initial_generation"), categories=cats, path=p
            )
        elif d.get("initial_generation") is not None:
            raise BaselineError(f"legacy receipt declares initial generation: {p}")
        if d.get("schema_version") in {4, 5, 6}:
            reservation = d.get("final_reservation_state")
            if not isinstance(reservation, dict):
                raise BaselineError(f"receipt final reservation state invalid: {p}")
            required_token = reservation.get("required_token")
            bound_len = reservation.get("bound_provider_history_len")
            bound_sha256 = reservation.get("bound_provider_history_sha256")
            completed_token = reservation.get("completed_token")
            released = reservation.get("released")
            if required_token is None:
                valid_reservation = (
                    bound_len is None
                    and bound_sha256 is None
                    and completed_token is None
                    and released is False
                )
            else:
                valid_reservation = bool(
                    isinstance(required_token, str)
                    and required_token.strip()
                    and isinstance(bound_len, int)
                    and not isinstance(bound_len, bool)
                    and 0 <= bound_len <= len(cats)
                    and bound_sha256
                    == _receipt_digest({"categories": cats[:bound_len]})
                    and completed_token in {None, required_token}
                    and isinstance(released, bool)
                    and (not released or completed_token is not None)
                )
            if not valid_reservation:
                raise BaselineError(
                    f"receipt final reservation state inconsistent: {p}"
                )
        elif d.get("final_reservation_state") is not None:
            raise BaselineError(
                f"legacy receipt unexpectedly declares final reservation state: {p}"
            )
        out.append(
            {
                "step_id": step,
                "total_calls": len(cats),
                "limit": limit,
                "sequence": cats,
                "breakdown": dict(breakdown),
                "repair_calls": breakdown.get("repair", 0),
                "logical_repair_attempts": len(logical_repairs),
                "logical_repair_classes": [
                    entry["repair_class"] for entry in logical_repairs
                ],
            }
        )
    if not out:
        raise BaselineError(f"no provider-call receipts found under: {run_dir}")
    return out


def aggregate_cost_records(run_dir: str, inputs: dict) -> dict:
    """Deduped union of every cost_records version (top-level + evidence/*)."""
    files = [os.path.join(run_dir, "cost_records.json")]
    files += sorted(glob.glob(os.path.join(run_dir, "evidence", "*cost_records*.json")))
    seen: set[str] = set()
    by_role: dict = defaultdict(
        lambda: {"n": 0, "prompt": 0, "completion": 0, "total": 0}
    )
    for f in files:
        if not os.path.exists(f):
            continue
        inputs[os.path.relpath(f, run_dir)] = _sha256_file(f)
        d = _load_json(f)
        if isinstance(d, list):
            recs = d
        elif isinstance(d, dict):
            recs = d.get("records", d.get("calls", []))
        else:
            raise BaselineError(f"cost_records has invalid top-level shape: {f}")
        if not isinstance(recs, list):
            raise BaselineError(f"cost_records has no record list: {f}")
        for r in recs:
            if not isinstance(r, dict):
                raise BaselineError(f"cost record is not an object: {f}")
            role = r.get("role")
            prompt = r.get("prompt_tokens")
            completion = r.get("completion_tokens")
            total = r.get(
                "total_tokens",
                (
                    prompt + completion
                    if isinstance(prompt, int) and isinstance(completion, int)
                    else None
                ),
            )
            if (
                not isinstance(role, str)
                or not role.strip()
                or isinstance(prompt, bool)
                or not isinstance(prompt, int)
                or prompt < 0
                or isinstance(completion, bool)
                or not isinstance(completion, int)
                or completion < 0
                or isinstance(total, bool)
                or not isinstance(total, int)
                or total < 0
            ):
                raise BaselineError(f"cost record has invalid role/token fields: {f}")
            key = _receipt_digest(r)
            if key in seen:
                continue
            seen.add(key)
            by_role[role]["n"] += 1
            by_role[role]["prompt"] += prompt
            by_role[role]["completion"] += completion
            by_role[role]["total"] += total
    if not seen:
        raise BaselineError(f"no cost records found under: {run_dir}")
    tot = {"n": 0, "prompt": 0, "completion": 0, "total": 0}
    for v in by_role.values():
        for k in tot:
            tot[k] += v[k]
    return {"by_role": {k: dict(v) for k, v in by_role.items()}, "total": tot}


def _parse_ts(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def _audit_sessions(records: list[dict], path: str) -> list[list[dict]]:
    """Return explicit run sessions from host-owned audit start/end events."""

    sessions: list[list[dict]] = []
    current: list[dict] | None = None
    previous_ts: datetime | None = None
    for record in records:
        event = record.get("event")
        raw_timestamp = record.get("timestamp")
        if not isinstance(event, str) or not isinstance(raw_timestamp, str):
            raise BaselineError(f"audit_log record lacks event/timestamp: {path}")
        try:
            timestamp = _parse_ts(raw_timestamp)
        except Exception as exc:
            raise BaselineError(f"audit_log timestamp is invalid: {path}") from exc
        if previous_ts is not None and timestamp < previous_ts:
            raise BaselineError(f"audit_log timestamps are not monotonic: {path}")
        previous_ts = timestamp
        record = dict(record)
        record["_parsed_timestamp"] = timestamp
        if event == RUN_SESSION_START:
            if current is not None:
                raise BaselineError(f"audit_log has nested run sessions: {path}")
            current = [record]
            continue
        if current is None:
            raise BaselineError(f"audit_log event falls outside a run session: {path}")
        current.append(record)
        if event == RUN_SESSION_END:
            sessions.append(current)
            current = None
    if current is not None:
        raise BaselineError(f"audit_log ends with an incomplete run session: {path}")
    if not sessions:
        raise BaselineError(f"audit_log contains no complete run session: {path}")
    return sessions


def read_wall_times(run_dir: str, inputs: dict) -> dict:
    # sandbox compute (exact, from step run.logs)
    sandbox = {}
    for log in sorted(glob.glob(os.path.join(run_dir, "steps", "*", "run.log"))):
        step = os.path.basename(os.path.dirname(log))
        secs = 0.0
        for m in re.finditer(
            r"duration_seconds:\s*([0-9.]+)",
            open(log, encoding="utf-8", errors="ignore").read(),
        ):
            secs += float(m.group(1))
        sandbox[step] = round(secs, 3)
    # active wall (from audit_log timestamps, idle-gap excluded)
    alog = os.path.join(run_dir, "audit_log.jsonl")
    if not os.path.exists(alog):
        raise BaselineError(f"audit_log.jsonl is missing: {run_dir}")
    inputs["audit_log.jsonl"] = _sha256_file(alog)
    records = []
    with open(alog, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(_load_json_line(line, alog))
    sessions = _audit_sessions(records, alog)
    session_active = 0.0
    step_active: dict[str, float] = defaultdict(float)
    for session in sessions:
        session_active += (
            session[-1]["_parsed_timestamp"] - session[0]["_parsed_timestamp"]
        ).total_seconds()
        by_step: dict[str, list[dict]] = defaultdict(list)
        for record in session:
            step_id = record.get("step_id")
            if isinstance(step_id, str) and step_id:
                by_step[step_id].append(record)
        for step_id, step_records in by_step.items():
            starts = [
                record
                for record in step_records
                if STEP_SESSION_START.match(str(record.get("event") or ""))
            ]
            if not starts:
                continue
            if len(starts) != 1:
                raise BaselineError(
                    f"step {step_id!r} has multiple starts in one audit session: {alog}"
                )
            start = starts[0]["_parsed_timestamp"]
            end = step_records[-1]["_parsed_timestamp"]
            if end < start:
                raise BaselineError(f"step {step_id!r} ends before it starts: {alog}")
            step_active[step_id] += (end - start).total_seconds()
    return {
        "sandbox_execution_seconds": sandbox,
        "sandbox_execution_seconds_total": round(sum(sandbox.values()), 3),
        "session_active_wall_seconds": round(session_active, 1),
        "run_session_count": len(sessions),
        "step_active_wall_seconds": {
            step_id: round(seconds, 1)
            for step_id, seconds in sorted(step_active.items())
        },
        "wall_method": "explicit_audit_session_boundaries",
    }


def _load_json_line(line: str, path: str) -> dict:
    try:
        return json.loads(line)
    except Exception as exc:
        raise BaselineError(f"corrupt audit_log line in {path}: {exc}") from exc


def count_resumes(run_dir: str, inputs: dict) -> int:
    paths = sorted(
        glob.glob(os.path.join(run_dir, "resume_environment_receipt_*.json"))
    )
    for expected_sequence, path in enumerate(paths, start=1):
        payload = _load_json(path)
        if (
            not isinstance(payload, dict)
            or payload.get("schema_version") != "easyicu.resume_environment_receipt/1"
            or payload.get("attempt_sequence") != expected_sequence
        ):
            raise BaselineError(f"resume receipt is invalid or non-monotonic: {path}")
        inputs[os.path.relpath(path, run_dir)] = _sha256_file(path)
    return len(paths)


def count_blocked_requests(run_dir: str, inputs: dict) -> int:
    rs = os.path.join(run_dir, "run_status.json")
    if not os.path.exists(rs):
        return 0
    inputs["run_status.json"] = _sha256_file(rs)
    blob = json.dumps(_load_json(rs))
    return len(re.findall(r"provider-call budget unavailable", blob))


def build_baseline(run_dir: str) -> dict:
    inputs: dict = {}
    receipts = read_receipts(run_dir, inputs)
    tokens = aggregate_cost_records(run_dir, inputs)
    walls = read_wall_times(run_dir, inputs)
    step_scoped = sum(r["total_calls"] for r in receipts)
    repair = sum(r["repair_calls"] for r in receipts)
    planner = tokens["by_role"].get("planner", {}).get("n", 0)
    blocked = count_blocked_requests(run_dir, inputs)
    all_provider_calls = step_scoped + planner
    cost_record_calls = tokens["total"]["n"]
    if all_provider_calls != cost_record_calls:
        raise BaselineError(
            "provider-call reconciliation failed: "
            f"step_scoped({step_scoped}) + planner({planner}) != "
            f"cost_records({cost_record_calls})"
        )
    return {
        "run_dir": run_dir,
        "calls": {
            "all_provider_calls": all_provider_calls,
            "step_scoped_calls": step_scoped,
            "run_level_planner_calls": planner,
            "cost_record_calls": cost_record_calls,
            "reconciliation_delta": 0,
            "repair_calls": repair,
            "blocked_provider_requests": blocked,
            "per_step": receipts,
        },
        "tokens_from_deduped_cost_records": tokens,
        "wall_time": walls,
        "resume_count": count_resumes(run_dir, inputs),
        "inputs_sha256": inputs,
        "notes": [
            "Calls from append-only, digest-VERIFIED provider-call receipts + planner from cost_records.",
            "blocked_provider_requests are budget-denied and are NOT real provider calls.",
            "sandbox_execution_seconds is in-sandbox compute only; active wall (LLM round-trips) dominates.",
            "cost_records lack per-step token attribution; totals are whole-run deduped union.",
        ],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    if not os.path.isdir(args.run_dir):
        raise BaselineError(f"not a directory: {args.run_dir}")
    base = build_baseline(args.run_dir)
    print(json.dumps(base, indent=2, ensure_ascii=False))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(base, fh, indent=2, ensure_ascii=False)
        print(f"\n[written] {args.json}", file=sys.stderr)


if __name__ == "__main__":
    try:
        main()
    except BaselineError as exc:
        print(f"BASELINE FAIL-CLOSED: {exc}", file=sys.stderr)
        sys.exit(2)
