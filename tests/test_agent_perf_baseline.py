"""Tests for tools/agent_perf_baseline.py (P0-5a performance baseline harness).

Self-contained: builds synthetic run dirs, does not depend on any real run.
Covers Codex's required cases: receipt digest pos/neg, corrupt-file fail-closed,
multi-version cost dedup, 12+3=15 reconciliation, blocked-not-counted, wall segmentation.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path

import pytest

_MOD_PATH = Path(__file__).resolve().parents[1] / "tools" / "agent_perf_baseline.py"
_spec = importlib.util.spec_from_file_location("agent_perf_baseline", _MOD_PATH)
apb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(apb)


def _write_receipt(
    run_dir: Path,
    step_id: str,
    categories: list[str],
    *,
    tamper: bool = False,
    logical_repairs: list[dict] | None = None,
    schema_version: int | None = None,
    final_reservation_state: dict | None = None,
) -> None:
    resolved_schema = schema_version or (3 if logical_repairs is not None else 2)
    payload = {
        "categories": categories,
        "limit": 7,
        "reserved_final_category": "concept_audit",
        "schema_version": resolved_schema,
        "step_id": step_id,
    }
    if logical_repairs is not None:
        payload["logical_repairs"] = logical_repairs
    if final_reservation_state is not None:
        payload["final_reservation_state"] = final_reservation_state
    payload["sha256"] = apb._receipt_digest({k: v for k, v in payload.items()})
    if tamper:
        payload["categories"] = categories + [
            "sneaky_extra"
        ]  # digest no longer matches
    suffix = hashlib.sha256(step_id.encode()).hexdigest()[:16]
    out = run_dir / ".runtime" / "provider_call_budgets"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{suffix}.json").write_text(json.dumps(payload), encoding="utf-8")


def _write_cost_records(run_dir: Path, name: str, records: list[dict]) -> None:
    ev = run_dir / "evidence"
    ev.mkdir(parents=True, exist_ok=True)
    (ev / name).write_text(json.dumps(records), encoding="utf-8")


def _base_run(tmp_path: Path) -> Path:
    run = tmp_path / "run"
    run.mkdir()
    # Step 01: 6 calls (init + 3 repair + audit + analyzer)
    _write_receipt(
        run,
        "01_cohort_flow",
        [
            "initial_generation",
            "runtime_repair_patch",
            "runtime_repair_full_rewrite",
            "runtime_repair_patch",
            "concept_audit",
            "analyzer",
        ],
    )
    # Step 02: 6 calls (init + 4 repair + audit)  -> 12 step-scoped, 7 repair total
    _write_receipt(
        run,
        "02_exposure_derivation_and_qc",
        [
            "initial_generation",
            "concept_repair_patch",
            "concept_repair_full_rewrite",
            "concept_repair_patch",
            "concept_audit",
            "concept_repair_patch",
        ],
    )
    # cost_records: 9 coder + 3 analyzer (the 12 step-scoped calls) +
    # 3 run-level planner calls across two versions, with duplicate snapshots.
    v1 = [
        {
            "timestamp": f"2026-07-16T07:29:0{i}Z",
            "role": "coder",
            "prompt_tokens": 100,
            "completion_tokens": 10,
            "total_tokens": 110,
        }
        for i in range(5)
    ]
    v1 += [
        {
            "timestamp": f"2026-07-16T07:30:0{i}Z",
            "role": "planner",
            "prompt_tokens": 200,
            "completion_tokens": 20,
            "total_tokens": 220,
        }
        for i in range(3)
    ]
    v1 += [
        {
            "timestamp": f"2026-07-16T07:30:1{i}Z",
            "role": "analyzer",
            "prompt_tokens": 150,
            "completion_tokens": 15,
            "total_tokens": 165,
        }
        for i in range(3)
    ]
    v2 = list(v1)  # full duplicate snapshot
    v2 += [
        {
            "timestamp": f"2026-07-16T07:31:0{i}Z",
            "role": "coder",
            "prompt_tokens": 100,
            "completion_tokens": 10,
            "total_tokens": 110,
        }
        for i in range(4)
    ]
    _write_cost_records(run, "cost_records__cost_records.json", v1)
    _write_cost_records(run, "cost_records_v2__cost_records.json", v2)
    (run / "cost_records.json").write_text(json.dumps(v2), encoding="utf-8")
    # audit_log with two explicit run sessions for step 02
    lines = [
        {
            "timestamp": "2026-07-16T07:28:59Z",
            "event": "Research context built.",
        },
        {
            "timestamp": "2026-07-16T07:29:00Z",
            "step_id": "02_exposure_derivation_and_qc",
            "event": "Step 2/12 started: 02_exposure_derivation_and_qc.",
        },
        {
            "timestamp": "2026-07-16T07:29:10Z",
            "step_id": "02_exposure_derivation_and_qc",
            "event": "Stopped after requested step: 02_exposure_derivation_and_qc.",
        },
        {
            "timestamp": "2026-07-16T07:29:11Z",
            "event": "Research-agent run complete.",
        },
        {
            "timestamp": "2026-07-16T07:40:00Z",
            "event": "Research context built.",
        },
        {
            "timestamp": "2026-07-16T07:40:01Z",
            "step_id": "02_exposure_derivation_and_qc",
            "event": "Step 2/12 started: 02_exposure_derivation_and_qc.",
        },
        {
            "timestamp": "2026-07-16T07:40:20Z",
            "step_id": "02_exposure_derivation_and_qc",
            "event": "Stopped after requested step: 02_exposure_derivation_and_qc.",
        },
        {
            "timestamp": "2026-07-16T07:40:21Z",
            "event": "Research-agent run complete.",
        },
    ]
    (run / "audit_log.jsonl").write_text(
        "\n".join(json.dumps(x) for x in lines), encoding="utf-8"
    )
    # run_status with one blocked budget request
    (run / "run_status.json").write_text(
        json.dumps(
            {
                "gates": {
                    "analysis_errors": [
                        "... LLM provider-call budget unavailable for step '02...': ..."
                    ]
                }
            }
        ),
        encoding="utf-8",
    )
    return run


def test_call_reconciliation_15_equals_12_plus_3(tmp_path):
    b = apb.build_baseline(str(_base_run(tmp_path)))
    c = b["calls"]
    assert c["step_scoped_calls"] == 12
    assert c["run_level_planner_calls"] == 3
    assert c["all_provider_calls"] == 15
    assert c["cost_record_calls"] == 15
    assert c["reconciliation_delta"] == 0
    assert c["repair_calls"] == 7


def test_blocked_request_not_counted_as_real_call(tmp_path):
    b = apb.build_baseline(str(_base_run(tmp_path)))
    assert b["calls"]["blocked_provider_requests"] == 1
    # blocked is NOT part of all_provider_calls
    assert b["calls"]["all_provider_calls"] == 15


def test_multi_version_cost_records_deduped(tmp_path):
    b = apb.build_baseline(str(_base_run(tmp_path)))
    tok = b["tokens_from_deduped_cost_records"]
    # 9 coder + 3 analyzer + 3 planner despite duplicated snapshots.
    assert tok["by_role"]["coder"]["n"] == 9
    assert tok["by_role"]["analyzer"]["n"] == 3
    assert tok["by_role"]["planner"]["n"] == 3
    assert tok["total"]["n"] == 15


def test_receipt_digest_valid_passes(tmp_path):
    run = _base_run(tmp_path)
    apb.build_baseline(str(run))  # no raise


def test_receipt_digest_tampered_fails_closed(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    _write_receipt(
        run, "01_cohort_flow", ["initial_generation", "concept_audit"], tamper=True
    )
    with pytest.raises(apb.BaselineError, match="digest invalid"):
        apb.read_receipts(str(run), {})


def test_schema_v3_logical_repair_ledger_is_reported(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    categories = ["initial_generation"]
    _write_receipt(
        run,
        "01_cohort_flow",
        categories,
        logical_repairs=[
            {
                "attempt_id": 1,
                "repair_class": "contract",
                "provider_history_len": 1,
                "provider_history_sha256": apb._receipt_digest(
                    {"categories": categories}
                ),
            }
        ],
    )

    receipts = apb.read_receipts(str(run), {})

    assert receipts[0]["total_calls"] == 1
    assert receipts[0]["logical_repair_attempts"] == 1
    assert receipts[0]["logical_repair_classes"] == ["contract"]


def test_schema_v3_logical_history_inconsistency_fails_closed(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    _write_receipt(
        run,
        "01_cohort_flow",
        ["initial_generation"],
        logical_repairs=[
            {
                "attempt_id": 1,
                "repair_class": "runtime",
                "provider_history_len": 1,
                "provider_history_sha256": "0" * 64,
            }
        ],
    )

    with pytest.raises(apb.BaselineError, match="logical repair history"):
        apb.read_receipts(str(run), {})


def test_schema_v4_final_audit_state_is_validated(tmp_path):
    run = tmp_path / "run_v4"
    run.mkdir()
    categories = ["initial_generation", "concept_audit"]
    _write_receipt(
        run,
        "01_model",
        categories,
        schema_version=4,
        logical_repairs=[],
        final_reservation_state={
            "required_token": "audit-authority",
            "bound_provider_history_len": 1,
            "bound_provider_history_sha256": apb._receipt_digest(
                {"categories": categories[:1]}
            ),
            "completed_token": "audit-authority",
            "released": False,
        },
    )
    assert apb.read_receipts(str(run), {})[0]["total_calls"] == 2

    path = next((run / ".runtime" / "provider_call_budgets").glob("*.json"))
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["final_reservation_state"]["released"] = True
    payload["final_reservation_state"]["completed_token"] = None
    body = {key: value for key, value in payload.items() if key != "sha256"}
    payload["sha256"] = apb._receipt_digest(body)
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(apb.BaselineError, match="reservation state"):
        apb.read_receipts(str(run), {})


def test_corrupt_cost_file_fails_closed(tmp_path):
    run = _base_run(tmp_path)
    (run / "evidence" / "cost_records_v9__cost_records.json").write_text(
        "{not json", encoding="utf-8"
    )
    with pytest.raises(apb.BaselineError, match="unreadable JSON"):
        apb.aggregate_cost_records(str(run), {})


def test_wall_time_uses_explicit_audit_sessions_and_excludes_inter_run_idle(tmp_path):
    b = apb.build_baseline(str(_base_run(tmp_path)))
    w = b["wall_time"]
    # Two host-marked sessions: 10s + 19s for step 02. The ~11-minute gap
    # between sessions is excluded without a tunable idle threshold.
    assert w["step_active_wall_seconds"]["02_exposure_derivation_and_qc"] == 29.0
    assert w["session_active_wall_seconds"] == 33.0
    assert w["run_session_count"] == 2
    assert w["wall_method"] == "explicit_audit_session_boundaries"
    assert "sandbox_execution_seconds" in w


def test_incomplete_audit_session_fails_closed(tmp_path):
    run = _base_run(tmp_path)
    with (run / "audit_log.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(
            "\n"
            + json.dumps(
                {
                    "timestamp": "2026-07-16T07:50:00Z",
                    "event": "Research context built.",
                }
            )
        )
    with pytest.raises(apb.BaselineError, match="incomplete run session"):
        apb.build_baseline(str(run))


def test_inputs_are_sha_recorded(tmp_path):
    b = apb.build_baseline(str(_base_run(tmp_path)))
    assert len(b["inputs_sha256"]) >= 4
    for sha in b["inputs_sha256"].values():
        assert len(sha) == 64
