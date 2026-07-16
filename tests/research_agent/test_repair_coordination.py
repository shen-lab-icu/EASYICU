"""Unit contract for repair_coordination (A2 batch-1 extraction).

These tests pin the EXACT behavioral surface of the four budget-accounting
closures extracted from ``pipeline_execute``: step_record key names and write
order, the neutral provider probe (never a consume), the persisted
``step_llm_repair_classes`` contract that resume replays monotonically, and
the all-or-nothing authorization semantics of the deterministic concept
repair.
"""

from __future__ import annotations

import json

from easyicu.research_agent.provider_budget import StepProviderCallBudget
from easyicu.research_agent.repair_coordination import (
    StepRepairBudget,
    authorized_deterministic_concept_repair,
)

STEP_ID = "02_exposure_derivation_and_qc"


def _budget(tmp_path, *, limit: int = 7):
    return StepProviderCallBudget(
        limit,
        step_id=STEP_ID,
        receipt_path=tmp_path / "receipt.json",
        reserved_final_category="concept_audit",
    )


def _repair_budget(tmp_path, *, limit: int = 7, max_llm: int = 3, initial: int = 0):
    provider = _budget(tmp_path, limit=limit)
    step_record: dict = {}
    budget = StepRepairBudget(
        provider_budget=provider,
        step_record=step_record,
        max_llm_repairs=max_llm,
        initial_llm_repair_attempts=initial,
        provider_receipt_relative_path=".runtime/provider_call_budgets/x.json",
    )
    return provider, step_record, budget


def test_sync_provider_writes_exact_key_set(tmp_path):
    provider, step_record, budget = _repair_budget(tmp_path)
    budget.sync_provider()
    assert list(step_record) == [
        "step_provider_call_budget_scope",
        "step_provider_call_budget",
        "step_provider_call_attempts",
        "step_provider_call_remaining",
        "step_provider_call_budget_exhausted",
        "step_provider_call_categories",
        "step_provider_call_reserved_category",
        "step_provider_call_reservation_released",
        "step_provider_call_receipt_version",
        "step_provider_call_receipt",
    ]
    assert (
        step_record["step_provider_call_budget_scope"]
        == "coder_generation_repair_concept_audit_and_analyzer"
    )
    # receipt path is only reported once something was actually paid
    assert step_record["step_provider_call_receipt"] is None
    provider.consume("initial_generation")
    budget.sync_provider()
    assert (
        step_record["step_provider_call_receipt"]
        == ".runtime/provider_call_budgets/x.json"
    )


def test_probe_never_consumes_or_touches_receipt(tmp_path):
    provider, step_record, budget = _repair_budget(tmp_path)
    assert budget.provider_available()
    assert provider.used == 0
    assert not (tmp_path / "receipt.json").exists()


def test_probe_refusal_records_unavailable_and_syncs(tmp_path):
    provider, step_record, budget = _repair_budget(tmp_path, limit=1)
    # only the reserved audit slot remains -> non-audit probe is refused
    assert not budget.provider_available()
    assert step_record["step_provider_call_repair_unavailable"] is True
    assert step_record["step_provider_call_budget"] == 1  # sync ran
    assert provider.used == 0  # still no consume


def test_consume_appends_repair_classes_in_order(tmp_path):
    provider, step_record, budget = _repair_budget(tmp_path, max_llm=3)
    assert budget.consume("concept")
    assert budget.consume("runtime")
    assert step_record["step_llm_repair_attempts"] == 2
    assert step_record["step_llm_repair_budget"] == 3
    assert step_record["step_llm_repair_classes"] == ["concept", "runtime"]


def test_logical_exhaustion_marks_record_and_refuses(tmp_path):
    provider, step_record, budget = _repair_budget(tmp_path, max_llm=1)
    assert budget.consume("concept")
    assert not budget.consume("contract")
    assert step_record["step_llm_repair_budget_exhausted"] is True
    assert step_record["step_llm_repair_classes"] == ["concept"]
    assert budget.llm_repair_attempts == 1


def test_resume_initial_attempts_count_against_allowance(tmp_path):
    provider, step_record, budget = _repair_budget(tmp_path, max_llm=3, initial=3)
    assert not budget.logical_available()
    assert not budget.consume("runtime")
    assert "step_llm_repair_classes" not in step_record  # nothing new appended


def test_authorized_repair_is_all_or_nothing():
    script = "helper_result = {}\nassert isinstance(helper_result, dict)\n"

    calls: list = []

    def approve(payload, **kwargs):
        calls.append(payload)
        return payload

    def deny(payload, **kwargs):
        calls.append(payload)
        return None

    # no matching mechanical repair -> untouched, no authorization attempted
    code, names = authorized_deterministic_concept_repair(
        "x = 1\n", ["unrelated"], authorize=approve, step=None, source="test"
    )
    assert (code, names) == ("x = 1\n", [])

    # a denied authorization rejects the WHOLE candidate even if it matched
    code, names = authorized_deterministic_concept_repair(
        script,
        ["never require `isinstance(helper_result, dict)`"],
        authorize=deny,
        step=None,
        source="test",
    )
    assert code == script
    assert names == []


def test_receipt_on_disk_matches_snapshot_projection(tmp_path):
    provider, step_record, budget = _repair_budget(tmp_path)
    provider.consume("initial_generation")
    provider.consume("concept_repair")
    budget.sync_provider()
    payload = json.loads((tmp_path / "receipt.json").read_text(encoding="utf-8"))
    assert payload["categories"] == step_record["step_provider_call_categories"]
    assert payload["limit"] == step_record["step_provider_call_budget"]
