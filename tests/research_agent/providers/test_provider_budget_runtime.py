from __future__ import annotations

from easyicu.research_agent.authority.provider_budget import (
    PROVIDER_CALL_BUDGET_RECEIPT_SCHEMA_VERSION,
)
from easyicu.research_agent.execution.provider_budget_runtime import (
    prepare_step_provider_budget,
)


def _prepare(tmp_path, *, prior_attempts=(), prior_record=None, step_record=None):
    record = {} if step_record is None else step_record
    runtime = prepare_step_provider_budget(
        prior_attempt_records=prior_attempts,
        prior_step_record=prior_record,
        run_dir=tmp_path,
        step_id="01_summary",
        step_record=record,
        max_provider_calls=3,
        max_llm_repairs=2,
        reserve_concept_audit=True,
        allow_terminal_initial_generation_restart=False,
    )
    return runtime, record


def test_fresh_step_budget_prepares_one_shared_owner(tmp_path) -> None:
    runtime, record = _prepare(tmp_path)

    assert runtime.integrity_error is None
    assert runtime.repair_budget.provider_budget is runtime.provider_budget
    assert runtime.repair_budget.llm_repair_attempts == 0
    assert runtime.reserved_final_category == "concept_audit"
    assert not runtime.receipt_path.exists()
    assert runtime.receipt_relative_path == str(
        runtime.receipt_path.relative_to(tmp_path)
    )
    assert runtime.provider_budget.snapshot()["limit"] == 3
    assert "step_llm_repair_attempts" not in record


def test_invalid_prior_snapshot_fails_closed_without_fresh_repair_budget(
    tmp_path,
) -> None:
    runtime, record = _prepare(
        tmp_path,
        prior_attempts=(
            {
                "step_llm_repair_attempts": 2,
                "step_llm_repair_classes": ["runtime", "contract"],
            },
        ),
        prior_record={
            "step_provider_call_budget": "unknown",
            "step_provider_call_attempts": 1,
            "step_provider_call_categories": ["coder_generation"],
        },
    )

    assert runtime.integrity_error == (
        "Prior provider-call budget snapshot is incomplete or invalid."
    )
    assert runtime.repair_budget.llm_repair_attempts == 2
    assert record["step_llm_repair_attempts"] == 2
    assert record["step_llm_repair_classes"] == ["runtime", "contract"]


def test_missing_required_receipt_is_not_reconstructed_from_snapshot(tmp_path) -> None:
    runtime, _ = _prepare(
        tmp_path,
        prior_record={
            "step_provider_call_receipt_version": (
                PROVIDER_CALL_BUDGET_RECEIPT_SCHEMA_VERSION
            ),
            "step_provider_call_receipt": (
                ".runtime/provider_call_budgets/01_summary.json"
            ),
            "step_provider_call_budget": 3,
            "step_provider_call_attempts": 1,
            "step_provider_call_categories": ["coder_generation"],
        },
    )

    assert runtime.integrity_error == (
        "Durable provider/repair receipt is missing for a prior reservation."
    )
    assert runtime.provider_budget.snapshot()["categories"] == ["coder_generation"]
