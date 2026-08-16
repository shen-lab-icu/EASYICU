from __future__ import annotations

import inspect
import threading
from types import SimpleNamespace

from easyicu.research_agent.authority.provider_budget import StepProviderCallBudget
from easyicu.research_agent.execution.phase import (
    _execute_step,
    _step_settle_initial_code,
)
from easyicu.research_agent.execution.repair_reservation import StepRepairReservation
from easyicu.research_agent.execution.step_attempt_bootstrap import (
    prepare_step_attempt_bootstrap,
)
from easyicu.research_agent.repairs.coordination import StepRepairBudget
from easyicu.research_agent.repairs.reasons import RepairPromptAuthority
from easyicu.research_agent.schema import AnalysisPlan, AnalysisStep


def _step() -> AnalysisStep:
    return AnalysisStep(
        step_id="01_summary",
        intent="Summarize the locked cohort.",
        inputs=["stay_id"],
        expected_outputs=["table:summary"],
        method="descriptive_summary",
    )


def test_bootstrap_restores_monotonic_attempt_identity_and_budget(tmp_path) -> None:
    step = _step()
    plan = AnalysisPlan(research_question="Summarize the cohort.", steps=[step])
    universe = tmp_path / "cohort_universe.parquet"
    cohort = tmp_path / "cohort_analysis.parquet"
    universe.write_bytes(b"universe")
    cohort.write_bytes(b"cohort")
    prior = {
        "step_id": step.step_id,
        "attempt_id": "run:01_summary:4",
        "attempt_sequence": 4,
        "step_llm_repair_attempts": 1,
        "step_llm_repair_classes": ["runtime"],
    }
    findings: list = []

    result = prepare_step_attempt_bootstrap(
        resume_state={"step_attempt_history": [prior]},
        per_step_records=[],
        shared_lock=threading.Lock(),
        step=step,
        plan=plan,
        run_id="run",
        run_dir=tmp_path,
        universe_path=universe,
        cohort_path=cohort,
        plan_scientific_signature=[{"step_id": step.step_id}],
        findings=findings,
        max_provider_calls=5,
        max_llm_repairs=2,
        reserve_concept_audit=False,
        allow_terminal_initial_generation_restart=False,
    )

    assert result.prior_step_record is prior
    assert result.attempt_sequence == 5
    assert result.attempt_id == "run:01_summary:5"
    assert result.review_checkpoint_id == ("run:01_summary:5:deterministic_review")
    assert result.step_record["plan_scientific_signature"] == [
        {"step_id": step.step_id}
    ]
    assert result.budget_runtime.repair_budget.llm_repair_attempts == 1
    assert result.step_record["step_llm_repair_classes"] == ["runtime"]


class _Checkpoint:
    def __init__(self) -> None:
        self.ensure_calls: list[tuple[str, str]] = []
        self.checkpoint_calls: list[tuple[str, dict]] = []

    def ensure_candidate(self, code: str, *, reason: str) -> None:
        self.ensure_calls.append((code, reason))

    def checkpoint_state(self, stage: str, *, extra: dict) -> None:
        self.checkpoint_calls.append((stage, extra))


def test_repair_reservation_binds_and_checkpoints_exact_attempt(tmp_path) -> None:
    step = _step()
    step_record: dict = {}
    provider_budget = StepProviderCallBudget(
        4,
        step_id=step.step_id,
        receipt_path=tmp_path / "provider.json",
    )
    repair_budget = StepRepairBudget(
        provider_budget=provider_budget,
        step_record=step_record,
        max_llm_repairs=2,
    )
    checkpoint = _Checkpoint()
    reservation = StepRepairReservation(
        step=step,
        repair_budget=repair_budget,
        checkpoint_authority=checkpoint,  # type: ignore[arg-type]
        attempt_state=SimpleNamespace(coordinates=None),  # type: ignore[arg-type]
        coder_context=SimpleNamespace(model_dump=lambda **_: {"question": "q"}),
        coder_authority=SimpleNamespace(payload=lambda: {"authority": "host"}),
        resolved_inputs_sha256="a" * 64,
        coder_provider_identity_sha256="b" * 64,
        prompt_version="test-prompts/1",
        run_input_capsule_sha256="c" * 64,
        deterministic_gate_stamp={"deterministic_gate_fingerprint": "d" * 64},
    )

    assert reservation.consume(
        "runtime",
        before_code="value = 1\n",
        repair_ticket="TypeError: test",
        repair_authority=RepairPromptAuthority(),
        provider_category="runtime_repair",
        failure_status="execution_failed",
    )

    assert checkpoint.ensure_calls == [("value = 1\n", "pre_repair_authority_binding")]
    assert checkpoint.checkpoint_calls[0][0] == "repair_transport_pending"
    extra = checkpoint.checkpoint_calls[0][1]
    assert extra["capsule_pending_repair_attempt_id"] == 1
    assert len(extra["capsule_pending_repair_binding_sha256"]) == 64
    assert extra["capsule_pending_repair_failure_status"] == "execution_failed"
    assert step_record["step_llm_repair_classes"] == ["runtime"]


def test_execute_worker_delegates_attempt_bootstrap_and_repair_reservation() -> None:
    source = (
        inspect.getsource(_execute_step)
        + "\n"
        + inspect.getsource(_step_settle_initial_code)
    )

    assert "prepare_step_attempt_bootstrap(" in source
    assert "StepRepairReservation(" in source
    assert "def _consume_llm_repair_budget(" not in source
