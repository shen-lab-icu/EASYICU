"""Host-owned bootstrap for one execute-phase step attempt.

This owner prepares only control-plane state that must exist before generated
code, deterministic repair, or scientific auditing can run:

* monotonic attempt identity and prior-record selection;
* the initial execution-cohort role and its content digest; and
* crash-safe provider/logical-repair budget restoration.

It deliberately does not choose a scientific method or mutate a plan.  The
returned ``step_record`` and budget objects are the same mutable instances the
execute coordinator continues to own.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableSequence, Sequence

from ..authority.evidence_store import sha256_of_file
from ..authority.runtime_artifacts import current_step_records
from ..contracts.primary_cohort import primary_analysis_cohort_producer_uses_universe
from ..repairs.semantic_boundary import SemanticRepairRecorder
from ..schema import AnalysisPlan, AnalysisStep, ValidationFinding
from .cohort_routing import step_execution_cohort_path
from .development_sample import DEVELOPMENT_PRIMARY_COHORT_CONFIRMATION_ROLE
from .provider_budget_runtime import (
    StepProviderBudgetRuntime,
    prepare_step_provider_budget,
)


RAW_UNIVERSE_EXECUTION_ROLE = "raw_universe_for_primary_analysis_cohort_producer"


@dataclass(frozen=True)
class StepAttemptBootstrap:
    """Prepared identity, route, and durable budgets for one attempt."""

    prior_attempt_records: List[Mapping[str, Any]]
    prior_step_record: Mapping[str, Any] | None
    attempt_sequence: int
    attempt_id: str
    review_checkpoint_id: str
    step_record: Dict[str, Any]
    execution_cohort_path: Path
    budget_runtime: StepProviderBudgetRuntime


def prepare_step_attempt_bootstrap(
    *,
    resume_state: Mapping[str, Any] | None,
    per_step_records: Sequence[Mapping[str, Any]],
    shared_lock: Any,
    step: AnalysisStep,
    plan: AnalysisPlan,
    run_id: str,
    run_dir: Path,
    universe_path: Path,
    cohort_path: Path,
    plan_scientific_signature: Any,
    findings: MutableSequence[ValidationFinding],
    max_provider_calls: int,
    max_llm_repairs: int,
    reserve_concept_audit: bool,
    allow_terminal_initial_generation_restart: bool,
) -> StepAttemptBootstrap:
    """Restore one attempt's host control plane without buying new authority."""

    with shared_lock:
        resume_history = (
            list(
                resume_state.get("step_attempt_history")
                or resume_state.get("per_step_records")
                or []
            )
            if isinstance(resume_state, Mapping)
            else []
        )
        candidate_history = resume_history + list(per_step_records)
        prior_attempt_records = [
            record
            for record in candidate_history
            if isinstance(record, Mapping)
            and str(record.get("step_id") or "") == step.step_id
        ]
        prior_step_record = next(
            (
                record
                for record in current_step_records(prior_attempt_records)
                if str(record.get("step_id") or "") == step.step_id
            ),
            None,
        )

    prior_attempt_sequences = [
        int(record.get("attempt_sequence"))
        for record in prior_attempt_records
        if isinstance(record.get("attempt_sequence"), int)
        and int(record.get("attempt_sequence")) >= 1
    ]
    attempt_sequence = (
        max(
            prior_attempt_sequences,
            default=len(prior_attempt_records),
        )
        + 1
    )
    attempt_id = f"{run_id}:{step.step_id}:{attempt_sequence}"
    review_checkpoint_id = f"{attempt_id}:deterministic_review"
    step_record: Dict[str, Any] = {
        "step_id": step.step_id,
        "intent": step.intent,
        "planned_analysis_role": step.planned_analysis_role,
        "attempt_id": attempt_id,
        "attempt_sequence": attempt_sequence,
        "review_checkpoint_id": review_checkpoint_id,
        "plan_scientific_signature": plan_scientific_signature,
    }

    execution_cohort_path = step_execution_cohort_path(
        step=step,
        plan=plan,
        run_dir=run_dir,
        universe_path=universe_path,
        cohort_path=cohort_path,
    )
    if execution_cohort_path == universe_path:
        step_record.update(
            {
                "execution_cohort_role": RAW_UNIVERSE_EXECUTION_ROLE,
                "execution_cohort_sha256": sha256_of_file(universe_path),
                "authoritative_analysis_cohort_sha256": sha256_of_file(cohort_path),
            }
        )
    elif primary_analysis_cohort_producer_uses_universe(step=step, plan=plan):
        step_record.update(
            {
                "execution_cohort_role": DEVELOPMENT_PRIMARY_COHORT_CONFIRMATION_ROLE,
                "execution_cohort_sha256": sha256_of_file(cohort_path),
                "authoritative_analysis_cohort_sha256": sha256_of_file(cohort_path),
                "paper_authority": False,
            }
        )

    budget_runtime = prepare_step_provider_budget(
        prior_attempt_records=prior_attempt_records,
        prior_step_record=prior_step_record,
        run_dir=run_dir,
        step_id=step.step_id,
        step_record=step_record,
        max_provider_calls=max_provider_calls,
        max_llm_repairs=max_llm_repairs,
        reserve_concept_audit=reserve_concept_audit,
        allow_terminal_initial_generation_restart=(
            allow_terminal_initial_generation_restart
        ),
    )
    budget_runtime.repair_budget.bind_semantic_escalation_recorder(
        SemanticRepairRecorder(
            step_record=step_record,
            findings=findings,
            lock=shared_lock,
            step_id=step.step_id,
            attempt_id=attempt_id,
        )
    )
    budget_runtime.repair_budget.sync_provider()

    return StepAttemptBootstrap(
        prior_attempt_records=prior_attempt_records,
        prior_step_record=prior_step_record,
        attempt_sequence=attempt_sequence,
        attempt_id=attempt_id,
        review_checkpoint_id=review_checkpoint_id,
        step_record=step_record,
        execution_cohort_path=execution_cohort_path,
        budget_runtime=budget_runtime,
    )
