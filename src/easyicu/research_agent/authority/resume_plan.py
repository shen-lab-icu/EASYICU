"""Digest-verified analysis-plan selection for step-level resume."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from ..cohort.schema import (
    COHORT_LOCK_FILENAME,
    _load_locked_cohort_definition,
)
from ..planning.cohort_contract import (
    cohort_definition_sha,
    ensure_cohort_definition,
)
from ..schema import AnalysisPlan, ResearchContext
from .evidence_snapshot import load_current_evidence_snapshot
from .evidence_store import EvidenceStore
from .plan_input_closure import (
    close_measurement_companion_inputs,
    register_measurement_companion_input_closure,
)
from .plan_scope import (
    _serializable_plan_scientific_scope_signature,
    completed_step_record_matches_plan,
    verified_plan_evidence_rank,
    verified_plan_scientific_scope_count,
)
from .runtime_artifacts import (
    current_successful_step_records,
    verified_run_evidence_path,
)

__all__ = ["load_compatible_resume_plan", "resume_plan_candidate_paths"]


def resume_plan_candidate_paths(
    *,
    run_dir: Path,
    resume_state: Optional[Dict[str, Any]],
) -> List[Path]:
    """Return digest-verified immutable plan evidence, newest first."""

    del resume_state  # Evidence authority supersedes a mutable manifest path.
    records = list(load_current_evidence_snapshot(run_dir).records)
    ranked: List[tuple[int, int, Path]] = []
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            continue
        revision = verified_plan_evidence_rank(record)
        if revision is None:
            continue
        verified_path = verified_run_evidence_path(run_dir, record)
        if verified_path is not None:
            ranked.append((revision, index, verified_path))

    unique: List[Path] = []
    seen: set[Path] = set()
    for _revision, _index, candidate in sorted(ranked, reverse=True):
        resolved = candidate.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(candidate)
    return unique


def _plan_matches_completed_steps(
    *,
    plan: AnalysisPlan,
    completed_records: Sequence[Dict[str, Any]],
    completed_step_ids: set[str],
    plan_scope_count: int,
    locked_cohort_sha256: Optional[str],
) -> bool:
    # Cohort locking and execution both define ``plan.cohort is None`` as the
    # canonical implicit primary cohort.  Resume must compare that same
    # normalized contract; treating the shorthand as a missing authority makes
    # a digest-verified, already executed plan impossible to resume.  A plan
    # that actually drops or changes a non-default cohort still fails because
    # its normalized digest differs from the immutable lock.
    normalized_plan = ensure_cohort_definition(plan)
    if locked_cohort_sha256 is not None and (
        cohort_definition_sha(normalized_plan.cohort) != locked_cohort_sha256
    ):
        return False
    step_by_id = {step.step_id: step for step in plan.steps}
    if not plan.steps or not completed_step_ids <= set(step_by_id):
        return False
    expected_plan_scope = _serializable_plan_scientific_scope_signature(plan)
    return all(
        completed_step_record_matches_plan(
            record,
            step=step_by_id[str(record.get("step_id") or "")],
            expected_plan_scope=expected_plan_scope,
            plan_scope_count=plan_scope_count,
            completed_records=completed_records,
        )
        for record in completed_records
    )


def _read_candidate(path: Path) -> Optional[AnalysisPlan]:
    try:
        return AnalysisPlan.model_validate(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, TypeError, ValueError):
        return None


def load_compatible_resume_plan(
    *,
    run_dir: Path,
    resume_state: Optional[Dict[str, Any]],
    context: Optional[ResearchContext] = None,
    evidence: Optional[EvidenceStore] = None,
    prompt_pack_version: Optional[str] = None,
) -> tuple[Optional[AnalysisPlan], Optional[Path]]:
    """Select exact plan authority, migrating only deterministic input closure.

    The optional migration is fail-closed: it starts from digest-verified plan
    evidence, applies the canonical structural closure, and persists the result
    only if it then matches every successful sealed step exactly.
    """

    locked_cohort_sha256 = None
    if (run_dir / COHORT_LOCK_FILENAME).exists():
        locked_cohort_sha256 = cohort_definition_sha(
            _load_locked_cohort_definition(run_dir)
        )
    completed_records = [
        dict(record)
        for record in current_successful_step_records(
            (resume_state or {}).get("per_step_records") or []
        )
        if record.get("step_id") and record.get("step_id") != "00_probe"
    ]
    completed_step_ids = {str(record.get("step_id")) for record in completed_records}
    candidates = resume_plan_candidate_paths(
        run_dir=run_dir,
        resume_state=resume_state,
    )
    plan_scope_count = verified_plan_scientific_scope_count(candidates)
    parsed_candidates = [
        (plan, candidate)
        for candidate in candidates
        if (plan := _read_candidate(candidate)) is not None
    ]
    for plan, candidate in parsed_candidates:
        if _plan_matches_completed_steps(
            plan=plan,
            completed_records=completed_records,
            completed_step_ids=completed_step_ids,
            plan_scope_count=plan_scope_count,
            locked_cohort_sha256=locked_cohort_sha256,
        ):
            return plan, candidate

    if context is None or evidence is None:
        return None, None
    for plan, _candidate in parsed_candidates:
        closed_plan, closure_findings = close_measurement_companion_inputs(
            plan=plan,
            context=context,
        )
        if not closure_findings or not _plan_matches_completed_steps(
            plan=closed_plan,
            completed_records=completed_records,
            completed_step_ids=completed_step_ids,
            plan_scope_count=plan_scope_count,
            locked_cohort_sha256=locked_cohort_sha256,
        ):
            continue
        registered = register_measurement_companion_input_closure(
            run_dir=run_dir,
            evidence=evidence,
            plan=closed_plan,
            prompt_pack_version=prompt_pack_version,
        )
        return closed_plan, registered.evidence_path
    return None, None
