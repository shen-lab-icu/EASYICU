"""Canonical fingerprints for Planner-owned scientific scope.

The functions here serialize existing Planner decisions for resume and typed
authority checks.  They do not choose or revise a cohort, exposure, outcome,
method, covariate set, or estimand.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, List, Mapping, Optional, Sequence, Tuple

from ..schema import AnalysisPlan, AnalysisStep
from .planned_role import verified_planned_analysis_role
from .run_input import (
    _HOST_COHORT_MATERIALIZER_AUTHORITY_KIND,
    _HOST_COHORT_MATERIALIZER_GENERATION_MODE,
)

__all__ = [
    "_normalise_scientific_text",
    "_plan_signature",
    "_plan_scientific_scope_signature",
    "_serializable_plan_scientific_scope_signature",
    "_step_scientific_signature",
    "completed_step_record_matches_plan",
    "legacy_host_checkpoint_may_inherit_plan_scope",
    "measurement_companion_input_closure_evidence_id",
    "verified_plan_scientific_scope_count",
    "verified_plan_evidence_rank",
]

_MEASUREMENT_COMPANION_INPUT_CLOSURE_ID_RE = re.compile(
    r"analysis_plan_input_closure_revision_(\d+)_([0-9a-f]{8})"
)


def measurement_companion_input_closure_evidence_id(
    *,
    revision: int,
    sha256: str,
) -> str:
    """Return the immutable id for one host-owned plan input closure."""

    if isinstance(revision, bool) or int(revision) < 0:
        raise ValueError("analysis plan revision must be a non-negative integer")
    digest = str(sha256 or "").strip().lower()
    if re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise ValueError("analysis plan input closure requires a SHA-256 digest")
    return f"analysis_plan_input_closure_revision_{int(revision)}_{digest[:8]}"


def legacy_host_checkpoint_may_inherit_plan_scope(
    record: Mapping[str, Any],
    *,
    plan_scope_count: int,
    completed_records: Sequence[Mapping[str, Any]],
) -> bool:
    """Permit one unambiguous pre-scope host checkpoint to migrate once."""

    return (
        not list(record.get("plan_scientific_signature") or [])
        and record.get("step_authority_kind")
        == _HOST_COHORT_MATERIALIZER_AUTHORITY_KIND
        and str(record.get("generation_mode") or "").strip().lower()
        == _HOST_COHORT_MATERIALIZER_GENERATION_MODE
        and (
            plan_scope_count == 1
            or any(
                list(item.get("plan_scientific_signature") or [])
                for item in completed_records
            )
        )
    )


def completed_step_record_matches_plan(
    record: Mapping[str, Any],
    *,
    step: AnalysisStep,
    expected_plan_scope: Sequence[Optional[str]],
    plan_scope_count: int,
    completed_records: Sequence[Mapping[str, Any]],
) -> bool:
    """Verify one completed step against a candidate immutable plan."""

    analysis_request = record.get("analysis_request")
    raw_step = (
        analysis_request.get("step") if isinstance(analysis_request, Mapping) else None
    )
    try:
        sealed_step = AnalysisStep.model_validate(raw_step)
    except (TypeError, ValueError):
        return False
    recorded_plan_scope = list(record.get("plan_scientific_signature") or [])
    legacy_host_without_scope = legacy_host_checkpoint_may_inherit_plan_scope(
        record,
        plan_scope_count=plan_scope_count,
        completed_records=completed_records,
    )
    return (
        verified_planned_analysis_role(record) is not None
        and _step_scientific_signature(sealed_step) == _step_scientific_signature(step)
        and (
            legacy_host_without_scope
            or recorded_plan_scope == list(expected_plan_scope)
        )
    )


def verified_plan_scientific_scope_count(paths: Sequence[Path]) -> int:
    """Count distinct valid plan-level scopes among immutable candidates."""

    scopes = set()
    for path in paths:
        try:
            plan = AnalysisPlan.model_validate_json(path.read_text(encoding="utf-8"))
        except (OSError, TypeError, ValueError):
            continue
        scopes.add(tuple(_serializable_plan_scientific_scope_signature(plan)))
    return len(scopes)


def verified_plan_evidence_rank(record: Mapping[str, Any]) -> Optional[int]:
    """Rank exact immutable plan evidence, rejecting unowned derivatives."""

    evidence_id = str(record.get("evidence_id") or "").strip()
    if evidence_id == "analysis_plan":
        return -1
    metadata = record.get("metadata")
    closure_authority = (
        record.get("producer") == "runtime_supervisor"
        and record.get("generation_mode") == "system"
        and isinstance(metadata, Mapping)
        and metadata.get("reason") == "measurement_companion_input_closure"
    )
    if evidence_id == "analysis_plan_input_closure":
        return 0 if closure_authority else None
    closure_match = _MEASUREMENT_COMPANION_INPUT_CLOSURE_ID_RE.fullmatch(evidence_id)
    if closure_match is not None:
        revision = int(closure_match.group(1))
        digest = str(record.get("sha256") or "").strip().lower()
        if (
            closure_authority
            and metadata.get("source_plan_revision") == revision
            and metadata.get("closure_sha256") == digest
            and closure_match.group(2) == digest[:8]
        ):
            return revision
        return None
    match = re.fullmatch(r"analysis_plan_revision_(\d+)(?:_[0-9a-f]{8})?", evidence_id)
    return int(match.group(1)) if match is not None else None


def _step_scientific_signature(step: AnalysisStep) -> Tuple[Any, ...]:
    """Fingerprint every Planner-owned field that can change execution.

    The current schema still carries exposure/outcome definitions, time windows,
    covariates, and missingness policy in ``intent``.  Until those coordinates
    are fully structured, ordinary semantic paraphrases cannot safely be
    distinguished from a changed estimand.  Only case/whitespace normalization
    is ignored.
    """

    return (
        step.step_id,
        step.method,
        tuple(step.inputs),
        tuple(step.expected_outputs),
        " ".join(str(step.intent or "").split()).casefold(),
        tuple(step.icu_rule_refs),
        step.planned_analysis_role,
        tuple(
            (
                requirement.requirement_id,
                requirement.outcome,
                requirement.outcome_type,
                requirement.method_family,
                requirement.exposure_source,
                requirement.analysis_role,
                requirement.analysis_set,
                requirement.required_for_step_success,
            )
            for requirement in step.model_requirements
        ),
        tuple(
            json.dumps(
                contract.model_dump(mode="json"),
                sort_keys=True,
                separators=(",", ":"),
            )
            for contract in step.input_consumption_contracts
        ),
        (
            json.dumps(
                step.trajectory_stability_spec.model_dump(mode="json"),
                sort_keys=True,
                separators=(",", ":"),
            )
            if step.trajectory_stability_spec is not None
            else None
        ),
    )


def _normalise_scientific_text(value: Any) -> Optional[str]:
    """Normalize cosmetic prose differences without erasing scientific edits."""

    if value is None:
        return None
    return " ".join(str(value).split()).casefold()


def _plan_scientific_scope_signature(plan: AnalysisPlan) -> Tuple[Optional[str], ...]:
    """Fingerprint Planner-owned science that applies to every plan step.

    ``revision`` is deliberately absent: it records plan history, not a change
    in the research question, analysis family, cohort, robustness contract, or
    rationale. Structured values use canonical JSON so the signature remains
    stable when it is serialized into a step record and loaded on resume.
    """

    plan_payload = plan.model_dump(
        mode="json",
        include={
            "cohort",
            "robustness_specs",
            "display_labels",
            "know_how_decisions",
        },
    )
    return (
        _normalise_scientific_text(plan.research_question),
        _normalise_scientific_text(plan.analysis_type),
        json.dumps(
            plan_payload.get("cohort"),
            sort_keys=True,
            separators=(",", ":"),
        ),
        json.dumps(
            plan_payload.get("robustness_specs", []),
            sort_keys=True,
            separators=(",", ":"),
        ),
        json.dumps(
            plan_payload.get("display_labels", {}),
            sort_keys=True,
            separators=(",", ":"),
        ),
        json.dumps(
            plan_payload.get("know_how_decisions", []),
            sort_keys=True,
            separators=(",", ":"),
        ),
        _normalise_scientific_text(plan.rationale),
    )


def _serializable_plan_scientific_scope_signature(
    plan: AnalysisPlan,
) -> List[Optional[str]]:
    """Return the plan-level signature in manifest-safe form."""

    return list(_plan_scientific_scope_signature(plan))


def _plan_signature(
    plan: AnalysisPlan,
) -> Tuple[Any, ...]:
    """Substantive fingerprint of a plan's step DAG and scientific requests.

    Intent remains authoritative because several estimand coordinates are not
    yet structured in :class:`AnalysisStep`; only case and whitespace changes
    are cosmetic. Structured model requirements, ICU rules, trajectory specs,
    typed DAG edges, and result roles are also included.
    """
    return (
        _plan_scientific_scope_signature(plan),
        tuple(_step_scientific_signature(step) for step in plan.steps),
    )
