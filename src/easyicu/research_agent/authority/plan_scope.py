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
            or _plan_scope_signatures_match(
                recorded_plan_scope,
                list(expected_plan_scope),
            )
        )
    )


def _drop_legacy_empty_literature_design_decisions(value: Any) -> Any:
    """Normalize one schema-added empty design field for legacy comparison."""

    if isinstance(value, list):
        return [
            _drop_legacy_empty_literature_design_decisions(item) for item in value
        ]
    if not isinstance(value, dict):
        return value
    return {
        key: _drop_legacy_empty_literature_design_decisions(item)
        for key, item in value.items()
        if not (key == "literature_design_decisions" and item == [])
    }


def _plan_scope_signatures_match(
    recorded: Sequence[Optional[str]],
    expected: Sequence[Optional[str]],
) -> bool:
    """Compare plan scope with one fail-closed legacy schema normalization."""

    if list(recorded) == list(expected):
        return True
    if len(recorded) != 4 or len(expected) != 4:
        return False
    if recorded[:2] != expected[:2] or recorded[3:] != expected[3:]:
        return False
    try:
        recorded_payload = json.loads(str(recorded[2]))
        expected_payload = json.loads(str(expected[2]))
    except (TypeError, ValueError, json.JSONDecodeError):
        return False
    return _drop_legacy_empty_literature_design_decisions(
        recorded_payload
    ) == _drop_legacy_empty_literature_design_decisions(expected_payload)


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

    ``scientific_capability`` is here because it selects the *execution owner*.
    It was added to ``AnalysisStep`` without being added here, and the omission
    was demonstrable rather than theoretical: flipping only that field on an
    association primary step moved the resolved owner from
    ``host_deterministic`` to ``agent_coded`` while this signature stayed
    byte-identical.  A step whose result would be computed by the LLM coder
    instead of the sealed host executor is not the same step, and a
    seal/resume comparison that cannot see the difference would accept the
    substitution under the approved plan's identity.

    ``scientific_action_id`` similarly binds the exact method/resource action
    published to Planner. Changing RMST to a PH diagnostic, or a reviewed
    DeLong kernel to an unrelated generated method, is a scientific change even
    if the free-text intent happens to remain the same.

    This is a comparison between two signatures computed by the same code --
    the sealed record is re-validated through ``AnalysisStep`` before it is
    fingerprinted -- so no stored digest changes: a pre-existing record
    validates the field to ``None`` and matches a live plan that still declares
    nothing.  What it newly refuses is a plan that changed it.
    """

    structured_payload = _analysis_step_scientific_authority_payload(step)
    return (
        step.step_id,
        step.method,
        tuple(step.inputs),
        tuple(step.expected_outputs),
        " ".join(str(step.intent or "").split()).casefold(),
        tuple(step.icu_rule_refs),
        step.planned_analysis_role,
        step.scientific_action_id,
        step.scientific_capability,
        json.dumps(structured_payload, sort_keys=True, separators=(",", ":")),
    )


# Every public AnalysisStep field belongs to exactly one authority class.  Most
# of the step is scientific authority today; the empty classes are deliberate,
# not omissions.  Keeping the classification explicit means a newly added
# public field cannot silently become invisible to seal/resume identity.
_ANALYSIS_STEP_CORE_SCIENTIFIC_AUTHORITY_FIELDS = frozenset(
    {
        "step_id",
        "method",
        "inputs",
        "expected_outputs",
        "intent",
        "icu_rule_refs",
        "planned_analysis_role",
        "scientific_action_id",
        "scientific_capability",
    }
)
_ANALYSIS_STEP_STRUCTURED_SCIENTIFIC_AUTHORITY_FIELDS = frozenset(
    {
        "literature_citation_keys",
        "literature_design_bindings",
        "sensitivity_spec_ids",
        "descriptive_claim",
        "figure_panels",
        "model_requirements",
        "family_primary_result_requirement",
        "input_consumption_contracts",
        "table_one_spec",
        "trajectory_stability_spec",
        "exposure_outcome_distribution_spec",
        "cohort_definition_spec",
        "measurement_audit_spec",
        "robustness_replay_spec",
    }
)
_ANALYSIS_STEP_PRESENTATION_ONLY_FIELDS = frozenset()
_ANALYSIS_STEP_RUNTIME_ONLY_FIELDS = frozenset()


def _analysis_step_scientific_authority_payload(
    step: AnalysisStep,
) -> dict[str, Any]:
    """Return the canonical structured portion of one step's authority.

    The previous fingerprint hand-copied eight fields from each
    ``PlannedModelRequirement`` and therefore omitted its covariates, model
    terms, level/reference and primary-contrast declarations.  It also stopped
    after ``trajectory_stability_spec`` and missed every later ``*_spec``.
    Serializing the classified fields through Pydantic keeps nested contracts
    complete and gives schema drift one fail-closed diagnostic surface.
    """

    classes = (
        _ANALYSIS_STEP_CORE_SCIENTIFIC_AUTHORITY_FIELDS,
        _ANALYSIS_STEP_STRUCTURED_SCIENTIFIC_AUTHORITY_FIELDS,
        _ANALYSIS_STEP_PRESENTATION_ONLY_FIELDS,
        _ANALYSIS_STEP_RUNTIME_ONLY_FIELDS,
    )
    classified: set[str] = set()
    overlaps: set[str] = set()
    for fields in classes:
        overlaps.update(classified.intersection(fields))
        classified.update(fields)
    public_fields = set(AnalysisStep.model_fields)
    if overlaps or classified != public_fields:
        missing = sorted(public_fields - classified)
        unknown = sorted(classified - public_fields)
        raise RuntimeError(
            "AnalysisStep authority classification drift: "
            f"missing={missing!r}, unknown={unknown!r}, overlaps={sorted(overlaps)!r}"
        )
    return step.model_dump(
        mode="json",
        include=_ANALYSIS_STEP_STRUCTURED_SCIENTIFIC_AUTHORITY_FIELDS,
    )


def _normalise_scientific_text(value: Any) -> Optional[str]:
    """Normalize cosmetic prose differences without erasing scientific edits."""

    if value is None:
        return None
    return " ".join(str(value).split()).casefold()


_ANALYSIS_PLAN_CORE_SCIENTIFIC_AUTHORITY_FIELDS = frozenset(
    {"research_question", "analysis_type", "rationale"}
)
_ANALYSIS_PLAN_STRUCTURED_SCIENTIFIC_AUTHORITY_FIELDS = frozenset(
    {
        "cohort",
        "endpoint",
        "robustness_specs",
        "know_how_decisions",
        "evalue_conversion_spec",
        "subgroup_analysis_spec",
        "design_selection",
    }
)
_ANALYSIS_PLAN_STEP_AUTHORITY_FIELDS = frozenset({"steps"})
_ANALYSIS_PLAN_PRESENTATION_ONLY_FIELDS = frozenset({"display_labels"})
_ANALYSIS_PLAN_RUNTIME_ONLY_FIELDS = frozenset({"revision"})


def _classified_plan_scientific_payload(plan: AnalysisPlan) -> dict[str, Any]:
    """Return structured plan science and fail when schema fields drift."""

    classes = (
        _ANALYSIS_PLAN_CORE_SCIENTIFIC_AUTHORITY_FIELDS,
        _ANALYSIS_PLAN_STRUCTURED_SCIENTIFIC_AUTHORITY_FIELDS,
        _ANALYSIS_PLAN_STEP_AUTHORITY_FIELDS,
        _ANALYSIS_PLAN_PRESENTATION_ONLY_FIELDS,
        _ANALYSIS_PLAN_RUNTIME_ONLY_FIELDS,
    )
    classified: set[str] = set()
    overlaps: set[str] = set()
    for fields in classes:
        overlaps.update(classified.intersection(fields))
        classified.update(fields)
    public_fields = set(AnalysisPlan.model_fields)
    if overlaps or classified != public_fields:
        raise RuntimeError(
            "AnalysisPlan authority classification drift: "
            f"missing={sorted(public_fields - classified)!r}, "
            f"unknown={sorted(classified - public_fields)!r}, "
            f"overlaps={sorted(overlaps)!r}"
        )
    return plan.model_dump(
        mode="json",
        include=_ANALYSIS_PLAN_STRUCTURED_SCIENTIFIC_AUTHORITY_FIELDS,
    )


def _plan_scientific_scope_signature(plan: AnalysisPlan) -> Tuple[Optional[str], ...]:
    """Fingerprint Planner-owned science that applies to every plan step.

    ``display_labels`` and ``revision`` are deliberately absent: presentation
    and plan history cannot change execution. ``steps`` have their own exact
    per-step authority signature. Every remaining public field is classified
    above so a schema addition cannot silently disappear from resume identity.
    """

    plan_payload = _classified_plan_scientific_payload(plan)
    return (
        _normalise_scientific_text(plan.research_question),
        _normalise_scientific_text(plan.analysis_type),
        json.dumps(
            plan_payload,
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
