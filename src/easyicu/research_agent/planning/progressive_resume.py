"""Dependency-bound development replay for Progressive Planner prefixes.

This module owns deterministic validation and reconstruction of a previously
materialized prefix.  It does not call a provider, persist evidence, or decide
whether a run is eligible for development replay; those responsibilities stay
with the Planner transport and pipeline configuration owners.
"""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from pydantic import ValidationError

from ..canonical_json import canonical_sha256
from ..schema import AnalysisPlan, ResearchContext
from .adjustment_authority import (
    AdjustmentAuthorityError,
    AdjustmentSetAuthority,
    validate_plan_against_adjustment_authority,
)
from .progressive_compiler import compile_progressive_plan
from .progressive_contract import (
    ProgressiveFoundationMaterialization,
    ProgressiveOutlineStep,
    ProgressivePlanCompileError,
    ProgressivePlanCompileReceipt,
    ProgressivePlanFoundation,
    ProgressivePlannerCheckpoint,
    ProgressivePlanOutline,
    ProgressiveOutputIntent,
    ProgressiveProductRef,
    ProgressivePlanSkeleton,
    ProgressiveSkeletonStep,
    ProgressiveStepMaterialization,
)
from .scientific_action_catalog import scientific_action_for_id


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_REQUIRED_RUNTIME_DEPENDENCIES = (
    "cohort_file_sha256",
    "llm_signature",
    "prompt_version",
)


StepSchemaAuthorityResolver = Callable[
    [ProgressiveOutlineStep, str, Sequence[tuple[str, str]]],
    str | None,
]


@dataclass(frozen=True)
class ProgressivePrefixState:
    """Current host-compiled prefix and its prompt-safe summary."""

    materializations: tuple[ProgressiveStepMaterialization, ...] = ()
    steps: tuple[ProgressiveSkeletonStep, ...] = ()
    plan: AnalysisPlan | None = None
    receipt: ProgressivePlanCompileReceipt | None = None
    available_product_refs: tuple[tuple[str, str], ...] = ()
    prompt_summary: tuple[Mapping[str, Any], ...] = ()


@dataclass(frozen=True)
class ProgressiveCheckpointAuthorities:
    """Full-request and semantic replay identities for one Planner run."""

    request_authority_sha256: str
    resume_dependency_authority_sha256: str


def _resume_context_projection(context: ResearchContext) -> dict[str, Any]:
    """Remove run-location entropy while retaining every scientific field."""

    payload = context.model_dump(mode="json")
    payload.pop("created_at", None)
    payload.pop("cohort_parquet", None)
    cohort = payload.get("cohort")
    if isinstance(cohort, dict):
        provenance = cohort.get("provenance")
        if isinstance(provenance, dict):
            provenance.pop("cohort_path", None)
            provenance.pop("materialized_cohort_provenance_sha256", None)
    return payload


def validate_progressive_resume_runtime_dependencies(
    runtime_dependencies: Mapping[str, Any] | None,
) -> None:
    """Require the minimum runtime identity before a provider can be called."""

    supplied = dict(runtime_dependencies or {})
    missing = [
        key
        for key in _REQUIRED_RUNTIME_DEPENDENCIES
        if not str(supplied.get(key) or "").strip()
    ]
    if missing:
        raise ProgressivePlanCompileError(
            "progressive_resume_runtime_dependency_missing",
            "development resume requires exact cohort, provider, and prompt "
            "runtime authority",
            path="resume_dependency_context",
            findings=[{"missing_keys": missing}],
        )
    cohort_digest = str(supplied["cohort_file_sha256"]).strip()
    if not _SHA256_RE.fullmatch(cohort_digest):
        raise ProgressivePlanCompileError(
            "progressive_resume_cohort_authority_invalid",
            "development resume cohort_file_sha256 is not a lowercase SHA-256",
            path="resume_dependency_context.cohort_file_sha256",
        )


def progressive_resume_dependency_sha256(
    *,
    context: ResearchContext,
    article_context: ResearchContext,
    authority: Mapping[str, Any],
) -> str:
    """Digest semantic inputs plus caller-owned runtime dependencies."""

    return canonical_sha256(
        {
            "research_context": _resume_context_projection(context),
            "article_context": _resume_context_projection(article_context),
            **dict(authority),
        }
    )


def build_progressive_checkpoint_authorities(
    *,
    context: ResearchContext,
    article_context: ResearchContext,
    scientific_authority: Mapping[str, Any],
    runtime_dependency_authority: Mapping[str, Any] | None,
) -> ProgressiveCheckpointAuthorities:
    """Build checkpoint identities without duplicating authority projections."""

    authority = dict(scientific_authority)
    return ProgressiveCheckpointAuthorities(
        request_authority_sha256=canonical_sha256(
            {
                "research_context": context.model_dump(mode="json"),
                "article_context": article_context.model_dump(mode="json"),
                **authority,
            }
        ),
        resume_dependency_authority_sha256=(
            progressive_resume_dependency_sha256(
                context=context,
                article_context=article_context,
                authority={
                    **authority,
                    "runtime_dependency_authority": dict(
                        runtime_dependency_authority or {}
                    ),
                },
            )
        ),
    )


def restore_progressive_resume_prompt_metrics(
    *,
    checkpoint: ProgressivePlannerCheckpoint,
    current_prompt_metrics: Mapping[str, Any],
    expected_dependency_sha256: str,
) -> dict[str, Any]:
    """Validate outline/runtime authority and seed current-run counters."""

    source_metrics = copy.deepcopy(checkpoint.prompt_metrics)
    source_dependency = str(
        source_metrics.get("resume_dependency_authority_sha256") or ""
    )
    if not _SHA256_RE.fullmatch(source_dependency):
        raise ProgressivePlanCompileError(
            "progressive_resume_dependency_authority_missing",
            "development checkpoint predates dependency-bound replay",
            path=(
                "resume_checkpoint.prompt_metrics."
                "resume_dependency_authority_sha256"
            ),
        )
    if source_dependency != expected_dependency_sha256:
        raise ProgressivePlanCompileError(
            "progressive_resume_dependency_authority_mismatch",
            "development checkpoint does not bind the current research, data, "
            "provider, prompt, literature, know-how, and cohort authority",
            path=(
                "resume_checkpoint.prompt_metrics."
                "resume_dependency_authority_sha256"
            ),
        )
    authority_keys = (
        "structured_output_authority_sha256",
        "selected_variable_roster_sha256",
        "candidate_analysis_types",
        "selected_scientific_action_ids",
        "planner_strategy",
        "foundation_cohort_owner",
        "required_primary_cohort_selection_mode",
    )
    mismatched_keys = [
        key
        for key in authority_keys
        if source_metrics.get(key) != current_prompt_metrics.get(key)
    ]
    if mismatched_keys:
        raise ProgressivePlanCompileError(
            "progressive_resume_outline_authority_mismatch",
            "development checkpoint outline authority differs from the current "
            "strict request",
            path="resume_checkpoint.prompt_metrics",
            findings=[{"mismatched_keys": mismatched_keys}],
        )
    source_metrics.update(
        {
            "resume_source_checkpoint_sha256": checkpoint.checkpoint_sha256,
            "resume_source_sequence": int(checkpoint.sequence),
            "resume_reused_materialization_count": len(
                checkpoint.materializations
            ),
            "current_run_compile_revision_count": 0,
            "current_run_step_materialization_count": 0,
            "current_run_step_materialization_attempt_payload_bytes": [],
            "current_run_step_materialization_attempt_schema_sha256": [],
        }
    )
    return source_metrics


def restore_progressive_resume_foundation(
    *,
    checkpoint: ProgressivePlannerCheckpoint,
    prompt_metrics: Mapping[str, Any],
    request_payload_bytes: int,
    schema_bytes: int,
    schema_authority_sha256: str | None,
) -> ProgressiveFoundationMaterialization | None:
    """Return a checkpoint foundation only after current request equivalence."""

    if checkpoint.foundation is None:
        return None
    mismatched_keys = [
        key
        for key, current_value in (
            ("foundation_request_payload_bytes", request_payload_bytes),
            ("foundation_schema_bytes", schema_bytes),
            (
                "foundation_structured_output_authority_sha256",
                schema_authority_sha256,
            ),
        )
        if prompt_metrics.get(key) != current_value
    ]
    if mismatched_keys:
        raise ProgressivePlanCompileError(
            "progressive_resume_foundation_authority_mismatch",
            "development checkpoint foundation differs from the current strict "
            "request",
            path="resume_checkpoint.foundation",
            findings=[{"mismatched_keys": mismatched_keys}],
        )
    return checkpoint.foundation


def assemble_progressive_skeleton(
    *,
    outline: ProgressivePlanOutline,
    foundation: ProgressivePlanFoundation,
    steps: Sequence[ProgressiveSkeletonStep],
) -> ProgressivePlanSkeleton:
    """Assemble the compiler input without giving the model wiring authority."""

    return ProgressivePlanSkeleton(
        analysis_type=outline.analysis_type,
        design_selection=outline.design_selection,
        cohort=foundation.cohort,
        display_labels=list(foundation.display_labels),
        robustness_intents=list(foundation.robustness_intents),
        know_how_decisions=list(foundation.know_how_decisions),
        steps=list(steps),
        rationale=outline.rationale,
    )


def validate_progressive_materialization_coordinate(
    materialization: ProgressiveStepMaterialization,
    *,
    outline_step: ProgressiveOutlineStep,
    outline_step_sha256: str,
    step_index: int,
    require_literature_roster: bool = True,
) -> None:
    """Keep outline-owned coordinates immutable during materialization."""

    step = materialization.step
    expected = {
        "step_id": outline_step.step_id,
        "planned_analysis_role": outline_step.planned_analysis_role,
        "module_id": outline_step.module_id,
        "objective": outline_step.objective,
        "depends_on": list(outline_step.depends_on),
        "scientific_action_id": outline_step.scientific_action_id,
    }
    actual = {
        "step_id": step.step_id,
        "planned_analysis_role": step.planned_analysis_role,
        "module_id": step.module_id,
        "objective": step.objective,
        "depends_on": list(step.depends_on),
        "scientific_action_id": step.scientific_action_id,
    }
    if materialization.outline_step_sha256 != outline_step_sha256:
        raise ProgressivePlanCompileError(
            "progressive_step_outline_digest_mismatch",
            "current-step materialization did not bind the host outline digest",
            step_id=outline_step.step_id,
            step_index=step_index,
            path="outline_step_sha256",
        )
    if actual != expected:
        changed = sorted(
            key for key, value in expected.items() if actual.get(key) != value
        )
        raise ProgressivePlanCompileError(
            "progressive_step_materialization_mismatch",
            "current-step materialization changed outline-owned fields: "
            + ", ".join(changed),
            step_id=outline_step.step_id,
            step_index=step_index,
            path=changed[0] if changed else "step",
        )
    expected_citations = tuple(outline_step.literature_citation_keys)
    actual_citations = tuple(
        binding.citation_key for binding in step.literature_bindings
    )
    if require_literature_roster and (
        len(actual_citations) != len(set(actual_citations))
        or set(actual_citations) != set(expected_citations)
    ):
        raise ProgressivePlanCompileError(
            "progressive_step_literature_roster_mismatch",
            "current-step materialization must bind every outline-sealed "
            "literature citation exactly once",
            step_id=outline_step.step_id,
            step_index=step_index,
            path="literature_bindings",
            findings=[
                {
                    "expected_citation_keys": list(expected_citations),
                    "observed_citation_keys": list(actual_citations),
                }
            ],
        )
    if materialization.foundation is not None:
        raise ProgressivePlanCompileError(
            "progressive_step_foundation_coordinate_mismatch",
            "current-step materializations must not repeat the separately sealed "
            "plan foundation",
            step_id=outline_step.step_id,
            step_index=step_index,
            path="foundation",
        )


def compile_progressive_prefix(
    state: ProgressivePrefixState,
    materialization: ProgressiveStepMaterialization,
    *,
    outline: ProgressivePlanOutline,
    foundation: ProgressivePlanFoundation,
    context: ResearchContext,
    allowed_literature_citation_keys: Sequence[str],
    allowed_know_how_decisions: Mapping[str, Mapping[str, Any]] | None,
    reporting_method_source_keys: Sequence[str],
) -> ProgressivePrefixState:
    """Compile one candidate step and return a new immutable prefix state."""

    candidate_steps = (*state.steps, materialization.step)
    try:
        skeleton = assemble_progressive_skeleton(
            outline=outline,
            foundation=foundation,
            steps=candidate_steps,
        )
    except ValidationError as exc:
        findings: list[dict[str, str]] = []
        for issue in exc.errors(
            include_context=False,
            include_input=False,
            include_url=False,
        )[:20]:
            location = ".".join(str(part) for part in issue.get("loc", ()))
            findings.append(
                {
                    "path": location or "steps",
                    "type": str(issue.get("type") or "value_error")[:80],
                    "message": str(issue.get("msg") or "invalid prefix")[:500],
                }
            )
        raise ProgressivePlanCompileError(
            "progressive_prefix_contract_invalid",
            "current-step materialization violates the typed prefix contract",
            step_id=materialization.step.step_id,
            step_index=len(state.steps),
            path=(findings[0]["path"] if findings else "steps"),
            findings=findings,
        ) from exc
    plan, receipt = compile_progressive_plan(
        skeleton=skeleton,
        context=context,
        allowed_literature_citation_keys=allowed_literature_citation_keys,
        allowed_know_how_decisions=allowed_know_how_decisions,
        host_reporting_method_source_keys=reporting_method_source_keys,
    )
    try:
        validate_plan_against_adjustment_authority(plan=plan, context=context)
    except AdjustmentAuthorityError as exc:
        authority = AdjustmentSetAuthority.from_context(context)
        raise ProgressivePlanCompileError(
            "progressive_adjustment_authority_mismatch",
            str(exc),
            step_id=materialization.step.step_id,
            step_index=len(state.steps),
            path="model_terms",
            findings=[
                {
                    "selection": authority.selection,
                    "required_covariates": list(authority.operational_covariates),
                    "repair_scope": "current_model_step_only",
                }
            ],
        ) from exc
    product_refs = tuple(
        (step.step_id, product_id)
        for step in plan.steps
        for product_id in step.expected_outputs
    )
    summary = tuple(
        {
            "step_id": step.step_id,
            "module_id": candidate_steps[index].module_id,
            "scientific_action_id": candidate_steps[index].scientific_action_id,
            "skeleton_sha256": receipt.compiled_steps[index].skeleton_sha256,
            "compiled_step_sha256": (
                receipt.compiled_steps[index].compiled_step_sha256
            ),
            "expected_outputs": list(step.expected_outputs),
        }
        for index, step in enumerate(plan.steps)
    )
    return ProgressivePrefixState(
        materializations=(*state.materializations, materialization),
        steps=candidate_steps,
        plan=plan,
        receipt=receipt,
        available_product_refs=product_refs,
        prompt_summary=summary,
    )


def restore_progressive_resume_prefix(
    *,
    checkpoint: ProgressivePlannerCheckpoint,
    outline: ProgressivePlanOutline,
    foundation: ProgressivePlanFoundation,
    context: ResearchContext,
    step_schema_authority: StepSchemaAuthorityResolver,
    allowed_literature_citation_keys: Sequence[str],
    allowed_know_how_decisions: Mapping[str, Mapping[str, Any]] | None,
    reporting_method_source_keys: Sequence[str],
    strict_step_schema_enabled: bool = False,
) -> ProgressivePrefixState:
    """Revalidate strict schemas and recompile every reused prefix step."""

    materializations = checkpoint.materializations
    if len(materializations) > len(outline.steps):
        raise ProgressivePlanCompileError(
            "progressive_resume_prefix_too_long",
            "development checkpoint contains more steps than its outline",
            path="resume_checkpoint.materializations",
        )
    stored_schema_authorities = checkpoint.prompt_metrics.get(
        "step_materialization_schema_sha256",
        [],
    )
    if not isinstance(stored_schema_authorities, list):
        raise ProgressivePlanCompileError(
            "progressive_resume_step_authority_invalid",
            "development checkpoint step schema authority is not a list",
            path=(
                "resume_checkpoint.prompt_metrics."
                "step_materialization_schema_sha256"
            ),
        )
    state = ProgressivePrefixState()
    stored_available_product_refs: tuple[tuple[str, str], ...] = ()
    runtime_contract_migration_started = False
    migrated_receipt_ids = {
        str(value)
        for value in checkpoint.prompt_metrics.get(
            "runtime_contract_migrated_step_ids", []
        )
        if str(value).strip()
    }
    migrated_positions = {
        index
        for index, item in enumerate(outline.steps)
        if item.step_id in migrated_receipt_ids
    }
    for step_index, materialization in enumerate(materializations):
        outline_step = outline.steps[step_index]
        outline_step_sha256 = canonical_sha256(
            outline_step.model_dump(mode="json")
        )
        current_schema_authority = step_schema_authority(
            outline_step,
            outline_step_sha256,
            stored_available_product_refs,
        )
        stored_schema_authority = (
            stored_schema_authorities[step_index]
            if step_index < len(stored_schema_authorities)
            else object()
        )
        host_materialized_replay = bool(
            strict_step_schema_enabled
            and stored_schema_authority is None
            and current_schema_authority is None
        )
        validate_progressive_materialization_coordinate(
            materialization,
            outline_step=outline_step,
            outline_step_sha256=outline_step_sha256,
            step_index=step_index,
            require_literature_roster=not host_materialized_replay,
        )
        schema_authority_matches = bool(
            step_index < len(stored_schema_authorities)
            and stored_schema_authority == current_schema_authority
        )
        migration_receipt_covers_drift = bool(
            migrated_positions and step_index > min(migrated_positions)
        )
        if not schema_authority_matches and not migration_receipt_covers_drift:
            raise ProgressivePlanCompileError(
                "progressive_resume_step_schema_authority_mismatch",
                "development checkpoint step differs from the current strict "
                "request",
                step_id=outline_step.step_id,
                step_index=step_index,
                path="resume_checkpoint.step_schema_authority",
            )
        stored_materialization = materialization
        materialization = _migrate_installed_runtime_contract(
            materialization,
            analysis_type=outline.analysis_type,
            available_product_refs=state.available_product_refs,
        )
        try:
            state = compile_progressive_prefix(
                state,
                materialization,
                outline=outline,
                foundation=foundation,
                context=context,
                allowed_literature_citation_keys=(
                    allowed_literature_citation_keys
                ),
                allowed_know_how_decisions=allowed_know_how_decisions,
                reporting_method_source_keys=reporting_method_source_keys,
            )
        except ProgressivePlanCompileError as exc:
            raise ProgressivePlanCompileError(
                "progressive_resume_prefix_invalid",
                f"current host rejected resumed prefix: {exc.reason_code}",
                step_id=exc.step_id or outline_step.step_id,
                step_index=(
                    exc.step_index
                    if exc.step_index is not None
                    else step_index
                ),
                path=exc.path or "resume_checkpoint.materializations",
                findings=[exc.details],
            ) from exc
        migrated = materialization != stored_materialization
        runtime_contract_migration_started = (
            runtime_contract_migration_started or migrated
        )
        if runtime_contract_migration_started:
            stored_available_product_refs = (
                *stored_available_product_refs,
                *(
                    (stored_materialization.step.step_id, output.product_id)
                    for output in stored_materialization.step.outputs
                ),
            )
        else:
            stored_available_product_refs = state.available_product_refs
    return state


def _migrate_installed_runtime_contract(
    materialization: ProgressiveStepMaterialization,
    *,
    analysis_type: str,
    available_product_refs: Sequence[tuple[str, str]],
) -> ProgressiveStepMaterialization:
    """Project a reused model step onto a newly installed owner contract.

    A development checkpoint can predate a deterministic adapter.  The signed
    model choices remain authoritative, but output product names and uniquely
    implied producer edges are implementation coordinates already fixed by the
    new owner.  Migrate only those coordinates; ambiguous or non-direct inputs
    remain untouched and fail through the ordinary compiler.
    """

    step = materialization.step
    if step.scientific_action_id is None:
        return materialization
    action = scientific_action_for_id(
        analysis_type=analysis_type,
        action_id=step.scientific_action_id,
    )
    contract = action.runtime_contract
    if contract is None:
        return materialization
    references: list[ProgressiveProductRef] = []
    for product_id in contract.required_product_inputs:
        owners = [
            producer
            for producer, available_product in available_product_refs
            if available_product == product_id
        ]
        if len(owners) != 1 or owners[0] not in step.depends_on:
            return materialization
        references.append(
            ProgressiveProductRef(
                producer_step_id=owners[0],
                product_id=product_id,
            )
        )
    outputs = [
        ProgressiveOutputIntent(product_id=product_id, semantic_role=semantic_role)
        for product_id, semantic_role in contract.outputs
    ]
    migrated_step = step.model_copy(
        update={"product_inputs": references, "outputs": outputs}
    )
    if migrated_step == step:
        return materialization
    return materialization.model_copy(update={"step": migrated_step})


__all__ = [
    "ProgressiveCheckpointAuthorities",
    "ProgressivePrefixState",
    "assemble_progressive_skeleton",
    "build_progressive_checkpoint_authorities",
    "compile_progressive_prefix",
    "progressive_resume_dependency_sha256",
    "restore_progressive_resume_foundation",
    "restore_progressive_resume_prefix",
    "restore_progressive_resume_prompt_metrics",
    "validate_progressive_materialization_coordinate",
    "validate_progressive_resume_runtime_dependencies",
]
