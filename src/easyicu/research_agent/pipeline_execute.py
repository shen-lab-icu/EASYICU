"""Execute phase of the research-agent pipeline.

Implements the probe → per-step analysis loop with optional replanning
and final figure visual-QA. Extracted from
:class:`ResearchAgentPipeline._run_execute_phase` (which is now a thin
delegate) so:

* the 1500-line execute loop reads as its own module;
* the planning / writing phases in :mod:`pipeline` don't have to scroll
  past it;
* a future graph-style runner (LangGraph or similar) has a single
  free-function entry point to wrap, rather than a method buried in a
  god-object.

The function is intentionally a free function, not a class. All state
that the execute phase mutates (``runtime_state``, ``per_step_records``,
``probe_summary``, ``findings``, ``plan``) is local to one call; nothing
needs to survive across calls. The pipeline instance is passed in only
as a *read-only collaborator* — execute-phase reads several ``_enable_*``
flags and calls ``pipeline._build_runner(...)``, but never mutates
pipeline state. The audit on 2026-05-15 confirmed zero ``self.* = ...``
writes inside the original method body.
"""

from __future__ import annotations

import ast
import csv
import hashlib
import importlib
import inspect
import json
import logging
import math
import os
import re
import shutil
import stat
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
)

from .agents import (
    AnalyzerAgent,
    ClinicalSemanticsAgent,
    CoderAgent,
    CriticAgent,
    DataExtractionAgent,
    ReplannerAgent,
    RuntimeSupervisor,
    StatisticalAnalysisAgent,
    VisualizationAgent,
)
from .article_contract import (
    article_contract_audit_payload,
    summarize_article_contract_coverage,
    validate_run_against_article_contract,
)
from .audits.validators import (
    ClinicalConstraintValidator,
    ConceptUsageAuditor,
    CrossStepCohortLockValidator,
    CrossStepRegisteredOutputValidator,
    CrossStepReconciliationTraceValidator,
    CrossStepSourceStatusValidator,
    FigureContractQualityValidator,
    FigureSourceDataValidator,
    LLMConceptAuditor,
    PrimaryModelContractValidator,
    StatisticalGuard,
    StatisticalValidator,
    StepSummaryFractionValidator,
    _reclassify_llm_concept_findings,
    _verified_authoritative_exposure_flow,
)
from .audits.patterns import AnalysisPatternAuditor
from .audits.step_summary_integrity import StepSummaryIntegrityValidator
from .code_repair import (
    _deterministic_runner_repair,
    _deterministic_summary_repair,
    deterministic_contract_repair,
    deterministic_concept_audit_repair,
)
from .code_hygiene import reorder_forward_references
from .code_preflight import audit_mechanical_code_contracts
from .concept_audit_cache import LLMConceptAuditCache
from .cohort_repair import extract_cohort_definition_from_prose
from .cohort_schema import (
    CohortDefinition,
    assert_cohort_definition_locked,
    materialize_locked_analysis_cohort,
    write_locked_cohort_definition,
)
from .contracts import ValidationFinding, _ExecutePhaseResult, _PlanPhaseResult
from .deterministic_descriptive import absolute_risk_context_code
from .deterministic_missingness import missingness_measurement_audit_code
from .deterministic_robustness import (
    replay_locked_memberships,
    robustness_sensitivity_preflight_code,
)
from .declared_product_contract import (
    authorize_declared_figure_product_slots,
    read_digest_bound_artifact_snapshot,
    typed_product_binding_contract,
    typed_product as _canonical_typed_product,
)
from .estimators import fit_robustness_rows_from_records
from .evidence import sha256_of_bytes, sha256_of_file
from .llm import MockLLMClient
from .method_compatibility import (
    detect_forbidden_pattern_usage,
    format_violation_message,
)
from .ordered_stratified_contract import ordered_stratified_numeric_findings
from .pipeline import (
    _build_probe_summary,
    _clear_output_dir,
    _distribution_availability_figure_step_matches_parent,
    _resolve_upstream_manifest_step,
    _sealed_renderer_figure_step_matches_parent,
    _sealed_renderer_parent_digest_seal,
    deterministic_figure_family_supported_for_upstream,
    deterministic_figure_repair_id_for_upstream,
    _has_figure_exports,
    _promote_prior_publication_bundle,
    _promote_sibling_figure_exports,
    _render_publication_bundle_from_prior_outputs_for_step,
    _semantic_aliases_for,
)
from .publication_figures import make_figure_contract
from .repair_reasons import typed_repair_ticket
from .plan_utils import (
    _augment_measurement_companion_inputs,
    _augment_report_typed_product_inputs,
    _cap_plan_preserving_figure_steps,
    _clustering_contract_applies,
    _cohort_definition_contract_findings,
    _cohort_definition_is_empty,
    _cohort_definition_prose,
    _normalised_expected_output_names,
    _normalised_structured_output_names,
    _output_declares_figure,
    _parent_step_id_for_figure_step,
    _plan_expects_analysis_cohort,
    _preserve_figure_steps_after_replan,
    _preserve_primary_estimand_step_after_replan,
    _primary_exposure_contract_findings,
    _primary_exposure_measurement_filter_findings,
    _primary_exposure_overadjustment_findings,
    _primary_model_leakage_findings,
    _step_contract_findings,
    _step_contract_repair_guidance,
    _step_expects_figure,
    _typed_plan_dag_findings,
)
from .pipeline_resume import (
    QuarantinedConceptDraft,
    ResumeController,
    clear_quarantined_concept_draft,
    store_quarantined_concept_draft,
    upsert_step_record,
)
from .schema import AnalysisPlan, AnalysisStep, EvidenceRef, ResearchContext
from .robustness_execution_contract import (
    ROBUSTNESS_COHORT_MEMBERSHIP_ALIASES,
    ROBUSTNESS_EXECUTION_CONTRACT_GUIDANCE,
    _executed_robustness_result_issues,
)
from .robustness_panel import (
    RobustnessSpec,
    assert_robustness_specs_locked,
    build_robustness_panel_from_records,
    robustness_specs_for_execution,
    robustness_specs_sha,
    write_robustness_panel,
)
from .trajectory_bundle import trajectory_bundle_findings
from .trajectory_plan_contract import (
    augment_trajectory_plan_products,
    trajectory_plan_contract_applies,
    trajectory_plan_dag_findings,
)
from .trajectory_stability_executor import (
    trajectory_stability_executor_code,
    trajectory_stability_executor_owns_step,
)
from .trajectory_resume_schema import materialize_legacy_trajectory_replay_schemas
from .repair_registry import (
    InvariantStatus,
    RepairLedger,
    RepairObservedState,
    automatic_repair_allowed,
    is_sealed_renderer_repair,
    repair_metadata_for,
)
from .provider_budget import (
    ProviderCallBudgetError,
    ProviderCallBudgetReceiptError,
    StepProviderCallBudget,
    complete_with_provider_budget,
    load_provider_call_budget_receipt,
    provider_call_budget_receipt_path,
)
from .run_input_capsule import (
    RunInputIdentityError,
    _HOST_COHORT_MATERIALIZER_AUTHORITY_FIELD,
    _HOST_COHORT_MATERIALIZER_AUTHORITY_KIND,
    _HOST_COHORT_MATERIALIZER_GENERATION_MODE,
    _HOST_PROBE_AUTHORITIES,
    _HOST_PROBE_AUTHORITY_KIND,
    _host_cohort_materializer_authority_error,
    _host_probe_authority_error,
    build_environment_identity,
    canonical_sha256,
    engine_code_sha256,
)
from .runtime_artifacts import (
    current_step_records,
    current_successful_step_records,
    verified_run_evidence_path,
    write_run_checkpoint,
)
from .scalar_utils import _expected_numeric_annotations_for_step
from .side_findings import SideFinding
from .skills import ClinicalSkill
from .summary_repair import salvage_step_summary
from .viability import (
    CohortViability,
    assess_cohort_viability,
    step_requires_model_performance,
    step_summary_block_signal,
)
from .visual_qa import VLMVisualQAAdapter, VisualQAAuditor

logger = logging.getLogger(__name__)


_STANDARD_EXECUTOR_INTERNAL_PENDING_ARTIFACTS = frozenset(
    {".cluster_stability_assignments.pending.csv"}
)
_FIGURE_CONTRACT_SOURCE_DATA_SCHEMA_REPAIR_ID = "figure_contract_source_data_schema_v1"
_DETERMINISTIC_GATE_SCHEMA_VERSION = "easyicu.deterministic_step_gate/1"
_COHORT_TRANSLATION_PROVIDER_CATEGORY = "cohort_definition_translation"
_HOST_COHORT_TRANSLATION_BUDGET_STEP_ID = "host_cohort_definition_translation"


def _declares_host_cohort_only_product(step: AnalysisStep) -> bool:
    declared = {
        str(value or "").strip().casefold()
        for value in (step.expected_outputs or [])
        if str(value or "").strip()
    }
    return declared == {"table:analysis_cohort"}


def _cohort_translation_budget_owner_step_id(plan: AnalysisPlan) -> str:
    """Return one stable budget owner without making a cohort decision.

    A single cohort-only product step is the natural owner because successful
    host materialisation completes exactly that planned step.  Ambiguous or
    mixed-product plans use a host pseudo-step instead of charging an arbitrary
    analysis step.  This helper only assigns provider-call accounting; the
    Planner's prose remains the sole source of inclusion/exclusion criteria.
    """

    cohort_only_step_ids = [
        str(step.step_id)
        for step in plan.steps
        if _declares_host_cohort_only_product(step)
    ]
    if len(cohort_only_step_ids) == 1:
        return cohort_only_step_ids[0]
    return _HOST_COHORT_TRANSLATION_BUDGET_STEP_ID


def _extract_cohort_definition_with_provider_budget(
    *,
    run_dir: Path,
    budget_owner_step_id: str,
    configured_limit: int,
    cohort_prose: str,
    universe_columns: Sequence[str],
    llm: Any,
    name: str,
) -> Tuple[Optional[CohortDefinition], Dict[str, Any]]:
    """Run cohort-prose translation under a crash-safe provider receipt.

    This call happens before ``_execute_one_step`` creates its ordinary
    per-step budget.  Reusing the same receipt namespace makes a cohort-only
    planned step inherit this paid call if translation fails and the Coder must
    later execute it.  Transport retries are charged by the active provider
    scope just like coder/auditor retries.
    """

    receipt_path = provider_call_budget_receipt_path(
        run_dir,
        step_id=budget_owner_step_id,
    )
    effective_limit = max(0, int(configured_limit))
    consumed_categories: Tuple[str, ...] = ()
    if receipt_path.exists():
        receipt_limit, consumed_categories = load_provider_call_budget_receipt(
            receipt_path,
            step_id=budget_owner_step_id,
        )
        effective_limit = min(effective_limit, receipt_limit)
    budget = StepProviderCallBudget(
        effective_limit,
        step_id=budget_owner_step_id,
        consumed_categories=consumed_categories,
        receipt_path=receipt_path,
    )
    definition = complete_with_provider_budget(
        budget=budget,
        category=_COHORT_TRANSLATION_PROVIDER_CATEGORY,
        call=lambda: extract_cohort_definition_from_prose(
            cohort_prose=cohort_prose,
            universe_columns=universe_columns,
            llm=llm,
            name=name,
        ),
    )
    snapshot = budget.snapshot()
    return definition, {
        "budget_owner_step_id": budget_owner_step_id,
        "step_provider_call_budget_scope": _COHORT_TRANSLATION_PROVIDER_CATEGORY,
        "step_provider_call_budget": snapshot["limit"],
        "step_provider_call_attempts": snapshot["used"],
        "step_provider_call_remaining": snapshot["remaining"],
        "step_provider_call_budget_exhausted": snapshot["exhausted"],
        "step_provider_call_categories": snapshot["categories"],
        "step_provider_call_receipt_version": 1,
        "step_provider_call_receipt": str(receipt_path.relative_to(run_dir)),
    }


def _deterministic_gate_stamp() -> Dict[str, str]:
    """Return the current host-owned deterministic gate identity.

    The full engine digest is intentionally included.  This safely
    over-invalidates when unrelated engine code changes, but never reuses a
    success reviewed under an older deterministic implementation.
    """

    engine_digest = engine_code_sha256()
    fingerprint = canonical_sha256(
        {
            "schema_version": _DETERMINISTIC_GATE_SCHEMA_VERSION,
            "engine_code_sha256": engine_digest,
        }
    )
    return {
        "deterministic_gate_schema_version": _DETERMINISTIC_GATE_SCHEMA_VERSION,
        "deterministic_gate_engine_code_sha256": engine_digest,
        "deterministic_gate_fingerprint": fingerprint,
    }


def _deterministic_code_gate_findings(
    *,
    context: ResearchContext,
    step: AnalysisStep,
    script_text: str,
    usage_auditor: Optional[ConceptUsageAuditor] = None,
    pattern_auditor: Optional[AnalysisPatternAuditor] = None,
) -> List[ValidationFinding]:
    """Run the shared deterministic pre-execution code gate.

    Fresh execution may add the optional LLM concept audit after this prefix.
    Resume drift replay deliberately stops here: it rechecks deterministic
    semantics and mechanics without invoking an LLM or mutating the script.
    """

    usage = usage_auditor if usage_auditor is not None else ConceptUsageAuditor()
    patterns = (
        pattern_auditor if pattern_auditor is not None else AnalysisPatternAuditor()
    )
    findings = usage.audit(
        context=context,
        script_text=script_text,
        step=step,
    )
    findings.extend(
        patterns.audit(
            context=context,
            script_text=script_text,
            step=step,
        )
    )
    compatibility_violations = detect_forbidden_pattern_usage(
        script_text,
        context,
        step,
    )
    if compatibility_violations:
        findings.append(
            ValidationFinding(
                validator="method_compatibility",
                severity="error",
                message=format_violation_message(compatibility_violations),
                detail={
                    "step_id": step.step_id,
                    "violations": compatibility_violations,
                },
            )
        )
    findings.extend(audit_mechanical_code_contracts(script_text, step))
    requires_primary_exposure_artifact = any(
        str(value).strip().casefold() == "artifact:primary_exposure_definition"
        for value in step.inputs or []
    )
    if (
        requires_primary_exposure_artifact
        and str(context.primary_exposure or "").strip()
        and not _verified_authoritative_exposure_flow(
            script_text,
            primary_exposure=str(context.primary_exposure),
        )
    ):
        findings.append(
            ValidationFinding(
                validator="typed_input_authority_flow",
                severity="error",
                message=(
                    f"Step {step.step_id} does not prove that the exact host-bound "
                    "primary exposure reaches its result-bearing model or figure "
                    "after finite/domain validation."
                ),
                detail={
                    "step_id": step.step_id,
                    "input_key": "artifact:primary_exposure_definition",
                    "issue": "typed_primary_exposure_not_consumed",
                },
            )
        )
    return findings


def _merge_monotonic_concept_constraints(
    existing: Sequence[ValidationFinding],
    candidates: Sequence[ValidationFinding],
) -> List[ValidationFinding]:
    """Merge binding concept errors without losing earlier repair constraints.

    A later repair may introduce a different error after an earlier error has
    already been removed from the current script.  Quarantine checkpoints must
    retain both constraints so a resumed repair cannot regress the earlier fix.
    """

    merged = list(existing)
    seen = {(finding.validator, finding.message) for finding in merged}
    for finding in candidates:
        key = (finding.validator, finding.message)
        if finding.severity != "error" or key in seen:
            continue
        merged.append(finding)
        seen.add(key)
    return merged


def _persisted_monotonic_concept_constraints(
    record: Mapping[str, Any] | None,
) -> List[ValidationFinding]:
    """Load binding constraints from the latest unfinished step record."""

    if not isinstance(record, Mapping) or str(record.get("status") or "") == "ok":
        return []
    raw_constraints = record.get("monotonic_concept_constraints")
    if not isinstance(raw_constraints, list):
        return []
    parsed: List[ValidationFinding] = []
    for payload in raw_constraints:
        if not isinstance(payload, Mapping):
            continue
        try:
            finding = ValidationFinding.model_validate(payload)
        except (TypeError, ValueError):
            continue
        parsed = _merge_monotonic_concept_constraints(parsed, [finding])
    return parsed


def _monotonic_step_llm_repair_history(
    records: Sequence[Mapping[str, Any]],
    *,
    limit: int,
) -> tuple[int, List[str], bool]:
    """Recover the largest durable logical-repair counter for one step.

    Step records are append-only attempts.  The latest attempt may terminate
    before copying the logical counter (for example, on a damaged provider
    receipt), so latest-record-only recovery can incorrectly buy a fresh
    repair budget.  A malformed explicit counter is treated conservatively as
    exhausted instead of being ignored.
    """

    attempts = 0
    classes: List[str] = []
    invalid_snapshot = False
    for record in records:
        if "step_llm_repair_attempts" in record:
            raw_attempts = record.get("step_llm_repair_attempts")
            if (
                isinstance(raw_attempts, bool)
                or not isinstance(raw_attempts, int)
                or raw_attempts < 0
            ):
                invalid_snapshot = True
            else:
                attempts = max(attempts, raw_attempts)
        raw_classes = record.get("step_llm_repair_classes")
        if not isinstance(raw_classes, list):
            continue
        normalized = [str(item).strip() for item in raw_classes]
        if any(not item for item in normalized):
            invalid_snapshot = True
            continue
        if len(normalized) > len(classes):
            classes = normalized
    if invalid_snapshot:
        attempts = max(attempts, max(0, int(limit)))
    return attempts, classes, invalid_snapshot


def _remove_standard_executor_pending_artifacts(out_dir: Path) -> None:
    """Remove private partial files before failed-run evidence discovery."""

    for name in _STANDARD_EXECUTOR_INTERNAL_PENDING_ARTIFACTS:
        (out_dir / name).unlink(missing_ok=True)


def _is_standard_executor_internal_artifact(path: Path) -> bool:
    """Return whether *path* is a private, never-evidence work product."""

    return path.name in _STANDARD_EXECUTOR_INTERNAL_PENDING_ARTIFACTS


class _EvidenceLineageResolutionError(RuntimeError):
    """A typed plan input could not be bound to current verified evidence."""

    def __init__(self, failures: Sequence[Mapping[str, Any]]) -> None:
        self.failures = [dict(failure) for failure in failures]
        super().__init__(
            "; ".join(
                f"{failure.get('input')}: {failure.get('reason')}"
                for failure in self.failures
            )
        )


_TYPED_INPUT_KINDS = frozenset(
    {
        "artifact",
        "dataset",
        "figure",
        "log",
        "manifest",
        "model",
        "statistic",
        "table",
    }
)

# Typed products describe the logical contract while EvidenceStore kinds describe
# the physical evidence class.  Most pairs are exact.  The three adapters below
# are deliberate and closed: tabular datasets are stored as tables, serialized
# models/manifests as logs, and generic artifacts may be either a table or a log.
# Code and figures are never compatible with a generic scientific artifact.
_TYPED_INPUT_EVIDENCE_KINDS: Mapping[str, frozenset[str]] = {
    "artifact": frozenset({"log", "table"}),
    "dataset": frozenset({"table"}),
    "figure": frozenset({"figure"}),
    "log": frozenset({"log"}),
    "manifest": frozenset({"log"}),
    "model": frozenset({"log"}),
    "statistic": frozenset({"statistic"}),
    "table": frozenset({"table"}),
}


def _evidence_kind_matches_typed_product(
    record: Any,
    typed_product: Tuple[str, str],
) -> bool:
    evidence_kind = str(_evidence_record_field(record, "kind") or "").strip().lower()
    return evidence_kind in _TYPED_INPUT_EVIDENCE_KINDS.get(
        typed_product[0], frozenset()
    )


def _normalise_typed_product_name(value: Any) -> str:
    parsed = _canonical_typed_product(f"artifact:{value}")
    return parsed[1] if parsed is not None else ""


def _typed_input_product(value: Any) -> Optional[Tuple[str, str]]:
    """Return a canonical ``(kind, product)`` for a typed plan dependency."""

    parsed = _canonical_typed_product(value)
    if parsed is None or parsed[0] not in _TYPED_INPUT_KINDS:
        return None
    return parsed


def _typed_artifact_name(value: Any) -> Optional[str]:
    """Backward-compatible artifact-only view of a typed plan dependency."""

    typed_product = _typed_input_product(value)
    if typed_product is None or typed_product[0] != "artifact":
        return None
    return typed_product[1]


def _evidence_record_field(record: Any, name: str) -> Any:
    if isinstance(record, Mapping):
        return record.get(name)
    return getattr(record, name, None)


def _registered_source_name(record: Any, verified_path: Path) -> Optional[str]:
    """Recover the registered source name from ``<evidence_id>__<filename>``."""

    evidence_id = str(_evidence_record_field(record, "evidence_id") or "")
    prefix = f"{evidence_id}__"
    if not evidence_id or not verified_path.name.startswith(prefix):
        return None
    return verified_path.name[len(prefix) :] or None


def _declared_typed_product_paths(
    step_summary: Any,
    *,
    typed_product: Tuple[str, str],
) -> Tuple[bool, List[str]]:
    """Return exact typed file mappings declared by the producer summary."""

    if not isinstance(step_summary, Mapping):
        return False, []
    declared = False
    paths: List[str] = []
    for container_name in ("output_files", "outputs"):
        container = step_summary.get(container_name)
        if isinstance(container, Mapping):
            for typed_key, value in container.items():
                if _typed_input_product(typed_key) != typed_product:
                    continue
                declared = True
                if isinstance(value, str) and value.strip():
                    paths.append(value.strip())
                elif isinstance(value, (list, tuple)):
                    paths.extend(
                        str(item).strip()
                        for item in value
                        if isinstance(item, str) and item.strip()
                    )
        elif isinstance(container, (list, tuple)):
            for item in container:
                if not isinstance(item, Mapping):
                    continue
                kind = item.get("kind") or item.get("product_type")
                name = item.get("name")
                named_product = _typed_input_product(name)
                kind_product = _typed_input_product(f"{kind}:placeholder")
                if named_product is not None:
                    descriptor_product = (
                        named_product
                        if kind_product is not None
                        and named_product[0] == kind_product[0]
                        else None
                    )
                else:
                    descriptor_product = _typed_input_product(f"{kind}:{name}")
                if descriptor_product != typed_product:
                    continue
                declared = True
                value = next(
                    (
                        item.get(key)
                        for key in ("path", "relative_path", "filename")
                        if isinstance(item.get(key), str)
                        and str(item.get(key)).strip()
                    ),
                    None,
                )
                if isinstance(value, str) and value.strip():
                    paths.append(value.strip())
                elif isinstance(value, (list, tuple)):
                    paths.extend(
                        str(path).strip()
                        for path in value
                        if isinstance(path, str) and path.strip()
                    )
    return declared, list(dict.fromkeys(paths))


def _declared_typed_artifact_paths(
    step_summary: Any,
    *,
    artifact_name: str,
) -> Tuple[bool, List[str]]:
    """Backward-compatible wrapper for artifact-specific callers."""

    return _declared_typed_product_paths(
        step_summary,
        typed_product=("artifact", artifact_name),
    )


def _lineage_failure_product_fields(
    typed_product: Tuple[str, str],
) -> Dict[str, str]:
    kind, product_name = typed_product
    fields = {"kind": kind, "product": product_name}
    if kind == "artifact":
        fields["artifact"] = product_name
    return fields


def _step_summary_statistic_values(
    step_summary: Any,
    statistic_name: str,
) -> List[float]:
    """Return finite scalar values bound to one exact statistic name."""

    values: List[float] = []

    def _append(value: Any) -> None:
        if isinstance(value, bool) or isinstance(value, (Mapping, list, tuple)):
            return
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return
        if math.isfinite(numeric):
            values.append(numeric)

    def _walk(value: Any) -> None:
        if isinstance(value, Mapping):
            declared_name = value.get("name") or value.get("statistic")
            if declared_name is not None and (
                _normalise_typed_product_name(declared_name) == statistic_name
            ):
                for result_key in ("value", "estimate", "result"):
                    if result_key in value:
                        _append(value[result_key])
            for key, nested in value.items():
                if (
                    _normalise_typed_product_name(key) == statistic_name
                    and nested is not None
                    and not isinstance(nested, (Mapping, list, tuple))
                ):
                    _append(nested)
                _walk(nested)
        elif isinstance(value, (list, tuple)):
            for item in value:
                _walk(item)

    _walk(step_summary)
    return values


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
        tuple(
            role
            for role in (
                "primary",
                "secondary",
                "sensitivity",
                "corroborative",
            )
            if re.search(rf"\b{role}\b", (step.intent or "").lower())
        ),
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
        include={"cohort", "robustness_specs"},
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
        _normalise_scientific_text(plan.rationale),
    )


def _serializable_plan_scientific_scope_signature(
    plan: AnalysisPlan,
) -> List[Optional[str]]:
    """Return the plan-level signature in manifest-safe form."""

    return list(_plan_scientific_scope_signature(plan))


def _preserve_completed_step_snapshots_after_replan(
    *,
    current_plan: AnalysisPlan,
    revised_plan: AnalysisPlan,
    completed_records: Sequence[Mapping[str, Any]],
) -> Tuple[AnalysisPlan, List[ValidationFinding]]:
    """Keep already-executed Planner steps immutable across replans.

    A replanner may change future work, but it cannot retroactively change the
    scientific request that produced registered evidence. The host-recorded
    ``analysis_request.step`` snapshot and the current plan-level scientific
    scope are execution authority. Replacing either would launder stale evidence
    or permanently block every downstream typed consumer, so restore them before
    accepting the revised DAG.
    """

    current_ids = {str(step.step_id) for step in current_plan.steps}
    snapshots: Dict[str, AnalysisStep] = {}
    completed_current_records = [
        record
        for record in current_successful_step_records(completed_records)
        if str(record.get("step_id") or "").strip() in current_ids
    ]
    for record in completed_current_records:
        step_id = str(record.get("step_id") or "").strip()
        analysis_request = record.get("analysis_request")
        raw_step = (
            analysis_request.get("step")
            if isinstance(analysis_request, Mapping)
            else None
        )
        if step_id not in current_ids or not isinstance(raw_step, Mapping):
            continue
        try:
            snapshot = AnalysisStep.model_validate(raw_step)
        except (TypeError, ValueError):
            continue
        if str(snapshot.step_id) == step_id:
            snapshots[step_id] = snapshot
    changed_ids: List[str] = []
    revised_steps: List[AnalysisStep] = []
    revised_ids: Set[str] = set()
    for step in revised_plan.steps:
        step_id = str(step.step_id)
        snapshot = snapshots.get(step_id)
        if snapshot is not None:
            revised_ids.add(step_id)
            if step.model_dump(mode="json") != snapshot.model_dump(mode="json"):
                changed_ids.append(step_id)
            revised_steps.append(snapshot)
        else:
            revised_steps.append(step)
            revised_ids.add(step_id)

    reinserted_ids: List[str] = []
    current_positions = {
        str(step.step_id): index for index, step in enumerate(current_plan.steps)
    }
    for step_id in sorted(
        snapshots,
        key=lambda value: current_positions.get(value, len(current_positions)),
    ):
        if step_id in revised_ids:
            continue
        insert_at = min(
            current_positions.get(step_id, len(revised_steps)), len(revised_steps)
        )
        revised_steps.insert(insert_at, snapshots[step_id])
        revised_ids.add(step_id)
        reinserted_ids.append(step_id)

    current_scope = _plan_scientific_scope_signature(current_plan)
    revised_scope = _plan_scientific_scope_signature(revised_plan)
    restored_plan_scope = bool(completed_current_records) and (
        revised_scope != current_scope
    )
    restored_plan_scope_fields: List[str] = []
    if restored_plan_scope:
        for field_name in (
            "research_question",
            "analysis_type",
            "cohort",
            "robustness_specs",
            "rationale",
        ):
            if getattr(revised_plan, field_name) != getattr(current_plan, field_name):
                restored_plan_scope_fields.append(field_name)

    if not changed_ids and not reinserted_ids and not restored_plan_scope:
        return revised_plan, []
    update: Dict[str, Any] = {"steps": revised_steps}
    if restored_plan_scope:
        update.update(
            {
                "research_question": current_plan.research_question,
                "analysis_type": current_plan.analysis_type,
                "cohort": current_plan.cohort,
                "robustness_specs": current_plan.robustness_specs,
                "rationale": current_plan.rationale,
            }
        )
    preserved = revised_plan.model_copy(update=update)
    return preserved, [
        ValidationFinding(
            validator="replanner",
            severity="warning",
            message=(
                "Replanner attempted to change completed execution authority; "
                "restored the host-recorded step snapshots and plan-level "
                "scientific scope so registered evidence remains bound to "
                "immutable scientific requests."
            ),
            detail={
                "restored_changed_step_ids": sorted(set(changed_ids)),
                "reinserted_step_ids": reinserted_ids,
                "restored_plan_scope": restored_plan_scope,
                "restored_plan_scope_fields": restored_plan_scope_fields,
                "reason": "completed_step_snapshot_immutable",
            },
        )
    ]


def _resolve_typed_input_evidence(
    *,
    input_name: str,
    plan: AnalysisPlan,
    evidence_records: Sequence[Any],
    per_step_records: Sequence[Mapping[str, Any]],
    run_dir: Path,
) -> Tuple[Optional[EvidenceRef], Optional[Dict[str, Any]]]:
    """Resolve one typed input through the current execution authority.

    The plan declaration identifies the producer; the latest outer step record
    authorizes its evidence ids.  Basename aliases are deliberately excluded:
    they are first-write-wins and can still point at a superseded resume
    artifact.  Every candidate must instead be owned by the successful current
    producer and pass the registered path/SHA check.
    """

    typed_product = _typed_input_product(input_name)
    if typed_product is None:
        return None, {"input": str(input_name), "reason": "invalid_typed_input"}
    product_fields = _lineage_failure_product_fields(typed_product)

    producer_ids = {
        str(step.step_id)
        for step in plan.steps
        if any(
            _typed_input_product(output) == typed_product
            for output in (step.expected_outputs or [])
        )
    }
    if not producer_ids:
        return None, {
            "input": str(input_name),
            **product_fields,
            "reason": "producer_not_declared",
        }
    if len(producer_ids) != 1:
        return None, {
            "input": str(input_name),
            **product_fields,
            "reason": "ambiguous_producer",
            "producer_step_ids": sorted(producer_ids),
        }

    producer_id = next(iter(producer_ids))
    latest_by_step = {
        str(record.get("step_id") or ""): record
        for record in current_step_records(per_step_records)
    }
    producer_record = latest_by_step.get(producer_id)
    producer_status = str((producer_record or {}).get("status") or "").lower()
    if producer_status != "ok":
        return None, {
            "input": str(input_name),
            **product_fields,
            "reason": "producer_not_successful",
            "producer_step_id": producer_id,
            "producer_status": producer_status or "missing",
        }

    active_producer_step = next(
        step for step in plan.steps if str(step.step_id) == producer_id
    )
    analysis_request = (producer_record or {}).get("analysis_request")
    executed_step_payload = (
        analysis_request.get("step") if isinstance(analysis_request, Mapping) else None
    )
    if not isinstance(executed_step_payload, Mapping):
        return None, {
            "input": str(input_name),
            **product_fields,
            "reason": "producer_plan_snapshot_missing",
            "producer_step_id": producer_id,
        }
    try:
        executed_step = AnalysisStep.model_validate(executed_step_payload)
    except (TypeError, ValueError):
        return None, {
            "input": str(input_name),
            **product_fields,
            "reason": "producer_plan_snapshot_invalid",
            "producer_step_id": producer_id,
        }
    if _step_scientific_signature(executed_step) != _step_scientific_signature(
        active_producer_step
    ):
        return None, {
            "input": str(input_name),
            **product_fields,
            "reason": "producer_plan_snapshot_mismatch",
            "producer_step_id": producer_id,
        }

    recorded_scope_signature = (producer_record or {}).get("plan_scientific_signature")
    if not isinstance(recorded_scope_signature, (list, tuple)):
        return None, {
            "input": str(input_name),
            **product_fields,
            "reason": "producer_plan_scope_snapshot_missing",
            "producer_step_id": producer_id,
        }
    if list(recorded_scope_signature) != (
        _serializable_plan_scientific_scope_signature(plan)
    ):
        return None, {
            "input": str(input_name),
            **product_fields,
            "reason": "producer_plan_scope_snapshot_mismatch",
            "producer_step_id": producer_id,
        }

    active_ids = {
        str(evidence_id)
        for evidence_id in (producer_record or {}).get("evidence_ids", [])
        if str(evidence_id).strip()
    }
    if typed_product[0] == "statistic":
        step_summary = (producer_record or {}).get("step_summary")
        recorded_values = _step_summary_statistic_values(
            step_summary,
            typed_product[1],
        )
        recorded_unique_values = sorted(set(recorded_values))
        if not recorded_unique_values:
            return None, {
                "input": str(input_name),
                **product_fields,
                "reason": "statistic_not_materialized",
                "producer_step_id": producer_id,
            }
        if len(recorded_unique_values) != 1:
            return None, {
                "input": str(input_name),
                **product_fields,
                "reason": "statistic_record_value_ambiguous",
                "producer_step_id": producer_id,
                "recorded_values": recorded_unique_values,
            }
        step_summary_evidence_id = str(
            (producer_record or {}).get("step_summary_evidence_id") or ""
        )
        candidates: List[Any] = []
        incompatible_evidence_kinds: Set[str] = set()
        for record in evidence_records:
            evidence_id = str(_evidence_record_field(record, "evidence_id") or "")
            if (
                evidence_id != step_summary_evidence_id
                or evidence_id not in active_ids
                or str(_evidence_record_field(record, "produced_by_step") or "")
                != producer_id
                or verified_run_evidence_path(run_dir, record) is None
            ):
                continue
            if not _evidence_kind_matches_typed_product(record, typed_product):
                incompatible_evidence_kinds.add(
                    str(_evidence_record_field(record, "kind") or "missing")
                )
                continue
            candidates.append(record)
        if len(candidates) != 1:
            return None, {
                "input": str(input_name),
                **product_fields,
                "reason": (
                    "evidence_kind_mismatch"
                    if incompatible_evidence_kinds and not candidates
                    else "no_verified_current_statistic"
                ),
                "producer_step_id": producer_id,
                "step_summary_evidence_id": step_summary_evidence_id or None,
                **(
                    {
                        "declared_kind": typed_product[0],
                        "observed_evidence_kinds": sorted(incompatible_evidence_kinds),
                    }
                    if incompatible_evidence_kinds and not candidates
                    else {}
                ),
            }
        record = candidates[0]
        verified_summary_path = verified_run_evidence_path(run_dir, record)
        try:
            evidence_summary = json.loads(
                verified_summary_path.read_text(encoding="utf-8")
            )
        except (AttributeError, OSError, TypeError, ValueError):
            return None, {
                "input": str(input_name),
                **product_fields,
                "reason": "statistic_evidence_payload_invalid",
                "producer_step_id": producer_id,
            }
        if not isinstance(evidence_summary, Mapping):
            return None, {
                "input": str(input_name),
                **product_fields,
                "reason": "statistic_evidence_payload_not_mapping",
                "producer_step_id": producer_id,
            }
        evidence_values = _step_summary_statistic_values(
            evidence_summary,
            typed_product[1],
        )
        evidence_unique_values = sorted(set(evidence_values))
        if not evidence_unique_values:
            return None, {
                "input": str(input_name),
                **product_fields,
                "reason": "statistic_evidence_value_missing",
                "producer_step_id": producer_id,
            }
        if len(evidence_unique_values) != 1:
            return None, {
                "input": str(input_name),
                **product_fields,
                "reason": "statistic_evidence_value_ambiguous",
                "producer_step_id": producer_id,
                "evidence_values": evidence_unique_values,
            }
        recorded_value = recorded_unique_values[0]
        evidence_value = evidence_unique_values[0]
        if not math.isclose(
            recorded_value,
            evidence_value,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            return None, {
                "input": str(input_name),
                **product_fields,
                "reason": "statistic_evidence_payload_mismatch",
                "producer_step_id": producer_id,
                "recorded_value": recorded_value,
                "evidence_value": evidence_value,
            }
        return (
            EvidenceRef(
                evidence_id=str(_evidence_record_field(record, "evidence_id") or ""),
                kind=_evidence_record_field(record, "kind"),
                description=_evidence_record_field(record, "description"),
                relative_path=_evidence_record_field(record, "relative_path"),
            ),
            None,
        )
    typed_mapping_declared, declared_paths = _declared_typed_product_paths(
        (producer_record or {}).get("step_summary"),
        typed_product=typed_product,
    )
    if typed_mapping_declared and not declared_paths:
        return None, {
            "input": str(input_name),
            **product_fields,
            "reason": "typed_mapping_not_verified",
            "producer_step_id": producer_id,
        }
    if len(declared_paths) > 1:
        return None, {
            "input": str(input_name),
            **product_fields,
            "reason": "ambiguous_typed_mapping",
            "producer_step_id": producer_id,
            "declared_paths": declared_paths,
        }
    declared_filename = Path(declared_paths[0]).name if declared_paths else None

    candidates: List[Tuple[Any, Path]] = []
    matching_current_ids: List[str] = []
    incompatible_evidence_kinds: Set[str] = set()
    for record in evidence_records:
        evidence_id = str(_evidence_record_field(record, "evidence_id") or "")
        if (
            evidence_id not in active_ids
            or str(_evidence_record_field(record, "produced_by_step") or "")
            != producer_id
        ):
            continue
        verified_path = verified_run_evidence_path(run_dir, record)
        if verified_path is None:
            continue
        source_name = _registered_source_name(record, verified_path)
        if source_name is None:
            continue
        if declared_filename is not None:
            matches_product = source_name == declared_filename
        else:
            matches_product = (
                _normalise_typed_product_name(source_name) == typed_product[1]
            )
        if not matches_product:
            continue
        if not _evidence_kind_matches_typed_product(record, typed_product):
            incompatible_evidence_kinds.add(
                str(_evidence_record_field(record, "kind") or "missing")
            )
            continue
        matching_current_ids.append(evidence_id)
        candidates.append((record, verified_path))

    if not candidates:
        return None, {
            "input": str(input_name),
            **product_fields,
            "reason": (
                "evidence_kind_mismatch"
                if incompatible_evidence_kinds
                else (
                    "typed_mapping_not_verified"
                    if declared_filename is not None
                    else "no_verified_current_artifact"
                )
            ),
            "producer_step_id": producer_id,
            **(
                {
                    "declared_kind": typed_product[0],
                    "observed_evidence_kinds": sorted(incompatible_evidence_kinds),
                }
                if incompatible_evidence_kinds
                else {}
            ),
            **(
                {"declared_path": declared_paths[0]}
                if declared_filename is not None
                else {}
            ),
        }
    if len(candidates) != 1:
        return None, {
            "input": str(input_name),
            **product_fields,
            "reason": "ambiguous_current_artifact",
            "producer_step_id": producer_id,
            "evidence_ids": sorted(matching_current_ids),
        }

    record, _ = candidates[0]
    return (
        EvidenceRef(
            evidence_id=str(_evidence_record_field(record, "evidence_id") or ""),
            kind=_evidence_record_field(record, "kind"),
            description=_evidence_record_field(record, "description"),
            relative_path=_evidence_record_field(record, "relative_path"),
        ),
        None,
    )


def _resolve_typed_artifact_evidence(
    *,
    input_name: str,
    plan: AnalysisPlan,
    evidence_records: Sequence[Any],
    per_step_records: Sequence[Mapping[str, Any]],
    run_dir: Path,
) -> Tuple[Optional[EvidenceRef], Optional[Dict[str, Any]]]:
    """Compatibility wrapper preserving the public artifact resolver."""

    if _typed_artifact_name(input_name) is None:
        return None, {"input": str(input_name), "reason": "invalid_artifact_input"}
    return _resolve_typed_input_evidence(
        input_name=input_name,
        plan=plan,
        evidence_records=evidence_records,
        per_step_records=per_step_records,
        run_dir=run_dir,
    )


def _resolved_typed_input_binding(
    *,
    input_name: str,
    evidence_ref: EvidenceRef,
    evidence_records: Sequence[Any],
    run_dir: Path,
    producer_step_records: Sequence[Mapping[str, Any]] = (),
    authoritative_cohort_path: Optional[Path] = None,
) -> Optional[Dict[str, Any]]:
    """Build the exact, digest-verified runtime binding for one typed input."""

    typed_product = _typed_input_product(input_name)
    if typed_product is None:
        return None
    record = next(
        (
            candidate
            for candidate in evidence_records
            if str(_evidence_record_field(candidate, "evidence_id") or "")
            == evidence_ref.evidence_id
        ),
        None,
    )
    if record is None:
        return None
    if not _evidence_kind_matches_typed_product(record, typed_product):
        return None
    verified_path = verified_run_evidence_path(run_dir, record)
    if verified_path is None:
        return None
    run_root = Path(run_dir).resolve()
    try:
        run_relative_path = verified_path.relative_to(run_root).as_posix()
    except ValueError:
        return None
    declared_kind, product_name = typed_product
    binding = {
        "evidence_id": evidence_ref.evidence_id,
        "declared_kind": declared_kind,
        "product": product_name,
        "evidence_kind": str(_evidence_record_field(record, "kind") or ""),
        "relative_path": run_relative_path,
        "absolute_path": str(verified_path),
        "sha256": str(_evidence_record_field(record, "sha256") or ""),
        "produced_by_step": str(
            _evidence_record_field(record, "produced_by_step") or ""
        ),
    }
    producer_contract: Optional[Dict[str, Any]] = None
    for step_record in reversed(list(producer_step_records)):
        if str(step_record.get("status") or "") != "ok":
            continue
        evidence_ids = {str(value) for value in (step_record.get("evidence_ids") or [])}
        if evidence_ref.evidence_id not in evidence_ids:
            continue
        step_summary = step_record.get("step_summary")
        if not isinstance(step_summary, Mapping):
            break
        product_contract = typed_product_binding_contract(
            product_name=product_name,
            step_summary=step_summary,
            artifact_path=verified_path,
            authoritative_cohort_path=authoritative_cohort_path,
        )
        if product_contract is not None:
            producer_contract = dict(product_contract)
        break
    contract_required = product_name in {
        "assignment_model",
        "primary_exposure_definition",
        "prespecified_confounder_set",
    }
    if contract_required and producer_contract is None:
        return None
    identity_row = {
        "input_key": str(input_name),
        "declared_kind": declared_kind,
        "product": product_name,
        "evidence_id": evidence_ref.evidence_id,
        "sha256": binding["sha256"],
        "produced_by_step": binding["produced_by_step"],
    }
    host_contract = dict(producer_contract or {})
    host_contract.update(
        {
            "schema_version": "easyicu.host_typed_product.v1",
            "identity_row": identity_row,
        }
    )
    binding["identity_row"] = identity_row
    binding["product_contract"] = host_contract
    return binding


def _write_resolved_inputs_manifest(
    *,
    run_dir: Path,
    step_id: str,
    bindings: Mapping[str, Mapping[str, Any]],
    context_path: Optional[Path] = None,
) -> Path:
    """Persist the step's authority capsule outside its writable overlay."""

    safe_step_id = str(step_id or "")
    if (
        not safe_step_id
        or safe_step_id in {".", ".."}
        or Path(safe_step_id).name != safe_step_id
        or "/" in safe_step_id
        or "\\" in safe_step_id
    ):
        raise ValueError("step_id must be a single safe path component")
    manifest_dir = Path(run_dir).resolve() / "resolved_inputs"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / f"{safe_step_id}.json"
    payload: Dict[str, Any] = {
        "schema_version": "2.0",
        "step_id": safe_step_id,
        "inputs": {str(key): dict(value) for key, value in bindings.items()},
    }
    if context_path is not None:
        resolved_context = Path(context_path).resolve()
        run_root = Path(run_dir).resolve()
        if not resolved_context.is_file():
            raise ValueError("context_path must name an existing context file")
        try:
            relative_context = resolved_context.relative_to(run_root).as_posix()
        except ValueError as exc:
            raise ValueError("context_path must be contained by run_dir") from exc
        payload["context"] = {
            "relative_path": relative_context,
            "absolute_path": str(resolved_context),
            "sha256": sha256_of_file(resolved_context),
        }
    temporary_path = manifest_path.with_suffix(".json.tmp")
    temporary_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(manifest_path)
    return manifest_path


class _InertPythonNodeStripper(ast.NodeTransformer):
    """Remove syntax that cannot repair analytical behavior."""

    def visit_Pass(self, node: ast.Pass) -> None:
        del node
        return None

    def visit_Expr(self, node: ast.Expr) -> Optional[ast.Expr]:
        node = self.generic_visit(node)
        if isinstance(node.value, ast.Constant):
            return None
        return node


def _python_semantic_sha256(code: str) -> Optional[str]:
    """Hash executable Python structure while ignoring comments/whitespace."""

    try:
        tree = _InertPythonNodeStripper().visit(ast.parse(code))
        normalized = ast.dump(
            tree,
            annotate_fields=True,
            include_attributes=False,
        )
    except (SyntaxError, TypeError, ValueError):
        return None
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _python_repair_is_materially_changed(before: str, after: str) -> bool:
    """Reject exact and AST-equivalent repair responses."""

    if (
        hashlib.sha256(before.encode("utf-8")).digest()
        == hashlib.sha256(after.encode("utf-8")).digest()
    ):
        return False
    before_semantic = _python_semantic_sha256(before)
    after_semantic = _python_semantic_sha256(after)
    if before_semantic is not None and before_semantic == after_semantic:
        return False
    return True


def _quarantined_errors_superseded_by_current_policy(
    *,
    prior_errors: Sequence[ValidationFinding],
    current_findings: Sequence[ValidationFinding],
    context: ResearchContext,
    script_text: str,
    quarantined_script_sha256: str,
) -> Optional[Tuple[List[ValidationFinding], List[Dict[str, Any]]]]:
    """Prove that stored errors were retired by a deterministic policy change.

    Absence of a finding from a new optional LLM audit is not evidence that an
    old quarantine is stale. The only no-code-change exit is to replay the
    current metadata-supported outcome reclassifier over every stored error,
    while the complete current audit independently has no errors.
    """

    if hashlib.sha256(script_text.encode("utf-8")).hexdigest() != str(
        quarantined_script_sha256 or ""
    ):
        return None
    if not prior_errors or any(
        finding.severity == "error" for finding in current_findings
    ):
        return None
    if any(finding.severity != "error" for finding in prior_errors):
        return None
    reclassified = _reclassify_llm_concept_findings(
        findings=prior_errors,
        context=context,
        script_text=script_text,
    )
    if len(reclassified) != len(prior_errors):
        return None

    provenance: List[Dict[str, Any]] = []
    for prior, current in zip(prior_errors, reclassified):
        prior_detail = dict(prior.detail or {})
        current_detail = dict(current.detail or {})
        reason = current_detail.get("downgraded_reason")
        same_finding = (
            current.validator == prior.validator
            and current.message == prior.message
            and current.evidence_ids == prior.evidence_ids
            and all(
                current_detail.get(key) == value for key, value in prior_detail.items()
            )
        )
        if (
            not same_finding
            or "downgraded_reason" in prior_detail
            or current.severity != "warning"
            or not isinstance(reason, str)
            or not reason.strip()
        ):
            return None
        provenance.append(
            {
                "validator": prior.validator,
                "message": prior.message,
                "prior_severity": prior.severity,
                "reclassified_severity": current.severity,
                "downgraded_reason": reason.strip(),
            }
        )
    return reclassified, provenance


def _repair_publication_figure_in_staging(
    *,
    run_dir: Path,
    current_step_id: str,
    out_dir: Path,
    authorizer: Callable[[str], bool],
    step_text: str = "",
    renderer: Callable[..., Optional[str]] = (
        _render_publication_bundle_from_prior_outputs_for_step
    ),
) -> Optional[str]:
    """Render into staging and replace agent exports only after success.

    A routing false-positive or strict renderer guard returning ``None`` must
    leave the agent-produced figure, source data, and contract untouched.  Once
    a staged renderer emits a real figure export, move the old directory into a
    same-filesystem backup, install the staged bundle, and roll back on any move
    failure.
    """

    out_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".publication-figure-repair-", dir=out_dir.parent
    ) as staging_name:
        staging_dir = Path(staging_name)
        try:
            repair_id = renderer(
                run_dir=run_dir,
                current_step_id=current_step_id,
                out_dir=staging_dir,
                step_text=step_text,
            )
        except Exception as exc:
            logger.warning(
                "Staged publication-figure repair failed for %s: %s",
                current_step_id,
                exc,
            )
            return None
        if (
            repair_id is None
            or is_sealed_renderer_repair(repair_id)
            or not _has_figure_exports(staging_dir)
        ):
            return None
        # Rendering into an isolated temporary directory is non-authoritative.
        # Ask the central repair policy before installing any generated bundle
        # into the live step directory.
        if not authorizer(repair_id):
            return None

        backup_dir = Path(
            tempfile.mkdtemp(prefix=".publication-figure-backup-", dir=out_dir.parent)
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        try:
            for child in list(out_dir.iterdir()):
                shutil.move(str(child), str(backup_dir / child.name))
            for child in list(staging_dir.iterdir()):
                shutil.move(str(child), str(out_dir / child.name))
        except Exception:
            _clear_output_dir(out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            for child in list(backup_dir.iterdir()):
                shutil.move(str(child), str(out_dir / child.name))
            raise
        finally:
            shutil.rmtree(backup_dir, ignore_errors=True)

        # Renderers may store absolute output paths in JSON summaries/contracts.
        # They were valid in staging; rewrite only that exact directory prefix
        # after the atomic-style move so provenance points to the installed bundle.
        for json_path in out_dir.rglob("*.json"):
            try:
                content = json_path.read_text(encoding="utf-8")
                rewritten = content.replace(str(staging_dir), str(out_dir))
                if rewritten != content:
                    json_path.write_text(rewritten, encoding="utf-8")
            except Exception:
                continue
        return repair_id


def _blocking_validator_findings(
    *finding_groups: Sequence[ValidationFinding],
) -> List[ValidationFinding]:
    """Keep only fail-closed findings that may drive code or Critic repair."""

    return [
        finding
        for group in finding_groups
        for finding in group
        if finding.severity == "error"
    ]


def _actionable_validator_messages(
    *finding_groups: Sequence[ValidationFinding],
) -> List[str]:
    """Return only blocking validator messages that require Critic action.

    Warning and informational audit records remain in the manifest and global
    findings, but the untyped Critic input cannot preserve their severity. If
    forwarded as bare strings they become ``needs_revision`` and incorrectly
    fail an otherwise valid step. Only fail-closed errors are actionable here.
    """

    return [
        finding.message
        for finding in _blocking_validator_findings(*finding_groups)
        if finding.message
    ]


_SUCCESS_REPLAN_REQUEST_FIELDS = (
    "replan_requested",
    "plan_revision_requested",
)


def _successful_step_requests_replan(record: Mapping[str, Any]) -> bool:
    """Return whether a clean agent step explicitly requests plan adaptation.

    The deterministic probe already receives one automatic replan and failed
    model steps have their own bounded directed-replan path. Calling the LLM
    replanner after every ordinary successful step adds latency and usually
    produces a no-op. Preserve adaptive agent behavior through exact boolean
    declarations in either the outer record or ``step_summary``; strings and
    other truthy values are intentionally not accepted.
    """

    if str(record.get("status") or "") != "ok":
        return False
    containers: List[Mapping[str, Any]] = [record]
    summary = record.get("step_summary")
    if isinstance(summary, Mapping):
        containers.append(summary)
    return any(
        container.get(field) is True
        for container in containers
        for field in _SUCCESS_REPLAN_REQUEST_FIELDS
    )


def _preserve_locked_robustness_specs_after_replan(
    *,
    current_plan: AnalysisPlan,
    revised_plan: AnalysisPlan,
    run_dir: Path,
) -> tuple[AnalysisPlan, Optional[ValidationFinding]]:
    """Keep probe/runtime replans from mutating the plan-time spec lock."""

    locked_specs = robustness_specs_for_execution(
        run_dir=run_dir,
        plan=current_plan,
    )
    revised_specs = list(revised_plan.robustness_specs or [])
    if robustness_specs_sha(revised_specs) == robustness_specs_sha(locked_specs):
        return revised_plan, None
    preserved = revised_plan.model_copy(update={"robustness_specs": list(locked_specs)})
    return preserved, ValidationFinding(
        validator="replanner",
        severity="warning",
        message=(
            "Replanner attempted to change the immutable plan-time robustness "
            "specifications; preserved the verified lock and retained only the "
            "other plan revisions."
        ),
        detail={
            "reason": "preserve_locked_robustness_specs",
            "locked_spec_ids": [spec.spec_id for spec in locked_specs],
        },
    )


def _step_status_from_contract_findings(
    *,
    contract_findings: Sequence[ValidationFinding],
    figure_source_findings: Sequence[ValidationFinding],
    stat_findings: Sequence[ValidationFinding],
    critique_status: Optional[str] = None,
) -> str:
    """Map deterministic review failures to the outer step status.

    A Critic ``needs_revision`` decision is not a successful scientific step.
    Contract validation normally catches objective defects early enough for an
    in-run coder repair, but the Critic is the final independent review layer;
    its negative decision must therefore remain fail-closed rather than being
    stored as a warning on an otherwise ``ok`` record.
    """

    has_contract_error = any(
        finding.severity == "error"
        for finding in (
            list(contract_findings) + list(figure_source_findings) + list(stat_findings)
        )
    )
    if has_contract_error:
        return "contract_failed"
    if str(critique_status or "").strip().lower() in {
        "needs_revision",
        "blocked",
    }:
        return "critic_failed"
    return "ok"


def _bind_findings_to_step_attempt(
    findings: Sequence[ValidationFinding],
    *,
    step_id: str,
    attempt_id: str,
    checkpoint_id: str,
) -> List[ValidationFinding]:
    """Attach host-owned execution identity to deterministic findings.

    Validator payloads are intentionally reusable and therefore do not know
    which resume attempt invoked them.  Supersession must never infer that
    identity from a message string: the orchestrator binds it at the review
    checkpoint before persisting either the finding or the outer step record.
    """

    bound: List[ValidationFinding] = []
    for finding in findings:
        detail = dict(finding.detail or {})
        detail.update(
            {
                "step_id": step_id,
                "attempt_id": attempt_id,
                "checkpoint_id": checkpoint_id,
            }
        )
        bound.append(finding.model_copy(update={"detail": detail}))
    return bound


_LOCKED_MEASUREMENT_DATA_QUALITY_ISSUES = frozenset(
    {
        "measurement_provenance_count_column_ambiguous",
        "measurement_provenance_count_flag_discordance",
        "measurement_provenance_host_replay_failed",
        "measurement_provenance_host_source_missing",
        "measurement_provenance_host_source_unreadable",
        "measurement_provenance_invalid_measured_values",
        "measurement_provenance_invalid_pairs",
        "measurement_provenance_measured_column_missing",
    }
)


def _locked_measurement_data_quality_issues(
    contract_findings: Sequence[ValidationFinding],
) -> List[str]:
    """Identify locked-cohort facts that generated code cannot repair."""

    return sorted(
        {
            str(finding.detail.get("issue"))
            for finding in contract_findings
            if finding.severity == "error"
            and finding.validator == StepSummaryIntegrityValidator.name
            and finding.detail.get("issue") in _LOCKED_MEASUREMENT_DATA_QUALITY_ISSUES
        }
    )


def _step_requires_publication_figure_exports(step: AnalysisStep) -> bool:
    """Return whether ``step`` structurally owns a figure export contract.

    Step ids and intents are narrative metadata and may mention a downstream
    publication figure without declaring one as this step's product.  The
    mandatory export gate therefore accepts only an exact publication-renderer
    method or the closed method/output evidence recognised by
    :func:`_step_expects_figure`.
    """

    method = str(step.method or "").strip().lower()
    return method == "publication_figure_generation" or _step_expects_figure(step)


def _step_has_figure_only_output_contract(step: AnalysisStep) -> bool:
    """Whether replacing ``outputs/`` can only replace presentation artifacts.

    Deterministic renderers install a complete staged bundle.  They are safe as
    a preflight or whole-directory repair only for an explicitly figure-only
    step; a mixed table/model + figure contract must stay with the coder so a
    renderer cannot erase or silently stand in for scientific products.
    """

    outputs = [
        str(output or "").strip()
        for output in (step.expected_outputs or [])
        if str(output or "").strip()
    ]

    def _is_typed_figure_product(output: str) -> bool:
        token = str(output or "").strip().lower()
        kind, separator, _product = token.partition(":")
        if separator:
            # The artifact kind is authoritative. A scientific table/model
            # whose product name happens to contain ``figure`` or ``plot`` is
            # still a mixed contract and must remain coder-owned.
            return kind.strip() in {"figure", "plot", "chart", "fig", "heatmap"}
        # Legacy bare declarations are figure-only only when they name an
        # actual image/vector export, never from a keyword in the stem.
        return token.endswith((".png", ".svg", ".pdf", ".tif", ".tiff"))

    return bool(outputs) and all(_is_typed_figure_product(output) for output in outputs)


def _read_locked_robustness_spec_dicts(run_dir: Path) -> List[Dict[str, Any]]:
    payload = json.loads(
        (Path(run_dir) / "robustness_specs_locked.json").read_text(encoding="utf-8")
    )
    raw_specs = payload.get("specs") if isinstance(payload, dict) else None
    if not isinstance(raw_specs, list):
        raise ValueError("robustness_specs_locked.json has no specs list")
    return [dict(spec) for spec in raw_specs if isinstance(spec, dict)]


_AGENT_OWNED_ROBUSTNESS_RESULT_METHODS = frozenset(
    {
        "cohort_definition_sensitivity",
        "prespecified_robustness_analysis",
    }
)
_AGENT_OWNED_ROBUSTNESS_RESULT_PRODUCTS = frozenset(
    {
        "cohort_definition_overlap_attrition",
        "cohort_overlap_and_attrition",
        "complete_case_n",
        "missingness_strategy_notes",
        "primary_or",
        "robustness_grid",
        "robustness_matrix",
        "robustness_summary",
        "sensitivity_comparison",
        "sensitivity_specification_matrix",
    }
)


def _is_cohort_definition_sensitivity_result_step(step: AnalysisStep) -> bool:
    """Return true for an agent-owned, plan-locked robustness result step.

    This predicate attaches specifications and validation; it does *not*
    dispatch a deterministic runner.  Ownership requires an exact controlled
    method head and a closed structured-product set so prose, step ids, or one
    stray robustness keyword cannot opt an unrelated analysis into the gate.
    """

    if _step_expects_figure(step):
        return False
    if _method_head(str(step.method or "")) not in (
        _AGENT_OWNED_ROBUSTNESS_RESULT_METHODS
    ):
        return False
    products = _closed_auxiliary_output_products(
        step.expected_outputs or [],
        supported_products=_AGENT_OWNED_ROBUSTNESS_RESULT_PRODUCTS,
    )
    return products is not None and bool(
        products
        & {
            "robustness_grid",
            "robustness_matrix",
            "robustness_summary",
            "sensitivity_comparison",
            "sensitivity_specification_matrix",
        }
    )


def _authoritative_primary_robustness_contract(
    *,
    completed_step_records: Sequence[Mapping[str, Any]],
    context: Optional[ResearchContext],
) -> Optional[Dict[str, Any]]:
    """Return one fitted primary model contract for robustness re-estimation.

    The robustness step is auxiliary: it may execute Planner-locked variants,
    but it may not select a different estimator, outcome, or exposure.  Bind it
    to the latest successful agent-produced primary contract that exactly
    matches the research context. Ambiguity remains fail-closed.
    """

    expected_exposure = str(
        (context.primary_exposure if context is not None else None) or ""
    ).strip()
    expected_outcome = str(
        (context.target_outcome if context is not None else None) or ""
    ).strip()
    for record in reversed(
        list(current_successful_step_records(completed_step_records))
    ):
        summary = record.get("step_summary")
        if not isinstance(summary, Mapping):
            continue
        candidates: List[Dict[str, Any]] = []
        for raw_contract in summary.get("model_contracts") or []:
            if not isinstance(raw_contract, Mapping):
                continue
            contract = dict(raw_contract)
            if str(contract.get("analysis_role") or "").strip().lower() != "primary":
                continue
            if str(contract.get("exposure_role") or "primary").strip().lower() != (
                "primary"
            ):
                continue
            if str(contract.get("fit_status") or "").strip().lower() != "fitted":
                continue
            if contract.get("converged") is not True:
                continue
            if expected_exposure and str(contract.get("exposure_source") or "") != (
                expected_exposure
            ):
                continue
            if expected_outcome and str(contract.get("outcome") or "") != (
                expected_outcome
            ):
                continue
            candidates.append(contract)
        if len(candidates) != 1:
            continue
        contract = candidates[0]
        contract["source_step_id"] = str(record.get("step_id") or "")
        final_terms = summary.get("final_design_terms")
        if isinstance(final_terms, list):
            contract["final_design_terms"] = list(final_terms)
        return contract
    return None


def _coder_context_with_locked_robustness_specs(
    *,
    context: ResearchContext,
    step: AnalysisStep,
    run_dir: Path,
) -> ResearchContext:
    """Attach the planner-locked variant contract to its execution step."""

    if not _is_cohort_definition_sensitivity_result_step(step):
        return context
    try:
        specs = _read_locked_robustness_spec_dicts(run_dir)
    except Exception:
        return context
    if not specs:
        return context
    fields = (
        "spec_id",
        "axis",
        "description",
        "cohort_override",
        "missing_override",
        "outcome_override",
    )
    locked_contract = [{field: spec.get(field) for field in fields} for spec in specs]
    primary_contract: Optional[Dict[str, Any]] = None
    for manifest_name in ("manifest_partial.json", "manifest.json"):
        manifest_path = Path(run_dir) / manifest_name
        if not manifest_path.is_file():
            continue
        try:
            manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        raw_records = (
            manifest_payload.get("per_step_records")
            if isinstance(manifest_payload, Mapping)
            else None
        )
        if not isinstance(raw_records, list):
            continue
        primary_contract = _authoritative_primary_robustness_contract(
            completed_step_records=raw_records,
            context=context,
        )
        if primary_contract is not None:
            break
    attachment = (
        "LOCKED ROBUSTNESS SPECIFICATIONS (binding plan-time state):\n"
        + json.dumps(locked_contract, ensure_ascii=False, separators=(",", ":"))
        + "\nExecute every spec_id exactly as declared; do not rename, replace, "
        "or invent specifications. Cohort-axis definitions that can recover "
        "rows outside the locked analysis cohort must be materialised from "
        "os.environ['EASYICU_UNIVERSE_PARQUET']; COHORT_PARQUET is the locked "
        "analysis cohort."
        "\n\n" + ROBUSTNESS_EXECUTION_CONTRACT_GUIDANCE
    )
    if primary_contract is not None:
        attachment += (
            "\n\nAUTHORITATIVE PRIMARY MODEL CONTRACT (binding; variants must "
            "re-estimate this model rather than substitute descriptive risks):\n"
            + json.dumps(
                primary_contract,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        )
    prior_notes = str(context.notes or "").strip()
    enriched_notes = f"{prior_notes}\n\n{attachment}" if prior_notes else attachment
    return context.model_copy(update={"notes": enriched_notes})


def _nonnegative_integral_value(value: Any) -> Optional[int]:
    if value is None or isinstance(value, bool):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not numeric.is_integer() or numeric < 0:
        return None
    return int(numeric)


def _declared_sensitivity_csv_paths(
    *,
    step_summary: Dict[str, Any],
    out_dir: Path,
) -> Tuple[List[Path], List[Path]]:
    """Return declared spec-table paths and denominator-table paths."""

    spec_roles = {
        "table:robustness_grid",
        "table:sensitivity_specification_matrix",
        "table:robustness_summary",
        "table:robustness_matrix",
        "robustness_grid",
        "sensitivity_specification_matrix",
        "robustness_summary",
        "robustness_matrix",
    }
    denominator_roles = spec_roles | {
        "table:cohort_definition_overlap_attrition",
        "table:cohort_overlap_and_attrition",
        "cohort_definition_overlap_attrition",
        "cohort_overlap_and_attrition",
    }
    spec_names = {
        "robustness_grid.csv",
        "sensitivity_specification_matrix.csv",
        "robustness_summary.csv",
        "robustness_matrix.csv",
    }
    denominator_names = spec_names | {
        "cohort_definition_overlap_attrition.csv",
        "cohort_overlap_and_attrition.csv",
    }
    root = Path(out_dir).resolve()

    def _local_csv(value: Any) -> Optional[Path]:
        text = str(value or "").strip()
        if not text:
            return None
        path = Path(text)
        if not path.is_absolute():
            path = root / path
        path = path.resolve()
        if not path.is_relative_to(root) or path.suffix.lower() != ".csv":
            return None
        return path if path.is_file() else None

    spec_paths: List[Path] = []
    denominator_paths: List[Path] = []
    output_files = step_summary.get("output_files")
    if isinstance(output_files, dict):
        for role, value in output_files.items():
            normalised_role = str(role or "").strip().lower()
            path = _local_csv(value)
            if path is None:
                continue
            if normalised_role in spec_roles:
                spec_paths.append(path)
            if normalised_role in denominator_roles:
                denominator_paths.append(path)
    for values in (
        output_files if isinstance(output_files, list) else [],
        (
            step_summary.get("outputs")
            if isinstance(step_summary.get("outputs"), list)
            else []
        ),
    ):
        for value in values:
            path = _local_csv(value)
            if path is None:
                continue
            if path.name in spec_names:
                spec_paths.append(path)
            if path.name in denominator_names:
                denominator_paths.append(path)
    for name in denominator_names:
        path = root / name
        if not path.is_file():
            continue
        if name in spec_names:
            spec_paths.append(path)
        denominator_paths.append(path)
    return list(dict.fromkeys(spec_paths)), list(dict.fromkeys(denominator_paths))


def _sensitivity_csv_rows(paths: Sequence[Path]) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for path in paths:
        try:
            with path.open("r", encoding="utf-8", newline="") as handle:
                for raw_row in csv.DictReader(handle):
                    row = dict(raw_row)
                    # ``definition_id`` is the natural identifier in cohort
                    # overlap/attrition tables, while the locked robustness
                    # contract calls the same key ``spec_id``.  Normalize the
                    # typed table role here; values are still checked against
                    # the digest-bound lock and deterministic membership
                    # replay below, so this cannot authorize an invented id.
                    if (
                        not str(row.get("spec_id") or "").strip()
                        and str(row.get("definition_id") or "").strip()
                    ):
                        row["spec_id"] = row["definition_id"]
                    rows.append(row)
        except (OSError, csv.Error):
            continue
    return rows


def _cohort_definition_sensitivity_contract_findings(
    *,
    step: AnalysisStep,
    step_summary: Dict[str, Any],
    out_dir: Path,
    run_dir: Path,
    universe_path: Path,
    cohort_path: Optional[Path] = None,
    context: Optional[ResearchContext] = None,
    completed_step_records: Sequence[Mapping[str, Any]] = (),
) -> List[ValidationFinding]:
    """Verify that an agent executed, rather than replaced, its locked specs."""

    if not _is_cohort_definition_sensitivity_result_step(step):
        return []
    # Findings emitted here belong to one exact planner step.  Keep that
    # ownership machine-readable so a successful retry/resume can supersede an
    # older failure without parsing prose or treating the whole robustness
    # subsystem as one global gate.
    step_detail = {"step_id": str(step.step_id)}
    try:
        locked_specs = _read_locked_robustness_spec_dicts(run_dir)
    except Exception as exc:
        return [
            ValidationFinding(
                validator="robustness_spec_lock",
                severity="error",
                message=f"Locked robustness definitions are unavailable: {exc}",
                detail={
                    **step_detail,
                    "lock_path": str(Path(run_dir) / "robustness_specs_locked.json"),
                },
            )
        ]

    locked_by_id = {
        str(spec.get("spec_id") or "").strip(): spec
        for spec in locked_specs
        if str(spec.get("spec_id") or "").strip()
    }
    executed_result_issues = _executed_robustness_result_issues(
        locked_by_id=locked_by_id,
        step_summary=step_summary,
        out_dir=out_dir,
        context=context,
        primary_model_contract=_authoritative_primary_robustness_contract(
            completed_step_records=completed_step_records,
            context=context,
        ),
    )
    reported_rows: List[Dict[str, Any]] = []
    raw_rows = step_summary.get("robustness_rows")
    if raw_rows is None and isinstance(step_summary.get("robustness_panel"), dict):
        raw_rows = step_summary["robustness_panel"].get("rows")
    if isinstance(raw_rows, list):
        for row in raw_rows:
            if not isinstance(row, dict):
                continue
            reported_rows.append(dict(row))

    spec_paths, denominator_paths = _declared_sensitivity_csv_paths(
        step_summary=step_summary,
        out_dir=out_dir,
    )
    reported_rows.extend(
        _sensitivity_csv_rows(list(dict.fromkeys([*spec_paths, *denominator_paths])))
    )

    rows_by_id: Dict[str, List[Dict[str, Any]]] = {}
    for row in reported_rows:
        spec_id = str(row.get("spec_id") or "").strip()
        if spec_id:
            rows_by_id.setdefault(spec_id, []).append(row)
    reported_ids = set(rows_by_id)

    locked_ids = set(locked_by_id)
    missing_ids = sorted(locked_ids - reported_ids)
    extra_ids = sorted(reported_ids - locked_ids - {"primary"})
    findings: List[ValidationFinding] = []
    if executed_result_issues:
        findings.append(
            ValidationFinding(
                validator="robustness_executed_result",
                severity="error",
                message=(
                    "Each locked robustness specification must have exactly one "
                    "typed executed-result row bound to its fitted model and "
                    "coefficient evidence. Declaration and membership tables "
                    "cannot substitute for execution. Issues="
                    f"{executed_result_issues}."
                ),
                detail={
                    **step_detail,
                    "required_spec_ids": sorted(locked_by_id),
                    "issues": executed_result_issues,
                    "point_only_policy": (
                        "Penalized point-only fits may be executed but must set "
                        "reportable=false and interval_method=unavailable."
                    ),
                },
            )
        )
    missing_axis_ids: List[str] = []
    axis_mismatches: List[Dict[str, str]] = []
    for spec_id, spec in locked_by_id.items():
        expected_axis = str(spec.get("axis") or "").strip().lower()
        reported_axes = {
            str(row.get("axis") or "").strip().lower()
            for row in rows_by_id.get(spec_id, [])
            if str(row.get("axis") or "").strip()
        }
        if spec_id in reported_ids and not reported_axes:
            missing_axis_ids.append(spec_id)
        for reported_axis in sorted(reported_axes - {expected_axis}):
            axis_mismatches.append(
                {
                    "spec_id": spec_id,
                    "expected_axis": expected_axis,
                    "reported_axis": reported_axis,
                }
            )
    if missing_ids or extra_ids or missing_axis_ids or axis_mismatches:
        missing_definitions = [locked_by_id[spec_id] for spec_id in missing_ids]
        findings.append(
            ValidationFinding(
                validator="robustness_spec_lock",
                severity="error",
                message=(
                    "Cohort-definition sensitivity outputs must cover every "
                    "plan-time locked spec_id and axis without substitutes. "
                    f"Missing={missing_ids}; extra={extra_ids}; "
                    f"missing_axis={missing_axis_ids}; axis_mismatches="
                    f"{axis_mismatches}; missing locked "
                    "definitions="
                    + json.dumps(
                        missing_definitions,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                ),
                detail={
                    **step_detail,
                    "locked_spec_ids": sorted(locked_ids),
                    "reported_spec_ids": sorted(reported_ids),
                    "missing_spec_ids": missing_ids,
                    "extra_spec_ids": extra_ids,
                    "missing_axis_spec_ids": missing_axis_ids,
                    "axis_mismatches": axis_mismatches,
                    "missing_spec_definitions": missing_definitions,
                    "specification_tables": [str(path) for path in spec_paths],
                },
            )
        )

    cohort_specs = [
        RobustnessSpec.from_dict(spec)
        for spec in locked_specs
        if str(spec.get("axis") or "").strip().lower() == "cohort"
    ]
    if cohort_specs:
        membership_issues: List[Dict[str, Any]] = []
        try:
            import pandas as pd  # type: ignore

            if cohort_path is None:
                raise ValueError("locked analysis cohort path is unavailable")
            universe = pd.read_parquet(universe_path)
            cohort = pd.read_parquet(cohort_path)
            replay_rows = replay_locked_memberships(
                specs=cohort_specs,
                cohort=cohort,
                universe=universe,
                context=context,
                exposure=str((context.primary_exposure if context else None) or ""),
            )
        except Exception as exc:
            findings.append(
                ValidationFinding(
                    validator="robustness_cohort_membership",
                    severity="error",
                    message=f"Could not replay locked cohort memberships: {exc}",
                    detail={
                        **step_detail,
                        "universe_path": str(universe_path),
                        "cohort_path": str(cohort_path) if cohort_path else None,
                    },
                )
            )
            return findings

        replay_by_id = {
            str(row.get("spec_id") or ""): row
            for row in replay_rows
            if str(row.get("spec_id") or "")
        }
        aliases = ROBUSTNESS_COHORT_MEMBERSHIP_ALIASES
        for spec in cohort_specs:
            spec_id = spec.spec_id
            expected = replay_by_id.get(spec_id) or {}
            expected_inflow = _nonnegative_integral_value(expected.get("inflow_n"))
            expected_outflow = _nonnegative_integral_value(expected.get("outflow_n"))
            expected_primary = _nonnegative_integral_value(
                expected.get("primary_membership_n")
            )
            expected_variant = _nonnegative_integral_value(
                expected.get("variant_membership_n")
            )
            expected_values = {
                "universe_n": _nonnegative_integral_value(expected.get("universe_n")),
                "variant_membership_n": expected_variant,
                "inflow_n": expected_inflow,
                "outflow_n": expected_outflow,
                "overlap_n": (
                    expected_primary - expected_outflow
                    if expected_primary is not None and expected_outflow is not None
                    else None
                ),
            }
            if not expected.get("membership_executable") or any(
                value is None for value in expected_values.values()
            ):
                membership_issues.append(
                    {
                        "spec_id": spec_id,
                        "issue": "locked_membership_replay_not_executable",
                        "replay": expected,
                    }
                )
                continue

            reported_for_spec = rows_by_id.get(spec_id, [])
            for field, field_aliases in aliases.items():
                claims = {
                    value
                    for row in reported_for_spec
                    for alias in field_aliases
                    if (value := _nonnegative_integral_value(row.get(alias)))
                    is not None
                }
                if not claims:
                    membership_issues.append(
                        {
                            "spec_id": spec_id,
                            "issue": "missing_membership_field",
                            "field": field,
                            "accepted_aliases": list(field_aliases),
                        }
                    )
                elif claims != {expected_values[field]}:
                    membership_issues.append(
                        {
                            "spec_id": spec_id,
                            "issue": "membership_value_mismatch",
                            "field": field,
                            "expected": expected_values[field],
                            "reported": sorted(claims),
                        }
                    )

        if membership_issues:
            findings.append(
                ValidationFinding(
                    validator="robustness_cohort_membership",
                    severity="error",
                    message=(
                        "Cohort-axis robustness rows must match deterministic "
                        "replay of their plan-locked predicates on "
                        "EASYICU_UNIVERSE_PARQUET, including retained N, overlap, "
                        "entries, and exits. "
                        f"Issues={membership_issues}."
                    ),
                    detail={
                        **step_detail,
                        "universe_path": str(universe_path),
                        "cohort_path": str(cohort_path),
                        "cohort_spec_ids": sorted(
                            spec.spec_id for spec in cohort_specs
                        ),
                        "issues": membership_issues,
                    },
                )
            )
    return findings


# Max directed full-replans fired when a model/estimation step self-blocks on a
# task-viable cohort. Two attempts give the replanner a fair chance to honour
# the override directive; beyond that the run falls back to an honest
# diagnostic_only rather than burning the replanner on a stuck plan.
_MAX_DIRECTED_MODEL_REPLANS = 2


def _contract_repair_log(
    findings: Sequence[ValidationFinding],
) -> str:
    """Serialize contract failures without discarding machine issue details.

    Coder repair only retains the tail of its run log.  Keep this compact JSON
    payload at the end of the repair request so model ids, allowed values, and
    expected/reported values survive that truncation.
    """

    return json.dumps(
        [
            {
                "validator": finding.validator,
                "severity": finding.severity,
                "message": finding.message,
                "detail": finding.detail,
            }
            for finding in findings
        ],
        ensure_ascii=False,
        default=str,
        separators=(",", ":"),
    )


def _visual_repair_request_log(
    findings: Sequence[ValidationFinding],
) -> str:
    """Keep visual-repair scope and structured collision details together."""

    payload = json.dumps(
        [
            {
                "validator": finding.validator,
                "severity": finding.severity,
                "message": finding.message,
                "detail": finding.detail,
            }
            for finding in findings
        ],
        ensure_ascii=False,
        default=str,
        separators=(",", ":"),
    )
    return (
        "LAYOUT-ONLY REPAIR BOUNDARY:\n"
        "- Preserve every source-data CSV value and row.\n"
        "- Preserve all numeric/statistical values in step_summary.json.\n"
        "- Preserve the figure contract's claims, evidence links, and panel roles.\n"
        "- Do not change source resolution, cohort/data transformations, estimates, "
        "or scientific labels.\n"
        "- Change only plotting/layout code needed to remove the reported collision; "
        "regenerate every declared figure format from the same data.\n\n"
        "STRUCTURED VISUAL FINDINGS (authoritative):\n" + payload
    )


def _is_terminal_publication_figure_repair_step(step: Any) -> bool:
    """Return true for rendering-only terminal publication figure repair steps."""

    expected_outputs = getattr(step, "expected_outputs", None) or []
    method = re.sub(
        r"[^a-z0-9]+",
        "_",
        str(getattr(step, "method", "") or "").strip().lower(),
    ).strip("_")
    rendering_methods = {
        "publication_figure_generation",
        "publication_figure_repair",
        "rendering_only_repair_from_primary_results",
    }
    if method not in rendering_methods or not expected_outputs:
        return False
    return all(_output_declares_figure(str(output)) for output in expected_outputs)


def _publication_bundle_has_primary_result_roles(outputs_dir: Path) -> bool:
    """Check whether an output directory already has a primary-result figure bundle."""

    contract_path = outputs_dir / "publication_figure.figure_contract.json"
    if not contract_path.exists():
        return False
    try:
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    panels = contract.get("panels") if isinstance(contract, Mapping) else None
    if not isinstance(panels, list):
        return False
    roles = {
        str(panel.get("role") or "").strip()
        for panel in panels
        if isinstance(panel, Mapping)
    }
    if not {"descriptive_result", "primary_estimand"}.issubset(roles):
        return False

    export_formats = contract.get("export_formats")
    if not isinstance(export_formats, list) or not export_formats:
        export_formats = ["svg", "png", "pdf", "tiff"]
    if not any(
        (outputs_dir / f"publication_figure.{str(ext).lstrip('.')}").exists()
        for ext in export_formats
    ):
        return False

    source_data = contract.get("source_data")
    if isinstance(source_data, list):
        source_paths = [
            outputs_dir / str(name)
            for name in source_data
            if isinstance(name, str) and Path(name).suffix
        ]
        if source_paths and not all(path.exists() for path in source_paths):
            return False
    return True


def _terminal_publication_repair_replan_skip_detail(
    *,
    plan: Any,
    completed_records: Optional[Sequence[Dict[str, Any]]],
    run_dir: Path,
) -> Optional[Dict[str, Any]]:
    """Return a skip reason when replanning would only delay deterministic repairs."""

    current_records = current_step_records(completed_records or [])
    completed_ok = {
        str(record.get("step_id") or "")
        for record in current_records
        if record.get("status") == "ok" and record.get("step_id")
    }
    remaining_steps = [
        step
        for step in getattr(plan, "steps", []) or []
        if str(getattr(step, "step_id", "") or "") not in completed_ok
    ]
    if not remaining_steps:
        return None
    if not all(
        _is_terminal_publication_figure_repair_step(step) for step in remaining_steps
    ):
        return None

    for record in reversed(current_records):
        if record.get("status") != "ok" or not record.get("step_id"):
            continue
        step_id = str(record["step_id"])
        outputs_dir = run_dir / "steps" / step_id / "outputs"
        if _publication_bundle_has_primary_result_roles(outputs_dir):
            return {
                "remaining_step_ids": [
                    str(getattr(step, "step_id", "") or "") for step in remaining_steps
                ],
                "satisfied_by_step_id": step_id,
                "satisfied_by_outputs_dir": str(outputs_dir),
            }
    return None


def _detached_figure_repair_binding(
    *,
    step: AnalysisStep,
    plan: AnalysisPlan,
    completed_records: Sequence[Mapping[str, Any]],
) -> Optional[Tuple[str, str, List[str]]]:
    """Bind a detached rendering-only repair to one failed figure target.

    The binding is orchestrator-owned: it comes from the current plan and
    latest outer step ledger, never from the renderer's self-reported
    ``parent_step`` text. Ambiguous repairs remain unbound and therefore cannot
    receive execution credit.
    """

    if not _is_terminal_publication_figure_repair_step(step):
        return None
    latest = {
        str(record.get("step_id") or ""): record
        for record in current_step_records(completed_records)
    }
    plan_steps = {
        str(candidate.step_id or ""): candidate for candidate in plan.steps or []
    }
    declared_step_inputs = {
        str(value or "").strip()
        for value in (step.inputs or [])
        if str(value or "").strip() in plan_steps
    }
    candidates: List[Tuple[str, str, List[str]]] = []
    for target_step_id, target_step in plan_steps.items():
        if target_step_id == str(step.step_id or ""):
            continue
        target_record = latest.get(target_step_id)
        target_status = str((target_record or {}).get("status") or "").strip().lower()
        if target_record is None or target_status not in {
            "execution_failed",
            "contract_failed",
            "repair_failed",
        }:
            continue
        if not _step_has_figure_only_output_contract(target_step):
            continue
        source_step_id = _parent_step_id_for_figure_step(target_step)
        if source_step_id is None:
            continue
        source_record = latest.get(source_step_id)
        if (
            source_record is None
            or str(source_record.get("status") or "").strip().lower() != "ok"
        ):
            continue
        if declared_step_inputs and not (
            {target_step_id, source_step_id} & declared_step_inputs
        ):
            continue
        source_evidence_ids = [
            str(evidence_id)
            for evidence_id in (source_record.get("evidence_ids") or [])
            if str(evidence_id).strip()
        ]
        if not source_evidence_ids:
            continue
        candidates.append((target_step_id, source_step_id, source_evidence_ids))
    if len(candidates) != 1:
        return None
    return candidates[0]


def _should_attempt_detached_figure_binding(
    *, out_dir: Path, sealed_renderer_authorized_code_sha256: Optional[str]
) -> bool:
    """Detached rescue lineage must never rewrite an authorized sealed summary."""

    return sealed_renderer_authorized_code_sha256 is None and _has_figure_exports(
        out_dir
    )


if TYPE_CHECKING:
    from .pipeline import ResearchAgentPipeline


def _sealed_renderer_source_digests(repair_id: str) -> Dict[str, str]:
    """Hash every repository module loaded by an exact sealed renderer."""

    if not is_sealed_renderer_repair(repair_id):
        raise ValueError(f"{repair_id!r} is not an exact sealed renderer")
    metadata = repair_metadata_for(repair_id)
    if not metadata.implementation_modules:
        raise ValueError(f"{repair_id!r} declares no implementation modules")
    digests: Dict[str, str] = {}
    for module_name in metadata.implementation_modules:
        module = importlib.import_module(module_name)
        module_file = getattr(module, "__file__", None)
        if not module_file:
            raise ValueError(f"Cannot locate implementation module {module_name!r}")
        digests[module_name] = sha256_of_file(Path(module_file))
    return dict(sorted(digests.items()))


def _sealed_renderer_implementation_digest(source_digests: Mapping[str, str]) -> str:
    """Return one stable authority digest for a renderer's source modules."""

    payload = json.dumps(
        dict(sorted(source_digests.items())),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256_of_bytes(payload)


def _sealed_parent_planner_anchors(
    *,
    run_dir: Path,
    figure_step_id: str,
) -> tuple[str, ...]:
    """Return only products and inputs from the host-recorded parent request.

    A physical filename or coder summary field can prove bytes and schema, but
    it cannot define the scientific subject that a Planner figure role claims.
    """

    request_step = _resolve_upstream_manifest_step(run_dir, figure_step_id)
    if not isinstance(request_step, Mapping):
        return ()
    anchors: list[str] = []
    for raw in request_step.get("expected_outputs") or []:
        parsed = _canonical_typed_product(raw)
        if parsed is not None:
            anchors.append(f"{parsed[0]}:{parsed[1]}")
    anchors.extend(
        str(raw).strip()
        for raw in (request_step.get("inputs") or [])
        if str(raw).strip()
    )
    return tuple(dict.fromkeys(anchors))


def _sealed_typed_figure_products(
    expected_outputs: Sequence[str],
) -> Optional[List[str]]:
    """Return unique typed figure roles, never legacy bare export filenames."""

    products = [
        str(product).strip() for product in expected_outputs if str(product).strip()
    ]
    typed_roles = [_canonical_typed_product(product) for product in products]
    if (
        not typed_roles
        or any(role is None or role[0] != "figure" for role in typed_roles)
        or len(typed_roles) != len(set(typed_roles))
    ):
        return None
    return products


_SEALED_AUTHORITY_SUMMARY_MARKERS = (
    "sealed_renderer_repair",
    "sealed_renderer_implementation_sha256",
    "sealed_renderer_parent_digests",
    "planner_bound_figure_products",
    "planner_product_slot_bindings",
    "planner_product_binding",
)


def _unowned_sealed_authority_markers(
    step_summary: Mapping[str, Any],
    *,
    authorized_code_sha256: Optional[str],
) -> List[str]:
    """Reject sealed provenance unless the host authorized it pre-execution."""

    if authorized_code_sha256 is not None:
        return []
    return [
        marker for marker in _SEALED_AUTHORITY_SUMMARY_MARKERS if marker in step_summary
    ]


_COSMETIC_VISUAL_REASON = "svg_text_overlap_spacing"
_LEGACY_COSMETIC_VISUAL_MESSAGE = re.compile(
    r"^svg figure '[^']+' has overlapping text elements; "
    r"multi-panel labels, annotations or axis text need more spacing\.?$",
    re.IGNORECASE,
)
_HARD_VISUAL_MESSAGE = re.compile(
    r"\b(?:blank|clip(?:ped|ping)?|crop(?:ped|ping)?|missing|absent|"
    r"unreadable|overflow|truncat(?:ed|ion)|numeric|mismatch|disagree)\b",
    re.IGNORECASE,
)


def _is_cosmetic_visual_finding(finding: ValidationFinding) -> bool:
    """Return true only for the closed SVG spacing reason safe to demote."""

    if finding.severity != "error" or finding.validator != "visual_qa":
        return False
    message = str(finding.message or "").strip()
    if _HARD_VISUAL_MESSAGE.search(message):
        return False
    detail = finding.detail if isinstance(finding.detail, Mapping) else {}
    if str(detail.get("reason") or "").strip() == _COSMETIC_VISUAL_REASON:
        return True
    # Preserve readiness for historical deterministic findings that predate
    # the reason enum, but require the complete canonical message rather than
    # two permissive substrings.
    return _LEGACY_COSMETIC_VISUAL_MESSAGE.fullmatch(message) is not None


def _demote_cosmetic_visual_findings(
    findings: Sequence[ValidationFinding],
) -> tuple[List[ValidationFinding], List[ValidationFinding]]:
    """Demote cosmetic visual errors and return remaining hard errors."""

    demoted: List[ValidationFinding] = []
    for finding in findings:
        if _is_cosmetic_visual_finding(finding):
            demoted.append(finding.model_copy(update={"severity": "warning"}))
        else:
            demoted.append(finding)
    blocking_errors = [f for f in demoted if f.severity == "error"]
    return demoted, blocking_errors


def _max_finding_severity(
    findings_for_step: Sequence[ValidationFinding],
) -> Optional[str]:
    """Return the strongest severity across findings (error > warning > info)."""
    if any(f.severity == "error" for f in findings_for_step):
        return "error"
    if any(f.severity == "warning" for f in findings_for_step):
        return "warning"
    if any(f.severity == "info" for f in findings_for_step):
        return "info"
    return None


def scope_findings_to_records(
    evidence_ids: Sequence[str],
    findings_for_step: Sequence[ValidationFinding],
) -> Dict[str, tuple[Optional[str], List[str]]]:
    """Map each step output record to the caveat that actually concerns it.

    A finding that names specific records (``finding.evidence_ids``) taints
    ONLY those records. A step-global finding — no evidence_ids, e.g. an
    "immortal-time-bias risk" or "cohort is keyed at the stay level"
    advisory — describes the ANALYSIS DESIGN, not any one artifact.
    Blanket-tainting every output record with a step-global WARNING made the
    primary result table uncitable and the manuscript unwinnable: one design
    advisory flags ``table_one`` / ``adjusted_association``, and the
    manifest-caveat gate then blocks any draft that cites them (which every
    real Results section must). Those advisories still live in the manifest
    findings list and reach the writer as limitations — they simply no longer
    masquerade as per-artifact taint.

    Step-global ERRORS keep the blanket behaviour (fail-closed: a step-level
    error means the step's outputs are not to be trusted).

    Returns ``{evidence_id: (severity_or_None, messages)}``.
    """
    targeted: Dict[str, List[ValidationFinding]] = {}
    for finding in findings_for_step:
        for eid in finding.evidence_ids or []:
            targeted.setdefault(str(eid), []).append(finding)

    global_error_findings = [
        f for f in findings_for_step if f.severity == "error" and not f.evidence_ids
    ]
    global_error_messages = [f.message for f in global_error_findings]

    scoped: Dict[str, tuple[Optional[str], List[str]]] = {}
    for evidence_id in evidence_ids:
        eid = str(evidence_id)
        relevant = targeted.get(eid, [])
        severity = _max_finding_severity(list(relevant) + global_error_findings)
        messages = [
            f.message for f in relevant if f.severity in {"warning", "error"}
        ] + global_error_messages
        scoped[eid] = (severity, messages)
    return scoped


def _filter_success_alias_bindings(
    bindings: Mapping[str, Sequence[str]],
    *,
    existing_aliases: Mapping[str, str],
    owners_by_evidence_id: Mapping[str, str],
    step_id: str,
    records_by_evidence_id: Optional[Mapping[str, Any]] = None,
) -> Tuple[Dict[str, List[str]], Dict[str, str], Set[str]]:
    """Keep cross-step product aliases on their existing authority.

    A figure step may legitimately repeat a parent analysis role such as
    ``primary_association``. Same-step retries may replace their own aliases,
    but a child must not steal the parent's semantic authority merely because
    both products mention that role.
    """

    filtered: Dict[str, List[str]] = {}
    retained: Dict[str, str] = {}
    for evidence_id, aliases in bindings.items():
        accepted: List[str] = []
        for alias in aliases:
            alias = str(alias).strip()
            if not alias:
                continue
            existing_id = str(existing_aliases.get(alias) or "").strip()
            existing_owner = str(owners_by_evidence_id.get(existing_id) or "").strip()
            if existing_id and existing_id != evidence_id and existing_owner != step_id:
                retained[alias] = existing_id
                continue
            accepted.append(alias)
        filtered[str(evidence_id)] = list(dict.fromkeys(accepted))
    records = records_by_evidence_id or {}

    def _record_source_name(evidence_id: str) -> str:
        record = records.get(evidence_id)
        relative_path = str(_evidence_record_field(record, "relative_path") or "")
        name = Path(relative_path).name
        prefix = f"{evidence_id}__"
        return name[len(prefix) :] if name.startswith(prefix) else name

    def _is_product_authority(evidence_id: str) -> bool:
        record = records.get(evidence_id)
        kind = str(_evidence_record_field(record, "kind") or "").lower()
        source_name = _record_source_name(evidence_id)
        if kind in {"table", "figure"}:
            return True
        return kind == "statistic" and source_name != "step_summary.json"

    implicit_basename_aliases = {
        evidence_id: Path(_record_source_name(evidence_id)).stem
        for evidence_id in filtered
        if _record_source_name(evidence_id)
    }
    alias_claimants: Dict[str, List[str]] = {}
    for evidence_id, aliases in filtered.items():
        for alias in aliases:
            alias_claimants.setdefault(alias, []).append(evidence_id)
        basename_alias = implicit_basename_aliases.get(evidence_id)
        if basename_alias:
            alias_claimants.setdefault(basename_alias, []).append(evidence_id)
    suppressed_basename_evidence_ids: Set[str] = set()
    for alias, claimants in alias_claimants.items():
        unique_claimants = list(dict.fromkeys(claimants))
        if len(unique_claimants) <= 1:
            continue
        product_claimants = [
            evidence_id
            for evidence_id in unique_claimants
            if _is_product_authority(evidence_id)
        ]
        selected_product: Optional[str] = None
        if len(product_claimants) == 1:
            selected_product = product_claimants[0]
        elif len(product_claimants) > 1:
            product_sources = {
                evidence_id: _record_source_name(evidence_id)
                for evidence_id in product_claimants
            }
            figure_claimants = [
                evidence_id
                for evidence_id in product_claimants
                if str(
                    _evidence_record_field(records.get(evidence_id), "kind") or ""
                ).lower()
                == "figure"
            ]
            stems = {
                Path(product_sources[evidence_id]).stem
                for evidence_id in figure_claimants
            }
            # PNG/SVG/PDF exports with one stem are formats of the same logical
            # figure, not competing scientific products. Prefer the editable
            # vector authority deterministically. Distinct real products keep
            # their duplicate claims so EvidenceStore fails closed.
            if len(figure_claimants) == len(product_claimants) and len(stems) == 1:
                format_rank = {
                    ".svg": 0,
                    ".pdf": 1,
                    ".png": 2,
                    ".tiff": 3,
                    ".tif": 3,
                }
                ranked = sorted(
                    product_claimants,
                    key=lambda evidence_id: (
                        format_rank.get(
                            Path(product_sources[evidence_id]).suffix.lower(), 99
                        ),
                        evidence_id,
                    ),
                )
                best_rank = format_rank.get(
                    Path(product_sources[ranked[0]]).suffix.lower(), 99
                )
                if (
                    sum(
                        format_rank.get(
                            Path(product_sources[evidence_id]).suffix.lower(), 99
                        )
                        == best_rank
                        for evidence_id in ranked
                    )
                    == 1
                ):
                    selected_product = ranked[0]
        if selected_product is None:
            continue
        for evidence_id in unique_claimants:
            if evidence_id != selected_product:
                filtered[evidence_id] = [
                    candidate
                    for candidate in filtered[evidence_id]
                    if candidate != alias
                ]
                if implicit_basename_aliases.get(evidence_id) == alias:
                    suppressed_basename_evidence_ids.add(evidence_id)
    return filtered, retained, suppressed_basename_evidence_ids


def _reader_label_from_stem(stem: str) -> str:
    words = [
        token for token in stem.replace("-", "_").replace(".", "_").split("_") if token
    ]
    if not words:
        return "Manuscript figure"
    return " ".join(
        word.capitalize() if len(word) > 3 else word.upper() for word in words
    )


def _infer_step_figure_panel_role(step: AnalysisStep, stem: str) -> str:
    text = " ".join(
        [
            step.step_id,
            step.intent or "",
            step.method or "",
            stem,
            " ".join(step.expected_outputs or []),
        ]
    ).lower()
    if any(token in text for token in ("robustness", "sensitivity", "specification")):
        return "robustness"
    if any(
        token in text
        for token in (
            "missingness",
            "measurement",
            "quality",
            "baseline",
            "table one",
            "attrition",
            "cohort",
            "audit",
        )
    ):
        return "audit"
    if any(
        token in text
        for token in ("association", "effect", "forest", "estimate", "outcome")
    ):
        return "relationship"
    return "overview"


def _step_summary_paths(
    value: Any,
    *,
    out_dir: Path,
    allowed_suffixes: Optional[set[str]] = None,
) -> List[Path]:
    raw_values: List[Any] = []
    if isinstance(value, (str, Path)):
        raw_values = [value]
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        raw_values = list(value)
    paths: List[Path] = []
    for raw in raw_values:
        path = Path(str(raw))
        if not path.is_absolute():
            path = out_dir / path
        if not path.exists() or not path.is_file():
            continue
        if allowed_suffixes is not None and path.suffix.lower() not in allowed_suffixes:
            continue
        paths.append(path)
    return sorted(dict.fromkeys(paths))


def _ensure_step_figure_contract(
    *,
    step: AnalysisStep,
    out_dir: Path,
    step_summary: Mapping[str, Any],
    evidence_ids: Sequence[str],
) -> Optional[Path]:
    """Create a minimal manuscript-facing contract for valid figure exports.

    Coder prompts already ask for ``*.figure_contract.json``. This runner-level
    fallback covers the common successful-plot / missing-boilerplate case without
    weakening result-bearing figure gates: association and robustness figures
    still keep their result-like roles, so the contract validator can require
    multi-panel evidence when appropriate.
    """

    if sorted(out_dir.glob("*.figure_contract.json")):
        return None
    figure_suffixes = {".svg", ".pdf", ".png", ".tiff", ".tif", ".pptx"}
    figure_paths = _step_summary_paths(
        step_summary.get("figure_files") or step_summary.get("figure_path"),
        out_dir=out_dir,
        allowed_suffixes=figure_suffixes,
    )
    if not figure_paths:
        figure_paths = sorted(
            path
            for path in out_dir.iterdir()
            if path.is_file() and path.suffix.lower() in figure_suffixes
        )
    if not figure_paths:
        return None
    source_paths = _step_summary_paths(
        step_summary.get("source_data_files")
        or step_summary.get("source_data")
        or step_summary.get("source_table"),
        out_dir=out_dir,
    )
    primary_stem = figure_paths[0].stem
    label = _reader_label_from_stem(primary_stem)
    role = _infer_step_figure_panel_role(step, primary_stem)
    contract = make_figure_contract(
        figure_id=primary_stem,
        core_claim=(
            f"{label} summarizes the planned manuscript figure from registered "
            "source data."
        ),
        panels=[
            {
                "panel_id": "A",
                "title": label,
                "role": role,
                "claim": (
                    "This panel displays the step result using registered "
                    "source data and preserved code provenance."
                ),
                "evidence_ids": list(evidence_ids),
                "review_risk": (
                    "Review the source data and upstream step contract before "
                    "using the panel in manuscript text."
                ),
            }
        ],
        export_formats=[
            suffix.lstrip(".")
            for suffix in (".svg", ".pdf", ".png", ".tiff")
            if any(path.suffix.lower() == suffix for path in figure_paths)
        ]
        or ["svg", "png"],
        source_data=[path.name for path in source_paths],
        statistics_note="Auto-generated by the runner from step summary metadata.",
        image_integrity_note="No values were invented or visually altered by this contract synthesis.",
    )
    contract_path = out_dir / f"{primary_stem}.figure_contract.json"
    contract_path.write_text(
        json.dumps(contract.model_dump(mode="json"), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return contract_path


def _load_step_summary_from_outputs(out_dir: Path) -> Dict[str, Any]:
    """Load the current staged summary without granting it evidence authority."""

    summary_path = out_dir / "step_summary.json"
    if not summary_path.exists():
        return {}
    try:
        loaded = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception:
        loaded = None
    return loaded if isinstance(loaded, dict) else {"raw": loaded}


def _write_host_input_binding_receipts(
    *,
    out_dir: Path,
    step_summary: Mapping[str, Any],
    resolved_input_bindings: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Any]:
    """Seal exact input receipts for a host-owned deterministic renderer.

    A sealed renderer consumes only the host-resolved artifacts authorized by
    its parent digest seal.  Generated code must normally report its own
    receipts, but asking a host-owned renderer to manufacture those receipts is
    both redundant and weaker than recording them here from the authority
    bindings.  An unreadable table is deliberately omitted so the downstream
    integrity validator fails closed on incomplete coverage.
    """

    receipts: List[Dict[str, Any]] = []
    for input_key, raw_binding in sorted(resolved_input_bindings.items()):
        if not isinstance(raw_binding, Mapping):
            continue
        binding = dict(raw_binding)
        path = Path(str(binding.get("absolute_path") or ""))
        receipt: Dict[str, Any] = {
            "input_key": str(input_key),
            "loaded": True,
        }
        for field in ("evidence_id", "sha256"):
            value = binding.get(field)
            if value is not None:
                receipt[field] = value
        if StepSummaryIntegrityValidator._is_tabular_binding(binding):
            try:
                receipt["row_count"] = StepSummaryIntegrityValidator._table_row_count(
                    path
                )
            except Exception:
                continue
        receipts.append(receipt)

    updated = dict(step_summary)
    updated["input_bindings"] = receipts
    summary_path = out_dir / "step_summary.json"
    temporary_path = summary_path.with_suffix(".json.tmp")
    temporary_path.write_text(
        json.dumps(updated, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(summary_path)
    return updated


def _figure_contract_source_data_canonicalization_candidate(
    *,
    contract_path: Path,
    out_dir: Path,
) -> Optional[Tuple[str, str, List[str]]]:
    """Return an exact legacy-descriptor -> flat-basename JSON rewrite.

    ``make_figure_contract`` accepts small path mappings as an in-memory input
    compatibility layer but persists canonical ``List[str]`` source data.
    Some legacy agent scripts wrote those mappings directly to JSON.  This
    representation-only migration is deliberately strict: every populated path
    alias must agree, every source must be an existing ordinary local CSV in
    the exact step output directory, and non-empty evidence references are not
    discarded.  Anything else is left untouched for the validator to block.
    """

    output_root = Path(out_dir).resolve()
    candidate_path = Path(contract_path)
    try:
        if (
            candidate_path.parent.resolve() != output_root
            or candidate_path.resolve(strict=True).parent != output_root
            or not candidate_path.is_file()
            or candidate_path.is_symlink()
            or candidate_path.stat().st_nlink != 1
        ):
            return None
        before = candidate_path.read_text(encoding="utf-8")
        payload = json.loads(before)
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None

    raw_sources = payload.get("source_data")
    if isinstance(raw_sources, Mapping):
        source_items: List[Any] = [raw_sources]
    elif isinstance(raw_sources, list):
        source_items = list(raw_sources)
    else:
        return None
    if not source_items or not any(isinstance(item, Mapping) for item in source_items):
        return None

    path_keys = ("file", "filename", "path", "relative_path")
    canonical_names: List[str] = []
    for item in source_items:
        if isinstance(item, str):
            source_name = item.strip()
        elif isinstance(item, Mapping):
            if item.get("evidence_id") not in (None, "") or item.get(
                "evidence_ids"
            ) not in (None, "", []):
                return None
            populated: List[str] = []
            for key in path_keys:
                value = item.get(key)
                if value in (None, ""):
                    continue
                if not isinstance(value, str) or not value.strip():
                    return None
                populated.append(value.strip())
            if len(set(populated)) != 1:
                return None
            source_name = populated[0]
        else:
            return None
        if (
            not source_name
            or Path(source_name).name != source_name
            or "/" in source_name
            or "\\" in source_name
            or Path(source_name).suffix.lower() != ".csv"
        ):
            return None
        source_path = output_root / source_name
        try:
            if (
                source_path.resolve(strict=True).parent != output_root
                or not source_path.is_file()
                or source_path.is_symlink()
                or source_path.stat().st_nlink != 1
            ):
                return None
        except OSError:
            return None
        canonical_names.append(source_name)

    canonical_payload = dict(payload)
    canonical_payload["source_data"] = canonical_names
    after = json.dumps(canonical_payload, indent=2, ensure_ascii=False) + "\n"
    if before == after:
        return None
    return before, after, canonical_names


def _install_figure_contract_source_data_canonicalization(
    *,
    contract_path: Path,
    expected_before: str,
    canonical_text: str,
) -> None:
    """Atomically install one pre-authorized contract-schema rewrite.

    The generated step controls its output directory, so a predictable temp
    path is unsafe: it could be pre-created as a symlink before the host writes.
    ``mkstemp`` gives us an exclusive random regular file.  The destination is
    also reopened without following symlinks and must still match the exact
    content reviewed by the authorization boundary.
    """

    contract_path = Path(contract_path)
    parent = contract_path.parent
    read_flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    contract_fd = os.open(contract_path, read_flags)
    try:
        opened_stat = os.fstat(contract_fd)
        if not stat.S_ISREG(opened_stat.st_mode) or opened_stat.st_nlink != 1:
            raise ValueError("figure contract must remain one ordinary file")
        with os.fdopen(contract_fd, "r", encoding="utf-8") as handle:
            contract_fd = -1
            observed_before = handle.read()
        if observed_before != expected_before:
            raise ValueError("figure contract changed after canonicalization review")

        temporary_fd, temporary_name = tempfile.mkstemp(
            prefix=f".{contract_path.name}.",
            suffix=".schema.tmp",
            dir=parent,
        )
        temporary_path = Path(temporary_name)
        try:
            with os.fdopen(temporary_fd, "w", encoding="utf-8") as handle:
                handle.write(canonical_text)
                handle.flush()
                os.fsync(handle.fileno())
            current_stat = os.stat(contract_path, follow_symlinks=False)
            if (
                not stat.S_ISREG(current_stat.st_mode)
                or current_stat.st_nlink != 1
                or current_stat.st_dev != opened_stat.st_dev
                or current_stat.st_ino != opened_stat.st_ino
            ):
                raise ValueError("figure contract identity changed before replace")
            os.replace(temporary_path, contract_path)
            try:
                directory_fd = os.open(
                    parent,
                    os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
                )
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
            except OSError:
                pass
        finally:
            temporary_path.unlink(missing_ok=True)
    finally:
        if contract_fd >= 0:
            os.close(contract_fd)


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


def build_self_block_replan_directive(
    *,
    failed_step: AnalysisStep,
    failed_record: Mapping[str, Any],
    completed_records: Sequence[Mapping[str, Any]],
    viability: "CohortViability",
) -> Optional[str]:
    """Return a viability-conditioned override directive when a model/estimation
    step self-blocked on a task-viable cohort, else ``None``.

    Pure and deterministic so the trigger logic is unit-testable without a run.
    Fires only when ALL hold: the failed step's contract requires model
    performance statistics (``statistic:auroc`` / ``statistic:brier_score``); the
    cohort cleared the viability floor; and a deliberate block signal is present
    on the failed step or an upstream completed step (e.g. a
    ``modeling_block_registration`` step). Stays silent otherwise — a genuinely
    non-viable cohort or a hard crash leaves blocking legitimate.

    Impartiality: the directive is conditioned on viability twice over — the
    trigger requires ``viability.viable`` and the directive text itself reaffirms
    that blocking stays legitimate on genuinely non-viable data. It never
    dictates which model to fit, only that a model must actually be fit.
    """
    if not step_requires_model_performance(failed_step.expected_outputs):
        return None
    if not viability.viable:
        return None
    block_reason = step_summary_block_signal(failed_record.get("step_summary") or {})
    if not block_reason:
        for rec in completed_records:
            if not isinstance(rec, Mapping):
                continue
            block_reason = step_summary_block_signal(rec.get("step_summary") or {})
            if block_reason:
                break
    if not block_reason:
        return None
    return (
        "The locked analysis cohort is task-viable (" + viability.note + "), yet "
        "the modeling step recorded a non-execution/blocked status "
        f'("{block_reason}") and produced no model and no required performance '
        "statistics (AUROC / Brier). On a cohort this populated, declaring the "
        "repaired artifacts unusable, registering a modeling block, or emitting a "
        "non-execution model stub is NOT an acceptable outcome for this task. "
        "Revise the remaining plan so the primary modeling step actually fits a "
        "model on the available predictors and emits the required performance "
        "statistics. Do NOT re-insert any step whose purpose is to gate, block, "
        "or declare the modeling unexecutable on this cohort. (Blocking would be "
        "legitimate only if "
        "the data were genuinely non-viable — too few rows, no outcome variation, "
        "or no usable predictors — which is not the case here.)"
    )


# No deterministic runner owns a primary scientific estimand.  Kept as an
# explicit empty compatibility surface for drift checks and legacy run records.
_PRIMARY_DETERMINISTIC_RUNNERS: set[str] = set()

# Method names the planner uses for a PRIMARY estimation step (not a
# prep/audit/figure step). A dose-response is routed to the ordinal runner only
# when a dose-response signal is ALSO present, so listing broad association
# methods here does not hijack a plain association step.
_ORDINAL_PRIMARY_METHODS = frozenset(
    {
        "dose_response",
        "dose_response_analysis",
        "ordinal_regression",
        "ordinal_logistic_regression",
        "trend_analysis",
        "association",
        "association_analysis",
        "stratified_analysis",
        "subgroup_analysis",
        "regression",
        "logistic_regression",
        "glm",
        "modeling",
        "model",
        "estimation",
        "ordinal",
    }
)
# Methods that are UNAMBIGUOUSLY a dose-response primary on their own.
_ORDINAL_EXPLICIT_METHODS = frozenset(
    {
        "dose_response",
        "dose_response_analysis",
    }
)
# General dose-response / graded-exposure vocabulary (case-neutral: never a
# specific score name). Present in the question, intent, or declared outputs.
_ORDINAL_OUTPUT_PRODUCTS = frozenset(
    {
        "dose_response",
        "per_stage",
        "per_stage_odds",
        "per_stage_odds_ratio",
        "per_stage_odds_ratios",
        "trend_or",
        "ordinal_trend",
        "ordinal_trend_model",
    }
)

# --- Cohort-definition-sensitivity routing (precise, not blunt keyword) -------
# A cohort-definition-sensitivity step VARIES the cohort/eligibility definition
# and compares the result across alternative definitions. The authoritative
# signal is the planner's own ``method`` key; the historical blunt test --
# ``"sensitivity" in blob and ("cohort"|"definition" in blob)`` -- false-positives
# on a primary estimand step that merely mentions a pre-specified within-cohort
# sensitivity sub-analysis. Require an alternative-definition signal instead.
_COHORT_DEF_SENSITIVITY_METHODS = frozenset(
    {
        "cohort_definition_sensitivity",
        "cohort_sensitivity",
        "definition_sensitivity",
    }
)
_COHORT_DEF_SENSITIVITY_OUTPUT_TOKENS = (
    "alternative_cohort_attrition",
    "cohort_overlap",
    "overlap_and_movement_across_cohorts",
    "sensitivity_grid",
    # Not "sensitivity_comparison": it substring-matches within-cohort comparison
    # outputs. Each kept token uniquely signals an across-definition comparison.
    "definition_sensitivity",
    "sensitivity_definition_summary",
    "outcome_by_definition",
    "adjustment_denominator_sensitivity",
)

_PRIMARY_COHORT_FLOW_METHODS = frozenset(
    {
        "cohort_construction",
        "cohort_definition",
        "eligibility_definition",
    }
)
_PRIMARY_COHORT_FLOW_OUTPUTS = frozenset(
    {
        "cohort_attrition",
        "cohort_denominator",
        "cohort_denominators",
        "cohort_flow",
        "attrition_by_rule",
        "eligibility_flow",
    }
)

_EFFECT_ASSOCIATION_METHOD_TOKENS = frozenset(
    {
        "association",
        "causal",
        "cox",
        "effect",
        "estimand",
        "hazard",
        "logistic",
        "logit",
        "mixed",
        "model",
        "prediction",
        "regression",
        "survival",
    }
)
_EFFECT_OUTPUT_FRAGMENTS = (
    "adjusted_effect",
    "association_estimate",
    "coefficient",
    "odds_ratio",
    "hazard_ratio",
    "risk_ratio",
    "risk_difference",
    "primary_estimate",
    "primary_or",
    "primary_hr",
    "c_statistic",
    "c_index",
    "auroc",
    "cox_summary",
)

_AUXILIARY_OUTPUT_KINDS = frozenset({"table", "statistic", "log"})


def _closed_auxiliary_output_products(
    expected_outputs: Sequence[str],
    *,
    supported_products: set[str] | frozenset[str],
) -> Optional[set[str]]:
    """Return all declared products only when one auxiliary owns all of them.

    Every non-empty output participates in the closed-contract decision,
    including bare filenames.  Unsupported artifact kinds and even one foreign
    product return ``None`` so a compact runner cannot silently ignore the rest
    of a mixed agent step.
    """

    products: set[str] = set()
    for raw in expected_outputs or []:
        value = str(raw or "").strip().lower()
        if not value:
            continue
        kind, separator, _product = value.partition(":")
        if separator and kind not in _AUXILIARY_OUTPUT_KINDS:
            return None
        normalized = _normalised_expected_output_names([value])
        if len(normalized) != 1:
            return None
        products.update(normalized)
    if not products or not products.issubset(set(supported_products)):
        return None
    return products


def _method_head(method: str) -> str:
    """Return the scientific owner of a ``<head>_with_<rider>`` method."""

    normalized = re.sub(r"[^a-z0-9]+", "_", str(method or "").strip().lower()).strip(
        "_"
    )
    return normalized.split("_with_", 1)[0]


def _method_is_effect_or_association(method: str) -> bool:
    head = _method_head(method)
    tokens = set(filter(None, re.split(r"[_\-\s]+", head)))
    return bool(tokens & _EFFECT_ASSOCIATION_METHOD_TOKENS)


def _declares_effect_output(expected_outputs: Sequence[str]) -> bool:
    """True for structured primary-effect/model outputs, including OR/HR."""

    for output in expected_outputs or []:
        value = str(output or "").strip().lower()
        if any(fragment in value for fragment in _EFFECT_OUTPUT_FRAGMENTS):
            return True
        tokens = set(re.findall(r"[a-z0-9]+", value))
        if tokens & {"or", "hr", "auc"}:
            return True
    return False


def _primary_cohort_flow_runner_owns_step(
    method: str,
    step_id: str,
    intent: str,
    expected_outputs: Sequence[str],
) -> bool:
    """True for the owner that defines the single locked primary cohort.

    Alternative-definition/overlap/sensitivity steps are deliberately excluded;
    those have separate deterministic runners.  The owner must declare an
    attrition/denominator output, so a generic preparation step is not hijacked.
    """

    del step_id, intent
    method_normalized = str(method or "").lower()
    expected_names = _closed_auxiliary_output_products(
        expected_outputs,
        supported_products=_PRIMARY_COHORT_FLOW_OUTPUTS,
    )
    if expected_names is None:
        return False
    method_head = _method_head(method_normalized)
    if _method_is_effect_or_association(method_head) or _declares_effect_output(
        expected_outputs
    ):
        return False
    return method_head in _PRIMARY_COHORT_FLOW_METHODS


# The compact missingness runner owns per-concept measurement counts only.  A
# richer exposure/source repair must retain the coder path until a runner that
# actually owns all of these contracts exists.
_RICH_EXPOSURE_AUDIT_OUTPUT_TOKENS = (
    "exposure_distribution",
    "joint_availability",
    "complete_case_attrition",
    "score_level_distribution",
    "score_completeness",
    "invalid_range",
    "model_availability",
    "source_reconciliation",
)

_COMPACT_MISSINGNESS_SUPPORTED_OUTPUTS = frozenset(
    {
        "missingness_audit",
        "missingness_measurement_audit",
        "measurement_audit",
        "measurement_process_audit",
        "data_quality_audit",
        "source_coverage",
        "cohort_flow",
        "analytic_denominator",
        "analytic_denominators",
    }
)
_COMPACT_MISSINGNESS_METHODS = frozenset(
    {
        "missingness_audit",
        "missingness",
        "measurement_audit",
        "measurement_process_audit",
        "data_quality_audit",
        "data_quality",
    }
)
_ABSOLUTE_RISK_CONTEXT_METHODS = frozenset(
    {
        "absolute_risk_context",
        "descriptive_context",
        "exposure_outcome_summary",
    }
)
_ROBUSTNESS_SENSITIVITY_METHODS = frozenset(
    {
        "prespecified_robustness",
        "robustness_sensitivity",
        "sensitivity_comparison",
    }
)


# An ordinal *trend test* can be a purely descriptive result.  The primary
# dose-response runner fits an adjusted model, so it may only claim a broadly
# named ordinal/association step when the declared contract or intent actually
# asks for a model/effect estimate.  This keeps exposure derivation/QC and
# stage-stratified descriptive steps with their own owners.
def _is_cohort_definition_sensitivity_step(
    method: str,
    step_id: str,
    intent: str,
    expected_outputs: Sequence[str],
) -> bool:
    """Pure routing test: is this an ACTUAL cohort-definition-sensitivity step?

    Require an exact method head plus a closed comparison product, or a pair of
    closed across-definition products. Step ids and prose never establish the
    role. This keeps ordinary within-cohort sensitivity language from vetoing a
    legitimate primary estimand step.
    """
    del step_id, intent
    head = _method_head(str(method or "").lower())
    expected_names = _normalised_expected_output_names(expected_outputs)
    matched_outputs = expected_names & set(_COHORT_DEF_SENSITIVITY_OUTPUT_TOKENS)
    return head in _COHORT_DEF_SENSITIVITY_METHODS and bool(matched_outputs)


def _cohort_definition_sensitivity_runner_owns_step(
    method: str,
    step_id: str,
    intent: str,
    expected_outputs: Sequence[str],
) -> bool:
    """Legacy comparator code is explicit-only and never a preflight owner.

    The historical script reconstructed cohorts, chose covariates, and refit a
    GLM.  Those are scientific decisions, so no method/output combination may
    automatically replace the coder with that script.
    """

    del method, step_id, intent, expected_outputs
    return False


def _cohort_definition_overlap_runner_owns_step(
    method: str,
    expected_outputs: Sequence[str],
) -> bool:
    """Legacy cohort-construction code is explicit-only, never automatic."""

    del method, expected_outputs
    return False


def _simple_missingness_audit_runner_owns_step(
    method: str,
    step_id: str,
    intent: str,
    expected_outputs: Sequence[str],
) -> bool:
    """True when the compact per-concept missingness runner owns the contract."""

    if _normalised_expected_output_names(expected_outputs) & set(
        _RICH_EXPOSURE_AUDIT_OUTPUT_TOKENS
    ):
        return False
    declared_names = _closed_auxiliary_output_products(
        expected_outputs,
        supported_products=_COMPACT_MISSINGNESS_SUPPORTED_OUTPUTS,
    )
    if declared_names is None:
        # A method label such as ``data_quality_audit`` is not sufficient
        # ownership.  If even one declared artefact belongs to a different
        # contract (e.g. representation reconciliation), leave the step to its
        # coder instead of returning a successful but irrelevant compact audit.
        return False

    method_head = _method_head(method)
    if method_head not in _COMPACT_MISSINGNESS_METHODS:
        return False
    if _declares_effect_output(expected_outputs):
        return False
    return True


def _absolute_risk_context_runner_owns_step(
    method: str,
    step_id: str,
    expected_outputs: Sequence[str],
) -> bool:
    """True for a descriptive exposure-prevalence / absolute-risk owner."""

    del step_id
    outputs = {str(item or "").lower() for item in (expected_outputs or [])}
    if any(item.startswith("figure:") for item in outputs):
        return False
    supported_products = {
        "exposure_outcome_summary",
        "exposure_prevalence_and_absolute_risk",
        "absolute_risk",
        "absolute_risk_context",
    }
    if _method_head(method) not in _ABSOLUTE_RISK_CONTEXT_METHODS:
        return False
    if _method_is_effect_or_association(method) or _declares_effect_output(
        expected_outputs
    ):
        return False
    structured_products = _closed_auxiliary_output_products(
        expected_outputs,
        supported_products=supported_products,
    )
    if structured_products is not None:
        return True
    # A reconciliation/audit step may mention absolute-risk context while
    # owning different artefacts (representation reconciliation, gap notes,
    # etc.).  The compact runner must not claim such a step merely because its
    # id contains ``absolute_risk_context``; it only owns the closed output
    # contract above.
    return False


def _robustness_sensitivity_runner_owns_step(
    method: str,
    step_id: str,
    expected_outputs: Sequence[str],
) -> bool:
    """True for a separate prespecified robustness-comparison owner."""

    del step_id
    outputs = {str(item or "").lower() for item in (expected_outputs or [])}
    if any(item.startswith("figure:") for item in outputs):
        return False
    method_head = _method_head(method)
    if method_head not in _ROBUSTNESS_SENSITIVITY_METHODS:
        return False
    supported_products = {
        "robustness_matrix",
        "robustness_summary",
        "complete_case_n",
    }
    structured_products = _closed_auxiliary_output_products(
        expected_outputs,
        supported_products=supported_products,
    )
    if structured_products is None:
        return False
    has_matrix = "robustness_matrix" in structured_products
    has_summary_contract = {
        "robustness_summary",
        "complete_case_n",
    }.issubset(structured_products)
    return has_matrix or has_summary_contract


def _method_has_ordinal_primary_token(method: str) -> bool:
    """True if ``method`` IS, or is a compound built from, a primary-estimation
    method token (e.g. ``multivariable_association`` -> ``association``,
    ``adjusted_logistic_regression`` -> ``regression``).

    Word-boundary token match (split on ``_`` / ``-``), NOT substring, so
    ``remodeling`` never matches ``model``. This is only ever reached after the
    closed ordinal-product gate in :func:`_ordinal_dose_response_step_matches`;
    a plain association label cannot establish ownership on its own.
    """
    if method in _ORDINAL_PRIMARY_METHODS:
        return True
    tokens = method.replace("-", "_").split("_")
    return any(tok in _ORDINAL_PRIMARY_METHODS for tok in tokens)


def _ordinal_dose_response_step_matches(
    method: str, blob: str, expected_blob: str
) -> bool:
    """Pure routing test: is this the PRIMARY dose-response estimation step?

    This legacy compatibility predicate is unit-testable without a full run. The
    caller supplies lowercased strings and has already excluded figure and
    cohort-definition-sensitivity steps.

    ``blob`` = step_id + intent + research_question + expected_outputs;
    ``expected_blob`` = expected_outputs only.
    """
    del blob
    head = _method_head(method)
    products = _normalised_structured_output_names(expected_blob)
    if not products.intersection(_ORDINAL_OUTPUT_PRODUCTS):
        return False
    return head in _ORDINAL_EXPLICIT_METHODS or _method_has_ordinal_primary_token(head)


# --- Trajectory-clustering compatibility audit ------------------------------
# Kept as a tested contract helper for legacy/resume inspection. Production has
# no clustering preflight or coder-failure runner: the agent owns feature/method/k
# and deterministic code only renders registered clustering products.
def _trajectory_clustering_step_matches(
    method: str,
    blob: str,
    expected_blob: str = "",
) -> bool:
    """Whether a legacy KMeans artifact contract is phenotype-compatible.

    The caller supplies lowercased strings and has already excluded figure steps.
    Compatibility requires an explicit KMeans method head plus at least two
    standard clustering products.  A primary EFFECT step (OR/HR/AUROC) is always
    excluded, and latent-class/GMM/unspecified phenotyping remains agent-owned so
    the auxiliary cannot silently replace the planned scientific method.
    """
    expected_outputs = re.split(r"[\s,]+", str(expected_blob or ""))
    if _declares_effect_output(expected_outputs):
        return False
    return _clustering_contract_applies(
        method=str(method or ""),
        intent=str(blob or ""),
        expected_outputs=str(expected_blob or ""),
        auxiliary_kmeans_only=True,
        minimum_output_signals=2,
    )


def _primary_runner_core_estimate_present(
    kind: Optional[str], step_summary: Mapping[str, Any]
) -> bool:
    """True when a PRIMARY deterministic runner emitted its core estimate.

    The runner's own ``status`` is the authority: it writes ``ok`` only when the
    estimate computed and ``blocked`` on genuinely non-viable data. When ``ok``
    and the effect key is present, the runner has satisfied the scientific
    contract for the step -- any extra planner-requested output tables it does
    not emit are advisory, not a reason to discard a trustworthy estimate.
    """
    if kind not in _PRIMARY_DETERMINISTIC_RUNNERS:
        return False
    if not isinstance(step_summary, Mapping):
        return False
    if str(step_summary.get("status") or "").lower() != "ok":
        return False
    if kind in ("causal_primary_iptw", "ordinal_dose_response"):
        # Both emit the scale-neutral ``adjusted_effect`` as their core estimate
        # (causal: marginal OR; ordinal: trend OR per +1 stage).
        return step_summary.get("adjusted_effect") is not None
    # survival_primary_cox
    if step_summary.get("hazard_ratio") is not None:
        return True
    primary_model = step_summary.get("primary_model")
    return (
        isinstance(primary_model, Mapping)
        and primary_model.get("hazard_ratio") is not None
    )


def _demote_step_contract_for_primary_runner(
    step_record: Mapping[str, Any],
    step_summary: Mapping[str, Any],
    findings: Sequence[ValidationFinding],
) -> List[ValidationFinding]:
    """Apply contract compatibility to legacy deterministic-primary records.

    When such a runner produced its core estimate, demote ``step_contract``
    missing-output ERRORS to advisory warnings. Otherwise a planner that
    over-specifies a step's ``expected_outputs`` (e.g. 17 documentation tables a
    causal step does not need) fail-closes the step and triggers a repair that
    replaces a validated legacy estimate with a repair. Integrity findings from
    other validators (exposure / overadjustment / leakage / figure) remain
    blocking. Live primary science is agent-owned; this is record compatibility.
    """
    kind = step_record.get("deterministic_standard_analysis")
    if not _primary_runner_core_estimate_present(kind, step_summary):
        return list(findings)
    demoted: List[ValidationFinding] = []
    for finding in findings:
        if (
            getattr(finding, "validator", "") == "step_contract"
            and finding.severity == "error"
        ):
            finding = finding.model_copy(
                update={
                    "severity": "warning",
                    "message": (
                        finding.message
                        + f" [advisory: step satisfied by deterministic {kind} "
                        "runner; extra planner-requested outputs are non-blocking]"
                    ),
                }
            )
        demoted.append(finding)
    return demoted


def _is_too_few_panels_figure_finding(finding: ValidationFinding) -> bool:
    """True for the ``figure_contract_quality`` "result figure has <2 panels"
    ERROR specifically.

    Keyed off ``detail['panel_count']`` (which only that finding sets) rather
    than the message text, so it stays robust if the wording changes. Blank-
    title / weak-claim / fallback-term figure errors are deliberately NOT
    matched -- only the panel-count shape rule is demoted below.
    """
    if getattr(finding, "validator", "") != "figure_contract_quality":
        return False
    if getattr(finding, "severity", "") != "error":
        return False
    detail = getattr(finding, "detail", None) or {}
    panel_count = detail.get("panel_count") if isinstance(detail, Mapping) else None
    return isinstance(panel_count, int) and panel_count < 2


def _family_has_deterministic_figure_renderer(context: Any) -> bool:
    """True when this study-design family builds its PRIMARY publication figure
    deterministically in the write phase (``render_family_figure``).

    Lazy import keeps ``pipeline_execute`` free of a ``figures`` /
    ``study_design`` import-order dependency and fail-safes to False (strict) if
    the family cannot be inferred.
    """
    try:
        from .figures import FAMILY_RENDERERS
        from .study_design import infer_study_design_family

        return str(infer_study_design_family(context)) in FAMILY_RENDERERS
    except Exception:
        return False


def _demote_result_figure_shape_for_family_renderer(
    context: Any,
    findings: Sequence[ValidationFinding],
) -> List[ValidationFinding]:
    """Demote a step-level "result figure has <2 panels" ERROR to a warning when
    the study-design family assembles its primary figure deterministically.

    A family in ``FAMILY_RENDERERS`` can have a deterministic multi-panel publication
    figure renderer, but it only runs in the WRITE phase -- which is gated behind
    ``execution_complete``. When the LLM's step-level figure is single-panel, the
    ``figure_contract_quality`` panel-count ERROR marks the step ``contract_
    failed`` -> ``execution_complete`` stays False -> the write phase is skipped
    -> the deterministic renderer (the very thing that would produce the >=2-panel
    primary) never runs. The step-level figure is NOT the manuscript's primary
    for these families, so its panel count is advisory here. The write-phase
    display-suite gate remains fully fail-closed: if the deterministic renderer
    cannot build a >=2-panel primary from the registered tables, the run still
    fails with "no primary publication result-bearing figure contract". Pure so
    both branches are unit-testable.
    """
    if not any(_is_too_few_panels_figure_finding(f) for f in findings):
        return list(findings)
    if not _family_has_deterministic_figure_renderer(context):
        return list(findings)
    demoted: List[ValidationFinding] = []
    for finding in findings:
        if _is_too_few_panels_figure_finding(finding):
            finding = finding.model_copy(
                update={
                    "severity": "warning",
                    "message": (
                        finding.message
                        + " [advisory: this study-design family builds its "
                        "manuscript-facing primary figure deterministically in "
                        "the write phase; the display-suite gate remains the "
                        "fail-closed backstop for panel count and role diversity]"
                    ),
                }
            )
        demoted.append(finding)
    return demoted


@dataclass(frozen=True)
class _FinalDeterministicGateFindings:
    """Attempt-bound finding groups produced by the final deterministic gate.

    The immutable grouping keeps evaluation separate from orchestration: the
    evaluator below reads sealed outputs and returns findings, while the caller
    remains responsible for publishing them to the run manifest, evidence
    metadata, and outer step status.  Resume revalidation can therefore reuse
    the same evaluator without duplicating its gate composition.
    """

    stat_findings: Tuple[ValidationFinding, ...]
    clinical_findings: Tuple[ValidationFinding, ...]
    guard_findings: Tuple[ValidationFinding, ...]
    contract_findings: Tuple[ValidationFinding, ...]
    figure_source_findings: Tuple[ValidationFinding, ...]

    def all_findings(self) -> Tuple[ValidationFinding, ...]:
        """Return all groups in the historical manifest publication order."""

        return (
            *self.stat_findings,
            *self.clinical_findings,
            *self.guard_findings,
            *self.contract_findings,
            *self.figure_source_findings,
        )


@dataclass(frozen=True)
class _ResumeDeterministicRevalidationResult:
    """Append-only resume ledger after selective deterministic replay."""

    resume_state: Dict[str, Any]
    revalidated_step_ids: Tuple[str, ...]
    invalidated_step_ids: Tuple[str, ...]


def _verified_explicit_step_authority(
    *,
    record: Mapping[str, Any],
    field: str,
    expected_kind: str,
    expected_source_name: Optional[str],
    evidence_by_id: Mapping[str, Any],
    run_dir: Path,
) -> Tuple[Any, Path]:
    """Resolve one exact checkpoint authority through owner/path/SHA checks."""

    step_id = str(record.get("step_id") or "").strip()
    evidence_id = str(record.get(field) or "").strip()
    listed = {
        str(value).strip()
        for value in (record.get("evidence_ids") or [])
        if str(value).strip()
    }
    if not evidence_id:
        raise ValueError(f"successful checkpoint is missing required {field}")
    if evidence_id not in listed:
        raise ValueError(f"{field} {evidence_id} is absent from evidence_ids")
    authority = evidence_by_id.get(evidence_id)
    if authority is None:
        raise ValueError(f"{field} references missing evidence {evidence_id}")
    if str(_evidence_record_field(authority, "produced_by_step") or "") != step_id:
        raise ValueError(f"{field} is not owned by step {step_id}")
    actual_kind = str(_evidence_record_field(authority, "kind") or "").lower()
    if actual_kind != expected_kind:
        raise ValueError(
            f"{field} has kind {actual_kind or '<missing>'}, expected {expected_kind}"
        )
    verified_path = verified_run_evidence_path(run_dir, authority)
    if verified_path is None:
        raise ValueError(f"{field} failed path/digest verification")
    source_name = _registered_source_name(authority, verified_path)
    if expected_source_name is not None and source_name != expected_source_name:
        raise ValueError(f"{field} does not name {expected_source_name}")
    return authority, verified_path


def _verified_resume_step_summary(
    *,
    record: Mapping[str, Any],
    evidence_by_id: Mapping[str, Any],
    run_dir: Path,
) -> Dict[str, Any]:
    """Load a summary only from the record's explicit digest-bound evidence."""

    field = (
        "probe_summary_evidence_id"
        if str(record.get("step_id") or "") == "00_probe"
        else "step_summary_evidence_id"
    )
    _, summary_path = _verified_explicit_step_authority(
        record=record,
        field=field,
        expected_kind="statistic",
        expected_source_name=(
            "probe_summary.json"
            if field == "probe_summary_evidence_id"
            else "step_summary.json"
        ),
        evidence_by_id=evidence_by_id,
        run_dir=run_dir,
    )
    try:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError) as exc:
        raise ValueError(f"{field} is not readable JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{field} payload is not an object")
    return payload


def _verify_resume_step_script_lineage(
    *,
    record: Mapping[str, Any],
    evidence_by_id: Mapping[str, Any],
) -> None:
    """Require every sealed non-code output to bind the reviewed script.

    Owner and digest checks alone are insufficient: a mutable checkpoint could
    list a second benign script from the same step and point
    ``script_evidence_id`` at it while retaining outputs produced by the real
    script.  Exact lineage closes that decoy-code path before preflight.
    """

    step_id = str(record.get("step_id") or "").strip()
    script_evidence_id = str(record.get("script_evidence_id") or "").strip()
    if not script_evidence_id:
        raise ValueError("successful checkpoint is missing script_evidence_id")
    for raw_id in record.get("evidence_ids") or []:
        evidence_id = str(raw_id).strip()
        authority = evidence_by_id.get(evidence_id)
        if authority is None:
            raise ValueError(f"listed evidence {evidence_id} is missing")
        owner = str(_evidence_record_field(authority, "produced_by_step") or "")
        if owner != step_id:
            raise ValueError(
                f"listed evidence {evidence_id} belongs to {owner or '<run-level>'}"
            )
        if evidence_id == script_evidence_id:
            if str(_evidence_record_field(authority, "kind") or "").lower() != "code":
                raise ValueError("script_evidence_id does not reference code evidence")
            continue
        bound_script_id = str(
            _evidence_record_field(authority, "script_evidence_id") or ""
        ).strip()
        if bound_script_id != script_evidence_id:
            raise ValueError(
                f"listed evidence {evidence_id} is bound to script "
                f"{bound_script_id or '<missing>'}, not {script_evidence_id}"
            )


def _trusted_resume_success_records(
    *,
    records: Sequence[Mapping[str, Any]],
    evidence_by_id: Mapping[str, Any],
    run_dir: Path,
) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    """Replace mutable checkpoint summaries with explicit evidence payloads."""

    trusted: List[Dict[str, Any]] = []
    errors: Dict[str, str] = {}
    for record in records:
        if str(record.get("status") or "").lower() != "ok":
            continue
        step_id = str(record.get("step_id") or "").strip()
        if (
            str(record.get("generation_mode") or "").strip().lower()
            == _HOST_COHORT_MATERIALIZER_GENERATION_MODE
            and record.get("step_authority_kind")
            == _HOST_COHORT_MATERIALIZER_AUTHORITY_KIND
        ):
            copy = dict(record)
            copy.pop("resolved_inputs", None)
            copy.pop("resolved_input_bindings", None)
            copy.pop("resolved_inputs_path", None)
            trusted.append(copy)
            continue
        try:
            summary = _verified_resume_step_summary(
                record=record,
                evidence_by_id=evidence_by_id,
                run_dir=run_dir,
            )
        except ValueError as exc:
            errors[step_id] = str(exc)
            continue
        copy = dict(record)
        copy["step_summary"] = summary
        # These mutable convenience receipts are never replay authority.
        copy.pop("resolved_inputs", None)
        copy.pop("resolved_input_bindings", None)
        copy.pop("resolved_inputs_path", None)
        trusted.append(copy)
    return trusted, errors


def _materialize_verified_step_output_view(
    *,
    record: Mapping[str, Any],
    evidence_by_id: Mapping[str, Any],
    run_dir: Path,
    destination: Path,
) -> None:
    """Copy only listed, verified same-step evidence under source filenames."""

    step_id = str(record.get("step_id") or "").strip()
    listed = [
        str(value).strip()
        for value in (record.get("evidence_ids") or [])
        if str(value).strip()
    ]
    if not listed:
        raise ValueError("successful checkpoint has no evidence_ids")
    destination.mkdir(parents=True, exist_ok=False)
    copied: Dict[str, str] = {}
    for evidence_id in listed:
        authority = evidence_by_id.get(evidence_id)
        if authority is None:
            raise ValueError(f"listed evidence {evidence_id} is missing")
        owner = str(_evidence_record_field(authority, "produced_by_step") or "")
        if owner != step_id:
            raise ValueError(
                f"listed evidence {evidence_id} belongs to {owner or '<run-level>'}"
            )
        verified_path = verified_run_evidence_path(run_dir, authority)
        if verified_path is None:
            raise ValueError(
                f"listed evidence {evidence_id} failed digest verification"
            )
        source_name = _registered_source_name(authority, verified_path)
        if (
            not source_name
            or Path(source_name).name != source_name
            or "/" in source_name
            or "\\" in source_name
        ):
            raise ValueError(
                f"listed evidence {evidence_id} has no safe source filename"
            )
        prior_id = copied.get(source_name)
        if prior_id is not None and prior_id != evidence_id:
            raise ValueError(f"multiple listed evidence records claim {source_name}")
        target = destination / source_name
        shutil.copyfile(verified_path, target)
        target.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
        copied[source_name] = evidence_id


def _resume_typed_input_bindings(
    *,
    step: AnalysisStep,
    plan: AnalysisPlan,
    evidence_records: Sequence[Any],
    trusted_step_records: Sequence[Mapping[str, Any]],
    run_dir: Path,
    cohort_path: Path,
) -> Tuple[Dict[str, Dict[str, Any]], List[str]]:
    """Rebuild typed bindings without reading mutable resolved-input receipts."""

    bindings: Dict[str, Dict[str, Any]] = {}
    evidence_ids: List[str] = []
    for raw_input in step.inputs or []:
        input_name = str(raw_input)
        if _typed_input_product(input_name) is None:
            continue
        ref, failure = _resolve_typed_input_evidence(
            input_name=input_name,
            plan=plan,
            evidence_records=evidence_records,
            per_step_records=trusted_step_records,
            run_dir=run_dir,
        )
        if failure is not None or ref is None:
            reason = failure or {"reason": "verified_reference_unavailable"}
            raise ValueError(
                f"typed input {input_name} could not be resolved: "
                + json.dumps(reason, sort_keys=True, default=str)
            )
        binding = _resolved_typed_input_binding(
            input_name=input_name,
            evidence_ref=ref,
            evidence_records=evidence_records,
            run_dir=run_dir,
            producer_step_records=trusted_step_records,
            authoritative_cohort_path=cohort_path,
        )
        if binding is None:
            raise ValueError(f"typed input {input_name} has no verified host binding")
        bindings[input_name] = binding
        evidence_ids.append(ref.evidence_id)
    return bindings, list(dict.fromkeys(evidence_ids))


def _resume_success_dependencies(
    *,
    plan: AnalysisPlan,
    current_records: Sequence[Mapping[str, Any]],
    evidence_by_id: Mapping[str, Any],
) -> Dict[str, Set[str]]:
    """Derive immutable plan/evidence producer edges for invalidation."""

    product_producers: Dict[Tuple[str, str], Set[str]] = {}
    for step in plan.steps:
        for raw_output in step.expected_outputs or []:
            product = _typed_input_product(raw_output)
            if product is not None:
                product_producers.setdefault(product, set()).add(step.step_id)
    dependencies: Dict[str, Set[str]] = {}
    steps_by_id = {step.step_id: step for step in plan.steps}
    for record in current_records:
        step_id = str(record.get("step_id") or "").strip()
        deps = dependencies.setdefault(step_id, set())
        step = steps_by_id.get(step_id)
        if step is not None:
            for raw_input in step.inputs or []:
                product = _typed_input_product(raw_input)
                producers = product_producers.get(product or ("", ""), set())
                if len(producers) == 1:
                    deps.update(producers - {step_id})
        pending = [
            str(value).strip()
            for evidence_id in (record.get("evidence_ids") or [])
            if (authority := evidence_by_id.get(str(evidence_id).strip())) is not None
            for value in (_evidence_record_field(authority, "inputs") or [])
            if str(value).strip()
        ]
        seen: Set[str] = set()
        while pending:
            evidence_id = pending.pop()
            if evidence_id in seen:
                continue
            seen.add(evidence_id)
            authority = evidence_by_id.get(evidence_id)
            if authority is None:
                continue
            owner = str(_evidence_record_field(authority, "produced_by_step") or "")
            if owner and owner != step_id:
                deps.add(owner)
                continue
            pending.extend(
                str(value).strip()
                for value in (_evidence_record_field(authority, "inputs") or [])
                if str(value).strip()
            )
    return dependencies


def _evaluate_final_deterministic_gates(
    *,
    context: ResearchContext,
    cohort_path: Path,
    universe_path: Path,
    run_dir: Path,
    out_dir: Path,
    step: AnalysisStep,
    step_summary: Dict[str, Any],
    step_record: Mapping[str, Any],
    completed_step_records: Sequence[Mapping[str, Any]],
    resolved_input_bindings: Mapping[str, Mapping[str, Any]],
    attempt_id: str,
    checkpoint_id: str,
    stat_validator: StatisticalValidator,
    clinical_validator: ClinicalConstraintValidator,
    statistical_guard: StatisticalGuard,
    cross_step_cohort_lock_validator: CrossStepCohortLockValidator,
    cross_step_registered_output_validator: CrossStepRegisteredOutputValidator,
    cross_step_reconciliation_trace_validator: CrossStepReconciliationTraceValidator,
    step_summary_integrity_validator: StepSummaryIntegrityValidator,
    step_summary_fraction_validator: StepSummaryFractionValidator,
    cross_step_source_status_validator: CrossStepSourceStatusValidator,
    primary_model_contract_validator: PrimaryModelContractValidator,
    figure_contract_validator: FigureContractQualityValidator,
    figure_source_validator: FigureSourceDataValidator,
) -> _FinalDeterministicGateFindings:
    """Evaluate the complete final deterministic review for one step attempt.

    This function deliberately does not append to the run-wide findings list,
    mutate ``step_record``, publish evidence, or decide the outer step status.
    Filesystem-reading validators make it only *pure-ish*, but all gate
    composition, compatibility demotions, and attempt binding now live here as
    one reusable authority.
    """

    stat_findings = stat_validator.audit(
        context=context,
        cohort_path=cohort_path,
        step=step,
        out_dir=out_dir,
        step_summary=step_summary,
    )
    clinical_findings = clinical_validator.audit(
        context=context,
        step=step,
        out_dir=out_dir,
        step_summary=step_summary,
    )
    guard_findings = statistical_guard.audit(
        context=context,
        cohort_path=cohort_path,
        step=step,
        out_dir=out_dir,
        step_summary=step_summary,
    )
    contract_findings = _step_contract_findings(
        step=step,
        step_summary=step_summary,
        completed_step_records=completed_step_records,
        resolved_input_bindings=resolved_input_bindings,
        out_dir=out_dir,
    )
    contract_findings.extend(
        _cohort_definition_sensitivity_contract_findings(
            step=step,
            step_summary=step_summary,
            out_dir=out_dir,
            run_dir=run_dir,
            universe_path=universe_path,
            cohort_path=cohort_path,
            context=context,
            completed_step_records=completed_step_records,
        )
    )
    contract_findings.extend(
        cross_step_cohort_lock_validator.audit(
            step=step,
            step_summary=step_summary,
            completed_step_records=completed_step_records,
        )
    )
    contract_findings.extend(
        cross_step_registered_output_validator.audit(
            step=step,
            step_summary=step_summary,
            completed_step_records=completed_step_records,
        )
    )
    contract_findings.extend(
        cross_step_reconciliation_trace_validator.audit(
            step=step,
            step_summary=step_summary,
            out_dir=out_dir,
        )
    )
    contract_findings.extend(
        step_summary_integrity_validator.audit(
            step=step,
            step_summary=step_summary,
            resolved_input_bindings=resolved_input_bindings,
            cohort_path=cohort_path,
        )
    )
    contract_findings.extend(
        step_summary_fraction_validator.audit(
            step=step,
            step_summary=step_summary,
        )
    )
    contract_findings.extend(
        cross_step_source_status_validator.audit(
            step=step,
            step_summary=step_summary,
            completed_step_records=completed_step_records,
        )
    )
    contract_findings.extend(
        primary_model_contract_validator.audit(
            step=step,
            step_summary=step_summary,
            context=context,
            completed_step_records=completed_step_records,
            out_dir=out_dir,
            cohort_path=cohort_path,
        )
    )
    contract_findings.extend(
        _primary_exposure_contract_findings(
            step=step,
            step_summary=step_summary,
            context=context,
        )
    )
    contract_findings.extend(
        _primary_exposure_measurement_filter_findings(
            step=step,
            step_summary=step_summary,
            context=context,
        )
    )
    contract_findings.extend(
        _primary_exposure_overadjustment_findings(
            step=step,
            context=context,
            out_dir=out_dir,
        )
    )
    contract_findings.extend(
        _primary_model_leakage_findings(
            step=step,
            context=context,
            out_dir=out_dir,
        )
    )
    contract_findings.extend(
        figure_contract_validator.audit(
            step=step,
            out_dir=out_dir,
            run_dir=run_dir,
            step_summary=step_summary,
        )
    )
    # Legacy deterministic-primary records own their historical core estimate;
    # only their excess step-output declarations are demoted.  All integrity,
    # exposure, leakage, and figure findings remain blocking.
    contract_findings = _demote_step_contract_for_primary_runner(
        step_record,
        step_summary,
        contract_findings,
    )
    # Some study-design families build the manuscript-facing multi-panel figure
    # in the write phase.  Preserve the existing narrow step-figure demotion;
    # the publication display-suite gate remains fail-closed.
    contract_findings = _demote_result_figure_shape_for_family_renderer(
        context,
        contract_findings,
    )
    figure_source_findings = figure_source_validator.audit(
        step=step,
        out_dir=out_dir,
        run_dir=run_dir,
        step_summary=step_summary,
        completed_step_records=completed_step_records,
        resolved_input_bindings=resolved_input_bindings,
    )

    def _bind(
        group: Sequence[ValidationFinding],
    ) -> Tuple[ValidationFinding, ...]:
        return tuple(
            _bind_findings_to_step_attempt(
                group,
                step_id=step.step_id,
                attempt_id=attempt_id,
                checkpoint_id=checkpoint_id,
            )
        )

    return _FinalDeterministicGateFindings(
        stat_findings=_bind(stat_findings),
        clinical_findings=_bind(clinical_findings),
        guard_findings=_bind(guard_findings),
        contract_findings=_bind(contract_findings),
        figure_source_findings=_bind(figure_source_findings),
    )


def _selectively_revalidate_resume_successes(
    *,
    resume_state: Dict[str, Any],
    plan: AnalysisPlan,
    context: ResearchContext,
    evidence: Any,
    run_dir: Path,
    cohort_path: Path,
    universe_path: Path,
    resume_from_step_id: Optional[str],
) -> _ResumeDeterministicRevalidationResult:
    """Replay changed deterministic gates against sealed evidence only.

    The function runs before :class:`ResumeController` applies skip decisions.
    It never invokes the coder, runner, analyzer, or LLM concept auditor.
    Successful replay appends a new ``ok`` authority checkpoint; any error
    appends ``resume_validator_invalid`` and makes the step executable again.
    """

    state = dict(resume_state)
    authority_history = [
        dict(record)
        for record in (resume_state.get("per_step_records") or [])
        if isinstance(record, Mapping)
    ]
    saved_attempt_history = [
        dict(record)
        for record in (resume_state.get("step_attempt_history") or [])
        if isinstance(record, Mapping)
    ]
    history = saved_attempt_history or list(authority_history)
    for authority_record in authority_history:
        if authority_record not in history:
            history.append(authority_record)
    current_records = [
        dict(record) for record in current_step_records(authority_history)
    ]
    current_successes = [
        record
        for record in current_records
        if str(record.get("status") or "").strip().lower() == "ok"
    ]
    steps_by_id = {step.step_id: step for step in plan.steps}
    step_order = {"00_probe": -1, **{s.step_id: i for i, s in enumerate(plan.steps)}}
    seeded_invalidated = {
        str(record.get("step_id") or "").strip(): (
            "prior checkpoint already lacks current resume authority "
            f"(status={str(record.get('status') or '').strip().lower()})"
        )
        for record in current_records
        if str(record.get("status") or "").strip().lower()
        in {"resume_evidence_invalid", "resume_validator_invalid"}
    }
    if resume_from_step_id and seeded_invalidated:
        cut = step_order.get(resume_from_step_id)
        earlier_invalid = sorted(
            step_id
            for step_id in seeded_invalidated
            if cut is not None and step_order.get(step_id, cut) < cut
        )
        if earlier_invalid:
            raise RunInputIdentityError(
                "Cannot start resume after an already-invalid upstream "
                "authority; resume at or before: " + ", ".join(earlier_invalid)
            )
    stamp = _deterministic_gate_stamp()
    stale_successes = [
        record
        for record in current_successes
        if record.get("deterministic_gate_fingerprint")
        != stamp["deterministic_gate_fingerprint"]
    ]
    if not stale_successes and not seeded_invalidated:
        return _ResumeDeterministicRevalidationResult(state, (), ())

    evidence_records = list(evidence.records())
    evidence_by_id = {
        str(_evidence_record_field(record, "evidence_id") or ""): record
        for record in evidence_records
    }
    trusted_records, trusted_summary_errors = _trusted_resume_success_records(
        records=current_successes,
        evidence_by_id=evidence_by_id,
        run_dir=run_dir,
    )
    trusted_by_step = {
        str(record.get("step_id") or ""): record for record in trusted_records
    }
    current_by_step = {
        str(record.get("step_id") or ""): record for record in current_successes
    }
    dependencies = _resume_success_dependencies(
        plan=plan,
        current_records=current_records,
        evidence_by_id=evidence_by_id,
    )
    invalidated: Dict[str, str] = dict(seeded_invalidated)
    revalidated: List[str] = []
    invalid_payloads: Dict[str, Dict[str, Any]] = {}
    retirement_records: Dict[str, Mapping[str, Any]] = {}

    def attempt_identity(step_id: str) -> Tuple[str, str]:
        sequence = 1 + sum(
            1
            for record in history
            if str(record.get("step_id") or "") == step_id
            and record.get("revalidated_without_execution") is True
        )
        attempt_id = f"{step_id}:resume_revalidation:{sequence}"
        return attempt_id, f"{attempt_id}:deterministic_review"

    def indexed_alias_evidence_ids(prior_record: Mapping[str, Any]) -> List[str]:
        step_id = str(prior_record.get("step_id") or "").strip()
        indexed_ids: List[str] = []
        for raw_id in prior_record.get("evidence_ids") or []:
            evidence_id = str(raw_id).strip()
            authority = evidence_by_id.get(evidence_id)
            if (
                authority is not None
                and str(_evidence_record_field(authority, "produced_by_step") or "")
                == step_id
            ):
                indexed_ids.append(evidence_id)
        return list(dict.fromkeys(indexed_ids))

    for invalid_step_id in seeded_invalidated:
        prior_success = next(
            (
                record
                for record in reversed(history)
                if str(record.get("step_id") or "").strip() == invalid_step_id
                and str(record.get("status") or "").strip().lower() == "ok"
            ),
            None,
        )
        if prior_success is not None:
            retirement_records[invalid_step_id] = prior_success

    def append_invalid(
        *,
        prior_record: Mapping[str, Any],
        reason: str,
        code_findings: Sequence[ValidationFinding] = (),
        gate_findings: Optional[_FinalDeterministicGateFindings] = None,
    ) -> None:
        step_id = str(prior_record.get("step_id") or "").strip()
        if step_id in invalidated:
            return
        attempt_id, checkpoint_id = attempt_identity(step_id)
        if not code_findings and gate_findings is None:
            code_findings = _bind_findings_to_step_attempt(
                [
                    ValidationFinding(
                        validator="resume_deterministic_revalidation",
                        severity="error",
                        message=(
                            f"Prior success for step {step_id} failed current "
                            "deterministic replay."
                        ),
                        detail={"reason": reason},
                    )
                ],
                step_id=step_id,
                attempt_id=attempt_id,
                checkpoint_id=checkpoint_id,
            )
        payload: Dict[str, Any] = {
            "step_id": step_id,
            "status": "resume_validator_invalid",
            "revalidated_without_execution": True,
            "attempt_id": attempt_id,
            "review_checkpoint_id": checkpoint_id,
            "resume_invalidation_reason": reason,
            "invalidated_evidence_ids": list(prior_record.get("evidence_ids") or []),
            "evidence_ids": [],
            "deterministic_code_findings": [
                finding.model_dump(mode="json") for finding in code_findings
            ],
            "retired_current_aliases": {},
            **stamp,
        }
        for key, value in prior_record.items():
            if key.startswith("step_provider_call_") or key.startswith(
                "step_llm_repair_"
            ):
                payload[key] = value
        if gate_findings is not None:
            payload.update(
                {
                    "stat_findings": [
                        finding.model_dump(mode="json")
                        for finding in gate_findings.stat_findings
                    ],
                    "clinical_findings": [
                        finding.model_dump(mode="json")
                        for finding in gate_findings.clinical_findings
                    ],
                    "guard_findings": [
                        finding.model_dump(mode="json")
                        for finding in gate_findings.guard_findings
                    ],
                    "contract_findings": [
                        finding.model_dump(mode="json")
                        for finding in gate_findings.contract_findings
                    ],
                    "figure_source_findings": [
                        finding.model_dump(mode="json")
                        for finding in gate_findings.figure_source_findings
                    ],
                }
            )
        invalidated[step_id] = reason
        invalid_payloads[step_id] = payload
        retirement_records[step_id] = prior_record
        history.append(payload)

    stale_successes.sort(
        key=lambda record: step_order.get(
            str(record.get("step_id") or ""), len(step_order)
        )
    )
    for prior_record in stale_successes:
        step_id = str(prior_record.get("step_id") or "").strip()
        invalid_upstream = sorted(
            dependencies.get(step_id, set()).intersection(invalidated)
        )
        if invalid_upstream:
            append_invalid(
                prior_record=prior_record,
                reason=(
                    "current success depends on invalidated upstream step(s): "
                    + ", ".join(invalid_upstream)
                ),
            )
            continue
        if step_id == "00_probe":
            summary_error = trusted_summary_errors.get(step_id)
            if summary_error is not None or step_id not in trusted_by_step:
                append_invalid(
                    prior_record=prior_record,
                    reason=(summary_error or "probe summary authority is unavailable"),
                )
                continue
            evidence_payloads = {
                evidence_id: (
                    record.model_dump(mode="json")
                    if hasattr(record, "model_dump")
                    else dict(record)
                )
                for evidence_id, record in evidence_by_id.items()
            }
            error = _host_probe_authority_error(
                record=prior_record,
                evidence_ids=list(prior_record.get("evidence_ids") or []),
                step_id=step_id,
                run_dir=run_dir,
                records=evidence_payloads,
            )
            if error is not None:
                append_invalid(prior_record=prior_record, reason=error)
                continue
            attempt_id, checkpoint_id = attempt_identity(step_id)
            summary = trusted_by_step[step_id]["step_summary"]
            replayed = {
                **prior_record,
                "status": "ok",
                "step_summary": dict(summary),
                "revalidated_without_execution": True,
                "attempt_id": attempt_id,
                "review_checkpoint_id": checkpoint_id,
                **stamp,
            }
            history.append(replayed)
            trusted_by_step[step_id] = replayed
            revalidated.append(step_id)
            continue

        is_host_cohort_materializer = (
            str(prior_record.get("generation_mode") or "").strip().lower()
            == _HOST_COHORT_MATERIALIZER_GENERATION_MODE
            or prior_record.get("step_authority_kind")
            == _HOST_COHORT_MATERIALIZER_AUTHORITY_KIND
        )
        if is_host_cohort_materializer:
            evidence_payloads = {
                evidence_id: (
                    record.model_dump(mode="json")
                    if hasattr(record, "model_dump")
                    else dict(record)
                )
                for evidence_id, record in evidence_by_id.items()
            }
            error = _host_cohort_materializer_authority_error(
                record=prior_record,
                evidence_ids=list(prior_record.get("evidence_ids") or []),
                step_id=step_id,
                run_dir=run_dir,
                records=evidence_payloads,
            )
            if error is not None:
                append_invalid(prior_record=prior_record, reason=error)
                continue
            attempt_id, checkpoint_id = attempt_identity(step_id)
            replayed = {
                **prior_record,
                "status": "ok",
                "step_summary": dict(prior_record["step_summary"]),
                "revalidated_without_execution": True,
                "attempt_id": attempt_id,
                "review_checkpoint_id": checkpoint_id,
                **stamp,
            }
            history.append(replayed)
            trusted_by_step[step_id] = replayed
            revalidated.append(step_id)
            continue

        step = steps_by_id.get(step_id)
        summary_error = trusted_summary_errors.get(step_id)
        if step is None or summary_error is not None:
            append_invalid(
                prior_record=prior_record,
                reason=(summary_error or "successful step is absent from active plan"),
            )
            continue
        trusted_record = trusted_by_step[step_id]
        attempt_id, checkpoint_id = attempt_identity(step_id)
        try:
            _verify_resume_step_script_lineage(
                record=prior_record,
                evidence_by_id=evidence_by_id,
            )
            _, script_path = _verified_explicit_step_authority(
                record=prior_record,
                field="script_evidence_id",
                expected_kind="code",
                expected_source_name=None,
                evidence_by_id=evidence_by_id,
                run_dir=run_dir,
            )
            script_text = script_path.read_text(encoding="utf-8")
            code_findings = _bind_findings_to_step_attempt(
                _deterministic_code_gate_findings(
                    context=context,
                    step=step,
                    script_text=script_text,
                ),
                step_id=step_id,
                attempt_id=attempt_id,
                checkpoint_id=checkpoint_id,
            )
            if any(finding.severity == "error" for finding in code_findings):
                append_invalid(
                    prior_record=prior_record,
                    reason="current deterministic code preflight failed",
                    code_findings=code_findings,
                )
                continue
            trusted_current_records = [
                record
                for record in trusted_by_step.values()
                if str(record.get("status") or "").lower() == "ok"
                and str(record.get("step_id") or "") not in invalidated
            ]
            resolved_bindings, resolved_input_evidence_ids = (
                _resume_typed_input_bindings(
                    step=step,
                    plan=plan,
                    evidence_records=evidence_records,
                    trusted_step_records=trusted_current_records,
                    run_dir=run_dir,
                    cohort_path=cohort_path,
                )
            )
            with tempfile.TemporaryDirectory(
                prefix=f".resume_gate_{step_id}_",
                dir=run_dir,
            ) as temporary_root:
                replay_out_dir = Path(temporary_root) / "outputs"
                _materialize_verified_step_output_view(
                    record=prior_record,
                    evidence_by_id=evidence_by_id,
                    run_dir=run_dir,
                    destination=replay_out_dir,
                )
                completed_records = [
                    record
                    for record in trusted_current_records
                    if str(record.get("step_id") or "") != step_id
                    and step_order.get(str(record.get("step_id") or ""), -1)
                    < step_order.get(step_id, len(step_order))
                ]
                gate_findings = _evaluate_final_deterministic_gates(
                    context=context,
                    cohort_path=cohort_path,
                    universe_path=universe_path,
                    run_dir=run_dir,
                    out_dir=replay_out_dir,
                    step=step,
                    step_summary=dict(trusted_record["step_summary"]),
                    step_record=prior_record,
                    completed_step_records=completed_records,
                    resolved_input_bindings=resolved_bindings,
                    attempt_id=attempt_id,
                    checkpoint_id=checkpoint_id,
                    stat_validator=StatisticalValidator(),
                    clinical_validator=ClinicalConstraintValidator(),
                    statistical_guard=StatisticalGuard(),
                    cross_step_cohort_lock_validator=CrossStepCohortLockValidator(),
                    cross_step_registered_output_validator=(
                        CrossStepRegisteredOutputValidator()
                    ),
                    cross_step_reconciliation_trace_validator=(
                        CrossStepReconciliationTraceValidator()
                    ),
                    step_summary_integrity_validator=StepSummaryIntegrityValidator(),
                    step_summary_fraction_validator=StepSummaryFractionValidator(),
                    cross_step_source_status_validator=CrossStepSourceStatusValidator(),
                    primary_model_contract_validator=PrimaryModelContractValidator(),
                    figure_contract_validator=FigureContractQualityValidator(),
                    figure_source_validator=FigureSourceDataValidator(),
                )
            if any(
                finding.severity == "error" for finding in gate_findings.all_findings()
            ):
                append_invalid(
                    prior_record=prior_record,
                    reason="current deterministic artifact gates failed",
                    code_findings=code_findings,
                    gate_findings=gate_findings,
                )
                continue
            prior_critique = prior_record.get("critique_report")
            prior_critique_status = (
                str(prior_critique.get("status") or "").strip().lower()
                if isinstance(prior_critique, Mapping)
                else ""
            )
            if prior_critique_status in {"blocked", "needs_revision"}:
                append_invalid(
                    prior_record=prior_record,
                    reason=(
                        "prior deterministic Critic status remains "
                        f"{prior_critique_status}"
                    ),
                    code_findings=code_findings,
                    gate_findings=gate_findings,
                )
                continue
            evidence_refs = [
                EvidenceRef(
                    evidence_id=str(_evidence_record_field(authority, "evidence_id")),
                    kind=_evidence_record_field(authority, "kind"),
                    description=str(
                        _evidence_record_field(authority, "description") or ""
                    ),
                    relative_path=str(
                        _evidence_record_field(authority, "relative_path") or ""
                    ),
                )
                for evidence_id in (prior_record.get("evidence_ids") or [])
                if (authority := evidence_by_id.get(str(evidence_id))) is not None
                and verified_run_evidence_path(run_dir, authority) is not None
            ]
            critique = CriticAgent().review_step(
                step=step,
                step_summary=dict(trusted_record["step_summary"]),
                evidence_refs=evidence_refs,
                findings=_actionable_validator_messages(
                    code_findings,
                    gate_findings.all_findings(),
                ),
            )
            if critique.status != "pass":
                append_invalid(
                    prior_record=prior_record,
                    reason=f"current deterministic Critic status={critique.status}",
                    code_findings=code_findings,
                    gate_findings=gate_findings,
                )
                continue
        except (OSError, TypeError, UnicodeError, ValueError) as exc:
            append_invalid(
                prior_record=prior_record,
                reason=f"{type(exc).__name__}: {exc}",
            )
            continue

        replayed = {
            **prior_record,
            "status": "ok",
            "step_summary": dict(trusted_record["step_summary"]),
            "resolved_input_evidence_ids": resolved_input_evidence_ids,
            "deterministic_code_findings": [
                finding.model_dump(mode="json") for finding in code_findings
            ],
            "stat_findings": [
                finding.model_dump(mode="json")
                for finding in gate_findings.stat_findings
            ],
            "clinical_findings": [
                finding.model_dump(mode="json")
                for finding in gate_findings.clinical_findings
            ],
            "guard_findings": [
                finding.model_dump(mode="json")
                for finding in gate_findings.guard_findings
            ],
            "contract_findings": [
                finding.model_dump(mode="json")
                for finding in gate_findings.contract_findings
            ],
            "figure_source_findings": [
                finding.model_dump(mode="json")
                for finding in gate_findings.figure_source_findings
            ],
            "critique_report": critique.model_dump(mode="json"),
            "revalidated_without_execution": True,
            "attempt_id": attempt_id,
            "review_checkpoint_id": checkpoint_id,
            **stamp,
        }
        replayed.pop("resolved_inputs", None)
        replayed.pop("resolved_input_bindings", None)
        replayed.pop("resolved_inputs_path", None)
        history.append(replayed)
        trusted_by_step[step_id] = replayed
        revalidated.append(step_id)

    # Propagate invalid authority through immutable plan/evidence edges, even
    # when a downstream success already carries the current fingerprint.
    while True:
        changed = False
        for step_id, prior_record in current_by_step.items():
            if step_id in invalidated:
                continue
            failed_dependencies = sorted(
                dependencies.get(step_id, set()).intersection(invalidated)
            )
            if not failed_dependencies:
                continue
            append_invalid(
                prior_record=prior_record,
                reason=(
                    "current success depends on invalidated upstream step(s): "
                    + ", ".join(failed_dependencies)
                ),
            )
            changed = True
        if not changed:
            break

    # An explicit cut after a newly detected invalid authority must fail
    # before any alias or manifest mutation.  The caller can restart at or
    # before the earliest invalid upstream step.
    if resume_from_step_id and invalidated:
        cut = step_order.get(resume_from_step_id)
        earlier_invalid = sorted(
            step_id
            for step_id in invalidated
            if cut is not None and step_order.get(step_id, cut) < cut
        )
        if earlier_invalid:
            raise RunInputIdentityError(
                "Cannot start resume after deterministic-validator-invalid "
                "upstream evidence; resume at or before: " + ", ".join(earlier_invalid)
            )

    if invalid_payloads:
        state_findings = list(resume_state.get("findings") or [])
        for step_id, payload in invalid_payloads.items():
            reason = str(payload.get("resume_invalidation_reason") or "")
            state_findings.append(
                ValidationFinding(
                    validator="resume_deterministic_revalidation",
                    severity="warning",
                    message=(
                        f"Prior success for step {step_id} was invalidated by "
                        "current deterministic gates and requires re-execution."
                    ),
                    detail={
                        "step_id": step_id,
                        "reason": reason,
                        "requires_reexecution": True,
                    },
                ).model_dump(mode="json")
            )
        state["findings"] = state_findings
    state["step_attempt_history"] = history
    state["per_step_records"] = [
        dict(record) for record in current_step_records(history)
    ]

    retirement_batch = {
        step_id: evidence_ids
        for step_id, prior_record in retirement_records.items()
        if (evidence_ids := indexed_alias_evidence_ids(prior_record))
    }
    current_aliases = evidence.aliases() if retirement_batch else {}
    for step_id, evidence_ids in retirement_batch.items():
        payload = invalid_payloads.get(step_id)
        if payload is not None:
            payload["retired_current_aliases"] = {
                alias: evidence_id
                for alias, evidence_id in current_aliases.items()
                if evidence_id in set(evidence_ids)
            }

    # Persist the append-only checkpoint before revoking aliases.  A failed
    # checkpoint write leaves aliases untouched; the batch retirement itself
    # is atomic across every invalid step.  If retirement fails, restore the
    # prior manifest so the two authority ledgers cannot disagree.
    checkpoint_path = run_dir / "manifest_partial.json"
    write_run_checkpoint(checkpoint_path, state)
    if retirement_batch:
        try:
            evidence.retire_steps_current_aliases(retirement_batch)
        except (KeyError, OSError, TypeError, ValueError) as exc:
            try:
                write_run_checkpoint(checkpoint_path, resume_state)
            except (OSError, TypeError, ValueError) as rollback_exc:
                raise RuntimeError(
                    "resume revalidation alias retirement and manifest rollback "
                    "both failed"
                ) from rollback_exc
            raise RuntimeError(
                "resume revalidation alias retirement failed; manifest was rolled back"
            ) from exc

    return _ResumeDeterministicRevalidationResult(
        resume_state=state,
        revalidated_step_ids=tuple(revalidated),
        invalidated_step_ids=tuple(sorted(invalidated)),
    )


def run_execute_phase(
    pipeline: "ResearchAgentPipeline",
    *,
    plan_result: _PlanPhaseResult,
    cohort_path: Path,
    run_dir: Path,
    run_id: str,
    skill_obj: Optional[ClinicalSkill],
    notes: Optional[str],
    emit_progress: Callable[..., None],
    resume_from_step_id: Optional[str] = None,
    stop_after_step_id: Optional[str] = None,
) -> _ExecutePhaseResult:
    """Execute probe + per-step analysis loop, with optional replanning."""
    context = plan_result.context
    agent_context = plan_result.agent_context
    evidence = plan_result.evidence
    findings = plan_result.findings
    plan = plan_result.plan
    plan_path = plan_result.plan_path
    plan, companion_input_findings = _augment_measurement_companion_inputs(
        plan=plan,
        context=context,
    )
    if companion_input_findings:
        findings.extend(companion_input_findings)
        plan_path = run_dir / "analysis_plan_input_closure.json"
        plan_path.write_text(plan.model_dump_json(indent=2), encoding="utf-8")
        if evidence.get("analysis_plan_input_closure") is None:
            evidence.register_file(
                kind="log",
                description=(
                    "Analysis plan with structural measurement-provenance "
                    "input closure."
                ),
                source_path=plan_path,
                evidence_id="analysis_plan_input_closure",
                producer="runtime_supervisor",
                generation_mode="system",
                prompt_pack_version=plan_result.prompt_version,
                metadata={"reason": "measurement_companion_input_closure"},
            )
        plan_result.plan_path = plan_path
    resume_controller = ResumeController(
        plan=plan,
        run_dir=run_dir,
        resume_state=plan_result.resume_state,
        resume_from_step_id=resume_from_step_id,
        stop_after_step_id=stop_after_step_id,
    )
    requested_resume_from_step_id = resume_controller.resume_from_step_id
    requested_stop_after_step_id = resume_controller.stop_after_step_id
    reuse_selected_step_code_opt_in = (
        requested_resume_from_step_id is not None
        and os.environ.get("EASYICU_RESUME_REUSE_STEP_CODE") == "1"
    )
    resumed_cohort_translation_budget: Optional[Dict[str, Any]] = None
    resumed_cohort_translation_budget_owner: Optional[str] = None
    if isinstance(plan_result.resume_state, Mapping):
        raw_cohort_translation_budget = plan_result.resume_state.get(
            "cohort_translation_provider_budget"
        )
        if isinstance(raw_cohort_translation_budget, Mapping):
            candidate_owner = str(
                raw_cohort_translation_budget.get("budget_owner_step_id") or ""
            ).strip()
            if candidate_owner:
                resumed_cohort_translation_budget = dict(raw_cohort_translation_budget)
                resumed_cohort_translation_budget_owner = candidate_owner
    # Replan convergence bookkeeping (see _maybe_replan). ``noop_streak``
    # counts consecutive substantively-identical revisions; ``total`` counts
    # substantive revisions; ``disabled`` latches once a guard trips.
    _replan_state = {
        "noop_streak": 0,
        "total": 0,
        "disabled": False,
        # Latches True when the substantive-revision count reaches
        # ``max_replans``; drives the fail-closed diagnostic_only demotion.
        "budget_exhausted": False,
        "cohort_contract_emitted": False,
        "cohort_materialized": False,
        # The first cohort-prose translation latches one provider-budget owner
        # for the run.  Later replans cannot buy a fresh allowance by renaming
        # or reshaping the cohort-definition step.
        "cohort_translation_budget_owner_step_id": (
            resumed_cohort_translation_budget_owner
        ),
        "cohort_translation_provider_budget": resumed_cohort_translation_budget,
        "cohort_translation_provider_budget_error_emitted": False,
        # Directed replans fired when a model/estimation step self-blocks on a
        # task-viable cohort (see _maybe_directed_model_replan). Bounded so a
        # run that keeps self-blocking falls back to an honest diagnostic_only
        # rather than looping the replanner indefinitely.
        "directed_model_replans": 0,
    }
    role_resolver = plan_result.role_resolver
    llm_signature = plan_result.llm_signature
    concept_audit_environment_sha256 = canonical_sha256(
        build_environment_identity(llm_signature=llm_signature)
    )
    prompt_version = plan_result.prompt_version
    prompt_files = plan_result.prompt_files
    assert_cohort_definition_locked(run_dir=run_dir, plan=plan)
    assert_robustness_specs_locked(run_dir=run_dir, plan=plan)

    # Dual-track cohort. If the plan phase materialised the locked cohort
    # definition into a filtered analysis cohort, every downstream consumer
    # (probe, statistical validators, robustness fitter, and the step runner)
    # reads THAT — so the declared inclusion/exclusion is enforced once,
    # consistently, instead of being silently re-implemented (or skipped) by
    # each generated step. The full universe stays reachable via the runner's
    # EASYICU_UNIVERSE_PARQUET env for explicit robustness steps.
    universe_path = cohort_path
    _analysis_cohort_path = run_dir / "cohort_analysis.parquet"
    if _analysis_cohort_path.exists():
        cohort_path = _analysis_cohort_path

    # Validator/code drift is resolved before ResumeController decides which
    # prior successes to skip and before any coder, runner, analyzer, or LLM
    # collaborator is constructed.  The replay reads sealed evidence only.
    if plan_result.resume_state is not None:
        resume_revalidation = _selectively_revalidate_resume_successes(
            resume_state=plan_result.resume_state,
            plan=plan,
            context=context,
            evidence=evidence,
            run_dir=run_dir,
            cohort_path=cohort_path,
            universe_path=universe_path,
            resume_from_step_id=requested_resume_from_step_id,
        )
        plan_result.resume_state = resume_revalidation.resume_state
        resume_controller.resume_state = resume_revalidation.resume_state

    coder = CoderAgent(role_resolver("coder"))
    # Opt-in altitude-2a: delegate script authoring + self-repair to a local
    # coding-agent CLI when EASYICU_AGENTIC_CODER_BACKEND is set. Off by default;
    # degrades back to ``coder`` when the CLI is unavailable. The script it
    # returns is still executed + evidence-bound by the instrumented runtime.
    from .agentic_coder import maybe_wrap_coder

    coder = maybe_wrap_coder(coder)
    analyzer = AnalyzerAgent(role_resolver("analyzer"))
    supervisor = RuntimeSupervisor(
        clinical_semantics=ClinicalSemanticsAgent(),
        data_extraction=DataExtractionAgent(),
        statistical_analysis=StatisticalAnalysisAgent(),
        visualization=VisualizationAgent(),
        critic=CriticAgent(role_resolver("analyzer")),
    )
    runner = pipeline._build_runner(
        run_dir=run_dir,
        cohort_path=cohort_path,
        target_outcome=context.target_outcome,
        universe_path=universe_path,
    )
    usage_auditor = ConceptUsageAuditor()
    pattern_auditor = AnalysisPatternAuditor()
    stat_validator = StatisticalValidator()
    figure_contract_validator = FigureContractQualityValidator()
    figure_source_validator = FigureSourceDataValidator()
    clinical_validator = ClinicalConstraintValidator()
    cross_step_cohort_lock_validator = CrossStepCohortLockValidator()
    cross_step_registered_output_validator = CrossStepRegisteredOutputValidator()
    cross_step_reconciliation_trace_validator = CrossStepReconciliationTraceValidator()
    cross_step_source_status_validator = CrossStepSourceStatusValidator()
    step_summary_fraction_validator = StepSummaryFractionValidator()
    step_summary_integrity_validator = StepSummaryIntegrityValidator()
    primary_model_contract_validator = PrimaryModelContractValidator()
    statistical_guard = StatisticalGuard()
    llm_concept_audit_cache = LLMConceptAuditCache(run_dir)
    llm_concept_auditor_source = inspect.getsourcefile(LLMConceptAuditor)
    llm_concept_auditor_implementation_sha256 = (
        sha256_of_file(Path(llm_concept_auditor_source))
        if llm_concept_auditor_source and Path(llm_concept_auditor_source).is_file()
        else ""
    )
    runtime_state = supervisor.bootstrap_state(run_id=run_id, context=context)
    repair_ledger = RepairLedger(run_dir / "repairs_applied.json")
    repair_ledger_lock = threading.Lock()

    per_step_records: List[Dict[str, Any]] = []
    step_attempt_history: List[Dict[str, Any]] = []
    probe_summary: Dict[str, Any] = {}
    resumed_step_ids: set = set()
    # Steps can finish before the ordinary plan execution loop.  Keep those
    # ids distinct from resume state: a probe-aware replan is allowed to retain
    # the probe in its returned plan, but that must not schedule a second coder
    # execution for work the host already completed deterministically.
    preexecuted_step_ids: set = set()
    if plan_result.resume_state is not None:
        resume_application = resume_controller.apply()
        step_attempt_history.extend(resume_application.audit_history)
        per_step_records.extend(resume_application.per_step_records)
        resumed_step_ids = set(resume_application.resumed_step_ids)
        preexecuted_step_ids.update(resumed_step_ids)
        findings.extend(resume_application.findings)
        probe_summary = resume_application.probe_summary
        findings.extend(
            materialize_legacy_trajectory_replay_schemas(
                plan=plan,
                context=context,
                run_dir=run_dir,
                evidence=evidence,
                per_step_records=per_step_records,
                prompt_pack_version=prompt_version,
            )
        )
        if resumed_step_ids:
            print(
                f"[research_agent] resume: skipping {len(resumed_step_ids)} "
                f"already-completed step(s) — {sorted(resumed_step_ids)}"
            )

    def _flush_partial_manifest(extra: Optional[Dict[str, Any]] = None) -> None:
        for record in per_step_records:
            snapshot = dict(record)
            if snapshot not in step_attempt_history:
                step_attempt_history.append(snapshot)
        payload: Dict[str, Any] = {
            "schema_version": "easyicu.research_manifest_partial/1",
            "run_id": run_id,
            "research_question": context.research_question,
            "started_at": plan_result.started_at.isoformat(),
            "context_path": str(plan_result.context_path.relative_to(run_dir)),
            "plan_path": str(plan_path.relative_to(run_dir)),
            "evidence": [r.model_dump(mode="json") for r in evidence.records()],
            "findings": [f.model_dump(mode="json") for f in findings],
            "per_step_records": per_step_records,
            "step_attempt_history": step_attempt_history,
            "llm_signature": llm_signature,
            "used_mock_llm": plan_result.used_mock_llm,
            "prompt_pack_version": prompt_version,
            "prompt_pack_files": prompt_files,
            "notes": notes,
            "runtime_state": runtime_state.model_dump(mode="json"),
            "repair_ledger_path": str(repair_ledger.path.relative_to(run_dir)),
            "repairs_applied": [record.__dict__ for record in repair_ledger.records],
            "cohort_translation_provider_budget": _replan_state.get(
                "cohort_translation_provider_budget"
            ),
        }
        if extra:
            payload.update(extra)
        write_run_checkpoint(run_dir / "manifest_partial.json", payload)

    runtime_packets = {
        "clinical_semantics_resolution": runtime_state.semantics,
        "data_extraction_request": runtime_state.extraction_request,
        "data_extraction_result": runtime_state.extraction_result,
    }
    for alias, packet in runtime_packets.items():
        if packet is None or evidence.get(alias) is not None:
            continue
        evidence.register_json(
            kind="log",
            description=f"Typed runtime packet: {alias}.",
            payload=packet.model_dump(mode="json"),
            filename=f"{alias}.json",
            evidence_id=alias,
            aliases=[alias],
            producer="runtime_supervisor",
            generation_mode="system",
            prompt_pack_version=prompt_version,
            metadata={"run_id": run_id},
        )

    _flush_partial_manifest()

    def _register_plan_revision(
        revised_plan: AnalysisPlan,
        *,
        reason: str,
    ) -> Path:
        revision_path = run_dir / f"analysis_plan_revision_{revised_plan.revision}.json"
        revision_path.write_text(
            revised_plan.model_dump_json(indent=2),
            encoding="utf-8",
        )
        base_id = f"analysis_plan_revision_{revised_plan.revision}"
        try:
            evidence.register_file(
                kind="log",
                description=f"Revised analysis plan (reason={reason}).",
                source_path=revision_path,
                evidence_id=base_id,
                producer="replanner",
                generation_mode="llm",
                prompt_pack_version=prompt_version,
                metadata={"reason": reason, "llm_signature": llm_signature},
            )
        except ValueError:
            # Resume + replan can legitimately re-emit the same revision number
            # with different content (the replanner is non-deterministic across
            # runs), which collides with the prior run's
            # ``analysis_plan_revision_N`` id. Keep both by versioning the id
            # with a content digest instead of crashing the resumed run. The
            # global evidence-id collision guard stays intact for every other
            # artefact.
            import hashlib

            digest = hashlib.sha256(revision_path.read_bytes()).hexdigest()[:8]
            evidence.register_file(
                kind="log",
                description=(
                    f"Revised analysis plan (reason={reason}; resume re-revision)."
                ),
                source_path=revision_path,
                evidence_id=f"{base_id}_{digest}",
                producer="replanner",
                generation_mode="llm",
                prompt_pack_version=prompt_version,
                metadata={
                    "reason": reason,
                    "llm_signature": llm_signature,
                    "resume_reregistration": True,
                },
            )
        return revision_path

    def _no_analysis_step_has_run() -> bool:
        """True while only the deterministic probe (00_probe) has executed.

        The cohort may be (re)materialised and the runner re-pointed only at
        this point; switching the cohort after analysis steps already ran on
        the universe would split a single run across two populations.
        """
        return not any(
            (rec.get("step_id") or "") != "00_probe" for rec in per_step_records
        )

    def _universe_columns() -> list:
        try:
            import pyarrow.parquet as pq  # type: ignore

            return list(pq.read_schema(universe_path).names)
        except Exception:
            try:
                import pandas as pd  # type: ignore

                return list(pd.read_parquet(universe_path).columns)
            except Exception:
                return []

    def _try_materialize_cohort_from_prose(
        candidate_plan: AnalysisPlan,
        *,
        reason: str,
    ) -> bool:
        """Extract the agent's prose 纳排 into typed predicates, materialise the
        filtered analysis cohort, and re-point the runner at it.

        Returns ``True`` when the cohort was materialised (so the caller skips
        the auditable contract error). The locked initial cohort was an empty
        placeholder for the bench's 0-step plan; locking the first real
        definition here is a provisional→real lock, fully provenance-recorded.
        """
        nonlocal cohort_path, runner
        if _replan_state["cohort_materialized"]:
            return True
        if _analysis_cohort_path.exists():
            return True
        if not _no_analysis_step_has_run():
            return False
        columns = _universe_columns()
        if not columns:
            return False
        budget_owner_step_id = _replan_state.get(
            "cohort_translation_budget_owner_step_id"
        )
        if not budget_owner_step_id:
            budget_owner_step_id = _cohort_translation_budget_owner_step_id(
                candidate_plan
            )
            _replan_state["cohort_translation_budget_owner_step_id"] = (
                budget_owner_step_id
            )
        try:
            definition, budget_snapshot = (
                _extract_cohort_definition_with_provider_budget(
                    run_dir=run_dir,
                    budget_owner_step_id=str(budget_owner_step_id),
                    configured_limit=pipeline._max_step_provider_calls,
                    cohort_prose=_cohort_definition_prose(candidate_plan),
                    universe_columns=columns,
                    llm=role_resolver("planner"),
                    name=getattr(
                        getattr(candidate_plan, "cohort", None),
                        "name",
                        "primary",
                    )
                    or "primary",
                )
            )
        except ProviderCallBudgetError as exc:
            error_detail = f"{type(exc).__name__}: {exc}"
            _replan_state["cohort_translation_provider_budget"] = {
                "budget_owner_step_id": str(budget_owner_step_id),
                "error": error_detail,
            }
            if not _replan_state.get(
                "cohort_translation_provider_budget_error_emitted"
            ):
                findings.append(
                    ValidationFinding(
                        validator="cohort_translation_provider_budget",
                        severity="error",
                        message=(
                            "Cohort-definition translation could not obtain a "
                            "trusted provider-call reservation; the host did not "
                            "infer or apply cohort criteria."
                        ),
                        detail={
                            "stage": "execute_repair",
                            "reason": reason,
                            "budget_owner_step_id": str(budget_owner_step_id),
                            "error": error_detail,
                        },
                    )
                )
                _replan_state["cohort_translation_provider_budget_error_emitted"] = True
            return False
        _replan_state["cohort_translation_provider_budget"] = budget_snapshot
        if definition is None:
            return False
        candidate_plan.cohort = definition
        try:
            write_locked_cohort_definition(
                run_dir=run_dir,
                plan=candidate_plan,
                evidence=evidence,
                prompt_pack_version=prompt_version,
                llm_signature=llm_signature,
                allow_empty_promotion=True,
            )
            result = materialize_locked_analysis_cohort(
                run_dir=run_dir,
                plan=candidate_plan,
                universe_path=universe_path,
            )
        except Exception as exc:  # never break the run; fall back to the error
            findings.append(
                ValidationFinding(
                    validator="cohort_materializer",
                    severity="warning",
                    message=(
                        "Extracted a cohort definition from step prose but could "
                        f"not materialise it: {type(exc).__name__}: {exc}"
                    ),
                    detail={"stage": "execute_repair", "reason": reason},
                )
            )
            return False
        if result.get("status") != "applied":
            return False
        cohort_path = _analysis_cohort_path
        runner = pipeline._build_runner(
            run_dir=run_dir,
            cohort_path=cohort_path,
            target_outcome=context.target_outcome,
            universe_path=universe_path,
        )
        cohort_product_steps = [
            step
            for step in candidate_plan.steps
            if _declares_host_cohort_only_product(step)
        ]
        cohort_product_step = (
            cohort_product_steps[0] if len(cohort_product_steps) == 1 else None
        )
        try:
            cohort_record = evidence.register_file(
                kind="table",
                description=(
                    "Analysis cohort materialised from the agent's prose 纳排, "
                    "translated to typed CTAS predicates during execution."
                ),
                source_path=cohort_path,
                evidence_id="analysis_cohort_execute_repair",
                produced_by_step=(
                    cohort_product_step.step_id if cohort_product_step else None
                ),
                producer="cohort_repair",
                generation_mode="llm",
                prompt_pack_version=prompt_version,
                metadata={"llm_signature": llm_signature, "reason": reason},
            )
        except ValueError:
            cohort_record = evidence.get("analysis_cohort_execute_repair")
        if cohort_product_step is not None and cohort_record is not None:
            # The deterministic materialiser has completely realised this
            # single-product step using the cohort the Agent selected.  Record
            # that product under the planned producer and do not ask the Coder
            # to recreate or reinterpret the cohort scientifically.
            cohort_checkpoint = {
                "step_id": cohort_product_step.step_id,
                "intent": cohort_product_step.intent,
                "status": "ok",
                "generation_mode": _HOST_COHORT_MATERIALIZER_GENERATION_MODE,
                "step_authority_kind": _HOST_COHORT_MATERIALIZER_AUTHORITY_KIND,
                _HOST_COHORT_MATERIALIZER_AUTHORITY_FIELD: (cohort_record.evidence_id),
                "step_summary": {
                    "output_files": {
                        "table:analysis_cohort": str(cohort_path.relative_to(run_dir))
                    },
                    "n_universe": int(result["n_universe"]),
                    "n_analysis_cohort": int(result["n_cohort"]),
                },
                "evidence_ids": [cohort_record.evidence_id],
                **_deterministic_gate_stamp(),
            }
            if budget_owner_step_id == cohort_product_step.step_id:
                cohort_checkpoint.update(
                    {
                        key: value
                        for key, value in budget_snapshot.items()
                        if key != "budget_owner_step_id"
                    }
                )
            cohort_authority_error = _host_cohort_materializer_authority_error(
                record=cohort_checkpoint,
                evidence_ids=[cohort_record.evidence_id],
                step_id=cohort_product_step.step_id,
                run_dir=run_dir,
                records={
                    record.evidence_id: record.model_dump(mode="json")
                    for record in evidence.records()
                },
            )
            if cohort_authority_error is None:
                per_step_records.append(cohort_checkpoint)
                preexecuted_step_ids.add(cohort_product_step.step_id)
            else:
                findings.append(
                    ValidationFinding(
                        validator="cohort_materializer_authority",
                        severity="error",
                        message=(
                            "Host cohort materializer could not seal its exact "
                            "single-product authority."
                        ),
                        detail={
                            "step_id": cohort_product_step.step_id,
                            "reason": cohort_authority_error,
                        },
                    )
                )
        findings.append(
            ValidationFinding(
                validator="cohort_materializer",
                severity="info",
                message=(
                    "Translated the cohort-definition step's prose into typed "
                    "predicates and applied them: analysis cohort "
                    f"n={result['n_cohort']} of universe n={result['n_universe']}. "
                    "Downstream steps now read the filtered cohort "
                    "(COHORT_PARQUET); the full universe stays available as "
                    "EASYICU_UNIVERSE_PARQUET."
                ),
                detail={
                    "stage": "execute_repair",
                    "reason": reason,
                    "n_universe": result["n_universe"],
                    "n_analysis_cohort": result["n_cohort"],
                },
            )
        )
        _replan_state["cohort_materialized"] = True
        return True

    def _enforce_cohort_contract_on_executing_plan(
        candidate_plan: AnalysisPlan,
        *,
        reason: str,
    ) -> None:
        """Re-check the structured-纳排 contract against the plan that actually
        executes.

        The plan-phase contract (``pipeline._run_plan_phase``) only sees the
        *initial* plan. For non-deterministic providers that initial plan is
        commonly a 0-step shell, and the real plan — which carries a
        cohort-definition step but leaves ``plan.cohort`` structurally empty —
        is grown here by the replanner. Without this re-check the contract is
        bypassed and downstream steps silently run on the unfiltered universe
        while each generated step re-applies 纳排 inconsistently (run12).

        Emitted once, as an auditable error, and only when the locked cohort
        was *not* materialised into a filtered analysis cohort (an applied
        definition already enforces 纳排 on the data).
        """
        if _replan_state["cohort_contract_emitted"]:
            return
        if _analysis_cohort_path.exists():
            return
        if not (
            _plan_expects_analysis_cohort(candidate_plan)
            and _cohort_definition_is_empty(candidate_plan)
        ):
            return
        for finding in _cohort_definition_contract_findings(candidate_plan):
            findings.append(
                finding.model_copy(
                    update={
                        "detail": {
                            **(finding.detail or {}),
                            "stage": "execute",
                            "reason": reason,
                        }
                    }
                )
            )
        _replan_state["cohort_contract_emitted"] = True

    def _resolve_cohort_definition(
        candidate_plan: AnalysisPlan,
        *,
        reason: str,
    ) -> None:
        """For an executing plan that implies a cohort but left it unstructured:
        first try to materialise it from the step prose (real enforcement); if
        that fails, surface the auditable contract error (visibility)."""
        if not (
            _plan_expects_analysis_cohort(candidate_plan)
            and _cohort_definition_is_empty(candidate_plan)
        ):
            return
        if _try_materialize_cohort_from_prose(candidate_plan, reason=reason):
            return
        _enforce_cohort_contract_on_executing_plan(candidate_plan, reason=reason)

    _resolve_cohort_definition(plan, reason="execute_start")

    def _maybe_replan(
        *,
        current_plan: AnalysisPlan,
        reason: str,
        probe_summary_payload: Optional[Dict[str, Any]] = None,
        completed_records: Optional[Sequence[Dict[str, Any]]] = None,
        directive: Optional[str] = None,
        force: bool = False,
    ) -> AnalysisPlan:
        nonlocal plan_path
        if not pipeline._enable_replanning or skill_obj is not None:
            return current_plan
        if _replan_state["disabled"] and not force:
            # A convergence guard already tripped earlier in this run; stop
            # paying for replanner calls that cannot change the outcome. A
            # ``force``d directed replan (bounded by its own caller-side budget)
            # bypasses this — it carries a new instruction the replanner has not
            # yet seen, so the prior no-op/budget verdict does not apply.
            return current_plan
        terminal_repair_skip = _terminal_publication_repair_replan_skip_detail(
            plan=current_plan,
            completed_records=completed_records,
            run_dir=run_dir,
        )
        if terminal_repair_skip is not None and not force:
            findings.append(
                ValidationFinding(
                    validator="replanner",
                    severity="info",
                    message=(
                        "Skipped replanner because only terminal rendering-only "
                        "publication-figure repair steps remain, and a completed "
                        "step already produced a primary-result publication bundle."
                    ),
                    detail={
                        "reason": reason,
                        **terminal_repair_skip,
                    },
                )
            )
            return current_plan
        replanner = ReplannerAgent(role_resolver("planner"))
        try:
            revised = replanner.run(
                context=agent_context,
                current_plan=current_plan,
                probe_summary=probe_summary_payload,
                completed_step_records=completed_records,
                directive=directive,
            )
        except Exception as exc:
            findings.append(
                ValidationFinding(
                    validator="replanner",
                    severity="warning",
                    message=f"Replanner failed; keeping existing plan: {exc}",
                    detail={"reason": reason},
                )
            )
            return current_plan
        revised, immutable_step_findings = (
            _preserve_completed_step_snapshots_after_replan(
                current_plan=current_plan,
                revised_plan=revised,
                completed_records=completed_records or [],
            )
        )
        findings.extend(immutable_step_findings)
        # Guard against the replanner silently dropping the primary
        # result-bearing MODEL step (the estimand) while inserting an
        # audit/reconciliation step. Run this before figure preservation so the re-attached
        # model step precedes any re-attached figure step.
        revised, estimand_findings = _preserve_primary_estimand_step_after_replan(
            current=current_plan,
            revised=revised,
        )
        if estimand_findings:
            findings.extend(estimand_findings)
        # Guard against the replanner silently dropping figure-producing
        # steps; task contracts (e.g. EasyICU experiment runner) still
        # require those artefacts regardless of the LLM's revised framing.
        revised, preservation_findings = _preserve_figure_steps_after_replan(
            current=current_plan,
            revised=revised,
        )
        if preservation_findings:
            findings.extend(preservation_findings)

        # Cap total plan size after a replan. A verbose replanner can grow a
        # simple analysis into many revisions without converging. The cap
        # truncates excess late-stage steps and forces the replanner
        # to revise existing steps in place on later passes. Cap of 0
        # disables the guard for backward compatibility.
        revised, report_input_findings = _augment_report_typed_product_inputs(
            plan=revised
        )
        findings.extend(report_input_findings)

        cap = pipeline._max_total_steps
        if cap > 0:
            protected_step_ids = [
                str(record.get("step_id"))
                for record in current_successful_step_records(completed_records or [])
                if record.get("step_id") and record.get("status") == "ok"
            ]
            capped_revised, cap_findings = _cap_plan_preserving_figure_steps(
                plan=revised,
                cap=cap,
                protected_step_ids=protected_step_ids,
            )
            revised = capped_revised
            findings.extend(
                finding.model_copy(
                    update={
                        "validator": "replanner",
                        "message": (finding.message or "").replace(
                            "Initial plan had",
                            "Replanner produced",
                        ),
                    }
                )
                for finding in cap_findings
            )

        revised, robustness_lock_finding = (
            _preserve_locked_robustness_specs_after_replan(
                current_plan=current_plan,
                revised_plan=revised,
                run_dir=run_dir,
            )
        )
        if robustness_lock_finding is not None:
            findings.append(robustness_lock_finding)
        revised, trajectory_product_findings = augment_trajectory_plan_products(
            plan=revised,
            context=context,
        )
        findings.extend(trajectory_product_findings)
        revised, companion_input_findings = _augment_measurement_companion_inputs(
            plan=revised,
            context=context,
        )
        findings.extend(companion_input_findings)
        revised, post_transform_snapshot_findings = (
            _preserve_completed_step_snapshots_after_replan(
                current_plan=current_plan,
                revised_plan=revised,
                completed_records=completed_records or [],
            )
        )
        findings.extend(post_transform_snapshot_findings)

        # No-op detection uses the scientific signature rather than the full
        # model_dump. Only casing/whitespace changes in intent are cosmetic;
        # semantic prose remains authoritative until every estimand coordinate
        # has a structured schema field.
        if _plan_signature(revised) == _plan_signature(current_plan):
            _replan_state["noop_streak"] += 1
            cap_noop = pipeline._max_consecutive_noop_replans
            if cap_noop and _replan_state["noop_streak"] >= cap_noop:
                _replan_state["disabled"] = True
                findings.append(
                    ValidationFinding(
                        validator="replanner",
                        severity="info",
                        message=(
                            f"Replanning disabled after {_replan_state['noop_streak']} "
                            "consecutive no-op revisions (unchanged step plan)."
                        ),
                        detail={"reason": reason},
                    )
                )
            return current_plan

        # Substantive revision: reset the no-op streak and register it.
        _replan_state["noop_streak"] = 0
        _replan_state["total"] += 1
        plan_path = _register_plan_revision(revised, reason=reason)
        plan_result.plan_path = plan_path
        _resolve_cohort_definition(revised, reason=reason)
        findings.append(
            ValidationFinding(
                validator="replanner",
                severity="info",
                message=f"Plan revised after {reason}.",
                detail={
                    "from_revision": current_plan.revision,
                    "to_revision": revised.revision,
                },
            )
        )
        cap_total = pipeline._max_replans
        if cap_total and _replan_state["total"] >= cap_total:
            _replan_state["disabled"] = True
            _replan_state["budget_exhausted"] = True
            # Fail closed: reaching the replan budget without the plan
            # converging is a runaway loop, not a clean run. The run is
            # demoted to diagnostic_only so a non-converging replan cascade
            # cannot launder a manuscript. The trigger is kept in ``detail``
            # (never the message) so a step-id-shaped reason cannot make the
            # readiness supersession rule drop this run-level latch.
            findings.append(
                ValidationFinding(
                    validator="replan_budget",
                    severity="error",
                    message=(
                        "Replan budget exhausted: "
                        f"{_replan_state['total']} substantive plan revisions "
                        f"reached the cap of {cap_total} without the plan "
                        "converging. Run demoted to diagnostic_only "
                        "(fail-closed) rather than emitting a manuscript from a "
                        "non-converging replan loop."
                    ),
                    detail={
                        "replan_budget_exhausted": True,
                        "cap": cap_total,
                        "substantive_revisions": _replan_state["total"],
                        "reason": reason,
                    },
                )
            )
        return revised

    trajectory_plan_blocked = False
    typed_plan_dag_blocked = False
    probe_step_id = "00_probe"
    if pipeline._enable_probe_step and probe_step_id not in resumed_step_ids:
        probe_summary, probe_files = _build_probe_summary(
            context=context,
            cohort_path=cohort_path,
            out_dir=run_dir / "steps" / probe_step_id / "outputs",
        )
        probe_evidence_ids: List[str] = []
        probe_authority_fields: Dict[str, str] = {}
        for probe_file in probe_files:
            kind = "statistic" if probe_file.name.endswith(".json") else "table"
            aliases = [probe_step_id]
            if probe_file.name == "probe_summary.json":
                aliases.extend(
                    [
                        "probe_summary",
                        "cohort_probe",
                    ]
                )
            rec = evidence.register_file(
                kind=kind,
                description=f"Probe artefact {probe_file.name}.",
                source_path=probe_file,
                produced_by_step=probe_step_id,
                producer="pipeline",
                generation_mode="deterministic_probe",
                aliases=aliases,
            )
            probe_evidence_ids.append(rec.evidence_id)
            for field, (_kind, source_name) in _HOST_PROBE_AUTHORITIES.items():
                if probe_file.name == source_name:
                    probe_authority_fields[field] = rec.evidence_id
        missing_probe_authorities = sorted(
            set(_HOST_PROBE_AUTHORITIES) - set(probe_authority_fields)
        )
        if missing_probe_authorities:
            raise RuntimeError(
                "Host probe did not produce its required authority fields: "
                + ", ".join(missing_probe_authorities)
            )
        probe_record = {
            "step_id": probe_step_id,
            "intent": "Probe distributions, missingness, and obvious anomalies before execution.",
            "status": "ok",
            "generation_mode": "deterministic_probe",
            "step_summary": probe_summary,
            "evidence_ids": probe_evidence_ids,
            "step_authority_kind": _HOST_PROBE_AUTHORITY_KIND,
            **probe_authority_fields,
            **_deterministic_gate_stamp(),
        }
        per_step_records.append(probe_record)
        preexecuted_step_ids.add(probe_step_id)
        _flush_partial_manifest()
        typed_plan_preflight = _typed_plan_dag_findings(plan)
        trajectory_preflight = trajectory_plan_dag_findings(
            plan=plan,
            context=context,
        )
        trajectory_directive = None
        typed_plan_directive = None
        if typed_plan_preflight:
            typed_plan_directive = (
                "Repair the plan's declared typed product DAG without changing "
                "its scientific choices. Every typed kind:product input must "
                "have exactly one declared producer, every required producer "
                "must remain in the plan, and producers must precede consumers. "
                "Do not invent an exposure, outcome, cohort, estimator, or "
                "analysis method. Contract findings: "
                + json.dumps(
                    [
                        {
                            "message": finding.message,
                            "detail": finding.detail,
                        }
                        for finding in typed_plan_preflight
                    ],
                    ensure_ascii=False,
                    default=str,
                )
            )
        if trajectory_preflight:
            trajectory_directive = (
                "Repair the agent-declared fixed-window trajectory plan DAG "
                "without changing its scientific choices. Preserve legitimate "
                "representation, candidate-selection, stability/freeze, and "
                "characterization step boundaries; repair only missing/ambiguous "
                "typed artifact edges, role declarations, and silent internal "
                "window-grid omissions. Do not choose a clustering method, k, "
                "eligibility threshold, or deterministic runner. Contract findings: "
                + json.dumps(
                    [
                        {
                            "message": finding.message,
                            "detail": finding.detail,
                        }
                        for finding in trajectory_preflight
                    ],
                    ensure_ascii=False,
                    default=str,
                )
            )
        plan = _maybe_replan(
            current_plan=plan,
            reason="probe_summary",
            probe_summary_payload=probe_summary,
            completed_records=[probe_record],
            directive="\n\n".join(
                directive
                for directive in (typed_plan_directive, trajectory_directive)
                if directive
            )
            or None,
            force=bool(typed_plan_preflight or trajectory_preflight),
        )

    final_typed_plan_findings = _typed_plan_dag_findings(plan)
    if final_typed_plan_findings:
        typed_plan_dag_blocked = True
        findings.extend(final_typed_plan_findings)
        _flush_partial_manifest(
            {
                "typed_plan_dag_blocked": True,
                "typed_plan_dag_error_count": len(final_typed_plan_findings),
            }
        )

    final_trajectory_plan_findings = trajectory_plan_dag_findings(
        plan=plan,
        context=context,
    )
    if final_trajectory_plan_findings:
        trajectory_plan_blocked = True
        findings.extend(final_trajectory_plan_findings)
        _flush_partial_manifest(
            {
                "trajectory_plan_contract_blocked": True,
                "trajectory_plan_contract_error_count": len(
                    final_trajectory_plan_findings
                ),
            }
        )

    shared_lock = threading.Lock()
    step_order = {s.step_id: i for i, s in enumerate(plan.steps)}
    total_steps = len(plan.steps)

    def _record_repair(
        *,
        repair_id: str,
        step_id: str,
        trigger: Dict[str, Any],
        transformation: str,
        before_code: Optional[str] = None,
        after_code: Optional[str] = None,
        selection_rule: Optional[str] = None,
        before_state: Optional[RepairObservedState] = None,
        after_state: Optional[RepairObservedState] = None,
        outcome: str = "applied",
    ) -> None:
        try:
            with repair_ledger_lock:
                provenance = repair_ledger.append_application(
                    repair_id=repair_id,
                    step_id=step_id,
                    trigger=trigger,
                    transformation=transformation,
                    outcome=outcome,
                    model_id=llm_signature,
                    before_text=before_code,
                    after_text=after_code,
                    selection_rule=selection_rule,
                    before_state=before_state,
                    after_state=after_state,
                )
            # P1: a runtime invariant that was actually checked and failed is a
            # non-blocking warning in soft mode; P2 will escalate this to a
            # fail-closed block for STRUCTURAL / CONTRACT_FILL repairs.
            if provenance.invariant_status == InvariantStatus.VERIFIED_FAIL.value:
                findings.append(
                    ValidationFinding(
                        validator="repair_invariant",
                        severity="warning",
                        message=(
                            f"Repair {repair_id} violated declared invariant(s) "
                            f"{list(provenance.invariant_failures)} on step {step_id}."
                        ),
                        detail={
                            "repair_id": repair_id,
                            "step_id": step_id,
                            "repair_class": provenance.repair_class,
                            "invariant_failures": list(provenance.invariant_failures),
                        },
                    )
                )
        except Exception as exc:
            findings.append(
                ValidationFinding(
                    validator="repair_ledger",
                    severity="warning",
                    message=(
                        f"Could not record repair provenance for {repair_id}: {exc}"
                    ),
                    detail={"repair_id": repair_id, "step_id": step_id},
                )
            )

    def _automatic_repair_authorized(
        repair_id: str,
        *,
        step: AnalysisStep,
        source: str,
        before_code: Optional[str] = None,
        after_code: Optional[str] = None,
        sealed_renderer_wrapper: bool = False,
    ) -> bool:
        """Apply the central no-auto-method-substitution policy.

        Code rewrites and artifact/rendering transforms share this boundary. A
        staged figure may be built speculatively, but it is not installed into
        the live step unless this policy authorizes its typed repair id.
        """

        step_id = str(step.step_id)
        if automatic_repair_allowed(
            repair_id,
            step=step,
            sealed_renderer_wrapper=sealed_renderer_wrapper,
        ):
            return True
        sealed_context_denied = is_sealed_renderer_repair(repair_id)
        policy_reason = (
            "sealed_renderer_requires_preexecution_wrapper"
            if sealed_context_denied
            else "method_substitution_default_deny"
        )
        _record_repair(
            repair_id=repair_id,
            step_id=step_id,
            trigger={
                "source": source,
                "automatic_repair_policy": policy_reason,
            },
            transformation=(
                "Candidate repair was not applied because its execution context "
                "did not satisfy the central automatic-repair policy."
            ),
            before_code=before_code,
            after_code=after_code,
            outcome="blocked_by_automatic_repair_policy",
        )
        findings.append(
            ValidationFinding(
                validator="automatic_repair_policy",
                severity="info",
                message=(
                    f"Blocked automatic repair {repair_id} for step {step_id}; "
                    f"policy={policy_reason}."
                ),
                detail={
                    "repair_id": repair_id,
                    "step_id": step_id,
                    "source": source,
                    "policy": policy_reason,
                    "outcome": "blocked_by_automatic_repair_policy",
                },
            )
        )
        return False

    def _authorize_automatic_repair(
        repair: Optional[Tuple[str, str]],
        *,
        step: AnalysisStep,
        source: str,
        before_code: str,
        sealed_renderer_wrapper: bool = False,
    ) -> Optional[Tuple[str, str]]:
        """Authorize a generated code repair before assigning live code."""

        if repair is None:
            return None
        repair_id, candidate_code = repair
        if not _automatic_repair_authorized(
            repair_id,
            step=step,
            source=source,
            before_code=before_code,
            after_code=candidate_code,
            sealed_renderer_wrapper=sealed_renderer_wrapper,
        ):
            return None
        return repair

    def _script_generation_mode(
        *,
        repair_attempts: int,
        fallback_used: bool,
        standard_executor_used: bool = False,
        runner_repair_name: Optional[str] = None,
        resumed_code_reuse: bool = False,
        concept_repair_used: bool = False,
        llm_repair_used: bool = False,
    ) -> str:
        if standard_executor_used:
            return "deterministic_standard"
        # Report the code that actually executed, not merely where its first
        # draft came from. A resumed script that required a fresh LLM repair is
        # repaired code; labelling it as pure reuse hides the model mutation and
        # can incorrectly trigger reuse-only audit shortcuts.
        if llm_repair_used:
            return "repaired"
        if fallback_used:
            return "fallback"
        if runner_repair_name:
            return "runner_repaired"
        if repair_attempts > 0 or concept_repair_used:
            return "repaired"
        if resumed_code_reuse:
            return "resumed_code_reuse"
        return "llm"

    def _propagate_findings_to_evidence(
        evidence_ids: Sequence[str],
        findings_for_step: Sequence[ValidationFinding],
        *,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        # Delegates to the module-level ``scope_findings_to_records`` so the
        # caveat-scoping rule (targeted taint + step-global-error fail-closed,
        # step-global warnings stay advisory) is unit-testable in isolation.
        scoped = scope_findings_to_records(evidence_ids, findings_for_step)
        for evidence_id in evidence_ids:
            severity, messages = scoped[str(evidence_id)]
            evidence.update_record(
                evidence_id,
                finding_severity=severity,
                finding_messages=messages,
                metadata=metadata,
            )

    def _evidence_refs_for_names(
        names: Sequence[str],
    ) -> Tuple[List[EvidenceRef], List[str], Dict[str, Dict[str, Any]]]:
        refs: List[EvidenceRef] = []
        typed_evidence_ids: List[str] = []
        typed_bindings: Dict[str, Dict[str, Any]] = {}
        seen: set[str] = set()
        failures: List[Dict[str, Any]] = []
        for name in names:
            value = str(name)
            if _typed_input_product(value) is not None:
                with shared_lock:
                    records_snapshot = list(per_step_records)
                evidence_snapshot = evidence.records()
                ref, failure = _resolve_typed_input_evidence(
                    input_name=value,
                    plan=plan,
                    evidence_records=evidence_snapshot,
                    per_step_records=records_snapshot,
                    run_dir=run_dir,
                )
                if failure is not None:
                    failures.append(failure)
                    continue
                if ref is not None and ref.evidence_id not in seen:
                    refs.append(ref)
                    seen.add(ref.evidence_id)
                    typed_evidence_ids.append(ref.evidence_id)
                if ref is not None:
                    binding = _resolved_typed_input_binding(
                        input_name=value,
                        evidence_ref=ref,
                        evidence_records=evidence_snapshot,
                        run_dir=run_dir,
                        producer_step_records=records_snapshot,
                        authoritative_cohort_path=cohort_path,
                    )
                    if binding is None:
                        failures.append(
                            {
                                "input": value,
                                "reason": "verified_binding_unavailable",
                            }
                        )
                    else:
                        typed_bindings[value] = binding
                continue

            rec = evidence.get(value)
            if rec is not None and rec.evidence_id not in seen:
                refs.append(
                    EvidenceRef(
                        evidence_id=rec.evidence_id,
                        kind=rec.kind,
                        description=rec.description,
                        relative_path=rec.relative_path,
                    )
                )
                seen.add(rec.evidence_id)
        if failures:
            raise _EvidenceLineageResolutionError(failures)
        return refs, typed_evidence_ids, typed_bindings

    def _validator_messages(
        *finding_groups: Sequence[ValidationFinding],
    ) -> List[str]:
        return _actionable_validator_messages(*finding_groups)

    def _failed_dependency_record(step: AnalysisStep) -> Optional[Dict[str, Any]]:
        parent_step_id = _parent_step_id_for_figure_step(step)
        if parent_step_id is None:
            return None
        with shared_lock:
            records = list(per_step_records)
        latest = {
            str(record.get("step_id") or ""): record
            for record in current_step_records(records)
        }
        record = latest.get(parent_step_id)
        if record is not None:
            if str(record.get("status") or "").lower() == "ok":
                return None
            return dict(record)
        return None

    def _execute_one_step(step: AnalysisStep) -> Dict[str, Any]:
        nonlocal runtime_state
        with shared_lock:
            resume_history = (
                list(
                    plan_result.resume_state.get("step_attempt_history")
                    or plan_result.resume_state.get("per_step_records")
                    or []
                )
                if isinstance(plan_result.resume_state, Mapping)
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
            max(prior_attempt_sequences, default=len(prior_attempt_records)) + 1
        )
        attempt_id = f"{run_id}:{step.step_id}:{attempt_sequence}"
        review_checkpoint_id = f"{attempt_id}:deterministic_review"
        step_record: Dict[str, Any] = {
            "step_id": step.step_id,
            "intent": step.intent,
            "attempt_id": attempt_id,
            "attempt_sequence": attempt_sequence,
            "review_checkpoint_id": review_checkpoint_id,
            "plan_scientific_signature": (
                _serializable_plan_scientific_scope_signature(plan)
            ),
        }
        (
            step_llm_repair_attempts,
            prior_repair_classes,
            repair_history_invalid,
        ) = _monotonic_step_llm_repair_history(
            prior_attempt_records,
            limit=pipeline._max_step_llm_repair_attempts,
        )
        if step_llm_repair_attempts:
            step_record["step_llm_repair_attempts"] = step_llm_repair_attempts
            step_record["step_llm_repair_budget"] = (
                pipeline._max_step_llm_repair_attempts
            )
        if prior_repair_classes:
            step_record["step_llm_repair_classes"] = list(prior_repair_classes)
        if repair_history_invalid:
            step_record["step_llm_repair_history_invalid"] = True
            step_record["step_llm_repair_budget_exhausted"] = True
        configured_provider_limit = pipeline._max_step_provider_calls
        effective_provider_limit = configured_provider_limit
        provider_receipt_path = provider_call_budget_receipt_path(
            run_dir,
            step_id=step.step_id,
        )
        provider_receipt_relative_path = str(provider_receipt_path.relative_to(run_dir))
        prior_provider_categories: tuple[str, ...] = ()
        prior_provider_attempts = 0
        provider_receipt_integrity_error: Optional[str] = None
        prior_snapshot_present = False
        if isinstance(prior_step_record, Mapping):
            snapshot_keys = {
                "step_provider_call_budget",
                "step_provider_call_attempts",
                "step_provider_call_categories",
            }
            prior_snapshot_present = any(
                key in prior_step_record for key in snapshot_keys
            )
            if prior_snapshot_present:
                prior_limit = prior_step_record.get("step_provider_call_budget")
                prior_attempts_raw = prior_step_record.get(
                    "step_provider_call_attempts"
                )
                prior_categories_raw = prior_step_record.get(
                    "step_provider_call_categories"
                )
                if (
                    isinstance(prior_limit, bool)
                    or not isinstance(prior_limit, int)
                    or prior_limit < 0
                    or isinstance(prior_attempts_raw, bool)
                    or not isinstance(prior_attempts_raw, int)
                    or prior_attempts_raw < 0
                    or not isinstance(prior_categories_raw, list)
                ):
                    provider_receipt_integrity_error = (
                        "Prior provider-call budget snapshot is incomplete or invalid."
                    )
                else:
                    normalized_categories = tuple(
                        str(item).strip() for item in prior_categories_raw
                    )
                    if any(
                        not item for item in normalized_categories
                    ) or prior_attempts_raw != len(normalized_categories):
                        provider_receipt_integrity_error = "Prior provider-call attempts and category history disagree."
                    else:
                        prior_provider_attempts = prior_attempts_raw
                        prior_provider_categories = normalized_categories
                        effective_provider_limit = min(
                            effective_provider_limit,
                            prior_limit,
                        )

        if provider_receipt_integrity_error is None and provider_receipt_path.exists():
            try:
                receipt_limit, receipt_categories = load_provider_call_budget_receipt(
                    provider_receipt_path,
                    step_id=step.step_id,
                )
                effective_provider_limit = min(
                    effective_provider_limit,
                    receipt_limit,
                )
                if prior_snapshot_present and (
                    len(receipt_categories) < len(prior_provider_categories)
                    or receipt_categories[: len(prior_provider_categories)]
                    != prior_provider_categories
                ):
                    raise ProviderCallBudgetReceiptError(
                        "Durable provider-call receipt conflicts with the latest "
                        "step snapshot."
                    )
                prior_provider_categories = receipt_categories
                prior_provider_attempts = len(receipt_categories)
            except ProviderCallBudgetReceiptError as exc:
                provider_receipt_integrity_error = str(exc)
        elif (
            provider_receipt_integrity_error is None
            and isinstance(prior_step_record, Mapping)
            and prior_step_record.get("step_provider_call_receipt_version") == 1
            and prior_provider_attempts > 0
        ):
            provider_receipt_integrity_error = (
                "Durable provider-call receipt is missing for a paid prior attempt."
            )

        provider_budget = StepProviderCallBudget(
            effective_provider_limit,
            step_id=step.step_id,
            consumed_categories=prior_provider_categories,
            receipt_path=provider_receipt_path,
        )

        def _sync_provider_budget() -> None:
            snapshot = provider_budget.snapshot()
            step_record["step_provider_call_budget_scope"] = (
                "coder_generation_repair_concept_audit_and_analyzer"
            )
            step_record["step_provider_call_budget"] = snapshot["limit"]
            step_record["step_provider_call_attempts"] = snapshot["used"]
            step_record["step_provider_call_remaining"] = snapshot["remaining"]
            step_record["step_provider_call_budget_exhausted"] = snapshot["exhausted"]
            step_record["step_provider_call_categories"] = snapshot["categories"]
            step_record["step_provider_call_receipt_version"] = 1
            step_record["step_provider_call_receipt"] = (
                provider_receipt_relative_path if snapshot["used"] else None
            )

        _sync_provider_budget()
        if provider_receipt_integrity_error is not None:
            step_record.update(
                {
                    "status": "contract_failed",
                    "generation_mode": "system",
                    "provider_call_budget_receipt_invalid": True,
                    "provider_call_budget_receipt_error": (
                        provider_receipt_integrity_error
                    ),
                }
            )
            receipt_finding = ValidationFinding(
                validator="provider_call_budget_receipt",
                severity="error",
                message=(
                    f"Step {step.step_id} cannot resume because its durable "
                    "provider-call receipt is missing, corrupt, or inconsistent."
                ),
                detail={
                    "step_id": step.step_id,
                    "receipt_path": provider_receipt_relative_path,
                    "reason": provider_receipt_integrity_error,
                },
            )
            with shared_lock:
                findings.append(receipt_finding)
                per_step_records.append(step_record)
                _flush_partial_manifest()
            emit_progress(
                "step",
                f"Step {step.step_id} failed closed: provider-call receipt invalid.",
                status="failed",
                run_id=run_id,
                step_id=step.step_id,
                current_step=step_order.get(step.step_id, 0) + 1,
                total_steps=total_steps,
            )
            return step_record
        coder_context = _coder_context_with_locked_robustness_specs(
            context=agent_context,
            step=step,
            run_dir=run_dir,
        )
        resumed_code_reuse_used = False
        critic_resume_repair_used = False
        resumed_quarantined_draft_used = False
        quarantined_draft_active = False
        quarantined_repair_materially_changed = False
        quarantined_repair_succeeded = False
        quarantine_superseded_by_fallback = False
        quarantine_policy_superseded = False
        pending_quarantined_errors: List[ValidationFinding] = []

        def _llm_repair_budget_available() -> bool:
            return step_llm_repair_attempts < pipeline._max_step_llm_repair_attempts

        def _consume_llm_repair_budget(repair_class: str) -> bool:
            nonlocal step_llm_repair_attempts
            if not _llm_repair_budget_available():
                step_record["step_llm_repair_budget_exhausted"] = True
                step_record["step_llm_repair_budget"] = (
                    pipeline._max_step_llm_repair_attempts
                )
                return False
            step_llm_repair_attempts += 1
            step_record["step_llm_repair_attempts"] = step_llm_repair_attempts
            step_record["step_llm_repair_budget"] = (
                pipeline._max_step_llm_repair_attempts
            )
            step_record.setdefault("step_llm_repair_classes", []).append(
                str(repair_class)
            )
            return True

        monotonic_concept_constraints = _persisted_monotonic_concept_constraints(
            prior_step_record
        )
        if monotonic_concept_constraints:
            step_record["monotonic_concept_constraints"] = [
                finding.model_dump(mode="json")
                for finding in monotonic_concept_constraints
            ]
        preexecution_runner_repair_name: Optional[str] = None
        runner_repair_name: Optional[str] = None
        sealed_renderer_repair_id: Optional[str] = None
        sealed_renderer_authorized_code_sha256: Optional[str] = None
        sealed_renderer_implementation_sha256: Optional[str] = None
        sealed_renderer_parent_digests: Dict[str, str] = {}
        sealed_renderer_authorized_product_slots: Dict[str, str] = {}
        step_current = step_order.get(step.step_id, 0) + 1
        dependency_record = _failed_dependency_record(step)
        if dependency_record is not None:
            parent_step_id = str(dependency_record.get("step_id") or "")
            step_record.update(
                {
                    "status": "skipped_dependency_failed",
                    "dependency_step_id": parent_step_id,
                    "diagnostic_only": True,
                    "generation_mode": "system",
                }
            )
            with shared_lock:
                findings.append(
                    ValidationFinding(
                        validator="dependency_gate",
                        severity="warning",
                        message=(
                            f"Skipped downstream figure step {step.step_id} because "
                            f"required analysis step {parent_step_id} did not pass."
                        ),
                        detail={
                            "step_id": step.step_id,
                            "dependency_step_id": parent_step_id,
                            "dependency_status": dependency_record.get("status"),
                            "diagnostic_only": True,
                        },
                    )
                )
                per_step_records.append(step_record)
                _flush_partial_manifest()
            emit_progress(
                "step",
                f"Skipped {step.step_id}; required step {parent_step_id} failed.",
                status="skipped",
                run_id=run_id,
                step_id=step.step_id,
                current_step=step_current,
                total_steps=total_steps,
            )
            return step_record
        emit_progress(
            "step",
            f"Step {step_current}/{total_steps} started: {step.step_id}.",
            run_id=run_id,
            step_id=step.step_id,
            current_step=step_current,
            total_steps=total_steps,
        )
        locked_measurement_findings = (
            step_summary_integrity_validator.audit_locked_measurement_data_quality(
                step=step,
                cohort_path=cohort_path,
            )
        )
        locked_measurement_issues = _locked_measurement_data_quality_issues(
            locked_measurement_findings
        )
        if locked_measurement_issues:
            step_record.update(
                {
                    "status": "contract_failed",
                    "diagnostic_only": True,
                    "measurement_provenance_preflight": True,
                    "measurement_provenance_repair_suppressed": True,
                    "measurement_provenance_terminal_reason": (
                        "locked_cohort_data_quality_failed"
                    ),
                    "measurement_provenance_terminal_issues": (
                        locked_measurement_issues
                    ),
                    "contract_findings": [
                        finding.model_dump() for finding in locked_measurement_findings
                    ],
                    "step_summary": {},
                    "llm_repair_used": False,
                    "generation_mode": "system",
                    "code_repair_attempts": 0,
                    "contract_repair_attempts": 0,
                }
            )
            with shared_lock:
                findings.extend(locked_measurement_findings)
                per_step_records.append(step_record)
                _flush_partial_manifest()
            emit_progress(
                "contract",
                (
                    "Locked-cohort measurement provenance failed before code "
                    f"generation for {step.step_id}; retained diagnostics "
                    "without attempting a repair."
                ),
                status="error",
                run_id=run_id,
                step_id=step.step_id,
                current_step=step_current,
                total_steps=total_steps,
            )
            return step_record
        try:
            (
                existing_refs,
                resolved_input_evidence_ids,
                resolved_input_bindings,
            ) = _evidence_refs_for_names(step.inputs)
        except _EvidenceLineageResolutionError as exc:
            step_record.update(
                {
                    "status": "blocked_dependency_evidence",
                    "diagnostic_only": True,
                    "generation_mode": "system",
                    "evidence_lineage_failures": exc.failures,
                }
            )
            lineage_finding = ValidationFinding(
                validator="typed_artifact_evidence_lineage",
                severity="error",
                message=(
                    f"Step {step.step_id} was blocked because one or more typed "
                    "artifact inputs lack a unique, current, digest-verified "
                    "producer output."
                ),
                detail={"step_id": step.step_id, "failures": exc.failures},
            )
            with shared_lock:
                findings.append(lineage_finding)
                per_step_records.append(step_record)
                _flush_partial_manifest()
            emit_progress(
                "audit",
                f"Blocked {step.step_id}; typed artifact evidence is unresolved.",
                status="error",
                run_id=run_id,
                step_id=step.step_id,
                current_step=step_current,
                total_steps=total_steps,
            )
            return step_record
        resolved_inputs_path = _write_resolved_inputs_manifest(
            run_dir=run_dir,
            step_id=step.step_id,
            bindings=resolved_input_bindings,
            context_path=plan_result.context_path,
        )
        step_record["resolved_inputs_path"] = str(
            resolved_inputs_path.relative_to(run_dir)
        )
        step_record["resolved_input_evidence_ids"] = list(resolved_input_evidence_ids)
        local_runtime_state = supervisor.prepare_step_state(
            state=runtime_state,
            context=context,
            step=step,
            evidence_refs=existing_refs,
        )
        step_record["analysis_request"] = (
            local_runtime_state.analysis_request.model_dump(mode="json")
            if local_runtime_state.analysis_request is not None
            else None
        )
        step_record["visualization_request"] = (
            local_runtime_state.visualization_request.model_dump(mode="json")
            if local_runtime_state.visualization_request is not None
            else None
        )
        step_record["semantics_family"] = local_runtime_state.analysis_family

        deterministic_fallback_used = False
        deterministic_standard_executor_used = False

        def _remember_concept_constraints(
            candidates: Sequence[ValidationFinding],
        ) -> None:
            """Keep repaired scientific defects binding across later repairs."""

            monotonic_concept_constraints[:] = _merge_monotonic_concept_constraints(
                monotonic_concept_constraints,
                candidates,
            )
            if monotonic_concept_constraints:
                step_record["monotonic_concept_constraints"] = [
                    finding.model_dump(mode="json")
                    for finding in monotonic_concept_constraints
                ]

        def _quarantine_error_payloads(
            candidates: Sequence[ValidationFinding],
        ) -> List[Dict[str, Any]]:
            """Serialize the complete cross-repair constraint set for resume."""

            _remember_concept_constraints(candidates)
            return [
                finding.model_dump(mode="json")
                for finding in monotonic_concept_constraints
            ]

        def _monotonic_concept_constraint_log() -> str:
            if not monotonic_concept_constraints:
                return ""
            payload = [
                {
                    "validator": finding.validator,
                    "message": finding.message,
                    "detail": dict(finding.detail or {}),
                }
                for finding in monotonic_concept_constraints
            ]
            return (
                "\n\nPREVIOUSLY REPAIRED CONCEPT FINDINGS (binding regression "
                "constraints; do not reintroduce them):\n"
                + json.dumps(payload, indent=2, ensure_ascii=False, default=str)
            )

        def _use_quarantined_draft(draft: QuarantinedConceptDraft) -> str:
            nonlocal resumed_quarantined_draft_used
            nonlocal quarantined_draft_active
            nonlocal pending_quarantined_errors
            resumed_quarantined_draft_used = True
            quarantined_draft_active = True
            pending_quarantined_errors = [
                ValidationFinding.model_validate(payload) for payload in draft.findings
            ]
            step_record["resumed_quarantined_draft"] = True
            step_record["quarantined_draft_sha256"] = draft.sha256
            step_record["quarantined_draft_relative_path"] = draft.relative_path
            step_record["quarantined_requires_repair"] = True
            step_record["quarantined_repair_succeeded"] = False
            emit_progress(
                "coder",
                f"Resuming rejected draft for mandatory repair: {step.step_id}.",
                status="warning",
                run_id=run_id,
                step_id=step.step_id,
                current_step=step_current,
                total_steps=total_steps,
            )
            return draft.code

        def _use_resumed_code(
            resumed_code: Tuple[str, Dict[str, Any]],
            *,
            error: Optional[BaseException] = None,
        ) -> str:
            nonlocal resumed_code_reuse_used
            resumed_code_reuse_used = True
            prior_code, resumed_record = resumed_code
            step_record["generation_mode"] = "resumed_code_reuse"
            step_record["resumed_code_evidence_id"] = resumed_record.get("evidence_id")
            step_record["resumed_code_relative_path"] = resumed_record.get(
                "relative_path"
            )
            resumed_evidence_generation_mode = str(
                resumed_record.get("generation_mode") or ""
            )
            resumed_from_generation_mode = resumed_evidence_generation_mode
            if resumed_evidence_generation_mode == "resumed_code_reuse":
                resumed_metadata = resumed_record.get("metadata")
                if isinstance(resumed_metadata, dict):
                    resumed_from_generation_mode = str(
                        resumed_metadata.get("resumed_from_generation_mode") or ""
                    )
            step_record["resumed_code_evidence_generation_mode"] = (
                resumed_evidence_generation_mode
            )
            step_record["resumed_from_generation_mode"] = resumed_from_generation_mode
            detail = {
                "step_id": step.step_id,
                "resume_from_step_id": requested_resume_from_step_id,
                "evidence_id": resumed_record.get("evidence_id"),
                "relative_path": resumed_record.get("relative_path"),
                "resumed_from_generation_mode": resumed_from_generation_mode,
            }
            if error is None:
                message = (
                    "Explicit resume reused prior agent-generated code "
                    f"(source mode: {resumed_from_generation_mode}) for step "
                    f"{step.step_id} before requesting a new coder script."
                )
            else:
                detail["error"] = str(error)
                message = (
                    f"Coder agent failed for step {step.step_id}; reused prior "
                    "agent-generated code from resume evidence "
                    f"(source mode: {resumed_from_generation_mode})."
                )
            with shared_lock:
                findings.append(
                    ValidationFinding(
                        validator="coder",
                        severity="warning",
                        message=message,
                        detail=detail,
                    )
                )
            emit_progress(
                "coder",
                f"Reused prior generated analysis script for {step.step_id}.",
                status="warning",
                run_id=run_id,
                step_id=step.step_id,
                current_step=step_current,
                total_steps=total_steps,
            )
            return prior_code

        def _resume_summary_repair_code() -> Optional[str]:
            nonlocal preexecution_runner_repair_name
            if (
                requested_resume_from_step_id != step.step_id
                or not pipeline._enable_deterministic_runner_repair
            ):
                return None
            resumed_code = resume_controller.prior_code_for_step(step.step_id)
            if resumed_code is None:
                return None
            prior_code, _resumed_record = resumed_code
            prior_summary_path = (
                run_dir / "steps" / step.step_id / "outputs" / "step_summary.json"
            )
            if not prior_summary_path.exists():
                return None
            try:
                prior_summary = json.loads(
                    prior_summary_path.read_text(encoding="utf-8")
                )
            except Exception:
                return None
            if not isinstance(prior_summary, dict) or not prior_summary:
                return None
            repair = _deterministic_summary_repair(
                code=prior_code,
                step_summary=prior_summary,
                previous_repair=None,
                analysis_family=(
                    local_runtime_state.analysis_family
                    or prior_summary.get("analysis_family")
                ),
            )
            repair = _authorize_automatic_repair(
                repair,
                step=step,
                source="resume_summary_repair_preflight",
                before_code=prior_code,
            )
            if repair is None:
                return None
            repair_name, repaired_code = repair
            _use_resumed_code(resumed_code)
            preexecution_runner_repair_name = repair_name
            step_record["runner_repair"] = repair_name
            step_record["resume_summary_repair"] = repair_name
            _record_repair(
                repair_id=repair_name,
                step_id=step.step_id,
                trigger={
                    "source": "resume_summary_repair_preflight",
                    "step_summary_path": str(prior_summary_path),
                    "step_summary_keys": sorted(str(k) for k in prior_summary),
                },
                transformation=(
                    "Reused the explicitly resumed step's prior generated code "
                    "after deterministic summary repair, before requesting a "
                    "new coder script."
                ),
                before_code=prior_code,
                after_code=repaired_code,
                selection_rule=(
                    "only when the prior step_summary triggers a case-neutral "
                    "deterministic summary repair"
                ),
            )
            with shared_lock:
                findings.append(
                    ValidationFinding(
                        validator="coder",
                        severity="info",
                        message=(
                            f"Applied deterministic resume-summary repair for "
                            f"step {step.step_id}: {repair_name}."
                        ),
                        detail={
                            "step_id": step.step_id,
                            "repair_id": repair_name,
                            "step_summary_path": str(prior_summary_path),
                        },
                    )
                )
            emit_progress(
                "runner_repair",
                (
                    f"Applied deterministic resume-summary repair for "
                    f"{step.step_id}: {repair_name}."
                ),
                run_id=run_id,
                step_id=step.step_id,
                current_step=step_current,
                total_steps=total_steps,
            )
            return repaired_code

        def _resume_critic_repair_code() -> Optional[str]:
            """Repair the selected prior script from structured Critic feedback."""

            nonlocal critic_resume_repair_used
            report = resume_controller.prior_negative_critic_report_for_step(
                step.step_id
            )
            if report is None:
                return None
            resumed_code = resume_controller.prior_code_for_step(step.step_id)
            if resumed_code is None:
                return None
            if not _consume_llm_repair_budget("critic_resume"):
                return None
            prior_code = _use_resumed_code(resumed_code)
            critique_log = (
                "PRIOR CRITIC REVIEW (binding repair requirements):\n"
                + json.dumps(report, indent=2, ensure_ascii=False, default=str)
            )
            emit_progress(
                "coder",
                f"Repairing prior Critic findings for {step.step_id}.",
                status="warning",
                run_id=run_id,
                step_id=step.step_id,
                current_step=step_current,
                total_steps=total_steps,
            )
            try:
                repaired = coder.repair(
                    context=coder_context,
                    step=step,
                    code=prior_code,
                    run_log=critique_log,
                    attempt=1,
                    provider_budget=provider_budget,
                    provider_category="critic_resume_repair",
                )
                _sync_provider_budget()
            except Exception as exc:
                _sync_provider_budget()
                with shared_lock:
                    findings.append(
                        ValidationFinding(
                            validator="critic_resume_repair",
                            severity="warning",
                            message=(
                                "Prior Critic-guided repair was unavailable; "
                                "falling back to ordinary code generation."
                            ),
                            detail={
                                "step_id": step.step_id,
                                "error_type": type(exc).__name__,
                                "error": str(exc)[:300],
                            },
                        )
                    )
                return None
            critic_resume_repair_used = True
            step_record["critic_resume_repair"] = True
            step_record["critic_resume_repair_status"] = report.get("status")
            return repaired

        def _publication_figure_preflight_supported() -> bool:
            # Preflight may replace the coder, so names/prose are insufficient.
            # Claim only a split figure whose direct parent recorded a controlled
            # figure-data family, exact method, or analysis family.  Legacy name
            # routing remains available only after an agent figure fails QA.
            if not _step_has_figure_only_output_contract(step):
                return False
            return deterministic_figure_family_supported_for_upstream(
                run_dir, step.step_id
            )

        def _deterministic_publication_figure_code(
            reason: str,
        ) -> Optional[str]:
            nonlocal deterministic_fallback_used, preexecution_runner_repair_name
            nonlocal sealed_renderer_repair_id
            nonlocal sealed_renderer_implementation_sha256
            nonlocal sealed_renderer_parent_digests
            nonlocal sealed_renderer_authorized_product_slots
            exact_repair_id = deterministic_figure_repair_id_for_upstream(
                run_dir, step.step_id
            )
            if (
                deterministic_fallback_used
                or not pipeline._enable_deterministic_runner_repair
                or not _step_has_figure_only_output_contract(step)
                or exact_repair_id is None
            ):
                return None
            sealed_renderer = is_sealed_renderer_repair(exact_repair_id)
            declared_figure_products = list(step.expected_outputs or [])
            sealed_source_digests: Dict[str, str] = {}
            sealed_implementation_digest = ""
            sealed_product_slots: Dict[str, str] = {}
            if sealed_renderer:
                typed_products = _sealed_typed_figure_products(declared_figure_products)
                if typed_products is None:
                    # Legacy bare exports are file requirements, not logical
                    # Planner product roles.  They retain the ordinary coder
                    # path rather than entering a sealed binder that cannot
                    # prove their semantics.
                    return None
                declared_figure_products = typed_products
                try:
                    sealed_source_digests = _sealed_renderer_source_digests(
                        exact_repair_id
                    )
                    sealed_implementation_digest = (
                        _sealed_renderer_implementation_digest(sealed_source_digests)
                    )
                except (ImportError, OSError, ValueError):
                    return None
                if not _sealed_renderer_figure_step_matches_parent(
                    run_dir,
                    step,
                    exact_repair_id,
                ):
                    return None
            sealed_parent_digests: Optional[Dict[str, str]] = None
            if sealed_renderer:
                sealed_parent_digests = _sealed_renderer_parent_digest_seal(
                    run_dir,
                    step.step_id,
                    exact_repair_id,
                )
                if not sealed_parent_digests:
                    return None
                try:
                    primary_descriptor = (
                        agent_context.variable(agent_context.primary_exposure)
                        if agent_context.primary_exposure
                        else None
                    )
                    sealed_product_slots = authorize_declared_figure_product_slots(
                        declared_products=declared_figure_products,
                        renderer_repair_id=exact_repair_id,
                        planner_parent_anchors=_sealed_parent_planner_anchors(
                            run_dir=run_dir,
                            figure_step_id=step.step_id,
                        ),
                        authoritative_display_subjects=(
                            [
                                value
                                for value in (
                                    agent_context.primary_exposure,
                                    (
                                        primary_descriptor.description
                                        if primary_descriptor is not None
                                        else None
                                    ),
                                )
                                if value
                            ]
                        ),
                    )
                except ValueError:
                    return None
            if exact_repair_id == (
                "distribution_availability_publication_bundle_from_parent_outputs_v1"
            ):
                if not _distribution_availability_figure_step_matches_parent(
                    run_dir, step
                ):
                    return None
            candidate_code = """
import hashlib
import importlib
import json
import os
from pathlib import Path

out_dir = Path(os.environ["STEP_OUT_DIR"])
run_dir = out_dir.parents[2]
current_step_id = out_dir.parent.name

expected_source_digests = __EXPECTED_SOURCE_DIGESTS__
loaded_modules = {}
actual_source_digests = {}
for module_name, expected_digest in expected_source_digests.items():
    module = importlib.import_module(module_name)
    module_path = Path(module.__file__)
    actual_digest = hashlib.sha256(module_path.read_bytes()).hexdigest()
    loaded_modules[module_name] = module
    actual_source_digests[module_name] = actual_digest
if actual_source_digests != expected_source_digests:
    raise RuntimeError(
        "A sealed renderer implementation module changed after authorization."
    )

pipeline_module = loaded_modules.get("easyicu.research_agent.pipeline")
if pipeline_module is None:
    pipeline_module = importlib.import_module("easyicu.research_agent.pipeline")
expected_repair_id = __EXPECTED_REPAIR_ID__
if __IS_SEALED_RENDERER__:
    render_publication_bundle = getattr(
        pipeline_module,
        "_render_authorized_sealed_publication_bundle",
    )
    repair_id = render_publication_bundle(
        repair_id=expected_repair_id,
        run_dir=run_dir,
        current_step_id=current_step_id,
        out_dir=out_dir,
        parent_artifact_digests=__PREVERIFIED_PARENT_DIGESTS__,
    )
else:
    render_publication_bundle = getattr(
        pipeline_module,
        "_render_publication_bundle_from_prior_outputs_for_step",
    )
    repair_id = render_publication_bundle(
        run_dir=run_dir,
        current_step_id=current_step_id,
        out_dir=out_dir,
        preverified_parent_digests=__PREVERIFIED_PARENT_DIGESTS__,
    )

if repair_id != expected_repair_id:
    summary = {
        "rendering_only": True,
        "deterministic_publication_figure_rescue": "typed_renderer_mismatch",
        "expected_repair_id": expected_repair_id,
        "observed_repair_id": repair_id,
        "figure_files": [],
        "warning": "The evidence-bound renderer did not return its authorized repair id.",
    }
    with open(out_dir / "step_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
else:
    if __IS_SEALED_RENDERER__:
        contract_module = loaded_modules[
            "easyicu.research_agent.declared_product_contract"
        ]
        bind_declared_figure_products = getattr(
            contract_module,
            "bind_declared_figure_products",
        )
        bind_declared_figure_products(
            out_dir=out_dir,
            declared_products=__DECLARED_FIGURE_PRODUCTS__,
            authorized_product_slots=__AUTHORIZED_PRODUCT_SLOTS__,
            renderer_repair_id=expected_repair_id,
            renderer_implementation_sha256=__IMPLEMENTATION_DIGEST__,
            renderer_parent_digests=__PREVERIFIED_PARENT_DIGESTS__,
        )
    print(json.dumps({"deterministic_publication_figure_rescue": repair_id}))
"""
            candidate_code = candidate_code.replace(
                "__EXPECTED_REPAIR_ID__", repr(exact_repair_id)
            )
            candidate_code = candidate_code.replace(
                "__PREVERIFIED_PARENT_DIGESTS__",
                repr(
                    dict(sorted(sealed_parent_digests.items()))
                    if sealed_parent_digests is not None
                    else None
                ),
            )
            candidate_code = candidate_code.replace(
                "__DECLARED_FIGURE_PRODUCTS__",
                repr(declared_figure_products),
            )
            candidate_code = candidate_code.replace(
                "__AUTHORIZED_PRODUCT_SLOTS__",
                repr(dict(sorted(sealed_product_slots.items()))),
            )
            candidate_code = candidate_code.replace(
                "__EXPECTED_SOURCE_DIGESTS__",
                repr(sealed_source_digests),
            )
            candidate_code = candidate_code.replace(
                "__IS_SEALED_RENDERER__",
                repr(sealed_renderer),
            )
            candidate_code = candidate_code.replace(
                "__IMPLEMENTATION_DIGEST__",
                repr(sealed_implementation_digest),
            )
            repair_id = exact_repair_id
            authorized = _authorize_automatic_repair(
                (repair_id, candidate_code),
                step=step,
                source=reason,
                before_code="",
                sealed_renderer_wrapper=sealed_renderer,
            )
            if authorized is None:
                return None
            deterministic_fallback_used = True
            preexecution_runner_repair_name = repair_id
            if sealed_renderer:
                sealed_renderer_repair_id = repair_id
                sealed_renderer_implementation_sha256 = sealed_implementation_digest
                sealed_renderer_parent_digests = dict(
                    sorted((sealed_parent_digests or {}).items())
                )
                sealed_renderer_authorized_product_slots = dict(
                    sorted(sealed_product_slots.items())
                )
                step_record["sealed_renderer_repair"] = repair_id
                step_record["post_execution_mutation_policy"] = "audit_only"
                step_record["sealed_renderer_source_digests"] = dict(
                    sealed_source_digests
                )
                step_record["sealed_renderer_implementation_sha256"] = (
                    sealed_implementation_digest
                )
                step_record["sealed_renderer_parent_digests"] = dict(
                    sealed_renderer_parent_digests
                )
                step_record["sealed_renderer_authorized_product_slots"] = dict(
                    sealed_renderer_authorized_product_slots
                )
                step_record["planner_product_slot_binding_source"] = (
                    "planner_parent_typed_product_prefix_v2"
                )
            step_record["deterministic_code_fallback"] = reason
            step_record["runner_repair"] = repair_id
            _record_repair(
                repair_id=repair_id,
                step_id=step.step_id,
                trigger={"source": reason},
                transformation=(
                    "Executed a rendering-only adapter over the typed direct "
                    "parent outputs; no estimand, cohort, or method was selected."
                ),
                before_code="",
                after_code=candidate_code,
            )
            return authorized[1]

        def _absolute_risk_context_preflight_supported() -> bool:
            if _step_expects_figure(step):
                return False
            return _absolute_risk_context_runner_owns_step(
                str(step.method or ""),
                str(step.step_id or ""),
                step.expected_outputs or [],
            )

        def _deterministic_absolute_risk_context_code(
            reason: str,
            *,
            preflight: bool = False,
        ) -> Optional[str]:
            nonlocal deterministic_fallback_used
            if (
                deterministic_fallback_used
                or not pipeline._enable_deterministic_runner_repair
                or (preflight and not _absolute_risk_context_preflight_supported())
            ):
                return None
            if not _absolute_risk_context_preflight_supported():
                return None
            deterministic_fallback_used = True
            step_record["deterministic_code_fallback"] = reason
            step_record["deterministic_standard_analysis"] = "absolute_risk_context"
            emit_progress(
                "coder",
                f"Using deterministic absolute-risk context runner for {step.step_id}.",
                run_id=run_id,
                step_id=step.step_id,
                current_step=step_current,
                total_steps=total_steps,
                fallback_reason=reason,
            )
            return absolute_risk_context_code()

        def _robustness_sensitivity_preflight_supported() -> bool:
            if _step_expects_figure(step):
                return False
            return _robustness_sensitivity_runner_owns_step(
                str(step.method or ""),
                str(step.step_id or ""),
                step.expected_outputs or [],
            )

        def _deterministic_robustness_sensitivity_code(
            reason: str,
            *,
            preflight: bool = False,
        ) -> Optional[str]:
            nonlocal deterministic_fallback_used
            if (
                deterministic_fallback_used
                or not pipeline._enable_deterministic_runner_repair
                or (preflight and not _robustness_sensitivity_preflight_supported())
            ):
                return None
            if not _robustness_sensitivity_preflight_supported():
                return None
            deterministic_fallback_used = True
            step_record["deterministic_code_fallback"] = reason
            step_record["deterministic_standard_analysis"] = "robustness_sensitivity"
            emit_progress(
                "coder",
                f"Using deterministic robustness runner for {step.step_id}.",
                run_id=run_id,
                step_id=step.step_id,
                current_step=step_current,
                total_steps=total_steps,
                fallback_reason=reason,
            )
            return robustness_sensitivity_preflight_code()

        def _missingness_audit_preflight_supported() -> bool:
            """True for a missingness / measurement-process AUDIT step.

            The audit is a pure per-concept count (measured vs missing fraction +
            structural-vs-measurement split); the LLM coder reliably exhausted its
            retry budget on it (~27.6 min then fail). The deterministic runner owns
            it so the audit never blocks the run. It must NOT claim a figure step
            nor a primary result step that merely mentions missingness. Trigger is
            case-neutral (the controlled ``method`` first, then audit vocabulary).
            """
            if _step_expects_figure(step):
                return False
            return _simple_missingness_audit_runner_owns_step(
                str(step.method or ""),
                str(step.step_id or ""),
                str(step.intent or ""),
                step.expected_outputs or [],
            )

        def _deterministic_missingness_audit_code(
            reason: str,
            *,
            preflight: bool = False,
        ) -> Optional[str]:
            nonlocal deterministic_fallback_used
            if (
                deterministic_fallback_used
                or not pipeline._enable_deterministic_runner_repair
                or (preflight and not _missingness_audit_preflight_supported())
            ):
                return None
            if not _missingness_audit_preflight_supported():
                return None
            deterministic_fallback_used = True
            step_record["deterministic_code_fallback"] = reason
            step_record["deterministic_standard_analysis"] = (
                "missingness_measurement_audit"
            )
            emit_progress(
                "coder",
                f"Using deterministic missingness/measurement audit runner for {step.step_id}.",
                run_id=run_id,
                step_id=step.step_id,
                current_step=step_current,
                total_steps=total_steps,
                fallback_reason=reason,
            )
            return missingness_measurement_audit_code()

        def _trajectory_stability_preflight_supported() -> bool:
            return trajectory_stability_executor_owns_step(step, plan=plan)

        def _deterministic_trajectory_stability_code(
            reason: str,
            *,
            preflight: bool = False,
        ) -> Optional[str]:
            nonlocal deterministic_standard_executor_used
            if deterministic_standard_executor_used or (
                preflight and not _trajectory_stability_preflight_supported()
            ):
                return None
            if not _trajectory_stability_preflight_supported():
                return None
            deterministic_standard_executor_used = True
            step_record["deterministic_standard_selection_reason"] = reason
            step_record["deterministic_standard_analysis"] = (
                "trajectory_cluster_stability"
            )
            emit_progress(
                "coder",
                f"Using planner-specified trajectory stability executor for {step.step_id}.",
                run_id=run_id,
                step_id=step.step_id,
                current_step=step_current,
                total_steps=total_steps,
                fallback_reason=reason,
            )
            return trajectory_stability_executor_code(step, plan=plan)

        # ``--resume-from-step-id`` means the selected step is intentionally
        # rerun. Completed predecessors stay checkpointed, but the selected
        # step does not reuse its old script before the current coder/
        # deterministic-standard path unless the operator explicitly enables
        # the diagnostic fast path. Reused code still runs through every
        # current execution audit and repair gate.
        preflight_trajectory_stability_code = _deterministic_trajectory_stability_code(
            "trajectory_stability_spec_preflight", preflight=True
        )
        preflight_figure_code = (
            None
            if preflight_trajectory_stability_code is not None
            else _deterministic_publication_figure_code(
                "publication_figure_parent_outputs_preflight"
            )
        )
        quarantined_resume_draft = (
            None
            if (
                preflight_trajectory_stability_code is not None
                or preflight_figure_code is not None
            )
            else resume_controller.quarantined_concept_draft_for_step(step.step_id)
        )
        resume_critic_repair_code = (
            None
            if (
                preflight_trajectory_stability_code is not None
                or preflight_figure_code is not None
                or quarantined_resume_draft is not None
            )
            else _resume_critic_repair_code()
        )
        resume_summary_repair_code = (
            None
            if (
                preflight_trajectory_stability_code is not None
                or preflight_figure_code is not None
                or quarantined_resume_draft is not None
                or resume_critic_repair_code is not None
            )
            else _resume_summary_repair_code()
        )
        preflight_resumed_code = None
        if (
            preflight_trajectory_stability_code is None
            and preflight_figure_code is None
            and quarantined_resume_draft is None
            and resume_summary_repair_code is None
            and resume_critic_repair_code is None
            and (
                requested_resume_from_step_id != step.step_id
                or reuse_selected_step_code_opt_in
            )
        ):
            preflight_resumed_code = resume_controller.prior_code_for_step(step.step_id)
        if preflight_trajectory_stability_code is not None:
            code = preflight_trajectory_stability_code
            with shared_lock:
                findings.append(
                    ValidationFinding(
                        validator="coder",
                        severity="info",
                        message=(
                            "Using the deterministic calculator for the complete "
                            "planner-owned trajectory stability specification in "
                            f"step {step.step_id}."
                        ),
                        detail={"step_id": step.step_id},
                    )
                )
        elif preflight_figure_code is not None:
            code = preflight_figure_code
            with shared_lock:
                findings.append(
                    ValidationFinding(
                        validator="coder",
                        severity="info",
                        message=(
                            "Using deterministic publication-figure renderer "
                            f"for figure step {step.step_id} before requesting "
                            "new coder code."
                        ),
                        detail={"step_id": step.step_id},
                    )
                )
        elif quarantined_resume_draft is not None:
            code = _use_quarantined_draft(quarantined_resume_draft)
        elif resume_critic_repair_code is not None:
            code = resume_critic_repair_code
        elif resume_summary_repair_code is not None:
            code = resume_summary_repair_code
        elif preflight_resumed_code is not None:
            code = _use_resumed_code(preflight_resumed_code)
        # Primary estimands and cohort selection stay agent-owned.  Deterministic
        # preflight below is limited to standard auxiliary products (descriptive
        # context, robustness replay, missingness audit, figures, and overlap
        # rendering); it must never replace a planned Cox/IPTW/ordinal method or
        # choose the analysis cohort before the coder runs.
        elif (
            _preflight_absolute_risk_code := (
                _deterministic_absolute_risk_context_code(
                    "absolute_risk_context_preflight", preflight=True
                )
            )
        ) is not None:
            code = _preflight_absolute_risk_code
            with shared_lock:
                findings.append(
                    ValidationFinding(
                        validator="coder",
                        severity="info",
                        message=(
                            "Using deterministic absolute-risk context runner "
                            f"before requesting new coder code for step {step.step_id}."
                        ),
                        detail={"step_id": step.step_id},
                    )
                )
        elif (
            _preflight_robustness_code := _deterministic_robustness_sensitivity_code(
                "robustness_sensitivity_preflight", preflight=True
            )
        ) is not None:
            code = _preflight_robustness_code
            with shared_lock:
                findings.append(
                    ValidationFinding(
                        validator="coder",
                        severity="info",
                        message=(
                            "Using deterministic robustness-sensitivity runner "
                            f"before requesting new coder code for step {step.step_id}."
                        ),
                        detail={"step_id": step.step_id},
                    )
                )
        elif (
            _preflight_missingness_code := _deterministic_missingness_audit_code(
                "missingness_audit_preflight", preflight=True
            )
        ) is not None:
            # The missingness/measurement audit is a deterministic per-concept
            # count; the LLM coder reliably timed out on it (~27.6 min then fail,
            # blocking the run). The runner produces the audit table + a
            # data_quality step_summary, so the figure step then renders via the
            # parent-family fallback (data_quality -> missingness renderer).
            code = _preflight_missingness_code
            with shared_lock:
                findings.append(
                    ValidationFinding(
                        validator="coder",
                        severity="info",
                        message=(
                            "Using deterministic missingness/measurement audit runner "
                            f"before requesting new coder code for step {step.step_id}."
                        ),
                        detail={"step_id": step.step_id},
                    )
                )
        else:
            preflight_figure_code = _deterministic_publication_figure_code(
                "publication_figure_parent_outputs_preflight",
            )
            if preflight_figure_code is not None:
                code = preflight_figure_code
                with shared_lock:
                    findings.append(
                        ValidationFinding(
                            validator="coder",
                            severity="info",
                            message=(
                                f"Using deterministic publication-figure renderer "
                                f"for figure step {step.step_id} before requesting "
                                "new coder code."
                            ),
                            detail={"step_id": step.step_id},
                        )
                    )
            else:
                try:
                    emit_progress(
                        "coder",
                        f"Generating analysis script for {step.step_id}.",
                        run_id=run_id,
                        step_id=step.step_id,
                        current_step=step_current,
                        total_steps=total_steps,
                    )
                    code = coder.run(
                        context=coder_context,
                        step=step,
                        provider_budget=provider_budget,
                    )
                    _sync_provider_budget()
                except Exception as exc:
                    _sync_provider_budget()
                    resumed_code = resume_controller.prior_code_for_step(step.step_id)
                    if resumed_code is not None:
                        code = _use_resumed_code(resumed_code, error=exc)
                    else:
                        fallback_code = _deterministic_publication_figure_code(
                            "publication_figure_coder_failed"
                        )
                        if fallback_code is not None:
                            code = fallback_code
                            with shared_lock:
                                findings.append(
                                    ValidationFinding(
                                        validator="coder",
                                        severity="warning",
                                        message=(
                                            f"Coder agent failed for step {step.step_id}; "
                                            "using its explicitly matched auxiliary "
                                            "deterministic fallback."
                                        ),
                                        detail={
                                            "step_id": step.step_id,
                                            "error": str(exc)[:300],
                                        },
                                    )
                                )
                        else:
                            with shared_lock:
                                findings.append(
                                    ValidationFinding(
                                        validator="coder",
                                        severity="error",
                                        message=f"Coder agent failed for step {step.step_id}: {exc}",
                                    )
                                )
                                step_record["status"] = "coder_failed"
                                per_step_records.append(step_record)
                                _flush_partial_manifest()
                            emit_progress(
                                "coder",
                                f"Coder failed for {step.step_id}.",
                                status="error",
                                run_id=run_id,
                                step_id=step.step_id,
                                current_step=step_current,
                                total_steps=total_steps,
                            )
                            return step_record

        def _deterministic_fallback_code(reason: str) -> Optional[str]:
            nonlocal deterministic_fallback_used
            if (
                deterministic_fallback_used
                or not pipeline._enable_deterministic_code_fallback
            ):
                return None
            deterministic_fallback_used = True
            plan_result.used_mock_llm = True
            step_record["deterministic_code_fallback"] = reason
            emit_progress(
                "coder",
                f"Using deterministic fallback script for {step.step_id}.",
                run_id=run_id,
                step_id=step.step_id,
                current_step=step_current,
                total_steps=total_steps,
                fallback_reason=reason,
            )
            fallback_coder = CoderAgent(MockLLMClient(context=coder_context))
            return fallback_coder.run(context=coder_context, step=step)

        llm_concept_audit_completed_digests: set[str] = set()

        def _concept_findings_for_code(
            script_text: str,
            *,
            include_llm: bool,
        ) -> List[ValidationFinding]:
            """Run deterministic code gates and, when requested, the LLM audit.

            Deterministic semantic/mechanical checks always run before execution.
            The comparatively expensive LLM audit is reserved for an exact code
            digest that has already executed successfully and passed the early
            host-owned output contracts.  Stored quarantine errors remain part of
            the deterministic pre-execution decision and therefore can never be
            bypassed by deferring a fresh LLM call.
            """

            nonlocal quarantined_draft_active
            nonlocal quarantine_policy_superseded
            nonlocal pending_quarantined_errors

            code_findings = _deterministic_code_gate_findings(
                context=context,
                step=step,
                script_text=script_text,
                usage_auditor=usage_auditor,
                pattern_auditor=pattern_auditor,
            )
            deterministic_errors = [
                finding
                for finding in code_findings
                if finding.severity == "error"
                and finding.validator != "llm_concept_auditor"
            ]
            if pending_quarantined_errors:
                # Policy supersession is a deterministic decision about the
                # exact quarantined digest.  Make it before the optional LLM
                # audit so a retired historical error cannot trigger (or be
                # regenerated by) an unnecessary repair call first.
                supersession = _quarantined_errors_superseded_by_current_policy(
                    prior_errors=pending_quarantined_errors,
                    current_findings=code_findings,
                    context=context,
                    script_text=script_text,
                    quarantined_script_sha256=str(
                        step_record.get("quarantined_draft_sha256") or ""
                    ),
                )
                if supersession is not None:
                    reclassified_findings, provenance = supersession
                    existing_keys = {
                        (finding.validator, finding.severity, finding.message)
                        for finding in code_findings
                    }
                    code_findings.extend(
                        finding
                        for finding in reclassified_findings
                        if (finding.validator, finding.severity, finding.message)
                        not in existing_keys
                    )
                    quarantine_policy_superseded = True
                    quarantined_draft_active = False
                    pending_quarantined_errors = []
                    step_record["quarantine_policy_superseded"] = True
                    step_record["quarantine_policy_superseded_findings"] = provenance
                    emit_progress(
                        "audit",
                        (
                            "Retiring stored concept errors under the current "
                            f"deterministic validator policy for {step.step_id}."
                        ),
                        status="warning",
                        run_id=run_id,
                        step_id=step.step_id,
                        current_step=step_current,
                        total_steps=total_steps,
                    )
            try:
                if (
                    include_llm
                    and pipeline._enable_llm_concept_audit
                    and (
                        deterministic_fallback_used
                        or deterministic_standard_executor_used
                    )
                ):
                    generation_mode = (
                        "deterministic_standard"
                        if deterministic_standard_executor_used
                        else "deterministic_fallback"
                    )
                    code_findings.append(
                        ValidationFinding(
                            validator="llm_concept_auditor",
                            severity="info",
                            message=(
                                "Skipped optional LLM concept audit for trusted "
                                f"{generation_mode} code in step {step.step_id}; "
                                "deterministic audits still ran."
                            ),
                            detail={
                                "step_id": step.step_id,
                                "generation_mode": generation_mode,
                            },
                        )
                    )
                elif (
                    include_llm
                    and pipeline._enable_llm_concept_audit
                    and deterministic_errors
                ):
                    code_findings.append(
                        ValidationFinding(
                            validator="llm_concept_auditor",
                            severity="info",
                            message=(
                                "Deferred optional LLM concept audit because the "
                                "deterministic mechanical/concept preflight already "
                                f"blocked step {step.step_id}. The repaired digest "
                                "will be audited after deterministic checks pass."
                            ),
                            detail={
                                "step_id": step.step_id,
                                "deterministic_error_validators": sorted(
                                    {
                                        finding.validator
                                        for finding in deterministic_errors
                                    }
                                ),
                            },
                        )
                    )
                elif include_llm and pipeline._enable_llm_concept_audit:
                    llm_audit_client = (
                        pipeline._llm_concept_auditor_client
                        or role_resolver("analyzer")
                    )
                    if llm_audit_client is not None:
                        llm_concept_auditor = LLMConceptAuditor(llm_audit_client)
                        audit_prompt = llm_concept_auditor._prompt(
                            context=context,
                            script_text=script_text,
                            step=step,
                        )
                        audit_key = llm_concept_audit_cache.key(
                            context=context,
                            step=step,
                            script_text=script_text,
                            audit_prompt=audit_prompt,
                            environment_sha256=(concept_audit_environment_sha256),
                            auditor_identity=pipeline._llm_signature(llm_audit_client),
                            authority_bindings=resolved_input_bindings,
                            validator_implementation_sha256=(
                                llm_concept_auditor_implementation_sha256
                            ),
                        )
                        cached_findings = llm_concept_audit_cache.get(audit_key)
                        if cached_findings is not None:
                            # Cache entries preserve the original audit output,
                            # but deterministic policy reclassifiers are the
                            # current authority.  Replay them on every cache hit
                            # so an old ERROR cannot survive a validator-policy
                            # fix merely because the LLM result was reusable.
                            cached_findings = _reclassify_llm_concept_findings(
                                findings=cached_findings,
                                context=context,
                                script_text=script_text,
                            )
                            code_findings.extend(cached_findings)
                            llm_concept_audit_completed_digests.add(
                                sha256_of_bytes(script_text.encode("utf-8"))
                            )
                            step_record["llm_concept_audit_cache_hits"] = (
                                int(
                                    step_record.get("llm_concept_audit_cache_hits") or 0
                                )
                                + 1
                            )
                        else:
                            llm_findings = llm_concept_auditor.audit(
                                context=context,
                                script_text=script_text,
                                step=step,
                                provider_budget=provider_budget,
                            )
                            _sync_provider_budget()
                            llm_concept_audit_cache.put(audit_key, llm_findings)
                            code_findings.extend(llm_findings)
                            llm_concept_audit_completed_digests.add(
                                sha256_of_bytes(script_text.encode("utf-8"))
                            )
            except ProviderCallBudgetError as exc:
                _sync_provider_budget()
                receipt_error = isinstance(exc, ProviderCallBudgetReceiptError)
                code_findings.append(
                    ValidationFinding(
                        validator=(
                            "provider_call_budget_receipt"
                            if receipt_error
                            else "provider_call_budget"
                        ),
                        severity="error",
                        message=(
                            f"Step {step.step_id} could not durably record its "
                            "provider call before concept approval."
                            if receipt_error
                            else f"Step {step.step_id} exhausted its shared LLM "
                            "provider-call budget before concept approval."
                        ),
                        detail={
                            "step_id": step.step_id,
                            "category": getattr(exc, "category", None),
                            "limit": getattr(exc, "limit", provider_budget.limit),
                            "used": getattr(exc, "used", provider_budget.used),
                            "reason": str(exc),
                        },
                    )
                )
            except BaseException:
                # An operator interrupt must propagate, but a draft already
                # rejected by deterministic findings remains resumable.
                error_payloads = _quarantine_error_payloads(code_findings)
                if error_payloads:
                    try:
                        store_quarantined_concept_draft(
                            run_dir=run_dir,
                            step_id=step.step_id,
                            code=script_text,
                            findings=error_payloads,
                        )
                    except Exception:
                        pass
                raise
            if pending_quarantined_errors:
                existing_keys = {
                    (finding.validator, finding.severity, finding.message)
                    for finding in code_findings
                }
                code_findings.extend(
                    finding
                    for finding in pending_quarantined_errors
                    if (finding.validator, finding.severity, finding.message)
                    not in existing_keys
                )
            return code_findings

        def _authorized_deterministic_concept_repair(
            *,
            script_text: str,
            error_messages: Sequence[str],
            source: str,
        ) -> Tuple[str, List[str]]:
            """Return an all-or-nothing centrally authorized mechanical repair."""

            candidate_code, repair_names = deterministic_concept_audit_repair(
                script_text,
                error_messages,
            )
            if not repair_names or candidate_code == script_text:
                return script_text, []
            for repair_name in repair_names:
                if (
                    _authorize_automatic_repair(
                        (repair_name, candidate_code),
                        step=step,
                        source=source,
                        before_code=script_text,
                    )
                    is None
                ):
                    return script_text, []
            return candidate_code, list(repair_names)

        concept_repair_attempts = 0
        llm_repair_used = critic_resume_repair_used
        concept_audit_error_count = 0
        deterministic_concept_repairs = 0
        _MAX_DETERMINISTIC_CONCEPT_REPAIRS = 3
        applied_concept_repair_names: List[str] = []
        concept_approved_code_digest: Optional[str] = None
        while True:
            # A quarantined checkpoint is digest-bound authority.  Do not
            # normalize it before testing deterministic policy supersession;
            # even a semantics-preserving rewrite would break the exact SHA
            # proof and force an otherwise unnecessary LLM repair.
            if not quarantined_draft_active:
                code = reorder_forward_references(code)
            usage_findings = _concept_findings_for_code(code, include_llm=False)
            step_record["usage_findings"] = [f.model_dump() for f in usage_findings]
            concept_audit_error_count += sum(
                1
                for f in usage_findings
                if f.validator == usage_auditor.name and f.severity == "error"
            )
            step_record["concept_audit_error_count"] = concept_audit_error_count
            step_record["concept_repair_attempts"] = concept_repair_attempts
            if not any(f.severity == "error" for f in usage_findings):
                concept_approved_code_digest = sha256_of_bytes(code.encode("utf-8"))
                step_record["concept_approved_code_sha256"] = (
                    concept_approved_code_digest
                )
                if sealed_renderer_repair_id is not None:
                    sealed_renderer_authorized_code_sha256 = (
                        concept_approved_code_digest
                    )
                    step_record["sealed_renderer_authorized_code_sha256"] = (
                        sealed_renderer_authorized_code_sha256
                    )
                if (
                    resumed_quarantined_draft_used
                    and quarantined_repair_materially_changed
                    and not quarantine_superseded_by_fallback
                ):
                    quarantined_repair_succeeded = True
                    step_record["quarantined_repair_succeeded"] = True
                with shared_lock:
                    findings.extend(usage_findings)
                break

            if sealed_renderer_repair_id is not None:
                terminal_finding = ValidationFinding(
                    validator="sealed_renderer_authority",
                    severity="error",
                    message=(
                        "The authorized rendering-only adapter failed the "
                        "pre-execution deterministic concept gate; execution was "
                        "blocked without coder repair."
                    ),
                    detail={
                        "step_id": step.step_id,
                        "repair_id": sealed_renderer_repair_id,
                        "reason": "preexecution_concept_gate_failed",
                    },
                )
                terminal_findings = [terminal_finding, *usage_findings]
                step_record.update(
                    {
                        "status": "blocked_by_concept_audit",
                        "diagnostic_only": True,
                        "sealed_renderer_terminal_reason": (
                            "preexecution_concept_gate_failed"
                        ),
                        "contract_findings": [
                            finding.model_dump() for finding in terminal_findings
                        ],
                        "llm_repair_used": False,
                        "generation_mode": "fallback",
                    }
                )
                with shared_lock:
                    findings.extend(terminal_findings)
                    per_step_records.append(step_record)
                    _flush_partial_manifest()
                emit_progress(
                    "audit",
                    f"Sealed renderer blocked for {step.step_id}.",
                    status="error",
                    run_id=run_id,
                    step_id=step.step_id,
                    current_step=step_current,
                    total_steps=total_steps,
                )
                return step_record

            if deterministic_standard_executor_used:
                terminal_finding = ValidationFinding(
                    validator="trajectory_stability_executor",
                    severity="error",
                    message=(
                        "The trusted trajectory stability adapter failed the "
                        "pre-execution deterministic concept gate; execution was "
                        "blocked without coder repair."
                    ),
                    detail={
                        "step_id": step.step_id,
                        "reason": "preexecution_concept_gate_failed",
                    },
                )
                terminal_findings = [terminal_finding, *usage_findings]
                step_record.update(
                    {
                        "status": "deterministic_standard_blocked",
                        "diagnostic_only": True,
                        "standard_executor_terminal_reason": (
                            "preexecution_concept_gate_failed"
                        ),
                        "contract_findings": [
                            finding.model_dump() for finding in terminal_findings
                        ],
                        "llm_repair_used": False,
                        "generation_mode": "deterministic_standard",
                    }
                )
                with shared_lock:
                    findings.extend(terminal_findings)
                    per_step_records.append(step_record)
                    _flush_partial_manifest()
                emit_progress(
                    "audit",
                    f"Trusted standard adapter blocked for {step.step_id}.",
                    status="error",
                    run_id=run_id,
                    step_id=step.step_id,
                    current_step=step_current,
                    total_steps=total_steps,
                )
                return step_record

            # Tier A — deterministic mechanical repair. For a closed set of
            # objectively-flagged ICU anti-patterns (e.g. silent fillna(0) on
            # a lab) there is a single neutral fix, so we apply it without a
            # model round-trip and re-audit. This does NOT consume the LLM
            # repair budget, and is bounded because each repair removes its
            # own pattern (a re-audit then finds nothing left to change).
            if deterministic_concept_repairs < _MAX_DETERMINISTIC_CONCEPT_REPAIRS:
                _audit_error_msgs = [
                    value
                    for finding in usage_findings
                    if finding.severity == "error"
                    for value in (
                        finding.message,
                        str((finding.detail or {}).get("reason") or ""),
                    )
                    if value
                ]
                _det_code, _det_names = _authorized_deterministic_concept_repair(
                    script_text=code,
                    error_messages=_audit_error_msgs,
                    source="deterministic_concept_audit_repair",
                )
                if _det_names and _det_code != code:
                    _det_before_code = code
                    deterministic_concept_repairs += 1
                    applied_concept_repair_names.extend(_det_names)
                    step_record["deterministic_concept_repairs"] = (
                        deterministic_concept_repairs
                    )
                    step_record["applied_concept_repair_names"] = list(
                        applied_concept_repair_names
                    )
                    for _name in _det_names:
                        _record_repair(
                            repair_id=_name,
                            step_id=step.step_id,
                            trigger={
                                "gate": "concept_audit",
                                "audit_errors": _audit_error_msgs,
                            },
                            transformation=(
                                "deterministic_concept_audit_repair: rewrote a "
                                "mechanical ICU anti-pattern flagged as an error "
                                "by the static concept-audit gate"
                            ),
                            before_code=code,
                            after_code=_det_code,
                            selection_rule=(
                                "applied only because an error finding "
                                "objectively named the anti-pattern"
                            ),
                        )
                    emit_progress(
                        "coder",
                        f"Auto-repaired concept-audit anti-pattern "
                        f"({', '.join(_det_names)}) for {step.step_id}.",
                        run_id=run_id,
                        step_id=step.step_id,
                        current_step=step_current,
                        total_steps=total_steps,
                    )
                    code = _det_code
                    if (
                        quarantined_draft_active
                        and _python_repair_is_materially_changed(
                            _det_before_code,
                            code,
                        )
                    ):
                        # The stored errors remain useful regression constraints,
                        # but they are not findings on a new, materially repaired
                        # digest.  Re-audit that digest from scratch just as the
                        # LLM-repair path below does.
                        quarantined_draft_active = False
                        quarantined_repair_materially_changed = True
                        pending_quarantined_errors = []
                        step_record["quarantined_repair_materially_changed"] = True
                    continue

            if (
                concept_repair_attempts >= pipeline._max_code_repair_attempts
                or not _llm_repair_budget_available()
                or provider_budget.exhausted
            ):
                if not _llm_repair_budget_available():
                    step_record["step_llm_repair_budget_exhausted"] = True
                    step_record["step_llm_repair_budget"] = (
                        pipeline._max_step_llm_repair_attempts
                    )
                _sync_provider_budget()
                fallback_code = _deterministic_fallback_code("concept_audit")
                if fallback_code is not None:
                    fallback_checkpoint_error: Optional[Exception] = None
                    if resumed_quarantined_draft_used:
                        try:
                            checkpoint = store_quarantined_concept_draft(
                                run_dir=run_dir,
                                step_id=step.step_id,
                                code=code,
                                findings=_quarantine_error_payloads(usage_findings),
                            )
                            step_record["quarantined_draft_sha256"] = checkpoint.sha256
                            step_record["quarantined_draft_relative_path"] = (
                                checkpoint.relative_path
                            )
                            step_record["quarantine_checkpoint_is_latest_candidate"] = (
                                True
                            )
                        except Exception as checkpoint_exc:
                            fallback_checkpoint_error = checkpoint_exc
                    # Surface the pattern/concept findings that
                    # forced the fallback; otherwise the manifest
                    # silently drops the original ICU rule
                    # violations that the LLM emitted. We dedupe by
                    # message so repeated retries don't spam.
                    with shared_lock:
                        if fallback_checkpoint_error is not None:
                            findings.append(
                                ValidationFinding(
                                    validator="resume",
                                    severity="warning",
                                    message=(
                                        "Could not update the concept-draft "
                                        "checkpoint before deterministic fallback "
                                        f"for step {step.step_id}: "
                                        f"{fallback_checkpoint_error}"
                                    ),
                                    detail={"step_id": step.step_id},
                                )
                            )
                        seen_msgs = {f.message for f in findings}
                        for f in usage_findings:
                            if f.message in seen_msgs:
                                continue
                            # Demote ``error`` severity to
                            # ``warning`` because the run is
                            # continuing on the deterministic
                            # fallback; reviewer still sees the
                            # original violation in the manifest.
                            if f.severity == "error":
                                f = f.model_copy(
                                    update={
                                        "severity": "warning",
                                        "message": (
                                            "[surfaced after fallback] " + f.message
                                        ),
                                    }
                                )
                            findings.append(f)
                    if resumed_quarantined_draft_used:
                        quarantined_draft_active = False
                        pending_quarantined_errors = []
                        quarantined_repair_succeeded = False
                        quarantine_superseded_by_fallback = True
                        step_record["quarantined_repair_succeeded"] = False
                        step_record["quarantine_superseded_by_fallback"] = True
                    code = fallback_code
                    continue
                step_record["status"] = "blocked_by_concept_audit"
                checkpoint_error: Optional[Exception] = None
                if not quarantine_superseded_by_fallback:
                    try:
                        checkpoint = store_quarantined_concept_draft(
                            run_dir=run_dir,
                            step_id=step.step_id,
                            code=code,
                            findings=_quarantine_error_payloads(usage_findings),
                        )
                        step_record["quarantined_draft_sha256"] = checkpoint.sha256
                        step_record["quarantined_draft_relative_path"] = (
                            checkpoint.relative_path
                        )
                        step_record["quarantined_requires_repair"] = True
                        step_record["quarantine_checkpoint_is_latest_candidate"] = True
                    except Exception as checkpoint_exc:
                        checkpoint_error = checkpoint_exc
                # Tier C — when auto-repair (deterministic + LLM) could not
                # clear the violation, do NOT just stop with a status code.
                # Emit an actionable repair ticket so a human can either add a
                # constraint and re-run, or knowingly accept the withheld
                # (diagnostic_only) result. We name candidate remedies without
                # mandating one — the analytical choice stays with the user.
                _block_errors = [
                    {"validator": f.validator, "message": f.message}
                    for f in usage_findings
                    if f.severity == "error"
                ]
                _offending_lines = [
                    ln.strip()
                    for ln in code.splitlines()
                    if any(
                        tok in ln
                        for tok in ("fillna(0)", "fillna(0.0)", ".mean()", "dropna(")
                    )
                ][:12]
                _remedies = [
                    "Add the violated ICU rule as an explicit coder/planner "
                    "constraint and re-run this question (e.g. 'do not impute a "
                    "lab with 0; handle missingness with complete-case or a "
                    "declared imputation + missingness indicator').",
                    "Use a stronger model for this question — the block was "
                    "triggered by generated code, not by the cohort or the "
                    "question itself.",
                    "Accept the withheld result: diagnostic_only is a valid "
                    "outcome. The fail-closed gate declined to report an "
                    "analysis it judged unsafe; nothing wrong was published.",
                ]
                step_record["concept_audit_block"] = {
                    "step_id": step.step_id,
                    "errors": _block_errors,
                    "deterministic_repairs_applied": list(applied_concept_repair_names),
                    "llm_repair_attempts": concept_repair_attempts,
                    "offending_code_lines": _offending_lines,
                    "candidate_remedies": _remedies,
                }
                try:
                    _ticket = [
                        f"# Concept-audit block — step `{step.step_id}`",
                        "",
                        "The static ICU concept-audit gate blocked this step "
                        "before execution and auto-repair could not clear it, "
                        "so the run withheld this analysis (`diagnostic_only`). "
                        "This is the fail-closed safety system working — but "
                        "here is how to move it forward.",
                        "",
                        "## What was flagged (objective errors)",
                        *[
                            f"- **{e['validator']}**: {e['message']}"
                            for e in _block_errors
                        ],
                        "",
                        "## Repair already attempted",
                        f"- deterministic: "
                        f"{applied_concept_repair_names or 'none matched'}",
                        f"- LLM coder repair attempts: {concept_repair_attempts}",
                        "",
                        "## Offending code lines",
                        "```python",
                        *(_offending_lines or ["(no obvious anti-pattern line)"]),
                        "```",
                        "",
                        "## How to resolve (pick one — your analytical choice)",
                        *[f"{i + 1}. {r}" for i, r in enumerate(_remedies)],
                        "",
                    ]
                    (run_dir / f"concept_audit_block_{step.step_id}.md").write_text(
                        "\n".join(_ticket), encoding="utf-8"
                    )
                except Exception:  # ticket is best-effort, never fatal
                    pass
                with shared_lock:
                    findings.extend(usage_findings)
                    if checkpoint_error is not None:
                        findings.append(
                            ValidationFinding(
                                validator="resume",
                                severity="warning",
                                message=(
                                    "Could not update the blocked concept-draft "
                                    f"checkpoint for step {step.step_id}: "
                                    f"{checkpoint_error}"
                                ),
                                detail={"step_id": step.step_id},
                            )
                        )
                    per_step_records.append(step_record)
                    _flush_partial_manifest()
                emit_progress(
                    "audit",
                    f"Concept audit blocked {step.step_id}; repair ticket written.",
                    status="error",
                    run_id=run_id,
                    step_id=step.step_id,
                    current_step=step_current,
                    total_steps=total_steps,
                )
                return step_record

            concept_repair_attempts += 1
            if not _consume_llm_repair_budget("concept"):
                raise AssertionError("LLM repair budget changed without mutation")
            step_record["concept_repair_attempts"] = concept_repair_attempts
            emit_progress(
                "coder",
                f"Repairing concept-audit violation for {step.step_id}.",
                run_id=run_id,
                step_id=step.step_id,
                current_step=step_current,
                total_steps=total_steps,
                repair_attempts=concept_repair_attempts,
            )
            blocking_usage_findings = _blocking_validator_findings(usage_findings)
            audit_log = "\n".join(
                (
                    f"{f.severity.upper()}: {f.message}"
                    + (
                        "\nDETAIL: "
                        + json.dumps(f.detail, ensure_ascii=False, sort_keys=True)
                        if f.detail
                        else ""
                    )
                )
                for f in blocking_usage_findings
            )
            structured_repair_ticket = typed_repair_ticket(blocking_usage_findings)
            _remember_concept_constraints(blocking_usage_findings)
            try:
                repaired_code = coder.repair(
                    context=coder_context,
                    step=step,
                    code=code,
                    run_log=(
                        "Static concept audit blocked this script before "
                        "execution. Fix all ICU-rule violations.\n\n"
                        "TYPED REPAIR TICKET (authoritative routing):\n"
                        + json.dumps(
                            structured_repair_ticket,
                            indent=2,
                            ensure_ascii=False,
                            default=str,
                        )
                        + "\n\nHUMAN-READABLE FINDINGS:\n"
                        + audit_log
                    ),
                    attempt=concept_repair_attempts,
                    provider_budget=provider_budget,
                    provider_category="concept_repair",
                )
                _sync_provider_budget()
                if (
                    quarantined_draft_active
                    and not _python_repair_is_materially_changed(code, repaired_code)
                ):
                    no_op_finding = ValidationFinding(
                        validator="resume",
                        severity="error",
                        message=(
                            "Quarantined concept-draft repair returned no material "
                            f"Python change for step {step.step_id}; the pending "
                            "concept errors remain binding."
                        ),
                        detail={
                            "step_id": step.step_id,
                            "quarantined_draft_sha256": step_record.get(
                                "quarantined_draft_sha256"
                            ),
                            "repair_attempt": concept_repair_attempts,
                            "semantic_noop": True,
                        },
                    )
                    if not any(
                        finding.message == no_op_finding.message
                        for finding in pending_quarantined_errors
                    ):
                        pending_quarantined_errors.append(no_op_finding)
                    step_record["quarantined_repair_noop_count"] = (
                        int(step_record.get("quarantined_repair_noop_count") or 0) + 1
                    )
                    step_record["quarantined_repair_succeeded"] = False
                    continue
                code = repaired_code
                llm_repair_used = True
                if quarantined_draft_active:
                    quarantined_draft_active = False
                    quarantined_repair_materially_changed = True
                    pending_quarantined_errors = []
                    step_record["quarantined_repair_materially_changed"] = True
            except BaseException as exc:
                _sync_provider_budget()
                checkpoint_error: Optional[Exception] = None
                try:
                    checkpoint = store_quarantined_concept_draft(
                        run_dir=run_dir,
                        step_id=step.step_id,
                        code=code,
                        findings=_quarantine_error_payloads(usage_findings),
                    )
                    step_record["quarantined_draft_sha256"] = checkpoint.sha256
                    step_record["quarantined_draft_relative_path"] = (
                        checkpoint.relative_path
                    )
                    step_record["quarantined_requires_repair"] = True
                except Exception as checkpoint_exc:
                    checkpoint_error = checkpoint_exc
                if not isinstance(exc, Exception):
                    raise
                fallback_code = _deterministic_fallback_code("concept_repair_failed")
                if fallback_code is not None:
                    quarantined_draft_active = False
                    pending_quarantined_errors = []
                    quarantined_repair_succeeded = False
                    if resumed_quarantined_draft_used:
                        quarantine_superseded_by_fallback = True
                        step_record["quarantined_repair_succeeded"] = False
                        step_record["quarantine_superseded_by_fallback"] = True
                    code = fallback_code
                    continue
                with shared_lock:
                    findings.extend(usage_findings)
                    if checkpoint_error is not None:
                        findings.append(
                            ValidationFinding(
                                validator="resume",
                                severity="warning",
                                message=(
                                    "Could not preserve the rejected concept-audit "
                                    f"draft for step {step.step_id}: {checkpoint_error}"
                                ),
                                detail={"step_id": step.step_id},
                            )
                        )
                    findings.append(
                        ValidationFinding(
                            validator="coder",
                            severity="error",
                            message=(
                                f"Coder repair failed after concept audit for "
                                f"step {step.step_id}: {exc}"
                            ),
                        )
                    )
                    step_record["status"] = "repair_failed"
                    per_step_records.append(step_record)
                    _flush_partial_manifest()
                emit_progress(
                    "coder",
                    f"Concept-audit repair failed for {step.step_id}.",
                    status="error",
                    run_id=run_id,
                    step_id=step.step_id,
                    current_step=step_current,
                    total_steps=total_steps,
                )
                return step_record

        if quarantined_draft_active and not quarantined_repair_succeeded:
            hard_gate_finding = ValidationFinding(
                validator="resume",
                severity="error",
                message=(
                    "Quarantined concept-audit draft cannot execute before a "
                    f"successful coder repair for step {step.step_id}."
                ),
                detail={
                    "step_id": step.step_id,
                    "quarantined_draft_sha256": step_record.get(
                        "quarantined_draft_sha256"
                    ),
                },
            )
            step_record["status"] = "blocked_quarantined_draft"
            with shared_lock:
                findings.append(hard_gate_finding)
                per_step_records.append(step_record)
                _flush_partial_manifest()
            emit_progress(
                "audit",
                f"Blocked unrepaired quarantined draft for {step.step_id}.",
                status="error",
                run_id=run_id,
                step_id=step.step_id,
                current_step=step_current,
                total_steps=total_steps,
            )
            return step_record

        if quarantined_repair_succeeded or quarantine_policy_superseded:
            try:
                clear_quarantined_concept_draft(
                    run_dir=run_dir,
                    step_id=step.step_id,
                )
                step_record["quarantined_requires_repair"] = False
                step_record["quarantine_retired"] = True
                if quarantine_policy_superseded:
                    step_record["quarantine_retired_by"] = (
                        "deterministic_validator_policy_supersession"
                    )
            except ValueError as exc:
                cleanup_finding = ValidationFinding(
                    validator="resume",
                    severity="error",
                    message=(
                        "Concept-approved code could not retire its stale "
                        f"quarantine safely for step {step.step_id}: {exc}"
                    ),
                    detail={"step_id": step.step_id},
                )
                step_record["status"] = "blocked_quarantine_cleanup"
                with shared_lock:
                    findings.append(cleanup_finding)
                    per_step_records.append(step_record)
                    _flush_partial_manifest()
                return step_record

        repair_attempts = 0
        contract_repair_attempts = 0
        visual_repair_attempts = 0
        # Contract, visual-layout, and runtime failures have independent repair
        # budgets. ``repair_attempts`` remains the total mutation count used for
        # provenance and generation-mode labels.
        runtime_repair_attempts = 0
        runner_repair_name = preexecution_runner_repair_name
        is_trajectory_stability_standard = bool(
            step.trajectory_stability_spec is not None
            and step_record.get("deterministic_standard_analysis")
            == "trajectory_cluster_stability"
        )
        standard_executor_terminal_block = False
        standard_executor_terminal_reason: Optional[str] = None
        standard_executor_terminal_summary: Dict[str, Any] = {}
        standard_executor_terminal_findings: List[ValidationFinding] = []
        deterministic_contract_approved_code_digest: Optional[str] = None
        final_concept_gate_approved_code_digest: Optional[str] = None
        while True:
            code = reorder_forward_references(code)
            candidate_code_digest = sha256_of_bytes(code.encode("utf-8"))
            if (
                sealed_renderer_authorized_code_sha256 is not None
                and candidate_code_digest != sealed_renderer_authorized_code_sha256
            ):
                authority_finding = ValidationFinding(
                    validator="sealed_renderer_authority",
                    severity="error",
                    message=(
                        "The active rendering-only adapter no longer matches its "
                        "authorized code digest; execution was blocked without "
                        "running or repairing the mutated code."
                    ),
                    detail={
                        "step_id": step.step_id,
                        "repair_id": sealed_renderer_repair_id,
                        "authorized_code_sha256": (
                            sealed_renderer_authorized_code_sha256
                        ),
                        "candidate_code_sha256": candidate_code_digest,
                    },
                )
                step_record.update(
                    {
                        "status": "execution_failed",
                        "diagnostic_only": True,
                        "sealed_renderer_terminal_reason": "code_digest_changed",
                        "llm_repair_used": False,
                        "generation_mode": "fallback",
                    }
                )
                with shared_lock:
                    findings.append(authority_finding)
                    per_step_records.append(step_record)
                    _flush_partial_manifest()
                return step_record
            final_llm_audit_due = bool(
                candidate_code_digest == deterministic_contract_approved_code_digest
                and candidate_code_digest != final_concept_gate_approved_code_digest
            )
            if (
                candidate_code_digest != concept_approved_code_digest
                or final_llm_audit_due
            ):
                # Every mutation still returns through deterministic semantic and
                # mechanical gates before execution.  The LLM concept auditor is
                # invoked only for the exact digest whose local run and early
                # deterministic contracts already passed, preventing runtime- or
                # contract-broken drafts from consuming repeated audit calls.
                usage_findings = _concept_findings_for_code(
                    code,
                    include_llm=final_llm_audit_due,
                )
                step_record["usage_findings"] = [
                    finding.model_dump() for finding in usage_findings
                ]
                post_mutation_errors = [
                    finding for finding in usage_findings if finding.severity == "error"
                ]
                if post_mutation_errors:
                    if final_llm_audit_due:
                        # These outputs came from a digest rejected by the final
                        # semantic audit.  They are never eligible for later
                        # sealing/current authority, and a repaired digest must
                        # execute afresh before it can regain contract approval.
                        deterministic_contract_approved_code_digest = None
                        _clear_output_dir(run_dir / "steps" / step.step_id / "outputs")
                    post_mutation_messages = [
                        value
                        for finding in post_mutation_errors
                        for value in (
                            finding.message,
                            str((finding.detail or {}).get("reason") or ""),
                        )
                        if value
                    ]
                    if (
                        deterministic_concept_repairs
                        < _MAX_DETERMINISTIC_CONCEPT_REPAIRS
                    ):
                        deterministic_code, deterministic_names = (
                            _authorized_deterministic_concept_repair(
                                script_text=code,
                                error_messages=post_mutation_messages,
                                source=("post_mutation_deterministic_concept_repair"),
                            )
                        )
                        if deterministic_names and deterministic_code != code:
                            before_code = code
                            code = deterministic_code
                            deterministic_concept_repairs += 1
                            applied_concept_repair_names.extend(deterministic_names)
                            step_record["deterministic_concept_repairs"] = (
                                deterministic_concept_repairs
                            )
                            step_record["applied_concept_repair_names"] = list(
                                applied_concept_repair_names
                            )
                            for repair_name in deterministic_names:
                                _record_repair(
                                    repair_id=repair_name,
                                    step_id=step.step_id,
                                    trigger={
                                        "gate": "post_mutation_concept_audit",
                                        "audit_errors": post_mutation_messages,
                                    },
                                    transformation=(
                                        "deterministic concept repair after a "
                                        "contract/runtime mutation"
                                    ),
                                    before_code=before_code,
                                    after_code=code,
                                    selection_rule=(
                                        "applied only because a typed mechanical "
                                        "error named the anti-pattern"
                                    ),
                                )
                            _clear_output_dir(
                                run_dir / "steps" / step.step_id / "outputs"
                            )
                            continue

                    if _llm_repair_budget_available():
                        concept_repair_attempts += 1
                        if not _consume_llm_repair_budget("post_mutation_concept"):
                            raise AssertionError(
                                "LLM repair budget changed without mutation"
                            )
                        step_record["concept_repair_attempts"] = concept_repair_attempts
                        emit_progress(
                            "coder",
                            (
                                "Repairing post-mutation concept violation for "
                                f"{step.step_id}."
                            ),
                            run_id=run_id,
                            step_id=step.step_id,
                            current_step=step_current,
                            total_steps=total_steps,
                            repair_attempts=step_llm_repair_attempts,
                        )
                        _remember_concept_constraints(post_mutation_errors)
                        post_mutation_ticket = typed_repair_ticket(post_mutation_errors)
                        post_mutation_log = "\n".join(
                            (
                                f"{finding.severity.upper()}: {finding.message}"
                                + (
                                    "\nDETAIL: "
                                    + json.dumps(
                                        finding.detail,
                                        ensure_ascii=False,
                                        sort_keys=True,
                                    )
                                    if finding.detail
                                    else ""
                                )
                            )
                            for finding in post_mutation_errors
                        )
                        try:
                            code = coder.repair(
                                context=coder_context,
                                step=step,
                                code=code,
                                run_log=(
                                    "A contract or runtime repair produced a new "
                                    "code digest that failed pre-execution audit. "
                                    "Fix every typed error with the smallest change; "
                                    "preserve the earlier contract repair and all "
                                    "Planner-owned science.\n\n"
                                    "TYPED REPAIR TICKET (authoritative routing):\n"
                                    + json.dumps(
                                        post_mutation_ticket,
                                        indent=2,
                                        ensure_ascii=False,
                                        default=str,
                                    )
                                    + "\n\nFINDINGS:\n"
                                    + post_mutation_log
                                    + _monotonic_concept_constraint_log()
                                ),
                                attempt=concept_repair_attempts,
                                provider_budget=provider_budget,
                                provider_category="post_mutation_concept_repair",
                            )
                            _sync_provider_budget()
                            llm_repair_used = True
                            _clear_output_dir(
                                run_dir / "steps" / step.step_id / "outputs"
                            )
                            continue
                        except Exception as exc:
                            _sync_provider_budget()
                            checkpoint_error: Optional[Exception] = None
                            try:
                                checkpoint = store_quarantined_concept_draft(
                                    run_dir=run_dir,
                                    step_id=step.step_id,
                                    code=code,
                                    findings=_quarantine_error_payloads(
                                        post_mutation_errors
                                    ),
                                )
                                step_record["quarantined_draft_sha256"] = (
                                    checkpoint.sha256
                                )
                                step_record["quarantined_draft_relative_path"] = (
                                    checkpoint.relative_path
                                )
                                step_record["quarantined_requires_repair"] = True
                            except Exception as checkpoint_exc:
                                checkpoint_error = checkpoint_exc
                            fallback_code = _deterministic_fallback_code(
                                "concept_repair_failed"
                            )
                            if fallback_code is not None:
                                code = fallback_code
                                _clear_output_dir(
                                    run_dir / "steps" / step.step_id / "outputs"
                                )
                                continue
                            with shared_lock:
                                findings.extend(usage_findings)
                                if checkpoint_error is not None:
                                    findings.append(
                                        ValidationFinding(
                                            validator="resume",
                                            severity="warning",
                                            message=(
                                                "Could not preserve the rejected final "
                                                "concept-audit draft for step "
                                                f"{step.step_id}: {checkpoint_error}"
                                            ),
                                            detail={"step_id": step.step_id},
                                        )
                                    )
                                findings.append(
                                    ValidationFinding(
                                        validator="coder",
                                        severity="error",
                                        message=(
                                            "Coder repair failed after post-mutation "
                                            "concept audit for step "
                                            f"{step.step_id}: {exc}"
                                        ),
                                        detail={"step_id": step.step_id},
                                    )
                                )
                                step_record["status"] = "repair_failed"
                                per_step_records.append(step_record)
                                _flush_partial_manifest()
                            emit_progress(
                                "coder",
                                f"Concept-audit repair failed for {step.step_id}.",
                                status="error",
                                run_id=run_id,
                                step_id=step.step_id,
                                current_step=step_current,
                                total_steps=total_steps,
                            )
                            return step_record

                    if not _llm_repair_budget_available():
                        step_record["step_llm_repair_budget_exhausted"] = True
                        step_record["step_llm_repair_budget"] = (
                            pipeline._max_step_llm_repair_attempts
                        )
                    checkpoint_error: Optional[Exception] = None
                    try:
                        checkpoint = store_quarantined_concept_draft(
                            run_dir=run_dir,
                            step_id=step.step_id,
                            code=code,
                            findings=_quarantine_error_payloads(post_mutation_errors),
                        )
                        step_record["quarantined_draft_sha256"] = checkpoint.sha256
                        step_record["quarantined_draft_relative_path"] = (
                            checkpoint.relative_path
                        )
                        step_record["quarantined_requires_repair"] = True
                    except Exception as checkpoint_exc:
                        checkpoint_error = checkpoint_exc
                    step_record["status"] = "blocked_by_concept_audit"
                    step_record["post_repair_concept_audit_block"] = {
                        "code_sha256": candidate_code_digest,
                        "errors": [
                            finding.model_dump(mode="json")
                            for finding in post_mutation_errors
                        ],
                    }
                    with shared_lock:
                        findings.extend(usage_findings)
                        if checkpoint_error is not None:
                            findings.append(
                                ValidationFinding(
                                    validator="resume",
                                    severity="warning",
                                    message=(
                                        "Could not preserve post-repair code rejected "
                                        f"by concept audit for step {step.step_id}: "
                                        f"{checkpoint_error}"
                                    ),
                                    detail={"step_id": step.step_id},
                                )
                            )
                        per_step_records.append(step_record)
                        _flush_partial_manifest()
                    emit_progress(
                        "audit",
                        f"Concept audit blocked mutated code for {step.step_id}.",
                        status="error",
                        run_id=run_id,
                        step_id=step.step_id,
                        current_step=step_current,
                        total_steps=total_steps,
                    )
                    return step_record
                with shared_lock:
                    findings.extend(usage_findings)
                concept_approved_code_digest = candidate_code_digest
                step_record["concept_approved_code_sha256"] = (
                    concept_approved_code_digest
                )
                step_record["deterministic_preflight_approved_code_sha256"] = (
                    concept_approved_code_digest
                )
                if final_llm_audit_due:
                    final_concept_gate_approved_code_digest = candidate_code_digest
                    step_record["final_concept_gate_approved_code_sha256"] = (
                        final_concept_gate_approved_code_digest
                    )
                    if candidate_code_digest in llm_concept_audit_completed_digests:
                        step_record["llm_concept_audit_status"] = "completed"
                        step_record["llm_concept_approved_code_sha256"] = (
                            candidate_code_digest
                        )
                    elif not pipeline._enable_llm_concept_audit:
                        step_record["llm_concept_audit_status"] = "disabled"
                    elif (
                        deterministic_fallback_used
                        or deterministic_standard_executor_used
                    ):
                        step_record["llm_concept_audit_status"] = (
                            "skipped_trusted_deterministic_code"
                        )
                    else:
                        step_record["llm_concept_audit_status"] = (
                            "skipped_no_auditor_client"
                        )
                    # Reuse the already validated outputs.  No second execution
                    # of unchanged code is needed after the digest-bound audit.
                    break

            concept_repair_used = bool(
                concept_repair_attempts or deterministic_concept_repairs
            )
            current_generation_mode = _script_generation_mode(
                repair_attempts=repair_attempts,
                fallback_used=deterministic_fallback_used,
                standard_executor_used=deterministic_standard_executor_used,
                runner_repair_name=runner_repair_name,
                resumed_code_reuse=resumed_code_reuse_used,
                concept_repair_used=concept_repair_used,
                llm_repair_used=llm_repair_used,
            )
            run_label = {
                "llm": "generated script",
                "resumed_code_reuse": "resumed script",
                "fallback": "fallback script",
                "deterministic_standard": "standard executor script",
            }.get(current_generation_mode, "repaired script")
            emit_progress(
                "runner",
                f"Running {run_label} for {step.step_id}.",
                run_id=run_id,
                step_id=step.step_id,
                current_step=step_current,
                total_steps=total_steps,
                repair_attempts=repair_attempts,
            )
            execution_runner = runner
            execution_timeout_seconds = pipeline._timeout_seconds
            if deterministic_standard_executor_used:
                # A registered standard executes the exact typed workload the
                # planner froze. Give it a distinct bounded runner rather than
                # widening the shared generated-code runner's timeout. This is
                # concurrency-safe and leaves every ordinary coder attempt on
                # the configured short budget.
                execution_timeout_seconds = pipeline._standard_executor_timeout_seconds
                execution_runner = pipeline._build_runner(
                    run_dir=run_dir,
                    cohort_path=cohort_path,
                    target_outcome=context.target_outcome,
                    universe_path=universe_path,
                    timeout_seconds=execution_timeout_seconds,
                )
            # DockerRunner must first prove any previous timed-out container
            # is quiescent before its bind-mounted output directory is reused;
            # it therefore owns cleanup inside ``run``. Other backends retain
            # the pipeline's established pre-execution clearing behaviour.
            if not bool(getattr(execution_runner, "manages_output_cleanup", False)):
                _clear_output_dir(run_dir / "steps" / step.step_id / "outputs")
            step_record["execution_timeout_seconds"] = execution_timeout_seconds
            run_result = execution_runner.run(
                step_id=step.step_id,
                code=code,
                resolved_inputs_path=resolved_inputs_path,
            )
            step_record["outputs_safe_to_collect"] = bool(
                run_result.outputs_safe_to_collect
            )
            if not run_result.outputs_safe_to_collect:
                # The backend could not prove that a process/container with a
                # writable output mount was stopped.  Those outputs remain
                # mutable and are therefore ineligible for inspection,
                # hashing, repair, cleanup, or evidence registration. Docker
                # keeps host-owned script/log control copies, but this step is
                # still terminal until a later explicit retry resolves the
                # teardown sentinel first.
                unsafe_reason = "runner_output_teardown_unconfirmed"
                step_record.update(
                    {
                        "status": (
                            "deterministic_standard_blocked"
                            if is_trajectory_stability_standard
                            else "execution_failed"
                        ),
                        "diagnostic_only": True,
                        "runner_output_safety_reason": unsafe_reason,
                    }
                )
                if is_trajectory_stability_standard:
                    step_record["standard_executor_terminal_reason"] = (
                        "executor_runtime_failure"
                    )
                elif sealed_renderer_authorized_code_sha256 is not None:
                    step_record.update(
                        {
                            "sealed_renderer_runtime_repair_suppressed": True,
                            "sealed_renderer_terminal_reason": unsafe_reason,
                            "llm_repair_used": False,
                            "generation_mode": "fallback",
                        }
                    )
                unsafe_finding = ValidationFinding(
                    validator="runner_output_safety",
                    severity="error",
                    message=(
                        f"Step {step.step_id} was stopped because the execution "
                        "backend could not confirm teardown of its writable "
                        "mount; no files from that mount were inspected or "
                        "registered."
                    ),
                    detail={
                        "step_id": step.step_id,
                        "reason": unsafe_reason,
                        "timed_out": bool(run_result.timed_out),
                        "returncode": int(run_result.returncode),
                    },
                )
                with shared_lock:
                    findings.append(unsafe_finding)
                    per_step_records.append(step_record)
                    _flush_partial_manifest()
                emit_progress(
                    "runner",
                    f"Execution mount teardown was not confirmed for {step.step_id}.",
                    status="error",
                    run_id=run_id,
                    step_id=step.step_id,
                    current_step=step_current,
                    total_steps=total_steps,
                )
                return step_record
            executed_code_digest = sha256_of_file(run_result.script_path)
            step_record["executed_code_sha256"] = executed_code_digest
            if sealed_renderer_authorized_code_sha256 is not None:
                step_record["sealed_renderer_executed_code_matches_authority"] = (
                    executed_code_digest == sealed_renderer_authorized_code_sha256
                )
            if (
                concept_approved_code_digest is None
                or executed_code_digest != concept_approved_code_digest
            ):
                integrity_finding = ValidationFinding(
                    validator="post_repair_concept_gate",
                    severity="error",
                    message=(
                        "The executed analysis script did not match the exact "
                        f"concept-approved code digest for step {step.step_id}; "
                        "outputs were rejected before evidence registration."
                    ),
                    detail={
                        "step_id": step.step_id,
                        "concept_approved_code_sha256": concept_approved_code_digest,
                        "executed_code_sha256": executed_code_digest,
                        "script_path": str(run_result.script_path),
                    },
                )
                _clear_output_dir(run_result.out_dir)
                step_record["status"] = "blocked_script_integrity"
                step_record["script_integrity_findings"] = [
                    integrity_finding.model_dump()
                ]
                if sealed_renderer_authorized_code_sha256 is not None:
                    step_record.update(
                        {
                            "sealed_renderer_terminal_reason": (
                                "executed_code_digest_mismatch"
                            ),
                            "llm_repair_used": False,
                            "generation_mode": "fallback",
                        }
                    )
                with shared_lock:
                    findings.append(integrity_finding)
                    per_step_records.append(step_record)
                    _flush_partial_manifest()
                emit_progress(
                    "audit",
                    f"Rejected script-integrity mismatch for {step.step_id}.",
                    status="error",
                    run_id=run_id,
                    step_id=step.step_id,
                    current_step=step_current,
                    total_steps=total_steps,
                )
                return step_record
            step_record["returncode"] = run_result.returncode
            step_record["timed_out"] = run_result.timed_out
            step_record["requested_network_policy"] = (
                run_result.requested_network_policy
            )
            step_record["effective_isolation"] = run_result.effective_isolation
            step_record["isolation_degraded"] = run_result.isolation_degraded
            if run_result.isolation_degradation_reason:
                step_record["isolation_degradation_reason"] = (
                    run_result.isolation_degradation_reason
                )
            step_record["code_repair_attempts"] = repair_attempts

            if current_generation_mode == "llm":
                script_description = (
                    f"Generated analysis script for step {step.step_id}."
                )
            elif current_generation_mode == "resumed_code_reuse":
                script_description = (
                    f"Reused prior agent-generated analysis script for step "
                    f"{step.step_id}."
                )
            elif current_generation_mode == "fallback":
                script_description = (
                    f"Deterministic fallback analysis script for step {step.step_id}."
                )
            elif current_generation_mode == "deterministic_standard":
                script_description = (
                    "Planner-selected deterministic standard executor adapter for "
                    f"step {step.step_id}."
                )
            else:
                total_repair_attempts = repair_attempts + concept_repair_attempts
                script_description = (
                    f"Repaired analysis script for step {step.step_id} "
                    f"(attempt {total_repair_attempts})."
                )
            script_digest = sha256_of_file(run_result.script_path)
            script_authority = "\0".join(
                (step.step_id, script_digest, current_generation_mode)
            )
            script_evidence_id = (
                "code_analysis_"
                + hashlib.sha256(script_authority.encode("utf-8")).hexdigest()[:16]
            )
            script_record = evidence.register_file(
                kind="code",
                description=script_description,
                source_path=run_result.script_path,
                produced_by_step=step.step_id,
                inputs=resolved_input_evidence_ids or None,
                evidence_id=script_evidence_id,
                producer=(
                    "standard_executor"
                    if current_generation_mode == "deterministic_standard"
                    else "coder"
                ),
                generation_mode=current_generation_mode,
                prompt_pack_version=prompt_version,
                metadata={
                    "repair_attempts": repair_attempts,
                    "concept_repair_attempts": concept_repair_attempts,
                    "deterministic_concept_repairs": deterministic_concept_repairs,
                    "llm_repair_used": llm_repair_used,
                    "fallback_reason": step_record.get("deterministic_code_fallback"),
                    "runner_repair": runner_repair_name,
                    "resumed_code_evidence_id": step_record.get(
                        "resumed_code_evidence_id"
                    ),
                    "resumed_code_relative_path": step_record.get(
                        "resumed_code_relative_path"
                    ),
                    "resumed_from_generation_mode": step_record.get(
                        "resumed_from_generation_mode"
                    ),
                    "resumed_code_evidence_generation_mode": step_record.get(
                        "resumed_code_evidence_generation_mode"
                    ),
                    "resumed_quarantined_draft": resumed_quarantined_draft_used,
                    "quarantined_draft_sha256": step_record.get(
                        "quarantined_draft_sha256"
                    ),
                    "quarantined_repair_succeeded": quarantined_repair_succeeded,
                    "quarantine_policy_superseded": quarantine_policy_superseded,
                    "quarantine_policy_superseded_findings": step_record.get(
                        "quarantine_policy_superseded_findings"
                    ),
                    "llm_signature": llm_signature,
                },
            )
            step_record["script_evidence_id"] = script_record.evidence_id
            log_path = run_result.runner_log_path or (run_result.cwd / "run.log")
            if log_path.exists():
                evidence.register_file(
                    kind="log",
                    description=f"stdout/stderr log for step {step.step_id}.",
                    source_path=log_path,
                    produced_by_step=step.step_id,
                    script_evidence_id=script_record.evidence_id,
                    producer="runner",
                    generation_mode=current_generation_mode,
                    metadata={
                        "repair_attempts": repair_attempts,
                        "concept_repair_attempts": concept_repair_attempts,
                        "deterministic_concept_repairs": (
                            deterministic_concept_repairs
                        ),
                        "llm_repair_used": llm_repair_used,
                        "fallback_reason": step_record.get(
                            "deterministic_code_fallback"
                        ),
                        "runner_repair": runner_repair_name,
                        "resumed_from_generation_mode": step_record.get(
                            "resumed_from_generation_mode"
                        ),
                    },
                )

            if run_result.succeeded:
                # Step-summary salvage reshapes the source from which numbers are
                # registered, so each salvage is recorded in the repair ledger
                # (ENG-REPAIR1 P1.5). The salvage decision lives in
                # salvage_step_summary() so it is unit-testable end-to-end; here
                # we only record what it did.
                salvage_outcome = salvage_step_summary(run_result, step=step)
                if salvage_outcome is not None:
                    if salvage_outcome.reset_artefacts:
                        run_result.artefacts = sorted(
                            p for p in run_result.out_dir.iterdir() if p.is_file()
                        )
                    _record_repair(
                        repair_id=salvage_outcome.repair_id,
                        step_id=step.step_id,
                        trigger={
                            "source": "summary_salvage",
                            "reason": salvage_outcome.trigger_reason,
                        },
                        transformation=salvage_outcome.transformation,
                        selection_rule=salvage_outcome.selection_rule,
                    )
                if not run_result.artefacts:
                    if is_trajectory_stability_standard:
                        standard_executor_terminal_block = True
                        standard_executor_terminal_reason = "missing_executor_outputs"
                        break
                    fallback_code = _deterministic_fallback_code("no_artefacts")
                    if fallback_code is not None:
                        code = fallback_code
                        _clear_output_dir(run_result.out_dir)
                        continue
                visual_step_summary: Dict[str, Any] = {}
                visual_summary_path = run_result.out_dir / "step_summary.json"
                if visual_summary_path.exists():
                    try:
                        vloaded = json.loads(
                            visual_summary_path.read_text(encoding="utf-8")
                        )
                    except Exception:
                        vloaded = None
                    if isinstance(vloaded, dict):
                        visual_step_summary = vloaded
                    else:
                        visual_step_summary = {"raw": vloaded}
                if runner_repair_name and is_sealed_renderer_repair(runner_repair_name):
                    visual_step_summary = _write_host_input_binding_receipts(
                        out_dir=run_result.out_dir,
                        step_summary=visual_step_summary,
                        resolved_input_bindings=resolved_input_bindings,
                    )
                if is_trajectory_stability_standard:
                    terminal_status = (
                        str(visual_step_summary.get("status") or "").strip().lower()
                    )
                    if terminal_status != "ok":
                        standard_executor_terminal_block = True
                        standard_executor_terminal_reason = "executor_reported_" + (
                            terminal_status or "missing_status"
                        )
                        standard_executor_terminal_summary = dict(visual_step_summary)
                        break
                step_figures = [
                    art
                    for art in run_result.artefacts
                    if art.suffix.lower() in {".png", ".svg", ".tiff", ".tif"}
                ]
                if pipeline._enable_visual_qa and step_figures:
                    expected_numeric = _expected_numeric_annotations_for_step(
                        step=step,
                        step_summary=visual_step_summary,
                    )
                    numeric_expectations = (
                        {
                            str(path): expected_numeric
                            for path in step_figures
                            if path.suffix.lower() == ".svg"
                        }
                        if expected_numeric
                        else None
                    )
                    visual_findings = VisualQAAuditor().audit_with_expected(
                        figure_paths=step_figures,
                        expected_numeric_by_path=numeric_expectations,
                    )
                    step_record["visual_findings"] = [
                        f.model_dump() for f in visual_findings
                    ]
                    visual_errors = [
                        f for f in visual_findings if f.severity == "error"
                    ]
                    if visual_errors:
                        if sealed_renderer_authorized_code_sha256 is not None:
                            demoted_findings, blocking_visual_errors = (
                                _demote_cosmetic_visual_findings(visual_findings)
                            )
                            step_record["visual_findings"] = [
                                finding.model_dump() for finding in demoted_findings
                            ]
                            step_record["sealed_renderer_visual_repair_suppressed"] = (
                                True
                            )
                            step_record["visual_qa_demoted"] = any(
                                original.severity == "error"
                                and demoted.severity == "warning"
                                for original, demoted in zip(
                                    visual_findings, demoted_findings
                                )
                            )
                            with shared_lock:
                                findings.extend(demoted_findings)
                            if blocking_visual_errors:
                                step_record.update(
                                    {
                                        "status": "execution_failed",
                                        "diagnostic_only": True,
                                        "sealed_renderer_terminal_reason": (
                                            "visual_qa_failed"
                                        ),
                                        "llm_repair_used": False,
                                        "generation_mode": "fallback",
                                    }
                                )
                                with shared_lock:
                                    per_step_records.append(step_record)
                                    _flush_partial_manifest()
                                emit_progress(
                                    "visual_qa",
                                    (
                                        "Visual QA blocked sealed renderer "
                                        f"{step.step_id}; coder repair was not "
                                        "authorized."
                                    ),
                                    status="error",
                                    run_id=run_id,
                                    step_id=step.step_id,
                                    current_step=step_current,
                                    total_steps=total_steps,
                                )
                                return step_record
                            emit_progress(
                                "visual_qa",
                                (
                                    "Cosmetic visual QA findings were retained as "
                                    "warnings for sealed renderer "
                                    f"{step.step_id}; its verified code and outputs "
                                    "were not rewritten."
                                ),
                                status="warning",
                                run_id=run_id,
                                step_id=step.step_id,
                                current_step=step_current,
                                total_steps=total_steps,
                            )
                        elif (
                            visual_repair_attempts >= pipeline._max_code_repair_attempts
                            or not _llm_repair_budget_available()
                        ):
                            fallback_code = _deterministic_fallback_code("visual_qa")
                            if fallback_code is not None:
                                code = fallback_code
                                _clear_output_dir(run_result.out_dir)
                                continue
                            demoted_findings, blocking_visual_errors = (
                                _demote_cosmetic_visual_findings(visual_findings)
                            )
                            step_record["visual_findings"] = [
                                finding.model_dump() for finding in demoted_findings
                            ]
                            with shared_lock:
                                findings.extend(demoted_findings)
                            step_record["visual_qa_demoted"] = any(
                                original.severity == "error"
                                and demoted.severity == "warning"
                                for original, demoted in zip(
                                    visual_findings, demoted_findings
                                )
                            )
                            if blocking_visual_errors:
                                step_record["status"] = "execution_failed"
                                with shared_lock:
                                    per_step_records.append(step_record)
                                    _flush_partial_manifest()
                                emit_progress(
                                    "visual_qa",
                                    (
                                        f"Visual QA blocked {step.step_id} after "
                                        f"{visual_repair_attempts} layout repair "
                                        "attempts."
                                    ),
                                    status="error",
                                    run_id=run_id,
                                    step_id=step.step_id,
                                    current_step=step_current,
                                    total_steps=total_steps,
                                )
                                return step_record
                            emit_progress(
                                "visual_qa",
                                (
                                    f"Cosmetic visual QA findings demoted to warning "
                                    f"for {step.step_id} after "
                                    f"{visual_repair_attempts} layout repair attempts."
                                ),
                                status="warning",
                                run_id=run_id,
                                step_id=step.step_id,
                                current_step=step_current,
                                total_steps=total_steps,
                            )
                            # Fall through to contract checks and evidence
                            # registration only when every remaining visual
                            # error was a deterministic layout/cosmetic issue.
                        else:
                            visual_repair_attempts += 1
                            if not _consume_llm_repair_budget("visual"):
                                raise AssertionError(
                                    "LLM repair budget changed without mutation"
                                )
                            repair_attempts += 1
                            step_record["code_repair_attempts"] = repair_attempts
                            step_record["visual_repair_attempts"] = (
                                visual_repair_attempts
                            )
                            emit_progress(
                                "visual_qa",
                                f"Repairing figure layout for {step.step_id}.",
                                run_id=run_id,
                                step_id=step.step_id,
                                current_step=step_current,
                                total_steps=total_steps,
                                repair_attempts=repair_attempts,
                                visual_repair_attempts=visual_repair_attempts,
                            )
                            qa_log = _visual_repair_request_log(visual_findings)
                            try:
                                code = coder.repair(
                                    context=coder_context,
                                    step=step,
                                    code=code,
                                    run_log=(
                                        "Visual QA rejected one or more figure outputs "
                                        "before evidence registration. Fix the figure "
                                        "layout, preserve all tables/statistics, save PNG "
                                        "and editable SVG with the same stem, include "
                                        "publication figure exports when requested, and rerun.\n\n"
                                        + qa_log
                                        + _monotonic_concept_constraint_log()
                                    ),
                                    attempt=visual_repair_attempts,
                                    provider_budget=provider_budget,
                                    provider_category="visual_repair",
                                )
                                _sync_provider_budget()
                                llm_repair_used = True
                                _clear_output_dir(run_result.out_dir)
                                continue
                            except Exception as exc:
                                _sync_provider_budget()
                                demoted_findings, blocking_visual_errors = (
                                    _demote_cosmetic_visual_findings(visual_findings)
                                )
                                if not blocking_visual_errors:
                                    provider_finding = ValidationFinding(
                                        validator="coder",
                                        severity="warning",
                                        message=(
                                            "Cosmetic visual-layout repair was "
                                            f"unavailable for step {step.step_id}; "
                                            "the current data-valid artifacts were "
                                            f"retained: {exc}"
                                        ),
                                        detail={
                                            "step_id": step.step_id,
                                            "error_type": type(exc).__name__,
                                            "visual_repair_attempts": (
                                                visual_repair_attempts
                                            ),
                                        },
                                    )
                                    step_record["visual_findings"] = [
                                        finding.model_dump()
                                        for finding in demoted_findings
                                    ]
                                    step_record["visual_qa_demoted"] = True
                                    step_record["visual_repair_provider_failed"] = True
                                    with shared_lock:
                                        findings.extend(demoted_findings)
                                        findings.append(provider_finding)
                                    emit_progress(
                                        "visual_qa",
                                        (
                                            "Cosmetic visual repair unavailable; "
                                            f"retained current artifacts for {step.step_id}."
                                        ),
                                        status="warning",
                                        run_id=run_id,
                                        step_id=step.step_id,
                                        current_step=step_current,
                                        total_steps=total_steps,
                                    )
                                else:
                                    fallback_code = _deterministic_fallback_code(
                                        "visual_qa_repair_failed"
                                    )
                                    if fallback_code is not None:
                                        code = fallback_code
                                        _clear_output_dir(run_result.out_dir)
                                        continue
                                    with shared_lock:
                                        findings.extend(visual_findings)
                                        findings.append(
                                            ValidationFinding(
                                                validator="coder",
                                                severity="error",
                                                message=(
                                                    "Coder repair failed after visual QA "
                                                    f"for step {step.step_id}: {exc}"
                                                ),
                                            )
                                        )
                                        step_record["status"] = "repair_failed"
                                        per_step_records.append(step_record)
                                        _flush_partial_manifest()
                                    emit_progress(
                                        "visual_qa",
                                        f"Visual QA repair failed for {step.step_id}.",
                                        status="error",
                                        run_id=run_id,
                                        step_id=step.step_id,
                                        current_step=step_current,
                                        total_steps=total_steps,
                                    )
                                    return step_record
                with shared_lock:
                    completed_records_snapshot = list(per_step_records)
                early_contract_findings = _step_contract_findings(
                    step=step,
                    step_summary=visual_step_summary,
                    completed_step_records=completed_records_snapshot,
                    resolved_input_bindings=resolved_input_bindings,
                    out_dir=run_result.out_dir,
                )
                early_contract_findings += (
                    _cohort_definition_sensitivity_contract_findings(
                        step=step,
                        step_summary=visual_step_summary,
                        out_dir=run_result.out_dir,
                        run_dir=run_dir,
                        universe_path=universe_path,
                        cohort_path=cohort_path,
                        context=context,
                        completed_step_records=completed_records_snapshot,
                    )
                )
                early_contract_findings += cross_step_cohort_lock_validator.audit(
                    step=step,
                    step_summary=visual_step_summary,
                    completed_step_records=completed_records_snapshot,
                )
                early_contract_findings += cross_step_registered_output_validator.audit(
                    step=step,
                    step_summary=visual_step_summary,
                    completed_step_records=completed_records_snapshot,
                )
                early_contract_findings += (
                    cross_step_reconciliation_trace_validator.audit(
                        step=step,
                        step_summary=visual_step_summary,
                        out_dir=run_result.out_dir,
                    )
                )
                early_contract_findings += step_summary_integrity_validator.audit(
                    step=step,
                    step_summary=visual_step_summary,
                    resolved_input_bindings=resolved_input_bindings,
                    cohort_path=cohort_path,
                )
                early_contract_findings += step_summary_fraction_validator.audit(
                    step=step,
                    step_summary=visual_step_summary,
                )
                early_contract_findings += cross_step_source_status_validator.audit(
                    step=step,
                    step_summary=visual_step_summary,
                    completed_step_records=completed_records_snapshot,
                )
                early_contract_findings += primary_model_contract_validator.audit(
                    step=step,
                    step_summary=visual_step_summary,
                    context=context,
                    completed_step_records=completed_records_snapshot,
                    out_dir=run_result.out_dir,
                    cohort_path=cohort_path,
                )
                # Exposure-contract audit: if the question names a required
                # primary exposure and this primary model estimated a clearly
                # different variable, flag it so the same in-run repair loop
                # re-fits the step with the correct exposure (no full restart).
                early_contract_findings += _primary_exposure_contract_findings(
                    step=step,
                    step_summary=visual_step_summary,
                    context=context,
                )
                early_contract_findings += (
                    _primary_exposure_measurement_filter_findings(
                        step=step,
                        step_summary=visual_step_summary,
                        context=context,
                    )
                )
                # Overadjustment hard-block: if the primary exposure is a
                # composite/derived score and this model conditioned on one of
                # its constituents, route an error through the same repair loop
                # so the step re-fits without the offending covariate.
                early_contract_findings += _primary_exposure_overadjustment_findings(
                    step=step,
                    context=context,
                    out_dir=run_result.out_dir,
                )
                # Outcome-leakage hard-block + treatment-mediator / other-endpoint
                # cautions: the declared outcome appearing among predictors is
                # target leakage (error → same re-fit loop); a treatment covariate
                # or a different endpoint as predictor surfaces as a non-gating
                # caution for the analyst to verify.
                early_contract_findings += _primary_model_leakage_findings(
                    step=step,
                    context=context,
                    out_dir=run_result.out_dir,
                )
                # Figure quality and source-data errors must enter the same
                # in-run repair loop as table/model contract errors. Checking
                # them only after evidence registration produces a terminal
                # contract_failed record with no opportunity to repair the
                # generated rendering script.
                for contract_path in sorted(
                    run_result.out_dir.glob("*.figure_contract.json")
                ):
                    schema_candidate = (
                        _figure_contract_source_data_canonicalization_candidate(
                            contract_path=contract_path,
                            out_dir=run_result.out_dir,
                        )
                    )
                    if schema_candidate is None:
                        continue
                    before_contract, after_contract, source_names = schema_candidate
                    repair_id = _FIGURE_CONTRACT_SOURCE_DATA_SCHEMA_REPAIR_ID
                    if not _automatic_repair_authorized(
                        repair_id,
                        step=step,
                        source="figure_contract_schema_canonicalization",
                        before_code=before_contract,
                        after_code=after_contract,
                    ):
                        continue
                    _install_figure_contract_source_data_canonicalization(
                        contract_path=contract_path,
                        expected_before=before_contract,
                        canonical_text=after_contract,
                    )
                    step_record.setdefault(
                        "figure_contract_schema_canonicalizations", []
                    ).append(
                        {
                            "contract": contract_path.name,
                            "source_data": list(source_names),
                            "repair_id": repair_id,
                        }
                    )
                    _record_repair(
                        repair_id=repair_id,
                        step_id=str(step.step_id),
                        trigger={
                            "source": "figure_contract_schema_canonicalization",
                            "contract": contract_path.name,
                        },
                        transformation=(
                            "Canonicalized an exact local source-data descriptor "
                            "to the persistent flat FigureContract basename schema."
                        ),
                        before_code=before_contract,
                        after_code=after_contract,
                    )
                early_contract_findings += figure_contract_validator.audit(
                    step=step,
                    out_dir=run_result.out_dir,
                    run_dir=run_dir,
                    step_summary=visual_step_summary,
                )
                early_contract_findings += figure_source_validator.audit(
                    step=step,
                    out_dir=run_result.out_dir,
                    run_dir=run_dir,
                    step_summary=visual_step_summary,
                    completed_step_records=completed_records_snapshot,
                    resolved_input_bindings=resolved_input_bindings,
                )
                # For the controlled ordered-stratified method, replay the
                # agent-authored tables from the locked cohort before evidence
                # registration. Numeric/method errors therefore return to the
                # existing coder repair loop instead of becoming a late warning.
                early_contract_findings += ordered_stratified_numeric_findings(
                    cohort_path=cohort_path,
                    step=step,
                    out_dir=run_result.out_dir,
                    step_summary=visual_step_summary,
                )
                unowned_sealed_markers = _unowned_sealed_authority_markers(
                    visual_step_summary,
                    authorized_code_sha256=(sealed_renderer_authorized_code_sha256),
                )
                if unowned_sealed_markers:
                    early_contract_findings.append(
                        ValidationFinding(
                            validator="sealed_renderer_authority",
                            severity="error",
                            message=(
                                "Generated code reported sealed-renderer authority "
                                "that the host did not authorize before execution."
                            ),
                            detail={
                                "step_id": step.step_id,
                                "unowned_authority_markers": unowned_sealed_markers,
                            },
                        )
                    )
                if (
                    sealed_renderer_authorized_code_sha256 is not None
                    and visual_step_summary.get("rendering_only") is not True
                ):
                    early_contract_findings.append(
                        ValidationFinding(
                            validator="sealed_renderer_authority",
                            severity="error",
                            message=(
                                "The authorized figure adapter did not report its "
                                "required rendering-only execution scope."
                            ),
                            detail={
                                "step_id": step.step_id,
                                "repair_id": sealed_renderer_repair_id,
                                "reported_rendering_only": visual_step_summary.get(
                                    "rendering_only"
                                ),
                            },
                        )
                    )
                reported_slot_bindings = visual_step_summary.get(
                    "planner_product_slot_bindings"
                )
                reported_product_slots = (
                    {
                        str(product): str(binding.get("slot") or "")
                        for product, binding in reported_slot_bindings.items()
                        if isinstance(binding, Mapping)
                    }
                    if isinstance(reported_slot_bindings, Mapping)
                    else {}
                )
                if sealed_renderer_authorized_code_sha256 is not None:
                    parent_step_id = str(step.step_id or "").removesuffix("_figure")
                    parent_out = run_dir / "steps" / parent_step_id / "outputs"
                    try:
                        read_digest_bound_artifact_snapshot(
                            parent_out=parent_out,
                            artifact_digests=sealed_renderer_parent_digests,
                        )
                        step_record["sealed_renderer_parent_receipt_verified"] = True
                    except ValueError:
                        step_record["sealed_renderer_parent_receipt_verified"] = False
                        early_contract_findings.append(
                            ValidationFinding(
                                validator="sealed_renderer_authority",
                                severity="error",
                                message=(
                                    "The sealed renderer's direct-parent inputs "
                                    "changed before host receipt."
                                ),
                                detail={
                                    "step_id": step.step_id,
                                    "repair_id": sealed_renderer_repair_id,
                                },
                            )
                        )
                if sealed_renderer_authorized_code_sha256 is not None and (
                    visual_step_summary.get("sealed_renderer_repair")
                    != sealed_renderer_repair_id
                    or visual_step_summary.get("sealed_renderer_implementation_sha256")
                    != sealed_renderer_implementation_sha256
                    or visual_step_summary.get("sealed_renderer_parent_digests")
                    != sealed_renderer_parent_digests
                    or reported_product_slots
                    != sealed_renderer_authorized_product_slots
                ):
                    early_contract_findings.append(
                        ValidationFinding(
                            validator="sealed_renderer_authority",
                            severity="error",
                            message=(
                                "The rendered summary did not preserve the exact "
                                "sealed renderer identity and implementation digest."
                            ),
                            detail={
                                "step_id": step.step_id,
                                "expected_repair_id": sealed_renderer_repair_id,
                                "reported_repair_id": visual_step_summary.get(
                                    "sealed_renderer_repair"
                                ),
                                "expected_implementation_sha256": (
                                    sealed_renderer_implementation_sha256
                                ),
                                "reported_implementation_sha256": (
                                    visual_step_summary.get(
                                        "sealed_renderer_implementation_sha256"
                                    )
                                ),
                                "expected_parent_digests": (
                                    sealed_renderer_parent_digests
                                ),
                                "reported_parent_digests": visual_step_summary.get(
                                    "sealed_renderer_parent_digests"
                                ),
                                "expected_product_slots": (
                                    sealed_renderer_authorized_product_slots
                                ),
                                "reported_product_slots": reported_product_slots,
                            },
                        )
                    )
                # A deterministic PRIMARY runner owns its step's contract: if it
                # produced the core estimate, planner-requested extra outputs it
                # does not emit are advisory, never a reason to repair-away the
                # trustworthy estimate.
                early_contract_findings = _demote_step_contract_for_primary_runner(
                    step_record, visual_step_summary, early_contract_findings
                )
                early_contract_errors = [
                    f for f in early_contract_findings if f.severity == "error"
                ]
                if early_contract_errors:
                    locked_data_quality_issues = (
                        _locked_measurement_data_quality_issues(early_contract_errors)
                    )
                    if locked_data_quality_issues:
                        step_record.update(
                            {
                                "status": "contract_failed",
                                "diagnostic_only": True,
                                "measurement_provenance_repair_suppressed": True,
                                "measurement_provenance_terminal_reason": (
                                    "locked_cohort_data_quality_failed"
                                ),
                                "measurement_provenance_terminal_issues": (
                                    locked_data_quality_issues
                                ),
                                "contract_findings": [
                                    finding.model_dump()
                                    for finding in early_contract_findings
                                ],
                                "step_summary": visual_step_summary,
                                "llm_repair_used": llm_repair_used,
                                "generation_mode": current_generation_mode,
                                "code_repair_attempts": repair_attempts,
                                "contract_repair_attempts": (contract_repair_attempts),
                            }
                        )
                        with shared_lock:
                            findings.extend(early_contract_findings)
                            per_step_records.append(step_record)
                            _flush_partial_manifest()
                        emit_progress(
                            "contract",
                            (
                                "Locked-cohort measurement provenance failed for "
                                f"{step.step_id}; retained diagnostics without "
                                "attempting a code repair."
                            ),
                            status="error",
                            run_id=run_id,
                            step_id=step.step_id,
                            current_step=step_current,
                            total_steps=total_steps,
                        )
                        return step_record
                    if sealed_renderer_authorized_code_sha256 is not None:
                        step_record.update(
                            {
                                "status": "contract_failed",
                                "diagnostic_only": True,
                                "sealed_renderer_contract_repair_suppressed": True,
                                "sealed_renderer_terminal_reason": (
                                    "output_contract_failed"
                                ),
                                "contract_findings": [
                                    finding.model_dump()
                                    for finding in early_contract_findings
                                ],
                                "step_summary": visual_step_summary,
                                "llm_repair_used": False,
                                "generation_mode": "fallback",
                            }
                        )
                        with shared_lock:
                            findings.extend(early_contract_findings)
                            per_step_records.append(step_record)
                            _flush_partial_manifest()
                        emit_progress(
                            "contract",
                            (
                                "Contract validation blocked sealed renderer "
                                f"{step.step_id}; its code and outputs were retained "
                                "without coder repair."
                            ),
                            status="error",
                            run_id=run_id,
                            step_id=step.step_id,
                            current_step=step_current,
                            total_steps=total_steps,
                        )
                        return step_record
                    if is_trajectory_stability_standard:
                        standard_executor_terminal_block = True
                        standard_executor_terminal_reason = (
                            "executor_output_contract_failed"
                        )
                        standard_executor_terminal_summary = dict(visual_step_summary)
                        standard_executor_terminal_findings = list(
                            early_contract_findings
                        )
                        break
                    if pipeline._enable_deterministic_runner_repair:
                        before_repair_code = code
                        summary_repair = _deterministic_summary_repair(
                            code=code,
                            step_summary=visual_step_summary,
                            previous_repair=runner_repair_name,
                            analysis_family=local_runtime_state.analysis_family,
                        )
                        summary_repair = _authorize_automatic_repair(
                            summary_repair,
                            step=step,
                            source="deterministic_summary_repair_before_contract",
                            before_code=before_repair_code,
                        )
                    else:
                        summary_repair = None
                    if summary_repair is not None:
                        contract_repair_attempts += 1
                        repair_attempts += 1
                        runner_repair_name, code = summary_repair
                        step_record["runner_repair"] = runner_repair_name
                        step_record["code_repair_attempts"] = repair_attempts
                        step_record["contract_repair_attempts"] = (
                            contract_repair_attempts
                        )
                        _record_repair(
                            repair_id=runner_repair_name,
                            step_id=step.step_id,
                            trigger={
                                "source": "deterministic_summary_repair",
                                "step_summary_keys": sorted(
                                    str(key) for key in visual_step_summary.keys()
                                ),
                                "contract_findings": [
                                    f.message for f in early_contract_errors
                                ],
                            },
                            transformation=(
                                "Deterministic repair before LLM contract repair."
                            ),
                            before_code=before_repair_code,
                            after_code=code,
                        )
                        emit_progress(
                            "runner_repair",
                            (
                                f"Applied deterministic summary repair for "
                                f"{step.step_id}: {runner_repair_name}."
                            ),
                            run_id=run_id,
                            step_id=step.step_id,
                            current_step=step_current,
                            total_steps=total_steps,
                        )
                        _clear_output_dir(run_result.out_dir)
                        continue
                    if pipeline._enable_deterministic_runner_repair:
                        before_repair_code = code
                        contract_repair = deterministic_contract_repair(
                            code=code,
                            findings=early_contract_errors,
                            previous_repair=runner_repair_name,
                        )
                        contract_repair = _authorize_automatic_repair(
                            contract_repair,
                            step=step,
                            source="deterministic_contract_repair",
                            before_code=before_repair_code,
                        )
                    else:
                        contract_repair = None
                    if contract_repair is not None:
                        contract_repair_attempts += 1
                        repair_attempts += 1
                        runner_repair_name, code = contract_repair
                        step_record["runner_repair"] = runner_repair_name
                        step_record["code_repair_attempts"] = repair_attempts
                        step_record["contract_repair_attempts"] = (
                            contract_repair_attempts
                        )
                        _record_repair(
                            repair_id=runner_repair_name,
                            step_id=step.step_id,
                            trigger={
                                "source": "deterministic_contract_repair",
                                "contract_findings": [
                                    f.message for f in early_contract_errors
                                ],
                            },
                            transformation=(
                                "Deterministically removed covariates named by "
                                "objective contract/audit findings."
                            ),
                            before_code=before_repair_code,
                            after_code=code,
                        )
                        emit_progress(
                            "runner_repair",
                            (
                                f"Applied deterministic contract repair for "
                                f"{step.step_id}: {runner_repair_name}."
                            ),
                            run_id=run_id,
                            step_id=step.step_id,
                            current_step=step_current,
                            total_steps=total_steps,
                        )
                        _clear_output_dir(run_result.out_dir)
                        continue
                    if (
                        contract_repair_attempts >= pipeline._max_code_repair_attempts
                        or not _llm_repair_budget_available()
                    ):
                        with shared_lock:
                            findings.extend(early_contract_findings)
                            step_record["status"] = "contract_failed"
                            step_record["contract_findings"] = [
                                f.model_dump() for f in early_contract_findings
                            ]
                            step_record["step_summary"] = visual_step_summary
                            per_step_records.append(step_record)
                            _flush_partial_manifest()
                        emit_progress(
                            "contract",
                            (
                                f"Contract violation could not be repaired for "
                                f"{step.step_id}; no LLM repair budget remains."
                            ),
                            status="error",
                            run_id=run_id,
                            step_id=step.step_id,
                            current_step=step_current,
                            total_steps=total_steps,
                        )
                        return step_record

                    contract_repair_attempts += 1
                    if not _consume_llm_repair_budget("contract"):
                        raise AssertionError(
                            "LLM repair budget changed without mutation"
                        )
                    repair_attempts += 1
                    step_record["code_repair_attempts"] = repair_attempts
                    step_record["contract_repair_attempts"] = contract_repair_attempts
                    emit_progress(
                        "coder",
                        f"Repairing contract violation for {step.step_id}.",
                        run_id=run_id,
                        step_id=step.step_id,
                        current_step=step_current,
                        total_steps=total_steps,
                        repair_attempts=repair_attempts,
                        contract_repair_attempts=contract_repair_attempts,
                    )
                    contract_log = _contract_repair_log(early_contract_errors)
                    structured_repair_ticket = typed_repair_ticket(
                        early_contract_errors
                    )
                    repair_guidance = _step_contract_repair_guidance(
                        step=step,
                        step_summary=visual_step_summary,
                        code=code,
                        input_bindings=resolved_input_bindings,
                    )
                    try:
                        code = coder.repair(
                            context=coder_context,
                            step=step,
                            code=code,
                            run_log=(
                                "The script executed but failed the machine-readable "
                                "step contract. Revise the analysis code; do not change "
                                "the research question. Ensure required primary metrics "
                                "are computed and written to step_summary.json with "
                                "explicit numeric keys or nested statistic fields.\n\n"
                                "STEP SUMMARY:\n"
                                + json.dumps(
                                    visual_step_summary,
                                    indent=2,
                                    ensure_ascii=False,
                                    default=str,
                                )
                                + "\n\nREPAIR GUIDANCE:\n"
                                + repair_guidance
                                + "\n\nTYPED REPAIR TICKET (authoritative routing):\n"
                                + json.dumps(
                                    structured_repair_ticket,
                                    indent=2,
                                    ensure_ascii=False,
                                    default=str,
                                )
                                + "\n\nSTRUCTURED CONTRACT FINDINGS (authoritative):\n"
                                + contract_log
                                + _monotonic_concept_constraint_log()
                            ),
                            attempt=contract_repair_attempts,
                            provider_budget=provider_budget,
                            provider_category="contract_repair",
                        )
                        _sync_provider_budget()
                        llm_repair_used = True
                        _clear_output_dir(run_result.out_dir)
                        continue
                    except Exception as exc:
                        _sync_provider_budget()
                        fallback_code = _deterministic_fallback_code(
                            "contract_repair_failed"
                        )
                        if fallback_code is not None:
                            code = fallback_code
                            _clear_output_dir(run_result.out_dir)
                            continue
                        with shared_lock:
                            findings.extend(early_contract_findings)
                            findings.append(
                                ValidationFinding(
                                    validator="coder",
                                    severity="error",
                                    message=(
                                        f"Coder repair failed after contract check "
                                        f"for step {step.step_id}: {exc}"
                                    ),
                                )
                            )
                            step_record["status"] = "repair_failed"
                            step_record["contract_findings"] = [
                                f.model_dump() for f in early_contract_findings
                            ]
                            step_record["step_summary"] = visual_step_summary
                            per_step_records.append(step_record)
                            _flush_partial_manifest()
                        emit_progress(
                            "coder",
                            f"Contract repair failed for {step.step_id}.",
                            status="error",
                            run_id=run_id,
                            step_id=step.step_id,
                            current_step=step_current,
                            total_steps=total_steps,
                        )
                        return step_record
                if (
                    pipeline._enable_deterministic_runner_repair
                    and sealed_renderer_authorized_code_sha256 is None
                ):
                    before_repair_code = code
                    summary_repair = _deterministic_summary_repair(
                        code=code,
                        step_summary=visual_step_summary,
                        previous_repair=runner_repair_name,
                        analysis_family=local_runtime_state.analysis_family,
                    )
                    summary_repair = _authorize_automatic_repair(
                        summary_repair,
                        step=step,
                        source="deterministic_summary_repair_after_contract",
                        before_code=before_repair_code,
                    )
                else:
                    summary_repair = None
                if summary_repair is not None:
                    runner_repair_name, code = summary_repair
                    step_record["runner_repair"] = runner_repair_name
                    _record_repair(
                        repair_id=runner_repair_name,
                        step_id=step.step_id,
                        trigger={
                            "source": "deterministic_summary_repair",
                            "step_summary_keys": sorted(
                                str(key) for key in visual_step_summary.keys()
                            ),
                        },
                        transformation="Deterministic repair after step_summary contract inspection.",
                        before_code=before_repair_code,
                        after_code=code,
                    )
                    emit_progress(
                        "runner_repair",
                        f"Applied deterministic summary repair for {step.step_id}: {runner_repair_name}.",
                        run_id=run_id,
                        step_id=step.step_id,
                        current_step=step_current,
                        total_steps=total_steps,
                    )
                    _clear_output_dir(run_result.out_dir)
                    continue
                deterministic_contract_approved_code_digest = candidate_code_digest
                step_record["deterministic_contract_approved_code_sha256"] = (
                    deterministic_contract_approved_code_digest
                )
                # Return once through the digest gate for the single final LLM
                # concept audit.  The output directory is intentionally retained;
                # on approval it proceeds without re-executing unchanged code.
                continue

            if log_path.exists():
                run_log = log_path.read_text(encoding="utf-8", errors="replace")
            else:
                run_log = (run_result.stdout or "") + "\n" + (run_result.stderr or "")
            if is_trajectory_stability_standard:
                # A timeout can interrupt the standard executor between its
                # private streaming write and atomic rename.  That file is an
                # implementation detail, not a diagnostic product, and must
                # be gone before the generic output-directory scan below can
                # register it as evidence.
                _remove_standard_executor_pending_artifacts(run_result.out_dir)
                standard_executor_terminal_block = True
                standard_executor_terminal_reason = "executor_runtime_failure"
                break
            if sealed_renderer_authorized_code_sha256 is not None:
                runtime_finding = ValidationFinding(
                    validator="sealed_renderer_authority",
                    severity="error",
                    message=(
                        "The authorized rendering-only adapter failed at runtime; "
                        "its diagnostics were retained and no deterministic or LLM "
                        "code repair was allowed."
                    ),
                    detail={
                        "step_id": step.step_id,
                        "repair_id": sealed_renderer_repair_id,
                        "returncode": run_result.returncode,
                        "timed_out": run_result.timed_out,
                    },
                )
                step_record.update(
                    {
                        "status": "execution_failed",
                        "diagnostic_only": True,
                        "sealed_renderer_runtime_repair_suppressed": True,
                        "sealed_renderer_terminal_reason": "runtime_failure",
                        "llm_repair_used": False,
                        "generation_mode": "fallback",
                    }
                )
                with shared_lock:
                    findings.append(runtime_finding)
                    per_step_records.append(step_record)
                    _flush_partial_manifest()
                emit_progress(
                    "runner",
                    (
                        f"Sealed renderer failed for {step.step_id}; coder repair "
                        "was not authorized."
                    ),
                    status="error",
                    run_id=run_id,
                    step_id=step.step_id,
                    current_step=step_current,
                    total_steps=total_steps,
                )
                return step_record
            if pipeline._enable_deterministic_runner_repair:
                before_repair_code = code
                plugin_repair = pipeline._case_plugin_registry.repair_code(
                    context=context,
                    step=step,
                    code=code,
                    run_log=(run_log + _monotonic_concept_constraint_log()),
                )
                if plugin_repair is not None and plugin_repair[0] != runner_repair_name:
                    runner_repair = plugin_repair
                else:
                    runner_repair = _deterministic_runner_repair(
                        code=code,
                        run_log=run_log,
                        previous_repair=runner_repair_name,
                        analysis_family=local_runtime_state.analysis_family,
                    )
                runner_repair = _authorize_automatic_repair(
                    runner_repair,
                    step=step,
                    source=(
                        "case_plugin_repair"
                        if plugin_repair is not None and runner_repair is plugin_repair
                        else "deterministic_runner_repair"
                    ),
                    before_code=before_repair_code,
                )
            else:
                runner_repair = None
            if runner_repair is not None:
                runner_repair_name, code = runner_repair
                step_record["runner_repair"] = runner_repair_name
                _record_repair(
                    repair_id=runner_repair_name,
                    step_id=step.step_id,
                    trigger={
                        "source": "deterministic_runner_repair",
                        "run_log_tail": run_log[-1200:],
                    },
                    transformation="Deterministic repair after runner failure.",
                    before_code=before_repair_code,
                    after_code=code,
                )
                emit_progress(
                    "runner_repair",
                    f"Applied deterministic runner repair for {step.step_id}: {runner_repair_name}.",
                    run_id=run_id,
                    step_id=step.step_id,
                    current_step=step_current,
                    total_steps=total_steps,
                )
                _clear_output_dir(run_result.out_dir)
                continue

            if (
                runtime_repair_attempts >= pipeline._max_code_repair_attempts
                or not _llm_repair_budget_available()
            ):
                fallback_code = _deterministic_fallback_code("execution_failure")
                if fallback_code is not None:
                    code = fallback_code
                    _clear_output_dir(run_result.out_dir)
                    continue
                with shared_lock:
                    findings.append(
                        ValidationFinding(
                            validator="runner",
                            severity="error",
                            message=(
                                f"Step {step.step_id} "
                                f"{'timed out' if run_result.timed_out else 'failed'} "
                                f"with returncode {run_result.returncode}."
                            ),
                        )
                    )
                    step_record["status"] = "execution_failed"
                    per_step_records.append(step_record)
                    _flush_partial_manifest()
                emit_progress(
                    "runner",
                    f"Execution failed for {step.step_id}.",
                    status="error",
                    run_id=run_id,
                    step_id=step.step_id,
                    current_step=step_current,
                    total_steps=total_steps,
                )
                return step_record

            runtime_repair_applied = False
            runtime_repair_fallback_applied = False
            while (
                runtime_repair_attempts < pipeline._max_code_repair_attempts
                and _llm_repair_budget_available()
            ):
                repair_attempts += 1
                runtime_repair_attempts += 1
                if not _consume_llm_repair_budget("runtime"):
                    raise AssertionError("LLM repair budget changed without mutation")
                step_record["code_repair_attempts"] = repair_attempts
                step_record["runtime_repair_attempts"] = runtime_repair_attempts
                emit_progress(
                    "coder",
                    f"Repairing failed script for {step.step_id}.",
                    run_id=run_id,
                    step_id=step.step_id,
                    current_step=step_current,
                    total_steps=total_steps,
                    repair_attempts=repair_attempts,
                )
                try:
                    repaired_code = coder.repair(
                        context=coder_context,
                        step=step,
                        code=code,
                        run_log=run_log,
                        attempt=repair_attempts,
                        provider_budget=provider_budget,
                        provider_category="runtime_repair",
                    )
                    _sync_provider_budget()
                    if not _python_repair_is_materially_changed(code, repaired_code):
                        raise RuntimeError(
                            "Runtime repair returned no material Python change."
                        )
                    code = repaired_code
                    llm_repair_used = True
                    runtime_repair_applied = True
                    _clear_output_dir(run_result.out_dir)
                    break
                except Exception as exc:
                    _sync_provider_budget()
                    # Transport/parse failure did not change the candidate. Retry
                    # the repair request itself with the same code and traceback;
                    # never pay to execute a digest whose failure is already known.
                    message = str(exc).lower()
                    is_noop_repair = "no material python change" in message
                    is_transient = (
                        isinstance(exc, json.JSONDecodeError)
                        or "expecting value" in message
                        or ("json" in message and "decode" in message)
                        or "503" in message
                        or "rate" in message
                    )
                    can_retry_repair = bool(
                        (is_transient or is_noop_repair)
                        and runtime_repair_attempts < pipeline._max_code_repair_attempts
                        and _llm_repair_budget_available()
                        and not provider_budget.exhausted
                    )
                    if can_retry_repair:
                        emit_progress(
                            "coder",
                            (
                                f"Repair attempt did not yield usable code for "
                                f"{step.step_id} "
                                f"(attempt {repair_attempts}): {type(exc).__name__}; "
                                "retrying the repair without re-executing unchanged code."
                            ),
                            run_id=run_id,
                            step_id=step.step_id,
                            current_step=step_current,
                            total_steps=total_steps,
                            repair_attempts=repair_attempts,
                        )
                        continue

                    # The causal failure is the unavailable repair, not a new
                    # runner failure. Preserve that reason even when the logical
                    # or provider-call budget became exhausted on this attempt.
                    fallback_code = _deterministic_fallback_code("repair_failed")
                    if fallback_code is not None:
                        code = fallback_code
                        runtime_repair_fallback_applied = True
                        _clear_output_dir(run_result.out_dir)
                        break
                    with shared_lock:
                        findings.append(
                            ValidationFinding(
                                validator="coder",
                                severity="error",
                                message=(
                                    f"Coder repair failed for step {step.step_id}: {exc}"
                                ),
                            )
                        )
                        step_record["status"] = "repair_failed"
                        per_step_records.append(step_record)
                        _flush_partial_manifest()
                    emit_progress(
                        "coder",
                        f"Repair failed for {step.step_id}.",
                        status="error",
                        run_id=run_id,
                        step_id=step.step_id,
                        current_step=step_current,
                        total_steps=total_steps,
                    )
                    return step_record

            if runtime_repair_applied or runtime_repair_fallback_applied:
                continue

        publication_step = _step_requires_publication_figure_exports(
            step
        ) and not step_record.get("deterministic_standard_analysis")
        # A deterministic data-only auxiliary produces registered tables rather
        # than an inline figure; a separate rendering step owns its export. Names
        # and narrative intent are deliberately absent from the predicate above.
        # A genuine figure method/output contract still fails closed here.
        figure_role = (
            "publication_figure"
            if publication_step
            else "analysis_figure" if _step_expects_figure(step) else None
        )
        if (
            publication_step
            and not _has_figure_exports(run_result.out_dir)
            and sealed_renderer_authorized_code_sha256 is None
        ):
            sibling_repair_id = "sibling_figure_exports_promote_v1"
            promoted = None
            if _automatic_repair_authorized(
                sibling_repair_id,
                step=step,
                source="publication_figure_sibling_promotion",
            ):
                promoted = _promote_sibling_figure_exports(out_dir=run_result.out_dir)
            if promoted is not None:
                runner_repair_name = promoted
                step_record["runner_repair"] = promoted
                _record_repair(
                    repair_id=promoted,
                    step_id=step.step_id,
                    trigger={"source": "publication_figure_sibling_promotion"},
                    transformation="Promoted sibling figure exports into canonical outputs directory.",
                )
            else:
                rescued = None
                if _step_has_figure_only_output_contract(
                    step
                ) and deterministic_figure_family_supported_for_upstream(
                    run_dir, step.step_id
                ):
                    rescued = _repair_publication_figure_in_staging(
                        run_dir=run_dir,
                        current_step_id=step.step_id,
                        out_dir=run_result.out_dir,
                        step_text=f"{step.intent} {step.method}",
                        authorizer=lambda repair_id: _automatic_repair_authorized(
                            repair_id,
                            step=step,
                            source="typed_publication_bundle_rescue",
                        ),
                    )
                if rescued is not None:
                    runner_repair_name = rescued
                    step_record["runner_repair"] = rescued
                    _record_repair(
                        repair_id=rescued,
                        step_id=step.step_id,
                        trigger={"source": "typed_publication_bundle_rescue"},
                        transformation=(
                            "Rendered deterministic publication figure bundle "
                            "from the registered parent outputs for this step type."
                        ),
                    )
                else:
                    parent_step_id = str(step.step_id or "").removesuffix("_figure")
                    direct_parent = run_dir / "steps" / parent_step_id
                    promoted = None
                    if (
                        parent_step_id != str(step.step_id or "")
                        and direct_parent.is_dir()
                        and _automatic_repair_authorized(
                            "publication_bundle_promote_v1",
                            step=step,
                            source="publication_figure_prior_bundle_promotion",
                        )
                    ):
                        promoted = _promote_prior_publication_bundle(
                            run_dir=run_dir,
                            current_step_id=step.step_id,
                            out_dir=run_result.out_dir,
                            require_declared_sources=True,
                        )
                    if promoted is not None:
                        runner_repair_name = promoted
                        step_record["runner_repair"] = promoted
                        _record_repair(
                            repair_id=promoted,
                            step_id=step.step_id,
                            trigger={
                                "source": "publication_figure_prior_bundle_promotion"
                            },
                            transformation="Promoted prior publication figure bundle into current outputs directory.",
                        )

        if _should_attempt_detached_figure_binding(
            out_dir=run_result.out_dir,
            sealed_renderer_authorized_code_sha256=(
                sealed_renderer_authorized_code_sha256
            ),
        ):
            with shared_lock:
                repair_binding_records = list(per_step_records)
            detached_repair_binding = _detached_figure_repair_binding(
                step=step,
                plan=plan,
                completed_records=repair_binding_records,
            )
        else:
            detached_repair_binding = None
        repair_source_evidence_ids: List[str] = []
        repair_evidence_metadata: Dict[str, Any] = {}
        if detached_repair_binding is not None:
            (
                repair_target_step_id,
                repair_source_step_id,
                repair_source_evidence_ids,
            ) = detached_repair_binding
            step_record["repair_target_step_id"] = repair_target_step_id
            step_record["source_evidence_ids"] = list(repair_source_evidence_ids)
            repair_evidence_metadata = {
                "repair_target_step_id": repair_target_step_id,
                "source_step_id": repair_source_step_id,
                "source_evidence_ids": list(repair_source_evidence_ids),
            }
            # Persist the same orchestrator binding in the registered summary.
            # The renderer may suggest a parent, but this exact value comes only
            # from the current plan + latest outer execution ledger above.
            summary_path = run_result.out_dir / "step_summary.json"
            try:
                summary_payload = (
                    json.loads(summary_path.read_text(encoding="utf-8"))
                    if summary_path.exists()
                    else {}
                )
            except Exception:
                summary_payload = {}
            if not isinstance(summary_payload, dict):
                summary_payload = {"raw": summary_payload}
            figure_exports = sorted(
                path.name
                for path in run_result.out_dir.iterdir()
                if path.is_file()
                and path.suffix.lower()
                in {".png", ".svg", ".pdf", ".tiff", ".tif", ".pptx"}
            )
            summary_payload.update(
                {
                    "rendering_only": True,
                    "source_step_id": repair_source_step_id,
                    "repair_target_step_id": repair_target_step_id,
                    "source_evidence_ids": list(repair_source_evidence_ids),
                    "figure_files": figure_exports,
                }
            )
            summary_path.write_text(
                json.dumps(
                    summary_payload,
                    indent=2,
                    ensure_ascii=False,
                    default=str,
                ),
                encoding="utf-8",
            )

        lineage_input_evidence_ids = list(
            dict.fromkeys([*resolved_input_evidence_ids, *repair_source_evidence_ids])
        )

        if standard_executor_terminal_block:
            # Defence in depth for every terminal path: only published
            # diagnostics may reach evidence enumeration.
            _remove_standard_executor_pending_artifacts(run_result.out_dir)
        if run_result.outputs_safe_to_collect:
            run_result.artefacts = sorted(
                p
                for p in run_result.out_dir.iterdir()
                if p.is_file()
                and not (
                    deterministic_standard_executor_used
                    and _is_standard_executor_internal_artifact(p)
                )
            )
        else:
            # A sandbox backend could not prove that a timed-out writer was
            # stopped. Never enumerate or hash its mutable mount. The script
            # and host-written run log remain available outside this list.
            run_result.artefacts = []

        if publication_step and not _has_figure_exports(run_result.out_dir):
            with shared_lock:
                findings.append(
                    ValidationFinding(
                        validator="publication_figure_outputs",
                        severity="error",
                        message=(
                            f"Step {step.step_id} completed without any publication-figure exports."
                        ),
                    )
                )
                step_record["status"] = "execution_failed"
                per_step_records.append(step_record)
                _flush_partial_manifest()
            emit_progress(
                "runner",
                f"Publication figure missing for {step.step_id}.",
                status="error",
                run_id=run_id,
                step_id=step.step_id,
                current_step=step_current,
                total_steps=total_steps,
            )
            return step_record

        # Finalise every result-bearing figure before any output is copied into
        # EvidenceStore or any numeric claim is registered.  The staged repair
        # replaces the entire output directory, so running it after registration
        # would leave evidence digests and claims bound to a retired draft.
        step_summary = _load_step_summary_from_outputs(run_result.out_dir)
        if runner_repair_name and is_sealed_renderer_repair(runner_repair_name):
            step_summary = _write_host_input_binding_receipts(
                out_dir=run_result.out_dir,
                step_summary=step_summary,
                resolved_input_bindings=resolved_input_bindings,
            )
        _ensure_step_figure_contract(
            step=step,
            out_dir=run_result.out_dir,
            step_summary=step_summary,
            evidence_ids=[script_record.evidence_id, *lineage_input_evidence_ids],
        )
        with shared_lock:
            preseal_completed_records = list(per_step_records)
        preseal_contract_findings = figure_contract_validator.audit(
            step=step,
            out_dir=run_result.out_dir,
            run_dir=run_dir,
            step_summary=step_summary,
        )
        preseal_source_findings = figure_source_validator.audit(
            step=step,
            out_dir=run_result.out_dir,
            run_dir=run_dir,
            step_summary=step_summary,
            completed_step_records=preseal_completed_records,
            resolved_input_bindings=resolved_input_bindings,
        )
        preseal_figure_errors = [
            finding
            for finding in preseal_contract_findings + preseal_source_findings
            if finding.severity == "error"
        ]
        repairable_publication_step = (
            publication_step
            and sealed_renderer_authorized_code_sha256 is None
            and _step_has_figure_only_output_contract(step)
            and deterministic_figure_family_supported_for_upstream(
                run_dir, step.step_id
            )
        )
        if repairable_publication_step and preseal_figure_errors:
            repaired = _repair_publication_figure_in_staging(
                run_dir=run_dir,
                current_step_id=step.step_id,
                out_dir=run_result.out_dir,
                step_text=f"{step.intent} {step.method}",
                authorizer=lambda repair_id: _automatic_repair_authorized(
                    repair_id,
                    step=step,
                    source="publication_figure_quality_repair",
                ),
            )
            if repaired is not None:
                runner_repair_name = repaired
                step_record["runner_repair"] = repaired
                _record_repair(
                    repair_id=repaired,
                    step_id=step.step_id,
                    trigger={
                        "source": "publication_figure_quality_repair",
                        "blocked_by": [
                            finding.message for finding in preseal_figure_errors[:5]
                        ],
                    },
                    transformation=(
                        "Replaced invalid figure-step exports with a deterministic "
                        "publication figure from the registered parent table for "
                        "this step type before evidence sealing."
                    ),
                )
                step_summary = _load_step_summary_from_outputs(run_result.out_dir)
                if is_sealed_renderer_repair(repaired):
                    step_summary = _write_host_input_binding_receipts(
                        out_dir=run_result.out_dir,
                        step_summary=step_summary,
                        resolved_input_bindings=resolved_input_bindings,
                    )
                _ensure_step_figure_contract(
                    step=step,
                    out_dir=run_result.out_dir,
                    step_summary=step_summary,
                    evidence_ids=[
                        script_record.evidence_id,
                        *lineage_input_evidence_ids,
                    ],
                )
                preseal_contract_findings = figure_contract_validator.audit(
                    step=step,
                    out_dir=run_result.out_dir,
                    run_dir=run_dir,
                    step_summary=step_summary,
                )
                preseal_source_findings = figure_source_validator.audit(
                    step=step,
                    out_dir=run_result.out_dir,
                    run_dir=run_dir,
                    step_summary=step_summary,
                    completed_step_records=preseal_completed_records,
                    resolved_input_bindings=resolved_input_bindings,
                )

        preseal_figure_errors = [
            finding
            for finding in preseal_contract_findings + preseal_source_findings
            if finding.severity == "error"
        ]
        if preseal_figure_errors:
            preseal_contract_findings = _bind_findings_to_step_attempt(
                preseal_contract_findings,
                step_id=step.step_id,
                attempt_id=attempt_id,
                checkpoint_id=review_checkpoint_id,
            )
            preseal_source_findings = _bind_findings_to_step_attempt(
                preseal_source_findings,
                step_id=step.step_id,
                attempt_id=attempt_id,
                checkpoint_id=review_checkpoint_id,
            )
            step_record.update(
                {
                    "status": "contract_failed",
                    "diagnostic_only": True,
                    "step_summary": step_summary,
                    "contract_findings": [
                        finding.model_dump() for finding in preseal_contract_findings
                    ],
                    "figure_source_findings": [
                        finding.model_dump() for finding in preseal_source_findings
                    ],
                    "evidence_ids": [script_record.evidence_id],
                    "result_evidence_sealed": False,
                }
            )
            with shared_lock:
                findings.extend(preseal_contract_findings)
                findings.extend(preseal_source_findings)
                per_step_records.append(step_record)
                _flush_partial_manifest()
            emit_progress(
                "contract",
                f"Figure validation failed before evidence sealing for {step.step_id}.",
                status="error",
                run_id=run_id,
                step_id=step.step_id,
                current_step=step_current,
                total_steps=total_steps,
            )
            return step_record

        # This is the seal boundary.  From here onward result artifacts are
        # immutable: validation may fail closed, but no repair can mutate them.
        if run_result.outputs_safe_to_collect:
            run_result.artefacts = sorted(
                path
                for path in run_result.out_dir.iterdir()
                if path.is_file()
                and not (
                    deterministic_standard_executor_used
                    and _is_standard_executor_internal_artifact(path)
                )
            )
        sealed_result_digests = {
            path.name: sha256_of_file(path) for path in run_result.artefacts
        }
        step_record["result_seal_sha256"] = sha256_of_bytes(
            json.dumps(
                sealed_result_digests,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        step_record["result_evidence_sealed"] = True

        evidence_ids_for_step: List[str] = [script_record.evidence_id]
        pending_success_aliases: Dict[str, List[str]] = {}
        step_summary_record_id: Optional[str] = None
        for art in run_result.artefacts:
            if not run_result.outputs_safe_to_collect:
                # Defence in depth if a custom runner supplied an artefact
                # list despite declaring its output mount unsafe.
                continue
            # Do not rely only on deletion/enumeration timing: an isolated
            # writer interrupted during teardown could recreate its private
            # streaming file.  Internal work products are never evidence,
            # even if a runner reports one explicitly.
            if deterministic_standard_executor_used and (
                _is_standard_executor_internal_artifact(art)
            ):
                continue
            step_aliases = _semantic_aliases_for(step, art)
            generation_mode = _script_generation_mode(
                repair_attempts=repair_attempts,
                fallback_used=deterministic_fallback_used,
                standard_executor_used=deterministic_standard_executor_used,
                runner_repair_name=runner_repair_name,
                resumed_code_reuse=resumed_code_reuse_used,
                concept_repair_used=concept_repair_used,
                llm_repair_used=llm_repair_used,
            )
            if art.name == "step_summary.json":
                summary_authority = "\0".join(
                    (
                        step.step_id,
                        sealed_result_digests.get(
                            art.name,
                            sha256_of_file(art),
                        ),
                        script_record.evidence_id,
                    )
                )
                summary_evidence_id = (
                    "statistic_step_summary_"
                    + hashlib.sha256(summary_authority.encode("utf-8")).hexdigest()[:16]
                )
                rec = evidence.register_file(
                    kind="statistic",
                    description=f"Machine-readable summary for step {step.step_id}.",
                    source_path=art,
                    produced_by_step=step.step_id,
                    inputs=lineage_input_evidence_ids or None,
                    script_evidence_id=script_record.evidence_id,
                    aliases=step_aliases,
                    producer="runner",
                    generation_mode=generation_mode,
                    evidence_id=summary_evidence_id,
                    publish_aliases=False,
                    metadata={
                        "script_evidence_id": script_record.evidence_id,
                        "figure_role": figure_role or "analysis_figure",
                        "diagnostic_only": standard_executor_terminal_block,
                        **repair_evidence_metadata,
                    },
                )
                step_summary_record_id = rec.evidence_id
            elif art.suffix.lower() in {".csv", ".tsv", ".parquet", ".feather"}:
                rec = evidence.register_file(
                    kind="table",
                    description=f"Table {art.stem} from step {step.step_id}.",
                    source_path=art,
                    produced_by_step=step.step_id,
                    inputs=lineage_input_evidence_ids or None,
                    script_evidence_id=script_record.evidence_id,
                    aliases=step_aliases,
                    producer="runner",
                    generation_mode=generation_mode,
                    publish_aliases=False,
                    metadata={
                        "script_evidence_id": script_record.evidence_id,
                        "diagnostic_only": standard_executor_terminal_block,
                        **repair_evidence_metadata,
                    },
                )
            elif art.suffix.lower() in {
                ".png",
                ".svg",
                ".pdf",
                ".tiff",
                ".tif",
                ".pptx",
            }:
                rec = evidence.register_file(
                    kind="figure",
                    description=f"Figure {art.stem} from step {step.step_id}.",
                    source_path=art,
                    produced_by_step=step.step_id,
                    inputs=lineage_input_evidence_ids or None,
                    script_evidence_id=script_record.evidence_id,
                    aliases=step_aliases,
                    producer="runner",
                    generation_mode=generation_mode,
                    publish_aliases=False,
                    metadata={
                        "script_evidence_id": script_record.evidence_id,
                        "figure_role": figure_role or "analysis_figure",
                        "diagnostic_only": standard_executor_terminal_block,
                        **repair_evidence_metadata,
                    },
                )
            else:
                rec = evidence.register_file(
                    kind="log",
                    description=f"Auxiliary artefact {art.name}.",
                    source_path=art,
                    produced_by_step=step.step_id,
                    inputs=lineage_input_evidence_ids or None,
                    script_evidence_id=script_record.evidence_id,
                    aliases=step_aliases,
                    producer="runner",
                    generation_mode=generation_mode,
                    publish_aliases=False,
                    metadata={
                        "script_evidence_id": script_record.evidence_id,
                        "diagnostic_only": standard_executor_terminal_block,
                        **repair_evidence_metadata,
                    },
                )
            pending_success_aliases[rec.evidence_id] = list(step_aliases)
            evidence_ids_for_step.append(rec.evidence_id)

        if step_summary_record_id is not None:
            step_record["step_summary_evidence_id"] = step_summary_record_id

        def _register_current_step_numeric_claims() -> None:
            """Publish numeric authority only after every step gate passes."""

            if (
                not step_summary
                or step_summary_record_id is None
                or standard_executor_terminal_block
            ):
                return
            # Value-level provenance (A-track): every numeric leaf in the
            # step's summary is registered as a NumericClaim so the
            # manuscript binder can reverse-link numbers in prose to the
            # exact field of the exact step output that produced them.
            try:
                cap = pipeline._max_numeric_claims_per_step
                evidence.register_step_summary_numerics(
                    step_id=step.step_id,
                    evidence_id=step_summary_record_id,
                    summary=step_summary,
                    max_leaves=cap if cap > 0 else None,
                )
            except Exception as exc:
                logger.warning(
                    "Failed to register numeric claims for step %s: %s",
                    step.step_id,
                    exc,
                )
            # Phase-1 derived-claim hook (Commit 2). After every leaf
            # is registered, evaluate any ``derived_claims`` the coder
            # declared in step_summary. Sources must resolve to claims
            # that ALREADY exist in the registry, so this runs second.
            # Errors surface as ``derived_claim_error`` findings rather
            # than aborting — a bad formula should not kill the step.
            try:
                _, derived_errors = evidence.register_step_derived_claims(
                    step_id=step.step_id,
                    evidence_id=step_summary_record_id,
                    summary=step_summary,
                )
                for err in derived_errors:
                    findings.append(
                        ValidationFinding(
                            validator="derived_claim",
                            severity="warning",
                            message=(
                                f"derived_claims entry {err['name']!r} for step "
                                f"{step.step_id} was rejected: {err['message']}"
                            ),
                            detail={
                                "step_id": step.step_id,
                                "claim_name": err["name"],
                                "reason": err["message"],
                            },
                        )
                    )
            except Exception as exc:
                logger.warning(
                    "Failed to register derived claims for step %s: %s",
                    step.step_id,
                    exc,
                )

        if standard_executor_terminal_block:
            terminal_summary = (
                step_summary if step_summary else standard_executor_terminal_summary
            )
            terminal_finding = ValidationFinding(
                validator="trajectory_stability_executor",
                severity="error",
                message=(
                    "The planner-specified trajectory stability computation failed "
                    "closed; its diagnostic outputs were preserved and no coder, "
                    "fallback method, seed change, or cluster-count change was used."
                ),
                detail={
                    "step_id": step.step_id,
                    "reason": standard_executor_terminal_reason,
                    "executor_errors": (
                        terminal_summary.get("errors")
                        if isinstance(terminal_summary, Mapping)
                        else None
                    ),
                },
            )
            terminal_findings = [
                terminal_finding,
                *standard_executor_terminal_findings,
            ]
            evidence_ids_for_step = list(dict.fromkeys(evidence_ids_for_step))
            step_record.update(
                {
                    "status": "deterministic_standard_blocked",
                    "diagnostic_only": True,
                    "standard_executor_terminal_reason": (
                        standard_executor_terminal_reason
                    ),
                    "step_summary": terminal_summary,
                    "contract_findings": [
                        finding.model_dump() for finding in terminal_findings
                    ],
                    "evidence_ids": evidence_ids_for_step,
                    "llm_repair_used": False,
                    "generation_mode": _script_generation_mode(
                        repair_attempts=repair_attempts,
                        fallback_used=deterministic_fallback_used,
                        standard_executor_used=(deterministic_standard_executor_used),
                        runner_repair_name=runner_repair_name,
                        resumed_code_reuse=resumed_code_reuse_used,
                        concept_repair_used=concept_repair_used,
                        llm_repair_used=False,
                    ),
                }
            )
            with shared_lock:
                findings.extend(terminal_findings)
                per_step_records.append(step_record)
                _flush_partial_manifest()
            emit_progress(
                "runner",
                f"Trajectory stability failed closed for {step.step_id}.",
                status="error",
                run_id=run_id,
                step_id=step.step_id,
                current_step=step_current,
                total_steps=total_steps,
                reason=standard_executor_terminal_reason,
            )
            return step_record
        with shared_lock:
            completed_records_snapshot = list(per_step_records)
        final_gate_findings = _evaluate_final_deterministic_gates(
            context=context,
            cohort_path=cohort_path,
            universe_path=universe_path,
            run_dir=run_dir,
            out_dir=run_result.out_dir,
            step=step,
            step_summary=step_summary,
            step_record=step_record,
            completed_step_records=completed_records_snapshot,
            resolved_input_bindings=resolved_input_bindings,
            attempt_id=attempt_id,
            checkpoint_id=review_checkpoint_id,
            stat_validator=stat_validator,
            clinical_validator=clinical_validator,
            statistical_guard=statistical_guard,
            cross_step_cohort_lock_validator=cross_step_cohort_lock_validator,
            cross_step_registered_output_validator=(
                cross_step_registered_output_validator
            ),
            cross_step_reconciliation_trace_validator=(
                cross_step_reconciliation_trace_validator
            ),
            step_summary_integrity_validator=step_summary_integrity_validator,
            step_summary_fraction_validator=step_summary_fraction_validator,
            cross_step_source_status_validator=cross_step_source_status_validator,
            primary_model_contract_validator=primary_model_contract_validator,
            figure_contract_validator=figure_contract_validator,
            figure_source_validator=figure_source_validator,
        )
        stat_findings = list(final_gate_findings.stat_findings)
        clinical_findings = list(final_gate_findings.clinical_findings)
        guard_findings = list(final_gate_findings.guard_findings)
        contract_findings = list(final_gate_findings.contract_findings)
        figure_source_findings = list(final_gate_findings.figure_source_findings)
        with shared_lock:
            findings.extend(stat_findings)
            findings.extend(clinical_findings)
            findings.extend(guard_findings)
            findings.extend(contract_findings)
            findings.extend(figure_source_findings)
        step_record["stat_findings"] = [f.model_dump() for f in stat_findings]
        step_record["clinical_findings"] = [f.model_dump() for f in clinical_findings]
        step_record["guard_findings"] = [f.model_dump() for f in guard_findings]
        step_record["contract_findings"] = [f.model_dump() for f in contract_findings]
        step_record["figure_source_findings"] = [
            f.model_dump() for f in figure_source_findings
        ]
        step_record["llm_repair_used"] = llm_repair_used
        step_record["generation_mode"] = _script_generation_mode(
            repair_attempts=repair_attempts,
            fallback_used=deterministic_fallback_used,
            standard_executor_used=deterministic_standard_executor_used,
            runner_repair_name=runner_repair_name,
            resumed_code_reuse=resumed_code_reuse_used,
            concept_repair_used=concept_repair_used,
            llm_repair_used=llm_repair_used,
        )
        raw_side_findings = step_summary.get("side_findings")
        if isinstance(raw_side_findings, list):
            side_findings = []
            for idx, raw in enumerate(raw_side_findings):
                if not isinstance(raw, dict):
                    continue
                payload = dict(raw)
                payload.setdefault("step_id", step.step_id)
                payload.setdefault("finding_id", f"{step.step_id}_side_{idx + 1}")
                side_findings.append(SideFinding.from_dict(payload).to_dict())
            if side_findings:
                step_record["side_findings"] = side_findings
        step_record["step_summary"] = step_summary
        evidence_refs_for_step, _, _ = _evidence_refs_for_names(evidence_ids_for_step)
        validator_messages = _validator_messages(
            usage_findings,
            stat_findings,
            clinical_findings,
            guard_findings,
            contract_findings,
            figure_source_findings,
        )
        local_runtime_state = supervisor.critique_step(
            state=local_runtime_state,
            step_summary=step_summary,
            evidence_refs=evidence_refs_for_step,
            findings=validator_messages,
        )
        critique = local_runtime_state.critique
        critique_findings: List[ValidationFinding] = []
        if critique is not None:
            critique_path = run_result.out_dir / "critique_report.json"
            critique_path.write_text(
                critique.model_dump_json(indent=2),
                encoding="utf-8",
            )
            critique_record = evidence.register_file(
                kind="log",
                description=f"Structured critique report for step {step.step_id}.",
                source_path=critique_path,
                produced_by_step=step.step_id,
                script_evidence_id=script_record.evidence_id,
                aliases=[f"{step.step_id}_critique"],
                producer="critic",
                generation_mode="system",
                publish_aliases=False,
                metadata={"script_evidence_id": script_record.evidence_id},
            )
            pending_success_aliases[critique_record.evidence_id] = [
                f"{step.step_id}_critique"
            ]
            evidence_ids_for_step.append(critique_record.evidence_id)
            step_record["critique_report"] = critique.model_dump(mode="json")
            if critique.status in {"needs_revision", "blocked"}:
                critique_finding = ValidationFinding(
                    validator="critic_agent",
                    severity=(
                        "warning" if critique.status == "needs_revision" else "error"
                    ),
                    message=(
                        f"CriticAgent marked {step.step_id} as {critique.status}: "
                        + "; ".join(
                            critique.concerns
                            or critique.suggested_repairs
                            or ["review required"]
                        )
                    ),
                    evidence_ids=[critique_record.evidence_id],
                    detail={
                        "step_id": step.step_id,
                        "critic_status": critique.status,
                    },
                )
                critique_findings.append(critique_finding)
                with shared_lock:
                    findings.append(critique_finding)

        step_record["evidence_ids"] = list(dict.fromkeys(evidence_ids_for_step))
        checkpoint_record = dict(step_record)
        checkpoint_record["status"] = "executed_pending_review"
        checkpoint_record["review_pending"] = True
        with shared_lock:
            upsert_step_record(
                per_step_records,
                checkpoint_record,
                replace_statuses={"executed_pending_review"},
            )
            _flush_partial_manifest()

        interp_generation_mode = "llm"
        final_generation_mode = str(step_record.get("generation_mode") or "")
        if final_generation_mode in {"resumed_code_reuse", "fallback"}:
            mode_label = (
                "resumed agent-generated code"
                if final_generation_mode == "resumed_code_reuse"
                else "deterministic fallback code"
            )
            interpretation = (
                f"Step `{step.step_id}` was executed from {mode_label}. "
                "Review the registered step summary and artefacts for numeric "
                "interpretation; no new LLM interpretation was requested."
            )
            interp_generation_mode = (
                "resumed_code_reuse"
                if final_generation_mode == "resumed_code_reuse"
                else "deterministic_fallback"
            )
        else:
            try:
                interpretation = analyzer.run(
                    context=agent_context,
                    step=step,
                    step_summary=step_summary,
                    evidence_ids=evidence_ids_for_step,
                    provider_budget=provider_budget,
                )
            except Exception as exc:
                interpretation = f"(analyzer failed: {exc})"
                interp_generation_mode = "system"
        _sync_provider_budget()
        # Content-addressing alone is insufficient for step-owned evidence:
        # two steps may legitimately receive identical analyzer text.  Bind
        # the identity to the producing step and exact script so a later
        # resume never reuses another step's first-written authority record.
        interpretation_authority = "\0".join(
            (step.step_id, script_record.evidence_id, interpretation)
        )
        interpretation_evidence_id = (
            "log_interpretation_"
            + hashlib.sha256(interpretation_authority.encode("utf-8")).hexdigest()[:16]
        )
        interp_record = evidence.register_text(
            kind="log",
            description=f"Analyzer interpretation for step {step.step_id}.",
            text=interpretation,
            filename=f"interpretation_{step.step_id}.md",
            produced_by_step=step.step_id,
            script_evidence_id=script_record.evidence_id,
            evidence_id=interpretation_evidence_id,
            producer="analyzer",
            generation_mode=interp_generation_mode,
            prompt_pack_version=prompt_version,
            publish_aliases=False,
        )
        pending_success_aliases[interp_record.evidence_id] = [
            f"interpretation_{step.step_id}"
        ]
        step_record["interpretation_evidence_id"] = interp_record.evidence_id
        evidence_ids_for_step.append(interp_record.evidence_id)
        step_record["evidence_ids"] = list(dict.fromkeys(evidence_ids_for_step))
        step_record.pop("review_pending", None)
        mutated_after_seal = [
            name
            for name, expected_digest in sealed_result_digests.items()
            if not (candidate := run_result.out_dir / name).is_file()
            or sha256_of_file(candidate) != expected_digest
        ]
        if mutated_after_seal:
            seal_finding = ValidationFinding(
                validator="result_evidence_seal",
                severity="error",
                message=(
                    f"Result artifacts for step {step.step_id} changed after the "
                    "validate-and-seal boundary; registered evidence was retired."
                ),
                detail={
                    "step_id": step.step_id,
                    "attempt_id": attempt_id,
                    "checkpoint_id": review_checkpoint_id,
                    "mutated_artifacts": sorted(mutated_after_seal),
                },
            )
            contract_findings.append(seal_finding)
            step_record["contract_findings"] = [
                finding.model_dump() for finding in contract_findings
            ]
            step_record["result_evidence_sealed"] = False
            with shared_lock:
                findings.append(seal_finding)
        _propagate_findings_to_evidence(
            evidence_ids_for_step,
            usage_findings
            + stat_findings
            + clinical_findings
            + guard_findings
            + contract_findings
            + figure_source_findings
            + critique_findings,
            metadata={
                "step_id": step.step_id,
                "generation_mode": step_record["generation_mode"],
            },
        )
        with shared_lock:
            runtime_state = local_runtime_state
        step_record["status"] = _step_status_from_contract_findings(
            contract_findings=contract_findings,
            figure_source_findings=figure_source_findings,
            stat_findings=stat_findings,
            critique_status=(critique.status if critique is not None else None),
        )
        has_contract_error = step_record["status"] == "contract_failed"
        final_cleanup_finding: Optional[ValidationFinding] = None
        if step_record["status"] == "ok":
            step_record.pop("monotonic_concept_constraints", None)
            try:
                clear_quarantined_concept_draft(
                    run_dir=run_dir,
                    step_id=step.step_id,
                )
                if resumed_quarantined_draft_used:
                    step_record["quarantined_requires_repair"] = False
                    step_record["quarantine_retired"] = True
                    if quarantine_superseded_by_fallback:
                        step_record["quarantine_retired_by"] = (
                            "successful_deterministic_fallback"
                        )
            except ValueError as exc:
                step_record["status"] = "blocked_quarantine_cleanup"
                final_cleanup_finding = ValidationFinding(
                    validator="resume",
                    severity="error",
                    message=(
                        "Successful step output could not retire its stale "
                        f"quarantine safely for step {step.step_id}: {exc}"
                    ),
                    detail={"step_id": step.step_id},
                )
        alias_publication_finding: Optional[ValidationFinding] = None
        if step_record["status"] == "ok":
            try:
                current_evidence_records = evidence.records()
                (
                    success_alias_bindings,
                    retained_cross_step_aliases,
                    suppressed_basename_evidence_ids,
                ) = _filter_success_alias_bindings(
                    pending_success_aliases,
                    existing_aliases=evidence.aliases(),
                    owners_by_evidence_id={
                        record.evidence_id: str(record.produced_by_step or "").strip()
                        for record in current_evidence_records
                    },
                    step_id=step.step_id,
                    records_by_evidence_id={
                        record.evidence_id: record
                        for record in current_evidence_records
                    },
                )
                evidence.publish_step_success_aliases(
                    success_alias_bindings,
                    step_id=step.step_id,
                    suppressed_basename_evidence_ids=(suppressed_basename_evidence_ids),
                )
                if retained_cross_step_aliases:
                    step_record["retained_cross_step_aliases"] = (
                        retained_cross_step_aliases
                    )
            except (KeyError, ValueError, OSError) as exc:
                step_record["status"] = "contract_failed"
                alias_publication_finding = ValidationFinding(
                    validator="result_evidence_authority",
                    severity="error",
                    message=(
                        "Validated result evidence could not be promoted to "
                        f"current authority for step {step.step_id}."
                    ),
                    detail={
                        "step_id": step.step_id,
                        "attempt_id": attempt_id,
                        "checkpoint_id": review_checkpoint_id,
                        "reason": str(exc),
                    },
                )
                contract_findings.append(alias_publication_finding)
                step_record["contract_findings"] = [
                    finding.model_dump() for finding in contract_findings
                ]
                _propagate_findings_to_evidence(
                    evidence_ids_for_step,
                    [alias_publication_finding],
                    metadata={
                        "step_id": step.step_id,
                        "generation_mode": step_record["generation_mode"],
                    },
                )
                has_contract_error = True
        if step_record["status"] == "ok":
            # This stamp is written only after deterministic artifact gates and
            # Critic review pass and current evidence aliases are published.
            step_record.update(_deterministic_gate_stamp())
            _register_current_step_numeric_claims()
        with shared_lock:
            if final_cleanup_finding is not None:
                findings.append(final_cleanup_finding)
            if alias_publication_finding is not None:
                findings.append(alias_publication_finding)
            upsert_step_record(
                per_step_records,
                step_record,
                replace_statuses={"executed_pending_review"},
            )
            _flush_partial_manifest()
        emit_progress(
            "step",
            (
                f"Step {step_current}/{total_steps} failed contract checks: "
                f"{step.step_id}."
                if has_contract_error
                else (
                    f"Step {step_current}/{total_steps} failed Critic review: "
                    f"{step.step_id}."
                    if step_record["status"] == "critic_failed"
                    else (
                        f"Step {step_current}/{total_steps} could not retire its "
                        f"quarantine: {step.step_id}."
                        if step_record["status"] == "blocked_quarantine_cleanup"
                        else f"Step {step_current}/{total_steps} complete: {step.step_id}."
                    )
                )
            ),
            status=("complete" if step_record["status"] == "ok" else "error"),
            run_id=run_id,
            step_id=step.step_id,
            current_step=step_current,
            total_steps=total_steps,
        )
        return step_record

    steps_to_run = (
        []
        if trajectory_plan_blocked or typed_plan_dag_blocked
        else resume_controller.remaining_steps(
            plan=plan,
            executed_step_ids=set(preexecuted_step_ids),
        )
    )
    has_typed_input_dependencies = any(
        _typed_input_product(input_name) is not None
        for step in steps_to_run
        for input_name in (step.inputs or [])
    )
    for skipped_step_id in sorted(resumed_step_ids):
        emit_progress(
            "resume",
            f"Skipped completed step from prior run: {skipped_step_id}.",
            status="complete",
            run_id=run_id,
            step_id=skipped_step_id,
        )
    for skipped_step_id in sorted(preexecuted_step_ids - resumed_step_ids):
        emit_progress(
            "step",
            f"Skipped step already completed by pre-execution: {skipped_step_id}.",
            status="complete",
            run_id=run_id,
            step_id=skipped_step_id,
        )
    if pipeline._enable_replanning and pipeline._max_concurrent_steps > 1:
        findings.append(
            ValidationFinding(
                validator="replanner",
                severity="info",
                message=(
                    "Replanning is enabled, so step execution was forced to sequential "
                    "mode to preserve run-internal plan revisions."
                ),
            )
        )
    elif has_typed_input_dependencies and pipeline._max_concurrent_steps > 1:
        findings.append(
            ValidationFinding(
                validator="typed_artifact_evidence_lineage",
                severity="info",
                message=(
                    "Typed product dependencies are present, so step execution "
                    "was forced to plan order before resolving producer evidence."
                ),
            )
        )

    if (
        pipeline._max_concurrent_steps <= 1
        or len(steps_to_run) <= 1
        or pipeline._enable_replanning
        or has_typed_input_dependencies
        or requested_stop_after_step_id is not None
    ):

        def _maybe_directed_model_replan(
            *,
            failed_step: AnalysisStep,
            failed_record: Dict[str, Any],
        ) -> Optional[AnalysisPlan]:
            """Fire a forced, directive-carrying replan when a model/estimation
            step self-blocks on a task-viable cohort, else return ``None``.

            This is the active half of the self-inflicted-block fix: the
            post-hoc scorecard only *labels* the self-paralysis, whereas here we
            give the replanner a viability-conditioned override so a populated
            cohort is not silently abandoned with a non-execution stub. Bounded
            by ``_MAX_DIRECTED_MODEL_REPLANS``; conservative — silent on a hard
            crash, an unreadable cohort, or genuinely non-viable data.
            """
            if not pipeline._enable_replanning:
                return None
            if failed_record.get("status") == "ok":
                return None
            if _replan_state["directed_model_replans"] >= _MAX_DIRECTED_MODEL_REPLANS:
                return None
            if not step_requires_model_performance(failed_step.expected_outputs):
                return None
            try:
                import pandas as pd  # lazy: only on the rare self-block path

                viability = assess_cohort_viability(
                    pd.read_parquet(cohort_path), outcome=None
                )
            except Exception:
                return None
            directive = build_self_block_replan_directive(
                failed_step=failed_step,
                failed_record=failed_record,
                completed_records=per_step_records,
                viability=viability,
            )
            if directive is None:
                return None
            _replan_state["directed_model_replans"] += 1
            findings.append(
                ValidationFinding(
                    validator="replanner",
                    severity="warning",
                    message=(
                        "Directed replan: modeling step "
                        f"{failed_step.step_id} self-blocked on a task-viable "
                        f"cohort ({viability.note}); issued a viability-conditioned "
                        "override to fit the model rather than register a block."
                    ),
                    detail={
                        "step_id": failed_step.step_id,
                        "directed_model_replans": _replan_state[
                            "directed_model_replans"
                        ],
                    },
                )
            )
            return _maybe_replan(
                current_plan=plan,
                reason=f"{failed_step.step_id}:self_inflicted_block_on_viable_cohort",
                probe_summary_payload=probe_summary,
                completed_records=per_step_records,
                directive=directive,
                force=True,
            )

        executed_step_ids = set(preexecuted_step_ids)
        remaining_steps = resume_controller.remaining_steps(
            plan=plan,
            executed_step_ids=executed_step_ids,
        )
        while remaining_steps:
            step = remaining_steps.pop(0)
            record = _execute_one_step(step)
            executed_step_ids.add(step.step_id)
            if step.step_id == requested_stop_after_step_id:
                emit_progress(
                    "pause",
                    f"Stopped after requested step: {step.step_id}.",
                    status="paused",
                    run_id=run_id,
                    step_id=step.step_id,
                )
                break
            directed_plan = _maybe_directed_model_replan(
                failed_step=step, failed_record=record
            )
            if directed_plan is not None:
                plan = directed_plan
                # Re-run the modeling step against the revised, de-gated plan.
                executed_step_ids.discard(step.step_id)
                step_order.clear()
                step_order.update({s.step_id: i for i, s in enumerate(plan.steps)})
                remaining_steps = resume_controller.remaining_steps(
                    plan=plan,
                    executed_step_ids=executed_step_ids,
                )
                total_steps = len(plan.steps)
                continue
            if (
                pipeline._enable_replanning
                and record.get("status") == "ok"
                and _successful_step_requests_replan(record)
                and remaining_steps
            ):
                plan = _maybe_replan(
                    current_plan=plan,
                    reason=step.step_id,
                    probe_summary_payload=probe_summary,
                    completed_records=per_step_records,
                )
                step_order.clear()
                step_order.update({s.step_id: i for i, s in enumerate(plan.steps)})
                remaining_steps = resume_controller.remaining_steps(
                    plan=plan,
                    executed_step_ids=executed_step_ids,
                )
                total_steps = len(plan.steps)
    else:
        workers = min(pipeline._max_concurrent_steps, len(steps_to_run))
        with ThreadPoolExecutor(
            max_workers=workers, thread_name_prefix="ra_step"
        ) as ex:
            futures = [ex.submit(_execute_one_step, s) for s in steps_to_run]
            for fut in as_completed(futures):
                exc = fut.exception()
                if exc is not None:
                    with shared_lock:
                        findings.append(
                            ValidationFinding(
                                validator="step_executor",
                                severity="error",
                                message=f"Worker raised an unhandled exception: {exc!r}",
                            )
                        )

    if (
        not trajectory_plan_blocked
        and not typed_plan_dag_blocked
        and trajectory_plan_contract_applies(plan=plan, context=context)
    ):
        run_level_trajectory_findings = trajectory_bundle_findings(
            context=context,
            plan=plan,
            per_step_records=per_step_records,
            evidence=evidence,
            run_dir=run_dir,
            cohort_path=cohort_path,
        )
        findings.extend(run_level_trajectory_findings)
        _flush_partial_manifest(
            {
                "trajectory_bundle_error_count": sum(
                    finding.severity == "error"
                    for finding in run_level_trajectory_findings
                )
            }
        )

    try:
        robustness_specs = robustness_specs_for_execution(run_dir=run_dir, plan=plan)
        if robustness_specs and not list(getattr(plan, "robustness_specs", []) or []):
            findings.append(
                ValidationFinding(
                    validator="robustness_panel",
                    severity="warning",
                    message=(
                        "Recovered robustness_specs from the plan-time lock because "
                        "the active replanned AnalysisPlan no longer carried them."
                    ),
                )
            )
        adapter_rows, adapter_warnings = fit_robustness_rows_from_records(
            specs=robustness_specs,
            per_step_records=per_step_records,
            primary_cohort=getattr(plan, "cohort", None),
            cohort_path=cohort_path,
            context=context,
            run_dir=run_dir,
            allow_implicit_cohort_refit=False,
        )
        for warning in adapter_warnings:
            findings.append(
                ValidationFinding(
                    validator="robustness_estimator",
                    severity="warning",
                    message=warning,
                )
            )
        robustness_panel = build_robustness_panel_from_records(
            specs=robustness_specs,
            per_step_records=per_step_records,
            adapter_rows=adapter_rows,
        )
        write_robustness_panel(
            run_dir=run_dir,
            panel=robustness_panel,
            evidence=evidence,
            prompt_pack_version=prompt_version,
        )
        _flush_partial_manifest(
            {
                "robustness_panel_path": "robustness_panel.json",
                "robustness_n_variants": robustness_panel.n_variants,
                "robustness_range_low": robustness_panel.range_low,
                "robustness_range_high": robustness_panel.range_high,
            }
        )
    except Exception as exc:
        findings.append(
            ValidationFinding(
                validator="robustness_panel",
                severity="warning",
                message=f"Robustness panel artifact could not be built: {exc}",
            )
        )

    if pipeline._enable_visual_qa:
        emit_progress(
            "visual_qa",
            "Auditing generated figures.",
            run_id=run_id,
        )
        fig_paths = [
            run_dir / r.relative_path for r in evidence.records() if r.kind == "figure"
        ]
        vlm_adapter = pipeline._visual_qa_adapter
        if vlm_adapter is None and pipeline._enable_vlm_visual_qa:
            client = pipeline._vlm_client or role_resolver("analyzer")
            if client is not None:
                vlm_adapter = VLMVisualQAAdapter(client)
        final_visual_findings = VisualQAAuditor(vlm_adapter=vlm_adapter).audit(
            figure_paths=fig_paths
        )
        demoted_final_findings, _ = _demote_cosmetic_visual_findings(
            final_visual_findings
        )
        findings += demoted_final_findings

    try:
        article_contract_status = summarize_article_contract_coverage(
            context=context,
            plan=plan,
            evidence_records=evidence.records(),
            per_step_records=per_step_records,
            run_dir=run_dir,
        )
        article_contract_path = run_dir / "article_contract_audit.json"
        article_contract_path.write_text(
            json.dumps(
                article_contract_audit_payload(article_contract_status),
                indent=2,
                ensure_ascii=False,
                default=str,
            ),
            encoding="utf-8",
        )
        if evidence.get("article_contract_audit") is None:
            evidence.register_file(
                kind="log",
                description=(
                    "Run-level article analysis contract audit: compares "
                    "registered artifacts against required article display roles."
                ),
                source_path=article_contract_path,
                evidence_id="article_contract_audit",
                producer="article_contract",
                generation_mode="system",
            )
        findings.extend(
            validate_run_against_article_contract(
                context=context,
                plan=plan,
                evidence_records=evidence.records(),
                per_step_records=per_step_records,
                run_dir=run_dir,
            )
        )
        _flush_partial_manifest(
            {"article_contract_audit": str(article_contract_path.relative_to(run_dir))}
        )
    except Exception as exc:
        findings.append(
            ValidationFinding(
                validator="article_analysis_contract",
                severity="warning",
                message=(
                    "Run-level article analysis contract audit failed: "
                    f"{type(exc).__name__}: {exc}"
                ),
            )
        )

    plan_result.plan = plan
    plan_result.plan_path = plan_path
    return _ExecutePhaseResult(
        plan=plan,
        per_step_records=per_step_records,
        probe_summary=probe_summary,
        runtime_state=runtime_state,
        flush_partial_manifest=_flush_partial_manifest,
    )
