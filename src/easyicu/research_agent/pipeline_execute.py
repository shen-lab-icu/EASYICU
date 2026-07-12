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
import json
import logging
import os
import re
import shutil
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
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
    _downgrade_metadata_supported_outcome_findings,
)
from .code_repair import (
    _deterministic_runner_repair,
    _deterministic_summary_repair,
    deterministic_contract_repair,
    deterministic_concept_audit_repair,
)
from .code_hygiene import reorder_forward_references
from .cohort_repair import extract_cohort_definition_from_prose
from .cohort_schema import (
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
from .estimators import fit_robustness_rows_from_records
from .evidence import sha256_of_bytes, sha256_of_file
from .llm import MockLLMClient
from .ordered_stratified_contract import ordered_stratified_numeric_findings
from .pipeline import (
    _build_probe_summary,
    _clear_output_dir,
    deterministic_figure_family_supported_for_upstream,
    deterministic_figure_repair_id_for_upstream,
    _has_figure_exports,
    _promote_prior_publication_bundle,
    _promote_sibling_figure_exports,
    _render_publication_bundle_from_prior_outputs_for_step,
    _semantic_aliases_for,
)
from .publication_figures import make_figure_contract
from .plan_utils import (
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
from .repair_registry import (
    InvariantStatus,
    RepairLedger,
    RepairObservedState,
    automatic_repair_allowed,
)
from .runtime_artifacts import current_step_records, current_successful_step_records
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

    if hashlib.sha256(before.encode("utf-8")).digest() == hashlib.sha256(
        after.encode("utf-8")
    ).digest():
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
    reclassified = _downgrade_metadata_supported_outcome_findings(
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
            and all(current_detail.get(key) == value for key, value in prior_detail.items())
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
        if repair_id is None or not _has_figure_exports(staging_dir):
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


def _actionable_validator_messages(
    *finding_groups: Sequence[ValidationFinding],
) -> List[str]:
    """Return only warning/error messages that require Critic action.

    Informational audit records remain in the manifest and global findings,
    but must not turn an otherwise clean deterministic step into
    ``needs_revision`` merely because the Critic receives a non-empty string.
    """

    return [
        finding.message
        for group in finding_groups
        for finding in group
        if finding.severity in {"warning", "error"} and finding.message
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
    preserved = revised_plan.model_copy(
        update={"robustness_specs": list(locked_specs)}
    )
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
) -> str:
    """Map deterministic contract errors to the outer step status."""

    has_contract_error = any(
        finding.severity == "error"
        for finding in (
            list(contract_findings)
            + list(figure_source_findings)
            + [
                finding
                for finding in stat_findings
                if finding.validator == "ordered_stratified_contract"
            ]
        )
    )
    return "contract_failed" if has_contract_error else "ok"


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


def _is_cohort_definition_sensitivity_result_step(step: AnalysisStep) -> bool:
    return (
        _method_head(str(step.method or "")) == "cohort_definition_sensitivity"
        and not _step_expects_figure(step)
    )


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
    locked_contract = [
        {field: spec.get(field) for field in fields}
        for spec in specs
    ]
    attachment = (
        "LOCKED ROBUSTNESS SPECIFICATIONS (binding plan-time state):\n"
        + json.dumps(locked_contract, ensure_ascii=False, separators=(",", ":"))
        + "\nExecute every spec_id exactly as declared; do not rename, replace, "
        "or invent specifications. Cohort-axis definitions that can recover "
        "rows outside the locked analysis cohort must be materialised from "
        "os.environ['EASYICU_UNIVERSE_PARQUET']; COHORT_PARQUET is the locked "
        "analysis cohort."
        "\n\n"
        + ROBUSTNESS_EXECUTION_CONTRACT_GUIDANCE
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
        "table:sensitivity_specification_matrix",
        "table:robustness_summary",
        "table:robustness_matrix",
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
        step_summary.get("outputs") if isinstance(step_summary.get("outputs"), list) else [],
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
                    if not str(row.get("spec_id") or "").strip() and str(
                        row.get("definition_id") or ""
                    ).strip():
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
        _sensitivity_csv_rows(
            list(dict.fromkeys([*spec_paths, *denominator_paths]))
        )
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
                        "cohort_spec_ids": sorted(spec.spec_id for spec in cohort_specs),
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
    return all(
        _output_declares_figure(str(output)) for output in expected_outputs
    )


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
        target_status = str(
            (target_record or {}).get("status") or ""
        ).strip().lower()
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
        if source_record is None or str(
            source_record.get("status") or ""
        ).strip().lower() != "ok":
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


if TYPE_CHECKING:
    from .pipeline import ResearchAgentPipeline


def _is_cosmetic_visual_finding(finding: ValidationFinding) -> bool:
    """Return true only for deterministic layout errors safe to demote."""

    if finding.severity != "error" or finding.validator != "visual_qa":
        return False
    message = (finding.message or "").lower()
    return "overlapping text elements" in message and "spacing" in message


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


def _plan_signature(
    plan: AnalysisPlan,
) -> Tuple[Tuple[Any, ...], ...]:
    """Substantive fingerprint of a plan's step DAG, ignoring ordinary prose.

    Two plans with the same step DAG are usually analytically identical even if
    the replanner reworded each step's ``intent``. Structured model requirements
    and primary/secondary/sensitivity/corroborative role markers are exceptions:
    changing either changes the estimand hierarchy and must not be suppressed as
    a no-op revision.
    """
    return tuple(
        (
            step.step_id,
            step.method,
            tuple(step.expected_outputs),
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
        )
        for step in plan.steps
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

    normalized = re.sub(
        r"[^a-z0-9]+", "_", str(method or "").strip().lower()
    ).strip("_")
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
    return head in _ORDINAL_EXPLICIT_METHODS or _method_has_ordinal_primary_token(
        head
    )


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
        # Directed replans fired when a model/estimation step self-blocks on a
        # task-viable cohort (see _maybe_directed_model_replan). Bounded so a
        # run that keeps self-blocking falls back to an honest diagnostic_only
        # rather than looping the replanner indefinitely.
        "directed_model_replans": 0,
    }
    role_resolver = plan_result.role_resolver
    llm_signature = plan_result.llm_signature
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
    from .audits.patterns import AnalysisPatternAuditor

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
    primary_model_contract_validator = PrimaryModelContractValidator()
    statistical_guard = StatisticalGuard()
    runtime_state = supervisor.bootstrap_state(run_id=run_id, context=context)
    repair_ledger = RepairLedger(run_dir / "repairs_applied.json")
    repair_ledger_lock = threading.Lock()

    per_step_records: List[Dict[str, Any]] = []
    probe_summary: Dict[str, Any] = {}
    resumed_step_ids: set = set()
    if plan_result.resume_state is not None:
        resume_application = resume_controller.apply()
        per_step_records.extend(resume_application.per_step_records)
        resumed_step_ids = set(resume_application.resumed_step_ids)
        findings.extend(resume_application.findings)
        probe_summary = resume_application.probe_summary
        if resumed_step_ids:
            print(
                f"[research_agent] resume: skipping {len(resumed_step_ids)} "
                f"already-completed step(s) — {sorted(resumed_step_ids)}"
            )

    def _flush_partial_manifest(extra: Optional[Dict[str, Any]] = None) -> None:
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
            "llm_signature": llm_signature,
            "used_mock_llm": plan_result.used_mock_llm,
            "prompt_pack_version": prompt_version,
            "prompt_pack_files": prompt_files,
            "notes": notes,
            "runtime_state": runtime_state.model_dump(mode="json"),
            "repair_ledger_path": str(repair_ledger.path.relative_to(run_dir)),
            "repairs_applied": [record.__dict__ for record in repair_ledger.records],
        }
        if extra:
            payload.update(extra)
        (run_dir / "manifest_partial.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )

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
        definition = extract_cohort_definition_from_prose(
            cohort_prose=_cohort_definition_prose(candidate_plan),
            universe_columns=columns,
            llm=role_resolver("planner"),
            name=getattr(getattr(candidate_plan, "cohort", None), "name", "primary")
            or "primary",
        )
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
        try:
            evidence.register_file(
                kind="table",
                description=(
                    "Analysis cohort materialised from the agent's prose 纳排, "
                    "translated to typed CTAS predicates during execution."
                ),
                source_path=cohort_path,
                evidence_id="analysis_cohort_execute_repair",
                producer="cohort_repair",
                generation_mode="llm",
                prompt_pack_version=prompt_version,
                metadata={"llm_signature": llm_signature, "reason": reason},
            )
        except ValueError:
            pass
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
        cap = pipeline._max_total_steps
        if cap > 0 and len(revised.steps) > cap:
            protected_step_ids = [
                str(record.get("step_id"))
                for record in current_successful_step_records(
                    completed_records or []
                )
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
            if not cap_findings:
                dropped = [s.step_id for s in revised.steps[cap:]]
                revised = revised.model_copy(
                    update={"steps": list(revised.steps[:cap])}
                )
                findings.append(
                    ValidationFinding(
                        validator="replanner",
                        severity="warning",
                        message=(
                            f"Replanner produced {len(dropped) + cap} steps; "
                            f"truncated to max_total_steps={cap}. Dropped: "
                            f"{', '.join(dropped[:6])}"
                            + (" ..." if len(dropped) > 6 else "")
                        ),
                        detail={"dropped_step_ids": dropped, "cap": cap},
                    )
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

        # No-op detection on the *substantive* step DAG, not the full
        # model_dump. A verbose replanner can rewrite each step's ``intent``
        # prose without changing the analysis; that must not count as a
        # revision or burn the convergence budget.
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

    probe_step_id = "00_probe"
    if pipeline._enable_probe_step and probe_step_id not in resumed_step_ids:
        probe_summary, probe_files = _build_probe_summary(
            context=context,
            cohort_path=cohort_path,
            out_dir=run_dir / "steps" / probe_step_id / "outputs",
        )
        probe_evidence_ids: List[str] = []
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
        probe_record = {
            "step_id": probe_step_id,
            "intent": "Probe distributions, missingness, and obvious anomalies before execution.",
            "status": "ok",
            "generation_mode": "deterministic_probe",
            "step_summary": probe_summary,
            "evidence_ids": probe_evidence_ids,
        }
        per_step_records.append(probe_record)
        _flush_partial_manifest()
        plan = _maybe_replan(
            current_plan=plan,
            reason="probe_summary",
            probe_summary_payload=probe_summary,
            completed_records=[probe_record],
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
    ) -> bool:
        """Apply the central no-auto-method-substitution policy.

        Code rewrites and artifact/rendering transforms share this boundary. A
        staged figure may be built speculatively, but it is not installed into
        the live step unless this policy authorizes its typed repair id.
        """

        step_id = str(step.step_id)
        if automatic_repair_allowed(repair_id, step=step):
            return True
        _record_repair(
            repair_id=repair_id,
            step_id=step_id,
            trigger={
                "source": source,
                "automatic_repair_policy": "method_substitution_default_deny",
            },
            transformation=(
                "Candidate repair was not applied because automatic method "
                "substitution is forbidden."
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
                    f"Blocked automatic method-substitution repair {repair_id} "
                    f"for step {step_id}; agent repair or fail-closed handling "
                    "retains ownership."
                ),
                detail={
                    "repair_id": repair_id,
                    "step_id": step_id,
                    "source": source,
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
        ):
            return None
        return repair

    def _script_generation_mode(
        *,
        repair_attempts: int,
        fallback_used: bool,
        runner_repair_name: Optional[str] = None,
        resumed_code_reuse: bool = False,
        concept_repair_used: bool = False,
        llm_repair_used: bool = False,
    ) -> str:
        if fallback_used:
            return "fallback"
        # Report the code that actually executed, not merely where its first
        # draft came from. A resumed script that required a fresh LLM repair is
        # repaired code; labelling it as pure reuse hides the model mutation and
        # can incorrectly trigger reuse-only audit shortcuts.
        if llm_repair_used:
            return "repaired"
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

    def _evidence_refs_for_names(names: Sequence[str]) -> List[EvidenceRef]:
        refs: List[EvidenceRef] = []
        seen: set[str] = set()
        for name in names:
            rec = evidence.get(str(name))
            if rec is None or rec.evidence_id in seen:
                continue
            refs.append(
                EvidenceRef(
                    evidence_id=rec.evidence_id,
                    kind=rec.kind,
                    description=rec.description,
                    relative_path=rec.relative_path,
                )
            )
            seen.add(rec.evidence_id)
        return refs

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
        step_record: Dict[str, Any] = {
            "step_id": step.step_id,
            "intent": step.intent,
        }
        coder_context = _coder_context_with_locked_robustness_specs(
            context=agent_context,
            step=step,
            run_dir=run_dir,
        )
        resumed_code_reuse_used = False
        resumed_quarantined_draft_used = False
        quarantined_draft_active = False
        quarantined_repair_materially_changed = False
        quarantined_repair_succeeded = False
        quarantine_superseded_by_fallback = False
        quarantine_policy_superseded = False
        pending_quarantined_errors: List[ValidationFinding] = []
        preexecution_runner_repair_name: Optional[str] = None
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
        existing_refs = _evidence_refs_for_names(step.inputs)
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

        def _use_quarantined_draft(draft: QuarantinedConceptDraft) -> str:
            nonlocal resumed_quarantined_draft_used
            nonlocal quarantined_draft_active
            nonlocal pending_quarantined_errors
            resumed_quarantined_draft_used = True
            quarantined_draft_active = True
            pending_quarantined_errors = [
                ValidationFinding.model_validate(payload)
                for payload in draft.findings
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
            step_record["resumed_from_generation_mode"] = (
                resumed_from_generation_mode
            )
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
            candidate_code = """
import json
import os
from pathlib import Path

out_dir = Path(os.environ["STEP_OUT_DIR"])
run_dir = out_dir.parents[2]
current_step_id = out_dir.parent.name

from easyicu.research_agent.pipeline import (
    _render_publication_bundle_from_prior_outputs_for_step,
)

repair_id = _render_publication_bundle_from_prior_outputs_for_step(
    run_dir=run_dir,
    current_step_id=current_step_id,
    out_dir=out_dir,
)

expected_repair_id = __EXPECTED_REPAIR_ID__
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
    print(json.dumps({"deterministic_publication_figure_rescue": repair_id}))
"""
            candidate_code = candidate_code.replace(
                "__EXPECTED_REPAIR_ID__", repr(exact_repair_id)
            )
            repair_id = exact_repair_id
            authorized = _authorize_automatic_repair(
                (repair_id, candidate_code),
                step=step,
                source=reason,
                before_code="",
            )
            if authorized is None:
                return None
            deterministic_fallback_used = True
            preexecution_runner_repair_name = repair_id
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

        # ``--resume-from-step-id`` means the selected step is intentionally
        # rerun. Completed predecessors stay checkpointed, but the selected
        # step does not reuse its old script before the current coder/
        # deterministic-standard path unless the operator explicitly enables
        # the diagnostic fast path. Reused code still runs through every
        # current execution audit and repair gate.
        quarantined_resume_draft = (
            resume_controller.quarantined_concept_draft_for_step(step.step_id)
        )
        resume_summary_repair_code = (
            None
            if quarantined_resume_draft is not None
            else _resume_summary_repair_code()
        )
        preflight_resumed_code = None
        if (
            quarantined_resume_draft is None
            and resume_summary_repair_code is None
            and (
            requested_resume_from_step_id != step.step_id
            or reuse_selected_step_code_opt_in
            )
        ):
            preflight_resumed_code = resume_controller.prior_code_for_step(step.step_id)
        if quarantined_resume_draft is not None:
            code = _use_quarantined_draft(quarantined_resume_draft)
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
                    code = coder.run(context=coder_context, step=step)
                except Exception as exc:
                    resumed_code = resume_controller.prior_code_for_step(
                        step.step_id
                    )
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

        def _concept_findings_for_code(script_text: str) -> List[ValidationFinding]:
            """Run the single pre-execution concept gate for one code digest."""

            nonlocal quarantined_draft_active
            nonlocal quarantine_policy_superseded
            nonlocal pending_quarantined_errors

            code_findings = usage_auditor.audit(
                context=context,
                script_text=script_text,
                step=step,
            )
            code_findings.extend(
                pattern_auditor.audit(
                    context=context,
                    script_text=script_text,
                    step=step,
                )
            )
            try:
                if pipeline._enable_llm_concept_audit and deterministic_fallback_used:
                    code_findings.append(
                        ValidationFinding(
                            validator="llm_concept_auditor",
                            severity="info",
                            message=(
                                f"Skipped optional LLM concept audit for deterministic "
                                f"fallback code in step {step.step_id}; deterministic "
                                "audits still ran."
                            ),
                            detail={
                                "step_id": step.step_id,
                                "generation_mode": "deterministic_fallback",
                            },
                        )
                    )
                elif pipeline._enable_llm_concept_audit:
                    llm_audit_client = (
                        pipeline._llm_concept_auditor_client
                        or role_resolver("analyzer")
                    )
                    if llm_audit_client is not None:
                        code_findings.extend(
                            LLMConceptAuditor(llm_audit_client).audit(
                                context=context,
                                script_text=script_text,
                                step=step,
                            )
                        )
            except BaseException:
                # An operator interrupt must propagate, but a draft already
                # rejected by deterministic findings remains resumable.
                error_payloads = [
                    finding.model_dump(mode="json")
                    for finding in code_findings
                    if finding.severity == "error"
                ]
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
                else:
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

        concept_repair_attempts = 0
        llm_repair_used = False
        concept_audit_error_count = 0
        deterministic_concept_repairs = 0
        _MAX_DETERMINISTIC_CONCEPT_REPAIRS = 3
        applied_concept_repair_names: List[str] = []
        concept_approved_code_digest: Optional[str] = None
        while True:
            code = reorder_forward_references(code)
            usage_findings = _concept_findings_for_code(code)
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

            # Tier A — deterministic mechanical repair. For a closed set of
            # objectively-flagged ICU anti-patterns (e.g. silent fillna(0) on
            # a lab) there is a single neutral fix, so we apply it without a
            # model round-trip and re-audit. This does NOT consume the LLM
            # repair budget, and is bounded because each repair removes its
            # own pattern (a re-audit then finds nothing left to change).
            if deterministic_concept_repairs < _MAX_DETERMINISTIC_CONCEPT_REPAIRS:
                _audit_error_msgs = [
                    f.message for f in usage_findings if f.severity == "error"
                ]
                _det_code, _det_names = deterministic_concept_audit_repair(
                    code, _audit_error_msgs
                )
                if _det_names and _det_code != code:
                    denied_names = [
                        name
                        for name in _det_names
                        if _authorize_automatic_repair(
                            (name, _det_code),
                            step=step,
                            source="deterministic_concept_audit_repair",
                            before_code=code,
                        )
                        is None
                    ]
                    if denied_names:
                        _det_code, _det_names = code, []
                if _det_names and _det_code != code:
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
                    continue

            if concept_repair_attempts >= pipeline._max_code_repair_attempts:
                fallback_code = _deterministic_fallback_code("concept_audit")
                if fallback_code is not None:
                    fallback_checkpoint_error: Optional[Exception] = None
                    if resumed_quarantined_draft_used:
                        try:
                            checkpoint = store_quarantined_concept_draft(
                                run_dir=run_dir,
                                step_id=step.step_id,
                                code=code,
                                findings=[
                                    finding.model_dump(mode="json")
                                    for finding in usage_findings
                                    if finding.severity == "error"
                                ],
                            )
                            step_record["quarantined_draft_sha256"] = (
                                checkpoint.sha256
                            )
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
                            findings=[
                                finding.model_dump(mode="json")
                                for finding in usage_findings
                                if finding.severity == "error"
                            ],
                        )
                        step_record["quarantined_draft_sha256"] = checkpoint.sha256
                        step_record["quarantined_draft_relative_path"] = (
                            checkpoint.relative_path
                        )
                        step_record["quarantined_requires_repair"] = True
                        step_record["quarantine_checkpoint_is_latest_candidate"] = (
                            True
                        )
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
            audit_log = "\n".join(
                f"{f.severity.upper()}: {f.message}" for f in usage_findings
            )
            try:
                repaired_code = coder.repair(
                    context=coder_context,
                    step=step,
                    code=code,
                    run_log=(
                        "Static concept audit blocked this script before "
                        "execution. Fix all ICU-rule violations.\n\n" + audit_log
                    ),
                    attempt=concept_repair_attempts,
                )
                if quarantined_draft_active and not _python_repair_is_materially_changed(
                    code, repaired_code
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
                    step_record["quarantined_repair_noop_count"] = int(
                        step_record.get("quarantined_repair_noop_count") or 0
                    ) + 1
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
                checkpoint_error: Optional[Exception] = None
                try:
                    checkpoint = store_quarantined_concept_draft(
                        run_dir=run_dir,
                        step_id=step.step_id,
                        code=code,
                        findings=[
                            finding.model_dump(mode="json")
                            for finding in usage_findings
                            if finding.severity == "error"
                        ],
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
        runner_repair_name: Optional[str] = preexecution_runner_repair_name
        while True:
            code = reorder_forward_references(code)
            candidate_code_digest = sha256_of_bytes(code.encode("utf-8"))
            if candidate_code_digest != concept_approved_code_digest:
                # Every code mutation after execution (visual, contract,
                # runtime, deterministic, or fallback) returns through this
                # single digest-gated pre-execution concept audit. Never let a
                # repair bypass the safety gate merely because it originated
                # inside the runner loop.
                usage_findings = _concept_findings_for_code(code)
                step_record["usage_findings"] = [
                    finding.model_dump() for finding in usage_findings
                ]
                post_mutation_errors = [
                    finding for finding in usage_findings if finding.severity == "error"
                ]
                if post_mutation_errors:
                    checkpoint_error: Optional[Exception] = None
                    try:
                        checkpoint = store_quarantined_concept_draft(
                            run_dir=run_dir,
                            step_id=step.step_id,
                            code=code,
                            findings=[
                                finding.model_dump(mode="json")
                                for finding in post_mutation_errors
                            ],
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

            concept_repair_used = bool(
                concept_repair_attempts or deterministic_concept_repairs
            )
            current_generation_mode = _script_generation_mode(
                repair_attempts=repair_attempts,
                fallback_used=deterministic_fallback_used,
                runner_repair_name=runner_repair_name,
                resumed_code_reuse=resumed_code_reuse_used,
                concept_repair_used=concept_repair_used,
                llm_repair_used=llm_repair_used,
            )
            run_label = {
                "llm": "generated script",
                "resumed_code_reuse": "resumed script",
                "fallback": "fallback script",
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
            _clear_output_dir(run_dir / "steps" / step.step_id / "outputs")
            run_result = runner.run(step_id=step.step_id, code=code)
            executed_code_digest = sha256_of_file(run_result.script_path)
            step_record["executed_code_sha256"] = executed_code_digest
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
            else:
                total_repair_attempts = repair_attempts + concept_repair_attempts
                script_description = (
                    f"Repaired analysis script for step {step.step_id} "
                    f"(attempt {total_repair_attempts})."
                )
            script_evidence_id = None
            if current_generation_mode != "llm":
                script_digest = sha256_of_file(run_result.script_path)
                script_evidence_id = (
                    f"code_analysis_{script_digest[:8]}_{current_generation_mode}"
                )
            script_record = evidence.register_file(
                kind="code",
                description=script_description,
                source_path=run_result.script_path,
                produced_by_step=step.step_id,
                evidence_id=script_evidence_id,
                producer="coder",
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
            log_path = run_result.cwd / "run.log"
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
                        if (
                            visual_repair_attempts
                            >= pipeline._max_code_repair_attempts
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
                                    ),
                                    attempt=visual_repair_attempts,
                                )
                                llm_repair_used = True
                                _clear_output_dir(run_result.out_dir)
                                continue
                            except Exception as exc:
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
                        contract_repair_attempts
                        >= pipeline._max_code_repair_attempts
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
                    repair_attempts += 1
                    step_record["code_repair_attempts"] = repair_attempts
                    step_record["contract_repair_attempts"] = (
                        contract_repair_attempts
                    )
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
                    repair_guidance = _step_contract_repair_guidance(
                        step=step,
                        step_summary=visual_step_summary,
                        code=code,
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
                                + "\n\nSTRUCTURED CONTRACT FINDINGS (authoritative):\n"
                                + contract_log
                            ),
                            attempt=contract_repair_attempts,
                        )
                        llm_repair_used = True
                        _clear_output_dir(run_result.out_dir)
                        continue
                    except Exception as exc:
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
                break

            if log_path.exists():
                run_log = log_path.read_text(encoding="utf-8", errors="replace")
            else:
                run_log = (run_result.stdout or "") + "\n" + (run_result.stderr or "")
            if pipeline._enable_deterministic_runner_repair:
                before_repair_code = code
                plugin_repair = pipeline._case_plugin_registry.repair_code(
                    context=context,
                    step=step,
                    code=code,
                    run_log=run_log,
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
                        if plugin_repair is not None
                        and runner_repair is plugin_repair
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

            if runtime_repair_attempts >= pipeline._max_code_repair_attempts:
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

            repair_attempts += 1
            runtime_repair_attempts += 1
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
                code = coder.repair(
                    context=coder_context,
                    step=step,
                    code=code,
                    run_log=run_log,
                    attempt=repair_attempts,
                )
                llm_repair_used = True
                _clear_output_dir(run_result.out_dir)
            except Exception as exc:
                # 🔧 2026-05-16: distinguish transient LLM/parse failures from
                # exhausted budget. JSON-parse errors after the OpenAIClient
                # retry chain already exhausted its own backoff still bubble up
                # here; treat them as one used repair attempt and loop instead
                # of immediately bailing out. Only fall through to the
                # deterministic fallback / repair_failed branch when we've
                # genuinely used up max_code_repair_attempts.
                _msg = str(exc).lower()
                _is_transient = (
                    isinstance(exc, json.JSONDecodeError)
                    or "expecting value" in _msg
                    or ("json" in _msg and "decode" in _msg)
                    or "503" in _msg
                    or "rate" in _msg
                )
                if (
                    _is_transient
                    and runtime_repair_attempts < pipeline._max_code_repair_attempts
                ):
                    emit_progress(
                        "coder",
                        f"Transient repair failure for {step.step_id} "
                        f"(attempt {repair_attempts}): {type(exc).__name__}; retrying.",
                        run_id=run_id,
                        step_id=step.step_id,
                        current_step=step_current,
                        total_steps=total_steps,
                        repair_attempts=repair_attempts,
                    )
                    # The retained `code` is unchanged → next loop iteration
                    # will re-run the same script, fail the same way, then
                    # come back here for repair attempt N+1 with the same
                    # traceback in run_log. That gives the LLM another shot
                    # at producing parseable output.
                    continue

                fallback_code = _deterministic_fallback_code("repair_failed")
                if fallback_code is not None:
                    code = fallback_code
                    _clear_output_dir(run_result.out_dir)
                    continue
                with shared_lock:
                    findings.append(
                        ValidationFinding(
                            validator="coder",
                            severity="error",
                            message=f"Coder repair failed for step {step.step_id}: {exc}",
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
            else "analysis_figure"
            if _step_expects_figure(step)
            else None
        )
        if publication_step and not _has_figure_exports(run_result.out_dir):
            sibling_repair_id = "sibling_figure_exports_promote_v1"
            promoted = None
            if _automatic_repair_authorized(
                sibling_repair_id,
                step=step,
                source="publication_figure_sibling_promotion",
            ):
                promoted = _promote_sibling_figure_exports(
                    out_dir=run_result.out_dir
                )
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
                if (
                    _step_has_figure_only_output_contract(step)
                    and deterministic_figure_family_supported_for_upstream(
                        run_dir, step.step_id
                    )
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

        if _has_figure_exports(run_result.out_dir):
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
            step_record["source_evidence_ids"] = list(
                repair_source_evidence_ids
            )
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

        run_result.artefacts = sorted(
            p for p in run_result.out_dir.iterdir() if p.is_file()
        )

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

        evidence_ids_for_step: List[str] = [script_record.evidence_id]
        step_summary_record_id: Optional[str] = None
        for art in run_result.artefacts:
            step_aliases = _semantic_aliases_for(step, art)
            generation_mode = _script_generation_mode(
                repair_attempts=repair_attempts,
                fallback_used=deterministic_fallback_used,
                runner_repair_name=runner_repair_name,
                resumed_code_reuse=resumed_code_reuse_used,
                concept_repair_used=concept_repair_used,
                llm_repair_used=llm_repair_used,
            )
            if art.name == "step_summary.json":
                rec = evidence.register_file(
                    kind="statistic",
                    description=f"Machine-readable summary for step {step.step_id}.",
                    source_path=art,
                    produced_by_step=step.step_id,
                    inputs=repair_source_evidence_ids or None,
                    script_evidence_id=script_record.evidence_id,
                    aliases=step_aliases,
                    producer="runner",
                    generation_mode=generation_mode,
                    metadata={
                        "script_evidence_id": script_record.evidence_id,
                        "figure_role": figure_role or "analysis_figure",
                        "diagnostic_only": False,
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
                    inputs=repair_source_evidence_ids or None,
                    script_evidence_id=script_record.evidence_id,
                    aliases=step_aliases,
                    producer="runner",
                    generation_mode=generation_mode,
                    metadata={
                        "script_evidence_id": script_record.evidence_id,
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
                    inputs=repair_source_evidence_ids or None,
                    script_evidence_id=script_record.evidence_id,
                    aliases=step_aliases,
                    producer="runner",
                    generation_mode=generation_mode,
                    metadata={
                        "script_evidence_id": script_record.evidence_id,
                        "figure_role": figure_role or "analysis_figure",
                        "diagnostic_only": False,
                        **repair_evidence_metadata,
                    },
                )
            else:
                rec = evidence.register_file(
                    kind="log",
                    description=f"Auxiliary artefact {art.name}.",
                    source_path=art,
                    produced_by_step=step.step_id,
                    inputs=repair_source_evidence_ids or None,
                    script_evidence_id=script_record.evidence_id,
                    aliases=step_aliases,
                    producer="runner",
                    generation_mode=generation_mode,
                    metadata={
                        "script_evidence_id": script_record.evidence_id,
                        **repair_evidence_metadata,
                    },
                )
            evidence_ids_for_step.append(rec.evidence_id)

        step_summary: Dict[str, Any] = {}
        ssj = run_result.out_dir / "step_summary.json"
        if ssj.exists():
            try:
                loaded = json.loads(ssj.read_text(encoding="utf-8"))
            except Exception:
                loaded = None
            if isinstance(loaded, dict):
                step_summary = loaded
            else:
                # The coder emitted a non-dict JSON (bare string /
                # list / number). Keep it accessible but coerce to
                # a dict so every downstream consumer that calls
                # ``.get(...)`` still works.
                step_summary = {"raw": loaded}
        if step_summary_record_id is not None:
            step_record["step_summary_evidence_id"] = step_summary_record_id
        if step_summary and step_summary_record_id is not None:
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
        auto_contract_path = _ensure_step_figure_contract(
            step=step,
            out_dir=run_result.out_dir,
            step_summary=step_summary,
            evidence_ids=evidence_ids_for_step,
        )
        if auto_contract_path is not None:
            generation_mode = _script_generation_mode(
                repair_attempts=repair_attempts,
                fallback_used=deterministic_fallback_used,
                runner_repair_name=runner_repair_name,
                resumed_code_reuse=resumed_code_reuse_used,
                concept_repair_used=concept_repair_used,
                llm_repair_used=llm_repair_used,
            )
            rec = evidence.register_file(
                kind="log",
                description=(
                    f"Auto-generated figure contract for step {step.step_id}."
                ),
                source_path=auto_contract_path,
                produced_by_step=step.step_id,
                inputs=repair_source_evidence_ids or None,
                script_evidence_id=script_record.evidence_id,
                aliases=_semantic_aliases_for(step, auto_contract_path),
                producer="runner",
                generation_mode=generation_mode,
                metadata={
                    "script_evidence_id": script_record.evidence_id,
                    "figure_role": figure_role or "analysis_figure",
                    "synthesis": "step_summary_figure_contract_v1",
                    **repair_evidence_metadata,
                },
            )
            evidence_ids_for_step.append(rec.evidence_id)
            run_result.artefacts = sorted(
                set([*run_result.artefacts, auto_contract_path])
            )
        stat_findings = stat_validator.audit(
            context=context,
            cohort_path=cohort_path,
            step=step,
            out_dir=run_result.out_dir,
            step_summary=step_summary,
        )
        clinical_findings = clinical_validator.audit(
            context=context,
            step=step,
            out_dir=run_result.out_dir,
            step_summary=step_summary,
        )
        guard_findings = statistical_guard.audit(
            context=context,
            cohort_path=cohort_path,
            step=step,
            out_dir=run_result.out_dir,
            step_summary=step_summary,
        )
        with shared_lock:
            completed_records_snapshot = list(per_step_records)
        contract_findings = _step_contract_findings(
            step=step,
            step_summary=step_summary,
            completed_step_records=completed_records_snapshot,
        )
        contract_findings.extend(
            _cohort_definition_sensitivity_contract_findings(
                step=step,
                step_summary=step_summary,
                out_dir=run_result.out_dir,
                run_dir=run_dir,
                universe_path=universe_path,
                cohort_path=cohort_path,
                context=context,
            )
        )
        contract_findings.extend(
            cross_step_cohort_lock_validator.audit(
                step=step,
                step_summary=step_summary,
                completed_step_records=completed_records_snapshot,
            )
        )
        contract_findings.extend(
            cross_step_registered_output_validator.audit(
                step=step,
                step_summary=step_summary,
                completed_step_records=completed_records_snapshot,
            )
        )
        contract_findings.extend(
            cross_step_reconciliation_trace_validator.audit(
                step=step,
                step_summary=step_summary,
                out_dir=run_result.out_dir,
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
                completed_step_records=completed_records_snapshot,
            )
        )
        contract_findings.extend(
            primary_model_contract_validator.audit(
                step=step,
                step_summary=step_summary,
                context=context,
                completed_step_records=completed_records_snapshot,
                out_dir=run_result.out_dir,
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
                out_dir=run_result.out_dir,
            )
        )
        contract_findings.extend(
            _primary_model_leakage_findings(
                step=step,
                context=context,
                out_dir=run_result.out_dir,
            )
        )
        contract_findings.extend(
            figure_contract_validator.audit(
                step=step,
                out_dir=run_result.out_dir,
                run_dir=run_dir,
                step_summary=step_summary,
            )
        )
        # A deterministic PRIMARY runner owns its step's contract (see the early
        # check above): demote step_contract missing-output errors to advisory so
        # planner output-bloat cannot fail-close a step whose core estimate the
        # runner already produced. Figure/exposure/leakage validators still block.
        contract_findings = _demote_step_contract_for_primary_runner(
            step_record, step_summary, contract_findings
        )
        # A study-design family whose PRIMARY publication figure is assembled
        # deterministically in the write phase (phenotyping / prediction /
        # time_to_event / causal_emulation) must not fail-close a step merely
        # because its LLM-declared step figure is single-panel: that keeps
        # execution_complete False and skips the very renderer that builds the
        # multi-panel primary. The display-suite gate stays the fail-closed
        # backstop. See _demote_result_figure_shape_for_family_renderer.
        contract_findings = _demote_result_figure_shape_for_family_renderer(
            context, contract_findings
        )
        figure_source_findings = figure_source_validator.audit(
            step=step,
            out_dir=run_result.out_dir,
            run_dir=run_dir,
            step_summary=step_summary,
            completed_step_records=completed_records_snapshot,
        )
        figure_gate_errors = [
            finding
            for finding in contract_findings + figure_source_findings
            if finding.severity == "error"
            and finding.validator in {"figure_contract_quality", "figure_source_data"}
        ]
        repairable_publication_step = (
            publication_step
            and _step_has_figure_only_output_contract(step)
            and deterministic_figure_family_supported_for_upstream(
                run_dir, step.step_id
            )
        )
        if repairable_publication_step and figure_gate_errors:
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
            transformation = (
                "Replaced invalid figure-step exports with a deterministic "
                "publication figure from the registered parent table for this "
                "step type."
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
                            finding.message for finding in figure_gate_errors[:5]
                        ],
                    },
                    transformation=transformation,
                )
                run_result.artefacts = sorted(
                    p for p in run_result.out_dir.iterdir() if p.is_file()
                )
                repaired_summary = run_result.out_dir / "step_summary.json"
                if repaired_summary.exists():
                    try:
                        loaded_summary = json.loads(
                            repaired_summary.read_text(encoding="utf-8")
                        )
                        if isinstance(loaded_summary, dict):
                            step_summary = loaded_summary
                    except Exception:
                        pass
                contract_findings = _step_contract_findings(
                    step=step,
                    step_summary=step_summary,
                    completed_step_records=completed_records_snapshot,
                )
                contract_findings.extend(
                    _cohort_definition_sensitivity_contract_findings(
                        step=step,
                        step_summary=step_summary,
                        out_dir=run_result.out_dir,
                        run_dir=run_dir,
                        universe_path=universe_path,
                        cohort_path=cohort_path,
                        context=context,
                    )
                )
                contract_findings.extend(
                    cross_step_cohort_lock_validator.audit(
                        step=step,
                        step_summary=step_summary,
                        completed_step_records=completed_records_snapshot,
                    )
                )
                contract_findings.extend(
                    cross_step_registered_output_validator.audit(
                        step=step,
                        step_summary=step_summary,
                        completed_step_records=completed_records_snapshot,
                    )
                )
                contract_findings.extend(
                    cross_step_reconciliation_trace_validator.audit(
                        step=step,
                        step_summary=step_summary,
                        out_dir=run_result.out_dir,
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
                        completed_step_records=completed_records_snapshot,
                    )
                )
                contract_findings.extend(
                    primary_model_contract_validator.audit(
                        step=step,
                        step_summary=step_summary,
                        context=context,
                        completed_step_records=completed_records_snapshot,
                        out_dir=run_result.out_dir,
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
                        out_dir=run_result.out_dir,
                    )
                )
                contract_findings.extend(
                    _primary_model_leakage_findings(
                        step=step,
                        context=context,
                        out_dir=run_result.out_dir,
                    )
                )
                contract_findings.extend(
                    figure_contract_validator.audit(
                        step=step,
                        out_dir=run_result.out_dir,
                        run_dir=run_dir,
                        step_summary=step_summary,
                    )
                )
                figure_source_findings = figure_source_validator.audit(
                    step=step,
                    out_dir=run_result.out_dir,
                    run_dir=run_dir,
                    step_summary=step_summary,
                    completed_step_records=completed_records_snapshot,
                )
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
        evidence_refs_for_step = _evidence_refs_for_names(evidence_ids_for_step)
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
                metadata={"script_evidence_id": script_record.evidence_id},
            )
            evidence_ids_for_step.append(critique_record.evidence_id)
            step_record["critique_report"] = critique.model_dump(mode="json")
            if critique.status in {"needs_revision", "blocked"}:
                with shared_lock:
                    findings.append(
                        ValidationFinding(
                            validator="critic_agent",
                            severity=(
                                "warning"
                                if critique.status == "needs_revision"
                                else "error"
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
                        )
                    )

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
                )
            except Exception as exc:
                interpretation = f"(analyzer failed: {exc})"
                interp_generation_mode = "system"
        interp_record = evidence.register_text(
            kind="log",
            description=f"Analyzer interpretation for step {step.step_id}.",
            text=interpretation,
            filename=f"interpretation_{step.step_id}.md",
            produced_by_step=step.step_id,
            script_evidence_id=script_record.evidence_id,
            producer="analyzer",
            generation_mode=interp_generation_mode,
            prompt_pack_version=prompt_version,
        )
        step_record["interpretation_evidence_id"] = interp_record.evidence_id
        evidence_ids_for_step.append(interp_record.evidence_id)
        step_record["evidence_ids"] = list(dict.fromkeys(evidence_ids_for_step))
        step_record.pop("review_pending", None)
        _propagate_findings_to_evidence(
            evidence_ids_for_step,
            usage_findings
            + stat_findings
            + clinical_findings
            + guard_findings
            + contract_findings
            + figure_source_findings,
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
        )
        has_contract_error = step_record["status"] == "contract_failed"
        final_cleanup_finding: Optional[ValidationFinding] = None
        if step_record["status"] == "ok":
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
        with shared_lock:
            if final_cleanup_finding is not None:
                findings.append(final_cleanup_finding)
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
                    f"Step {step_current}/{total_steps} could not retire its "
                    f"quarantine: {step.step_id}."
                    if step_record["status"] == "blocked_quarantine_cleanup"
                    else f"Step {step_current}/{total_steps} complete: {step.step_id}."
                )
            ),
            status=("complete" if step_record["status"] == "ok" else "error"),
            run_id=run_id,
            step_id=step.step_id,
            current_step=step_current,
            total_steps=total_steps,
        )
        return step_record

    steps_to_run = resume_controller.remaining_steps(
        plan=plan,
        executed_step_ids=set(resumed_step_ids),
    )
    for skipped_step_id in sorted(resumed_step_ids):
        emit_progress(
            "resume",
            f"Skipped completed step from prior run: {skipped_step_id}.",
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

    if (
        pipeline._max_concurrent_steps <= 1
        or len(steps_to_run) <= 1
        or pipeline._enable_replanning
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

        executed_step_ids = set(resumed_step_ids)
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
