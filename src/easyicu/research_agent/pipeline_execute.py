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

import json
import logging
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
    FigureContractQualityValidator,
    FigureSourceDataValidator,
    LLMConceptAuditor,
    StatisticalGuard,
    StatisticalValidator,
)
from .code_repair import (
    _deterministic_runner_repair,
    _deterministic_summary_repair,
    deterministic_contract_repair,
    deterministic_concept_audit_repair,
)
from .cohort_repair import extract_cohort_definition_from_prose
from .cohort_schema import (
    assert_cohort_definition_locked,
    materialize_locked_analysis_cohort,
    write_locked_cohort_definition,
)
from .contracts import ValidationFinding, _ExecutePhaseResult, _PlanPhaseResult
from .deterministic_sensitivity import (
    cohort_definition_overlap_code,
    cohort_definition_sensitivity_comparison_code,
)
from .deterministic_causal import causal_primary_analysis_code
from .deterministic_ordinal import ordinal_dose_response_analysis_code
from .deterministic_survival import survival_primary_analysis_code
from .estimators import fit_robustness_rows_from_records
from .llm import MockLLMClient
from .pipeline import (
    _build_probe_summary,
    _clear_output_dir,
    _has_figure_exports,
    _promote_prior_publication_bundle,
    _promote_sibling_figure_exports,
    _render_publication_bundle_from_prior_outputs_for_step,
    _semantic_aliases_for,
)
from .publication_figures import make_figure_contract
from .plan_utils import (
    _cap_plan_preserving_figure_steps,
    _cohort_definition_contract_findings,
    _cohort_definition_is_empty,
    _cohort_definition_prose,
    _parent_step_id_for_figure_step,
    _plan_expects_analysis_cohort,
    _preserve_figure_steps_after_replan,
    _primary_exposure_contract_findings,
    _primary_exposure_measurement_filter_findings,
    _primary_exposure_overadjustment_findings,
    _primary_model_leakage_findings,
    _step_contract_findings,
    _step_contract_repair_guidance,
    _step_expects_figure,
)
from .pipeline_resume import ResumeController, upsert_step_record
from .schema import AnalysisPlan, AnalysisStep, EvidenceRef
from .robustness_panel import (
    assert_robustness_specs_locked,
    build_robustness_panel_from_records,
    robustness_specs_for_execution,
    write_robustness_panel,
)
from .repair_registry import InvariantStatus, RepairLedger, RepairObservedState
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

# Max directed full-replans fired when a model/estimation step self-blocks on a
# task-viable cohort. Two attempts give the replanner a fair chance to honour
# the override directive; beyond that the run falls back to an honest
# diagnostic_only rather than burning the replanner on a stuck plan.
_MAX_DIRECTED_MODEL_REPLANS = 2


def _is_terminal_publication_figure_repair_step(step: Any) -> bool:
    """Return true for rendering-only terminal publication figure repair steps."""

    expected_outputs = getattr(step, "expected_outputs", None) or []
    haystack = " ".join(
        str(part or "")
        for part in (
            getattr(step, "step_id", ""),
            getattr(step, "intent", ""),
            getattr(step, "method", ""),
            " ".join(str(item) for item in expected_outputs),
        )
    )
    lowered = haystack.lower().replace("_", "-")
    if not all(token in lowered for token in ("publication", "figure", "repair")):
        return False
    step_id = str(getattr(step, "step_id", "") or "").lower()
    return (
        "rendering-only" in lowered
        or "rendering only" in lowered
        or step_id.endswith("publication_figure_repair")
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

    completed_ok = {
        str(record.get("step_id") or "")
        for record in (completed_records or [])
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

    for record in reversed(list(completed_records or [])):
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
) -> Tuple[Tuple[str, Optional[str], Tuple[str, ...]], ...]:
    """Substantive fingerprint of a plan's step DAG, ignoring prose.

    Two plans with the same ``(step_id, method, expected_outputs)`` per step
    are analytically identical even if the replanner reworded each step's
    ``intent``. Used by ``_maybe_replan`` to suppress no-op revisions that
    would otherwise burn an LLM call and the convergence budget without
    changing the analysis.
    """
    return tuple(
        (step.step_id, step.method, tuple(step.expected_outputs)) for step in plan.steps
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


_PRIMARY_DETERMINISTIC_RUNNERS = {
    "causal_primary_iptw",
    "survival_primary_cox",
    "ordinal_dose_response",
}

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
        "ordinal_regression",
        "ordinal_logistic_regression",
        "trend_analysis",
    }
)
# General dose-response / graded-exposure vocabulary (case-neutral: never a
# specific score name). Present in the question, intent, or declared outputs.
_ORDINAL_DOSE_SIGNAL_TOKENS = (
    "dose-response",
    "dose response",
    "dose–response",
    "per-stage",
    "per stage",
    "graded exposure",
    "severity gradient",
    "stage gradient",
    "ordinal trend",
)
_ORDINAL_OUTPUT_TOKENS = (
    "dose_response",
    "per_stage",
    "per-stage",
    "trend_or",
    "ordinal_trend",
)

# --- Cohort-definition-sensitivity routing (precise, not blunt keyword) -------
# A cohort-definition-sensitivity step VARIES the cohort/eligibility definition
# and compares the result across alternative definitions. The authoritative
# signal is the planner's own ``method`` key; the historical blunt test --
# ``"sensitivity" in blob and ("cohort"|"definition" in blob)`` -- false-positives
# on a PRIMARY estimand step that merely mentions a pre-specified within-cohort
# sensitivity sub-analysis. E3's ordinal dose-response primary step said
# "reconciled primary COHORT denominator" and "survivor-only LOS SENSITIVITY",
# so the blunt test both (a) vetoed the ordinal runner and (b) let the
# cohort-sensitivity runner claim the step, which then skipped for lack of an
# alternative-cohort input and deadlocked the run. Require the alternative-
# definition signal instead. Stays case-neutral: no score/variable literals.
_COHORT_DEF_SENSITIVITY_METHODS = frozenset(
    {
        "cohort_definition_sensitivity",
        "cohort_sensitivity",
        "definition_sensitivity",
    }
)
_COHORT_DEF_SENSITIVITY_ID_TOKENS = (
    "cohort_definition_sensitivity",
    "cohort-definition-sensitivity",
    "definition_sensitivity",
)
_COHORT_DEF_SENSITIVITY_OUTPUT_TOKENS = (
    "alternative_cohort_attrition",
    "cohort_overlap",
    "overlap_and_movement_across_cohorts",
    "sensitivity_grid",
    # NB: NOT "sensitivity_comparison" -- it substring-matches a primary step's
    # "sensitivity_comparison_grid" output and over-claimed E3's merged
    # association_with_cohort_sensitivity step. Each kept token uniquely signals
    # an across-DEFINITION comparison; "sensitivity_grid" is not a substring of
    # "sensitivity_comparison_grid".
    "definition_sensitivity",
    "sensitivity_definition_summary",
    "outcome_by_definition",
    "adjustment_denominator_sensitivity",
)
_COHORT_DEF_SENSITIVITY_PHRASES = (
    "cohort definition sensitivity",
    "cohort-definition sensitivity",
    "alternative cohort definition",
    "alternative cohort definitions",
    "alternative eligibility",
    "alternative inclusion",
    "alternative exclusion",
    "vary the cohort definition",
    "varying the cohort definition",
)


def _is_cohort_definition_sensitivity_step(
    method: str,
    step_id: str,
    intent: str,
    expected_outputs: Sequence[str],
) -> bool:
    """Pure routing test: is this an ACTUAL cohort-definition-sensitivity step?

    True only when the step varies the cohort/eligibility DEFINITION and compares
    across alternatives -- signalled by the planner ``method`` key, the step_id,
    alternative-cohort output tables, or an explicit alternative-definition
    phrase. A primary estimand step that merely mentions a within-cohort
    "sensitivity" sub-analysis (on a single, already-defined cohort) returns
    False, so it keeps its own primary runner rather than being hijacked here.

    Extracted from the two preflight closures so the discriminator is shared and
    unit-testable without a full bench.
    """
    m = str(method or "").lower()
    if m in _COHORT_DEF_SENSITIVITY_METHODS:
        return True
    # A hybrid "<head>_with_<rider>" method belongs to its HEAD. A definition-
    # sensitivity head (e.g. cohort_definition_sensitivity_with_binomial_glm) is
    # a genuine definition comparison; a PRIMARY head with a sensitivity rider
    # (association_with_cohort_sensitivity) is NOT -- its trailing
    # "cohort_sensitivity" is a robustness rider, so it must not be claimed here.
    # Match only the head, never the tail.
    if m.split("_with_", 1)[0] in _COHORT_DEF_SENSITIVITY_METHODS:
        return True
    sid = str(step_id or "").lower()
    if any(tok in sid for tok in _COHORT_DEF_SENSITIVITY_ID_TOKENS):
        return True
    expected_blob = " ".join(
        str(item or "") for item in (expected_outputs or [])
    ).lower()
    if any(tok in expected_blob for tok in _COHORT_DEF_SENSITIVITY_OUTPUT_TOKENS):
        return True
    blob = " ".join([sid, str(intent or "").lower(), expected_blob])
    return any(phrase in blob for phrase in _COHORT_DEF_SENSITIVITY_PHRASES)


def _method_has_ordinal_primary_token(method: str) -> bool:
    """True if ``method`` IS, or is a compound built from, a primary-estimation
    method token (e.g. ``multivariable_association`` -> ``association``,
    ``adjusted_logistic_regression`` -> ``regression``).

    Word-boundary token match (split on ``_`` / ``-``), NOT substring, so
    ``remodeling`` never matches ``model``. This is only ever reached AFTER the
    dose-response-signal gate in :func:`_ordinal_dose_response_step_matches`, so a
    plain association method still routes here only when a dose signal is also
    present — the anti-hijack guard is preserved. Motivated by E3's real primary
    step, whose planner ``method`` was ``multivariable_association`` (a genuine
    ordinal-trend estimation for a "dose-response gradient / ordered exposure"
    question) that the exact-match allowlist missed, starving the deterministic
    ordinal runner and leaving the run with no traceable primary figure.
    """
    if method in _ORDINAL_PRIMARY_METHODS:
        return True
    tokens = method.replace("-", "_").split("_")
    return any(tok in _ORDINAL_PRIMARY_METHODS for tok in tokens)


def _ordinal_dose_response_step_matches(
    method: str, blob: str, expected_blob: str
) -> bool:
    """Pure routing test: is this the PRIMARY dose-response estimation step?

    Extracted from the preflight closure so it is unit-testable without a full
    bench (E3's first bench routed its primary step to the LLM coder because the
    method ``association_analysis`` was not recognised here). The caller supplies
    lowercased strings and has already excluded figure / cohort-definition-
    sensitivity steps.

    ``blob`` = step_id + intent + research_question + expected_outputs;
    ``expected_blob`` = expected_outputs only.
    """
    # A hybrid "<head>_with_<rider>" method is owned by its primary HEAD, so a
    # merged "association_with_cohort_sensitivity" step is claimed by the ordinal
    # runner (head "association") when the dose-response narrative signal is
    # present -- rather than being blocked wholesale by the cohort-sensitivity
    # runner. The dose-signal requirement below still guards a plain non-graded
    # association step from being hijacked.
    head = method.split("_with_", 1)[0]
    if method in _ORDINAL_EXPLICIT_METHODS or head in _ORDINAL_EXPLICIT_METHODS:
        return True
    if any(tok in expected_blob for tok in _ORDINAL_OUTPUT_TOKENS):
        return True
    # otherwise require BOTH a dose-response narrative signal AND a primary-
    # estimation method, so a plain association step that merely mentions a
    # "trend" is not hijacked.
    if not any(tok in blob for tok in _ORDINAL_DOSE_SIGNAL_TOKENS):
        return False
    # Tokenise only the HEAD (owner) of a ``<head>_with_<rider>`` method, never
    # the rider: a ``cohort_definition_sensitivity_with_binomial_glm`` step is a
    # definition comparison whose ``glm`` rider must NOT pull it into the ordinal
    # runner (the head "cohort_definition_sensitivity" has no ordinal token). For
    # a non-hybrid method, ``head`` == ``method``.
    return _method_has_ordinal_primary_token(head)


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
    """A deterministic PRIMARY runner owns its step's contract.

    When such a runner produced its core estimate, demote ``step_contract``
    missing-output ERRORS to advisory warnings. Otherwise a planner that
    over-specifies a step's ``expected_outputs`` (e.g. 17 documentation tables a
    causal step does not need) fail-closes the step and triggers a repair that
    replaces the trustworthy deterministic estimate with fragile LLM code -- the
    exact failure that left the H2 causal run with ``adjusted_effect=None`` even
    though the IPTW runner had produced OR 3.04. Integrity findings from other
    validators (exposure / overadjustment / leakage / figure) are left untouched
    -- they still block.
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

    Deadlock this breaks (2026-07-07, M3 subphenotype): phenotyping (and any
    family in ``FAMILY_RENDERERS``) has a deterministic multi-panel publication
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
                    f"Revised analysis plan (reason={reason}; " f"resume re-revision)."
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
        # Guard against the replanner silently dropping figure-producing
        # steps; task contracts (e.g. EasyICU experiment runner) still
        # require those artefacts regardless of the LLM's revised framing.
        revised, preservation_findings = _preserve_figure_steps_after_replan(
            current=current_plan,
            revised=revised,
        )
        if preservation_findings:
            findings.extend(preservation_findings)

        # C1 (pilot 20260515 fix): cap total plan size after a replan.
        # The pilot saw the replanner grow a simple SOFA-2 association
        # to 30 steps with 13 revisions and never converge. The cap
        # truncates excess late-stage steps and forces the replanner
        # to revise existing steps in place on later passes. Cap of 0
        # disables the guard for backward compatibility.
        cap = pipeline._max_total_steps
        if cap > 0 and len(revised.steps) > cap:
            protected_step_ids = [
                str(record.get("step_id"))
                for record in (completed_records or [])
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

        # No-op detection on the *substantive* step DAG, not the full
        # model_dump. A verbose replanner can rewrite each step's ``intent``
        # prose without changing the analysis; that must not count as a
        # revision or burn the convergence budget. (E1 20260611: revisions
        # 4-6 carried an identical DAG, each a wasted LLM call, and the run
        # was killed mid-step-7 before finishing.)
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
    ) -> None:
        try:
            with repair_ledger_lock:
                provenance = repair_ledger.append_application(
                    repair_id=repair_id,
                    step_id=step_id,
                    trigger=trigger,
                    transformation=transformation,
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

    def _script_generation_mode(
        *,
        repair_attempts: int,
        fallback_used: bool,
        runner_repair_name: Optional[str] = None,
        resumed_code_reuse: bool = False,
    ) -> str:
        if resumed_code_reuse:
            return "resumed_code_reuse"
        if fallback_used:
            return "fallback"
        if repair_attempts > 0:
            return "repaired"
        if runner_repair_name:
            return "runner_repaired"
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
        messages: List[str] = []
        for group in finding_groups:
            for finding in group:
                if finding.message:
                    messages.append(finding.message)
        return messages

    def _failed_dependency_record(step: AnalysisStep) -> Optional[Dict[str, Any]]:
        parent_step_id = _parent_step_id_for_figure_step(step)
        if parent_step_id is None:
            return None
        with shared_lock:
            records = list(per_step_records)
        for record in records:
            if record.get("step_id") != parent_step_id:
                continue
            if str(record.get("status") or "").lower() == "ok":
                return None
            return record
        return None

    def _execute_one_step(step: AnalysisStep) -> Dict[str, Any]:
        nonlocal runtime_state
        step_record: Dict[str, Any] = {
            "step_id": step.step_id,
            "intent": step.intent,
        }
        resumed_code_reuse_used = False
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
            detail = {
                "step_id": step.step_id,
                "resume_from_step_id": requested_resume_from_step_id,
                "evidence_id": resumed_record.get("evidence_id"),
                "relative_path": resumed_record.get("relative_path"),
            }
            if error is None:
                message = (
                    f"Explicit resume reused prior agent-generated code for step "
                    f"{step.step_id} before requesting a new coder script."
                )
            else:
                detail["error"] = str(error)
                message = (
                    f"Coder agent failed for step {step.step_id}; reused prior "
                    "agent-generated code from resume evidence."
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
            # Match on the step id with the router's own token groups so the
            # preflight only claims figure steps the deterministic renderer
            # can actually serve. Matching intent/method text here used to
            # hijack steps whose prose merely mentioned "cohort"/"quality"
            # (e.g. a baseline/absolute-risk figure step), replacing the LLM
            # coder with a rescue that then emitted no figure exports.
            from .pipeline import deterministic_figure_family_supported

            return _step_expects_figure(step) and deterministic_figure_family_supported(
                step.step_id
            )

        def _deterministic_publication_figure_code(
            reason: str,
            *,
            preflight: bool = False,
        ) -> Optional[str]:
            nonlocal deterministic_fallback_used
            if (
                deterministic_fallback_used
                or not pipeline._enable_deterministic_runner_repair
                or not _step_expects_figure(step)
                or (preflight and not _publication_figure_preflight_supported())
            ):
                return None
            deterministic_fallback_used = True
            step_record["deterministic_code_fallback"] = reason
            return """
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

if repair_id is None:
    summary = {
        "rendering_only": True,
        "deterministic_publication_figure_rescue": "no_parent_outputs",
        "figure_files": [],
        "warning": "No compatible parent outputs were available for deterministic figure rendering.",
    }
    with open(out_dir / "step_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
else:
    print(json.dumps({"deterministic_publication_figure_rescue": repair_id}))
"""

        def _cohort_definition_sensitivity_preflight_supported() -> bool:
            if _step_expects_figure(step):
                return False
            # Guard: a PRIMARY result-bearing analysis (Cox/KM survival,
            # prediction model, causal contrast, clustering) can legitimately
            # mention "sensitivity" and a "definition" in its intent without being
            # a cohort-definition-sensitivity comparison. The deterministic
            # cohort-sensitivity code emits alternative_cohort_attrition /
            # cohort_overlap outputs, never cox_summary / auroc / cluster tables,
            # so it must NOT claim such a step -- doing so blocks the real
            # estimator on a missing alternative-cohort input (H1 survival step
            # was hijacked this way and never fit its Cox model).
            method = str(step.method or "").lower()
            if method in (
                "survival_analysis",
                "prediction_model",
                "dynamic_prediction",
                "causal_inference",
                "treatment_response",
                "clustering",
                "validation",
            ):
                return False
            expected_blob = " ".join(
                str(item or "") for item in (step.expected_outputs or [])
            ).lower()
            if any(
                token in expected_blob
                for token in (
                    "cox_summary",
                    "km_curve",
                    "hazard_ratio",
                    "kaplan",
                    "auroc",
                    "roc_curve",
                    "calibration_curve",
                    "cluster_characteristics",
                    "silhouette",
                    "causal_effect",
                    "covariate_balance",
                )
            ):
                return False
            # Only claim a step that ACTUALLY varies the cohort/eligibility
            # definition (method/id/output-table/phrase signal). The former blunt
            # "sensitivity"+"cohort" co-occurrence hijacked primary estimand steps
            # (ordinal/association/descriptive) whose intent merely mentions a
            # within-cohort sensitivity sub-analysis on the primary cohort.
            return _is_cohort_definition_sensitivity_step(
                str(step.method or ""),
                str(step.step_id or ""),
                str(step.intent or ""),
                step.expected_outputs or [],
            )

        def _cohort_definition_overlap_preflight_supported() -> bool:
            if _step_expects_figure(step):
                return False
            blob = " ".join(
                [
                    str(step.step_id or ""),
                    str(step.intent or ""),
                    str(step.method or ""),
                    *[str(item or "") for item in (step.expected_outputs or [])],
                ]
            ).lower()
            expected = {str(item or "").lower() for item in step.expected_outputs or []}
            has_overlap_outputs = any(
                token in expected
                for token in (
                    "table:alternative_cohort_attrition",
                    "table:cohort_overlap_matrix",
                )
            )
            if has_overlap_outputs:
                return True
            return (
                "cohort_definition_sensitivity" in blob
                and any(token in blob for token in ("alternative", "eligibility"))
                and any(
                    token in blob for token in ("overlap", "attrition", "denominator")
                )
            )

        def _survival_primary_analysis_preflight_supported() -> bool:
            """True for the PRIMARY time-to-event result step.

            The deterministic Cox runner owns the survival estimand (exposure +
            Cox + KM data) so this result path is reproducible instead of
            varying run-to-run. It must NOT claim a figure step (the family
            figure renderer handles those) nor a sensitivity/prep step.
            """
            if _step_expects_figure(step):
                return False
            method = str(step.method or "").lower()
            if method not in ("survival_analysis", "time_to_event", "cox", "cox_ph"):
                return False
            expected_blob = " ".join(
                str(item or "") for item in (step.expected_outputs or [])
            ).lower()
            # A step that DECLARES primary Cox/KM outputs IS the primary survival
            # step, never a cohort-definition-sensitivity re-fit — even when its
            # thorough output set also includes innocuously-named tables
            # (sensitivity_results, cohort_flow, survival_time_definition) or its
            # intent narrates sensitivity analyses. Keying the exclusion on those
            # words dropped fix3i's PRIMARY step to the LLM coder (swapped-column
            # garbage). Declaring the primary estimand wins. This mirrors the
            # sibling cohort-sensitivity predicate, which already declines when
            # cox_summary/km_curve/hazard_ratio are declared.
            if any(
                token in expected_blob
                for token in (
                    "cox_summary",
                    "cox_model",
                    "hazard_ratio",
                    "km_curve",
                    "kaplan",
                )
            ):
                return True
            # Otherwise fall back to the identity heuristic: a genuine
            # cohort-definition-sensitivity step (one that declares NO primary Cox
            # output) is handled by a different deterministic runner.
            blob = " ".join(
                [
                    str(step.step_id or ""),
                    str(step.intent or ""),
                    expected_blob,
                ]
            ).lower()
            if "sensitivity" in blob and any(
                t in blob for t in ("cohort", "definition", "eligibility")
            ):
                return False
            return True

        def _deterministic_survival_primary_analysis_code(
            reason: str,
            *,
            preflight: bool = False,
        ) -> Optional[str]:
            nonlocal deterministic_fallback_used
            if (
                deterministic_fallback_used
                or not pipeline._enable_deterministic_runner_repair
                or (preflight and not _survival_primary_analysis_preflight_supported())
            ):
                return None
            if not _survival_primary_analysis_preflight_supported():
                return None
            deterministic_fallback_used = True
            step_record["deterministic_code_fallback"] = reason
            step_record["deterministic_standard_analysis"] = "survival_primary_cox"
            emit_progress(
                "coder",
                f"Using deterministic Cox survival runner for {step.step_id}.",
                run_id=run_id,
                step_id=step.step_id,
                current_step=step_current,
                total_steps=total_steps,
                fallback_reason=reason,
            )
            return survival_primary_analysis_code()

        def _causal_primary_analysis_preflight_supported() -> bool:
            """True for the PRIMARY causal-inference result step.

            The deterministic IPTW runner owns the causal estimand (propensity +
            stabilised weights + weighted marginal odds ratio + balance data) so
            this result path is reproducible instead of varying run-to-run
            through the LLM coder. It must NOT claim a figure step (the family
            figure renderer handles those) nor a cohort-definition-sensitivity
            step.
            """
            if _step_expects_figure(step):
                return False
            method = str(step.method or "").lower()
            if method not in (
                "causal_inference",
                "causal_emulation",
                "iptw",
                "ipw",
                "psm",
                "propensity",
                "propensity_score",
                "target_trial",
                "target_trial_emulation",
            ):
                return False
            expected_blob = " ".join(
                str(item or "") for item in (step.expected_outputs or [])
            ).lower()
            # A step that DECLARES the primary causal-effect / balance / propensity
            # outputs IS the primary causal step, even when its output set also
            # includes innocuously-named tables or its intent narrates sensitivity.
            if any(
                token in expected_blob
                for token in (
                    "causal_effect",
                    "balance_pre_post",
                    "covariate_balance",
                    "propensity",
                    "adjusted_effect",
                    "max_smd",
                    "primary_causal",
                )
            ):
                return True
            blob = " ".join(
                [
                    str(step.step_id or ""),
                    str(step.intent or ""),
                    expected_blob,
                ]
            ).lower()
            if "sensitivity" in blob and any(
                t in blob for t in ("cohort", "definition", "eligibility")
            ):
                return False
            return True

        def _deterministic_causal_primary_analysis_code(
            reason: str,
            *,
            preflight: bool = False,
        ) -> Optional[str]:
            nonlocal deterministic_fallback_used
            if (
                deterministic_fallback_used
                or not pipeline._enable_deterministic_runner_repair
                or (preflight and not _causal_primary_analysis_preflight_supported())
            ):
                return None
            if not _causal_primary_analysis_preflight_supported():
                return None
            deterministic_fallback_used = True
            step_record["deterministic_code_fallback"] = reason
            step_record["deterministic_standard_analysis"] = "causal_primary_iptw"
            emit_progress(
                "coder",
                f"Using deterministic IPTW causal runner for {step.step_id}.",
                run_id=run_id,
                step_id=step.step_id,
                current_step=step_current,
                total_steps=total_steps,
                fallback_reason=reason,
            )
            return causal_primary_analysis_code()

        def _ordinal_dose_response_preflight_supported() -> bool:
            """True for the PRIMARY dose-response result step (a graded ORDINAL
            exposure vs a binary outcome).

            The deterministic ordinal runner owns the trend estimand (adjusted
            odds ratio per +1 stage + the per-stage forest), so a dose-response
            headline does not vary run-to-run through the LLM coder. It must NOT
            claim a figure step, a cohort-definition-sensitivity step, nor a
            plain (non-graded) association step. The trigger stays case-neutral:
            general dose-response vocabulary, never a specific score name.
            """
            if _step_expects_figure(step):
                return False
            method = str(step.method or "").lower()
            expected_blob = " ".join(
                str(item or "") for item in (step.expected_outputs or [])
            ).lower()
            blob = " ".join(
                [
                    str(step.step_id or ""),
                    str(step.intent or ""),
                    str(getattr(context, "research_question", "") or ""),
                    expected_blob,
                ]
            ).lower()
            # a cohort-definition-sensitivity step is owned by another runner;
            # use the precise discriminator, NOT a blunt "sensitivity"+"cohort"
            # co-occurrence. The blunt test vetoed this very (primary ordinal)
            # step because its intent mentions the "reconciled primary cohort
            # denominator" and a pre-specified within-cohort LOS "sensitivity"
            # sub-analysis -- neither of which makes it a definition comparison.
            if _is_cohort_definition_sensitivity_step(
                method, step.step_id, step.intent, step.expected_outputs
            ):
                return False
            return _ordinal_dose_response_step_matches(method, blob, expected_blob)

        def _deterministic_ordinal_dose_response_code(
            reason: str,
            *,
            preflight: bool = False,
        ) -> Optional[str]:
            nonlocal deterministic_fallback_used
            if (
                deterministic_fallback_used
                or not pipeline._enable_deterministic_runner_repair
                or (preflight and not _ordinal_dose_response_preflight_supported())
            ):
                return None
            if not _ordinal_dose_response_preflight_supported():
                return None
            deterministic_fallback_used = True
            step_record["deterministic_code_fallback"] = reason
            step_record["deterministic_standard_analysis"] = "ordinal_dose_response"
            emit_progress(
                "coder",
                f"Using deterministic ordinal dose-response runner for {step.step_id}.",
                run_id=run_id,
                step_id=step.step_id,
                current_step=step_current,
                total_steps=total_steps,
                fallback_reason=reason,
            )
            return ordinal_dose_response_analysis_code()

        def _deterministic_cohort_definition_overlap_code(
            reason: str,
            *,
            preflight: bool = False,
        ) -> Optional[str]:
            nonlocal deterministic_fallback_used
            if (
                deterministic_fallback_used
                or not pipeline._enable_deterministic_runner_repair
                or (preflight and not _cohort_definition_overlap_preflight_supported())
            ):
                return None
            if not _cohort_definition_overlap_preflight_supported():
                return None
            deterministic_fallback_used = True
            step_record["deterministic_code_fallback"] = reason
            step_record["deterministic_standard_analysis"] = "cohort_definition_overlap"
            emit_progress(
                "coder",
                f"Using deterministic cohort-definition overlap script for {step.step_id}.",
                run_id=run_id,
                step_id=step.step_id,
                current_step=step_current,
                total_steps=total_steps,
                fallback_reason=reason,
            )
            return cohort_definition_overlap_code()

        def _deterministic_cohort_definition_sensitivity_code(
            reason: str,
            *,
            preflight: bool = False,
        ) -> Optional[str]:
            nonlocal deterministic_fallback_used
            if (
                deterministic_fallback_used
                or not pipeline._enable_deterministic_runner_repair
                or (
                    preflight
                    and not _cohort_definition_sensitivity_preflight_supported()
                )
            ):
                return None
            if not _cohort_definition_sensitivity_preflight_supported():
                return None
            deterministic_fallback_used = True
            step_record["deterministic_code_fallback"] = reason
            step_record["deterministic_standard_analysis"] = (
                "cohort_definition_sensitivity"
            )
            emit_progress(
                "coder",
                f"Using deterministic cohort-definition sensitivity script for {step.step_id}.",
                run_id=run_id,
                step_id=step.step_id,
                current_step=step_current,
                total_steps=total_steps,
                fallback_reason=reason,
            )
            return cohort_definition_sensitivity_comparison_code()

        # ``--resume-from-step-id`` means the selected step is intentionally
        # rerun. Completed predecessors stay checkpointed, but the selected
        # step must not silently reuse its old script before the current
        # coder/deterministic-standard path has a chance to run.
        resume_summary_repair_code = _resume_summary_repair_code()
        preflight_resumed_code = None
        if (
            resume_summary_repair_code is None
            and requested_resume_from_step_id != step.step_id
        ):
            preflight_resumed_code = resume_controller.prior_code_for_step(step.step_id)
        if resume_summary_repair_code is not None:
            code = resume_summary_repair_code
        elif preflight_resumed_code is not None:
            code = _use_resumed_code(preflight_resumed_code)
        elif (
            _preflight_survival_code := _deterministic_survival_primary_analysis_code(
                "survival_primary_analysis_preflight", preflight=True
            )
        ) is not None:
            # The PRIMARY time-to-event result runs deterministically (Cox +
            # KM data, correct exposure, no figures) so the survival estimand
            # does not vary run-to-run through the LLM coder.
            code = _preflight_survival_code
            with shared_lock:
                findings.append(
                    ValidationFinding(
                        validator="coder",
                        severity="info",
                        message=(
                            "Using deterministic Cox survival runner before "
                            f"requesting new coder code for step {step.step_id}."
                        ),
                        detail={"step_id": step.step_id},
                    )
                )
        elif (
            _preflight_causal_code := _deterministic_causal_primary_analysis_code(
                "causal_primary_analysis_preflight", preflight=True
            )
        ) is not None:
            # The PRIMARY causal-inference result runs deterministically (IPTW
            # propensity + weighted marginal OR + balance data, no figures) so the
            # causal estimand does not vary run-to-run through the LLM coder.
            code = _preflight_causal_code
            with shared_lock:
                findings.append(
                    ValidationFinding(
                        validator="coder",
                        severity="info",
                        message=(
                            "Using deterministic IPTW causal runner before "
                            f"requesting new coder code for step {step.step_id}."
                        ),
                        detail={"step_id": step.step_id},
                    )
                )
        elif (
            _preflight_ordinal_code := _deterministic_ordinal_dose_response_code(
                "ordinal_dose_response_preflight", preflight=True
            )
        ) is not None:
            # The PRIMARY dose-response result runs deterministically (adjusted
            # trend OR per +1 stage + per-stage forest, no figures) so the
            # graded-exposure estimand does not vary run-to-run through the LLM
            # coder.
            code = _preflight_ordinal_code
            with shared_lock:
                findings.append(
                    ValidationFinding(
                        validator="coder",
                        severity="info",
                        message=(
                            "Using deterministic ordinal dose-response runner "
                            f"before requesting new coder code for step {step.step_id}."
                        ),
                        detail={"step_id": step.step_id},
                    )
                )
        else:
            preflight_figure_code = _deterministic_publication_figure_code(
                "publication_figure_parent_outputs_preflight",
                preflight=True,
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
                preflight_overlap_code = _deterministic_cohort_definition_overlap_code(
                    "cohort_definition_overlap_preflight",
                    preflight=True,
                )
                if preflight_overlap_code is not None:
                    code = preflight_overlap_code
                    with shared_lock:
                        findings.append(
                            ValidationFinding(
                                validator="coder",
                                severity="info",
                                message=(
                                    "Using deterministic cohort-definition "
                                    "overlap analysis before requesting new "
                                    f"coder code for step {step.step_id}."
                                ),
                                detail={"step_id": step.step_id},
                            )
                        )
                else:
                    preflight_sensitivity_code = (
                        _deterministic_cohort_definition_sensitivity_code(
                            "cohort_definition_sensitivity_preflight",
                            preflight=True,
                        )
                    )
                    if preflight_sensitivity_code is not None:
                        code = preflight_sensitivity_code
                        with shared_lock:
                            findings.append(
                                ValidationFinding(
                                    validator="coder",
                                    severity="info",
                                    message=(
                                        "Using deterministic cohort-definition "
                                        "sensitivity comparison before requesting "
                                        f"new coder code for step {step.step_id}."
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
                            code = coder.run(context=agent_context, step=step)
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
                                                    f"Coder agent failed for figure step {step.step_id}; "
                                                    "using deterministic publication-figure renderer "
                                                    "from parent outputs."
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
            fallback_coder = CoderAgent(MockLLMClient(context=agent_context))
            return fallback_coder.run(context=agent_context, step=step)

        concept_repair_attempts = 0
        concept_audit_error_count = 0
        deterministic_concept_repairs = 0
        _MAX_DETERMINISTIC_CONCEPT_REPAIRS = 3
        applied_concept_repair_names: List[str] = []
        while True:
            usage_findings = usage_auditor.audit(
                context=context,
                script_text=code,
                step=step,
            )
            # O-generic: analysis-pattern auditor (clustering /
            # prediction / survival footguns). Runs alongside the
            # concept-usage auditor so both sets of findings are
            # merged before the error-gate decision.
            usage_findings.extend(
                pattern_auditor.audit(
                    context=context,
                    script_text=code,
                    step=step,
                )
            )
            if pipeline._enable_llm_concept_audit and (
                resumed_code_reuse_used or deterministic_fallback_used
            ):
                usage_findings.append(
                    ValidationFinding(
                        validator="llm_concept_auditor",
                        severity="warning",
                        message=(
                            f"Skipped optional LLM concept audit for deterministic "
                            f"or resumed code in step {step.step_id}; deterministic "
                            "audits still ran."
                        ),
                        detail={
                            "step_id": step.step_id,
                            "generation_mode": (
                                "resumed_code_reuse"
                                if resumed_code_reuse_used
                                else "deterministic_fallback"
                            ),
                        },
                    )
                )
            elif pipeline._enable_llm_concept_audit:
                llm_audit_client = (
                    pipeline._llm_concept_auditor_client or role_resolver("analyzer")
                )
                if llm_audit_client is not None:
                    usage_findings.extend(
                        LLMConceptAuditor(llm_audit_client).audit(
                            context=context,
                            script_text=code,
                            step=step,
                        )
                    )
            step_record["usage_findings"] = [f.model_dump() for f in usage_findings]
            concept_audit_error_count += sum(
                1
                for f in usage_findings
                if f.validator == usage_auditor.name and f.severity == "error"
            )
            step_record["concept_audit_error_count"] = concept_audit_error_count
            step_record["concept_repair_attempts"] = concept_repair_attempts
            if not any(f.severity == "error" for f in usage_findings):
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
                    # Surface the pattern/concept findings that
                    # forced the fallback; otherwise the manifest
                    # silently drops the original ICU rule
                    # violations that the LLM emitted. We dedupe by
                    # message so repeated retries don't spam.
                    with shared_lock:
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
                    code = fallback_code
                    continue
                step_record["status"] = "blocked_by_concept_audit"
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
                    per_step_records.append(step_record)
                    _flush_partial_manifest()
                emit_progress(
                    "audit",
                    f"Concept audit blocked {step.step_id}; " f"repair ticket written.",
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
                code = coder.repair(
                    context=agent_context,
                    step=step,
                    code=code,
                    run_log=(
                        "Static concept audit blocked this script before "
                        "execution. Fix all ICU-rule violations.\n\n" + audit_log
                    ),
                    attempt=concept_repair_attempts,
                )
            except Exception as exc:
                fallback_code = _deterministic_fallback_code("concept_repair_failed")
                if fallback_code is not None:
                    code = fallback_code
                    continue
                with shared_lock:
                    findings.extend(usage_findings)
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

        repair_attempts = 0
        # A runtime crash (returncode != 0) is a distinct, always-actionable
        # failure class (a real Python traceback) and gets its own repair
        # budget. Otherwise a success-path repair (contract / visual QA) that
        # *introduces* a crash could consume the only shared attempt, leaving
        # nothing to fix the traceback — the step would fail-closed even though
        # the analysis it produced (e.g. the primary OR) was already valid.
        runtime_repair_attempts = 0
        runner_repair_name: Optional[str] = preexecution_runner_repair_name
        while True:
            run_label = "repaired script" if repair_attempts else "generated script"
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

            script_description = (
                f"Generated analysis script for step {step.step_id}."
                if repair_attempts == 0
                else f"Repaired analysis script for step {step.step_id} (attempt {repair_attempts})."
            )
            script_record = evidence.register_file(
                kind="code",
                description=script_description,
                source_path=run_result.script_path,
                produced_by_step=step.step_id,
                producer="coder",
                generation_mode=_script_generation_mode(
                    repair_attempts=repair_attempts,
                    fallback_used=deterministic_fallback_used,
                    runner_repair_name=runner_repair_name,
                    resumed_code_reuse=resumed_code_reuse_used,
                ),
                prompt_pack_version=prompt_version,
                metadata={
                    "repair_attempts": repair_attempts,
                    "fallback_reason": step_record.get("deterministic_code_fallback"),
                    "runner_repair": runner_repair_name,
                    "resumed_code_evidence_id": step_record.get(
                        "resumed_code_evidence_id"
                    ),
                    "resumed_code_relative_path": step_record.get(
                        "resumed_code_relative_path"
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
                    generation_mode=_script_generation_mode(
                        repair_attempts=repair_attempts,
                        fallback_used=deterministic_fallback_used,
                        runner_repair_name=runner_repair_name,
                        resumed_code_reuse=resumed_code_reuse_used,
                    ),
                    metadata={
                        "repair_attempts": repair_attempts,
                        "fallback_reason": step_record.get(
                            "deterministic_code_fallback"
                        ),
                        "runner_repair": runner_repair_name,
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
                        if repair_attempts >= pipeline._max_code_repair_attempts:
                            fallback_code = _deterministic_fallback_code("visual_qa")
                            if fallback_code is not None:
                                code = fallback_code
                                _clear_output_dir(run_result.out_dir)
                                continue
                            demoted_findings, blocking_visual_errors = (
                                _demote_cosmetic_visual_findings(visual_findings)
                            )
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
                                        f"{repair_attempts} repair attempts."
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
                                    f"{repair_attempts} repair attempts."
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
                            repair_attempts += 1
                            step_record["code_repair_attempts"] = repair_attempts
                            emit_progress(
                                "visual_qa",
                                f"Repairing figure layout for {step.step_id}.",
                                run_id=run_id,
                                step_id=step.step_id,
                                current_step=step_current,
                                total_steps=total_steps,
                                repair_attempts=repair_attempts,
                            )
                            qa_log = "\n".join(
                                f"{f.severity.upper()}: {f.message}"
                                for f in visual_findings
                            )
                            try:
                                code = coder.repair(
                                    context=agent_context,
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
                                    attempt=repair_attempts,
                                )
                                _clear_output_dir(run_result.out_dir)
                                continue
                            except Exception as exc:
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
                                                f"Coder repair failed after visual QA "
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
                    else:
                        summary_repair = None
                    if summary_repair is not None:
                        repair_attempts += 1
                        runner_repair_name, code = summary_repair
                        step_record["runner_repair"] = runner_repair_name
                        step_record["code_repair_attempts"] = repair_attempts
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
                    else:
                        contract_repair = None
                    if contract_repair is not None:
                        repair_attempts += 1
                        runner_repair_name, code = contract_repair
                        step_record["runner_repair"] = runner_repair_name
                        step_record["code_repair_attempts"] = repair_attempts
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
                    if repair_attempts >= pipeline._max_code_repair_attempts:
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

                    repair_attempts += 1
                    step_record["code_repair_attempts"] = repair_attempts
                    emit_progress(
                        "coder",
                        f"Repairing contract violation for {step.step_id}.",
                        run_id=run_id,
                        step_id=step.step_id,
                        current_step=step_current,
                        total_steps=total_steps,
                        repair_attempts=repair_attempts,
                    )
                    contract_log = "\n".join(
                        f"{f.severity.upper()}: {f.message}"
                        for f in early_contract_findings
                        if f.message
                    )
                    repair_guidance = _step_contract_repair_guidance(
                        step=step,
                        step_summary=visual_step_summary,
                        code=code,
                    )
                    try:
                        code = coder.repair(
                            context=agent_context,
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
                                + "\n\nCONTRACT FINDINGS:\n"
                                + contract_log
                                + "\n\nREPAIR GUIDANCE:\n"
                                + repair_guidance
                            ),
                            attempt=repair_attempts,
                        )
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
                    context=agent_context,
                    step=step,
                    code=code,
                    run_log=run_log,
                    attempt=repair_attempts,
                )
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

        publication_step = (
            step.method == "publication_figure_generation"
            or "publication_figure" in (step.step_id or "").lower()
            or "publication figure" in (step.intent or "").lower()
            or "publication-ready figure" in (step.intent or "").lower()
            or any(
                str(item).startswith("figure:publication_figure")
                for item in step.expected_outputs
            )
        ) and not step_record.get("deterministic_standard_analysis")
        # A step that ran a deterministic DATA-only runner (survival Cox / KM
        # tables, cohort overlap, cohort sensitivity) produces CSV evidence, not
        # an inline figure — the separate ``*_figure`` step renders it. The
        # planner often narrates "the publication figure" in such a step's intent
        # (describing that downstream figure step), which would otherwise mis-tag
        # the analysis step as a publication-figure step and fail it for a missing
        # inline figure — killing the primary Cox result and cascading to skip the
        # real figure step (H1 fix4: 01_survival_analysis produced HR 1.83 then was
        # marked failed). Genuine publication-figure steps carry no such marker and
        # still fail closed when they emit no figure.
        figure_role = (
            "publication_figure"
            if publication_step
            else "analysis_figure" if _step_expects_figure(step) else None
        )
        if publication_step and not _has_figure_exports(run_result.out_dir):
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
                rescued = _render_publication_bundle_from_prior_outputs_for_step(
                    run_dir=run_dir,
                    current_step_id=step.step_id,
                    out_dir=run_result.out_dir,
                    step_text=f"{step.intent} {step.method}",
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
                    promotion_text = (
                        f"{step.step_id} {step.intent} {step.method}".lower()
                    )
                    association_promotion_roles: Optional[Sequence[str]] = None
                    if any(
                        token in promotion_text
                        for token in (
                            "association",
                            "odds",
                            "effect",
                            "forest",
                            "primary_result",
                            "primary results",
                            "main_result",
                            "main results",
                        )
                    ):
                        association_promotion_roles = (
                            "descriptive_result",
                            "primary_estimand",
                        )
                    promoted = _promote_prior_publication_bundle(
                        run_dir=run_dir,
                        current_step_id=step.step_id,
                        out_dir=run_result.out_dir,
                        required_roles=association_promotion_roles,
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
            )
            if art.name == "step_summary.json":
                rec = evidence.register_file(
                    kind="statistic",
                    description=f"Machine-readable summary for step {step.step_id}.",
                    source_path=art,
                    produced_by_step=step.step_id,
                    script_evidence_id=script_record.evidence_id,
                    aliases=step_aliases,
                    producer="runner",
                    generation_mode=generation_mode,
                    metadata={
                        "script_evidence_id": script_record.evidence_id,
                        "figure_role": figure_role or "analysis_figure",
                        "diagnostic_only": False,
                    },
                )
                step_summary_record_id = rec.evidence_id
            elif art.suffix.lower() in {".csv", ".tsv", ".parquet", ".feather"}:
                rec = evidence.register_file(
                    kind="table",
                    description=f"Table {art.stem} from step {step.step_id}.",
                    source_path=art,
                    produced_by_step=step.step_id,
                    script_evidence_id=script_record.evidence_id,
                    aliases=step_aliases,
                    producer="runner",
                    generation_mode=generation_mode,
                    metadata={"script_evidence_id": script_record.evidence_id},
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
                    script_evidence_id=script_record.evidence_id,
                    aliases=step_aliases,
                    producer="runner",
                    generation_mode=generation_mode,
                    metadata={
                        "script_evidence_id": script_record.evidence_id,
                        "figure_role": figure_role or "analysis_figure",
                        "diagnostic_only": False,
                    },
                )
            else:
                rec = evidence.register_file(
                    kind="log",
                    description=f"Auxiliary artefact {art.name}.",
                    source_path=art,
                    produced_by_step=step.step_id,
                    script_evidence_id=script_record.evidence_id,
                    aliases=step_aliases,
                    producer="runner",
                    generation_mode=generation_mode,
                    metadata={"script_evidence_id": script_record.evidence_id},
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
            )
            rec = evidence.register_file(
                kind="log",
                description=(
                    f"Auto-generated figure contract for step {step.step_id}."
                ),
                source_path=auto_contract_path,
                produced_by_step=step.step_id,
                script_evidence_id=script_record.evidence_id,
                aliases=_semantic_aliases_for(step, auto_contract_path),
                producer="runner",
                generation_mode=generation_mode,
                metadata={
                    "script_evidence_id": script_record.evidence_id,
                    "figure_role": figure_role or "analysis_figure",
                    "synthesis": "step_summary_figure_contract_v1",
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
        )
        figure_gate_errors = [
            finding
            for finding in contract_findings + figure_source_findings
            if finding.severity == "error"
            and finding.validator in {"figure_contract_quality", "figure_source_data"}
        ]
        association_publication_step = (
            publication_step
            and "association" in f"{step.step_id} {step.intent} {step.method}".lower()
        )
        sensitivity_publication_step = publication_step and any(
            token in f"{step.step_id} {step.intent} {step.method}".lower()
            for token in ("sensitivity", "robustness")
        )
        cohort_publication_step = publication_step and any(
            token in f"{step.step_id} {step.intent} {step.method}".lower()
            for token in ("cohort", "eligibility", "overlap", "attrition", "definition")
        )
        missingness_publication_step = publication_step and any(
            token in f"{step.step_id} {step.intent} {step.method}".lower()
            for token in ("missingness", "measurement", "data_quality", "quality")
        )
        if (
            association_publication_step
            or sensitivity_publication_step
            or cohort_publication_step
            or missingness_publication_step
        ) and figure_gate_errors:
            _clear_output_dir(run_result.out_dir)
            repaired = _render_publication_bundle_from_prior_outputs_for_step(
                run_dir=run_dir,
                current_step_id=step.step_id,
                out_dir=run_result.out_dir,
                step_text=f"{step.intent} {step.method}",
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
        step_record["generation_mode"] = _script_generation_mode(
            repair_attempts=repair_attempts,
            fallback_used=deterministic_fallback_used,
            runner_repair_name=runner_repair_name,
            resumed_code_reuse=resumed_code_reuse_used,
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
        if resumed_code_reuse_used or deterministic_fallback_used:
            mode_label = (
                "resumed agent-generated code"
                if resumed_code_reuse_used
                else "deterministic fallback code"
            )
            interpretation = (
                f"Step `{step.step_id}` was executed from {mode_label}. "
                "Review the registered step summary and artefacts for numeric "
                "interpretation; no new LLM interpretation was requested."
            )
            interp_generation_mode = (
                "resumed_code_reuse"
                if resumed_code_reuse_used
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
        has_contract_error = any(
            finding.severity == "error"
            for finding in contract_findings + figure_source_findings
        )
        step_record["status"] = "contract_failed" if has_contract_error else "ok"
        with shared_lock:
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
                else f"Step {step_current}/{total_steps} complete: {step.step_id}."
            ),
            status="error" if has_contract_error else "complete",
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
