"""Article-level analysis contracts for research-agent runs.

The study-design brief teaches the planner what a journal-style analysis
usually needs. This module turns that guidance into a reusable contract that
can be checked at three boundaries:

* before planning, as a compact prompt block;
* after planning, as role/module coverage;
* after execution/readiness, against registered artifacts and figure contracts.

The contract is deliberately case-neutral. It encodes display roles such as
cohort accounting, data quality, primary estimand, calibration, robustness, and
transportability; it does not name one benchmark variable or database.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set

from pydantic import BaseModel, ConfigDict, Field

from .schema import AnalysisPlan, AnalysisStep, ResearchContext, ValidationFinding
from .study_design import StudyDesignBrief, build_study_design_brief
from .study_design_playbook import (
    DisplayModuleSpec,
    DisplayTier,
    StudyDesignFamily,
    role_check_terms,
)


ARTICLE_ANALYSIS_CONTRACT_SCHEMA_VERSION = "easyicu.article_analysis_contract/1"
ARTICLE_CONTRACT_AUDIT_SCHEMA_VERSION = "easyicu.article_contract_audit/1"

_REQUIRED_TIERS: Set[DisplayTier] = {"core", "conditional"}
_COUNTED_ARTIFACT_KINDS = {"table", "figure", "statistic"}

_ROLE_ALIASES: Dict[str, Sequence[str]] = {
    "cohort_accounting": ("cohort flow", "attrition", "eligibility", "denominator"),
    "baseline_context": ("table 1", "table_one", "baseline characteristics"),
    "data_quality": ("audit", "missingness", "measurement", "coverage", "quality"),
    "primary_estimand": (
        "relationship",
        "association",
        "adjusted estimate",
        "effect estimate",
        "forest plot",
    ),
    "robustness": ("sensitivity", "robustness", "specification", "alternative"),
    "validation": ("validation", "external validation", "train-test"),
    "model_performance": ("roc", "auroc", "discrimination", "precision-recall"),
    "calibration": ("calibration", "brier"),
    "temporal_absolute_risk": ("kaplan", "risk table", "cumulative incidence"),
    "survival_effect": ("hazard ratio", "cox", "survival contrast"),
    "diagnostics": ("diagnostic", "assumption", "censoring"),
    "phenotype_structure": ("embedding", "umap", "pca", "cluster heatmap"),
    "phenotype_profile": ("phenotype profile", "cluster characteristics"),
    "stability": ("stability", "bootstrap", "consensus"),
    "causal_protocol": ("target trial", "time zero", "estimand"),
    "balance_positivity": ("balance", "positivity", "weight distribution"),
    "causal_contrast": ("causal contrast", "iptw", "g-computation"),
    "distribution": ("distribution", "prevalence", "density"),
    "descriptive_result": (
        "prevalence",
        "incidence",
        "event rate",
        "outcome by exposure",
        "outcome-by-exposure",
    ),
    "transportability": (
        "cross database",
        "cross-database",
        "database-specific",
        "site-specific",
        "transportability",
    ),
}


class ArticleDisplayRequirement(BaseModel):
    model_config = ConfigDict(extra="forbid")

    module_id: str
    role: str
    tier: DisplayTier
    required: bool = True
    rationale: str
    acceptable_outputs: List[str] = Field(default_factory=list)
    search_terms: List[str] = Field(default_factory=list)


class ArticleAnalysisContract(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = ARTICLE_ANALYSIS_CONTRACT_SCHEMA_VERSION
    analysis_family: StudyDesignFamily
    reporting_guidelines: List[str] = Field(default_factory=list)
    requirements: List[ArticleDisplayRequirement] = Field(default_factory=list)
    required_roles: List[str] = Field(default_factory=list)
    recommended_roles: List[str] = Field(default_factory=list)
    minimum_required_role_count: int = 0
    anti_patterns: List[str] = Field(default_factory=list)
    design_reference_queries: List[str] = Field(default_factory=list)
    source_brief_schema_version: str = ""


def _normalise_space(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip().lower())


def _requirement_terms(module: DisplayModuleSpec) -> List[str]:
    raw_terms: List[str] = [
        module.module_id,
        module.module_id.replace("_", " "),
        module.role,
        module.role.replace("_", " "),
        module.rationale,
        *module.acceptable_outputs,
        *role_check_terms(module.role),
        *_ROLE_ALIASES.get(module.role, ()),
    ]
    terms: List[str] = []
    seen: Set[str] = set()
    for term in raw_terms:
        cleaned = _normalise_space(term)
        if not cleaned or cleaned in seen:
            continue
        terms.append(cleaned)
        seen.add(cleaned)
    return terms


def build_article_analysis_contract(
    context: ResearchContext,
    *,
    brief: Optional[StudyDesignBrief] = None,
) -> ArticleAnalysisContract:
    """Build the article-level output contract for a research context."""

    resolved_brief = brief or build_study_design_brief(context)
    requirements: List[ArticleDisplayRequirement] = []
    required_roles: Set[str] = set()
    recommended_roles: Set[str] = set()
    for module in resolved_brief.display_modules:
        if module.tier == "supplementary":
            continue
        required = module.tier in _REQUIRED_TIERS
        requirement = ArticleDisplayRequirement(
            module_id=module.module_id,
            role=module.role,
            tier=module.tier,
            required=required,
            rationale=module.rationale,
            acceptable_outputs=list(module.acceptable_outputs),
            search_terms=_requirement_terms(module),
        )
        requirements.append(requirement)
        if required:
            required_roles.add(module.role)
        else:
            recommended_roles.add(module.role)
    return ArticleAnalysisContract(
        analysis_family=resolved_brief.analysis_family,
        reporting_guidelines=list(resolved_brief.reporting_guidelines),
        requirements=requirements,
        required_roles=sorted(required_roles),
        recommended_roles=sorted(recommended_roles),
        minimum_required_role_count=len(required_roles),
        anti_patterns=list(resolved_brief.anti_patterns),
        design_reference_queries=list(resolved_brief.exemplar_search_queries),
        source_brief_schema_version=resolved_brief.schema_version,
    )


def render_article_analysis_contract_for_prompt(
    contract: ArticleAnalysisContract,
) -> str:
    """Render a compact, planner-facing contract block."""

    required = [req for req in contract.requirements if req.required]
    recommended = [req for req in contract.requirements if not req.required]
    lines = [
        "ARTICLE ANALYSIS CONTRACT:",
        f"- analysis_family: {contract.analysis_family}",
        "- reporting_guidelines: " + "; ".join(contract.reporting_guidelines),
        "- required_article_roles: " + ", ".join(contract.required_roles),
        "- required_modules:",
    ]
    for req in required:
        lines.append(
            "  - "
            f"{req.module_id} (role={req.role}; tier={req.tier}; "
            f"acceptable={', '.join(req.acceptable_outputs[:4])})"
        )
    if recommended:
        lines.append(
            "- recommended_roles: "
            + ", ".join(f"{req.module_id}:{req.role}" for req in recommended)
        )
    if contract.design_reference_queries:
        lines.append(
            "- design_reference_queries: "
            + "; ".join(contract.design_reference_queries[:3])
        )
    lines.append(
        "- rule: a technically valid single result figure is insufficient unless "
        "the artifact suite covers the required article roles."
    )
    return "\n".join(lines)


def _step_text(step: AnalysisStep) -> str:
    return _normalise_space(
        "\n".join(
            [
                step.step_id,
                step.intent,
                step.method or "",
                " ".join(step.inputs or []),
                " ".join(step.expected_outputs or []),
                " ".join(step.icu_rule_refs or []),
            ]
        )
    )


def _text_matches_requirement(text: str, requirement: ArticleDisplayRequirement) -> bool:
    haystack = _normalise_space(text)
    return any(term and term in haystack for term in requirement.search_terms)


def roles_covered_by_plan(
    plan: Optional[AnalysisPlan],
    contract: ArticleAnalysisContract,
) -> Set[str]:
    if plan is None:
        return set()
    texts = [_step_text(step) for step in plan.steps or []]
    covered: Set[str] = set()
    for requirement in contract.requirements:
        if any(_text_matches_requirement(text, requirement) for text in texts):
            covered.add(requirement.role)
    return covered


def validate_plan_against_article_contract(
    *,
    plan: Optional[AnalysisPlan],
    contract: ArticleAnalysisContract,
) -> List[ValidationFinding]:
    covered_roles = roles_covered_by_plan(plan, contract)
    required_roles = set(contract.required_roles)
    missing_roles = sorted(required_roles - covered_roles)
    if not missing_roles:
        return []
    missing_modules = [
        req.module_id
        for req in contract.requirements
        if req.required and req.role in missing_roles
    ]
    return [
        ValidationFinding(
            validator="article_analysis_contract",
            severity="warning",
            message=(
                "Analysis plan does not cover all article-level roles required "
                f"for {contract.analysis_family} studies."
            ),
            detail={
                "analysis_family": contract.analysis_family,
                "covered_roles": sorted(covered_roles),
                "missing_roles": missing_roles,
                "missing_modules": missing_modules,
            },
        )
    ]


def _record_to_text(record: Any) -> str:
    parts = [
        getattr(record, "evidence_id", ""),
        getattr(record, "kind", ""),
        getattr(record, "description", ""),
        getattr(record, "relative_path", ""),
        getattr(record, "produced_by_step", ""),
    ]
    metadata = getattr(record, "metadata", None)
    if metadata:
        parts.append(json.dumps(metadata, ensure_ascii=False, default=str))
    return _normalise_space("\n".join(str(part or "") for part in parts))


def _step_summary_text(record: Mapping[str, Any]) -> str:
    if record.get("status") != "ok":
        return ""
    summary = record.get("step_summary")
    if not summary:
        return ""
    return _normalise_space(json.dumps(summary, ensure_ascii=False, default=str))


def _figure_contract_paths(run_dir: Path) -> List[Path]:
    candidates = [
        *run_dir.glob("publication_figures/*.figure_contract.json"),
        *run_dir.glob("steps/*/outputs/*.figure_contract.json"),
    ]
    seen: Set[str] = set()
    unique: List[Path] = []
    for path in sorted(candidates):
        key = str(path.resolve())
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    return unique


def _figure_contract_text(path: Path) -> str:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return ""
    if not isinstance(raw, dict):
        return ""
    parts: List[str] = [
        str(raw.get("figure_id") or ""),
        str(raw.get("title") or ""),
        str(raw.get("core_claim") or ""),
        str(raw.get("statistics_note") or ""),
    ]
    panels = raw.get("panels")
    if isinstance(panels, list):
        for panel in panels:
            if not isinstance(panel, dict):
                continue
            parts.extend(
                [
                    str(panel.get("panel_id") or ""),
                    str(panel.get("title") or ""),
                    str(panel.get("role") or ""),
                    str(panel.get("claim") or ""),
                    str(panel.get("review_risk") or ""),
                ]
            )
    return _normalise_space("\n".join(parts))


def _artifact_texts(
    *,
    evidence_records: Sequence[Any],
    per_step_records: Sequence[Mapping[str, Any]],
    run_dir: Path,
) -> List[str]:
    texts: List[str] = []
    for record in evidence_records:
        kind = str(getattr(record, "kind", "") or "")
        if kind in _COUNTED_ARTIFACT_KINDS:
            texts.append(_record_to_text(record))
    for record in per_step_records:
        text = _step_summary_text(record)
        if text:
            texts.append(text)
    for path in _figure_contract_paths(run_dir):
        text = _figure_contract_text(path)
        if text:
            texts.append(text)
    return texts


def roles_covered_by_artifacts(
    *,
    contract: ArticleAnalysisContract,
    evidence_records: Sequence[Any],
    per_step_records: Sequence[Mapping[str, Any]],
    run_dir: Path,
) -> Set[str]:
    texts = _artifact_texts(
        evidence_records=evidence_records,
        per_step_records=per_step_records,
        run_dir=run_dir,
    )
    covered: Set[str] = set()
    for requirement in contract.requirements:
        if any(_text_matches_requirement(text, requirement) for text in texts):
            covered.add(requirement.role)
    return covered


def summarize_article_contract_coverage(
    *,
    context: ResearchContext,
    plan: Optional[AnalysisPlan],
    evidence_records: Sequence[Any],
    per_step_records: Sequence[Mapping[str, Any]],
    run_dir: Path,
) -> Dict[str, Any]:
    contract = build_article_analysis_contract(context)
    plan_roles = roles_covered_by_plan(plan, contract)
    artifact_roles = roles_covered_by_artifacts(
        contract=contract,
        evidence_records=evidence_records,
        per_step_records=per_step_records,
        run_dir=run_dir,
    )
    required_roles = set(contract.required_roles)
    missing_plan_roles = sorted(required_roles - plan_roles)
    missing_artifact_roles = sorted(required_roles - artifact_roles)
    missing_artifact_modules = [
        req.module_id
        for req in contract.requirements
        if req.required and req.role in missing_artifact_roles
    ]
    errors: List[str] = []
    if missing_artifact_roles:
        errors.append(
            "Missing required article artifact role(s): "
            + ", ".join(missing_artifact_roles)
        )
    if len(artifact_roles & required_roles) < contract.minimum_required_role_count:
        errors.append(
            "Artifact suite covers fewer required article roles than the "
            f"{contract.analysis_family} contract expects."
        )
    return {
        "article_contract_audit_schema_version": ARTICLE_CONTRACT_AUDIT_SCHEMA_VERSION,
        "article_contract_complete": not errors,
        "article_contract_family": contract.analysis_family,
        "article_required_roles": sorted(required_roles),
        "article_plan_roles": sorted(plan_roles),
        "article_artifact_roles": sorted(artifact_roles),
        "article_missing_plan_roles": missing_plan_roles,
        "article_missing_artifact_roles": missing_artifact_roles,
        "article_missing_artifact_modules": missing_artifact_modules,
        "article_contract_errors": errors,
        "article_contract": contract.model_dump(mode="json"),
    }


def validate_run_against_article_contract(
    *,
    context: ResearchContext,
    plan: Optional[AnalysisPlan],
    evidence_records: Sequence[Any],
    per_step_records: Sequence[Mapping[str, Any]],
    run_dir: Path,
) -> List[ValidationFinding]:
    status = summarize_article_contract_coverage(
        context=context,
        plan=plan,
        evidence_records=evidence_records,
        per_step_records=per_step_records,
        run_dir=run_dir,
    )
    if status["article_contract_complete"]:
        return []
    return [
        ValidationFinding(
            validator="article_analysis_contract",
            severity="warning",
            message=(
                "Run artifacts do not yet satisfy the article-level analysis "
                f"contract for {status['article_contract_family']} studies."
            ),
            detail={
                "missing_artifact_roles": status["article_missing_artifact_roles"],
                "missing_artifact_modules": status[
                    "article_missing_artifact_modules"
                ],
                "artifact_roles": status["article_artifact_roles"],
            },
        )
    ]


def _iter_missing_requirements(
    contract: ArticleAnalysisContract,
    covered_roles: Iterable[str],
) -> List[ArticleDisplayRequirement]:
    covered = set(covered_roles)
    missing: List[ArticleDisplayRequirement] = []
    seen_roles: Set[str] = set()
    for requirement in contract.requirements:
        if not requirement.required or requirement.role in covered:
            continue
        if requirement.role in seen_roles:
            continue
        missing.append(requirement)
        seen_roles.add(requirement.role)
    return missing


def _slug(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip().lower()).strip("_")
    return cleaned or "article_display"


def _unique_step_id(base: str, used: Set[str]) -> str:
    candidate = base
    suffix = 2
    while candidate in used:
        candidate = f"{base}_{suffix}"
        suffix += 1
    used.add(candidate)
    return candidate


def _expected_outputs_for_requirement(
    requirement: ArticleDisplayRequirement,
) -> List[str]:
    joined = " ".join(requirement.acceptable_outputs).lower()
    output_kind = "figure" if any(
        token in joined
        for token in ("figure", "plot", "curve", "heatmap", "panel", "diagram")
    ) else "table"
    if requirement.role in {"primary_estimand", "causal_contrast", "survival_effect"}:
        output_kind = "table"
    if requirement.role in {
        "model_performance",
        "calibration",
        "temporal_absolute_risk",
        "phenotype_structure",
        "robustness",
    }:
        output_kind = "figure"
    return [f"{output_kind}:{_slug(requirement.module_id)}"]


def augment_plan_for_article_contract(
    *,
    plan: AnalysisPlan,
    contract: ArticleAnalysisContract,
) -> tuple[AnalysisPlan, List[ValidationFinding]]:
    """Return a plan with missing article-role display steps appended.

    This helper is intentionally pure and opt-in. The main pipeline can use it
    when it wants deterministic expansion; tests and callers can use it to
    verify that a narrow plan is not treated as article-complete.
    """

    covered_roles = roles_covered_by_plan(plan, contract)
    missing = _iter_missing_requirements(contract, covered_roles)
    if not missing:
        return plan, []
    used_ids = {step.step_id for step in plan.steps or []}
    new_steps = list(plan.steps or [])
    base_index = len(new_steps) + 1
    for offset, requirement in enumerate(missing):
        step_id = _unique_step_id(
            f"{base_index + offset:02d}_{_slug(requirement.module_id)}",
            used_ids,
        )
        new_steps.append(
            AnalysisStep(
                step_id=step_id,
                intent=(
                    f"Produce the article-facing {requirement.role} display "
                    f"required by the {contract.analysis_family} contract: "
                    f"{requirement.rationale}"
                ),
                inputs=[],
                expected_outputs=_expected_outputs_for_requirement(requirement),
                method="article_contract_display",
                icu_rule_refs=[],
            )
        )
    revised = plan.model_copy(
        update={
            "steps": new_steps,
            "revision": max(1, plan.revision) + 1,
            "rationale": (
                (plan.rationale or "").rstrip()
                + "\n\nArticle contract augmentation added missing display roles: "
                + ", ".join(req.role for req in missing)
            ).strip(),
        }
    )
    finding = ValidationFinding(
        validator="article_analysis_contract",
        severity="info",
        message=(
            "Augmented analysis plan with missing article-level display roles "
            f"for {contract.analysis_family} studies."
        ),
        detail={
            "added_roles": [req.role for req in missing],
            "added_modules": [req.module_id for req in missing],
            "added_step_ids": [step.step_id for step in new_steps[-len(missing):]],
        },
    )
    return revised, [finding]


__all__ = [
    "ARTICLE_ANALYSIS_CONTRACT_SCHEMA_VERSION",
    "ARTICLE_CONTRACT_AUDIT_SCHEMA_VERSION",
    "ArticleAnalysisContract",
    "ArticleDisplayRequirement",
    "augment_plan_for_article_contract",
    "build_article_analysis_contract",
    "render_article_analysis_contract_for_prompt",
    "roles_covered_by_artifacts",
    "roles_covered_by_plan",
    "summarize_article_contract_coverage",
    "validate_plan_against_article_contract",
    "validate_run_against_article_contract",
]
