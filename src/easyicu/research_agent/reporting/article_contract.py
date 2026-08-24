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
import hashlib
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set

from pydantic import BaseModel, ConfigDict, Field

from ..figures.contracts import figure_contract_paths
from ..authority.runtime_artifacts import (
    current_evidence_records,
    current_successful_step_records,
    verified_run_evidence_path,
)
from ..authority.planned_role import (
    unique_verified_primary_record,
    verified_planned_analysis_role,
)
from ..schema import AnalysisPlan, AnalysisStep, ResearchContext, ValidationFinding
from ..contracts.declared_product import typed_product
from ..planning.study_design import (
    StudyDesignBrief,
    _declaration_matches_term,
    _structured_plan_declarations,
    build_study_design_brief,
)
from ..planning.analysis_types import canonical_analysis_family, infer_analysis_type
from ..planning.scientific_action_catalog import scientific_action_for_id
from ..planning.study_design_playbook import (
    DisplayModuleSpec,
    DisplayTier,
    StudyDesignFamily,
    primary_result_roles_for_analysis_type,
    role_check_terms,
)

ARTICLE_ANALYSIS_CONTRACT_SCHEMA_VERSION = "easyicu.article_analysis_contract/2"
ARTICLE_CONTRACT_AUDIT_SCHEMA_VERSION = "easyicu.article_contract_audit/1"

_REQUIRED_TIERS: Set[DisplayTier] = {"core", "conditional"}
_COUNTED_ARTIFACT_KINDS = {"table", "figure", "statistic"}
_AMBIGUOUS_ARTIFACT_ROLE_TERMS: Dict[str, Set[str]] = {
    # These words are useful while matching structured plan declarations, but
    # far too broad for artifact prose (for example, "model specification" in
    # a review-risk sentence is not a completed sensitivity analysis).
    "robustness": {"variant", "alternative", "specification"},
}

_ROLE_ALIASES: Dict[str, Sequence[str]] = {
    "cohort_accounting": (
        "cohort flow",
        "attrition",
        "eligibility",
        "denominator",
        "cohort_attrition",
    ),
    "baseline_context": (
        "table 1",
        "table_one",
        "baseline characteristics",
        "baseline_characteristics",
    ),
    "data_quality": (
        "audit",
        "missingness",
        "measurement",
        "coverage",
        "quality",
        "feature availability",
        "feature_availability",
        "missingness_profile",
        "data_quality_source_status",
        "measurement_missingness",
        "measurement_process",
        "measurement_source",
    ),
    "primary_estimand": (
        "relationship",
        "association",
        "adjusted estimate",
        "effect estimate",
        "forest plot",
        "adjusted_association_estimates",
    ),
    # Keep these aliases role-specific. Bare words such as ``specification`` or
    # ``alternative`` occur in ordinary model-review prose and previously made
    # a two-panel primary figure falsely satisfy the robustness artifact role.
    # The requirement's acceptable outputs still recognise the precise phrases
    # ``specification curve`` and ``alternative-definition panel``.
    "robustness": (
        "sensitivity",
        "robustness",
        "robustness_matrix",
        "robustness_summary",
    ),
    "validation": ("validation", "external validation", "train-test"),
    "model_performance": ("roc", "auroc", "discrimination", "precision-recall"),
    "calibration": ("calibration", "brier"),
    "temporal_absolute_risk": ("kaplan", "risk table", "cumulative incidence"),
    "survival_effect": ("hazard ratio", "cox", "survival contrast"),
    "diagnostics": ("diagnostic", "assumption", "censoring"),
    "phenotype_structure": (
        "embedding",
        "umap",
        "pca",
        "cluster heatmap",
        "cluster assignments",
        "cluster_assignments",
    ),
    "phenotype_profile": (
        "phenotype profile",
        "phenotype_profiles",
        "trajectory profile",
        "trajectory_profiles",
        "cluster characteristics",
        "cluster summary",
        "cluster_summary",
        "cluster outcomes",
        "cluster_outcomes",
    ),
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
        "exposure_outcome_distribution",
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
    source_analysis_type: str
    planner_owned_result_roles: List[str] = Field(default_factory=list)
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
    ambiguous = _AMBIGUOUS_ARTIFACT_ROLE_TERMS.get(module.role, set())
    for term in raw_terms:
        cleaned = _normalise_space(term)
        if not cleaned or cleaned in seen or cleaned in ambiguous:
            continue
        terms.append(cleaned)
        seen.add(cleaned)
    return terms


def build_article_analysis_contract(
    context: ResearchContext,
    *,
    brief: Optional[StudyDesignBrief] = None,
    analysis_type: Optional[str] = None,
) -> ArticleAnalysisContract:
    """Build the article-level output contract for a research context."""

    source_analysis_type = canonical_analysis_family(analysis_type)
    if analysis_type is not None and source_analysis_type is None:
        raise ValueError(f"unknown analysis_type {analysis_type!r}")
    if source_analysis_type is None:
        source_analysis_type = infer_analysis_type(context).key
    resolved_brief = brief or build_study_design_brief(
        context,
        analysis_type=source_analysis_type,
    )
    requirements: List[ArticleDisplayRequirement] = []
    required_roles: Set[str] = set()
    recommended_roles: Set[str] = set()
    display_modules = list(resolved_brief.display_modules)
    if source_analysis_type == "data_quality_audit":
        # A measurement-completeness question is intentionally narrower than
        # descriptive epidemiology. Reusing the descriptive playbook must not
        # silently turn it into cohort characterization, prevalence estimation,
        # or cross-database result reporting.
        display_modules = [
            module for module in display_modules if module.role == "data_quality"
        ]
    from ..planning.dependence_authority import context_counts_only_authority

    if context_counts_only_authority(context):
        # The only typed baseline-context owner is Table One, which this
        # authority forbids. The generic distribution module is also redundant:
        # the required typed exposure/outcome product owns those counts and its
        # downstream figure owns the display. Retaining either module encourages
        # a second, untyped or mislabeled result product.
        display_modules = [
            module
            for module in display_modules
            if module.role != "baseline_context"
            and module.module_id != "distribution_prevalence"
        ]
    for module in display_modules:
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
        source_analysis_type=source_analysis_type,
        planner_owned_result_roles=sorted(
            primary_result_roles_for_analysis_type(source_analysis_type)
        ),
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
    headline_roles = set(contract.planner_owned_result_roles)
    lines = [
        "ARTICLE ANALYSIS CONTRACT:",
        f"- analysis_family: {contract.analysis_family}",
        f"- source_analysis_type: {contract.source_analysis_type}",
        "- reporting_guidelines: " + "; ".join(contract.reporting_guidelines),
        "- required_article_roles: " + ", ".join(contract.required_roles),
        "- required_modules:",
    ]
    for req in required:
        lines.append(
            "  - "
            f"{req.module_id} (role={req.role}; tier={req.tier}; "
            + ("headline_owned=true; " if req.role in headline_roles else "")
            + f"typed_example=table:{req.module_id}; "
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
    lines.append(
        "- rule: every required role must be owned by an explicit analysis step; "
        "put its typed_example (or an equally explicit typed product using an "
        "acceptable term) in expected_outputs. Intent-only prose does not count."
    )
    # The rule above is true for ordinary roles and INCOMPLETE for the
    # headline-owned ones, which roles_covered_by_plan credits only from steps
    # on the primary lineage. Measured 2026-07-30 on a recorded
    # survival-analysis failure: the scientifically natural survival plan
    # -- Cox model as the single primary step, Kaplan-Meier curve as its own
    # secondary display step reading the analysis cohort -- covers
    # survival_effect and misses temporal_absolute_risk, because the display
    # step is off the lineage. The plan schema permits at most one primary step
    # and refuses a primary step whose products are all rendering, so a Planner
    # obeying the published rule literally has no way to satisfy the role. Five
    # attempts, three distinct violations, nothing executed. Six of fifteen
    # families declare a headline-owned role whose natural output is a display,
    # so this is not one family's quirk.
    if headline_roles & set(contract.required_roles):
        lines.append(
            "- rule: a role marked headline_owned=true is credited only when the "
            "declaring step is on the primary lineage -- either the single "
            "planned_analysis_role='primary' step itself, or a "
            "secondary/auxiliary step whose inputs include a typed product that "
            "only a lineage step produces. A plan may declare at most one primary "
            "step, and that step must itself declare a non-rendering scientific "
            "result product, so a display for a headline_owned role belongs in "
            "the primary step's expected_outputs beside that result, or in a step "
            "that consumes the primary step's typed output. A step reading only "
            "the cohort does not join the lineage."
        )
    # This contract was compiled for source_analysis_type, which the host
    # inferred from the research context. A plan that declares a different
    # analysis_type is judged against *that* family's contract, whose required
    # roles are not listed above. Measured on 2026-07-29: a Planner shown the
    # survival contract declared association_study -- the better label for a
    # binary in-hospital-mortality outcome -- and was then judged on
    # primary_estimand and robustness, which it had never seen, while the
    # survival roles it had been shown stopped applying. Five attempts produced
    # five different violations and nothing executed. Re-declaring may well be
    # right; doing it unknowingly is what costs the run.
    lines.append(
        f"- rule: this contract is compiled for analysis_type="
        f"{contract.source_analysis_type}. Declaring a different analysis_type "
        "REPLACES it with that family's contract, whose required roles are not "
        "listed here. Either keep this analysis_type, or declare the family you "
        "intend and cover its roles -- do not re-declare casually."
    )
    return "\n".join(lines)


def _artifact_text_matches_requirement(
    text: str, requirement: ArticleDisplayRequirement
) -> bool:
    haystack = _normalise_space(text)
    return any(term and term in haystack for term in requirement.search_terms)


def _typed_declaration_set(value: Any) -> Set[tuple[str, str]]:
    if not isinstance(value, (list, tuple, set, frozenset)):
        return set()
    return {parsed for item in value if (parsed := typed_product(item)) is not None}


def _has_scientific_result_product(products: Set[tuple[str, str]]) -> bool:
    return any(
        kind not in {"figure", "log", "report", "code", "test"}
        for kind, _product in products
    )


def empty_primary_lineage_reason(plan: AnalysisPlan) -> Optional[str]:
    """Why no step is on the primary lineage, in the plan's own terms, or None.

    ``_declared_primary_lineage_step_ids`` returns the empty set for three
    structural reasons, and while it does, a headline-owned role cannot be
    credited *anywhere* -- declaring its product in any step changes nothing.
    The host reported only the symptom ("missing role X") and advised declaring
    the product in the primary step, which the Planner may already have done.
    canary5 spent 2 of its 5 attempts there.

    Only ONE of its three reasons is reachable here, which is why this is four
    lines rather than three branches.  ``AnalysisPlan`` already refuses the
    other two outright -- more than one primary step, and a primary step whose
    outputs are all displays -- so a plan holding either never reaches the
    article contract at all.  Both were written and deleted after construction
    showed the plan cannot be built; canary5's "four primary steps" rejection
    was a *different* attempt from its missing-role ones.

    What remains is a plan with NO primary step, which ``AnalysisPlan`` permits
    and which silently makes every headline role uncreditable.
    """

    if any(step.planned_analysis_role == "primary" for step in plan.steps):
        return None
    return (
        "no step declares planned_analysis_role='primary', so the primary "
        "lineage is empty and no headline-owned role can be credited in any "
        "step, wherever its product is declared"
    )


def _declared_primary_lineage_step_ids(plan: AnalysisPlan) -> Set[str]:
    """Return Planner-declared primary result descendants.

    Plan coverage may use declarations because execution has not happened yet,
    but it must still require a unique typed producer and may not route a
    sensitivity step into the headline lineage.
    """

    primary_steps = [
        step for step in plan.steps if step.planned_analysis_role == "primary"
    ]
    if len(primary_steps) != 1:
        return set()
    primary = primary_steps[0]
    primary_outputs = _typed_declaration_set(primary.expected_outputs)
    if not _has_scientific_result_product(primary_outputs):
        return set()
    producers: Dict[tuple[str, str], Set[str]] = {}
    for step in plan.steps:
        for product in _typed_declaration_set(step.expected_outputs):
            producers.setdefault(product, set()).add(step.step_id)
    allowed = {primary.step_id}
    changed = True
    while changed:
        changed = False
        for step in plan.steps:
            if step.step_id in allowed or step.planned_analysis_role not in {
                "secondary",
                "auxiliary",
            }:
                continue
            for product in _typed_declaration_set(step.inputs):
                owners = producers.get(product, set())
                if len(owners) == 1 and owners <= allowed:
                    allowed.add(step.step_id)
                    changed = True
                    break
    return allowed


def declared_primary_lineage_step_ids(plan: AnalysisPlan) -> frozenset[str]:
    """Expose the typed pre-execution primary lineage to adjacent owners.

    Figure planning and article-content planning must use the same lineage
    definition.  Returning an immutable projection keeps callers from
    modifying the reporting owner's internal working set.
    """

    return frozenset(_declared_primary_lineage_step_ids(plan))


#: Roles for which the SCHEMA, not this contract, fixes the product name.
#:
#: ``AnalysisStep`` refuses a step carrying a ``table_one_spec`` unless its
#: expected outputs include ``table:table_one`` -- there is exactly one way to
#: declare a Table 1, and it is not the contract's display id.  Every playbook
#: family names this role's module ``baseline_table`` or ``descriptive_table``,
#: so a remediation hint built from the module id alone names the one product
#: the schema will reject, while ``table:table_one`` -- which ``_ROLE_ALIASES``
#: already credits for this role -- is never mentioned.
#:
#: MEASURED on a survival-analysis fixture, 2026-08-03: five planner attempts,
#: five distinct rejections, two of them this loop -- ``missing required article
#: contract role(s): baseline_context ... 'table:baseline_table'`` followed by
#: ``table_one_spec requires expected output 'table:table_one'``.  The task has
#: never produced a plan.
#:
#: This is a schema law with one entry, not a name allowlist: the accompanying
#: test derives the check from ``AnalysisStep`` itself and fails if a hinted
#: product stops being legally emittable, so a future law cannot drift from it.
SCHEMA_MANDATED_ROLE_PRODUCTS: Dict[str, str] = {
    "baseline_context": "table:table_one",
}


def hinted_typed_products(role: str, module_ids: Sequence[str]) -> List[str]:
    """The typed products to offer the Planner for an unmet role.

    Returns the schema-mandated product where one exists, because a hint the
    schema refuses is worse than no hint: it costs an attempt and returns the
    Planner to the same wall.
    """

    mandated = SCHEMA_MANDATED_ROLE_PRODUCTS.get(role)
    if mandated is not None:
        return [mandated]
    return [f"table:{module_id}" for module_id in module_ids]


def _plan_outputs_match_requirement(
    output_declarations: Set[str],
    requirement: ArticleDisplayRequirement,
) -> bool:
    terms = [
        requirement.module_id,
        requirement.role,
        *requirement.acceptable_outputs,
        *role_check_terms(requirement.role),
        *_ROLE_ALIASES.get(requirement.role, ()),
    ]
    return any(
        _declaration_matches_term(declaration, term)
        for declaration in output_declarations
        for term in terms
    )


def roles_covered_by_plan(
    plan: Optional[AnalysisPlan],
    contract: ArticleAnalysisContract,
) -> Set[str]:
    if plan is None:
        return set()
    _method_declarations, output_declarations = _structured_plan_declarations(plan)
    primary_lineage_ids = _declared_primary_lineage_step_ids(plan)
    primary_output_declarations: Set[str] = set()
    if primary_lineage_ids:
        _primary_methods, primary_output_declarations = _structured_plan_declarations(
            plan.model_copy(
                update={
                    "steps": [
                        step
                        for step in plan.steps
                        if step.step_id in primary_lineage_ids
                    ]
                }
            )
        )
    planner_owned_roles = set(contract.planner_owned_result_roles)
    runtime_roles_by_step: Dict[str, Set[str]] = {}
    for step in plan.steps:
        if step.scientific_action_id is None:
            continue
        try:
            action = scientific_action_for_id(
                analysis_type=plan.analysis_type,
                action_id=step.scientific_action_id,
            )
        except ValueError:
            continue
        runtime_contract = action.runtime_contract
        if runtime_contract is None or tuple(step.expected_outputs) != tuple(
            product_id for product_id, _role in runtime_contract.outputs
        ):
            continue
        runtime_roles_by_step[step.step_id] = set(runtime_contract.article_roles)
    covered: Set[str] = set()
    for requirement in contract.requirements:
        candidate_outputs = (
            primary_output_declarations
            if requirement.role in planner_owned_roles
            else output_declarations
        )
        eligible_runtime_roles = {
            role
            for step_id, roles in runtime_roles_by_step.items()
            if requirement.role not in planner_owned_roles
            or step_id in primary_lineage_ids
            for role in roles
        }
        if requirement.role in eligible_runtime_roles or _plan_outputs_match_requirement(
            candidate_outputs, requirement
        ):
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
    def field(name: str) -> Any:
        if isinstance(record, Mapping):
            return record.get(name)
        return getattr(record, name, None)

    parts = [
        field("evidence_id"),
        field("kind"),
        field("description"),
        field("relative_path"),
        field("produced_by_step"),
    ]
    metadata = field("metadata")
    if metadata:
        parts.append(json.dumps(metadata, ensure_ascii=False, default=str))
    return _normalise_space("\n".join(str(part or "") for part in parts))


def _step_summary_text(record: Mapping[str, Any]) -> str:
    if record.get("status") != "ok":
        return ""
    summary = record.get("step_summary")
    if not isinstance(summary, Mapping):
        return ""
    # Article-role ownership comes from structured artifact identity, not free
    # prose buried in notes/diagnostics. Otherwise a baseline table that merely
    # mentions a future "sensitivity analysis" can falsely satisfy robustness.
    identity = {
        "output_files": summary.get("output_files"),
        "figure_files": summary.get("figure_files"),
        "contract_files": summary.get("contract_files"),
    }
    return _normalise_space(json.dumps(identity, ensure_ascii=False, default=str))


def _evidence_field(record: Any, name: str) -> Any:
    if isinstance(record, Mapping):
        return record.get(name)
    return getattr(record, name, None)


def _verified_resolved_input_bindings(
    *,
    record: Mapping[str, Any],
    run_dir: Path,
    evidence_by_id: Mapping[str, Any],
) -> Optional[Mapping[str, Mapping[str, Any]]]:
    """Load one exact resolved-input receipt, failing closed on any drift."""

    relative_text = str(record.get("resolved_inputs_path") or "").strip()
    if not relative_text:
        # Resume revalidation deliberately removes the former execution-time
        # path from the current step record.  The immutable receipt remains at
        # the host-owned canonical location and is still bound by its digest.
        # Recover only that one dependency-neutral location; arbitrary or
        # legacy paths continue to fail closed below.
        step_id = str(record.get("step_id") or "").strip()
        if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", step_id) is None:
            return None
        relative_text = f"resolved_inputs/{step_id}.json"
    expected_sha = str(record.get("resolved_inputs_sha256") or "").strip().lower()
    relative = Path(relative_text)
    if (
        not relative_text
        or relative.is_absolute()
        or any(part in {"", ".", ".."} for part in relative.parts)
        or re.fullmatch(r"[0-9a-f]{64}", expected_sha) is None
    ):
        return None
    root = Path(run_dir).resolve()
    candidate = root / relative
    current = root
    try:
        for part in relative.parts:
            current = current / part
            if current.is_symlink():
                return None
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(root)
        if not resolved.is_file():
            return None
        if hashlib.sha256(resolved.read_bytes()).hexdigest() != expected_sha:
            return None
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    if (
        not isinstance(payload, Mapping)
        or payload.get("schema_version") != "2.1"
        or payload.get("step_id") != record.get("step_id")
        or not isinstance(payload.get("inputs"), Mapping)
    ):
        return None
    bindings = payload["inputs"]
    receipt_ids = {
        str(value)
        for value in (record.get("resolved_input_evidence_ids") or [])
        if str(value).strip()
    }
    bound_ids: Set[str] = set()
    for input_key, raw_binding in bindings.items():
        if not isinstance(input_key, str) or not isinstance(raw_binding, Mapping):
            return None
        product = typed_product(input_key)
        identity = raw_binding.get("identity_row")
        evidence_id = str(raw_binding.get("evidence_id") or "").strip()
        if product is None or not isinstance(identity, Mapping) or not evidence_id:
            return None
        expected_identity = {
            "input_key": input_key,
            "declared_kind": product[0],
            "product": product[1],
            "evidence_id": evidence_id,
            "sha256": raw_binding.get("sha256"),
            "produced_by_step": raw_binding.get("produced_by_step"),
        }
        if any(identity.get(key) != value for key, value in expected_identity.items()):
            return None
        if (
            raw_binding.get("declared_kind") != product[0]
            or raw_binding.get("product") != product[1]
        ):
            return None
        evidence_record = evidence_by_id.get(evidence_id)
        if (
            evidence_record is None
            or verified_run_evidence_path(run_dir, evidence_record) is None
            or str(_evidence_field(evidence_record, "sha256") or "")
            != str(raw_binding.get("sha256") or "")
            or str(_evidence_field(evidence_record, "produced_by_step") or "")
            != str(raw_binding.get("produced_by_step") or "")
        ):
            return None
        bound_ids.add(evidence_id)
    if bound_ids != receipt_ids:
        return None
    return bindings


def _verified_primary_lineage_step_ids(
    *,
    current_records: Sequence[Mapping[str, Any]],
    evidence_records: Sequence[Any],
    run_dir: Path,
) -> Set[str]:
    """Return runtime descendants with digest-bound primary evidence inputs."""

    primary_record = unique_verified_primary_record(current_records)
    if primary_record is None:
        return set()
    primary_step_id = str(primary_record.get("step_id") or "").strip()
    request = primary_record.get("analysis_request")
    primary_payload = request.get("step") if isinstance(request, Mapping) else None
    primary_outputs = _typed_declaration_set(
        primary_payload.get("expected_outputs")
        if isinstance(primary_payload, Mapping)
        else None
    )
    if not primary_step_id or not _has_scientific_result_product(primary_outputs):
        return set()

    current_evidence = current_evidence_records(evidence_records, current_records)
    evidence_by_id = {
        str(_evidence_field(record, "evidence_id") or ""): record
        for record in current_evidence
        if str(_evidence_field(record, "evidence_id") or "").strip()
    }
    producers: Dict[tuple[str, str], Set[str]] = {}
    record_by_step: Dict[str, Mapping[str, Any]] = {}
    for record in current_records:
        step_id = str(record.get("step_id") or "").strip()
        analysis_request = record.get("analysis_request")
        payload = (
            analysis_request.get("step")
            if isinstance(analysis_request, Mapping)
            else None
        )
        if not step_id or not isinstance(payload, Mapping):
            continue
        products = _typed_declaration_set(payload.get("expected_outputs"))
        record_by_step[step_id] = record
        for product in products:
            producers.setdefault(product, set()).add(step_id)

    allowed = {primary_step_id}
    changed = True
    while changed:
        changed = False
        for step_id, record in record_by_step.items():
            if step_id in allowed or verified_planned_analysis_role(record) not in {
                "secondary",
                "auxiliary",
            }:
                continue
            request = record.get("analysis_request")
            payload = request.get("step") if isinstance(request, Mapping) else None
            if not isinstance(payload, Mapping):
                continue
            bindings = _verified_resolved_input_bindings(
                record=record,
                run_dir=run_dir,
                evidence_by_id=evidence_by_id,
            )
            if bindings is None:
                continue
            for raw_input in payload.get("inputs") or []:
                product = typed_product(raw_input)
                owners = producers.get(product, set()) if product is not None else set()
                binding = (
                    bindings.get(raw_input) if isinstance(raw_input, str) else None
                )
                if (
                    len(owners) != 1
                    or not owners <= allowed
                    or not isinstance(binding, Mapping)
                ):
                    continue
                owner_id = next(iter(owners))
                owner_record = record_by_step.get(owner_id)
                evidence_id = str(binding.get("evidence_id") or "")
                if (
                    owner_record is not None
                    and str(binding.get("produced_by_step") or "") == owner_id
                    and evidence_id
                    in {
                        str(value) for value in (owner_record.get("evidence_ids") or [])
                    }
                ):
                    allowed.add(step_id)
                    changed = True
                    break
    return allowed


# Shared with figure_strategy / display_suite via figures.contracts so all
# article-level audits see the identical contract list.
_figure_contract_paths = figure_contract_paths


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
    allowed_step_ids: Optional[Set[str]] = None,
) -> List[str]:
    texts: List[str] = []
    for record in current_evidence_records(evidence_records, per_step_records):
        produced_by_step = str(
            _evidence_field(record, "produced_by_step") or ""
        ).strip()
        if per_step_records is not None and not produced_by_step:
            # A run-level audit/package record does not prove that a current
            # execution step delivered this scientific role. Step summaries
            # and current figure contracts remain the execution authority.
            continue
        if allowed_step_ids is not None and produced_by_step not in allowed_step_ids:
            continue
        kind = str(_evidence_field(record, "kind") or "")
        if kind in _COUNTED_ARTIFACT_KINDS:
            texts.append(_record_to_text(record))
    current_records = current_successful_step_records(per_step_records)
    for record in current_records:
        step_id = str(record.get("step_id") or "").strip()
        if allowed_step_ids is not None and step_id not in allowed_step_ids:
            continue
        text = _step_summary_text(record)
        if text:
            texts.append(text)
    figure_records: Sequence[Mapping[str, Any]] = per_step_records
    if allowed_step_ids is not None:
        figure_records = [
            record
            for record in current_records
            if str(record.get("step_id") or "").strip() in allowed_step_ids
        ]
    for path in _figure_contract_paths(
        run_dir,
        per_step_records=figure_records,
        include_publication_figures=allowed_step_ids is None,
    ):
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
    current_records = current_successful_step_records(per_step_records)
    primary_lineage_ids = _verified_primary_lineage_step_ids(
        current_records=current_records,
        evidence_records=evidence_records,
        run_dir=run_dir,
    )
    primary_texts: List[str] = []
    if primary_lineage_ids:
        primary_texts = _artifact_texts(
            evidence_records=evidence_records,
            per_step_records=per_step_records,
            run_dir=run_dir,
            allowed_step_ids=primary_lineage_ids,
        )
    planner_owned_roles = set(contract.planner_owned_result_roles)
    covered: Set[str] = set()
    for requirement in contract.requirements:
        candidate_texts = (
            primary_texts if requirement.role in planner_owned_roles else texts
        )
        if any(
            _artifact_text_matches_requirement(text, requirement)
            for text in candidate_texts
        ):
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
    contract = build_article_analysis_contract(
        context,
        analysis_type=plan.analysis_type if plan is not None else None,
    )
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


def article_contract_audit_payload(status: Mapping[str, Any]) -> Dict[str, Any]:
    """Canonical on-disk shape of ``article_contract_audit.json``.

    Both writers (the execute-phase crash-resilience snapshot and the final
    report-phase write) must produce this shape, otherwise the registered
    evidence copy and the run-dir file diverge structurally under the same
    schema id.
    """
    return {
        "schema_version": status["article_contract_audit_schema_version"],
        "article_contract_complete": status["article_contract_complete"],
        "analysis_family": status["article_contract_family"],
        "required_roles": status["article_required_roles"],
        "plan_roles": status["article_plan_roles"],
        "artifact_roles": status["article_artifact_roles"],
        "missing_plan_roles": status["article_missing_plan_roles"],
        "missing_artifact_roles": status["article_missing_artifact_roles"],
        "missing_artifact_modules": status["article_missing_artifact_modules"],
        "errors": status["article_contract_errors"],
        "contract": status["article_contract"],
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
                "missing_artifact_modules": status["article_missing_artifact_modules"],
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
    output_kind = (
        "figure"
        if any(
            token in joined
            for token in ("figure", "plot", "curve", "heatmap", "panel", "diagram")
        )
        else "table"
    )
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
    all_missing = _iter_missing_requirements(contract, covered_roles)
    planner_owned_roles = set(contract.planner_owned_result_roles)
    planner_owned_missing = [
        requirement
        for requirement in all_missing
        if requirement.role in planner_owned_roles
    ]
    missing = [
        requirement
        for requirement in all_missing
        if requirement.role not in planner_owned_roles
    ]
    if not missing:
        if planner_owned_missing:
            return plan, [
                ValidationFinding(
                    validator="article_analysis_contract",
                    severity="warning",
                    message=(
                        "The article contract is missing a Planner-owned headline "
                        "result; deterministic display augmentation cannot invent "
                        "or assign that scientific role."
                    ),
                    detail={
                        "reason": "planner_owned_headline_result_missing",
                        "missing_roles": [
                            requirement.role for requirement in planner_owned_missing
                        ],
                    },
                )
            ]
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
                planned_analysis_role="auxiliary",
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
            "added_step_ids": [step.step_id for step in new_steps[-len(missing) :]],
            "planner_owned_missing_roles": [
                requirement.role for requirement in planner_owned_missing
            ],
        },
    )
    findings = [finding]
    if planner_owned_missing:
        findings.append(
            ValidationFinding(
                validator="article_analysis_contract",
                severity="warning",
                message=(
                    "Planner-owned headline results remain missing; deterministic "
                    "display augmentation added only non-scientific support roles."
                ),
                detail={
                    "reason": "planner_owned_headline_result_missing",
                    "missing_roles": [
                        requirement.role for requirement in planner_owned_missing
                    ],
                },
            )
        )
    return revised, findings


__all__ = [
    "ARTICLE_ANALYSIS_CONTRACT_SCHEMA_VERSION",
    "ARTICLE_CONTRACT_AUDIT_SCHEMA_VERSION",
    "ArticleAnalysisContract",
    "ArticleDisplayRequirement",
    "augment_plan_for_article_contract",
    "build_article_analysis_contract",
    "declared_primary_lineage_step_ids",
    "empty_primary_lineage_reason",
    "render_article_analysis_contract_for_prompt",
    "roles_covered_by_artifacts",
    "roles_covered_by_plan",
    "summarize_article_contract_coverage",
    "validate_plan_against_article_contract",
    "validate_run_against_article_contract",
]
