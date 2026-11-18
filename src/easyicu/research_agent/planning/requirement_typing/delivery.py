"""Resolve generic plan deliverables against the typed owners in this draft."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Sequence

from ...contracts.descriptive_execution import exposure_outcome_distribution_execution_verdict
from ...contracts.model_terms import PreparationRefusal, preparation_for_step
from ...contracts.primary_cohort import primary_analysis_cohort_producer_uses_universe
from ...contracts.prediction_subgroup import prediction_subgroup_step_matches
from ...contracts.product_files import RUN_DELIVERY_SCHEMA_VERSION
from ...contracts.product_identity import typed_product
from ..question_quote import normalize_question_quote
from ..scientific_action_catalog import scientific_action_for_id

if TYPE_CHECKING:
    from ...schema import AnalysisPlan, AnalysisStep


PlanDeliverableKind = Literal[
    "literature_grounded_design", "population_description", "reproducible_analysis",
    "missing_data_handling", "repeated_admission_handling", "diagnostics", "sensitivity_analyses",
]

# Whole neutral requests only. Users who name a method, time window, input,
# population restriction or estimand still need the corresponding owner.
_NAMES: dict[PlanDeliverableKind, tuple[str, ...]] = {
    "literature_grounded_design": (
        "literature-grounded study design", "literature-informed design", "literature-grounded design",
        "基于文献的研究设计", "有文献依据的研究设计",
    ),
    "population_description": (
        "data and population description", "data and cohort description", "description of the data and population",
        "study population description", "cohort description", "数据与研究人群描述", "数据和人群描述", "队列描述",
    ),
    "reproducible_analysis": (
        "reproducible analysis", "reproducible analysis plan", "可复现的分析", "可复现分析", "可复现的分析计划",
    ),
    "missing_data_handling": (
        "missing data handling", "missing-data handling", "handling of missing data", "缺失数据处理", "缺失值处理",
    ),
    "repeated_admission_handling": (
        "handling of repeated admissions", "repeated admission handling", "repeated stay handling",
        "handling of repeat stays", "重复入院处理", "重复住院处理", "重复入住处理",
    ),
    "diagnostics": ("diagnostics", "analysis diagnostics", "诊断检查", "分析诊断"),
    "sensitivity_analyses": (
        "sensitivity analysis", "sensitivity analyses", "sensitivity checks", "敏感性分析",
    ),
}
_BY_NAME = {name: kind for kind, names in _NAMES.items() for name in names}
_ORDERED_NAMES = sorted(_BY_NAME, key=len, reverse=True)
_JOINERS = (", and ", " and ", ",", "、", "，", "以及", "和", "与", "及")
_PREFIXES = (
    "please provide ", "please include ", "provide ", "include ", "deliver ", "report ",
    "请提供", "请包含", "请报告", "提供", "包含", "报告",
)


def requested_plan_deliverables(quote: str) -> tuple[PlanDeliverableKind, ...]:
    text = normalize_question_quote(quote).rstrip(".。?!？！")
    prefix = next((prefix for prefix in _PREFIXES if text.startswith(prefix)), "")
    text = text[len(prefix):]
    kinds = []
    while text:
        text = text.lstrip().removeprefix("a ").removeprefix("an ").removeprefix("the ")
        name = next((name for name in _ORDERED_NAMES if text.startswith(name)), None)
        if name is None:
            break
        kinds.append(_BY_NAME[name])
        text = text[len(name):]
        if not text:
            return tuple(dict.fromkeys(kinds))
        joiner = next((joiner for joiner in _JOINERS if text.startswith(joiner)), None)
        if joiner is None:
            break
        text = text[len(joiner):]
    return ()


def _preparation_has_missing_policy(step: AnalysisStep) -> bool:
    try:
        preparation = preparation_for_step(step)
    except PreparationRefusal:
        return False
    return preparation is not None and preparation.missing_data is not None


def _registered_analysis(step: AnalysisStep, plan: AnalysisPlan) -> bool:
    if not plan.analysis_type or not step.scientific_action_id:
        return False
    try:
        scientific_action_for_id(analysis_type=plan.analysis_type, action_id=step.scientific_action_id)
    except ValueError:
        return False
    return any(map(typed_product, step.expected_outputs))


def _owns(kind: PlanDeliverableKind, step: AnalysisStep, plan: AnalysisPlan) -> bool:
    primary = step.planned_analysis_role == "primary"
    if kind == "literature_grounded_design":
        return primary and bool(step.literature_citation_keys)
    if kind == "population_description":
        return step.table_one_spec is not None or primary_analysis_cohort_producer_uses_universe(step=step, plan=plan)
    if kind == "reproducible_analysis":
        return primary and _registered_analysis(step, plan)
    if kind == "missing_data_handling":
        return primary and (
            exposure_outcome_distribution_execution_verdict(step).claimed or _preparation_has_missing_policy(step)
        )
    if kind == "repeated_admission_handling":
        distribution = step.exposure_outcome_distribution_spec
        return primary and (
            step.runtime_dependence_contract is not None
            or any(model.dependence is not None for model in step.model_requirements)
            or (distribution is not None and distribution.dependence is not None
                and exposure_outcome_distribution_execution_verdict(step).claimed)
        )
    if kind == "diagnostics":
        if any((
            step.measurement_audit_spec, step.covariate_balance_spec, step.cluster_size_spec,
            step.trajectory_stability_spec,
        )):
            return True
        try:
            return prediction_subgroup_step_matches(step)
        except ValueError:
            return False
    return step.robustness_replay_spec is not None or (
        step.planned_analysis_role == "sensitivity" and _registered_analysis(step, plan)
    )


def planned_delivery_evidence(
    kinds: Sequence[PlanDeliverableKind], *, plan: AnalysisPlan,
) -> dict[str, object] | None:
    owners = {kind: [step.step_id for step in plan.steps if _owns(kind, step, plan)] for kind in kinds}
    if not owners or not all(owners.values()):
        return None
    return {
        "source": "host_verified",
        "contract": RUN_DELIVERY_SCHEMA_VERSION,
        "deliverable_kinds": list(owners),
        "artifact_names": ["analysis_plan.json"],
        "owner_step_ids_by_kind": owners,
        "reason": "The host verified a typed owner for each requested deliverable in this plan.",
    }
