"""Resolve generic plan deliverables against the typed owners in this draft."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Mapping, Sequence

from ...contracts.claim_ceiling import first_result
from ...contracts.descriptive_execution import exposure_outcome_distribution_execution_verdict
from ...contracts.model_terms import PreparationRefusal, preparation_for_step
from ...contracts.primary_cohort import primary_analysis_cohort_producer_uses_universe
from ...contracts.prediction_subgroup import prediction_subgroup_step_matches
from ...contracts.product_files import (
    DeliverableKind, DeliveryRequest, HOST_DELIVERY_KINDS, HostDeliveryKind, PlanDeliverables, RUN_DELIVERY_SCHEMA_VERSION,
    delivery_requests, host_delivery_coverage, host_delivery_owners,
)
from ...contracts.product_identity import is_canonical_typed_product_token, typed_product
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


def prefill_plan_deliverables(
    question: str, *, defaults: Sequence[HostDeliveryKind] = (),
) -> PlanDeliverables:
    requests = delivery_requests(question)
    check = tuple(kind for kind, state in requests.items() if state is DeliveryRequest.CHECK)
    return PlanDeliverables(
        requested=tuple(
            kind for kind, state in requests.items()
            if state is DeliveryRequest.REQUESTED or (state is DeliveryRequest.UNMENTIONED and kind in defaults)
        ),
        excluded=tuple(kind for kind, state in requests.items() if state is DeliveryRequest.NOT_REQUESTED),
        check=check, confirmed=not check, source="detector_prefill",
    )


def bind_confirmed_host_delivery(plan: AnalysisPlan, selection: PlanDeliverables) -> AnalysisPlan:
    """Bind the selected terminal owners to the current scientific plan."""
    from ...schema import AnalysisStep

    kept = [step for step in plan.steps if step.host_delivery_kind is None]
    kinds = (
        tuple(kind for kind in HOST_DELIVERY_KINDS if kind in selection.requested)
        if selection.confirmed and not first_result(plan) else ()
    )
    used_ids = {step.step_id for step in kept}
    intents = {
        "article_preview": "Deliver the evidence-bound article preview",
        "manuscript_pdf": "Deliver the compiled manuscript PDF",
        "references": "Deliver the run's bibliography",
    }
    added = []
    for kind in kinds:
        step_id = f"host_delivery_{kind}"
        while step_id in used_ids:
            step_id += "_host"
        added.append(AnalysisStep(step_id=step_id, intent=intents[kind], host_delivery_kind=kind))
    return plan.model_copy(update={"deliverables": selection, "steps": [*kept, *added]})


def bind_requested_host_delivery(plan: AnalysisPlan, *, question: str | None = None) -> AnalysisPlan:
    """Initialize a fresh candidate with the host's reading of the original question."""
    return bind_confirmed_host_delivery(plan, prefill_plan_deliverables(question or plan.research_question))


def bind_run_deliverables(
    plan: AnalysisPlan, *, question: str, defaults: Sequence[HostDeliveryKind] | None,
    edit: Mapping[str, object] | None,
) -> AnalysisPlan:
    """Apply tier defaults and record what the human changed on the review card."""
    first = first_result(plan)
    prefill = prefill_plan_deliverables(question, defaults=() if first else defaults or ())
    if edit is None:
        return bind_confirmed_host_delivery(plan, prefill)

    edited = PlanDeliverables.model_validate(edit)
    deferred = set(edited.deferred)
    if first and not edited.requested:
        # Older cards saved only the requested list. Keep known earlier
        # requests unless the current edit explicitly excludes them.
        prior = set(prefill.requested)
        if plan.deliverables is not None:
            prior.update(plan.deliverables.requested)
        deferred.update(prior.difference(edited.excluded))
    excluded = (set(edited.excluded) | set(prefill.excluded)).difference(edited.requested, deferred)
    selection = PlanDeliverables(
        requested=edited.requested, confirmed=True, source="user_edit",
        deferred=tuple(kind for kind in HOST_DELIVERY_KINDS if kind in deferred),
        excluded=tuple(kind for kind in HOST_DELIVERY_KINDS if kind in excluded),
    )
    return bind_confirmed_host_delivery(plan, selection)


def planned_run_delivery_evidence(
    kinds: Sequence[DeliverableKind], *, plan: AnalysisPlan,
) -> dict[str, object] | None:
    """Bind each run deliverable to the plan that this judgment will review."""
    product_kinds: dict[DeliverableKind, str] = {"figures": "figure", "result_tables": "table"}
    owners = host_delivery_owners(plan.model_dump(mode="json")) if getattr(plan, "deliverables", None) is not None else {}
    return host_delivery_coverage({
        kind: [
            step.step_id for step in plan.steps
            if step.step_id == owners.get(kind) or any(
                is_canonical_typed_product_token(output) and output.partition(":")[0] == product_kinds.get(kind)
                for output in step.expected_outputs
            )
        ]
        for kind in kinds
    })


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
