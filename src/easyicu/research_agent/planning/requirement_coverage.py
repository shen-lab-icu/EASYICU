"""Digest-bound coverage projection for explicit study requirements.

The scientific-review owner has already decided whether a typed outcome,
sensitivity, or exact covariate is covered, and the question-requirements
owner (``planning.question_requirements``) whether an analysis the question
asks for is.  This module records that decision as a stable row and binds it
to the exact context and plan. It also binds generic deliverables to their
typed plan owners, without granting execution or reporting authority.

The structured adjustment reading is a required subset independent of exact
rosters. Its coverage is checked here against primary fitted covariate terms.
A stated subgroup is checked here too, against the analysis steps that
answer it in the plan's own family; review inside the execution kernel reads
both without the Planner-facing question-requirement owner.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, Mapping, Optional, Sequence

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..contracts.analysis_design import PlanShapeRemediation as PlanShapeRemediation
from ..contracts.claim_ceiling import first_result
from ..contracts.descriptive_execution import exposure_outcome_distribution_execution_verdict
from ..contracts.model_terms import PreparationRefusal, preparation_for_step
from ..contracts.prediction_subgroup import prediction_subgroup_step_matches
from ..contracts.primary_cohort import primary_analysis_cohort_producer_uses_universe
from ..contracts.product_files import (
    DeliverableKind,
    DeliveryRequest,
    HOST_DELIVERY_KINDS,
    HOST_DELIVERY_REASON,
    HostDeliveryKind,
    PlanDeliverables,
    RUN_DELIVERY_SCHEMA_VERSION,
    delivery_requests,
    host_delivery_coverage,
    host_delivery_owners,
)
from ..contracts.product_identity import is_canonical_typed_product_token, typed_product
from ..methods.descriptive_inputs import REQUESTED_CONCEPT_NO_DATA, context_columns_without_data
from .adjustment_authority import concept_relatives
from .question_quote import normalize_question_quote

if TYPE_CHECKING:
    from ..schema import AnalysisPlan, AnalysisStep, ResearchContext


RequirementKind = Literal["outcome", "sensitivity", "covariate", "question"]
RequirementCoverageStatus = Literal["covered", "missing", "unsupported"]
PlanDeliverableKind = Literal[
    "literature_grounded_design", "population_description", "reproducible_analysis",
    "missing_data_handling", "repeated_admission_handling", "diagnostics", "sensitivity_analyses",
]


class PlanRequirementCoverageRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    requirement_id: str = Field(min_length=3, max_length=256)
    kind: RequirementKind
    #: What a question requirement asks for (``benchmark``, ``subgroup`` ...);
    #: set on question rows only.
    question_kind: Optional[str] = Field(default=None, pattern=r"^[a-z][a-z_]{2,31}$")
    concept_identity: str = Field(min_length=1, max_length=256)
    source_ref: str = Field(min_length=3, max_length=512)
    status: RequirementCoverageStatus
    owner_step_ids: tuple[str, ...] = ()
    delivery_kind: Optional[DeliverableKind | PlanDeliverableKind] = Field(default=None, exclude_if=lambda value: value is None)
    statement_ref: Optional[str] = Field(
        default=None, min_length=3, max_length=512, exclude_if=lambda value: value is None,
    )
    reason_code: str = Field(min_length=3, max_length=160)

    @model_validator(mode="after")
    def _owner_status_is_consistent(self) -> "PlanRequirementCoverageRecord":
        if len(self.owner_step_ids) != len(set(self.owner_step_ids)):
            raise ValueError("requirement coverage owner step ids must be unique")
        if self.delivery_kind is not None and (
            self.kind != "question" or self.question_kind not in {"analysis", "delivery"}
            or self.status != "covered"
            or self.reason_code != HOST_DELIVERY_REASON
            or self.concept_identity != self.delivery_kind
        ):
            raise ValueError("host delivery covers only its typed output")
        if self.statement_ref is not None and (
            self.kind != "question" or self.question_kind != "analysis"
            or self.status != "covered" or self.owner_step_ids or self.delivery_kind
            or self.reason_code != "question_method_clause_stated"
        ):
            raise ValueError("a methods statement covers only its recorded question clause")
        if self.status == "covered" and not (self.owner_step_ids or self.delivery_kind or self.statement_ref):
            raise ValueError("covered requirement must name an owner step or host delivery kind or a recorded methods statement")
        if self.status != "covered" and self.owner_step_ids:
            raise ValueError("uncovered requirement cannot name an owner step")
        if (self.kind == "question") != (self.question_kind is not None):
            raise ValueError(
                "a question requirement row states what the question asks for"
            )
        return self


class PlanRequirementCoverage(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal["easyicu.plan_requirement_coverage/1"] = (
        "easyicu.plan_requirement_coverage/1"
    )
    context_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    plan_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    records: tuple[PlanRequirementCoverageRecord, ...]
    complete: bool

    @model_validator(mode="after")
    def _records_are_unique_and_complete(self) -> "PlanRequirementCoverage":
        identities = [record.requirement_id for record in self.records]
        if len(identities) != len(set(identities)):
            raise ValueError("plan requirement coverage ids must be unique")
        expected = all(record.status == "covered" for record in self.records)
        if self.complete != expected:
            raise ValueError("plan requirement coverage completeness drifted")
        return self


def with_requirement_data_availability(
    coverage: PlanRequirementCoverage, context: ResearchContext,
) -> PlanRequirementCoverage:
    """An owner cannot cover a requirement whose clinical values are all empty."""

    records = tuple(
        record.model_copy(update={
            "status": "unsupported", "owner_step_ids": (),
            "reason_code": REQUESTED_CONCEPT_NO_DATA,
        }) if not (record.delivery_kind or record.statement_ref) and context_columns_without_data(context, (record.concept_identity,)) else record
        for record in coverage.records
    )
    return coverage.model_copy(update={
        "records": records, "complete": all(record.status == "covered" for record in records),
    })


def build_plan_requirement_coverage(
    *,
    context_sha256: str,
    plan_sha256: str,
    requested_outcomes: Sequence[str],
    outcome_owner_step_ids: Mapping[str, Sequence[str]],
    sensitivity_spec_ids: Sequence[str],
    sensitivity_owner_step_ids: Mapping[str, Sequence[str]],
    unsupported_sensitivity_spec_ids: Sequence[str],
    exact_covariates: Sequence[str],
    covariate_owner_step_ids: Mapping[str, Sequence[str]],
) -> PlanRequirementCoverage:
    """Build rows only from typed requirements and prior owner validation."""

    records: list[PlanRequirementCoverageRecord] = []
    for outcome in dict.fromkeys(str(value).strip() for value in requested_outcomes):
        if not outcome:
            continue
        owners = tuple(dict.fromkeys(outcome_owner_step_ids.get(outcome, ())))
        records.append(
            PlanRequirementCoverageRecord(
                requirement_id=f"outcome:{outcome}",
                kind="outcome",
                concept_identity=outcome,
                source_ref="research_context.json.cohort.requested_outcome_columns",
                status="covered" if owners else "missing",
                owner_step_ids=owners,
                reason_code=(
                    "typed_plan_owner_present"
                    if owners
                    else "typed_plan_owner_missing"
                ),
            )
        )

    unsupported = set(unsupported_sensitivity_spec_ids)
    for spec_id in dict.fromkeys(str(value).strip() for value in sensitivity_spec_ids):
        if not spec_id:
            continue
        owners = tuple(dict.fromkeys(sensitivity_owner_step_ids.get(spec_id, ())))
        if owners:
            status: RequirementCoverageStatus = "covered"
            reason_code = "typed_plan_owner_present"
        elif spec_id in unsupported:
            status = "unsupported"
            reason_code = "runtime_capability_unavailable"
        else:
            status = "missing"
            reason_code = "typed_plan_owner_missing"
        records.append(
            PlanRequirementCoverageRecord(
                requirement_id=f"sensitivity:{spec_id}",
                kind="sensitivity",
                concept_identity=spec_id,
                source_ref=(
                    "research_context.json.user_preferences.sensitivity_specs"
                ),
                status=status,
                owner_step_ids=owners,
                reason_code=reason_code,
            )
        )

    for covariate in dict.fromkeys(str(value).strip() for value in exact_covariates):
        if not covariate:
            continue
        owners = tuple(dict.fromkeys(covariate_owner_step_ids.get(covariate, ())))
        records.append(
            PlanRequirementCoverageRecord(
                requirement_id=f"covariate:{covariate}",
                kind="covariate",
                concept_identity=covariate,
                source_ref="research_context.json.user_preferences.covariates",
                status="covered" if owners else "missing",
                owner_step_ids=owners,
                reason_code=(
                    "typed_plan_owner_present"
                    if owners
                    else "typed_plan_owner_missing"
                ),
            )
        )

    return PlanRequirementCoverage(
        context_sha256=context_sha256,
        plan_sha256=plan_sha256,
        records=tuple(records),
        complete=all(record.status == "covered" for record in records),
    )


def stated_adjustment_coverage(context, plan, *, runtime_authority=None) -> PlanRequirementCoverage:
    """Only primary fitted covariate terms cover the question's required subset.

    A value appearing in Table 1, inputs, robustness or an exposure term is
    not adjustment. Every primary model must keep it; added covariates remain
    governed by the existing rationale and temporal-authority owners.
    """

    from ..canonical_json import canonical_sha256
    from .adjustment_authority import (
        context_stated_adjustments, stated_adjustment_columns,
    )

    primary = [
        (step.step_id, set(requirement.covariates or ()).intersection(
            term.name for term in requirement.model_terms or () if term.role == "covariate"
        ))
        for step in plan.steps if step.planned_analysis_role == "primary"
        for requirement in step.model_requirements
        if requirement.analysis_role == "primary"
    ]
    # The signed spline family replaces the draft model requirement with its
    # digest-bound execution contract. Inputs or a proposal alone cannot prove
    # adjustment: let that owner validate the primary before reading its roster.
    from ..authority.current_case_scientific_runtime import (
        CurrentCaseScientificAuthorityError, LandmarkSplineRuntimeAuthority,
    )

    if isinstance(runtime_authority, LandmarkSplineRuntimeAuthority):
        try:
            signed = runtime_authority.governed_step(plan)
        except CurrentCaseScientificAuthorityError:
            pass
        else:
            if (runtime_authority.exposure_column == context.primary_exposure
                and runtime_authority.outcome_column == context.target_outcome):
                primary.append((signed.step_id, set(runtime_authority.required_adjustment_columns)))
    records = []
    for adjustment in context_stated_adjustments(context):
        identity = adjustment.concept or adjustment.variable
        columns = set(stated_adjustment_columns(context, adjustment))
        owners = [
            step_id for step_id, covariates in primary if columns.intersection(covariates)
        ]
        unavailable = adjustment.source_status == "unavailable"
        no_data = bool(columns) and set(columns) <= set(context_columns_without_data(context, columns))
        # Without a concept no column can be checked; the review shows it.
        unresolved = adjustment.concept is None
        covered = (
            bool(primary) and len(owners) == len(primary)
            and not unavailable and not unresolved and not no_data
        )
        if no_data:
            status, reason = "unsupported", REQUESTED_CONCEPT_NO_DATA
        elif unavailable:
            status, reason = "unsupported", "source_concept_unavailable"
        elif unresolved:
            status, reason = "unsupported", "stated_adjustment_concept_unresolved"
        elif covered:
            status, reason = "covered", "typed_primary_adjustment_present"
        else:
            status, reason = "missing", "typed_primary_adjustment_missing"
        records.append(PlanRequirementCoverageRecord(
            requirement_id=f"stated_adjustment:{identity}", kind="covariate",
            concept_identity=identity,
            source_ref="research_context.json.user_preferences.data_constraints.stated_adjustments",
            status=status,
            owner_step_ids=tuple(dict.fromkeys(owners)) if covered else (),
            reason_code=reason,
        ))
    # Several clauses may state the same concept; it remains one requirement.
    unique = {record.requirement_id: record for record in records}
    return PlanRequirementCoverage(
        context_sha256=canonical_sha256(context.model_dump(mode="json")),
        plan_sha256=canonical_sha256(plan.model_dump(mode="json")),
        records=tuple(unique.values()),
        complete=all(record.status == "covered" for record in unique.values()),
    )



#: The step roles that analyse.  Cohort accounting, Table 1, audits, figures
#: and the report are auxiliary: reading a concept there answers nothing.
ANALYSIS_ROLES = frozenset({"primary", "secondary", "sensitivity"})

#: Actions that analyse a named subgroup.
SUBGROUP_ACTIONS = frozenset(
    {
        "association.effect_modification",
        "prediction.subgroup_fairness",
        "time_to_event.subgroup_hr",
    }
)


def _family_actions(analysis_type: str) -> tuple[Any, ...]:
    from .scientific_action_catalog import scientific_actions_for_analysis_type

    try:
        actions = scientific_actions_for_analysis_type(analysis_type).actions
    except ValueError:
        return ()
    return tuple(action for action in actions if action.execution_mode != "not_available")


def subgroup_actions_for_family(analysis_type: str) -> frozenset[str]:
    """Executable subgroup owners offered by the existing family catalog."""

    return frozenset(
        action.action_id for action in _family_actions(analysis_type)
        if action.action_id in SUBGROUP_ACTIONS
    )


def family_describes_by_group(analysis_type: str) -> bool:
    """Whether the family answers a grouping by describing each of its levels.

    A descriptive family estimates no stratum effects: "describe the outcome
    in each ICU type" is answered by a result reported level by level of that
    grouping, as ``_judge`` accepts for a stated subgroup requirement.
    """

    return any(action.analysis_family == "descriptive" for action in _family_actions(analysis_type))


def _grouped_result_steps(plan: AnalysisPlan, columns: frozenset[str]) -> tuple[str, ...]:
    """Analysis steps whose typed result is reported in each level of a column."""

    from ..contracts.continuous_outcome_summary import (
        CONTINUOUS_OUTCOME_SUMMARY, continuous_outcome_columns,
    )

    owners = []
    for step in plan.steps:
        if step.planned_analysis_role not in ANALYSIS_ROLES:
            continue
        distribution = step.exposure_outcome_distribution_spec
        if distribution is not None:
            grouping = (distribution.exposure,)
        elif str(step.method or "").strip().casefold() == CONTINUOUS_OUTCOME_SUMMARY:
            grouping = continuous_outcome_columns(step)[:1]
        else:
            grouping = ()
        if columns.intersection(grouping):
            owners.append(step.step_id)
    return tuple(owners)


def stated_subgroup_owner_steps(context: ResearchContext, plan: AnalysisPlan, concept: str) -> tuple[str, ...]:
    """The analysis steps that answer a stated subgroup in this plan's family.

    A family with subgroup estimates answers only with a subgroup-capable step
    reading the variable; a Table 1 grouping or a covariate term is no stratum
    estimate.  A descriptive family answers with a result reported in each
    level of the variable.  Reuse the question owner's concept identities,
    excluding metadata companions.
    """

    columns = concept_relatives(context)(concept) & {
        variable.name for variable in context.variables
        if variable.role.value not in {"meta", "time", "id", "index"}
    }
    actions = subgroup_actions_for_family(plan.analysis_type)
    if not actions and family_describes_by_group(plan.analysis_type):
        return _grouped_result_steps(plan, frozenset(columns))
    return tuple(
        step.step_id for step in plan.steps
        if step.scientific_action_id in actions
        and step.planned_analysis_role in ANALYSIS_ROLES
        and step.table_one_spec is None
        and columns.intersection(step.inputs)
    )


def stated_subgroup_coverage(context: ResearchContext, plan: AnalysisPlan) -> PlanRequirementCoverage:
    """Bind conversation subgroup requirements even when an outline states none.

    Only the static prediction template plans a static prediction primary,
    and it drafts no subgroup step, so such a plan has no route to an owner.
    """

    from ..canonical_json import canonical_sha256
    from ..contracts.prediction_execution import PREDICTION_PRIMARY_ACTION
    from .adjustment_authority import context_stated_variables

    subgroups = context_stated_variables(context, "stated_subgroups")
    answerable = bool(subgroups) and (
        bool(subgroup_actions_for_family(plan.analysis_type))
        or family_describes_by_group(plan.analysis_type)
    ) and not any(step.scientific_action_id == PREDICTION_PRIMARY_ACTION for step in plan.steps)
    records = {}
    for item in subgroups:
        identity = item.concept or item.variable
        owners = stated_subgroup_owner_steps(context, plan, item.concept) if item.concept else ()
        if item.source_status == "unavailable":
            status, reason = "unsupported", "source_concept_unavailable"
        elif item.source_status != "available" or item.concept is None:
            status, reason = "unsupported", "stated_subgroup_concept_unresolved"
        elif owners:
            status, reason = "covered", "typed_subgroup_owner_present"
        elif not answerable:
            status, reason = "unsupported", "stated_subgroup_owner_unavailable"
        else:
            status, reason = "missing", "typed_subgroup_owner_missing"
        records[identity] = PlanRequirementCoverageRecord(
            requirement_id=f"stated_subgroup:{identity}", kind="question", question_kind="subgroup",
            concept_identity=identity,
            source_ref="research_context.json.user_preferences.data_constraints.stated_subgroups",
            status=status, owner_step_ids=owners if status == "covered" else (), reason_code=reason,
        )
    return PlanRequirementCoverage(
        context_sha256=canonical_sha256(context.model_dump(mode="json")),
        plan_sha256=canonical_sha256(plan.model_dump(mode="json")),
        records=tuple(records.values()),
        complete=all(record.status == "covered" for record in records.values()),
    )


AnalysisRole = Literal["primary", "secondary", "sensitivity", "auxiliary"]


class RequiredPlanShape(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    key: str
    kind: Literal[
        "article_role", "primary_summary", "measurement_audit", "outcome", "comparison", "adjustment",
        "subgroup", "effect_measure",
    ]
    reason_code: str
    planned_analysis_role: AnalysisRole | None = None
    action: str | None = None
    module: str | None = None
    inputs: tuple[str, ...] = ()
    article_role: str | None = None
    effect_measure: Literal["odds_ratio", "risk_ratio"] | None = None




class PlanStructuralChange(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    finding_code: str
    remediation_index: int
    step_id: str
    remediation: PlanShapeRemediation


class PlanRevisionReceipt(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["easyicu.plan_revision_receipt/1"] = "easyicu.plan_revision_receipt/1"
    source_plan_sha256: str
    revised_plan_sha256: str
    structural_changes: tuple[PlanStructuralChange, ...] = ()


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
    from .scientific_action_catalog import scientific_action_for_id

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
    from ..schema import AnalysisStep

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
    requested = edited.requested
    excluded = set(edited.excluded) | set(prefill.excluded)
    deferred = set(edited.deferred) if first else set()
    if not first:
        requested += tuple(kind for kind in edited.deferred if kind not in excluded)
    elif not requested:
        # Older cards saved only the requested list. Keep known earlier
        # requests when the current question and edit allow it.
        prior = set(prefill.requested)
        if plan.deliverables is not None:
            prior.update(plan.deliverables.requested)
        deferred.update(prior.difference(excluded, prefill.check))
    unanswered = (
        set(edited.unanswered).difference(requested, deferred, prefill.excluded)
        | set(prefill.check).difference(requested, deferred, excluded)
    )
    excluded = (excluded | unanswered).difference(requested, deferred)
    selection = PlanDeliverables(
        requested=requested, confirmed=True, source="user_edit",
        deferred=tuple(kind for kind in HOST_DELIVERY_KINDS if kind in deferred),
        excluded=tuple(kind for kind in HOST_DELIVERY_KINDS if kind in excluded),
        unanswered=tuple(kind for kind in HOST_DELIVERY_KINDS if kind in unanswered),
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


__all__ = [
    "PlanDeliverableKind",
    "bind_confirmed_host_delivery",
    "bind_requested_host_delivery",
    "bind_run_deliverables",
    "planned_delivery_evidence",
    "planned_run_delivery_evidence",
    "prefill_plan_deliverables",
    "requested_plan_deliverables",
    "ANALYSIS_ROLES",
    "RequiredPlanShape",
    "PlanShapeRemediation",
    "PlanStructuralChange",
    "PlanRevisionReceipt",
    "SUBGROUP_ACTIONS",
    "PlanRequirementCoverage",
    "PlanRequirementCoverageRecord",
    "build_plan_requirement_coverage",
    "concept_relatives",
    "family_describes_by_group",
    "stated_adjustment_coverage",
    "stated_subgroup_coverage",
    "stated_subgroup_owner_steps",
    "subgroup_actions_for_family",
]
