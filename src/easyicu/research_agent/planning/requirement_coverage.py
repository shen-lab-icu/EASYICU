"""Digest-bound coverage projection for explicit study requirements.

The scientific-review owner has already decided whether a typed outcome,
sensitivity, or exact covariate is covered, and the question-requirements
owner (``planning.question_requirements``) whether an analysis the question
asks for is.  This module records that decision as a stable row and binds it
to the exact context and plan.  It does not infer requirements from prose or
grant execution/reporting authority.

The structured adjustment reading is a required subset independent of exact
rosters. Its coverage is checked here against primary fitted covariate terms.
A stated subgroup is checked here too, against the analysis steps that
answer it in the plan's own family; review inside the execution kernel reads
both without the Planner-facing question-requirement owner.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, Mapping, Optional, Sequence

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..methods.descriptive_inputs import REQUESTED_CONCEPT_NO_DATA, context_columns_without_data
from .adjustment_authority import concept_relatives
from ..contracts.analysis_design import PlanShapeRemediation as PlanShapeRemediation

if TYPE_CHECKING:
    from ..schema import AnalysisPlan, ResearchContext


RequirementKind = Literal["outcome", "sensitivity", "covariate", "question"]
RequirementCoverageStatus = Literal["covered", "missing", "unsupported"]


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
    reason_code: str = Field(min_length=3, max_length=160)

    @model_validator(mode="after")
    def _owner_status_is_consistent(self) -> "PlanRequirementCoverageRecord":
        if len(self.owner_step_ids) != len(set(self.owner_step_ids)):
            raise ValueError("requirement coverage owner step ids must be unique")
        if self.status == "covered" and not self.owner_step_ids:
            raise ValueError("covered requirement must name at least one owner step")
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
        }) if context_columns_without_data(context, (record.concept_identity,)) else record
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


__all__ = [
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
