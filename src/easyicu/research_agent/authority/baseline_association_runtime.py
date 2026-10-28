"""Digest-bound baseline association coordinates; no temporal eligibility bypass.

The Web projector proves baseline timing before signing. This owner checks the
complete model and binds its reviewed terms, dependence and subgroup roster to
the same digest used by the deterministic executor.
"""

from __future__ import annotations

import json
from typing import Literal, get_args

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..canonical_json import canonical_sha256
from ..contracts.association_execution import (
    ASSOCIATION_BINARY_SENSITIVITY_CAPABILITY_ID, association_execution_verdict, association_binary_sensitivity_plan_verdict,
    sole_primary_model_requirement,
)
from ..contracts.cohort_product_keys import sole_typed_cohort_input
from ..contracts.dependence import PlannedDependenceRequirement
from ..contracts.exposure_group_rules import PlannedGroupColumn, read_grouping_rules
from ..contracts.figure_plan import COHORT_FLOW_FIGURE_PANELS, DATA_QUALITY_FIGURE_PANELS
from ..contracts.model_terms import EffectModificationBinding, ModelTermSpec
from ..contracts.runtime_outcomes import RuntimeOutcomeContract
from ..schema import AnalysisPlan, AnalysisStep, ArtifactConsumptionContract

BASELINE_METHOD = "adjusted_association_models"
BASELINE_RISK_METHOD = "signed_baseline_absolute_risk"
BASELINE_CAPABILITY = "association_baseline_adjusted_v1"
BASELINE_STEPS = ("cohort_definition", "table_one", "measurement_audit", "adjusted_association",
                  "absolute_risk_context", "visualization", "report")


AssociationAuthorityReason = Literal[
    "association_adjustment_exposure_conflict",
    "baseline_association_roster_mismatch",
    "baseline_association_design_mismatch",
    "baseline_association_step_roster_mismatch",
]


class BaselineAssociationAuthorityError(ValueError):
    """A signed association refused a plan; only fixed codes and fields leave it."""

    reason_codes = frozenset(get_args(AssociationAuthorityReason))
    mismatch_field_names = (
        "primary_model", "method", "model_requirements", "covariates", "outcome_type",
        "exposure_source", "outcome", "dependence", "exposure_levels",
        "exposure_reference_level", "primary_contrast_level", "exposure_encoding",
        "scientific_action_id", "inputs", "effect_modifiers",
        "grouping.source_columns", "grouping.window",
        "step_roster", "scientific_capability", "signed_reference", "expected_outputs",
        "runtime_outcome_contract", "input_consumption_contracts",
    )

    def __init__(self, message: str, *, reason_code: AssociationAuthorityReason = "baseline_association_design_mismatch",
                 mismatch_fields: tuple[str, ...] = (), step_id: str = "adjusted_association"):
        super().__init__(message)
        self.reason_code = reason_code
        self._mismatch_fields = tuple(mismatch_fields)
        self.step_id = step_id if step_id in (*BASELINE_STEPS, "plan") else "plan"

    @property
    def mismatch_fields(self) -> tuple[str, ...]:
        """Bounded coordinate names, never exception text or compared values."""
        return tuple(field for field in self.mismatch_field_names if field in self._mismatch_fields)


def _refusal(message, *, reason_code="baseline_association_design_mismatch", mismatch_fields=(), step_id="adjusted_association"):
    return BaselineAssociationAuthorityError(message, reason_code=reason_code, mismatch_fields=mismatch_fields, step_id=step_id)


def _article_display_profile(step, core):
    """Recognize only the two source-bound article companions; never new fits.

    The seven scientific owners keep their order. A companion may render an
    existing product exactly once, with the same panels and all-row contract
    as its deterministic renderer. No raw input or scientific contract enters.
    """
    profiles = (
        ("figure:cohort_flow", ("table:cohort_flow",), COHORT_FLOW_FIGURE_PANELS),
        ("figure:data_quality", ("table:measurement_audit_missingness", "table:measurement_audit_process"), DATA_QUALITY_FIGURE_PANELS),
    )
    display_fields = {"step_id", "intent", "inputs", "expected_outputs", "method", "icu_rule_refs",
                      "input_consumption_contracts", "figure_panels"}
    if step.method != "visualization" or step.model_dump(exclude=display_fields, exclude_defaults=True, exclude_none=True):
        return None
    for output, sources, templates in profiles:
        if step.expected_outputs != [output] or tuple(step.inputs) != sources:
            continue
        if any(sum(source in owner.expected_outputs for owner in core) != 1 for source in sources):
            return None
        if step.input_consumption_contracts != [ArtifactConsumptionContract(input_key=source, mode="all_rows") for source in sources]:
            return None
        panels = [(panel.panel_id, panel.article_role, panel.chart_type, tuple(panel.source_products)) for panel in step.figure_panels]
        expected = [(panel.panel_id, panel.article_role, panel.chart_type, (source,)) for panel, source in zip(templates, sources)]
        if panels == expected and all(panel.figure_output == output and not panel.policy_alternative_chart_types for panel in step.figure_panels):
            return output
    return None


class BaselineAssociationRuntimeAuthority(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal["easyicu.baseline_association_runtime_authority/1"] = "easyicu.baseline_association_runtime_authority/1"
    authority_kind: Literal["baseline_adjusted_association"] = "baseline_adjusted_association"
    protocol_content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    execution_contract_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    exposure_column: str
    outcome_column: str
    exposure_levels: tuple[str, ...] = ()
    exposure_reference_level: str | None = None
    primary_contrast_level: str | None = None
    required_adjustment_columns: tuple[str, ...] = ()
    subgroup_terms: tuple[ModelTermSpec, ...] = ()
    dependence: PlannedDependenceRequirement | None = None
    exposure_encoding: Literal["treatment_contrast", "restricted_cubic_spline"]
    spline_knot_quantiles: tuple[float, float, float] = (0.1, 0.5, 0.9)
    interpretation: Literal["descriptive_prognostic_association_not_causal"] = "descriptive_prognostic_association_not_causal"
    model_requirement_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def _closed(self):
        if self.exposure_column == self.outcome_column or {self.exposure_column, self.outcome_column} & set(self.required_adjustment_columns):
            raise ValueError("baseline design columns overlap")
        if len(set(self.required_adjustment_columns)) != len(self.required_adjustment_columns):
            raise ValueError("baseline adjustment roster must be unique")
        if self.spline_knot_quantiles != (0.1, 0.5, 0.9):
            raise ValueError("baseline continuous encoding follows the landmark spline policy")
        if self.exposure_encoding == "treatment_contrast":
            if (len(self.exposure_levels) < 2 or len(set(self.exposure_levels)) != len(self.exposure_levels)
                or self.exposure_reference_level not in self.exposure_levels
                or self.primary_contrast_level not in self.exposure_levels
                or self.exposure_reference_level == self.primary_contrast_level):
                raise ValueError("baseline categorical contrast is not closed")
        elif self.exposure_levels or self.exposure_reference_level or self.primary_contrast_level:
            raise ValueError("a continuous baseline exposure has no categorical contrast")
        names = [term.name for term in self.subgroup_terms]
        if len(names) != len(set(names)) or {self.exposure_column, self.outcome_column} & set(names):
            raise ValueError("subgroups must be distinct from the exposure and outcome")
        if any(term.transform != "treatment_contrast" or not term.levels for term in self.subgroup_terms):
            raise ValueError("subgroups require declared treatment-coded levels")
        if canonical_sha256(self.model_dump(mode="json", exclude={"execution_contract_sha256"})) != self.execution_contract_sha256:
            raise ValueError("baseline authority digest mismatch")
        return self

    @property
    def plan_rule_ref(self):
        return f"scientific_runtime_contract:{self.execution_contract_sha256}"

    @property
    def required_columns(self):
        return tuple(dict.fromkeys((self.exposure_column, self.outcome_column, *self.required_adjustment_columns,
                                   *(term.name for term in self.subgroup_terms),
                                   *((self.dependence.group_source,) if self.dependence else ()))))

    def for_grouped_exposure(self, column: PlannedGroupColumn):
        """Re-sign an unreviewed source exposure from a run-verified grouping."""
        if self.model_requirement_sha256 or self.exposure_encoding != "restricted_cubic_spline":
            return self
        sources = set(read_grouping_rules(column.parameters).read_columns)
        if self.exposure_column not in sources:
            return self
        if sources != {self.exposure_column} or (column.window and column.window[1] > 0):
            raise _refusal("baseline grouping changes the signed source or observation window", mismatch_fields=tuple(
                field for field, differs in (("grouping.source_columns", sources != {self.exposure_column}),
                                            ("grouping.window", bool(column.window and column.window[1] > 0))) if differs))
        if {column.variable, *sources} & set(self.required_adjustment_columns):
            raise _refusal("baseline adjustment includes the exposure or its source",
                           reason_code="association_adjustment_exposure_conflict", mismatch_fields=("covariates",))
        return build_baseline_authority(**{
            **self.model_dump(mode="python", exclude={"execution_contract_sha256"}),
            "exposure_column": column.variable, "exposure_encoding": "treatment_contrast",
            "exposure_levels": tuple(str(code) for code in range(1, column.levels + 1)),
            "exposure_reference_level": str(column.contrast.reference),
            "primary_contrast_level": str(column.contrast.contrast),
        })

    def _primary(self, plan: AnalysisPlan):
        primary = [step for step in plan.steps if step.planned_analysis_role == "primary"]
        if len(primary) != 1:
            raise _refusal("baseline authority requires exactly one primary model", mismatch_fields=("primary_model",))
        return primary[0]

    def _requirement(self, primary: AnalysisStep):
        if primary.method != BASELINE_METHOD:
            raise _refusal("baseline primary estimator method drifted", mismatch_fields=("method",))
        requirement = sole_primary_model_requirement(primary)
        unsigned = primary.model_copy(update={
            "method": "adjusted_association_models",
            "scientific_action_id": "association.adjusted_association",
            "scientific_capability": "association_adjusted_v1",
            "model_requirements": [requirement.model_copy(update={"effect_modifiers": []})] if requirement else [],
        })
        verdict = association_execution_verdict(unsigned)
        if not verdict.claimed or requirement is None:
            raise _refusal("baseline primary model is incomplete: " + verdict.reason, mismatch_fields=("model_requirements",))
        if tuple(requirement.covariates or ()) != self.required_adjustment_columns:
            raise _refusal("baseline adjustment roster drifted from signed coordinates",
                           reason_code="baseline_association_roster_mismatch", mismatch_fields=("covariates",))
        mismatches = tuple(field for field, actual, expected in (
            ("outcome_type", requirement.outcome_type, "binary"),
            ("exposure_source", requirement.exposure_source, self.exposure_column),
            ("outcome", requirement.outcome, self.outcome_column),
            ("dependence", requirement.dependence, self.dependence),
        ) if actual != expected)
        if mismatches:
            raise _refusal("baseline primary model drifted from signed coordinates", mismatch_fields=mismatches)
        exposure = next(term for term in requirement.model_terms if term.role == "exposure")
        if self.exposure_encoding == "treatment_contrast":
            mismatches = tuple(field for field, actual, expected in (
                ("exposure_levels", tuple(exposure.levels or ()), self.exposure_levels),
                ("exposure_reference_level", exposure.reference_level, self.exposure_reference_level),
                ("primary_contrast_level", requirement.primary_contrast_level, self.primary_contrast_level),
            ) if actual != expected)
            if mismatches:
                raise _refusal("baseline exposure contrast drifted", mismatch_fields=mismatches)
        elif exposure.coding != "continuous" or exposure.transform != "identity":
            raise _refusal("baseline spline requires the continuous primary term intent", mismatch_fields=("exposure_encoding",))
        expected_action = "association.effect_modification" if self.subgroup_terms and primary.scientific_capability == BASELINE_CAPABILITY else "association.adjusted_association"
        if primary.scientific_action_id != expected_action or not set(self.required_columns).issubset(primary.inputs):
            raise _refusal("baseline subgroup parameter or input roster drifted", mismatch_fields=tuple(
                field for field, differs in (("scientific_action_id", primary.scientific_action_id != expected_action),
                                            ("inputs", not set(self.required_columns).issubset(primary.inputs))) if differs))
        if primary.scientific_capability == BASELINE_CAPABILITY and tuple(requirement.effect_modifiers) != self.subgroup_terms:
            raise _refusal("baseline subgroup parameter drifted", mismatch_fields=("effect_modifiers",))
        requirement = type(requirement).model_validate({
            **requirement.model_dump(mode="python"), "effect_modifiers": list(self.subgroup_terms),
        })
        if self.model_requirement_sha256 and canonical_sha256(requirement.model_dump(mode="json")) != self.model_requirement_sha256:
            raise _refusal("baseline reviewed model terms drifted", mismatch_fields=("model_requirements",))
        return requirement

    def seal_adjustment_roster(self, plan: AnalysisPlan):
        requirement = self._requirement(self._primary(plan))
        if self.model_requirement_sha256:
            return self
        return build_baseline_authority(**{
            **self.model_dump(mode="python", exclude={"execution_contract_sha256"}),
            "model_requirement_sha256": canonical_sha256(requirement.model_dump(mode="json")),
        })

    def bind_plan(self, plan: AnalysisPlan):
        if not self.model_requirement_sha256:
            raise _refusal("baseline model must be sealed from the reviewed plan", mismatch_fields=("signed_reference",))
        if self._primary(plan).scientific_capability == BASELINE_CAPABILITY:
            self.validate_plan(plan)
            return plan
        requirement = self._requirement(self._primary(plan))
        steps = []
        for step in plan.steps:
            if (step.scientific_capability == ASSOCIATION_BINARY_SENSITIVITY_CAPABILITY_ID
                    and step.method in {"first_stay_association", "restricted_cubic_spline_sensitivity"}):
                extra = [key for key in step.inputs if ":" not in key and key not in self._primary(plan).inputs]
                step = step.model_copy(update={"inputs": list(dict.fromkeys([
                    "artifact:analysis_cohort", *self.required_columns, *extra, "table:adjusted_association_estimates",
                ]))})
            if step.step_id in {"adjusted_association", "absolute_risk_context"}:
                primary = step.step_id == "adjusted_association"
                inputs = ["artifact:analysis_cohort", *self.required_columns]
                if not primary:
                    inputs.append("table:adjusted_association_estimates")
                step = step.model_copy(update={
                    "method": BASELINE_METHOD if primary else BASELINE_RISK_METHOD,
                    "scientific_capability": BASELINE_CAPABILITY if primary else step.scientific_capability,
                    "scientific_action_id": ("association.effect_modification" if self.subgroup_terms else "association.adjusted_association") if primary else step.scientific_action_id,
                    "inputs": inputs,
                    "icu_rule_refs": list(dict.fromkeys([*step.icu_rule_refs, self.plan_rule_ref])),
                    "runtime_outcome_contract": RuntimeOutcomeContract(owner_ref=self.plan_rule_ref, outcomes=(self.outcome_column,)),
                    "input_consumption_contracts": [ArtifactConsumptionContract(input_key=key, mode="all_rows") for key in inputs if ":" in key],
                    **({"model_requirements": [requirement]} if primary else {}),
                })
            steps.append(step)
        bound = plan.model_copy(update={"steps": steps})
        self.validate_plan(bound)
        return bound

    def governed_primary_step(self, plan: AnalysisPlan):
        primary = self._primary(plan)
        if primary.method != BASELINE_METHOD:
            raise _refusal("baseline signed primary is absent", mismatch_fields=("method",))
        if primary.scientific_capability != BASELINE_CAPABILITY:
            raise _refusal("baseline primary capability drifted", mismatch_fields=("scientific_capability",))
        self._requirement(primary)
        if self.plan_rule_ref not in primary.icu_rule_refs or not self.model_requirement_sha256:
            raise _refusal("baseline signed reference is absent", mismatch_fields=("signed_reference",))
        return primary

    def validate_plan(self, plan: AnalysisPlan):
        primary = self.governed_primary_step(plan)
        required_steps = BASELINE_STEPS
        if getattr(plan.design_selection, "execution_tier", "article") == "first_result":
            required_steps = BASELINE_STEPS[:6] if any(step.step_id == "visualization" for step in plan.steps) else BASELINE_STEPS[:5]
        checks = [step for step in plan.steps if self._prespecified_check(step, plan)]
        remaining = [step for step in plan.steps if step not in checks]
        core, companions = remaining[:len(required_steps)], remaining[len(required_steps):]
        profiles = [_article_display_profile(step, core) for step in companions]
        if (tuple(step.step_id for step in core) != required_steps
            or (companions and required_steps != BASELINE_STEPS)
            or None in profiles or len(profiles) != len(set(profiles))
            or any(plan.steps.index(step) <= plan.steps.index(next(s for s in plan.steps if s.step_id == "absolute_risk_context"))
                   or any(plan.steps.index(step) > plan.steps.index(s) for s in core if s.step_id in {"visualization", "report"})
                   for step in checks)
            or len({step.step_id for step in plan.steps}) != len(plan.steps)):
            raise _refusal("baseline template step roster drifted", reason_code="baseline_association_step_roster_mismatch",
                           mismatch_fields=("step_roster",), step_id="plan")
        if sole_typed_cohort_input(primary) != "artifact:analysis_cohort" or primary.expected_outputs != ["table:adjusted_association_estimates"]:
            raise _refusal("baseline primary product or cohort drifted", mismatch_fields=("expected_outputs", "inputs"))
        if primary.inputs != ["artifact:analysis_cohort", *self.required_columns]:
            raise _refusal("baseline primary input roster drifted", mismatch_fields=("inputs",))
        risk = plan.steps[4]
        if (risk.method != BASELINE_RISK_METHOD or self.plan_rule_ref not in risk.icu_rule_refs
            or risk.expected_outputs != ["table:absolute_risk_context"]
            or risk.inputs != ["artifact:analysis_cohort", *self.required_columns, "table:adjusted_association_estimates"]):
            raise _refusal("baseline absolute-risk binding drifted", mismatch_fields=("inputs", "expected_outputs", "signed_reference"), step_id=risk.step_id)
        for step in (primary, risk):
            if step.runtime_outcome_contract != RuntimeOutcomeContract(owner_ref=self.plan_rule_ref, outcomes=(self.outcome_column,)):
                raise _refusal("baseline runtime outcome binding drifted", mismatch_fields=("runtime_outcome_contract",), step_id=step.step_id)
            for key in (key for key in step.inputs if ":" in key):
                if not any(contract.input_key == key and contract.mode == "all_rows" for contract in step.input_consumption_contracts):
                    raise _refusal("baseline evidence consumption binding drifted", mismatch_fields=("input_consumption_contracts",), step_id=step.step_id)

    def _prespecified_check(self, step, plan):
        """Admit only exact inherited refits and the stated modifier companion."""
        if step.family_primary_result_requirement is not None:
            return False
        if (step.step_id == "prespecified_effect_modification"
                and step.method == "prespecified_effect_modification"
                and step.scientific_action_id == "association.effect_modification"
                and step.planned_analysis_role == "secondary"
                and self.subgroup_terms
                and step.expected_outputs == ["table:prespecified_effect_modification"]):
            expected = self._requirement(self._primary(plan)).model_copy(update={
                "requirement_id": step.step_id, "analysis_role": "secondary",
            })
            try:
                requirement = EffectModificationBinding.requirement_for(step, self._primary(plan).model_requirements[0])
            except ValueError:
                return False
            return (not step.model_requirements and requirement == expected and
                    set(step.inputs) == {"artifact:analysis_cohort", "table:adjusted_association_estimates", *self.required_columns})
        if (step.model_requirements or self.subgroup_terms or self.exposure_encoding != "treatment_contrast"
                or step.method not in {"first_stay_association", "restricted_cubic_spline_sensitivity"}
                or not association_binary_sensitivity_plan_verdict(step, plan_steps=plan.steps).claimed
                or len(step.sensitivity_spec_ids) != 1
                or {key for key in step.inputs if ":" in key} != {"artifact:analysis_cohort", "table:adjusted_association_estimates"}):
            return False
        form = step.functional_form_spec
        if step.method == "restricted_cubic_spline_sensitivity":
            requirement = self._primary(plan).model_requirements[0]
            return (form is not None and form.comparison == "restricted_cubic_spline_vs_linear"
                    and form.target_column in {term.name for term in requirement.model_terms
                                               if term.role == "covariate" and term.coding == "continuous"})
        return form is None and len({key for key in step.inputs if ":" not in key} - set(self.required_columns)) == 1

    def planner_disclosure(self):
        return "BASELINE ASSOCIATION: preserve the signed exposure, outcome, adjustment and subgroup rosters.\n" + self.model_dump_json()


def build_baseline_authority(**coordinates) -> BaselineAssociationRuntimeAuthority:
    draft = BaselineAssociationRuntimeAuthority.model_construct(**coordinates)
    body = draft.model_dump(mode="json", exclude={"execution_contract_sha256"})
    body["execution_contract_sha256"] = canonical_sha256(body)
    return BaselineAssociationRuntimeAuthority.model_validate_json(json.dumps(body))
