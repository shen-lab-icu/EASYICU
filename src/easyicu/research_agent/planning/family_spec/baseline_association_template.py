"""Baseline association plans and exact, separately approved result additions.

The host fixes the layout. A subgroup is a parameter of the primary interaction
model, never a second stratified fit. Continuous exposures carry the same term
intent as the landmark spline template; the signed runtime owns their basis.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Callable, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ...authority.baseline_association_runtime import BaselineAssociationRuntimeAuthority, build_baseline_authority
from ...contracts.model_terms import ModelTermSpec
from ...schema import AnalysisPlan, AnalysisStep, ArtifactConsumptionContract, ResearchContext
from ..sensitivity_authority import PrespecifiedSensitivitySpec
from ..sensitivity_plan_shaping import ensure_prespecified_sensitivity_steps

from ...canonical_json import canonical_sha256
from ..design_selection import ResearchDesignCandidate, ResearchDesignSelection
from ..progressive_contract import (
    ProgressiveDisplayLabel, ProgressiveFoundationMaterialization,
    ProgressiveModelTermIntent, ProgressiveOutputIntent, ProgressivePlanFoundation,
    ProgressivePlanOutline, ProgressiveSkeletonStep, ProgressiveTableOneVariable,
)
from .contract import (
    BASELINE_ASSOCIATION_FAMILY_ID, FamilyPlanSpec, FamilySpecError, FamilySpecRequest,
    table_one_group_column,
)
from .contract import first_result_subset
from .landmark_categorical_template import (
    FamilySkeletonDraft, _cohort_intent, _covariate_terms, _label,
    _materialization, _MEASUREMENT_AUDIT_OUTPUTS, _method_card_elements,
    _outline_step, _ref, _step_bindings, _table_one_summary,
)


def build_baseline_association_skeleton(
    request: FamilySpecRequest, spec: FamilyPlanSpec, *,
    bind_outline: Callable[[ProgressivePlanOutline], ProgressivePlanOutline] | None = None,
) -> FamilySkeletonDraft:
    if request.family_id != BASELINE_ASSOCIATION_FAMILY_ID:
        raise FamilySpecError("family_spec_template_mismatch", "Expected a baseline association request.")
    if spec.request_sha256 != request.request_sha256:
        raise FamilySpecError("family_spec_request_digest_mismatch", "The spec does not bind this request.")
    exposure, outcome = request.primary_exposure, request.outcome
    terms = _covariate_terms(request, spec)
    covariates = [term.name for term in terms]
    continuous = request.exposure_kind == "continuous"
    table_one_group = table_one_group_column(request)
    group_reason = (
        "the exposure is continuous and no threshold grouping was stated"
        if continuous else "the exposure has declared groups"
    )
    table_one_decision = (
        f"Host decision: Table 1 is grouped by {_label(spec, table_one_group)} "
        f"because {group_reason}."
    )
    subgroups = request.subgroup_columns
    # The ordinary compiler validates the primary model intent first. The
    # runtime owner then binds its typed modifiers and interaction action
    # before scientific review and approval, as for the landmark families.
    action = "association.adjusted_association"
    roster = list(dict.fromkeys([request.identity_column, exposure, outcome, *covariates, *subgroups]))
    labels = [_label(spec, name) for name in covariates]
    adjustment = "adjusted for " + ", ".join(labels) if labels else "without adjustment covariates"
    encoding = "restricted cubic spline (10/50/90 knots, median reference)" if continuous else "declared treatment contrasts"
    estimand = f"Observational association of {_label(spec, exposure)} with {_label(spec, outcome)}, {adjustment}."
    keys = list(dict.fromkeys([
        *(key for key in request.allowed_literature_citation_keys if _method_card_elements(key)),
        *request.direct_comparator_literature_keys,
    ]))[:12]
    selected = ResearchDesignCandidate(
        design_id=BASELINE_ASSOCIATION_FAMILY_ID, analysis_type="association_study",
        estimand=estimand, time_zero="ICU admission; the exposure is a baseline attribute.",
        observation_window=f"From ICU admission through ascertainment of {_label(spec, outcome)}.",
        primary_method=f"Logistic model with {encoding}, {adjustment}.",
        required_variables=roster,
        assumptions=["The exposure is known at baseline.", "Inference follows the declared dependence contract."],
        literature_citation_keys=keys[:8], literature_design_decisions=spec.literature_design_decisions,
        novelty_positioning="Compare the observational design with the screened sources without claiming novelty.",
        figure_role="One figure presents the primary association and any stated stratum estimates.",
        supports=estimand, cannot_prove="Observational associations do not establish causation or transportability.",
        reviewable_plan=[
            "Apply the typed cohort eligibility and report selection counts.",
            f"Describe baseline characteristics and measurement completeness. {table_one_decision}",
            f"Fit one logistic model with {encoding}, {adjustment}.",
            "Report complete-case denominators and absolute risks with the count cross-check.",
            "Report stratum-specific contrasts and the interaction test from the same model." if subgroups
            else "Report the prespecified primary contrast without additional exploratory models.",
            "Present one figure and a report with observational conclusion ceilings.",
        ], disposition="selected",
        decision_reason="A baseline attribute is available at admission, so no later landmark restriction is required.",
    )
    rejected = selected.model_copy(update={
        "design_id": "baseline_unadjusted_description", "disposition": "rejected",
        "primary_method": "Exposure-outcome counts and proportions only.",
        "estimand": "Unadjusted outcome frequencies by the baseline exposure.",
        "supports": "Descriptive counts and proportions only.", "reviewable_plan": None,
        "decision_reason": "Description alone does not answer the requested model-based association.",
    })
    objectives = {
        "cohort_definition": "Apply typed cohort eligibility and publish the analysis cohort and selection counts.",
        "table_one": f"Describe baseline characteristics with standardized differences only. {table_one_decision}",
        "measurement_audit": "Audit source, completeness, missingness, measurement process and analytic denominators.",
        "adjusted_association": f"Estimate {estimand} Use {encoding}." + (
            " Report stratum estimates and the interaction test from this same model." if subgroups else ""
        ),
        "absolute_risk_context": "Report absolute risks and risk differences in the primary model population, cross-checked against counts.",
        "visualization": "Present the primary association and any stated stratum estimates in one figure."
        + (" Pair them with observed outcome risks and denominators by exposure group." if not continuous else ""),
        "report": "Report sources, cohort flow, denominators, estimates and limitations under the observational conclusion ceiling.",
    }
    dependencies = {
        "cohort_definition": [], "table_one": ["cohort_definition"],
        "measurement_audit": ["cohort_definition"],
        "adjusted_association": ["cohort_definition", "measurement_audit"],
        "absolute_risk_context": ["adjusted_association"],
        "visualization": ["adjusted_association", "absolute_risk_context"],
        "report": list(objectives)[:-1],
    }
    roles = {"adjusted_association": "primary", "absolute_risk_context": "secondary"}
    outline_steps = [
        _outline_step(
            step_id=step_id, role=roles.get(step_id, "auxiliary"), module_id=step_id,
            objective=objective, depends_on=dependencies[step_id], variable_names=roster,
            citations=keys if step_id in roles else [],
            action=action if step_id == "adjusted_association" else None,
            population_scope="primary_model" if step_id == "absolute_risk_context" else None,
        )
        for step_id, objective in objectives.items()
    ]
    outline = ProgressivePlanOutline(
        analysis_type="association_study", cohort_objective=estimand,
        design_selection=ResearchDesignSelection(candidates=[selected, rejected]),
        steps=outline_steps, rationale="The host fixes the steps and adjustment roster; the spec supplies reader labels.",
    )
    if bind_outline is not None:
        outline = bind_outline(outline)
    bound = {step.step_id: step for step in outline.steps}
    def common(key):
        return dict(step_id=key, module_id=key, objective=objectives[key],
                    planned_analysis_role=roles.get(key, "auxiliary"), depends_on=dependencies[key])
    steps = [
        ProgressiveSkeletonStep(**common("cohort_definition"), raw_inputs=roster),
        ProgressiveSkeletonStep(
            **common("table_one"), raw_inputs=list(dict.fromkeys([exposure, outcome, *covariates, *subgroups])),
            table_one_group_by=table_one_group, table_one_mode="descriptive_smd_only",
            table_one_variables=[
                *([] if continuous else [ProgressiveTableOneVariable(name=outcome, summary="count_percent")]),
                *([ProgressiveTableOneVariable(name=exposure, summary="median_iqr")] if continuous else []),
                *(ProgressiveTableOneVariable(name=term.name, summary=_table_one_summary(term.coding)) for term in terms),
                *(ProgressiveTableOneVariable(name=name, summary="count_percent") for name in subgroups if name not in covariates),
            ],
        ),
        ProgressiveSkeletonStep(
            **common("measurement_audit"), raw_inputs=roster[1:],
            product_inputs=[_ref("cohort_definition", "artifact:analysis_cohort")],
            outputs=[ProgressiveOutputIntent(product_id=product, semantic_role=role) for product, role in _MEASUREMENT_AUDIT_OUTPUTS],
        ),
        ProgressiveSkeletonStep(
            **common("adjusted_association"), raw_inputs=roster,
            product_inputs=[_ref("cohort_definition", "artifact:analysis_cohort")],
            scientific_action_id=action, primary_exposure=exposure, outcome=outcome, outcome_type="binary",
            model_terms=[ProgressiveModelTermIntent(
                name=exposure, role="exposure", coding="continuous" if continuous else "categorical" if len(request.exposure_levels) > 2 else "binary",
                reference_level_index=None if continuous else request.reference_level_index,
            ), *terms], event_level_index=request.event_level_index,
            reference_exposure_level_index=None if continuous else request.reference_level_index,
            comparison_exposure_level_index=None if continuous else request.primary_contrast_level_index,
            primary_contrast_level_index=None if continuous else request.primary_contrast_level_index,
            confidence_level=0.95,
            literature_bindings=_step_bindings(bound["adjusted_association"], module="adjusted_association",
                                               coordinate="baseline primary association", comparator_applications=spec.applications),
        ),
        ProgressiveSkeletonStep(
            **common("absolute_risk_context"), raw_inputs=[exposure, outcome],
            product_inputs=[_ref("adjusted_association", "table:adjusted_association_estimates")],
            primary_exposure=exposure, outcome=outcome, population_scope="primary_model",
            literature_bindings=_step_bindings(bound["absolute_risk_context"], module="absolute_risk_context",
                                               coordinate="primary population absolute risks", comparator_applications=spec.applications),
        ),
        ProgressiveSkeletonStep(
            **common("visualization"), raw_inputs=[],
            product_inputs=[_ref("adjusted_association", "table:adjusted_association_estimates"),
                            *([] if continuous else [_ref("absolute_risk_context", "table:absolute_risk_context")])],
            outputs=[ProgressiveOutputIntent(product_id="figure:visualization", semantic_role="figure")],
        ),
        ProgressiveSkeletonStep(
            **common("report"), raw_inputs=[],
            product_inputs=[_ref("cohort_definition", "table:cohort_flow"), _ref("table_one", "table:table_one"),
                            *(_ref("measurement_audit", product) for product, _ in _MEASUREMENT_AUDIT_OUTPUTS),
                            _ref("adjusted_association", "table:adjusted_association_estimates"),
                            _ref("absolute_risk_context", "table:absolute_risk_context"),
                            _ref("visualization", "figure:visualization")],
            outputs=[ProgressiveOutputIntent(product_id="report:report", semantic_role="report")],
        ),
    ]
    return first_result_subset(FamilySkeletonDraft(
        outline=outline,
        foundation=ProgressiveFoundationMaterialization(
            outline_sha256=canonical_sha256(outline.model_dump(mode="json")),
            foundation=ProgressivePlanFoundation(
                cohort=_cohort_intent(request, spec.population),
                display_labels=[ProgressiveDisplayLabel(key=key, value=value) for key, value in spec.labels.items()],
                robustness_intents=[], know_how_decisions=[],
            ),
        ),
        materializations=tuple(_materialization(bound[step.step_id], step) for step in steps),
    ), request)


# Result additions have their own typed roster and signed runtime. They reuse
# sealed source fields without rerunning the Planner or widening the source run.
AppendKind = Literal["sensitivity", "exposure", "subgroup", "article"]
LABELS = {
    "sensitivity": ("Add a sensitivity analysis", "加敏感性分析"),
    "exposure": ("Change the exposure definition", "换暴露定义"),
    "subgroup": ("Add a subgroup", "加亚组"),
    "article": ("Extend to a full article", "完善为论文级分析"),
}


@dataclass(frozen=True)
class AppendCompilation:
    kind: AppendKind
    plan: AnalysisPlan
    model_authority: BaselineAssociationRuntimeAuthority | None = None
    exposure_source: str | None = None


def _plan(source, steps, *, article=False):
    design = source.design_selection
    if article:
        if design is None or design.execution_tier != "first_result":
            raise ValueError("An article extension requires a first-result run.")
        design = design.model_copy(update={"execution_tier": "article"})
    return AnalysisPlan.model_validate({
        **source.model_dump(mode="python"), "steps": steps,
        "revision": source.revision + 1, "design_selection": design,
    })


def _model_step(source, authority, *, kind, requirement, intent):
    original = authority.governed_primary_step(source)
    step = original.model_copy(deep=True, update={
        "step_id": f"append_{kind}", "intent": intent,
        # Product names are scoped by run id. The original table remains in
        # its source run; this contract's fixed result name is a new product.
        "expected_outputs": ["table:adjusted_association_estimates"],
        "model_requirements": [requirement],
    })
    return step


def _authority(authority, requirement, *, exposure_encoding=None):
    exposure = next(term for term in requirement.model_terms if term.role == "exposure")
    return build_baseline_authority(**{
        **authority.model_dump(mode="python", exclude={"execution_contract_sha256"}),
        "model_requirement_sha256": None,
        "exposure_column": requirement.exposure_source,
        "exposure_encoding": exposure_encoding or authority.exposure_encoding,
        "exposure_levels": tuple(exposure.levels or ()),
        "exposure_reference_level": exposure.reference_level,
        "primary_contrast_level": requirement.primary_contrast_level,
        "subgroup_terms": tuple(requirement.effect_modifiers),
    })


@dataclass(frozen=True)
class SensitivityAppend:
    @staticmethod
    def parent_step(source, authority):
        primary = authority.governed_primary_step(source)
        if authority.exposure_encoding != "treatment_contrast" or authority.subgroup_terms:
            raise ValueError("The sensitivity owner needs a categorical model without existing interactions.")
        # The generic sensitivity owner implements this exact linear model.
        # Supply its capability spelling while retaining every model term and
        # the source product's approval/digest in the append runtime contract.
        return primary.model_copy(deep=True, update={"scientific_capability": "association_adjusted_v1"})

    def compile(self, source: AnalysisPlan, context: ResearchContext, authority) -> AppendCompilation:
        from ...contracts.association_execution import association_binary_sensitivity_plan_verdict

        primary = self.parent_step(source, authority)
        # The current refit owner is linear in the source exposure and does not
        # carry interaction terms. Do not silently refit a different model.
        preferences = context.user_preferences
        specs = list(preferences.sensitivity_specs if preferences else ())
        # A functional-form check has a closed, reviewed target in the source
        # model. The proposal names that target before its one approval.
        specs += [PrespecifiedSensitivitySpec(
            spec_id="append_form_" + term.name, axis="functional_form",
            strategy="restricted_cubic_spline", execution_variables=(term.name,),
        ) for term in primary.model_requirements[0].model_terms
            if term.role == "covariate" and term.coding == "continuous"]
        from ...schema import UserPreferences

        known = {variable.name for variable in context.variables}
        for spec in specs:
            if spec.strategy not in {"first_stay", "restricted_cubic_spline"} or not set(spec.execution_variables) <= known:
                continue
            if len(spec.execution_variables) != 1:
                continue
            if spec.strategy == "restricted_cubic_spline" and spec.execution_variables[0] not in {
                term.name for term in primary.model_requirements[0].model_terms if term.role == "covariate" and term.coding == "continuous"
            }:
                continue
            if spec.strategy == "first_stay" and set(spec.execution_variables) & set(authority.required_columns):
                continue
            scoped = (preferences or UserPreferences()).model_copy(deep=True, update={"sensitivity_specs": [spec]})
            shaped, _ = ensure_prespecified_sensitivity_steps(
                plan=source, context=context.model_copy(update={"user_preferences": scoped}),
            )
            old_ids = {step.step_id for step in source.steps}
            additions = [step for step in shaped.steps if step.step_id not in old_ids]
            if len(additions) != 1:
                continue
            step = additions[0]
            if spec.strategy == "restricted_cubic_spline" and len(spec.execution_variables) == 1:
                step = AnalysisStep.model_validate({**step.model_dump(mode="python"), "functional_form_spec": {
                    "target_column": spec.execution_variables[0], "comparison": "restricted_cubic_spline_vs_linear",
                    "knot_quantiles": [0.1, 0.5, 0.9],
                }})
            if association_binary_sensitivity_plan_verdict(step, plan_steps=[primary, step]).claimed:
                # The ordinary capability requires an in-plan parent. This
                # run's parent is instead sealed by ResultAppendRuntimeAuthority;
                # only that runtime may dispatch the inherited refit owner.
                step = step.model_copy(update={"scientific_capability": None})
                return AppendCompilation("sensitivity", _plan(source, [step]))
        raise ValueError("No sensitivity with sealed inputs is executable; state a sensitivity in the analysis owner.")


@dataclass(frozen=True)
class ExposureAppend:
    def compile(self, source: AnalysisPlan, context: ResearchContext, authority) -> AppendCompilation:
        from ...contracts.exposure_group_rules import EXPOSURE_GROUP_LABELS_KEY, read_grouping_rules

        primary = authority.governed_primary_step(source)
        records = context.cohort.provenance.get(EXPOSURE_GROUP_LABELS_KEY, ())
        rows = [row for row in records if row.get("variable") == authority.exposure_column]
        if len(rows) != 1:
            raise ValueError("State a different exposure definition in the exposure owner first.")
        rules = read_grouping_rules(rows[0]["parameters"])
        columns = tuple(dict.fromkeys(rules.read_columns))
        known = {variable.name for variable in context.variables}
        if len(columns) != 1 or columns[0] not in known or authority.subgroup_terms:
            raise ValueError("The exposure owner needs one sealed numeric source and no existing interactions.")
        raw = columns[0]
        dtype = next(variable.dtype for variable in context.variables if variable.name == raw) or ""
        if not dtype.lower().startswith(("int", "uint", "float", "double", "decimal", "number", "numeric")):
            raise ValueError("The alternative exposure requires a sealed numeric source column.")
        requirement = primary.model_requirements[0]
        if raw in {requirement.outcome, *requirement.covariates} or "append_exposure_value" in known:
            raise ValueError("The alternative exposure overlaps a bound model variable.")
        exposure = ModelTermSpec(name="append_exposure_value", role="exposure", coding="continuous", transform="identity")
        requirement = type(requirement).model_validate({
            **requirement.model_dump(mode="python"), "requirement_id": "append_exposure_model",
            "exposure_source": exposure.name, "exposure_levels": None,
            "exposure_reference_level": None, "primary_contrast_level": None,
            "model_terms": [exposure, *(term for term in requirement.model_terms if term.role != "exposure")],
        })
        runtime = _authority(authority, requirement, exposure_encoding="restricted_cubic_spline")
        derive = AnalysisStep(
            step_id="append_exposure_definition", planned_analysis_role="auxiliary",
            method="append_exposure_definition", intent=f"Use the continuous source {raw} as a different exposure definition.",
            inputs=["artifact:analysis_cohort", raw], expected_outputs=["artifact:append_exposure_cohort"],
        )
        step = _model_step(source, authority, kind="exposure", requirement=requirement,
                           intent="Estimate a different exposure definition; retain the previously reported estimate.")
        step.inputs = ["artifact:append_exposure_cohort", *runtime.required_columns]
        step.input_consumption_contracts = [ArtifactConsumptionContract(input_key=step.inputs[0], mode="all_rows")]
        risk = AnalysisStep(
            step_id="append_exposure_absolute_risk", planned_analysis_role="secondary",
            method="append_exposure_absolute_risk",
            intent="Report absolute risks for the different exposure definition beside its new association estimate.",
            inputs=["artifact:append_exposure_cohort", "table:adjusted_association_estimates"],
            expected_outputs=["table:append_exposure_absolute_risk"],
        )
        return AppendCompilation("exposure", _plan(source, [derive, step, risk]), runtime, raw)


@dataclass(frozen=True)
class SubgroupAppend:
    def compile(self, source: AnalysisPlan, context: ResearchContext, authority) -> AppendCompilation:
        primary = authority.governed_primary_step(source)
        requirement = primary.model_requirements[0]
        existing = {term.name for term in requirement.effect_modifiers}
        eligible = [term for term in requirement.model_terms if term.role == "covariate"
                    and term.transform == "treatment_contrast" and term.levels and term.name not in existing]
        if not eligible:
            raise ValueError("No eligible stratifier is sealed; choose one in the subgroup owner.")
        term = eligible[0]
        requirement = type(requirement).model_validate({
            **requirement.model_dump(mode="python"), "requirement_id": "append_subgroup_model",
            "effect_modifiers": [*requirement.effect_modifiers, term],
        })
        runtime = _authority(authority, requirement)
        step = _model_step(source, authority, kind="subgroup", requirement=requirement,
                           intent=f"Report {term.name} stratum estimates and the interaction test from the same model.")
        step.scientific_action_id = "association.effect_modification"
        step.inputs = ["artifact:analysis_cohort", *runtime.required_columns]
        return AppendCompilation("subgroup", _plan(source, [step]), runtime)


@dataclass(frozen=True)
class ArticleAppend:
    def compile(self, source: AnalysisPlan, context: ResearchContext, authority) -> AppendCompilation:
        # These are the report inputs of the baseline article family. Their
        # producers remain in the source run; the article run cites the bytes.
        products = [output for step in source.steps for output in step.expected_outputs
                    if output.startswith(("table:", "figure:"))]
        if not products:
            raise ValueError("The article owner requires registered result products.")
        steps = []
        if not any(product.startswith("figure:") for product in products):
            steps.append(AnalysisStep(
                step_id="append_article_figure", planned_analysis_role="auxiliary", method="adjusted_association_figure",
                intent="Present the existing primary association and its interval without refitting the model.",
                inputs=["table:adjusted_association_estimates"], expected_outputs=["figure:append_article_figure"],
            ))
            products.append("figure:append_article_figure")
        step = AnalysisStep(
            step_id="append_article_report", planned_analysis_role="auxiliary", method="scientific_reporting",
            intent="Extend this study to article-tier reporting of its sources, cohort flow, denominators, estimates and limitations.",
            inputs=products, expected_outputs=["report:append_article"],
        )
        return AppendCompilation("article", _plan(source, [*steps, step], article=True))


OWNERS = {"sensitivity": SensitivityAppend(), "exposure": ExposureAppend(),
          "subgroup": SubgroupAppend(), "article": ArticleAppend()}


def compile_result_append(*, kind: AppendKind, source: AnalysisPlan, context: ResearchContext, authority) -> AppendCompilation:
    authority.validate_plan(source)
    if kind not in OWNERS:
        raise ValueError("Unknown result append kind.")
    compiled = OWNERS[kind].compile(source, context, authority)
    known = {variable.name for variable in context.variables}
    if any(value not in known and value != "append_exposure_value" for step in compiled.plan.steps
           for value in step.inputs if ":" not in value):
        raise ValueError("The append needs an input absent from the sealed extraction.")
    return compiled


class ResultAppendRuntimeAuthority(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["easyicu.result_append_runtime/1"] = "easyicu.result_append_runtime/1"
    authority_kind: Literal["result_append"] = "result_append"
    kind: AppendKind
    source_run_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,159}$")
    source_run_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_plan_json: str
    source_context_json: str
    baseline: BaselineAssociationRuntimeAuthority
    append_plan_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    execution_contract_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def _closed(self):
        if canonical_sha256(self.model_dump(mode="json", exclude={"execution_contract_sha256"})) != self.execution_contract_sha256:
            raise ValueError("append runtime digest mismatch")
        if canonical_sha256(self.compilation().plan.model_dump(mode="json")) != self.append_plan_sha256:
            raise ValueError("append runtime roster differs from its typed owner")
        return self

    def compilation(self):
        return compile_result_append(
            kind=self.kind, source=AnalysisPlan.model_validate_json(self.source_plan_json),
            context=ResearchContext.model_validate_json(self.source_context_json), authority=self.baseline,
        )

    def validate_plan(self, plan: AnalysisPlan):
        if canonical_sha256(plan.model_dump(mode="json")) != self.append_plan_sha256:
            raise ValueError("append runtime roster changed")

    def bind_plan(self, plan: AnalysisPlan):
        self.validate_plan(plan)
        return plan

    @classmethod
    def create(cls, *, kind, source_run_id, source_run_sha256, source_plan, context, baseline):
        compiled = compile_result_append(kind=kind, source=source_plan, context=context, authority=baseline)
        body = dict(
            schema_version="easyicu.result_append_runtime/1", authority_kind="result_append", kind=kind,
            source_run_id=source_run_id, source_run_sha256=source_run_sha256,
            source_plan_json=source_plan.model_dump_json(), source_context_json=context.model_dump_json(),
            baseline=baseline.model_dump(mode="json"), append_plan_sha256=canonical_sha256(compiled.plan.model_dump(mode="json")),
        )
        return cls.model_validate_json(json.dumps({**body, "execution_contract_sha256": canonical_sha256(body)}))
