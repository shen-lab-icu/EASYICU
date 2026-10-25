"""Seven-step baseline association, compiled by the ordinary progressive owners.

The host fixes the layout. A subgroup is a parameter of the primary interaction
model, never a second stratified fit. Continuous exposures carry the same term
intent as the landmark spline template; the signed runtime owns their basis.
"""

from __future__ import annotations

from typing import Callable

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
        "visualization": "Present the primary association and any stated stratum estimates in one figure.",
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
        steps=outline_steps, rationale="The host fixes the seven steps; the spec supplies only the bound roster and reader labels.",
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
            product_inputs=[_ref("adjusted_association", "table:adjusted_association_estimates")],
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
