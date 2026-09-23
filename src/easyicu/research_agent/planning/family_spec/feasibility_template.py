"""Host template for the sealed fail-closed source-feasibility family.

A causal question whose reviewed protocol found the requested treatment
contrast non-identifiable from the current source capture (for example, the
source can prove recorded administration but not verified non-use, so no
control arm exists).  The signing ``SourceFeasibilityRuntimeAuthority`` owns
that decision; the Planner decides nothing scientific here and supplies only
comparator application sentences.  The template projects one auxiliary step
that names the sealed owner with its exact products; the host's ``bind_plan``
then replaces the draft with the single signed step, so the run's only result
is the fail-closed feasibility decision.

Step layout (before the host binds the authority):

1. ``cohort_accounting``   denominators of the audited source rows
2. ``source_feasibility``  the sealed fail-closed decision owner
3. ``report``              zero-patient-row plan report
"""

from __future__ import annotations

from typing import Callable

from ...canonical_json import canonical_sha256
from ..design_selection import ResearchDesignCandidate, ResearchDesignSelection
from ..progressive_contract import (
    ProgressiveDisplayLabel,
    ProgressiveFoundationMaterialization,
    ProgressiveLiteratureBinding,
    ProgressiveOutlineStep,
    ProgressiveOutputIntent,
    ProgressivePlanFoundation,
    ProgressivePlanOutline,
    ProgressiveProductRef,
    ProgressiveSkeletonStep,
    ProgressiveStepMaterialization,
)
from .contract import (
    SOURCE_FEASIBILITY_FAMILY_ID,
    FamilyPlanSpec,
    FamilySpecError,
    FamilySpecRequest,
)
from .landmark_categorical_template import (
    FamilySkeletonDraft,
    _cohort_intent,
    _method_card_elements,
    _method_card_ids,
)

_DECISION_DESIGN_ELEMENTS = ("estimand", "exposure", "time_zero", "reporting")


def _bindings(
    outline_step: ProgressiveOutlineStep,
    *,
    comparator_applications: dict[str, str],
) -> list[ProgressiveLiteratureBinding]:
    desired = set(_DECISION_DESIGN_ELEMENTS)
    bindings: list[ProgressiveLiteratureBinding] = []
    for key in outline_step.literature_citation_keys:
        if key in comparator_applications:
            bindings.append(
                ProgressiveLiteratureBinding(
                    citation_key=key,
                    design_elements=["population", "exposure", "estimand"],
                    application=comparator_applications[key],
                    divergence=(
                        "The current source cannot open the comparator's control arm; "
                        "no effect estimate is produced."
                    ),
                )
            )
            continue
        elements = sorted(desired & _method_card_elements(key))
        if not elements:
            continue
        bindings.append(
            ProgressiveLiteratureBinding(
                citation_key=key,
                design_elements=elements,
                application=(
                    f"Apply the host-curated method card(s) {_method_card_ids(key, desired)} "
                    "to state the requested target-trial contrast and why the sealed source "
                    "audit fails it closed; no estimator is fitted."
                ),
                divergence=None,
            )
        )
    return bindings


def _design_variables(request: FamilySpecRequest) -> list[str]:
    """The identity column plus the audited source columns the reviewer labels."""

    audited = [
        name for name in request.measurement_audit_columns if name != request.identity_column
    ]
    if not audited:
        raise FamilySpecError(
            "family_spec_feasibility_roster_empty",
            "the source feasibility decision needs at least one audited source column "
            "beside the identity column",
            path="measurement_audit_columns",
        )
    return [request.identity_column, *audited]


def _design_selection(
    request: FamilySpecRequest,
    *,
    method_keys: list[str],
) -> ResearchDesignSelection:
    sealed = request.sealed_feasibility
    assert sealed is not None
    window = f"ICU hours {sealed.audited_window_hours[0]}–{sealed.audited_window_hours[1]}"
    design_variables = _design_variables(request)
    comparator_keys = [
        key for key in request.comparison_literature_keys if key in request.allowed_literature_citation_keys
    ]
    selected = ResearchDesignCandidate(
        design_id="source_feasibility_fail_closed",
        analysis_type="causal_inference",
        estimand=(
            "No treatment effect is estimated: the reviewed protocol found the requested "
            f"contrast not identifiable from {sealed.source}, so the formal result is the "
            f"signed decision {sealed.decision} ({sealed.reason_code})."
        ),
        time_zero=f"ICU admission; the source capture is audited over {window}.",
        observation_window=window,
        primary_method=(
            "Sealed source-feasibility audit: verify what the source capture can and cannot "
            "prove (recorded administration versus verified non-use), emit the fail-closed "
            "decision table and its runtime receipt; no control arm, weighting, matching or "
            "effect estimate."
        ),
        required_variables=design_variables,
        assumptions=[
            "The source capture contract describes every record the source can provide "
            "for the audited window.",
        ],
        literature_citation_keys=[*method_keys, *comparator_keys][:8],
        literature_design_decisions=[],
        novelty_positioning=(
            "No novelty is claimed; the result states a source-specific identifiability "
            "boundary against the screened comparators."
        ),
        figure_role="No figure: the signed feasibility table is the result.",
        supports=(
            "A formal statement that the requested causal contrast cannot be identified "
            "from the current source, with the capture boundary that would unblock it."
        ),
        cannot_prove=(
            "Any treatment effect, absence of effect, balance, or positivity; absence of a "
            "record is not verified non-use."
        ),
        reviewable_plan=[
            f"Population and unit: cohort {request.cohort_name}; every audited source row "
            "is one ICU stay.",
            f"Exposure and timing: recorded administration in {sealed.source} over {window}; "
            "verified non-use is unavailable, so no comparator arm is formed.",
            "Outcome and follow-up: no outcome is analysed; the reviewed protocol declares "
            "the treatment contrast non-identifiable.",
            f"Adjustment and model: none; the sealed audit emits the decision {sealed.decision} "
            f"with reason code {sealed.reason_code} and forbids "
            + ", ".join(sealed.forbidden_plan_tokens) + ".",
            "Missing data: absence of an administration record is not verified non-use and "
            "is never imputed as a control arm.",
            "Sensitivity and feasibility: not applicable; the future unblock design is not "
            "authorized in this run and the products are the signed table and receipt.",
        ],
        disposition="selected",
        decision_reason=(
            "The reviewed protocol declares the contrast non-identifiable; the sealed "
            "fail-closed decision is the only current-run result the host may execute."
        ),
    )
    rejected = ResearchDesignCandidate(
        design_id="weighted_effect_estimation",
        analysis_type="causal_inference",
        estimand="An inverse-probability-weighted treatment effect on the outcome.",
        time_zero="ICU admission.",
        observation_window=window,
        primary_method="Propensity weighting with balance and positivity diagnostics.",
        required_variables=design_variables,
        assumptions=["A verified untreated comparator arm exists in the source."],
        literature_citation_keys=[
            key for key in ("strobe_2007", "record_2015") if key in request.allowed_literature_citation_keys
        ],
        literature_design_decisions=[],
        novelty_positioning="Recorded as the rejected alternative for audit; no novelty is claimed.",
        figure_role="A balance plot and effect contrast.",
        supports="A causal contrast when both arms are verifiable.",
        cannot_prove=(
            "Anything under the current capture: absence of a record cannot define the "
            "comparator arm."
        ),
        reviewable_plan=None,
        disposition="rejected",
        decision_reason=(
            "Rejected because the sealed source audit finds verified non-use unavailable; "
            "the reviewed protocol forbids constructing a control arm from absent records."
        ),
    )
    return ResearchDesignSelection(candidates=[selected, rejected])


def _outline_step(
    *,
    step_id: str,
    role: str,
    module_id: str,
    objective: str,
    depends_on: list[str],
    variable_names: list[str],
    citations: list[str],
) -> ProgressiveOutlineStep:
    return ProgressiveOutlineStep(
        step_id=step_id,
        planned_analysis_role=role,
        module_id=module_id,
        objective=objective,
        depends_on=depends_on,
        variable_names=list(dict.fromkeys(variable_names)),
        literature_citation_keys=list(dict.fromkeys(citations))[:12],
        scientific_action_id=None,
    )


def _ref(producer: str, product: str) -> ProgressiveProductRef:
    return ProgressiveProductRef(producer_step_id=producer, product_id=product)


def build_source_feasibility_skeleton(
    request: FamilySpecRequest,
    spec: FamilyPlanSpec,
    *,
    bind_outline: Callable[[ProgressivePlanOutline], ProgressivePlanOutline] | None = None,
) -> FamilySkeletonDraft:
    """Project outline, foundation, and step materializations from the sealed decision."""

    if request.family_id != SOURCE_FEASIBILITY_FAMILY_ID or request.sealed_feasibility is None:
        raise FamilySpecError(
            "family_spec_template_mismatch",
            "the source feasibility template received a request for another family",
            path="family_id",
        )
    if spec.request_sha256 != request.request_sha256:
        raise FamilySpecError(
            "family_spec_request_digest_mismatch",
            "the spec does not bind this request",
            path="request_sha256",
        )
    sealed = request.sealed_feasibility
    identity = request.identity_column
    method_keys = [key for key in request.allowed_literature_citation_keys if _method_card_elements(key)]
    decision_keys = list(
        dict.fromkeys(
            [
                *(k for k in method_keys if _method_card_elements(k) & set(_DECISION_DESIGN_ELEMENTS)),
                *request.direct_comparator_literature_keys,
            ]
        )
    )[:12]
    design = _design_selection(
        request, method_keys=[k for k in method_keys if k in set(decision_keys)][:6]
    )
    design_variables = design.selected.required_variables
    table_product = next(value for value in sealed.plan_outputs if value.startswith("table:"))
    objectives = {
        "cohort_accounting": (
            "Account for every audited source row and record the denominator the "
            "feasibility decision applies to."
        ),
        "source_feasibility": sealed.plan_intent,
        "report": (
            "Produce the zero-patient-row plan report for human review: the requested "
            "contrast, the source capture boundary, the fail-closed decision and its "
            "reason code, and the analysis-only boundary."
        ),
    }
    outline_steps = [
        _outline_step(
            step_id="cohort_accounting", role="auxiliary", module_id="cohort_definition",
            objective=objectives["cohort_accounting"], depends_on=[],
            variable_names=design_variables, citations=[],
        ),
        _outline_step(
            step_id="source_feasibility", role="auxiliary", module_id="custom_analysis",
            objective=objectives["source_feasibility"], depends_on=["cohort_accounting"],
            variable_names=design_variables, citations=decision_keys,
        ),
    ]
    outline_steps.append(
        _outline_step(
            step_id="report", role="auxiliary", module_id="report",
            objective=objectives["report"], depends_on=[step.step_id for step in outline_steps],
            variable_names=[identity], citations=[],
        )
    )
    outline = ProgressivePlanOutline(
        analysis_type="causal_inference",
        cohort_objective=(
            f"State the sealed source-feasibility decision for cohort {request.cohort_name}: "
            "the requested treatment contrast is not identifiable from the current source "
            "capture, so no effect is estimated."
        ),
        design_selection=design,
        steps=outline_steps,
        rationale=(
            f"Family template {request.family_id}: the decision, its products and its "
            "forbidden actions are the sealed runtime authority's; the Planner supplied "
            "comparator applications only. All results stay at plan level under the "
            "analysis-only claim ceiling."
        ),
    )
    if bind_outline is not None:
        outline = bind_outline(outline)
    bound = {step.step_id: step for step in outline.steps}
    steps = [
        ProgressiveSkeletonStep(
            step_id="cohort_accounting", planned_analysis_role="auxiliary", module_id="cohort_definition",
            objective=objectives["cohort_accounting"], depends_on=[],
            raw_inputs=list(design_variables), literature_bindings=[],
        ),
        ProgressiveSkeletonStep(
            step_id="source_feasibility", planned_analysis_role="auxiliary", module_id="custom_analysis",
            objective=objectives["source_feasibility"], depends_on=["cohort_accounting"],
            # Exactly the sealed products: the host's bind_plan replaces this
            # draft with the single signed step and refuses any drift.
            raw_inputs=list(design_variables),
            outputs=[
                ProgressiveOutputIntent(product_id=product, semantic_role="custom")
                for product in sealed.plan_outputs
            ],
            custom_method=sealed.sealed_owner,
            literature_bindings=_bindings(
                bound["source_feasibility"], comparator_applications=spec.applications,
            ),
        ),
        ProgressiveSkeletonStep(
            step_id="report", planned_analysis_role="auxiliary", module_id="report",
            objective=objectives["report"], depends_on=bound["report"].depends_on, raw_inputs=[],
            product_inputs=[
                _ref("cohort_accounting", "artifact:analysis_cohort"),
                _ref("cohort_accounting", "table:cohort_flow"),
                _ref("source_feasibility", table_product),
            ],
            outputs=[ProgressiveOutputIntent(product_id="report:report", semantic_role="report")],
            literature_bindings=[],
        ),
    ]
    if [step.step_id for step in steps] != [step.step_id for step in outline.steps]:
        raise FamilySpecError(
            "family_spec_template_step_mismatch",
            "template outline and materialization rosters diverged",
            path="steps",
        )
    outline_sha256 = canonical_sha256(outline.model_dump(mode="json"))
    foundation = ProgressiveFoundationMaterialization(
        outline_sha256=outline_sha256,
        foundation=ProgressivePlanFoundation(
            cohort=_cohort_intent(request),
            display_labels=[
                ProgressiveDisplayLabel(key=key, value=value)
                for key, value in spec.labels.items()
                if key in set(request.variable_roster)
            ],
            robustness_intents=[],
            know_how_decisions=[],
        ),
    )
    materializations = tuple(
        ProgressiveStepMaterialization(
            outline_step_sha256=canonical_sha256(bound[step.step_id].model_dump(mode="json")),
            foundation=None,
            step=step,
        )
        for step in steps
    )
    return FamilySkeletonDraft(outline=outline, foundation=foundation, materializations=materializations)


__all__ = ["build_source_feasibility_skeleton"]
