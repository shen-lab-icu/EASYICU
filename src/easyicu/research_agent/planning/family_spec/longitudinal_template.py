"""Shared projection mechanics for explicit longitudinal templates.

This owner builds ordinary Progressive coordinates. It does not bypass the
compiler, choose runtime authorities, or register a question-routing rule.
"""

from __future__ import annotations

from typing import Callable

from pydantic import ValidationError

from ...canonical_json import canonical_sha256
from ..design_selection import ResearchDesignCandidate, ResearchDesignSelection
from ..progressive_contract import (
    ProgressiveDisplayLabel, ProgressiveFoundationMaterialization, ProgressiveLiteratureBinding,
    ProgressiveOutlineStep, ProgressiveOutputIntent, ProgressivePlanFoundation,
    ProgressivePlanOutline, ProgressiveProductRef, ProgressiveSkeletonStep,
    ProgressiveStepMaterialization,
)
from ..progressive_resume import validate_progressive_materialization_coordinate
from .contract import FamilyPlanSpec, FamilySpecError, FamilySpecRequest, validate_family_plan_spec
from .landmark_categorical_template import FamilySkeletonDraft, _cohort_intent, _MEASUREMENT_AUDIT_OUTPUTS

OutlineBinder = Callable[[ProgressivePlanOutline], ProgressivePlanOutline]


def checked_request(request: FamilySpecRequest, spec: FamilyPlanSpec, family: str) -> FamilySpecRequest:
    if request.family_id != family:
        raise FamilySpecError("family_spec_template_mismatch", "This builder requires its own family.", path="family_id")
    try:
        # model_copy is not validation. Recheck nested authorities as JSON so
        # strict tuple fields retain their ordinary transport representation.
        request = FamilySpecRequest.model_validate_json(request.model_dump_json())
    except ValidationError as exc:
        cause = exc.errors()[0].get("ctx", {}).get("error")
        if isinstance(cause, FamilySpecError):
            raise cause from exc
        raise FamilySpecError("family_spec_longitudinal_coordinates", "Invalid host coordinates.", path="family_id") from exc
    validate_family_plan_spec(spec, request)
    return request


def ref(producer: str, product: str) -> ProgressiveProductRef:
    return ProgressiveProductRef(producer_step_id=producer, product_id=product)


def outputs(products: tuple[str, ...] | list[str], role: str = "custom") -> list[ProgressiveOutputIntent]:
    return [ProgressiveOutputIntent(product_id=product, semantic_role=role) for product in products]


def preparation(request: FamilySpecRequest, columns: list[str]) -> list[ProgressiveSkeletonStep]:
    return [
        ProgressiveSkeletonStep(
            step_id="cohort_accounting", module_id="cohort_definition", planned_analysis_role="auxiliary",
            objective="Account for every bound source stay before the longitudinal owner's eligibility and risk-set rules.",
            raw_inputs=columns,
        ),
        ProgressiveSkeletonStep(
            step_id="measurement_audit", module_id="measurement_audit", planned_analysis_role="auxiliary",
            objective="Audit source measurements, completeness, missingness and denominators before constructing longitudinal risk sets.",
            depends_on=["cohort_accounting"],
            raw_inputs=list(dict.fromkeys([*columns[1:], *request.measurement_audit_columns])),
            product_inputs=[ref("cohort_accounting", "artifact:analysis_cohort")],
            outputs=[ProgressiveOutputIntent(product_id=product, semantic_role=role) for product, role in _MEASUREMENT_AUDIT_OUTPUTS],
        ),
    ]


def append_report(steps: list[ProgressiveSkeletonStep], *, limitations: str) -> None:
    products = [ref("cohort_accounting", "table:cohort_flow")]
    for step in steps:
        products.extend(ref(step.step_id, item.product_id) for item in step.outputs)
        if step.module_id == "table_one":
            products.append(ref(step.step_id, "table:table_one"))
    steps.append(ProgressiveSkeletonStep(
        step_id="report", module_id="report", planned_analysis_role="auxiliary",
        objective="Report cohort accounting, methods, owned results and limitations for human review. " + limitations,
        depends_on=[step.step_id for step in steps], product_inputs=products,
        outputs=outputs(["report:report"], "report"),
    ))


def draft(
    request: FamilySpecRequest, spec: FamilyPlanSpec, steps: list[ProgressiveSkeletonStep], *,
    estimand: str, time_zero: str, observation_window: str, primary_method: str,
    assumptions: list[str], cannot_prove: str, reviewable_plan: list[str],
    bind_outline: OutlineBinder | None,
) -> FamilySkeletonDraft:
    selected = ResearchDesignCandidate(
        design_id=request.family_id, analysis_type=request.analysis_type, estimand=estimand,
        time_zero=time_zero, observation_window=observation_window, primary_method=primary_method,
        required_variables=request.variable_roster, assumptions=assumptions,
        literature_citation_keys=list(spec.applications),
        literature_design_decisions=spec.literature_design_decisions,
        novelty_positioning="The explicit host design is retained; no novelty claim is inferred.",
        figure_role="Display only the products that the named analysis owners produce.",
        supports=estimand, cannot_prove=cannot_prove, reviewable_plan=reviewable_plan,
        disposition="selected", decision_reason="The caller supplied this family's complete host coordinates.",
    )
    rejected = selected.model_copy(update={
        "design_id": "single_window_description", "disposition": "rejected", "reviewable_plan": None,
        "primary_method": "Single-window descriptive summary.",
        "decision_reason": "Collapsing time changes the prespecified risk sets and does not answer this design.",
    })
    outline = ProgressivePlanOutline(
        analysis_type=request.analysis_type, cohort_objective=estimand,
        design_selection=ResearchDesignSelection(candidates=[selected, rejected]),
        steps=[ProgressiveOutlineStep(
            step_id=step.step_id, module_id=step.module_id, objective=step.objective,
            planned_analysis_role=step.planned_analysis_role, depends_on=step.depends_on,
            variable_names=step.raw_inputs or [request.outcome], scientific_action_id=step.scientific_action_id,
            literature_citation_keys=list(spec.applications) if step.planned_analysis_role == "primary" else [],
        ) for step in steps],
        rationale="Explicit host family template; existing routing and scientific review remain authoritative. Analysis only.",
    )
    if bind_outline is not None:
        outline = bind_outline(outline)
    if [step.step_id for step in outline.steps] != [step.step_id for step in steps]:
        raise FamilySpecError("family_spec_template_step_mismatch", "The outline binder changed the host step roster.", path="steps")
    materializations = []
    for index, (outline_step, step) in enumerate(zip(outline.steps, steps, strict=True)):
        bindings = [ProgressiveLiteratureBinding(
            citation_key=key, design_elements=["estimand", "time_zero", "reporting"],
            application=spec.applications[key], divergence=None,
        ) for key in outline_step.literature_citation_keys if key in spec.applications]
        item = ProgressiveStepMaterialization(
            outline_step_sha256=canonical_sha256(outline_step.model_dump(mode="json")),
            foundation=None, step=step.model_copy(update={"literature_bindings": bindings}),
        )
        validate_progressive_materialization_coordinate(
            item, outline_step=outline_step, outline_step_sha256=item.outline_step_sha256, step_index=index,
        )
        materializations.append(item)
    return FamilySkeletonDraft(
        outline=outline,
        foundation=ProgressiveFoundationMaterialization(
            outline_sha256=canonical_sha256(outline.model_dump(mode="json")),
            foundation=ProgressivePlanFoundation(
                cohort=_cohort_intent(request, spec.population),
                display_labels=[ProgressiveDisplayLabel(key=key, value=value) for key, value in spec.labels.items()],
                robustness_intents=[], know_how_decisions=[],
            ),
        ), materializations=tuple(materializations),
    )
