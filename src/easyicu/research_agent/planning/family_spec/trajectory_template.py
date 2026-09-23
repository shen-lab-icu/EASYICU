"""Host template for the sealed fixed-window trajectory suite family.

A longitudinal trajectory-clustering question whose runtime is already sealed
by a ``TrajectoryScientificRuntimeAuthority`` (fixed-grid representation,
the signed candidate grid with BIC selection, the stability design and the
selection-diagnostics figure).  The Planner decides nothing scientific here:
it labels the sealed coordinate concepts and the outcome and writes the
comparator application sentences.  The template projects steps that name the
two signed owners with their exact products; the host's ``bind_plan`` then
compiles the four signed steps from the authority, so every analysis
coordinate is the authority's.

Step layout (before the host binds the authority):

1. ``cohort_accounting``          denominators and cohort flow
2. ``coordinate_audit``           coordinate availability and missingness
3. ``trajectory_representation``  the sealed representation owner
4. ``candidate_selection``        the sealed candidate-grid owner (primary)
5. ``cluster_stability``          the signed stability design
6. ``selection_figure``           candidate-grid and availability diagnostics
7. ``report``                     zero-patient-row plan report
"""

from __future__ import annotations

from typing import Callable

from ...canonical_json import canonical_sha256
from ...trajectory.plan_contract import (
    STABILITY_CHARACTERIZATION_EXECUTOR_OUTPUTS,
    TRAJECTORY_STABILITY_CHARACTERIZATION_METHOD_HEAD,
)
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
    FIXED_WINDOW_TRAJECTORY_FAMILY_ID,
    FamilyPlanSpec,
    FamilySpecError,
    FamilySpecRequest,
)
from .landmark_categorical_template import (
    FamilySkeletonDraft,
    _cohort_intent,
    _label,
    _method_card_elements,
    _method_card_ids,
)

#: Products the signed candidate owner registers (``trajectory/plan_contract``).
CANDIDATE_OUTPUTS = (
    "artifact:candidate_cluster_assignments",
    "manifest:cluster_selection",
    "manifest:candidate_cluster_solution_schema",
    "table:trajectory_candidate_selection",
)
#: Representation-side audit products the signed owner registers beside its
#: required outputs; the cohort-flow table stays with cohort accounting here
#: because the host compiles both into the signed representation step.
REPRESENTATION_AUDIT_OUTPUTS = (
    "table:feature_availability",
    "manifest:trajectory_window_manifest",
)
SELECTION_FIGURE = "figure:trajectory_selection_diagnostics"
_AUDIT_OUTPUTS = (
    ("table:measurement_missingness", "measurement_missingness"),
    ("table:measurement_process_audit", "measurement_process"),
)
_PRIMARY_DESIGN_ELEMENTS = ("estimand", "missing_data", "reporting", "robustness", "time_zero")


def _bindings(
    outline_step: ProgressiveOutlineStep,
    *,
    comparator_applications: dict[str, str],
) -> list[ProgressiveLiteratureBinding]:
    desired = set(_PRIMARY_DESIGN_ELEMENTS)
    bindings: list[ProgressiveLiteratureBinding] = []
    for key in outline_step.literature_citation_keys:
        if key in comparator_applications:
            bindings.append(
                ProgressiveLiteratureBinding(
                    citation_key=key,
                    design_elements=["population", "time_zero", "estimand"],
                    application=comparator_applications[key],
                    divergence=None,
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
                    f"Apply the host-curated method card(s) {_method_card_ids(key, desired)} to the "
                    "sealed trajectory suite; the family template binds only the curated design "
                    "elements and retains the analysis-only claim ceiling."
                ),
                divergence=None,
            )
        )
    return bindings


def _design_selection(
    request: FamilySpecRequest,
    spec: FamilyPlanSpec,
    *,
    method_keys: list[str],
) -> ResearchDesignSelection:
    sealed = request.sealed_trajectory
    assert sealed is not None
    outcome = _label(spec, request.outcome)
    concept_text = ", ".join(_label(spec, name) for name in sealed.coordinate_concepts)
    window = f"{sealed.window_hours[0]}–{sealed.window_hours[1]} h after ICU admission"
    grid = f"{sealed.grid_width_hours} h"
    candidates = ", ".join(str(value) for value in sealed.candidate_cluster_counts)
    unit_text = (
        "each analysis row is one ICU stay; patient-level dependence is declared"
        if request.cluster_unit == "patient"
        else "each analysis row is one ICU stay and rows are not assumed to be distinct patients"
    )
    comparator_keys = [
        key for key in request.comparison_literature_keys if key in request.allowed_literature_citation_keys
    ]
    required_variables = [request.identity_column, request.outcome]
    selected = ResearchDesignCandidate(
        design_id="fixed_window_trajectory_suite",
        analysis_type="trajectory_clustering",
        estimand=(
            f"Candidate organ-dysfunction trajectory classes over {window} on a fixed {grid} grid "
            f"of {concept_text}, selected by BIC from the sealed candidate grid ({candidates}) "
            "and frozen only when the signed stability design holds; classes are candidates, not "
            "validated phenotypes, and imply no causal contrast."
        ),
        time_zero="ICU admission; every trajectory is aligned to the same fixed grid.",
        observation_window=f"{window} in {grid} windows; rows outside the window are not observed.",
        primary_method=(
            "Sealed fixed-window trajectory suite: fixed-grid representation with an explicit "
            "missingness policy, observed-data diagonal Gaussian-mixture candidates, minimum-BIC "
            "selection with fail-closed boundary rules, and resampling stability."
        ),
        required_variables=required_variables,
        assumptions=[
            "The materialized panel carries the sealed coordinate concepts on the fixed grid.",
            f"{unit_text[0].upper()}{unit_text[1:]}.",
        ],
        literature_citation_keys=[*method_keys, *comparator_keys][:8],
        literature_design_decisions=[],
        novelty_positioning=(
            "No novelty is claimed before completion; candidate classes are positioned against each "
            "screened comparator on population, window, features and selection rule."
        ),
        figure_role=(
            "Candidate-grid selection criterion and coordinate availability as the diagnostic "
            "figure; no class is displayed as a validated phenotype."
        ),
        supports=(
            "A prespecified, reproducible candidate trajectory partition with its selection and "
            f"stability evidence and a descriptive link to {outcome}."
        ),
        cannot_prove=(
            "No causal effect of a class, no external reproducibility, and no clinical phenotype "
            "beyond a stable candidate partition."
        ),
        reviewable_plan=[
            f"Population and unit: cohort {request.cohort_name}; {unit_text}.",
            f"Representation: {concept_text} aggregated per {grid} window over {window}.",
            f"Candidates: diagonal Gaussian mixtures with {candidates} classes; minimum BIC selects.",
            "Missing data: the sealed availability rule admits rows with enough observed windows; "
            "availability is audited per coordinate and window.",
            "Robustness: the signed resampling stability design must hold before any class is "
            "frozen; a boundary or unstable solution is a formal no-solution result.",
            f"Outcome: {outcome} is described by class only after the partition is frozen.",
        ],
        disposition="selected",
        decision_reason=(
            "The question asks for aligned longitudinal trajectory classes with explicit handling "
            "of unequal follow-up and missingness; the sealed fixed-window suite fixes those rules "
            "before any data are read."
        ),
    )
    rejected = ResearchDesignCandidate(
        design_id="cross_sectional_phenotyping",
        analysis_type="trajectory_clustering",
        estimand="Clusters of one aggregated feature vector per row.",
        time_zero="ICU admission.",
        observation_window=f"A single aggregate over {window}.",
        primary_method="Cross-sectional clustering of per-row aggregates.",
        required_variables=required_variables,
        assumptions=["One aggregate per row represents the trajectory."],
        literature_citation_keys=[
            key for key in ("strobe_2007", "record_2015") if key in request.allowed_literature_citation_keys
        ],
        literature_design_decisions=[],
        novelty_positioning="Recorded as the rejected alternative for audit; no novelty is claimed.",
        figure_role="An embedding of aggregate features.",
        supports="A cross-sectional partition when time structure is not the question.",
        cannot_prove="It discards the trajectory shape the question asks about.",
        reviewable_plan=None,
        disposition="rejected",
        decision_reason=(
            "Rejected because the question asks for trajectory classes and the sealed panel carries "
            "the fixed-grid longitudinal structure."
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


def build_fixed_window_trajectory_skeleton(
    request: FamilySpecRequest,
    spec: FamilyPlanSpec,
    *,
    bind_outline: Callable[[ProgressivePlanOutline], ProgressivePlanOutline] | None = None,
) -> FamilySkeletonDraft:
    """Project outline, foundation, and step materializations from the sealed suite."""

    if request.family_id != FIXED_WINDOW_TRAJECTORY_FAMILY_ID or request.sealed_trajectory is None:
        raise FamilySpecError(
            "family_spec_template_mismatch",
            "the trajectory suite template received a request for another family",
            path="family_id",
        )
    if spec.request_sha256 != request.request_sha256:
        raise FamilySpecError(
            "family_spec_request_digest_mismatch",
            "the spec does not bind this request",
            path="request_sha256",
        )
    sealed = request.sealed_trajectory
    identity = request.identity_column
    outcome = request.outcome
    audit_columns = list(request.measurement_audit_columns)
    method_keys = [key for key in request.allowed_literature_citation_keys if _method_card_elements(key)]
    primary_keys = list(
        dict.fromkeys(
            [
                *(k for k in method_keys if _method_card_elements(k) & set(_PRIMARY_DESIGN_ELEMENTS)),
                *request.direct_comparator_literature_keys,
            ]
        )
    )[:12]
    design = _design_selection(request, spec, method_keys=[k for k in method_keys if k in set(primary_keys)][:6])
    window = f"{sealed.window_hours[0]}–{sealed.window_hours[1]} h"
    objectives = {
        "cohort_accounting": (
            "Account for every input analysis row and record the denominator before the sealed "
            "availability rule is applied."
        ),
        "coordinate_audit": (
            f"Audit availability and missingness of the sealed coordinate concepts over {window} "
            "so the representation's missingness policy is applied to known gaps."
        ),
        "trajectory_representation": (
            f"Build the sealed fixed-grid representation of {len(sealed.coordinate_concepts)} "
            f"coordinate concepts over {window} with its explicit missingness policy."
        ),
        "candidate_selection": (
            "Fit every signed candidate mixture and select by minimum BIC with the sealed "
            "boundary rules; candidates are not validated phenotypes."
        ),
        "cluster_stability": (
            "Execute the signed resampling stability design and characterize the frozen "
            "partition; an unstable or boundary solution is a formal no-solution result."
        ),
        "selection_figure": (
            "Render the candidate-grid criterion and coordinate availability diagnostics "
            "without presenting candidate labels as validated phenotypes."
        ),
        "report": (
            "Produce the zero-patient-row plan report for human review: sources, denominators, "
            "the representation rule, selection and stability evidence, limitations, and the "
            "analysis-only boundary."
        ),
    }
    outline_steps = [
        _outline_step(
            step_id="cohort_accounting", role="auxiliary", module_id="cohort_definition",
            objective=objectives["cohort_accounting"], depends_on=[],
            variable_names=[identity, outcome], citations=[],
        ),
        _outline_step(
            step_id="coordinate_audit", role="auxiliary", module_id="measurement_audit",
            objective=objectives["coordinate_audit"], depends_on=["cohort_accounting"],
            variable_names=[identity, *audit_columns], citations=[],
        ),
        _outline_step(
            step_id="trajectory_representation", role="auxiliary", module_id="custom_analysis",
            objective=objectives["trajectory_representation"], depends_on=["cohort_accounting"],
            variable_names=[identity], citations=[],
        ),
        _outline_step(
            step_id="candidate_selection", role="primary", module_id="custom_analysis",
            objective=objectives["candidate_selection"], depends_on=["trajectory_representation"],
            variable_names=[identity], citations=primary_keys,
        ),
        _outline_step(
            step_id="cluster_stability", role="auxiliary", module_id="custom_analysis",
            objective=objectives["cluster_stability"],
            depends_on=["trajectory_representation", "candidate_selection"],
            variable_names=[identity, outcome], citations=[],
        ),
        _outline_step(
            step_id="selection_figure", role="auxiliary", module_id="visualization",
            objective=objectives["selection_figure"],
            depends_on=["trajectory_representation", "candidate_selection"],
            variable_names=[identity], citations=[],
        ),
    ]
    outline_steps.append(
        _outline_step(
            step_id="report", role="auxiliary", module_id="report",
            objective=objectives["report"], depends_on=[step.step_id for step in outline_steps],
            variable_names=[identity, outcome], citations=[],
        )
    )
    outline = ProgressivePlanOutline(
        analysis_type="trajectory_clustering",
        cohort_objective=(
            f"Derive candidate trajectory classes on cohort {request.cohort_name} with the sealed "
            "fixed-window suite, keeping availability, missingness, and repeated stays visible."
        ),
        design_selection=design,
        steps=outline_steps,
        rationale=(
            f"Family template {request.family_id}: every executable coordinate is the sealed "
            "trajectory authority's; the Planner supplied reader labels and comparator "
            "applications only. All results stay at plan level under the analysis-only claim ceiling."
        ),
    )
    if bind_outline is not None:
        outline = bind_outline(outline)
    bound = {step.step_id: step for step in outline.steps}
    representation_outputs = [*sealed.representation_outputs, *REPRESENTATION_AUDIT_OUTPUTS]
    steps = [
        ProgressiveSkeletonStep(
            step_id="cohort_accounting", planned_analysis_role="auxiliary", module_id="cohort_definition",
            objective=objectives["cohort_accounting"], depends_on=[],
            raw_inputs=[identity, outcome], literature_bindings=[],
        ),
        ProgressiveSkeletonStep(
            step_id="coordinate_audit", planned_analysis_role="auxiliary", module_id="measurement_audit",
            objective=objectives["coordinate_audit"], depends_on=["cohort_accounting"],
            raw_inputs=list(dict.fromkeys([identity, *audit_columns])),
            product_inputs=[_ref("cohort_accounting", "artifact:analysis_cohort")],
            outputs=[
                ProgressiveOutputIntent(product_id=product, semantic_role=role)
                for product, role in _AUDIT_OUTPUTS
            ],
            literature_bindings=[],
        ),
        ProgressiveSkeletonStep(
            step_id="trajectory_representation", planned_analysis_role="auxiliary",
            module_id="custom_analysis", objective=objectives["trajectory_representation"],
            depends_on=["cohort_accounting"], raw_inputs=[identity],
            product_inputs=[_ref("cohort_accounting", "artifact:analysis_cohort")],
            outputs=[
                ProgressiveOutputIntent(product_id=product, semantic_role="custom")
                for product in representation_outputs
            ],
            custom_method=sealed.representation_owner,
            literature_bindings=[],
        ),
        ProgressiveSkeletonStep(
            step_id="candidate_selection", planned_analysis_role="primary", module_id="custom_analysis",
            objective=objectives["candidate_selection"], depends_on=["trajectory_representation"],
            raw_inputs=[],
            product_inputs=[
                _ref("trajectory_representation", "artifact:trajectory_representation"),
                _ref("trajectory_representation", "manifest:trajectory_representation_schema"),
            ],
            outputs=[
                ProgressiveOutputIntent(product_id=product, semantic_role="custom")
                for product in CANDIDATE_OUTPUTS
            ],
            custom_method=sealed.candidate_owner,
            literature_bindings=_bindings(
                bound["candidate_selection"], comparator_applications=spec.applications,
            ),
        ),
        ProgressiveSkeletonStep(
            step_id="cluster_stability", planned_analysis_role="auxiliary", module_id="custom_analysis",
            objective=objectives["cluster_stability"],
            depends_on=["trajectory_representation", "candidate_selection"],
            raw_inputs=[],
            product_inputs=[
                _ref("trajectory_representation", "artifact:trajectory_representation"),
                _ref("candidate_selection", "artifact:candidate_cluster_assignments"),
                _ref("candidate_selection", "manifest:cluster_selection"),
                _ref("trajectory_representation", "manifest:trajectory_representation_schema"),
                _ref("candidate_selection", "manifest:candidate_cluster_solution_schema"),
            ],
            outputs=[
                ProgressiveOutputIntent(product_id=product, semantic_role="custom")
                for product in sorted(STABILITY_CHARACTERIZATION_EXECUTOR_OUTPUTS)
            ],
            custom_method=TRAJECTORY_STABILITY_CHARACTERIZATION_METHOD_HEAD,
            literature_bindings=[],
        ),
        ProgressiveSkeletonStep(
            step_id="selection_figure", planned_analysis_role="auxiliary", module_id="visualization",
            objective=objectives["selection_figure"],
            depends_on=["trajectory_representation", "candidate_selection"],
            raw_inputs=[],
            product_inputs=[
                _ref("candidate_selection", "table:trajectory_candidate_selection"),
                _ref("trajectory_representation", "table:feature_availability"),
            ],
            outputs=[ProgressiveOutputIntent(product_id=SELECTION_FIGURE, semantic_role="figure")],
            literature_bindings=[],
        ),
        ProgressiveSkeletonStep(
            step_id="report", planned_analysis_role="auxiliary", module_id="report",
            objective=objectives["report"], depends_on=bound["report"].depends_on, raw_inputs=[],
            product_inputs=[
                _ref("cohort_accounting", "artifact:analysis_cohort"),
                _ref("cohort_accounting", "table:cohort_flow"),
                *(_ref("coordinate_audit", product) for product, _role in _AUDIT_OUTPUTS),
                _ref("trajectory_representation", "table:feature_availability"),
                _ref("candidate_selection", "table:trajectory_candidate_selection"),
                *(
                    _ref("cluster_stability", product)
                    for product in sorted(STABILITY_CHARACTERIZATION_EXECUTOR_OUTPUTS)
                    if product.startswith("table:")
                ),
                _ref("selection_figure", SELECTION_FIGURE),
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


__all__ = ["CANDIDATE_OUTPUTS", "SELECTION_FIGURE", "build_fixed_window_trajectory_skeleton"]
