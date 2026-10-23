"""Existing step repairs retain scientific scope and consume one source step."""

from __future__ import annotations

import pytest

from easyicu.research_agent.planning.scientific_review import PlanScientificReview
from easyicu.research_agent.contracts.figure_plan import FigurePresentationSpec
from easyicu.research_agent.schema import AnalysisPlan, AnalysisStep
from easyicu.webserver import research_plan_revision as owner
from tests.research_agent.planning.scientific_review_fixtures import _context, _plan


def _binding_repair_guard(plan: AnalysisPlan) -> owner.PlanRevisionGuard:
    review = PlanScientificReview(
        status="changes_required", approval_allowed=False, top_journal_candidate=False,
        score=50, dimension_scores={}, findings=[], context_sha256="a" * 64,
        plan_sha256=owner.canonical_sha256(plan.model_dump(mode="json")),
        literature_sha256="c" * 64, figure_strategy_sha256="d" * 64,
        generated_at="2026-09-01T00:00:00Z",
    )
    return owner.PlanRevisionGuard.from_review(plan=plan.model_dump(mode="json"), review=review)


@pytest.fixture
def binding_replay() -> tuple[AnalysisPlan, AnalysisPlan]:
    """Anonymized seven-step checkpoint structure; no observations or levels.

    The stopped revision rebuilt two data-quality panels before the article
    placement projection. Only their placement differs from the saved plan.
    """
    steps = [
        AnalysisStep(
            step_id=step_id, planned_analysis_role=role, method=method,
            intent="Produce the declared synthetic product.", inputs=inputs,
            expected_outputs=outputs,
        )
        for step_id, role, method, inputs, outputs in [
            ("select_records", "auxiliary", "cohort_definition_and_attrition",
             ["record_key", "measurement"], ["artifact:analysis_cohort", "table:cohort_flow"]),
            ("summarize_groups", "primary", "grouped_empirical_distribution_summary",
             ["record_key", "category", "measurement", "artifact:analysis_cohort"],
             ["table:cohort_summary", "table:distribution_summary"]),
            ("summarize_measurement", "secondary", "continuous_outcome_summary",
             ["artifact:analysis_cohort", "category", "measurement"],
             ["table:continuous_outcome_summary"]),
            ("audit_coverage", "auxiliary", "missing_data",
             ["record_key", "category", "measurement", "artifact:analysis_cohort"],
             ["table:measurement_audit", "table:measurement_process_audit"]),
            ("result_display", "auxiliary", "visualization",
             ["table:distribution_summary"], ["figure:primary_result"]),
            ("cohort_display", "auxiliary", "visualization",
             ["table:cohort_flow"], ["figure:cohort_flow"]),
            ("quality_display", "auxiliary", "visualization",
             ["table:measurement_audit", "table:measurement_process_audit"], ["figure:data_quality"]),
        ]
    ]
    figure = AnalysisStep(**{
        **steps[-1].model_dump(mode="json"),
        "figure_panels": [
            {"panel_id": panel_id, "article_role": "data_quality", "chart_type": chart,
             "figure_output": "figure:data_quality", "placement": "supplementary",
             "source_products": [source]}
            for panel_id, chart, source in [
                ("source_availability", "availability_panel", "table:measurement_audit"),
                ("measurement_process_coverage", "coverage_heatmap", "table:measurement_process_audit"),
            ]
        ],
        "input_consumption_contracts": [
            {"input_key": source, "mode": "all_rows"} for source in steps[-1].inputs
        ],
    })
    previous = AnalysisPlan(
        research_question="Describe a measurement across category levels.",
        analysis_type="descriptive_epidemiology", steps=[*steps[:-1], figure],
    )
    rebuilt = figure.model_copy(update={"figure_panels": [
        panel.model_copy(update={"placement": "main"}) for panel in figure.figure_panels
    ]})
    return previous, previous.model_copy(update={"steps": [*steps[:-1], rebuilt]})


def test_checkpoint_structure_replay_admits_rebuilt_figure_placement(binding_replay):
    previous, revised = binding_replay
    before = previous.model_dump(mode="json")
    guard = _binding_repair_guard(previous)
    guard.validate_plan(plan=revised, context=_context())
    assert guard.previous_plan == before == previous.model_dump(mode="json")


@pytest.mark.parametrize("change", ["repoint", "repoint_revised_product", "reduce", "step_reference", "known_variable", "presentation", "panel_identity", "figure_output"])
def test_same_id_figure_can_repair_inputs_and_display_bindings(binding_replay, change):
    previous, _ = binding_replay
    figure = previous.steps[-1]
    if change in {"presentation", "panel_identity", "figure_output"}:
        panels = [panel.model_copy(update={
            "presentation": {"presentation": FigurePresentationSpec(layout="column")},
            "panel_identity": {"panel_id": f"renamed_{index}"},
            "figure_output": {"figure_output": "figure:repaired_display"},
        }[change]) for index, panel in enumerate(figure.figure_panels)]
        update = {"figure_panels": panels}
        if change == "figure_output":
            update["expected_outputs"] = ["figure:repaired_display"]
    else:
        inputs = {
            "repoint": ["table:cohort_summary"],
            "repoint_revised_product": ["table:repaired_audit"],
            "reduce": figure.inputs[:1],
            "step_reference": ["audit_coverage"],
            "known_variable": ["category"],
        }[change]
        panels = [figure.figure_panels[0].model_copy(update={"source_products": inputs})] if change in {"repoint", "repoint_revised_product", "reduce"} else []
        update = {"inputs": inputs, "figure_panels": panels}
    figure = figure.model_copy(update=update)
    steps = [*previous.steps[:-1], figure]
    if change == "repoint_revised_product":
        steps[3] = steps[3].model_copy(update={"expected_outputs": [*steps[3].expected_outputs, "table:repaired_audit"]})
    revised = previous.model_copy(update={"steps": steps})
    _binding_repair_guard(previous).validate_plan(plan=revised, context=_context())


@pytest.mark.parametrize("change", ["variable", "missing_product", "missing_panel_product", "missing_figure_output", "missing_step", "method", "module", "role", "sensitivity", "chart", "new_id"])
def test_same_id_cannot_hide_new_coordinates_or_dangling_references(binding_replay, change):
    previous, revised = binding_replay
    figure = revised.steps[-1]
    update = {
        "variable": {"inputs": [*figure.inputs, "unrequested_measurement"]},
        "missing_product": {"inputs": ["table:absent_product"]},
        "missing_panel_product": {"figure_panels": [figure.figure_panels[0].model_copy(update={"source_products": ["table:absent_product"]})]},
        "missing_figure_output": {"figure_panels": [figure.figure_panels[0].model_copy(update={"figure_output": "figure:absent_output"})]},
        "missing_step": {"inputs": ["absent_step"]},
        "method": {"method": "new_analysis"},
        "module": {"scientific_action_id": "descriptive.descriptive_summary"},
        "role": {"planned_analysis_role": "secondary"},
        "sensitivity": {"sensitivity_spec_ids": ["unrequested_sensitivity"]},
        "chart": {"figure_panels": [figure.figure_panels[0].model_copy(update={"chart_type": "forest"})]},
        "new_id": {"step_id": "new_display"},
    }[change]
    revised = revised.model_copy(update={"steps": [*revised.steps[:-1], figure.model_copy(update=update)]})
    with pytest.raises(owner.ResearchPipelineRunError) as refused:
        _binding_repair_guard(previous).validate_plan(plan=revised, context=_context())
    assert refused.value.code == "revision_added_unrequested_analysis"
    assert refused.value.details["step_id"] == revised.steps[-1].step_id


def test_a_repaired_step_cannot_be_consumed_twice(binding_replay):
    previous, revised = binding_replay
    duplicate = previous.steps[-1].model_copy(update={"step_id": "another_display"})
    revised = revised.model_copy(update={"steps": [*revised.steps, duplicate]})
    with pytest.raises(owner.ResearchPipelineRunError):
        _binding_repair_guard(previous).validate_plan(plan=revised, context=_context())


def test_matching_prefers_identity_before_equal_coordinates(binding_replay):
    previous, revised = binding_replay
    sibling = previous.steps[-1].model_copy(update={"step_id": "sibling_display"})
    previous = previous.model_copy(update={"steps": [*previous.steps, sibling]})
    revised = revised.model_copy(update={"steps": [*revised.steps[:-1], sibling, revised.steps[-1]]})
    _binding_repair_guard(previous).validate_plan(plan=revised, context=_context())


def test_an_unchanged_coordinate_can_still_be_renamed(binding_replay):
    previous, _ = binding_replay
    renamed = previous.steps[-1].model_copy(update={"step_id": "renamed_display"})
    revised = previous.model_copy(update={"steps": [*previous.steps[:-1], renamed]})
    _binding_repair_guard(previous).validate_plan(plan=revised, context=_context())


@pytest.mark.parametrize("used_by", ["other_step", "model_input", None])
def test_a_primary_model_cannot_gain_an_unrequested_covariate(used_by):
    previous = _plan()
    if used_by == "other_step":
        audit = AnalysisStep(step_id="other_product", method="missing_data", intent="Audit an existing source.",
                             inputs=["unrequested_predictor"], expected_outputs=["table:other_product"])
        previous = previous.model_copy(update={"steps": [*previous.steps, audit]})
    elif used_by == "model_input":
        primary = previous.steps[0].model_copy(update={"inputs": [*previous.steps[0].inputs, "unrequested_predictor"]})
        previous = previous.model_copy(update={"steps": [primary, *previous.steps[1:]]})
    primary = previous.steps[0]
    requirement = primary.model_requirements[0]
    extra = requirement.model_terms[-1].model_copy(update={"name": "unrequested_predictor"})
    changed = primary.model_copy(update={
        "inputs": [*primary.inputs, extra.name],
        "model_requirements": [requirement.model_copy(update={
            "covariates": [*requirement.covariates, extra.name],
            "model_terms": [*requirement.model_terms, extra],
        })],
    })
    revised = previous.model_copy(update={"steps": [changed, *previous.steps[1:]]})
    with pytest.raises(owner.ResearchPipelineRunError):
        _binding_repair_guard(previous).validate_plan(plan=revised, context=_context())


def test_unchanged_typed_coordinates_cannot_hide_a_new_input_variable():
    previous = _plan()
    changed = previous.steps[0].model_copy(update={"inputs": [*previous.steps[0].inputs, "unrequested_predictor"]})
    revised = previous.model_copy(update={"steps": [changed, *previous.steps[1:]]})
    with pytest.raises(owner.ResearchPipelineRunError):
        _binding_repair_guard(previous).validate_plan(plan=revised, context=_context())


def test_an_existing_primary_method_cannot_change_under_the_same_action():
    previous = _plan()
    primary = previous.steps[0].model_copy(update={"scientific_action_id": "association.adjusted_association"})
    previous = previous.model_copy(update={"steps": [primary, *previous.steps[1:]]})
    changed = primary.model_copy(update={"method": "unrequested_model_method"})
    revised = previous.model_copy(update={"steps": [changed, *previous.steps[1:]]})
    with pytest.raises(owner.ResearchPipelineRunError):
        _binding_repair_guard(previous).validate_plan(plan=revised, context=_context())


@pytest.mark.parametrize("kind", ["subgroup", "functional_form", "robustness", "evalue"])
def test_an_existing_step_cannot_gain_a_scientific_spec(kind):
    from easyicu.research_agent.contracts.post_analysis import EValueConversionSpec, SubgroupAnalysisSpec
    from easyicu.research_agent.contracts.functional_form import FunctionalFormSpec
    from easyicu.research_agent.schema import RobustnessReplaySpec

    method, role = {
        "subgroup": ("subgroup_analysis", "secondary"),
        "functional_form": ("restricted_cubic_spline_sensitivity", "sensitivity"),
        "robustness": ("robustness_analysis", "sensitivity"),
        "evalue": ("e_value", "auxiliary"),
    }[kind]
    step = AnalysisStep(
        step_id="existing_analysis", planned_analysis_role=role, method=method,
        intent="Perform the existing synthetic analysis.", inputs=["exposure", "death", "age"],
        expected_outputs=["table:existing"],
        sensitivity_spec_ids=["existing_authority"] if kind == "functional_form" else [],
    )
    previous = _plan().model_copy(update={"steps": [*_plan().steps, step]})
    revised = previous
    if kind == "subgroup":
        revised = previous.model_copy(update={"subgroup_analysis_spec": SubgroupAnalysisSpec(
            primary_model_requirement_id="primary", predictor="exposure", outcome="death",
            subgroup_columns=["age"], multiplicity_family_id="subgroups",
        )})
    elif kind == "evalue":
        revised = previous.model_copy(update={"evalue_conversion_spec": EValueConversionSpec(
            baseline_risk_column="death", population_column="analysis_set", baseline_population="source_aware",
        )})
    else:
        update = {"functional_form_spec": FunctionalFormSpec(target_column="age", knot_quantiles=[0.1, 0.5, 0.9])} if kind == "functional_form" else {
            "robustness_replay_spec": RobustnessReplaySpec(products=[{"product_id": "existing", "output": "robustness_matrix"}]),
        }
        revised = previous.model_copy(update={"steps": [*previous.steps[:-1], step.model_copy(update=update)]})
    with pytest.raises(owner.ResearchPipelineRunError):
        _binding_repair_guard(previous).validate_plan(plan=revised, context=_context())
