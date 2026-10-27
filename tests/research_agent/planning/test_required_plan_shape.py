"""One context-derived checklist reaches planning and scientific review."""

from __future__ import annotations

import json

import pytest

from easyicu.research_agent.agents.family_spec_planner import family_spec_messages
from easyicu.research_agent.agents.progressive_planner import ProgressivePlannerAgent
from easyicu.research_agent.planning.family_spec import build_family_spec_request
from easyicu.research_agent.planning.scientific_review import build_plan_scientific_review
from tests.support.descriptive_plans import descriptive_context


def _complete_article_roles(context, plan):
    """Give each required article role a separate typed product on the primary lineage."""
    from easyicu.research_agent.planning.publication_readiness import required_plan_shape
    from easyicu.research_agent.schema import AnalysisStep

    primary = next(step for step in plan.steps if step.planned_analysis_role == "primary")
    roles = [item for item in required_plan_shape(context, analysis_type=plan.analysis_type) if item.kind == "article_role"]
    return plan.model_copy(update={"steps": [*plan.steps, *[
        AnalysisStep(step_id=f"role_{index}", intent="Declare the required evidence product.",
                     method="visualization", inputs=[next(value for value in primary.expected_outputs if value.startswith("table:"))],
                     expected_outputs=[f"figure:{item.module}"])
        for index, item in enumerate(roles)
    ]]})


def _summary_case(*, grouped=False):
    from easyicu.research_agent.schema import AnalysisPlan, AnalysisStep, ConceptDescriptor, EndpointSpec, VariableRole

    context = descriptive_context(variable="value", question="Describe the measured response distribution.")
    context.variables[-1] = context.variables[-1].model_copy(update={"role": VariableRole.OUTCOME})
    steps = [AnalysisStep(
        step_id="summary", planned_analysis_role="primary", method="empirical_summary",
        scientific_action_id="descriptive.descriptive_summary", intent="Describe the response distribution.",
        inputs=["artifact:analysis_cohort", "value"],
        expected_outputs=["table:cohort_summary", "table:distribution_summary"],
    )]
    if grouped:
        context.variables.append(ConceptDescriptor(
            name="ward", role="demographic", dtype="int64",
            observed_domain={"levels": [0, 1], "n_unique": 2, "is_binary": True},
        ))
        context = context.model_copy(update={
            "research_question": "Describe the measured response in each ward.",
            "primary_exposure": "ward", "target_outcome": "value",
            "endpoint": EndpointSpec(name="value", kind="continuous", absence_semantics="no_absent_rows"),
            "cohort": context.cohort.model_copy(update={"requested_outcome_columns": ["value"], "outcome_columns": ["value"]}),
        })
        steps.append(AnalysisStep(
            step_id="grouped", planned_analysis_role="secondary", method="continuous_outcome_summary",
            intent="Describe response in each ward without inference.",
            inputs=["artifact:analysis_cohort", "ward", "value"], expected_outputs=["table:continuous_outcome_summary"],
        ))
    plan = AnalysisPlan(research_question=context.research_question, analysis_type="descriptive_epidemiology", steps=steps)
    return context, _complete_article_roles(context, plan)


def _cluster_case(*, trajectory=False):
    from tests.research_agent.planning.family_spec_fixtures import (
        _phenotyping_context, _phenotyping_payload, _request, _run,
    )

    context = _phenotyping_context()
    if trajectory:
        from easyicu.research_agent.contracts.trajectory_design import load_trajectory_design, sealed_trajectory_authority_body
        from easyicu.research_agent.orchestration.scientific_runtime import ScientificRuntimeAuthorities
        from easyicu.research_agent.schema import AnalysisStep, ConceptDescriptor
        from easyicu.research_agent.trajectory.scientific_runtime_authority import build_trajectory_scientific_runtime_authority

        coordinates = ("sofa2_cardio", "sofa2_renal", "lact")
        context = context.model_copy(update={
            "research_question": "Which trajectory classes emerge from repeated measurements over 72 hours, and how does hospital mortality differ by class?",
            "variables": [*context.variables, *[
                ConceptDescriptor(name=name, role="lab", dtype="float64", analysis_window="icu_admission[0,72]h")
                for name in coordinates
            ]],
        })
        authority = build_trajectory_scientific_runtime_authority(sealed_trajectory_authority_body(
            load_trajectory_design({"coordinate_concepts": list(coordinates)}), protocol_content_sha256="0" * 64,
        ))
        plan = authority.development_execution_only_plan(research_question=context.research_question)
        comparison = AnalysisStep(
            step_id="outcome_by_class", planned_analysis_role="secondary", method="descriptive_outcome_by_cluster",
            scientific_action_id="phenotyping.outcome_by_cluster", intent="Describe the outcome by frozen class.",
            inputs=["stay_id", "death", "artifact:analysis_cohort", "table:cluster_assignments", "artifact:stability_freeze"],
            expected_outputs=["table:outcome_by_cluster"], phenotype_comparison_spec={
                "identity_column": "stay_id", "outcome_columns": ["death"], "variables": [{
                    "name": "death", "variable_kind": "categorical", "summary": "count_percent",
                    "test": "none_descriptive_smd_only", "levels": [0, 1],
                }],
            },
        )
        plan, _ = ScientificRuntimeAuthorities(trajectory=authority, current_case=None).bind_plan(
            plan.model_copy(update={"steps": [*plan.steps, comparison], "endpoint": context.endpoint})
        )
        return context, _complete_article_roles(context, plan)
    request = _request(context)
    _, result = _run(context, [json.dumps(_phenotyping_payload(
        request, features=["hr_max", "lactate_max", "map_min"], baseline=[], membership="phenotype_flag",
    ))])
    return context, _complete_article_roles(context, result.output)


def _association_case():
    from tests.research_agent.planning.scientific_review_fixtures import _context, _plan

    context = _context()
    context = context.model_copy(update={
        "cohort": context.cohort.model_copy(update={"n_stays": 0}),
        "user_preferences": context.user_preferences.model_copy(update={"data_constraints": json.dumps({
            "stated_adjustments": [{"variable": "age", "concept": "age", "words": "adjusted for age", "source_status": "available"}],
        })}),
    })
    return context, _complete_article_roles(context, _plan())


@pytest.mark.parametrize("family", ["descriptive", "grouped_continuous", "association", "phenotyping", "trajectory"])
def test_context_shape_and_review_agree_across_families(family):
    from easyicu.research_agent.planning.publication_readiness import required_plan_shape

    if family in {"descriptive", "grouped_continuous"}:
        context, plan = _summary_case(grouped=family == "grouped_continuous")
    elif family == "association":
        context, plan = _association_case()
    else:
        context, plan = _cluster_case(trajectory=family == "trajectory")
    shape = required_plan_shape(context, analysis_type=plan.analysis_type)
    codes = {item.reason_code for item in shape}
    review = build_plan_scientific_review(context=context, plan=plan)
    assert not codes & {f.code for f in review.findings}
    if family == "association":
        primary = plan.steps[0]
        requirement = primary.model_requirements[0].model_copy(update={"covariates": [], "model_terms": primary.model_requirements[0].model_terms[:1]})
        steps = [primary.model_copy(update={"model_requirements": [requirement]}), *plan.steps[1:]]
        expected = {"STATED_ADJUSTMENT_MISSING"}
    elif family == "descriptive":
        steps = [plan.steps[0].model_copy(update={"scientific_action_id": None}), *plan.steps[1:]]
        expected = {"DESCRIPTIVE_PRIMARY_SUMMARY_MISSING"}
    elif family == "grouped_continuous":
        steps = [step for step in plan.steps if step.step_id != "grouped"]
        expected = {"REQUESTED_OUTCOME_COVERAGE_INCOMPLETE"}
    else:
        steps = [step for step in plan.steps if step.phenotype_comparison_spec is None]
        expected = {"PHENOTYPING_OUTCOME_COMPARISON_INCOMPLETE"}
        if family == "phenotyping":
            expected.add("REQUESTED_OUTCOME_COVERAGE_INCOMPLETE")
    removed = build_plan_scientific_review(context=context, plan=plan.model_copy(update={"steps": steps}))
    assert codes & {f.code for f in removed.findings} == expected


def test_grouped_continuous_requires_the_primary_summary_and_the_grouped_secondary():
    from easyicu.research_agent.agents.progressive_prompt_contracts import DESCRIPTIVE_SUMMARY_CONTRACT
    from easyicu.research_agent.planning.publication_readiness import render_required_plan_shape, required_plan_shape

    context, plan = _summary_case(grouped=True)
    requirements = required_plan_shape(context)
    primary = next(item for item in requirements if item.kind == "primary_summary")
    secondary = next(item for item in requirements if item.kind == "outcome")
    assert (primary.action, primary.planned_analysis_role, primary.inputs) == ("descriptive.descriptive_summary", "primary", ("value", "ward"))
    assert (secondary.module, secondary.planned_analysis_role, secondary.inputs) == ("continuous_outcome_summary", "secondary", ("ward", "value"))
    # The prompt contract lets the primary summarize the measurement overall or
    # within the grouping, and so do the checklist and the review.
    assert "overall or within" in DESCRIPTIVE_SUMMARY_CONTRACT
    assert "value only, overall or within ward" in render_required_plan_shape(context)
    grouped_primary = plan.model_copy(update={"steps": [
        step.model_copy(update={"inputs": [*step.inputs, "ward"]}) if step.step_id == "summary" else step
        for step in plan.steps
    ]})
    codes = {f.code for f in build_plan_scientific_review(context=context, plan=grouped_primary).findings}
    assert not codes & {"DESCRIPTIVE_PRIMARY_SUMMARY_MISSING", "PRIMARY_DESCRIPTIVE_SUMMARY_SCOPE_UNRESOLVED"}


def _with_measurements(context, plan, names):
    from easyicu.research_agent.schema import ConceptDescriptor

    context = context.model_copy(update={"variables": [
        *context.variables,
        *[ConceptDescriptor(name=name, role="lab", dtype="float64") for name in names],
    ]})
    plan = plan.model_copy(update={"steps": [
        step.model_copy(update={"inputs": [*step.inputs, *names]}) if step.step_id == "summary" else step
        for step in plan.steps
    ]})
    return context, plan


def test_one_primary_over_several_measurements_draws_exactly_one_finding():
    from easyicu.research_agent.agents.progressive_prompt_contracts import DESCRIPTIVE_SUMMARY_CONTRACT
    from easyicu.research_agent.planning.publication_readiness import render_required_plan_shape, required_plan_shape

    context, plan = _with_measurements(*_summary_case(), ["hr"])
    shape = required_plan_shape(context)
    # Several measurements: no one-measurement summary is demanded, and the
    # checklist names the audit-led primary the prompt contract prescribes for
    # a whether/how-often/when-recorded question.
    assert not [item for item in shape if item.kind == "primary_summary"]
    audit = next(item for item in shape if item.kind == "measurement_audit")
    assert (audit.module, audit.planned_analysis_role, audit.inputs) == ("measurement_audit", "primary", ("value", "hr"))
    assert "make that step the primary" in DESCRIPTIVE_SUMMARY_CONTRACT
    assert "primary measurement_audit: value,hr if asked whether/how often/when recorded" in render_required_plan_shape(context)
    review = build_plan_scientific_review(context=context, plan=plan)
    codes = [f.code for f in review.findings
             if f.code in {"DESCRIPTIVE_PRIMARY_SUMMARY_MISSING", "PRIMARY_DESCRIPTIVE_SUMMARY_SCOPE_UNRESOLVED"}]
    assert codes == ["PRIMARY_DESCRIPTIVE_SUMMARY_SCOPE_UNRESOLVED"]
    finding = next(f for f in review.findings if f.code == "PRIMARY_DESCRIPTIVE_SUMMARY_SCOPE_UNRESOLVED")
    (repair,) = finding.structural_remediations
    assert (repair.operation, repair.step_id, repair.module, repair.planned_analysis_role) == (
        "replace_module", "summary", "measurement_audit", "primary",
    )
    assert {"value", "hr"} <= set(repair.inputs)


def test_a_one_measurement_primary_with_an_unknown_column_keeps_one_finding():
    context, plan = _summary_case()
    plan = plan.model_copy(update={"steps": [
        step.model_copy(update={"inputs": [*step.inputs, "not_a_column"]}) if step.step_id == "summary" else step
        for step in plan.steps
    ]})
    review = build_plan_scientific_review(context=context, plan=plan)
    codes = [f.code for f in review.findings
             if f.code in {"DESCRIPTIVE_PRIMARY_SUMMARY_MISSING", "PRIMARY_DESCRIPTIVE_SUMMARY_SCOPE_UNRESOLVED"}]
    assert codes == ["PRIMARY_DESCRIPTIVE_SUMMARY_SCOPE_UNRESOLVED"]
    finding = next(f for f in review.findings if f.code == "PRIMARY_DESCRIPTIVE_SUMMARY_SCOPE_UNRESOLVED")
    (repair,) = finding.structural_remediations
    assert (repair.operation, repair.step_id, repair.action) == ("replace_module", "summary", "descriptive.descriptive_summary")


def test_only_the_actual_context_contributes_requirements_and_article_roles():
    from easyicu.research_agent.planning.publication_readiness import required_plan_shape
    from easyicu.research_agent.reporting.article_contract import build_article_analysis_contract

    context, _ = _association_case()
    shape = required_plan_shape(context)
    assert [item.inputs for item in shape if item.kind == "adjustment"] == [("age",)]
    assert not [item for item in shape if item.kind in {"subgroup", "effect_measure", "primary_summary", "comparison"}]
    assert {item.article_role for item in shape if item.kind == "article_role"} == set(build_article_analysis_contract(context).required_roles)
    assert not [item for item in required_plan_shape(context, execution_tier="first_result") if item.kind == "article_role"]


@pytest.mark.parametrize("grouped", [False, True])
def test_the_review_checks_every_required_article_role(grouped):
    from easyicu.research_agent.planning.publication_readiness import required_plan_shape

    context, plan = _summary_case(grouped=grouped)
    required = {item.article_role for item in required_plan_shape(context) if item.kind == "article_role"}
    review = build_plan_scientific_review(context=context, plan=plan.model_copy(update={"steps": []}))
    assert set(review.facts["content_roles"]["missing_roles"]) == required
    assert {f.code for f in review.findings if f.code.startswith("ARTICLE_CONTENT")} == {"ARTICLE_CONTENT_ROLES_INCOMPLETE"}


def test_stated_subgroups_and_effect_measures_share_the_existing_coverage_owners():
    from easyicu.research_agent.planning.question_requirements import judge_stated_effect_measures, question_effect_measures
    from easyicu.research_agent.planning.publication_readiness import required_plan_shape
    from easyicu.research_agent.planning.requirement_coverage import stated_subgroup_coverage
    from easyicu.research_agent.schema import AnalysisStep

    context, plan = _association_case()
    context = context.model_copy(update={
        "research_question": "Estimate the association as a risk ratio, within age strata.",
        "user_preferences": context.user_preferences.model_copy(update={"data_constraints": json.dumps({
            "stated_subgroups": [{"variable": "age", "concept": "age", "words": "within age strata", "source_status": "available"}],
        })}),
    })
    shape = required_plan_shape(context)
    subgroup = next(item for item in shape if item.kind == "subgroup")
    measure = next(item for item in shape if item.kind == "effect_measure")
    assert subgroup.inputs == ("age",)
    assert measure.effect_measure == "risk_ratio"
    assert stated_subgroup_coverage(context, plan).records[0].status == "missing"
    subgroup_step = AnalysisStep(
        step_id="subgroup", planned_analysis_role="secondary", method="subgroup_analysis",
        scientific_action_id=subgroup.action, intent="Estimate associations within age strata.",
        inputs=["exposure", "death", "age"], expected_outputs=["table:stratum_estimates"],
    )
    with_subgroup = plan.model_copy(update={"steps": [*plan.steps, subgroup_step]})
    assert stated_subgroup_coverage(context, with_subgroup).complete
    assert "STATED_SUBGROUP_MISSING" in {f.code for f in build_plan_scientific_review(context=context, plan=plan).findings}
    assert "STATED_SUBGROUP_MISSING" not in {f.code for f in build_plan_scientific_review(context=context, plan=with_subgroup).findings}
    requirement = plan.steps[0].model_requirements[0].model_copy(update={"method_family": "statsmodels_logit_mle"})
    plan = plan.model_copy(update={"steps": [plan.steps[0].model_copy(update={"model_requirements": [requirement]}), *plan.steps[1:]]})
    judgments = judge_stated_effect_measures(question_effect_measures(context.research_question), plan=plan, requirements=[])
    assert judgments[0].reason_code == measure.reason_code
    requirement = plan.steps[0].model_requirements[0].model_copy(update={"method_family": "statsmodels_glm_poisson_log_robust"})
    with_measure = plan.model_copy(update={"steps": [plan.steps[0].model_copy(update={"model_requirements": [requirement]}), *plan.steps[1:]]})
    assert judge_stated_effect_measures(question_effect_measures(context.research_question), plan=with_measure, requirements=[])[0].disposition == "covered"


def test_outline_receives_the_host_required_plan_shape():
    context = descriptive_context()
    families, variables, _, rows = ProgressivePlannerAgent._request_authorities(context)
    prompt = ProgressivePlannerAgent._user_prompt(
        context, analysis_types=families, variables=variables, action_rows=rows,
    )
    assert "Required plan shape (host):" in prompt
    assert "primary descriptive.descriptive_summary" in prompt


def test_family_receives_the_same_context_derived_checklist():
    from easyicu.research_agent.planning.publication_readiness import render_required_plan_shape
    from tests.research_agent.planning.family_spec_fixtures import _descriptive_context

    context = _descriptive_context()
    request = build_family_spec_request(
        context, analysis_types=["descriptive_epidemiology"],
        variable_roster=[v.name for v in context.variables], allowed_literature_citation_keys=(),
    )
    messages = family_spec_messages(
        request, context=context, variable_descriptions={},
    )
    assert render_required_plan_shape(context) in messages[1].content


def test_review_records_the_same_shape_and_missing_outcome_remediation():
    from easyicu.research_agent.planning.publication_readiness import required_plan_shape
    from tests.research_agent.planning.family_spec_fixtures import (
        _phenotyping_context, _phenotyping_payload, _request, _run,
    )

    context = _phenotyping_context()
    request = _request(context)
    llm, result = _run(context, [json.dumps(_phenotyping_payload(
        request, features=["hr_max", "lactate_max", "map_min"], baseline=[],
        membership="phenotype_flag",
    ))])
    assert "Required plan shape (host):" in llm.calls[0][0][1].content
    plan = result.output
    plan = plan.model_copy(update={"steps": [
        step for step in plan.steps if step.phenotype_comparison_spec is None
    ]})
    review = build_plan_scientific_review(context=context, plan=plan)
    assert review.facts["required_plan_shape"] == [
        item.model_dump(mode="json") for item in required_plan_shape(context, analysis_type=plan.analysis_type)
    ]
    finding = next(f for f in review.findings if f.code == "PHENOTYPING_OUTCOME_COMPARISON_INCOMPLETE")
    assert finding.structural_remediations
    repair = finding.structural_remediations[0]
    assert repair.operation == "add_step"
    assert repair.action == "phenotyping.outcome_by_cluster"
    assert repair.planned_analysis_role == "secondary"
    assert "death" in repair.inputs
    assert review.schema_version == "easyicu.plan_scientific_review/18"


@pytest.mark.parametrize("family", ["grouped_continuous", "phenotyping", "trajectory"])
def test_the_reviews_derived_scope_admits_the_actual_missing_step(family):
    from easyicu.webserver.research_plan_revision import PlanRevisionGuard

    context, plan = _summary_case(grouped=True) if family == "grouped_continuous" else _cluster_case(trajectory=family == "trajectory")
    missing_id = "grouped" if family == "grouped_continuous" else next(step.step_id for step in plan.steps if step.phenotype_comparison_spec)
    source = plan.model_copy(update={"steps": [step for step in plan.steps if step.step_id != missing_id]})
    review = build_plan_scientific_review(context=context, plan=source)
    guard = PlanRevisionGuard.from_review(plan=source.model_dump(mode="json"), review=review)
    receipt = guard.validate_plan(plan=plan, context=context)
    assert len(receipt.structural_changes) == 1
    assert receipt.structural_changes[0].step_id == missing_id
    assert receipt.structural_changes[0].finding_code in {
        "REQUESTED_OUTCOME_COVERAGE_INCOMPLETE", "PHENOTYPING_OUTCOME_COMPARISON_INCOMPLETE",
    }


def test_article_advice_cannot_authorize_an_already_covered_role():
    context, plan = _summary_case(grouped=True)
    # Keep cohort/data-quality roles, omit only the descriptive result's role.
    source = plan.model_copy(update={"steps": [
        step.model_copy(update={"scientific_action_id": None, "expected_outputs": ["table:unrelated"]})
        if step.step_id == "summary" else step
        for step in plan.steps if "figure:distribution_prevalence" not in step.expected_outputs
    ]})
    review = build_plan_scientific_review(context=context, plan=source)
    finding = next(f for f in review.findings if f.code == "ARTICLE_CONTENT_ROLES_INCOMPLETE")
    assert "cohort_accounting" not in review.facts["content_roles"]["missing_roles"]
    assert "data_quality" not in review.facts["content_roles"]["missing_roles"]
    assert not finding.structural_remediations
