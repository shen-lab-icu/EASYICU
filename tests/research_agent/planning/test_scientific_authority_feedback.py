"""Synthetic current-case refusals belong inside the Planner's repair budget."""

import json
from types import SimpleNamespace

import pytest

from easyicu.research_agent.agents import progressive_planner as owner
from easyicu.research_agent.authority.current_case_scientific_runtime import (
    CurrentCaseScientificAuthorityError,
    build_current_case_scientific_runtime_authority,
)
from easyicu.research_agent.contracts.association_execution import (
    ASSOCIATION_BINARY_SENSITIVITY_CAPABILITY_ID,
)
from easyicu.research_agent.orchestration.scientific_runtime import ScientificRuntimeAuthorities
from easyicu.research_agent.planning.progressive_contract import ProgressivePlanCompileError
from easyicu.research_agent.providers.mocks import ScriptedMockLLMClient
from easyicu.research_agent.providers.structured_retry import call_llm_with_structured_retry, StructuredResponseFailure
from easyicu.research_agent.contracts.functional_form import FunctionalFormSpec
from easyicu.research_agent.schema import AnalysisPlan, AnalysisStep

from .progressive_planner_fixtures import _context, _outline_payload


def _authority():
    return build_current_case_scientific_runtime_authority({
        "schema_version": "easyicu.landmark_spline_runtime_authority/1",
        "authority_kind": "landmark_spline_association",
        "protocol_content_sha256": "a" * 64,
        "plan_method": "signed_landmark_restricted_cubic_spline",
        "plan_intent": "Estimate the approved synthetic landmark association.",
        "plan_outputs": ["table:curve", "table:contrasts", "table:linear_sensitivity", "log:receipt"],
        "exposure_column": "exposure_flag", "outcome_column": "outcome_flag",
        "outcome_time_column": "event_hours", "observation_duration_column": "followup_hours",
        "observation_duration_unit": "hours", "landmark_hours": 24,
        "required_adjustment_columns": ["age_years"], "categorical_adjustment_columns": [],
        "alternative_exposure_columns": [], "dependence": None,
        "adjusted_absolute_risk_product": None, "population_flow_product": None,
        "variable_opportunity_sensitivity_product": None,
        "spline_knot_quantiles": [0.1, 0.5, 0.9], "spline_reference": "median_in_primary_population",
        "curve_quantile_range": [0.1, 0.9], "curve_points": 9,
        "linear_sensitivity_per_unit": 1.0,
        "interpretation": "descriptive_prognostic_association_not_causal",
    })


def _plan(*, sensitivity=True):
    steps = [AnalysisStep(
        step_id="primary", planned_analysis_role="primary", intent="Approved synthetic model",
        method="adjusted_association_models", inputs=["artifact:analysis_cohort", "exposure_flag", "outcome_flag"],
        expected_outputs=["table:adjusted_association_estimates"],
        model_requirements=[{
            "requirement_id": "primary_model", "outcome": "outcome_flag", "outcome_type": "binary",
            "method_family": "statsmodels_logit_mle", "exposure_source": "exposure_flag",
            "analysis_role": "primary", "analysis_set": "source_aware", "covariates": ["age_years"],
            "model_terms": [
                {"name": "exposure_flag", "role": "exposure", "coding": "continuous", "transform": "identity"},
                {"name": "age_years", "role": "covariate", "coding": "continuous", "transform": "identity"},
            ],
        }],
    )]
    if sensitivity:
        steps.append(AnalysisStep(
            step_id="optional_adjustment", planned_analysis_role="sensitivity",
            intent="Optional alternative adjustment check", method="alternative_adjustment",
            scientific_capability=ASSOCIATION_BINARY_SENSITIVITY_CAPABILITY_ID,
            sensitivity_spec_ids=["optional_spec"], inputs=["table:adjusted_association_estimates"],
            expected_outputs=["table:optional_sensitivity"],
        ))
    return AnalysisPlan(research_question="Synthetic measurement and outcome association.", steps=steps)


def test_final_acceptance_checks_current_authority_before_returning_a_plan(monkeypatch):
    authority, plan = _authority(), _plan()
    # The original owner refuses it, but formerly only after the Planner returned.
    with pytest.raises(CurrentCaseScientificAuthorityError) as refused:
        authority.bind_plan(plan)
    assert refused.value.reason_code == "landmark_spline_binary_sensitivity_unsupported"
    receipt = SimpleNamespace(model_copy=lambda **_kwargs: None)
    monkeypatch.setattr(owner, "compile_progressive_plan", lambda **_kwargs: (plan, receipt))
    monkeypatch.setattr(owner, "_accept_compiled_plan", lambda **_kwargs: None)
    agent = owner.ProgressivePlannerAgent(SimpleNamespace())
    agent.scientific_authorities = ScientificRuntimeAuthorities(trajectory=None, current_case=authority)
    with pytest.raises(ProgressivePlanCompileError) as feedback:
        agent._compile_and_accept(
            SimpleNamespace(analysis_type="association_study"),
            agent_context=_context(), article_context=_context(),
            allowed_literature_citation_keys=(), direct_comparator_literature_keys=(),
            allowed_know_how_decisions=None, enforce_article_contract=False,
        )
    assert feedback.value.reason_code == "progressive_scientific_authority_repairable"
    assert feedback.value.step_id == "optional_adjustment"
    assert "optional_spec" in str(feedback.value)


def _authorities():
    return ScientificRuntimeAuthorities(trajectory=None, current_case=_authority())


def _functional_plan():
    plan = _plan()
    plan.steps[-1] = plan.steps[-1].model_copy(update={
        "method": "linear_per_unit_sensitivity",
        "functional_form_spec": FunctionalFormSpec(target_column="exposure_flag", knot_quantiles=(0.1, 0.5, 0.9)),
    })
    return plan


@pytest.mark.parametrize("repaired", [_plan(sensitivity=False), _functional_plan()], ids=["drop-optional", "supported-form"])
def test_refusal_is_repaired_within_the_same_structured_budget(repaired):
    authorities, original = _authorities(), _plan()
    before = original.model_dump_json()
    llm = ScriptedMockLLMClient([original.model_dump_json(), repaired.model_dump_json()])
    events = []
    context = _context()
    assert context.cohort.database == "miiv"  # MIMIC-IV fixture, no patient records.
    result = call_llm_with_structured_retry(
        llm, [], parser=lambda raw: authorities.check_planner_plan(AnalysisPlan.model_validate_json(raw), context),
        retryable_parse_error=authorities.planner_retryable_error, max_retries=1, progress_callback=events.append,
    )
    assert len(llm.calls) == 2
    assert [event.phase for event in events] == ["started", "rejected", "started", "accepted"]
    feedback = llm.calls[1][0][-1].content
    for expected in ("optional_adjustment", "optional_spec", ASSOCIATION_BINARY_SENSITIVITY_CAPABILITY_ID, "functional_form_spec", "approved exposure"):
        assert expected in feedback
    assert result == repaired
    authorities.current_case.validate_plan(authorities.current_case.bind_plan(result))
    assert original.model_dump_json() == before


def test_supported_sensitivity_is_kept_without_spending_a_retry():
    authorities, plan = _authorities(), _functional_plan()
    before = plan.model_dump_json()
    assert authorities.check_planner_plan(plan, _context()) is plan
    assert plan.model_dump_json() == before
    bound = authorities.current_case.bind_plan(plan)
    authorities.current_case.validate_plan(bound)
    assert bound.steps[-1].functional_form_spec == plan.steps[-1].functional_form_spec


def test_authority_refusal_exhaustion_records_every_attempt(tmp_path):
    from easyicu.research_agent.orchestration.scientific_runtime import ScientificAuthorityRepairError

    authorities, plan = _authorities(), _plan()
    llm = ScriptedMockLLMClient([plan.model_dump_json()] * 4)
    with pytest.raises(StructuredResponseFailure) as caught:
        call_llm_with_structured_retry(
            llm, [], parser=lambda raw: authorities.check_planner_plan(AnalysisPlan.model_validate_json(raw), _context()),
            retryable_parse_error=authorities.planner_retryable_error, max_retries=2,
        )
    assert len(llm.calls) == 3
    history = caught.value.easyicu_structured_attempt_metadata
    assert len(history) == 3
    assert {item["reason_code"] for item in history} == {"progressive_scientific_authority_repairable"}
    assert isinstance(caught.value.__cause__, ScientificAuthorityRepairError)
    from easyicu.webserver.agent_pipeline_runs import _pipeline_failure_code, _record_pipeline_failure

    _record_pipeline_failure(
        wrapper_dir=tmp_path, study={"id": "synthetic"}, provider={}, exc=caught.value,
        code=_pipeline_failure_code(caught.value), execution_retry_id=None,
    )
    receipt = json.loads((tmp_path / "diagnostics/research_pipeline_failure.json").read_text())
    assert receipt["structured_retry_history"] == history
    assert not receipt["raw_model_output_recorded"]
    with pytest.raises(CurrentCaseScientificAuthorityError):
        authorities.current_case.bind_plan(plan)  # Repair never relaxes the owner.


@pytest.mark.parametrize("coordinate", ["exposure_source", "outcome"])
def test_approved_design_refusal_is_terminal(coordinate):
    authorities, plan = _authorities(), _plan(sensitivity=False)
    requirement = plan.steps[0].model_requirements[0]
    plan.steps[0].model_requirements = [requirement.model_copy(update={coordinate: "different_variable"})]
    llm = ScriptedMockLLMClient(["{}", "{}"])
    with pytest.raises(CurrentCaseScientificAuthorityError):
        call_llm_with_structured_retry(
            llm, [], parser=lambda _raw: authorities.check_planner_plan(plan, _context()),
            retryable_parse_error=authorities.planner_retryable_error, max_retries=2,
        )
    assert len(llm.calls) == 1


def test_a_risk_ratio_primary_is_refused_by_the_odds_ratio_spline_suite():
    # The signed spline suite fits a logistic model: an approved plan that
    # declares a risk ratio is a terminal design refusal, not a repair that
    # would hand back an odds ratio under the risk-ratio request.
    authorities, plan = _authorities(), _plan(sensitivity=False)
    requirement = plan.steps[0].model_requirements[0]
    plan.steps[0].model_requirements = [type(requirement).model_validate({
        **requirement.model_dump(mode="json"), "method_family": "statsmodels_glm_poisson_log_robust",
    })]
    llm = ScriptedMockLLMClient(["{}", "{}"])
    with pytest.raises(CurrentCaseScientificAuthorityError) as refused:
        call_llm_with_structured_retry(
            llm, [], parser=lambda _raw: authorities.check_planner_plan(plan, _context()),
            retryable_parse_error=authorities.planner_retryable_error, max_retries=2,
        )
    assert refused.value.reason_code == "landmark_spline_risk_ratio_unsupported"
    assert len(llm.calls) == 1


def test_unknown_authority_refusal_is_terminal():
    authorities, plan = _authorities(), _plan(sensitivity=False)
    plan.steps[0].inputs = ["exposure_flag", "outcome_flag"]
    with pytest.raises(CurrentCaseScientificAuthorityError) as caught:
        authorities.check_planner_plan(plan, _context())
    assert not authorities.planner_retryable_error(caught.value)


def _outline_without_optional():
    payload = _outline_payload()
    payload["steps"] = [step for step in payload["steps"] if step["step_id"] != "06_sensitivity"]
    return payload


def _unsupported_outline():
    payload = _outline_payload()
    next(step for step in payload["steps"] if step["step_id"] == "06_sensitivity")["scientific_action_id"] = "association.multiple_adjustment"
    return payload


def test_progressive_outline_repairs_optional_step_before_sealing():
    rejected = _unsupported_outline()
    repaired = _outline_without_optional()
    llm = ScriptedMockLLMClient([json.dumps(rejected), json.dumps(repaired)])
    agent = owner.ProgressivePlannerAgent(llm, scientific_authorities=_authorities())
    result = agent.run(_context(), stop_after_outline=True)
    assert len(llm.calls) == 2
    assert "06_sensitivity" not in {step.step_id for step in result.steps}
    history = agent.last_result.facts.prompt_metrics["structured_retry_history"]
    assert len(history) == 1
    assert history[0]["reason_code"] == "progressive_scientific_authority_repairable"


def test_progressive_outline_refusal_stops_at_existing_budget():
    llm = ScriptedMockLLMClient([json.dumps(_unsupported_outline())] * 5)
    agent = owner.ProgressivePlannerAgent(llm, scientific_authorities=_authorities())
    with pytest.raises(StructuredResponseFailure) as caught:
        agent.run(_context(), stop_after_outline=True)
    assert len(llm.calls) == owner._MAX_INITIAL_PARSE_RETRIES + 1
    assert len(caught.value.easyicu_compile_retry_metadata) == len(llm.calls)


def _baseline_plan_and_authorities():
    from easyicu.research_agent.authority.baseline_association_runtime import build_baseline_authority
    from tests.support.baseline_association import baseline_context, baseline_plan_attempt, baseline_runtime

    context = baseline_context(covariates=("age",))
    attempt, _ = baseline_plan_attempt(context)
    plan = attempt.output
    original = baseline_runtime(context, plan).current_case
    authority = build_baseline_authority(**{
        **original.model_dump(mode="python", exclude={"execution_contract_sha256"}),
        "exposure_reference_level": "1", "primary_contrast_level": "0",
    })
    return plan, ScientificRuntimeAuthorities(trajectory=None, current_case=authority)


def test_baseline_contrast_feedback_names_the_signed_values():
    from easyicu.research_agent.orchestration.scientific_runtime import ScientificAuthorityRepairError

    plan, authorities = _baseline_plan_and_authorities()
    with pytest.raises(ScientificAuthorityRepairError) as caught:
        authorities.check_planner_plan(plan)
    support = caught.value.details["findings"][0]["supported_repair"]
    assert support["reference_level"] == "1" and support["primary_contrast_level"] == "0"
    assert caught.value.step_id == "adjusted_association"
    assert caught.value.path == "model_requirements"
    requirement = plan.steps[3].model_requirements[0]
    plan.steps[3].model_requirements = [requirement.model_copy(update={
        "model_terms": [requirement.model_terms[0].model_copy(update={"reference_level": "1"}), requirement.model_terms[1]],
        "exposure_reference_level": "1", "primary_contrast_level": "0",
    })]
    assert authorities.check_planner_plan(plan) is plan


def test_contrast_repair_cannot_hide_an_adjustment_roster_change():
    from easyicu.research_agent.authority.baseline_association_runtime import BaselineAssociationAuthorityError

    plan, authorities = _baseline_plan_and_authorities()
    requirement = plan.steps[3].model_requirements[0]
    plan.steps[3].model_requirements = [requirement.model_copy(update={"covariates": []})]
    with pytest.raises(BaselineAssociationAuthorityError) as caught:
        authorities.check_planner_plan(plan)
    assert not authorities.planner_retryable_error(caught.value)


def test_later_design_stop_preserves_prior_retry_history():
    authorities, optional, changed = _authorities(), _plan(), _plan(sensitivity=False)
    changed.steps[0].model_requirements[0].exposure_source = "different_variable"
    candidates = iter([optional, changed])
    llm = ScriptedMockLLMClient(["{}"] * 3)
    with pytest.raises(CurrentCaseScientificAuthorityError) as caught:
        call_llm_with_structured_retry(
            llm, [], parser=lambda _raw: authorities.check_planner_plan(next(candidates), _context()),
            retryable_parse_error=authorities.planner_retryable_error, max_retries=2,
        )
    assert len(llm.calls) == 2
    assert len(caught.value.easyicu_structured_attempt_metadata) == 1


def test_supported_outline_is_not_forced_to_drop_a_functional_form_check():
    from easyicu.research_agent.planning.progressive_contract import ProgressivePlanOutline

    outline = _outline_payload()
    for action in (None, "association.rcs_spline"):
        next(step for step in outline["steps"] if step["step_id"] == "06_sensitivity")["scientific_action_id"] = action
        _authorities().check_planner_outline(ProgressivePlanOutline.model_validate(outline))


def test_full_plan_agent_repairs_and_retains_its_retry_receipt():
    from easyicu.research_agent.agents.planner import PlannerAgent

    original, repaired = _plan(), _plan(sensitivity=False)
    for plan in (original, repaired):
        plan.steps[0].scientific_action_id = "association.adjusted_association"
    original.steps[-1].scientific_action_id = "association.multiple_adjustment"
    llm = ScriptedMockLLMClient([original.model_dump_json(), repaired.model_dump_json()])
    agent = PlannerAgent(llm, scientific_authorities=_authorities())
    result = agent.run(_context())
    assert len(llm.calls) == 2
    assert [step.step_id for step in result.steps] == ["primary"]
    assert agent.last_prompt_metrics["structured_retry_history"][0]["reason_code"] == "progressive_scientific_authority_repairable"


@pytest.mark.parametrize("coordinate", ["population", "estimand"])
def test_untyped_design_refusals_do_not_become_schema_retries(monkeypatch, coordinate):
    authorities = _authorities()

    def refuse(_self, _plan):
        raise ValueError(f"signed {coordinate} differs")

    monkeypatch.setattr(type(authorities.current_case), "validate_plan", refuse)
    llm = ScriptedMockLLMClient(["{}", "{}"])
    with pytest.raises(CurrentCaseScientificAuthorityError):
        call_llm_with_structured_retry(
            llm, [], parser=lambda _raw: authorities.check_planner_plan(_plan(sensitivity=False), _context()),
            retryable_parse_error=authorities.planner_retryable_error, max_retries=1,
        )
    assert len(llm.calls) == 1
