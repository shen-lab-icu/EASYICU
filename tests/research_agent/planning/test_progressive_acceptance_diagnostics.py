"""Acceptance failures keep their owner coordinate without exposing candidate text."""

import json
from types import SimpleNamespace

import pytest

from easyicu.research_agent.agents import progressive_planner as planner
from easyicu.research_agent.planning.progressive_compiler import required_reader_display_label_keys
from easyicu.research_agent.planning.progressive_contract import ProgressivePlanCompileError

from .progressive_planner_fixtures import (
    _context, _foundation_payload, _materialization_payloads, _outline_payload, _payload,
)


def test_unsupported_optional_design_input_retries_outline_and_completes_plan():
    from easyicu.research_agent.providers.mocks import ScriptedMockLLMClient

    context = _context()
    optional = context.variables[-1].model_copy(update={
        "name": "optional_repeat_visit", "source_concept": "icu_readmission",
    })
    context = context.model_copy(update={
        "cohort": context.cohort.model_copy(update={"database": "miiv"}),
        "variables": [*context.variables, optional],
    })
    invalid = _outline_payload()
    selected = next(c for c in invalid["design_selection"]["candidates"]
                    if c["disposition"] == "selected")
    selected["required_variables"].append(optional.name)
    responses = [invalid, _outline_payload(), _foundation_payload(), *_materialization_payloads()]
    llm = ScriptedMockLLMClient([json.dumps(item) for item in responses])
    llm.supports_strict_json_schema = True
    before = context.model_dump_json()
    plan = planner.ProgressivePlannerAgent(llm).run(context)
    assert len(plan.steps) == 7
    assert len(llm.calls) == 10
    assert "progressive_design_input_structurally_unavailable" in llm.calls[1][0][-1].content
    assert "optional_repeat_visit" in llm.calls[1][0][-1].content
    assert [call[1]["structured_output"].name for call in llm.calls[:3]] == [
        "easyicu_progressive_plan_outline_v1", "easyicu_progressive_plan_outline_v1",
        "easyicu_progressive_plan_foundation_v1",
    ]
    assert context.model_dump_json() == before


def test_selected_family_conflict_is_inside_outline_retry_boundary():
    from easyicu.research_agent.schema import UserPreferences
    from easyicu.research_agent.planning.progressive_contract import ProgressivePlanOutline

    context = _context().model_copy(update={"user_preferences": UserPreferences(
        data_constraints=json.dumps({"analysis_design": {
            "analysis_unit": "icu_stay", "variance_estimator": "none_counts_only",
        }}),
    )})
    before = context.model_dump_json()
    with pytest.raises(ProgressivePlanCompileError) as caught:
        planner.ProgressivePlannerAgent._validate_outline_authority(
            ProgressivePlanOutline.model_validate(_outline_payload()),
            analysis_types=["association_study", "descriptive_epidemiology"],
            variable_names=[v.name for v in context.variables],
            allowed_literature_citation_keys=[], article_context=context,
        )
    assert caught.value.reason_code == "progressive_selected_family_counts_only_incompatible"
    assert caught.value.path == "analysis_type"
    assert context.model_dump_json() == before


def test_contradictory_saved_design_requires_host_revision_without_provider():
    from easyicu.research_agent.schema import UserPreferences
    from easyicu.research_agent.planning.dependence_authority import DependenceAuthorityError

    context = _context().model_copy(update={"user_preferences": UserPreferences(
        inferred_analysis_family="prediction_model",
        data_constraints=json.dumps({"analysis_design": {
            "analysis_family": "prediction_model", "analysis_unit": "icu_stay",
            "variance_estimator": "none_counts_only",
        }}),
    )})
    before = context.model_dump_json()
    with pytest.raises(DependenceAuthorityError) as caught:
        planner.ProgressivePlannerAgent.request_messages(context)
    assert caught.value.code == "counts_only_family_incompatible"
    assert context.model_dump_json() == before


def test_counts_only_selection_feedback_retries_without_changing_host_ceiling():
    from easyicu.research_agent.schema import UserPreferences
    from easyicu.research_agent.planning.progressive_contract import ProgressivePlanOutline
    from easyicu.research_agent.providers.mocks import ScriptedMockLLMClient
    from easyicu.research_agent.providers.protocol import LLMMessage
    from easyicu.research_agent.providers.structured_retry import call_llm_with_structured_retry

    context = _context().model_copy(update={
        "research_question": "Describe exposure and outcome counts in this cohort.",
        "user_preferences": UserPreferences(data_constraints=json.dumps({"analysis_design": {
            "analysis_unit": "icu_stay", "variance_estimator": "none_counts_only",
        }})),
    })
    payload = _payload()
    payload.update(analysis_type="descriptive_epidemiology", robustness_intents=[])
    payload["steps"] = payload["steps"][:4]
    payload["steps"][1].update(planned_analysis_role="secondary", scientific_action_id="descriptive.table_one")
    payload["steps"][2]["planned_analysis_role"] = "primary"
    corrected = _outline_payload(payload)
    selected = next(c for c in corrected["design_selection"]["candidates"] if c["disposition"] == "selected")
    for field in ("estimand", "primary_method", "figure_role", "supports"):
        selected[field] = "Describe counts and proportions in the registered cohort."
    selected["reviewable_plan"] = ["Recommended counts and proportions with explicit denominators."] * 6

    def parse(raw):
        outline = ProgressivePlanOutline.model_validate_json(raw)
        planner.ProgressivePlannerAgent._validate_outline_authority(
            outline, analysis_types=["association_study", "descriptive_epidemiology"],
            variable_names=[v.name for v in context.variables],
            allowed_literature_citation_keys=[], article_context=context,
        )
        return outline

    llm = ScriptedMockLLMClient([json.dumps(_outline_payload()), json.dumps(corrected)])
    before = context.model_dump_json()
    result = call_llm_with_structured_retry(
        llm, [LLMMessage(role="user", content=context.research_question)],
        parser=parse, role="progressive_planner_outline", max_retries=1,
        include_failed_response_on_retry=False,
    )
    assert result.analysis_type == "descriptive_epidemiology"
    assert len(llm.calls) == 2
    assert "progressive_selected_family_counts_only_incompatible" in llm.calls[1][0][-1].content
    assert context.model_dump_json() == before


def test_final_acceptance_runs_once_after_all_validated_prefixes(monkeypatch):
    from easyicu.research_agent.providers.mocks import ScriptedMockLLMClient

    prefixes, accepted = [], []
    compile_prefix = planner.compile_progressive_prefix
    accept = planner._accept_compiled_plan

    def record_prefix(*args, **kwargs):
        result = compile_prefix(*args, **kwargs)
        prefixes.append(len(result.steps))
        return result

    def record_accept(**kwargs):
        accepted.append(len(kwargs["plan"].steps))
        return accept(**kwargs)

    monkeypatch.setattr(planner, "compile_progressive_prefix", record_prefix)
    monkeypatch.setattr(planner, "_accept_compiled_plan", record_accept)
    llm = ScriptedMockLLMClient([json.dumps(p) for p in [
        _outline_payload(), _foundation_payload(), *_materialization_payloads(),
    ]])
    planner.ProgressivePlannerAgent(llm).run(_context())
    assert prefixes == list(range(1, 8))
    assert accepted == [7]


@pytest.mark.parametrize(
    ("validator", "gate"),
    [
        ("validate_literature_citation_bindings", "literature_citation_bindings"),
        ("validate_plan_typed_bindings_against_context", "context_bindings"),
        ("validate_plan_against_adjustment_authority", "adjustment_authority"),
        ("validate_required_primary_result", "primary_result"),
    ],
)
def test_fresh_acceptance_attributes_rejection_without_message_leak(monkeypatch, validator, gate):
    validators = [
        "validate_literature_citation_bindings", "validate_plan_typed_bindings_against_context",
        "validate_plan_against_adjustment_authority", "validate_required_primary_result",
    ]
    for name in validators:
        monkeypatch.setattr(planner, name, lambda *a, **k: None)
    monkeypatch.setattr(planner, "primary_analysis_cohort_plan_findings", lambda **k: [])
    monkeypatch.setattr(planner, "llm_is_mockish", lambda _: True)
    original = ValueError("untrusted_candidate_value_and_private_path")

    def reject(*a, **k):
        raise original

    monkeypatch.setattr(planner, validator, reject)
    step = SimpleNamespace(step_id="primary_estimate", planned_analysis_role="primary")
    plan = SimpleNamespace(steps=[step], robustness_specs=[])
    with pytest.raises(ProgressivePlanCompileError) as caught:
        planner._accept_compiled_plan(
            plan=plan, agent_context=_context(), article_context=_context(),
            allowed_literature_citation_keys=[], direct_comparator_literature_keys=[],
            allowed_know_how_decisions=None, enforce_article_contract=False, llm=object(),
        )
    assert caught.value.reason_code == f"progressive_{gate}_invalid"
    assert caught.value.path == gate
    assert caught.value.__cause__ is original
    assert "untrusted_candidate" not in str(caught.value.easyicu_safe_diagnostic)


@pytest.mark.parametrize("source_field", ["source_concept", "derived_from_concepts"])
def test_reader_label_request_rejects_unsupported_source_before_copywriting(source_field):
    context = _context()
    source = "icu_readmission" if source_field == "source_concept" else ["icu_readmission"]
    variable = context.variables[0].model_copy(update={source_field: source})
    context = context.model_copy(update={
        "cohort": context.cohort.model_copy(update={"database": "miiv"}),
        "variables": [variable, *context.variables[1:]],
    })
    selection = SimpleNamespace(selected=SimpleNamespace(required_variables=[variable.name]))
    before = context.model_dump_json()
    with pytest.raises(ProgressivePlanCompileError) as caught:
        required_reader_display_label_keys(context, selection)
    assert caught.value.reason_code == "progressive_design_input_structurally_unavailable"
    assert caught.value.path == "design_selection.required_variables"
    assert context.model_dump_json() == before


def test_reader_label_request_keeps_unknown_local_source_and_full_required_roster():
    context = _context()
    selection = SimpleNamespace(selected=SimpleNamespace(
        required_variables=["stay_id", "exposure_flag", "outcome_flag", "age_years"]
    ))
    assert required_reader_display_label_keys(context, selection) == (
        "exposure_flag", "outcome_flag", "age_years",
    )


def test_missing_article_roles_reach_safe_diagnostic_without_candidate_text(monkeypatch):
    from easyicu.webserver.agent_pipeline_runs import _safe_pipeline_typed_failure

    monkeypatch.setattr(planner, "validate_literature_citation_bindings", lambda *a, **k: None)
    monkeypatch.setattr(planner, "build_article_analysis_contract", lambda *a, **k: SimpleNamespace(
        required_roles=("causal_protocol", "balance_positivity"),
        requirements=(),
    ))
    monkeypatch.setattr(planner, "validate_plan_against_article_contract", lambda **k: [
        SimpleNamespace(detail={"missing_roles": ["causal_protocol", "balance_positivity"]}),
    ])
    plan = SimpleNamespace(
        steps=[SimpleNamespace(step_id="effect_plan", planned_analysis_role="primary",
                               expected_outputs=[], scientific_action_id=None, method="custom")],
        robustness_specs=[], analysis_type="causal_inference",
    )
    with pytest.raises(ProgressivePlanCompileError) as caught:
        planner._accept_compiled_plan(
            plan=plan, agent_context=_context(), article_context=_context(),
            allowed_literature_citation_keys=[], direct_comparator_literature_keys=[],
            allowed_know_how_decisions=None, enforce_article_contract=True, llm=object(),
        )
    diagnostic = _safe_pipeline_typed_failure(caught.value)
    assert diagnostic["reason_code"] == "progressive_article_required_roles_missing"
    assert diagnostic["path"] == "article_analysis_contract.balance_positivity"
    assert caught.value.details["findings"] == [{
        "missing_roles": ["balance_positivity", "causal_protocol"],
        "role_owner_indices": {"balance_positivity": [], "causal_protocol": []},
        "repair_localization": "unlocated_full_materialization",
    }]
    assert caught.value.step_index == 0
