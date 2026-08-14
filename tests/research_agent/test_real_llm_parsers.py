"""Parser-robustness tests for the real-LLM smoke path (T1.3).

Free-tier OpenRouter models (gemini-2.0-flash-exp, llama-3.x, etc.)
routinely:

* wrap JSON in a ```json fence with prose before/after,
* prefix Python code with "Sure! Here is the script:" and ```python,
* surround the manuscript markdown with ```markdown … ```,
* mention an opening brace inside a string literal that earlier
  parsers miscounted.

This module pins the agent helpers' tolerance for those quirks so a
regression in ``_strip_code_fence`` / ``_first_json_block`` shows up
without anyone having to spend a real LLM call.
"""

from __future__ import annotations

import importlib
import json
import sys
import types
from pathlib import Path

import pytest

from easyicu.research_agent.providers.mocks import ScriptedMockLLMClient


def _load_agents_helpers(ra):
    """Load parser helpers from their canonical implementation module."""
    return importlib.import_module(ra.__name__ + ".agents.core")


def test_strip_code_fence_handles_leading_prose(ra):
    helpers = _load_agents_helpers(ra)
    raw = (
        "Sure! Here is the analysis plan you asked for:\n\n"
        "```json\n"
        '{"research_question": "x", "steps": []}\n'
        "```\n\n"
        "Let me know if you need more steps."
    )
    out = helpers._strip_code_fence(raw)
    assert out.strip().startswith("{")
    assert "Sure!" not in out
    assert "Let me know" not in out


def test_strip_code_fence_python_block(ra):
    helpers = _load_agents_helpers(ra)
    raw = "Here you go:\n```python\nimport pandas as pd\ndf = pd.read_parquet('x')\n```"
    out = helpers._strip_code_fence(raw).strip()
    assert out.startswith("import pandas")


def test_strip_code_fence_markdown_block(ra):
    helpers = _load_agents_helpers(ra)
    raw = "Here is the manuscript:\n```markdown\n# Title\n\nBody.\n```\n"
    out = helpers._strip_code_fence(raw)
    assert "# Title" in out
    assert "Here is" not in out


def test_strip_code_fence_no_fence_passthrough(ra):
    helpers = _load_agents_helpers(ra)
    raw = '{"a": 1}'
    assert helpers._strip_code_fence(raw) == raw


def test_first_json_block_skips_braces_in_strings(ra):
    """Earlier parsers miscounted braces inside string literals; pin the fix."""
    helpers = _load_agents_helpers(ra)
    raw = (
        'Some prose before. {"intent": "the {evidence:foo} placeholder is required",'
        ' "steps": [{"step_id": "01"}]}'
    )
    block = helpers._first_json_block(raw)
    assert block is not None
    import json as _json

    parsed = _json.loads(block)
    assert parsed["steps"][0]["step_id"] == "01"


def test_planner_parse_recovers_fenced_json(ra):
    """End-to-end: PlannerAgent._parse must accept fenced JSON."""
    raw = (
        "Sure, here's the plan:\n```json\n"
        '{"research_question": "Is sofa2 -> death?", "steps":'
        ' [{"step_id":"01_table_one","planned_analysis_role":"auxiliary",'
        '"intent":"t1","inputs":[],"expected_outputs":[]}]}\n'
        "```"
    )
    schema = ra.schema
    ctx = schema.ResearchContext(
        research_question="Is sofa2 -> death?",
        cohort=schema.CohortDescriptor(
            cohort_name="c", database="d", n_patients=1, n_stays=1
        ),
        variables=[],
    )

    class _DummyLLM:
        name = "dummy"

        def complete(self, messages, **kwargs):
            return raw

    from easyicu.research_agent.agents.core import PlannerAgent

    plan = PlannerAgent(_DummyLLM())._parse(raw, ctx)
    assert plan.steps and plan.steps[0].step_id == "01_table_one"


def test_fresh_planner_measurement_audit_requires_typed_product_authority(ra):
    raw = json.dumps(
        {
            "research_question": "Audit measurement availability.",
            "analysis_type": "descriptive_epidemiology",
            "steps": [
                {
                    "step_id": "01_measurement_audit",
                    "planned_analysis_role": "auxiliary",
                    "intent": "Report count-only measurement missingness.",
                    "inputs": [],
                    "expected_outputs": ["table:missingness_data_quality"],
                    "method": "measurement_audit",
                }
            ],
        }
    )
    schema = ra.schema
    ctx = schema.ResearchContext(
        research_question="Audit measurement availability.",
        cohort=schema.CohortDescriptor(
            cohort_name="c",
            database="d",
            n_patients=1,
            n_stays=1,
        ),
        variables=[],
    )

    class _ProviderLikeLLM:
        name = "provider-like"

        def complete(self, messages, **kwargs):
            return raw

    from easyicu.research_agent.agents.core import PlannerAgent

    with pytest.raises(ValueError, match="measurement_audit_spec.products"):
        PlannerAgent(_ProviderLikeLLM())._parse(raw, ctx)


def test_planner_parse_preserves_declared_display_labels(ra):
    raw = (
        '{"research_question":"Estimate an association.",'
        '"display_labels":{"death":"In-hospital mortality",'
        '"primary":"Primary analysis"},'
        '"steps":[{"step_id":"01_model",'
        '"planned_analysis_role":"primary","intent":"fit",'
        '"inputs":[],"expected_outputs":["statistic:adjusted_effect"]}]}'
    )
    schema = ra.schema
    ctx = schema.ResearchContext(
        research_question="Estimate an association.",
        cohort=schema.CohortDescriptor(
            cohort_name="c", database="d", n_patients=1, n_stays=1
        ),
        variables=[],
    )

    class _DummyLLM:
        name = "dummy"

        def complete(self, messages, **kwargs):
            return raw

    from easyicu.research_agent.agents.core import PlannerAgent

    plan = PlannerAgent(_DummyLLM())._parse(raw, ctx)

    assert plan.display_labels == {
        "death": "In-hospital mortality",
        "primary": "Primary analysis",
    }


def test_planner_parse_drops_extra_step_fields(ra):
    raw = (
        '{"research_question": "Is sofa2 -> death?", "extra": "drop me", "steps":'
        ' [{"step_id":"06_cross_database","planned_analysis_role":"auxiliary",'
        '"intent":"protocol","inputs":[], '
        '"expected_outputs":[],"note":"external cohort unavailable"}]}'
    )
    schema = ra.schema
    ctx = schema.ResearchContext(
        research_question="Is sofa2 -> death?",
        cohort=schema.CohortDescriptor(
            cohort_name="c", database="d", n_patients=1, n_stays=1
        ),
        variables=[],
    )

    class _DummyLLM:
        name = "dummy"

        def complete(self, messages, **kwargs):
            return raw

    from easyicu.research_agent.agents.core import PlannerAgent

    plan = PlannerAgent(_DummyLLM())._parse(raw, ctx)
    assert plan.steps[0].step_id == "06_cross_database"
    assert not hasattr(plan.steps[0], "note")


def test_planner_uses_enough_completion_budget(ra):
    """Reasoning models can spend part of max_tokens before final JSON."""
    schema = ra.schema
    ctx = schema.ResearchContext(
        research_question="Is sofa2 -> death?",
        cohort=schema.CohortDescriptor(
            cohort_name="c", database="d", n_patients=1, n_stays=1
        ),
        variables=[],
    )

    from easyicu.research_agent.agents.core import PlannerAgent

    llm = ScriptedMockLLMClient(
        [
            '{"research_question": "Is sofa2 -> death?", "steps":'
            ' [{"step_id":"01_table_one","planned_analysis_role":"auxiliary",'
            '"intent":"t1","inputs":[],"expected_outputs":[]}]}'
        ]
    )
    PlannerAgent(llm).run(ctx)
    assert llm.calls[0][1]["max_tokens"] >= 8192


def test_planner_retries_primary_cohort_step_with_side_output(ra):
    """Raw-universe cohort ownership must be repaired before probe execution."""

    schema = ra.schema
    ctx = schema.ResearchContext(
        research_question="Describe a closed primary cohort.",
        cohort=schema.CohortDescriptor(
            cohort_name="c", database="d", n_patients=1, n_stays=1
        ),
        variables=[],
    )
    invalid = json.dumps(
        {
            "research_question": ctx.research_question,
            "analysis_type": "descriptive_epidemiology",
            "steps": [
                {
                    "step_id": "01_cohort",
                    "planned_analysis_role": "auxiliary",
                    "intent": "Materialize the cohort and report attrition.",
                    "inputs": [],
                    "expected_outputs": [
                        "artifact:analysis_cohort",
                        "table:cohort_flow",
                        "table:cohort_summary",
                    ],
                    "method": "cohort_definition_and_attrition",
                }
            ],
        }
    )
    repaired = json.dumps(
        {
            "research_question": ctx.research_question,
            "analysis_type": "descriptive_epidemiology",
            "steps": [
                {
                    "step_id": "01_cohort",
                    "planned_analysis_role": "auxiliary",
                    "intent": "Materialize the cohort and report attrition.",
                    "inputs": [],
                    "expected_outputs": [
                        "artifact:analysis_cohort",
                        "table:cohort_flow",
                    ],
                    "method": "cohort_definition_and_attrition",
                },
                {
                    "step_id": "02_summary",
                    "planned_analysis_role": "auxiliary",
                    "intent": "Describe the closed analysis cohort.",
                    "inputs": ["artifact:analysis_cohort"],
                    "expected_outputs": ["table:cohort_summary"],
                    "method": "descriptive",
                },
            ],
        }
    )
    llm = ScriptedMockLLMClient([invalid, repaired])

    from easyicu.research_agent.agents.core import PlannerAgent

    plan = PlannerAgent(llm).run(ctx)

    assert len(llm.calls) == 2
    assert plan.steps[0].expected_outputs == [
        "artifact:analysis_cohort",
        "table:cohort_flow",
    ]
    assert plan.steps[1].inputs == ["artifact:analysis_cohort"]
    assert "strict execution boundary" in llm.calls[0][0][-1].content
    feedback = llm.calls[1][0][-1].content
    assert "primary-cohort output contract is not executable" in feedback
    assert "table:cohort_summary" in feedback
    assert "downstream steps" in feedback


def test_planner_retry_projection_keeps_structure_and_bounds_prose() -> None:
    from easyicu.research_agent.agents.core import (
        _PLANNER_RETRY_PROJECTION_BYTE_LIMIT,
        _planner_retry_response_projection,
    )

    raw = json.dumps(
        {
            "analysis_type": "association",
            "rationale": "long prose " * 5_000,
            "steps": [
                {
                    "step_id": "01_primary",
                    "planned_analysis_role": "primary",
                    "intent": "long intent " * 2_000,
                    "inputs": ["exposure_max", "death"],
                    "expected_outputs": ["table:adjusted_association_estimates"],
                    "method": "adjusted_association_models",
                    "sensitivity_spec_ids": ["timing_landmark_24h"],
                    "model_requirements": [
                        {
                            "requirement_id": "primary",
                            "outcome": "death",
                            "exposure_source": "exposure_max",
                        }
                    ],
                    "exposure_outcome_distribution_spec": {
                        "schema_version": "easyicu.exposure_outcome_distribution/2",
                        "exposure": "exposure_max",
                        "exposure_levels": [0, 1],
                        "outcome": "death",
                        "outcome_levels": [0, 1],
                        "outcome_positive_value": 1,
                        "level_match_policy": "exact_typed",
                        "denominator_policy": "all_declared_rows",
                        "missing_outcome_policy": "structural_absence_is_non_event",
                        "risk_difference_contrast": {
                            "reference_exposure_level": 0,
                            "comparison_exposure_level": 1,
                        },
                        "confidence_level": 0.95,
                    },
                    "descriptive_claim": {
                        "claim_ceiling": "descriptive_only",
                        "unresolved_limitations": [
                            "post_baseline_exposure_opportunity_unresolved"
                        ],
                    },
                }
            ],
            "robustness_specs": [
                {
                    "spec_id": "complete_case",
                    "axis": "missing",
                    "missing_override": {"strategy": "complete_case"},
                }
            ],
            "evalue_conversion_spec": {
                "baseline_risk_evidence_id": "outcome_rate",
                "baseline_risk_column": "event_rate",
                "population_column": "population",
                "baseline_population": "unexposed",
            },
            "subgroup_analysis_spec": {
                "predictor": "exposure_max",
                "outcome": "death",
                "subgroup_columns": ["sex"],
                "multiplicity_family_id": "secondary:subgroups",
            },
        }
    )

    projected = _planner_retry_response_projection(raw)
    payload = json.loads(projected)

    assert len(projected.encode("utf-8")) <= _PLANNER_RETRY_PROJECTION_BYTE_LIMIT
    assert payload["steps"][0]["inputs"] == ["exposure_max", "death"]
    assert payload["steps"][0]["sensitivity_spec_ids"] == ["timing_landmark_24h"]
    assert payload["steps"][0]["model_requirements"][0]["outcome"] == "death"
    assert (
        payload["steps"][0]["exposure_outcome_distribution_spec"][
            "risk_difference_contrast"
        ]["comparison_exposure_level"]
        == 1
    )
    assert payload["steps"][0]["descriptive_claim"]["claim_ceiling"] == (
        "descriptive_only"
    )
    assert payload["robustness_specs"][0]["spec_id"] == "complete_case"
    assert payload["evalue_conversion_spec"]["baseline_population"] == "unexposed"
    assert payload["subgroup_analysis_spec"]["subgroup_columns"] == ["sex"]
    assert "rationale" not in payload
    assert "intent" not in payload["steps"][0]


def test_planner_retries_dictionary_concept_absent_from_sealed_typed_input(
    tmp_path,
):
    """A legal global concept is not executable authority for every cohort."""

    from easyicu.research_agent.agents.core import PlannerAgent
    from tests.research_agent.test_materialized_column_metadata import (
        _build_v2_context,
    )

    context = _build_v2_context(tmp_path)

    def response(concept_id: str) -> str:
        return (
            '{"research_question":"Describe the sealed cohort.",'
            '"analysis_type":"descriptive_epidemiology",'
            '"cohort":{"name":"primary","inclusion":[{'
            f'"concept_id":"{concept_id}",'
            '"time_window":{"anchor":"icu_admission",'
            '"start_offset_hours":0,"end_offset_hours":24},'
            '"aggregation":"max","op":"not_missing","value":null}],'
            '"exclusion":[]},'
            '"steps":[{"step_id":"01_define_cohort",'
            '"planned_analysis_role":"auxiliary",'
            '"intent":"Materialize the declared analysis cohort.",'
            '"inputs":["lact_max"],'
            '"expected_outputs":["artifact:analysis_cohort"],'
            '"method":"cohort_definition"}]}'
        )

    llm = ScriptedMockLLMClient([response("hr"), response("lact")])
    plan = PlannerAgent(llm).run(context)

    assert len(llm.calls) == 2
    assert plan.cohort is not None
    assert plan.cohort.inclusion[0].concept_id == "lact"
    feedback = llm.calls[1][0][-1].content
    assert "not executable against this sealed input" in feedback
    assert "lact_max" in feedback


def test_cohort_concept_allowlist_includes_sofa2_overlay() -> None:
    from easyicu.research_agent.planning.cohort_contract import known_concept_ids

    assert {"sofa2", "sep3_sofa2", "sofa2_resp"} <= known_concept_ids()


def test_planner_retries_non_column_step_and_model_references(tmp_path) -> None:
    """Semantic labels must not survive as executable typed dataframe fields."""

    from easyicu.research_agent.agents.core import PlannerAgent
    from tests.research_agent.test_materialized_column_metadata import (
        _build_v2_context,
    )

    context = _build_v2_context(tmp_path)

    def response(exposure: str, outcome: str) -> str:
        return (
            '{"research_question":"Estimate the sealed association.",'
            '"analysis_type":"association_study",'
            '"steps":[{"step_id":"01_model",'
            '"planned_analysis_role":"primary",'
            '"intent":"Fit the declared adjusted association.",'
            f'"inputs":["{exposure}","{outcome}"],'
            '"expected_outputs":["table:adjusted_association_estimates"],'
            '"method":"adjusted_association_models",'
            '"model_requirements":[{'
            '"requirement_id":"primary_sealed_association",'
            f'"outcome":"{outcome}","outcome_type":"binary",'
            '"method_family":"logistic_regression",'
            f'"exposure_source":"{exposure}",'
            f'"covariates":[],"model_terms":[{{"name":"{exposure}",'
            '"role":"exposure","coding":"continuous",'
            '"transform":"identity"}],'
            '"analysis_role":"primary","analysis_set":"complete_case",'
            '"required_for_step_success":true}]}]}'
        )

    llm = ScriptedMockLLMClient(
        [response("lactate", "mortality"), response("lact_max", "death")]
    )
    plan = PlannerAgent(llm).run(context)

    assert len(llm.calls) == 2
    assert plan.steps[0].inputs == ["lact_max", "death"]
    feedback = llm.calls[1][0][-1].content
    assert "typed plan references are not executable" in feedback
    assert "raw name 'lactate'" in feedback
    assert "'step inputs': 1" in feedback
    assert "'model exposures': 1" in feedback
    assert "raw name 'mortality'" in feedback
    assert "'model outcomes': 1" in feedback


def test_typed_binding_gate_covers_robustness_fields(tmp_path) -> None:
    from types import SimpleNamespace

    from easyicu.research_agent.cohort.schema import (
        CohortSchemaError,
        validate_plan_typed_bindings_against_context,
    )
    from tests.research_agent.test_materialized_column_metadata import (
        _build_v2_context,
    )

    context = _build_v2_context(tmp_path)
    plan = SimpleNamespace(
        cohort=None,
        steps=[],
        robustness_specs=[
            SimpleNamespace(
                spec_id="missing_aliases",
                cohort_override=None,
                missing_override={
                    "strategy": "complete_case",
                    "variables": ["lactate", "death"],
                    "audit_flags": ["measurement_status"],
                },
                outcome_override=None,
            ),
            SimpleNamespace(
                spec_id="outcome_alias",
                cohort_override=None,
                missing_override=None,
                outcome_override={
                    "concept_id": "mortality",
                    "time_column": "mortality_time",
                },
            ),
        ],
    )

    with pytest.raises(CohortSchemaError) as caught:
        validate_plan_typed_bindings_against_context(plan=plan, context=context)

    message = str(caught.value)
    assert "raw name 'lactate'" in message
    assert "'robustness missing variables': 1" in message
    assert "raw name 'measurement_status'" in message
    assert "'robustness audit flags': 1" in message
    assert "raw name 'mortality'" in message
    assert "raw name 'mortality_time'" in message
    assert message.count("'robustness outcome fields': 1") == 2


def test_typed_binding_gate_rejects_direct_suffix_with_wrong_window(tmp_path) -> None:
    from types import SimpleNamespace

    from easyicu.research_agent.cohort.schema import (
        CohortSchemaError,
        validate_plan_typed_bindings_against_context,
    )
    from easyicu.research_agent.planning.cohort_contract import (
        CohortDefinition,
        ConceptPredicate,
        TimeWindow,
    )
    from tests.research_agent.test_materialized_column_metadata import (
        _build_v2_context,
    )

    context = _build_v2_context(tmp_path)
    producer = SimpleNamespace(
        step_id="01_cohort",
        inputs=["lact_max", "death"],
        expected_outputs=["artifact:analysis_cohort"],
        method="cohort_definition",
    )
    six_hour = CohortDefinition(
        name="six_hour",
        inclusion=[
            ConceptPredicate(
                concept_id="lact",
                time_window=TimeWindow(
                    anchor="icu_admission",
                    start_offset_hours=0,
                    end_offset_hours=6,
                ),
                aggregation="max",
                op=">=",
                value=0,
            )
        ],
    )
    plan = SimpleNamespace(
        cohort=None,
        steps=[producer],
        robustness_specs=[
            SimpleNamespace(
                spec_id="six_hour",
                cohort_override=six_hour,
                missing_override=None,
                outcome_override=None,
            )
        ],
    )

    with pytest.raises(CohortSchemaError, match="proven matching aggregation and time"):
        validate_plan_typed_bindings_against_context(plan=plan, context=context)


def test_typed_binding_gate_accepts_direct_static_column_without_window(
    tmp_path,
) -> None:
    from types import SimpleNamespace

    from easyicu.research_agent.cohort.schema import (
        validate_plan_typed_bindings_against_context,
    )
    from easyicu.research_agent.planning.cohort_contract import (
        CohortDefinition,
        ConceptPredicate,
        TimeWindow,
    )
    from easyicu.research_agent.schema import AnalysisStep
    from tests.research_agent.test_materialized_column_metadata import (
        _build_v2_context,
    )

    context = _build_v2_context(tmp_path)
    producer = AnalysisStep(
        step_id="01_cohort",
        planned_analysis_role="auxiliary",
        intent="Materialize the adult analysis cohort and its flow.",
        inputs=["age", "death"],
        expected_outputs=["cohort:analysis_cohort", "table:cohort_flow"],
        method="cohort_definition",
    )
    adult_cohort = CohortDefinition(
        name="adult_cohort",
        inclusion=[
            ConceptPredicate(
                concept_id="age",
                time_window=TimeWindow(
                    anchor="icu_admission",
                    start_offset_hours=0,
                    end_offset_hours=24,
                ),
                aggregation="first",
                op=">=",
                value=18,
            )
        ],
    )
    plan = SimpleNamespace(
        cohort=adult_cohort,
        steps=[producer],
        robustness_specs=[],
    )

    validate_plan_typed_bindings_against_context(plan=plan, context=context)


def test_typed_binding_gate_rejects_identity_coordinate_as_raw_step_input(
    tmp_path,
) -> None:
    from types import SimpleNamespace

    from easyicu.research_agent.cohort.schema import (
        CohortSchemaError,
        validate_plan_typed_bindings_against_context,
    )
    from easyicu.research_agent.schema import AnalysisStep
    from tests.research_agent.test_materialized_column_metadata import (
        _build_v2_context,
    )

    context = _build_v2_context(tmp_path)
    plan = SimpleNamespace(
        cohort=None,
        steps=[
            AnalysisStep(
                step_id="01_invalid_coordinate",
                intent="Use a sealed coordinate as an analysis variable.",
                inputs=["stay_id", "death"],
            )
        ],
        robustness_specs=[],
    )

    with pytest.raises(CohortSchemaError, match="raw name 'stay_id'") as caught:
        validate_plan_typed_bindings_against_context(plan=plan, context=context)

    message = str(caught.value)
    assert "reserved for host navigation" in message
    assert "reserved navigation coordinates=['stay_id']" in message
    assert (
        "'stay_id'"
        not in message.split("executable cohort columns=", 1)[1].split(
            "; reserved navigation coordinates=", 1
        )[0]
    )

    from easyicu.research_agent.agents.core import PlannerAgent

    prompt = PlannerAgent.request_messages(context)[1].content
    assert "HOST NAVIGATION COORDINATES" in prompt
    assert "not executable analysis fields" in prompt


def test_planner_retries_robustness_window_absent_from_sealed_input(tmp_path) -> None:
    """A plausible column suffix cannot authorize a different scientific window."""

    import json

    from easyicu.research_agent.agents.core import PlannerAgent
    from tests.research_agent.test_materialized_column_metadata import (
        _build_v2_context,
    )

    context = _build_v2_context(tmp_path)
    primary_step = {
        "step_id": "01_model",
        "planned_analysis_role": "primary",
        "intent": "Fit the declared adjusted association.",
        "inputs": ["lact_max", "death"],
        "expected_outputs": ["table:adjusted_association_estimates"],
        "method": "adjusted_association_models",
        "model_requirements": [
            {
                "requirement_id": "primary_sealed_association",
                "outcome": "death",
                "outcome_type": "binary",
                "method_family": "logistic_regression",
                "exposure_source": "lact_max",
                "covariates": [],
                "model_terms": [
                    {
                        "name": "lact_max",
                        "role": "exposure",
                        "coding": "continuous",
                        "transform": "identity",
                    }
                ],
                "analysis_role": "primary",
                "analysis_set": "complete_case",
                "required_for_step_success": True,
            }
        ],
    }
    unsupported_window = {
        "research_question": "Estimate the sealed association.",
        "analysis_type": "association_study",
        "steps": [primary_step],
        "robustness_specs": [
            {
                "spec_id": "six_hour_lactate",
                "axis": "cohort",
                "description": "Restrict by a six-hour maximum.",
                "cohort_override": {
                    "name": "six_hour_lactate",
                    "inclusion": [
                        {
                            "concept_id": "lact",
                            "time_window": {
                                "anchor": "icu_admission",
                                "start_offset_hours": 0,
                                "end_offset_hours": 6,
                            },
                            "aggregation": "max",
                            "op": ">=",
                            "value": 0,
                        }
                    ],
                    "exclusion": [],
                },
            }
        ],
    }
    supported_missingness = {
        **unsupported_window,
        "robustness_specs": [
            {
                "spec_id": "complete_case",
                "axis": "missing",
                "description": "Restrict the model to complete cases.",
                "missing_override": {
                    "strategy": "complete_case",
                    "variables": ["lact_max", "death"],
                },
            }
        ],
    }
    llm = ScriptedMockLLMClient(
        [json.dumps(unsupported_window), json.dumps(supported_missingness)]
    )

    plan = PlannerAgent(llm).run(context)

    assert len(llm.calls) == 2
    assert plan.robustness_specs[0].spec_id == "complete_case"
    feedback = llm.calls[1][0][-1].content
    assert "proven matching aggregation and time" in feedback
    assert "icu_admission[0.0,6.0]h/max" in feedback
    assert "icu_admission[0,24]h" in feedback


def test_planner_retries_primary_cohort_that_erases_its_closed_comparison(
    tmp_path,
) -> None:
    """Eligibility cannot erase a Planner-declared downstream contrast."""

    from easyicu.research_agent.agents.core import PlannerAgent
    from tests.research_agent.test_materialized_column_metadata import (
        _build_v2_context,
    )

    context = _build_v2_context(tmp_path)

    def response(cohort_op: str, cohort_value: str) -> str:
        return (
            '{"research_question":"Compare a closed exposure in the sealed cohort.",'
            '"analysis_type":"association_study",'
            '"cohort":{"name":"primary","inclusion":[{'
            '"concept_id":"mech_vent",'
            '"time_window":{"anchor":"icu_admission",'
            '"start_offset_hours":0,"end_offset_hours":24},'
            f'"aggregation":"max","op":"{cohort_op}","value":{cohort_value}'
            '}],"exclusion":[]},'
            '"steps":[{"step_id":"01_define_cohort",'
            '"planned_analysis_role":"auxiliary",'
            '"intent":"Materialize the declared analysis cohort.",'
            '"inputs":["mech_vent_max"],'
            '"expected_outputs":["artifact:analysis_cohort"],'
            '"method":"cohort_definition"},'
            '{"step_id":"02_table_one",'
            '"planned_analysis_role":"auxiliary",'
            '"intent":"Describe both closed exposure groups.",'
            '"inputs":["artifact:analysis_cohort","mech_vent_max","age"],'
            '"expected_outputs":["table:table_one"],'
            '"method":"descriptive",'
            '"table_one_spec":{"group_by":"mech_vent_max",'
            '"group_levels":["__easyicu_level_1__","__easyicu_level_2__"],'
            '"variables":[{"name":"age","variable_kind":"continuous",'
            '"summary":"median_iqr","test":"mann_whitney_or_kruskal",'
            '"levels":[]}],"include_overall":true,'
            '"missing_group_policy":"fail_closed",'
            '"missingness_display":"n_percent_by_group",'
            '"p_values_required":true,'
            '"p_value_adjustment":"none_descriptive_table"}},'
            '{"step_id":"03_primary_association",'
            '"planned_analysis_role":"primary",'
            '"intent":"Fit the required primary adjusted association.",'
            '"inputs":["artifact:analysis_cohort","lact_max","death"],'
            '"expected_outputs":["table:adjusted_association_estimates"],'
            '"method":"adjusted_association_models",'
            '"model_requirements":[{'
            '"requirement_id":"primary_adjusted",'
            '"outcome":"death","outcome_type":"binary",'
            '"method_family":"logistic_regression",'
            '"exposure_source":"lact_max",'
            '"covariates":[],"model_terms":[{'
            '"name":"lact_max","role":"exposure",'
            '"coding":"continuous","transform":"identity"}],'
            '"analysis_role":"primary","analysis_set":"source_aware",'
            '"required_for_step_success":true}]}]}'
        )

    llm = ScriptedMockLLMClient(
        [
            response(">=", "1"),
            response("not_missing", "null"),
        ]
    )
    plan = PlannerAgent(llm).run(context)

    assert len(llm.calls) == 2
    assert plan.cohort is not None
    assert plan.cohort.inclusion[0].op == "not_missing"
    feedback = llm.calls[1][0][-1].content
    assert "collapse a downstream closed comparison" in feedback
    assert "below two retained levels" in feedback
    assert "retained_values" not in feedback


def test_openai_client_passes_provider_extra_body(ra, monkeypatch):
    """OpenRouter reasoning controls must reach the SDK request."""
    calls = {}

    class _FakeCompletions:
        def create(self, **kwargs):
            calls["create"] = kwargs
            message = types.SimpleNamespace(content="ok")
            choice = types.SimpleNamespace(message=message, finish_reason="stop")
            usage = types.SimpleNamespace(
                prompt_tokens=1,
                completion_tokens=1,
                total_tokens=2,
            )
            return types.SimpleNamespace(choices=[choice], usage=usage)

    class _FakeOpenAI:
        def __init__(self, **kwargs):
            calls["client"] = kwargs
            self.chat = types.SimpleNamespace(
                completions=_FakeCompletions(),
            )

    monkeypatch.setitem(
        sys.modules, "openai", types.SimpleNamespace(OpenAI=_FakeOpenAI)
    )

    from easyicu.research_agent.providers.factory import authorize_provider_client
    from easyicu.research_agent.providers.llm import LLMMessage, OpenAIClient

    extra_body = {"reasoning": {"effort": "none", "exclude": True}}
    client = OpenAIClient(
        model="z-ai/glm-4.5-air:free",
        api_key="test-key",
        base_url="https://openrouter.ai/api/v1",
        extra_body=extra_body,
    )
    authorize_provider_client(
        client,
        provider="openai",
        model="z-ai/glm-4.5-air:free",
        base_url="https://openrouter.ai/api/v1",
        destination="external",
        environment={"EASYICU_ALLOW_EXTERNAL_LLM": "1"},
    )
    assert client.complete([LLMMessage(role="user", content="hi")]) == "ok"
    assert calls["create"]["extra_body"] == extra_body


def test_writer_strips_markdown_fence(ra, tmp_path: Path):
    """If the LLM wraps the manuscript in ```markdown, the binder must
    still see raw markdown so it can locate ``{evidence:*}``."""
    raw = "```markdown\n# Title\n\nCohort: {evidence:table_one}.\n```"

    from easyicu.research_agent.agents.core import WriterAgent

    schema = ra.schema
    ctx = schema.ResearchContext(
        research_question="x",
        cohort=schema.CohortDescriptor(
            cohort_name="c", database="d", n_patients=1, n_stays=1
        ),
        variables=[],
    )
    out = WriterAgent(ScriptedMockLLMClient([raw], repeat_last=True)).run(
        context=ctx, evidence_ids=["table_one"]
    )
    # The fence must be stripped so the binder regex matches.
    assert "{evidence:table_one}" in out
    assert "```markdown" not in out


def test_writer_language_prompt_preserves_evidence_ids(ra):
    """The Chinese writer mode should ask for zh prose but keep evidence ids ASCII."""
    from easyicu.research_agent.agents.core import WriterAgent

    schema = ra.schema
    ctx = schema.ResearchContext(
        research_question="x",
        cohort=schema.CohortDescriptor(
            cohort_name="c", database="d", n_patients=1, n_stays=1
        ),
        variables=[],
    )

    llm = ScriptedMockLLMClient(
        ["# 标题\n\n结果：12 例 {evidence:table_one}。\n"],
        repeat_last=True,
    )
    out = WriterAgent(llm, language="zh").run(
        context=ctx,
        evidence_ids=["table_one"],
    )

    prompts = "\n".join(
        message.content for messages, _kwargs in llm.calls for message in messages
    )
    assert "Simplified Chinese" in prompts
    assert "do not translate evidence ids" in prompts
    assert "{evidence:table_one}" in out


def test_writer_prompt_discourages_tbd_and_manifest_narration(ra):
    # The writer contract (writer.txt → _WRITER_GUIDE) lands in the
    # *system* message of every per-section LLM call. Capture the full
    # joined prompt across every section so we can assert on contract
    # text regardless of which section was last called.
    from easyicu.research_agent.agents.core import WriterAgent

    schema = ra.schema
    ctx = schema.ResearchContext(
        research_question="x",
        cohort=schema.CohortDescriptor(
            cohort_name="c", database="d", n_patients=1, n_stays=1
        ),
        variables=[],
    )

    llm = ScriptedMockLLMClient(
        [
            "# Title\n\n## Results\n\nBaseline characteristics are summarised "
            "in Table 1 {evidence:table_one}.\n"
        ],
        repeat_last=True,
    )
    out = WriterAgent(llm).run(context=ctx, evidence_ids=["table_one"])

    captured = {"system": "", "user": ""}
    for messages, _kwargs in llm.calls:
        for message in messages:
            captured[message.role] += message.content + "\n"

    # Writer contract assertions land in the system prompt.
    assert "`[TBD]`" in captured["system"]
    assert "warning: see manifest" in captured["system"]
    assert (
        "Only cite `table_one`, `outcome_rate`, or `primary_association`"
        in captured["system"]
    )
    # Writer contract should reference `model_performance` as a fallback
    # baseline source for prediction tasks. Exact wording has shifted; we
    # assert on the alias token rather than a specific sentence.
    assert "`model_performance`" in captured["system"]
    assert "Use exactly single braces" in captured["user"]
    assert "SCIENTIFIC CLAIM RULE" in captured["user"]
    assert "complete standalone sentence" in captured["user"]
    assert "Run-bound typed methodology applications" in captured["user"]
    assert "mechanisms, strengths, or limitations" in captured["user"]
    assert "must either be one exact host-authorized claim token" in captured["user"]
    assert "TBD by author" not in captured["user"]
    assert "Funding information was not available" in captured["user"]
    # The dummy LLM's stock response should land in the bound output.
    assert "{evidence:table_one}" in out


def test_writer_evidence_repair_returns_cite_or_drop_decisions():
    from easyicu.research_agent.reporting.writer_evidence_repair import (
        decide_writer_evidence_repairs,
    )

    raw = json.dumps(
        {
            "decisions": [
                {
                    "index": 0,
                    "action": "cite",
                    "evidence_ids": ["literature_prisma"],
                },
                {"index": 1, "action": "drop", "evidence_ids": []},
            ]
        }
    )
    decisions = decide_writer_evidence_repairs(
        ScriptedMockLLMClient([raw], repeat_last=True),
        evidence_ids=["literature_prisma", "primary_association"],
        evidence_digest="literature_prisma: background evidence",
        missing_sentences=[
            "Sepsis is clinically important.",
            "No estimate was available for reporting.",
        ],
    )

    assert decisions == [
        {
            "index": 0,
            "action": "cite",
            "evidence_ids": ["literature_prisma"],
        },
        {"index": 1, "action": "drop", "evidence_ids": []},
    ]


def test_writer_evidence_repair_rejects_unregistered_evidence_id():
    from easyicu.research_agent.providers.structured_retry import (
        StructuredResponseFailure,
    )
    from easyicu.research_agent.reporting.writer_evidence_repair import (
        decide_writer_evidence_repairs,
    )

    raw = json.dumps(
        {
            "decisions": [
                {
                    "index": 0,
                    "action": "cite",
                    "evidence_ids": ["invented_id"],
                }
            ]
        }
    )

    with pytest.raises(StructuredResponseFailure):
        decide_writer_evidence_repairs(
            ScriptedMockLLMClient([raw], repeat_last=True),
            evidence_ids=["literature_prisma"],
            evidence_digest="literature_prisma: background evidence",
            missing_sentences=["Sepsis is clinically important."],
        )


def test_writer_evidence_repair_can_select_only_an_exact_host_claim():
    from easyicu.research_agent.reporting.writer_evidence_repair import (
        decide_writer_evidence_repairs,
    )

    sentence = "The exposed group had higher observed mortality."
    claim_ref = "03_primary.observed_risk_difference"
    raw = json.dumps(
        {
            "decisions": [
                {
                    "index": 0,
                    "action": "claim",
                    "evidence_ids": [],
                    "claim_ref": claim_ref,
                }
            ]
        }
    )

    decisions = decide_writer_evidence_repairs(
        ScriptedMockLLMClient([raw], repeat_last=True),
        evidence_ids=["primary_summary"],
        evidence_digest="primary summary",
        missing_sentences=[sentence],
        scientific_claims={claim_ref: "Host-rendered descriptive claim."},
        claim_required_sentences=[sentence],
    )

    assert decisions == [
        {
            "index": 0,
            "action": "claim",
            "evidence_ids": [],
            "claim_ref": claim_ref,
        }
    ]


def test_writer_evidence_repair_cannot_cite_around_claim_authority():
    from easyicu.research_agent.providers.structured_retry import (
        StructuredResponseFailure,
    )
    from easyicu.research_agent.reporting.writer_evidence_repair import (
        decide_writer_evidence_repairs,
    )

    sentence = "The exposed group had higher observed mortality."
    raw = json.dumps(
        {
            "decisions": [
                {
                    "index": 0,
                    "action": "cite",
                    "evidence_ids": ["primary_summary"],
                }
            ]
        }
    )

    with pytest.raises(StructuredResponseFailure):
        decide_writer_evidence_repairs(
            ScriptedMockLLMClient([raw], repeat_last=True),
            evidence_ids=["primary_summary"],
            evidence_digest="primary summary",
            missing_sentences=[sentence],
            scientific_claims={
                "03_primary.observed_risk_difference": "Host claim."
            },
            claim_required_sentences=[sentence],
        )


def test_openrouter_reasoning_extra_body_skips_gpt_oss(ra):
    from easyicu.research_agent.providers.llm import openrouter_reasoning_extra_body

    assert openrouter_reasoning_extra_body("openai/gpt-oss-120b:free") is None
    assert openrouter_reasoning_extra_body("z-ai/glm-4.5-air:free") == {
        "reasoning": {"effort": "none", "exclude": True}
    }
