"""Accepted baseline obligations describe clinical rows, not primary results or lineage."""

from __future__ import annotations

import json

import pytest

from easyicu.research_agent.agents.progressive_planner import ProgressivePlannerAgent
from easyicu.research_agent.contracts.cohort_summary import is_primary_descriptive_summary_step
from easyicu.research_agent.planning.baseline_requirements import (
    baseline_outline_coverage,
    baseline_requirement_coverage,
    bind_baseline_requirements,
    candidate_baseline_requirements,
)
from easyicu.research_agent.planning.capability_registry import resolve_primary_capability
from easyicu.research_agent.planning.progressive_compiler import compile_progressive_plan
from easyicu.research_agent.planning.progressive_contract import ProgressivePlanCompileError
from easyicu.research_agent.providers.mocks import ScriptedMockLLMClient
from easyicu.research_agent.schema import AnalysisPlan, ConceptDescriptor, VariableRole
from tests.support.baseline_requirement_plans import _bound_context, _plan, _requirements

from tests.support.descriptive_plans import (
    descriptive_answer,
    descriptive_context,
    descriptive_skeleton,
)


def _grouped_replay():
    # Same owners/edges as a hosted descriptive plan: cohort, grouped primary,
    # continuous secondary, audit. Only synthetic column/domain metadata.
    context = descriptive_context(
        database="synthetic", variable="measurement",
        question="Describe the synthetic measurement distribution within each category.",
    )
    context.variable("measurement").role = VariableRole.OUTCOME
    context = context.model_copy(update={
        "cohort": context.cohort.model_copy(update={"provenance": {}}),
        "primary_exposure": "category",
        "variables": [*context.variables, ConceptDescriptor(
            name="category", role="other", dtype="str",
            observed_domain={"levels": ["alpha", "beta"], "n_unique": 2},
        )],
    })
    payload = descriptive_answer("measurement")
    for candidate in payload["design_selection"]["candidates"]:
        candidate["required_variables"] = ["measurement", "category"]
    for step in payload["steps"]:
        step["variable_names"] = ["stay_id", "category", "measurement"]
    payload["steps"].insert(2, {
        "step_id": "secondary", "module_id": "continuous_outcome_summary",
        "planned_analysis_role": "secondary", "variable_names": ["category", "measurement"],
        "objective": "Summarize the recorded measurement within each declared category.",
        "depends_on": ["cohort"],
    })
    return context, payload


def test_accepted_grouped_primary_replans_under_its_own_result_contract():
    context, payload = _grouped_replay()
    llm = ScriptedMockLLMClient([json.dumps(payload)])
    outline = ProgressivePlannerAgent(llm).run(context, stop_after_outline=True)
    skeleton = descriptive_skeleton(outline, grouping="category")
    skeleton = type(skeleton).model_validate({**skeleton.model_dump(), "display_labels": [
        {"key": "measurement", "value": "Measurement"},
        {"key": "category", "value": "Category"},
        {"key": "category=alpha", "value": "Alpha"},
        {"key": "category=beta", "value": "Beta"},
    ]})
    accepted, _ = compile_progressive_plan(skeleton=skeleton, context=context)
    required = candidate_baseline_requirements(
        plan=accepted.model_dump(mode="json"), source_plan_sha256="a" * 64,
        selected_concepts=["measurement", "category"],
        catalog_columns=[v.name for v in context.variables],
    )
    assert required is None
    rebound = bind_baseline_requirements(context, required)
    llm = ScriptedMockLLMClient([json.dumps(payload)])
    replanned = ProgressivePlannerAgent(llm).run(rebound, stop_after_outline=True)
    assert len(llm.calls) == 1
    assert baseline_outline_coverage(rebound, [s.model_dump() for s in replanned.steps])["status"] == "not_bound"
    replanned_skeleton = descriptive_skeleton(replanned, grouping="category")
    replanned_skeleton = replanned_skeleton.model_copy(update={"display_labels": skeleton.display_labels})
    result, _ = compile_progressive_plan(skeleton=replanned_skeleton, context=rebound)
    primary = next(s for s in result.steps if s.planned_analysis_role == "primary")
    assert is_primary_descriptive_summary_step(primary)
    assert primary.expected_outputs == ["table:cohort_summary", "table:distribution_summary"]
    assert resolve_primary_capability(analysis_type=result.analysis_type, plan=result).owner_claimed
    corrupted = result.model_copy(update={"steps": [
        s.model_copy(update={"scientific_action_id": None}) if s == primary else s
        for s in result.steps
    ]})
    assert not resolve_primary_capability(analysis_type=corrupted.analysis_type, plan=corrupted).owner_claimed


@pytest.mark.parametrize("role", ["id", "index"])
@pytest.mark.parametrize("source", [None, "lineage_concept"])
def test_lineage_roles_never_become_missing_roster_coordinates(role, source):
    context = _bound_context("age", "lineage")
    context = context.model_copy(update={"variables": [
        *context.variables, ConceptDescriptor(name="lineage", role=role, dtype="int64", source_concept=source),
    ]})
    for coverage in (
        baseline_requirement_coverage(context, _plan("age")),
        baseline_outline_coverage(context, [{
            "step_id": "baseline", "module_id": "table_one", "variable_names": ["exposure", "age"],
        }]),
    ):
        assert coverage["status"] == "complete"
        assert coverage["tables"][0]["missing_variables"] == []
        assert coverage["tables"][0]["unavailable_coordinates"] == []
        assert [row["required"] for row in coverage["tables"][0]["variables"]] == ["age"]


@pytest.mark.parametrize("owner", ["table_one", "auxiliary_summary"])
@pytest.mark.parametrize("clinical", [True, False])
def test_accepted_cohort_identity_is_excluded_even_when_extraction_renames_it(owner, clinical):
    names = ["age", "original_identity"] if clinical else ["original_identity"]
    plan = _plan(*names).model_dump(mode="json")
    if owner == "auxiliary_summary":
        plan["steps"][0].update(
            method="descriptive_cohort_summary", table_one_spec=None,
            inputs=["artifact:analysis_cohort", *names], expected_outputs=["table:cohort_summary"],
        )
    plan["steps"].insert(0, {
        "step_id": "cohort", "method": "cohort_definition_and_attrition",
        "intent": "Declare the accepted row identity.", "planned_analysis_role": "auxiliary",
        "inputs": ["original_identity"], "expected_outputs": ["artifact:analysis_cohort", "table:cohort_flow"],
        "cohort_definition_spec": {"identity_column": "original_identity", "analysis_set": "bound_universe"},
    })
    required = candidate_baseline_requirements(
        plan=plan, source_plan_sha256="b" * 64,
        selected_concepts=["age", "exposure"], catalog_columns=["age", "exposure", "original_identity"],
    )
    if clinical:
        assert [v.name for v in required.tables[0].variables] == ["age"]
    else:
        assert required is None


def test_a_real_table_one_roster_still_requires_every_clinical_row_after_extraction():
    context = _bound_context("age", "charlson")
    assert baseline_requirement_coverage(context, _plan("age", "cci_value"))["status"] == "complete"
    dropped = baseline_requirement_coverage(context, _plan("age"))
    assert dropped["status"] == "incomplete"
    assert dropped["tables"][0]["missing_variables"] == ["charlson"]
    assert dropped["tables"][0]["unavailable_coordinates"] == []


def test_a_closed_auxiliary_summary_keeps_the_schema_default_role():
    # Host/internal AnalysisStep construction defaults to auxiliary. The
    # accepted-reader must use that contract even for an older compact payload.
    plan = {"steps": [{
        "step_id": "baseline", "method": "descriptive_cohort_summary",
        "intent": "Describe the accepted clinical roster.",
        "inputs": ["artifact:analysis_cohort", "age"], "expected_outputs": ["table:cohort_summary"],
    }]}
    required = candidate_baseline_requirements(
        plan=plan, source_plan_sha256="c" * 64, selected_concepts=["age"], catalog_columns=["age"],
    )
    assert required is not None
    assert [v.name for v in required.tables[0].variables] == ["age"]


def test_a_primary_result_cannot_discharge_an_accepted_auxiliary_roster():
    context = _bound_context("age")
    payload = _requirements("age").model_dump(mode="json")
    payload["schema_version"] = "easyicu.accepted_baseline_requirements/2"
    payload["tables"][0]["group_by"] = None
    context.cohort.provenance["accepted_baseline_requirements"] = payload
    plan = AnalysisPlan(research_question=context.research_question, steps=[{
        "step_id": "primary", "method": "empirical_summary", "planned_analysis_role": "primary",
        "intent": "Describe the primary empirical distribution.",
        "scientific_action_id": "descriptive.descriptive_summary",
        "inputs": ["artifact:analysis_cohort", "age"],
        "expected_outputs": ["table:cohort_summary", "table:distribution_summary"],
    }])
    assert baseline_requirement_coverage(context, plan)["status"] == "incomplete"


def test_a_lineage_only_roster_has_no_clinical_obligation():
    context = _bound_context("stay_id")
    context.cohort.id_columns = ["stay_id"]
    coverage = baseline_outline_coverage(context, [])
    assert coverage["status"] == "complete"
    assert coverage["tables"] == []


def test_a_lineage_group_is_not_an_unavailable_clinical_coordinate():
    context = _bound_context("age")
    payload = _requirements("age").model_dump(mode="json")
    payload["tables"][0]["group_by"] = {"name": "row_index"}
    context.cohort.provenance["accepted_baseline_requirements"] = payload
    context.variables.append(ConceptDescriptor(name="row_index", role="index", dtype="int64"))
    coverage = baseline_requirement_coverage(context, _plan("age"))
    assert coverage["status"] == "complete"
    assert coverage["tables"][0]["group_by"]["required"] is None
    assert coverage["tables"][0]["unavailable_coordinates"] == []


def test_lineage_source_mapping_cannot_erase_a_clinical_row():
    context = _bound_context("age", "sequence")
    context.variables.extend([
        ConceptDescriptor(name="row_number", role="index", dtype="int64", source_concept="sequence"),
        ConceptDescriptor(name="lineage", role="id", dtype="int64", source_concept="age"),
    ])
    coverage = baseline_requirement_coverage(context, _plan("age"))
    assert coverage["status"] == "complete"
    assert [row["required"] for row in coverage["tables"][0]["variables"]] == ["age"]
    assert baseline_requirement_coverage(context, _plan("unrelated"))["tables"][0]["missing_variables"] == ["age"]


@pytest.mark.parametrize("strategy", ["progressive_v2", "family_spec_v1"])
@pytest.mark.parametrize("missing", ["variable", "group"])
@pytest.mark.parametrize("separate_article_context", [False, True])
def test_unsatisfiable_accepted_coordinate_stops_before_any_provider_call(strategy, missing, separate_article_context):
    context, _ = _grouped_replay()
    payload = _requirements("missing_measurement").model_dump(mode="json")
    table = payload["tables"][0]
    table["group_by"]["name"] = "category" if missing == "variable" else "missing_category"
    table["group_by"]["source_concept"] = None
    if missing == "group":
        table["variables"] = [{"name": "measurement"}]
    bound = bind_baseline_requirements(context, payload)
    llm = ScriptedMockLLMClient([])
    with pytest.raises(ProgressivePlanCompileError) as raised:
        ProgressivePlannerAgent(llm).run(
            context if separate_article_context else bound, planner_strategy=strategy,
            **({"article_contract_context": bound} if separate_article_context else {}),
        )
    assert raised.value.reason_code == "progressive_accepted_baseline_materialization_missing"
    coordinate = "missing_measurement" if missing == "variable" else "missing_category"
    assert raised.value.easyicu_safe_diagnostic["concepts"] == [coordinate]
    assert raised.value.step_id == table["source_step_id"]
    assert coordinate in str(raised.value)
    assert llm.calls == []
