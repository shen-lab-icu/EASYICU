"""Accepted baseline content survives a fresh package-bound Planner pass."""

from __future__ import annotations

import pytest

from easyicu.research_agent.orchestration.config import PipelineConfig
from easyicu.research_agent.planning.baseline_requirements import (
    baseline_requirement_coverage,
    bind_baseline_requirements,
    candidate_baseline_requirements,
)
from easyicu.research_agent.planning.scientific_review import (
    build_plan_scientific_review,
    render_plan_scientific_guardrails,
)
from easyicu.research_agent.schema import AnalysisPlan, ConceptDescriptor

from .scientific_review_fixtures import _context, _traditional_table_one_step


def _plan(*names: str, group: str = "exposure") -> AnalysisPlan:
    step = _traditional_table_one_step().model_dump(mode="json")
    step["step_id"] = "renamed_baseline"
    step["inputs"] = ["cohort:analysis_set", group, *names]
    spec = step["table_one_spec"]
    spec["group_by"] = group
    spec["variables"] = [dict(spec["variables"][0], name=name) for name in names]
    return AnalysisPlan.model_validate({
        "research_question": _context().research_question, "steps": [step],
    })


def _requirements(*names: str):
    return candidate_baseline_requirements(
        plan=_plan(*names).model_dump(mode="json"),
        source_plan_sha256="a" * 64,
        selected_concepts=["exposure", *names],
        catalog_columns=["exposure", *names],
    )


def _bound_context(*names: str):
    context = _context().model_copy(update={"variables": [
        *_context().variables,
        ConceptDescriptor(name="cci_value", source_concept="charlson", role="other", dtype="float64"),
        ConceptDescriptor(name="charlson_n", source_concept="charlson", role="meta", dtype="int64"),
        ConceptDescriptor(name="charlson_time", source_concept="charlson", role="time", dtype="float64"),
        ConceptDescriptor(name="unrelated", source_concept="other_score", role="other", dtype="float64"),
    ]})
    return bind_baseline_requirements(context, _requirements(*names).model_dump(mode="json"))


def test_missing_confirmed_variable_blocks_approval_and_lists_all_missing() -> None:
    context = _bound_context("age", "charlson")
    review = build_plan_scientific_review(context=context, plan=_plan("age"))
    finding = next(f for f in review.findings if f.code == "ACCEPTED_BASELINE_CONTENT_MISSING")
    assert finding.severity == "blocker"
    assert finding.remediation_route == "agent_plan_revision"
    assert "charlson" in finding.message
    assert review.approval_allowed is False
    assert review.dimension_scores["content_completeness"] < 100
    assert review.facts["accepted_baseline_coverage"]["tables"][0]["missing_variables"] == ["charlson"]


def test_planner_and_review_use_same_source_bound_roster() -> None:
    context = _bound_context("age", "charlson")
    prompt = render_plan_scientific_guardrails(context)
    assert "charlson" in prompt and "cci_value" in prompt
    assert "charlson_n" not in prompt
    coverage = baseline_requirement_coverage(context, _plan("age", "cci_value"))
    assert coverage["status"] == "complete"
    assert coverage["tables"][0]["matched_step_id"] == "renamed_baseline"


@pytest.mark.parametrize("column", ["charlson_n", "charlson_time", "unrelated", "charlson_first"])
def test_counts_times_other_concepts_and_invented_suffix_do_not_count(column: str) -> None:
    coverage = baseline_requirement_coverage(_bound_context("age", "charlson"), _plan("age", column))
    assert coverage["status"] == "incomplete"
    assert coverage["tables"][0]["missing_variables"] == ["charlson"]


def test_exact_concept_name_still_cannot_substitute_metadata_for_a_value() -> None:
    context = _bound_context("age", "charlson")
    context = context.model_copy(update={"variables": [
        *context.variables,
        ConceptDescriptor(name="charlson", source_concept="charlson", role="meta", dtype="int64"),
    ]})
    assert baseline_requirement_coverage(context, _plan("age", "charlson"))["status"] == "incomplete"


def test_partial_tables_cannot_pool_their_rows_to_satisfy_one_accepted_table() -> None:
    context = _bound_context("age", "charlson")
    first = _plan("age")
    second = _plan("cci_value").steps[0].model_copy(update={"step_id": "second_baseline"})
    plan = first.model_copy(update={"steps": [*first.steps, second]})
    assert baseline_requirement_coverage(context, plan)["status"] == "incomplete"


def test_input_or_prose_mention_and_wrong_group_do_not_satisfy_table_rows() -> None:
    plan = _plan("age")
    step = plan.steps[0].model_copy(update={
        "inputs": [*plan.steps[0].inputs, "cci_value"],
        "intent": "Includes Charlson in baseline table.",
    })
    context = _bound_context("age", "charlson")
    assert baseline_requirement_coverage(context, plan.model_copy(update={"steps": [step]}))["status"] == "incomplete"
    assert baseline_requirement_coverage(context, _plan("age", "cci_value", group="death"))["status"] == "incomplete"


def test_no_confirmed_roster_does_not_make_available_columns_mandatory() -> None:
    context = _context()
    assert baseline_requirement_coverage(context, _plan("age"))["status"] == "not_bound"
    review = build_plan_scientific_review(context=context, plan=_plan("age"))
    assert not any(f.code.startswith("ACCEPTED_BASELINE") for f in review.findings)


def test_unavailable_requirement_stays_visible_and_routes_to_data_owner() -> None:
    context = bind_baseline_requirements(_context(), _requirements("age", "score_x").model_dump(mode="json"))
    review = build_plan_scientific_review(context=context, plan=_plan("age"))
    finding = next(f for f in review.findings if f.code == "ACCEPTED_BASELINE_MATERIALIZATION_MISSING")
    assert finding.severity == "blocker"
    assert finding.remediation_route == "runtime_capability"
    assert "score_x" in finding.message


def test_already_operationalized_coordinate_cannot_be_replaced_by_another_summary() -> None:
    requirement = candidate_baseline_requirements(
        plan=_plan("age", "score_initial").model_dump(mode="json"),
        source_plan_sha256="a" * 64,
        selected_concepts=["age", "score", "exposure"],
        catalog_columns=["age", "score_initial", "exposure"],
    )
    context = _context().model_copy(update={"variables": [
        *_context().variables,
        ConceptDescriptor(name="score_average", source_concept="score", role="other", dtype="float64"),
    ]})
    context = bind_baseline_requirements(context, requirement.model_dump(mode="json"))
    assert baseline_requirement_coverage(context, _plan("age", "score_average"))["status"] == "incomplete"


def test_context_and_config_recovery_preserve_requirement_identity(tmp_path) -> None:
    payload = _requirements("age", "charlson").model_dump(mode="json")
    config = PipelineConfig(workdir=tmp_path, require_human_plan_review=True, bound_baseline_requirements=payload)
    original_digest = config.canonical_digest()
    payload["tables"][0]["variables"].pop()
    assert config.canonical_digest() == original_digest
    restored = PipelineConfig.from_recovery_payload(config.recovery_payload(), expected_digest=original_digest)
    assert restored.bound_baseline_requirements == config.bound_baseline_requirements
    context = bind_baseline_requirements(_context(), config.bound_baseline_requirements)
    assert bind_baseline_requirements(context, restored.bound_baseline_requirements, restoring=True) is context
    with pytest.raises(ValueError, match="binding_drift"):
        bind_baseline_requirements(context, payload, restoring=True)
    with pytest.raises(ValueError, match="binding_drift"):
        bind_baseline_requirements(_context(), restored.bound_baseline_requirements, restoring=True)


def test_projection_does_not_truncate_or_accept_unknown_catalog_coordinates() -> None:
    names = [f"variable_{index}" for index in range(40)]
    requirement = _requirements(*names)
    assert len(requirement.tables[0].variables) == 40
    with pytest.raises(ValueError, match="outside the catalog"):
        candidate_baseline_requirements(
            plan=_plan("age", "invented").model_dump(mode="json"),
            source_plan_sha256="a" * 64, selected_concepts=["age"],
            catalog_columns=["age", "exposure"],
        )


def test_invalid_host_payload_is_not_silently_ignored() -> None:
    with pytest.raises(ValueError):
        bind_baseline_requirements(_context(), {"tables": []})


def test_prompt_retrieval_cannot_hide_accepted_baseline_columns() -> None:
    from easyicu.research_agent.research_context.builder import build_retrieved_research_context

    context = _bound_context("age", "charlson")
    retrieved = build_retrieved_research_context(context, query="death", top_k=1)
    assert {"age", "cci_value"}.issubset(v.name for v in retrieved.variables)
    assert baseline_requirement_coverage(retrieved, _plan("age", "cci_value"))["status"] == "complete"


def test_failed_revision_retains_requirements_for_the_next_fresh_plan() -> None:
    first = build_plan_scientific_review(context=_bound_context("age", "charlson"), plan=_plan("age"))
    inherited = first.facts["accepted_baseline_requirements"]
    context = bind_baseline_requirements(_context(), inherited)
    second = build_plan_scientific_review(context=context, plan=_plan("age"))
    assert second.facts["accepted_baseline_requirements"] == inherited
    assert second.facts["accepted_baseline_coverage"]["tables"][0]["missing_variables"] == ["charlson"]
    assert second.approval_allowed is False


def test_bound_requirements_cannot_disable_the_review_gate(tmp_path) -> None:
    with pytest.raises(ValueError, match="requires require_human_plan_review"):
        PipelineConfig(workdir=tmp_path, bound_baseline_requirements=_requirements("age").model_dump(mode="json"))


@pytest.mark.parametrize("ambiguous", [False, True])
def test_composite_group_uses_owner_mapping_without_selecting_between_definitions(ambiguous) -> None:
    requirements = candidate_baseline_requirements(
        plan=_plan("age", group="comorbidity_loader").model_dump(mode="json"),
        source_plan_sha256="a" * 64,
        selected_concepts=["age", "comorbidity_loader"], catalog_columns=["age", "comorbidity_loader"],
    )
    values = [ConceptDescriptor(name="score_value", source_concept="charlson", role="other", dtype="int64")]
    if ambiguous:
        values.append(ConceptDescriptor(name="other_score_value", source_concept="elixhauser", role="other", dtype="int64"))
    context = _context().model_copy(update={"variables": [*_context().variables, *values]})
    context = bind_baseline_requirements(context, requirements.model_dump(mode="json"))
    coverage = baseline_requirement_coverage(context, _plan("age", group="score_value"))
    assert (coverage["status"] == "complete") is (not ambiguous)
