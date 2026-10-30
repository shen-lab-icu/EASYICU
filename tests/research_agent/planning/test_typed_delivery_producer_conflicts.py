"""Final typed outputs have one producer even when no step consumes them."""

import pytest

from easyicu.research_agent.planning.plan_graph import typed_plan_dag_findings
from easyicu.research_agent.planning.replan_gate import (
    replan_candidate_contract_findings,
)
from easyicu.research_agent.schema import (
    AnalysisPlan,
    AnalysisStep,
    CohortDescriptor,
    ResearchContext,
)


def _step(step_id, outputs):
    return AnalysisStep(step_id=step_id, intent=step_id, expected_outputs=outputs)


@pytest.mark.parametrize("order", [("a", "b"), ("b", "a")])
def test_duplicate_final_product_is_rejected_independently_of_completion_order(order):
    plan = AnalysisPlan(
        research_question="Describe each group",
        steps=[_step(step, ["table:mortality"]) for step in order],
    )
    findings = typed_plan_dag_findings(plan)
    conflicts = [f for f in findings if f.detail.get("reason") == "typed_output_producer_ambiguous"]
    assert len(conflicts) == 1
    assert conflicts[0].severity == "error"
    assert conflicts[0].detail == {
        "reason": "typed_output_producer_ambiguous",
        "typed_product": "table:mortality",
        "producer_step_ids": ["a", "b"],
    }


def test_distinct_products_and_duplicate_spelling_within_one_step_are_allowed():
    plan = AnalysisPlan(
        research_question="Report a table and figure",
        steps=[_step("a", ["table:mortality", "table:mortality"]),
               _step("b", ["figure:mortality"])],
    )
    assert not [f for f in typed_plan_dag_findings(plan)
                if f.detail.get("reason") == "typed_output_producer_ambiguous"]


def test_a_replan_with_competing_final_outputs_is_rejected():
    plan = AnalysisPlan(
        research_question="Describe each group",
        steps=[_step("a", ["table:mortality"]), _step("b", ["table:mortality"])],
        revision=2,
    )
    context = ResearchContext(
        research_question=plan.research_question,
        cohort=CohortDescriptor(cohort_name="demo", database="synthetic", n_patients=10, n_stays=10),
        variables=[],
    )
    assert any(
        f.severity == "error" and f.detail.get("reason") == "typed_output_producer_ambiguous"
        for f in replan_candidate_contract_findings(plan=plan, context=context)
    )
