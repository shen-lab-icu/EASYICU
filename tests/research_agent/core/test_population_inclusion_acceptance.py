"""An omission decision covers exact criteria before full-plan approval."""

from types import SimpleNamespace

import pytest

from easyicu.research_agent.orchestration.progressive_planning import (
    population_approval_findings,
)
from easyicu.research_agent.orchestration.workflow import (
    HumanReviewDecision,
    HumanReviewStateDrift,
    build_pipeline_workflow,
    human_review_requests_for_plan,
)
from easyicu.research_agent.planning.population_compile import compile_population
from easyicu.research_agent.planning.population_spec import PopulationSpec
from easyicu.research_agent.schema import (
    AnalysisPlan,
    AnalysisStep,
    CohortDescriptor,
    ResearchContext,
)


def paused_population(*, extraction=False):
    spec = PopulationSpec.model_validate(
        {
            "criteria": [
                {
                    "id": name,
                    "kind": "not_typed",
                    "role": "include",
                    "source": "question",
                    "quote": f"Only the {name} population",
                    "why": f"No exact supported kind for {name}",
                }
                for name in ["c1", "c2"]
            ]
            + (
                [
                    {
                        "id": "c3",
                        "kind": "first_icu_stay",
                        "role": "include",
                        "source": "question",
                        "quote": "First ICU stays",
                    }
                ]
                if extraction
                else []
            )
        }
    )
    context = ResearchContext(
        research_question="Describe this population",
        cohort=CohortDescriptor(
            cohort_name="synthetic",
            database="miiv",
            n_stays=0,
        ),
        variables=[],
    )
    population = compile_population(spec, context)
    handoff = SimpleNamespace(
        aborted_result=None,
        evidence=SimpleNamespace(records=lambda: (), verified_records=lambda: ()),
        plan=AnalysisPlan(
            research_question=context.research_question,
            steps=[
                AnalysisStep(
                    step_id="describe", intent="Describe the actual input population"
                )
            ],
            cohort=population.cohort_definition(),
        ),
        findings=population_approval_findings(population),
    )
    calls, records = [], []
    workflow = build_pipeline_workflow(
        plan_invoker=lambda: handoff,
        human_review_invoker=lambda value: human_review_requests_for_plan(
            findings=value.findings,
            plan=value.plan,
            require_plan_review=True,
        ),
        execute_invoker=lambda _: calls.append("execute"),
        write_invoker=lambda *_: calls.append("write"),
        finalise_invoker=lambda *_: calls.append("finalise"),
        human_review_recorder=lambda value: records.extend(value),
    )
    pause = workflow.start()
    return workflow, pause, calls, records, handoff


def approve(request, ids=()):
    return HumanReviewDecision(
        review_id=request.review_id,
        authority_sha256=request.authority_sha256,
        decision="approved",
        reviewer="synthetic_reviewer",
        decided_at="2026-10-16T08:00:00Z",
        population_criterion_acceptances=ids,
    )


@pytest.mark.parametrize(
    "ids", [(), ("c1",), ("c2",), ("c1", "c2", "extra"), ("c1", "c2", "c2")]
)
def test_an_incomplete_omission_decision_cannot_start_execution(ids):
    workflow, pause, calls, records, _ = paused_population()
    with pytest.raises(ValueError, match="cannot be approved"):
        workflow.resume([approve(pause.requests[0], ids)])
    assert calls == records == []


def test_complete_acceptance_is_recorded_without_rewriting_the_plan():
    workflow, pause, calls, records, handoff = paused_population()
    original = handoff.plan.model_dump(mode="json")
    request = pause.requests[0]
    assert request.payload["approval_allowed"] is False
    assert request.payload["population_compile"]["criteria"]
    workflow.resume([approve(request, ("c2", "c1"))])
    assert calls == ["execute", "write", "finalise"]
    assert handoff.plan.model_dump(mode="json") == original
    acceptance = records[0]["population_acceptance"]
    assert acceptance["accepted_criterion_ids"] == ["c1", "c2"]
    assert [row["criterion"]["quote"] for row in acceptance["criteria"]] == [
        "Only the c1 population",
        "Only the c2 population",
    ]
    assert (
        acceptance["plan_sha256"]
        == request.payload["plan_review_authority"]["plan_sha256"]
    )


def test_accepting_omissions_does_not_waive_an_extraction_stop():
    workflow, pause, calls, records, _ = paused_population(extraction=True)
    with pytest.raises(ValueError, match="cannot be approved"):
        workflow.resume([approve(request, ("c1", "c2")) for request in pause.requests])
    assert calls == records == []


def test_acceptance_still_refuses_a_changed_plan():
    workflow, pause, calls, records, handoff = paused_population()
    handoff.plan.research_question = "A changed research question"
    with pytest.raises(HumanReviewStateDrift):
        workflow.resume([approve(pause.requests[0], ("c1", "c2"))])
    assert calls == records == []


def test_ordinary_decision_serialization_keeps_its_previous_fields():
    _, pause, *_ = paused_population()
    assert "population_criterion_acceptances" not in approve(
        pause.requests[0]
    ).model_dump(mode="json")
