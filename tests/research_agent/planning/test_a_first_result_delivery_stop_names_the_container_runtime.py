"""A requested paper on a first-result run says the container runtime is what it needs (UX rc_826).

A desktop install without a usable Docker falls back to the host runner, and
the host runner selects the first-result profile, which writes no paper. The
stop told the researcher to "start a separate article run", which that
computer cannot do.
"""
import json

import pytest

from easyicu.research_agent.agents.progressive_planner import FAMILY_SPEC_STRATEGY, ProgressivePlannerAgent
from easyicu.research_agent.contracts.claim_ceiling import first_result_instructions
from easyicu.research_agent.contracts.product_files import FIRST_RESULT_DELIVERY_STOP, PlanDeliverables
from easyicu.research_agent.orchestration.profiles import get_submission_profile
from easyicu.research_agent.planning.approval_stops import first_result_delivery_stop
from easyicu.research_agent.providers.mocks import ScriptedMockLLMClient
from easyicu.webserver.research_launch_runtime import _submission_profile_ref
from tests.research_agent.planning.family_spec_fixtures import (
    ALLOWED_CITATIONS, DIRECT_COMPARATORS, _descriptive_context, _descriptive_payload, _request,
)


@pytest.mark.parametrize("live_pubmed", (False, True))
def test_only_the_host_runner_selects_the_first_result_tier(live_pubmed):
    host = get_submission_profile(_submission_profile_ref(
        budget_mode="full_reviewed", live_pubmed=live_pubmed, runner_kind="host",
    ))
    container = get_submission_profile(_submission_profile_ref(
        budget_mode="full_reviewed", live_pubmed=live_pubmed, runner_kind="docker",
    ))

    assert host.execution_tier == "first_result" and host.generate_manuscript is False
    assert container.execution_tier != "first_result"


def _first_result_plan():
    question = "在成人 ICU 住院中，按记录表型描述住院死亡情况。写成论文，交付 PDF 和参考文献（.bib）。"
    context = _descriptive_context().model_copy(update={"research_question": question})
    request = _request(context, cohort_mode=None).model_copy(update={"execution_tier": "first_result"})
    payload = _descriptive_payload(request, baseline_variables=["age", "sex"])
    attempt = ProgressivePlannerAgent(ScriptedMockLLMClient([json.dumps(payload)] * 3)).run_attempt(
        context, planner_strategy=FAMILY_SPEC_STRATEGY,
        allowed_literature_citation_keys=ALLOWED_CITATIONS,
        direct_comparator_literature_keys=DIRECT_COMPARATORS,
        comparison_literature_keys=DIRECT_COMPARATORS,
        enforce_article_contract=True, planning_contract_context=first_result_instructions(),
        required_primary_cohort_selection_mode=None,
    )
    plan = attempt.output
    assert plan.design_selection.execution_tier == "first_result"
    return plan.model_copy(update={"deliverables": PlanDeliverables(
        requested=("manuscript_pdf", "references"), confirmed=True, source="detector_prefill",
    )})


def test_a_requested_paper_on_a_first_result_plan_names_the_container_runtime():
    finding = first_result_delivery_stop(_first_result_plan())

    assert finding is not None and finding.detail["reason"] == FIRST_RESULT_DELIVERY_STOP
    assert finding.detail["deliverable_kinds"] == ["manuscript_pdf", "references"]
    assert "container runtime" in finding.message and "Docker" in finding.message
    assert "runner image" in finding.message
    assert "separate article run" not in finding.message
