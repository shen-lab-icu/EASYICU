"""A question that asks for the paper is answered by its documents (UX rc_70e).

「写成论文，交付 PDF 和参考文献（.bib）」 named no unknown product, yet the
host reader returned nothing for the whole quote: 「写成论文」 and 「.bib」
left a residue. The plan that owned the PDF and the references was refused
as ``question_requirement_not_covered``. On regeneration the Planner split
the quote into 「写成论文」 and 「参考文献（.bib）」, and both were refused.
"""
import json

import pytest

from easyicu.research_agent.agents.progressive_planner import FAMILY_SPEC_STRATEGY, ProgressivePlannerAgent
from easyicu.research_agent.contracts.product_files import (
    DeliveryRequest, delivery_requests, manuscript_request_only, neutral_delivery_kinds,
    requested_deliverable_kinds,
)
from easyicu.research_agent.orchestration.progressive_planning import question_requirement_outcome
from easyicu.research_agent.providers.mocks import ScriptedMockLLMClient
from tests.research_agent.planning.family_spec_fixtures import (
    ALLOWED_CITATIONS, DIRECT_COMPARATORS, _context, _request, _spec_payload,
)

PDF, REFERENCES, PREVIEW = "manuscript_pdf", "references", "article_preview"
UX_QUOTE = "写成论文，交付 PDF 和参考文献（.bib）"
PAPER_WITH_PDF_AND_REFERENCES = (
    UX_QUOTE,
    "写成论文，并提供 PDF 和参考文献",
    "写成论文，并给出 PDF 和参考文献",
    "write it up as a paper with PDF and references",
    "Write it up as a paper, and provide the PDF and references",
    "produce a paper with PDF and references",
)
ONE_DOCUMENT = (
    ("撰写论文，并提供 PDF", (PDF,)),
    ("生成一篇完整论文，交付 PDF", (PDF,)),
    ("整理成论文，交付参考文献", (REFERENCES,)),
    ("turn it into a manuscript and provide the PDF", (PDF,)),
    ("Turn the results into a full paper, and include the references", (REFERENCES,)),
    ("参考文献（.bib）", (REFERENCES,)),
)
PAPER_ONLY = ("写成论文", "请写成论文", "撰写论文", "write a paper", "please write it up as a paper")
PAPER_WITH_SCIENCE = (
    "写成论文并做 Cox 回归，提供 PDF",
    "用 Cox 回归写成论文，提供 PDF",
    "写成论文，提供按性别分层的 PDF",
    "write it up as a paper with a sex-stratified PDF",
    "produce a paper comparing the groups, with PDF",
)
QUESTION = (
    "Among adults in the ICU, how is first-day injury stage associated with in-hospital death after "
    "the landmark, and how does ICU length of stay differ by stage?"
)


@pytest.mark.parametrize("quote", PAPER_WITH_PDF_AND_REFERENCES)
def test_a_paper_request_reads_as_its_named_documents(quote):
    assert neutral_delivery_kinds(quote) == (PDF, REFERENCES)
    requests = delivery_requests(quote)
    assert requests[PDF] is DeliveryRequest.REQUESTED
    assert requests[REFERENCES] is DeliveryRequest.REQUESTED
    assert requests[PREVIEW] is DeliveryRequest.UNMENTIONED
    assert not manuscript_request_only(quote)


@pytest.mark.parametrize(("quote", "kinds"), ONE_DOCUMENT)
def test_a_paper_request_with_one_document_names_that_document(quote, kinds):
    assert neutral_delivery_kinds(quote) == kinds


def test_a_bib_file_names_the_references():
    assert requested_deliverable_kinds("provide a .bib file") == (REFERENCES,)
    assert requested_deliverable_kinds("交付参考文献（.bib）") == (REFERENCES,)


@pytest.mark.parametrize("quote", PAPER_ONLY)
def test_a_paper_alone_names_no_document_but_is_a_paper_request(quote):
    assert neutral_delivery_kinds(quote) == ()
    assert manuscript_request_only(quote)


@pytest.mark.parametrize("quote", PAPER_WITH_SCIENCE)
def test_a_paper_request_carrying_science_names_nothing(quote):
    assert neutral_delivery_kinds(quote) == ()
    assert not manuscript_request_only(quote)


def _judged(tmp_path, question_tail, *requirements):
    context = _context().model_copy(update={"research_question": f"{QUESTION} {question_tail}。"})
    context.user_preferences.must_have_outputs = (
        "Use the typed association.ordinal_trend action for the secondary ICU length-of-stay analysis."
    )
    request = _request(context, cohort_mode="predicate_filtered")
    payload = _spec_payload(request)
    payload["question_requirements"] = [
        {"id": f"r{index}", "kind": "delivery", "quote": quote, "concepts": [],
         "coverage": "plan", "deliverable_kinds": list(kinds)}
        for index, (quote, kinds) in enumerate(requirements, 1)
    ]
    llm = ScriptedMockLLMClient([json.dumps(payload)] * 3)
    attempt = ProgressivePlannerAgent(llm).run_attempt(
        context, planner_strategy=FAMILY_SPEC_STRATEGY,
        allowed_literature_citation_keys=ALLOWED_CITATIONS,
        direct_comparator_literature_keys=DIRECT_COMPARATORS,
        comparison_literature_keys=DIRECT_COMPARATORS,
        enforce_article_contract=True, planning_contract_context="",
        required_primary_cohort_selection_mode="predicate_filtered",
    )
    assert attempt.output.deliverables.requested == (PDF, REFERENCES)
    findings = question_requirement_outcome(
        context=context, plan=attempt.output, facts=attempt.facts,
        run_dir=tmp_path, planning_contract_context="",
    )
    record = json.loads((tmp_path / "question_requirements.json").read_text())
    return findings, {row["id"]: row for row in record["judged"]}


OWNERS = {PDF: ["host_delivery_manuscript_pdf"], REFERENCES: ["host_delivery_references"]}


@pytest.mark.parametrize("quote", PAPER_WITH_PDF_AND_REFERENCES[:4])
def test_a_plan_that_owns_the_pdf_and_references_answers_the_paper_request(tmp_path, quote):
    findings, rows = _judged(tmp_path, quote, (quote, (PDF, REFERENCES)))

    assert rows["r1"]["disposition"] == "covered", rows["r1"]
    assert rows["r1"]["delivery"]["owner_step_ids_by_kind"] == OWNERS
    assert not [finding for finding in findings if "question_requirement" in str(finding)]


@pytest.mark.parametrize("paper_kinds", ((), (PDF,)))
def test_a_split_paper_request_is_answered_by_the_documents_on_the_card(tmp_path, paper_kinds):
    findings, rows = _judged(
        tmp_path, UX_QUOTE, ("写成论文", paper_kinds), ("参考文献（.bib）", (REFERENCES,)),
    )

    assert rows["r1"]["disposition"] == "covered", rows["r1"]
    # The card asked for the PDF, not the browser preview: the PDF answers the paper.
    assert rows["r1"]["delivery"]["owner_step_ids_by_kind"] == {PDF: OWNERS[PDF]}
    assert rows["r2"]["disposition"] == "covered", rows["r2"]
    assert rows["r2"]["delivery"]["owner_step_ids_by_kind"] == {REFERENCES: OWNERS[REFERENCES]}
    assert not [finding for finding in findings if "question_requirement" in str(finding)]


def test_a_paper_request_carrying_science_is_still_judged_on_its_science(tmp_path):
    quote = "写成论文并按损伤分期比较住院时长"
    _findings, rows = _judged(tmp_path, f"{quote}，交付 PDF 和参考文献", (quote, ()))

    assert rows["r1"]["disposition"] != "covered", rows["r1"]
    assert "delivery" not in rows["r1"]
