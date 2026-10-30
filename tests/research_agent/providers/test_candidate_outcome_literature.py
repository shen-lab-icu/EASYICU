"""Candidate labels support literature search without inventing source bindings."""
import json

from easyicu.research_agent.gates.data_answerability import analysis_answerability_findings
from easyicu.research_agent.literature import (
    CitationRecord, HypothesisBlueprintAgent, LiteratureBundle,
    _clinical_axis_matches, _literature_outcome_name, _screening_decision_for_record,
    build_pubmed_protocol_queries_for_context,
)
from easyicu.research_agent.research_context.temporal_semantics import TimeWindowSemanticParser
from easyicu.research_agent.schema import CohortDescriptor, ConceptDescriptor, ResearchContext, TimeWindow, UserPreferences
from easyicu.webserver.agent_pipeline_runs import _research_user_preferences


LABEL = "从首次转入 ICU 或中间监护至病例关闭的时长"


def _context(question="SICdb 患者首次进入 ICU 或中间监护后前 24 小时乳酸与病例关闭时长的关联"):
    preferences = _research_user_preferences({"question": question, "outcome": LABEL})
    return ResearchContext(
        research_question=question,
        cohort=CohortDescriptor(cohort_name="synthetic", database="sic", n_stays=0),
        variables=[ConceptDescriptor(name="lact", dtype="float64")], primary_exposure="lact",
        user_preferences=UserPreferences(**preferences),
        temporal_constraints=TimeWindowSemanticParser().parse(question),
        time_windows=[TimeWindow(name="physical", anchor="icu_admission", start_hours=0, end_hours=24)],
    )


def test_candidate_is_available_to_search_but_not_an_executable_endpoint():
    context = _context()
    assert _literature_outcome_name(context) == LABEL
    assert LABEL in build_pubmed_protocol_queries_for_context(context)[0]
    assert context.target_outcome is None and context.endpoint is None
    blueprint = HypothesisBlueprintAgent().run(
        context=context, literature=LiteratureBundle(research_question=context.research_question, citations=[]),
    )
    assert blueprint.feasibility_status == "blocked" and "target_outcome" in blueprint.missing_variables
    saved = json.loads(context.user_preferences.data_constraints)["candidate_outcome_for_literature"]
    assert saved["source_binding"] == "unverified"


def test_chinese_candidate_cannot_collapse_to_generic_icu_comparator():
    context = _context()
    paper = CitationRecord(key="unrelated", year="2024", title="Lactate and ICU length of stay in critically ill adults",
                           relevance="Source excerpt: ICU patients had lactate measured.",
                           publication_types=["Observational Study"])
    decision = _screening_decision_for_record(context=context, record=paper, source="pubmed", query=None)
    assert decision.outcome_match is False and decision.disposition == "exclude"
    assert not _clinical_axis_matches(LABEL, "icu mortality and icu length of stay", axis="outcome")
    assert _clinical_axis_matches(LABEL, "研究" + LABEL + "的关联", axis="outcome")


def test_unresolved_care_unit_origin_blocks_icu_materialization_only():
    context = _context()
    assert any(item.anchor_event == "unresolved" for item in context.temporal_constraints)
    assert any((finding.detail or {}).get("kind") == "unresolved_clinical_origin_vs_icu_materialization"
               for finding in analysis_answerability_findings(context))
    # Even a separately supplied outcome column must not defeat the time
    # origin gate: a ready endpoint does not establish equivalent clocks.
    with_endpoint = context.model_copy(deep=True, update={
        "target_outcome": "case_days",
        "variables": [*context.variables, ConceptDescriptor(name="case_days", dtype="float64")],
    })
    blueprint = HypothesisBlueprintAgent().run(
        context=with_endpoint, literature=LiteratureBundle(research_question=context.research_question, citations=[]),
    )
    assert blueprint.feasibility_status == "blocked"
    assert "target_outcome" not in blueprint.missing_variables
    assert any("time origin is unresolved" in note for note in blueprint.domain_gate_notes)
    standard = _context("ICU 患者入 ICU 后前 24 小时乳酸与病例关闭时长的关联")
    assert not any((finding.detail or {}).get("kind") == "unresolved_clinical_origin_vs_icu_materialization"
                   for finding in analysis_answerability_findings(standard))


def test_untrusted_or_missing_marker_is_not_a_search_label():
    context = _context()
    context.user_preferences.data_constraints = json.dumps({"candidate_outcome_for_literature": {"label": LABEL}})
    assert _literature_outcome_name(context) is None
