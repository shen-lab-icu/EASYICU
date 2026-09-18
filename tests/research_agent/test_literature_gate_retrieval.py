"""Track 3 literature hard dependency: retrieval-backed plan gate (offline only).

Covers the green path, the cut-retrieval red path, per-decision coverage,
plan-binding enforcement, and the pure ``contract_from_bundle`` projection
over existing live-PubMed lineage (stubbed client, no network/Provider).
"""

from __future__ import annotations

import pytest

from easyicu.ai_optin import is_offline_llm_choice
from easyicu.research_agent.gates.literature_retrieval_gate import (
    VALIDATOR,
    literature_retrieval_findings,
    literature_retrieval_gate_blocks,
)
from easyicu.research_agent.planning.literature_retrieval_contract import (
    LiteratureRetrievalContract,
    LiteratureRetrievalHit,
    LiteratureRetrievalSearch,
    LiteratureRetrievalSupport,
    contract_from_bundle,
)
from easyicu.research_agent.schema import AnalysisPlan, AnalysisStep


def _searches() -> list[LiteratureRetrievalSearch]:
    return [
        LiteratureRetrievalSearch(
            source="pubmed", query="sepsis ICU cohort mortality"
        ),
        LiteratureRetrievalSearch(
            source="pubmed", query="ICU time zero landmark mortality"
        ),
    ]


def _hits() -> list[LiteratureRetrievalHit]:
    return [
        LiteratureRetrievalHit(
            citation_key="pop_2024",
            source="pubmed",
            query="sepsis ICU cohort mortality",
            pmid="11111111",
        ),
        LiteratureRetrievalHit(
            citation_key="out_2023",
            source="pubmed",
            query="ICU time zero landmark mortality",
            pmid="22222222",
        ),
        LiteratureRetrievalHit(
            citation_key="meth_2022",
            source="pubmed",
            query="sepsis ICU cohort mortality",
            pmid="33333333",
        ),
    ]


def _supports() -> list[LiteratureRetrievalSupport]:
    return [
        LiteratureRetrievalSupport(
            decision="population",
            citation_key="pop_2024",
            supporting_statement="Adults admitted to ICU with suspected sepsis were eligible.",
        ),
        LiteratureRetrievalSupport(
            decision="outcome_window",
            citation_key="out_2023",
            supporting_statement="Follow-up started at ICU admission over 28 days.",
        ),
        LiteratureRetrievalSupport(
            decision="method_applicability",
            citation_key="meth_2022",
            supporting_statement="Adjusted logistic model applies when outcome is binary.",
        ),
    ]


def _contract(
    *,
    searches: list | None = None,
    hits: list | None = None,
    supports: list | None = None,
    search_conducted: bool = True,
) -> LiteratureRetrievalContract:
    return LiteratureRetrievalContract(
        research_question="Is exposure associated with outcome in ICU patients?",
        searches=_searches() if searches is None else searches,
        hits=_hits() if hits is None else hits,
        supports=_supports() if supports is None else supports,
        search_conducted=search_conducted,
        sources_returning=["pubmed"],
    )


def _plan(keys: tuple[str, str, str] = ("pop_2024", "out_2023", "meth_2022")) -> AnalysisPlan:
    pop_key, out_key, meth_key = keys
    return AnalysisPlan(
        research_question="Is exposure associated with outcome in ICU patients?",
        steps=[
            AnalysisStep(
                step_id="s_pop",
                intent="Define the analysis cohort.",
                literature_citation_keys=[pop_key],
                literature_design_bindings=[
                    {
                        "citation_key": pop_key,
                        "design_elements": ["population"],
                        "application": "Adopt the adult ICU sepsis eligibility.",
                    }
                ],
            ),
            AnalysisStep(
                step_id="s_out",
                intent="Define outcome and follow-up window.",
                literature_citation_keys=[out_key],
                literature_design_bindings=[
                    {
                        "citation_key": out_key,
                        "design_elements": ["outcome", "time_zero"],
                        "application": "Start follow-up at ICU admission for 28 days.",
                    }
                ],
            ),
            AnalysisStep(
                step_id="s_meth",
                intent="Fit the prespecified adjusted model.",
                literature_citation_keys=[meth_key],
                literature_design_bindings=[
                    {
                        "citation_key": meth_key,
                        "design_elements": ["estimand"],
                        "application": "Report the adjusted association as planned.",
                    }
                ],
            ),
        ],
    )


def test_green_path_passes_with_mock_transport_and_no_opt_in() -> None:
    # Two-tier contract (review finding): lineage plus citation binding is
    # clean here, so nothing blocks; the recorded sentences are not
    # source-backed, which the gate reports as warnings, not errors.
    assert is_offline_llm_choice("MockLLMClient") is True
    findings = literature_retrieval_findings(
        plan=_plan(), contract=_contract(), llm_choice="MockLLMClient"
    )
    assert findings
    assert all(f.severity == "warning" for f in findings)
    assert {f.detail["kind"] for f in findings} == {
        "literature_support_not_source_backed"
    }
    assert literature_retrieval_gate_blocks(findings) is False


def test_fully_backed_path_reports_no_findings() -> None:
    from datetime import datetime, timezone

    from easyicu.research_agent.literature import LiteratureBundle
    from easyicu.research_agent.planning.literature_design_authority import (
        LiteratureDesignEvidenceCard,
    )

    def _card(key: str, dimension: str) -> LiteratureDesignEvidenceCard:
        return LiteratureDesignEvidenceCard.model_validate(
            {
                "citation_key": key,
                "evidence_role": "design_analogue",
                "access_mode": "open_access_fulltext",
                "full_text_locator": f"pmc:{key}",
                "full_text_sha256": "ab" * 32,
                "supplement_status": "unknown",
                "reviewed_at": datetime(2026, 9, 18, tzinfo=timezone.utc),
                "evidence": [
                    {
                        "dimension": dimension,
                        "source_backed_summary": (
                            "Reviewed full-text fact shaping the planned "
                            "design decision."
                        ),
                    }
                ],
            },
            strict=True,
        )

    from easyicu.research_agent.literature import LiteratureSearchProvenance

    bundle = LiteratureBundle(
        research_question="Is exposure associated with outcome in ICU patients?",
        citations=[],
        search_provenance=LiteratureSearchProvenance(
            curated_seed_count=0,
            sources_enabled=["pubmed"],
            sources_returning=["pubmed"],
            search_queries={
                "pubmed": [
                    "sepsis ICU cohort mortality",
                    "ICU time zero landmark mortality",
                ]
            },
            record_queries={
                "pop_2024": ["sepsis ICU cohort mortality"],
                "out_2023": ["ICU time zero landmark mortality"],
                "meth_2022": ["sepsis ICU cohort mortality"],
            },
            search_conducted=True,
        ),
        screening_decisions=[],
        design_evidence_cards=[
            _card("pop_2024", "study_population"),
            _card("out_2023", "time_zero_and_windows"),
            _card("meth_2022", "primary_model_and_sensitivities"),
        ],
    )
    contract = contract_from_bundle(bundle, supports=_supports())
    assert contract.search_conducted is True
    assert {h.citation_key for h in contract.hits} == {
        "pop_2024",
        "out_2023",
        "meth_2022",
    }
    assert contract.source_backed == {
        "pop_2024": ["population"],
        "out_2023": ["outcome_window"],
        "meth_2022": ["method_applicability"],
    }
    findings = literature_retrieval_findings(plan=_plan(), contract=contract)
    assert findings == []
    assert literature_retrieval_gate_blocks(findings) is False


def test_cut_retrieval_turns_gate_red() -> None:
    empty = LiteratureRetrievalContract(
        research_question="Is exposure associated with outcome in ICU patients?",
        searches=[],
        hits=[],
        supports=[],
        search_conducted=False,
        sources_returning=[],
    )
    findings = literature_retrieval_findings(plan=_plan(), contract=empty)
    assert findings
    assert all(f.validator == VALIDATOR and f.severity == "error" for f in findings)
    assert literature_retrieval_gate_blocks(findings) is True
    assert findings[0].detail["kind"] == "literature_search_not_conducted"
    assert findings[0].detail["provider_called"] is False


def test_cut_hits_turns_gate_red() -> None:
    contract = _contract(hits=[], supports=[])
    findings = literature_retrieval_findings(plan=_plan(), contract=contract)
    assert literature_retrieval_gate_blocks(findings) is True
    assert findings[0].detail["kind"] == "literature_hits_missing"


def test_missing_one_decision_reports_that_decision() -> None:
    supports = [s for s in _supports() if s.decision != "method_applicability"]
    contract = _contract(supports=supports)
    findings = literature_retrieval_findings(plan=_plan(), contract=contract)
    kinds = {(f.detail or {}).get("missing_decision") for f in findings}
    assert "method_applicability" in kinds
    assert literature_retrieval_gate_blocks(findings) is True


def test_support_key_not_bound_by_plan_turns_gate_red() -> None:
    plan = _plan(keys=("pop_2024", "out_2023", "other_2021"))
    findings = literature_retrieval_findings(plan=plan, contract=_contract())
    assert literature_retrieval_gate_blocks(findings) is True
    assert any(
        (f.detail or {}).get("kind") == "literature_support_not_bound_by_plan"
        for f in findings
    )


def test_non_bibliographic_hits_turn_gate_red() -> None:
    searches = [
        LiteratureRetrievalSearch(source="curated", query="sepsis ICU cohort")
    ]
    hits = [
        LiteratureRetrievalHit(
            citation_key="pop_2024", source="curated", query="sepsis ICU cohort"
        )
    ]
    supports = [
        LiteratureRetrievalSupport(
            decision="population",
            citation_key="pop_2024",
            supporting_statement="Adults admitted to ICU with suspected sepsis were eligible.",
        )
    ]
    contract = LiteratureRetrievalContract(
        research_question="Is exposure associated with outcome in ICU patients?",
        searches=searches,
        hits=hits,
        supports=supports,
        search_conducted=True,
        sources_returning=["curated"],
    )
    findings = literature_retrieval_findings(plan=_plan(), contract=contract)
    assert literature_retrieval_gate_blocks(findings) is True
    assert findings[0].detail["kind"] == "literature_source_not_bibliographic"


def test_support_for_non_retrieved_record_fails_closed() -> None:
    with pytest.raises(ValueError, match="non-retrieved"):
        LiteratureRetrievalContract(
            research_question="Is exposure associated with outcome in ICU patients?",
            searches=_searches(),
            hits=_hits(),
            supports=[
                LiteratureRetrievalSupport(
                    decision="population",
                    citation_key="ghost_2099",
                    supporting_statement="This record was never retrieved by any query.",
                )
            ],
            search_conducted=True,
            sources_returning=["pubmed"],
        )


def test_contract_from_bundle_reuses_live_pubmed_lineage_without_network() -> None:
    from easyicu.research_agent.literature import (
        CitationRecord,
        LiteratureBundle,
        LiteratureScreeningDecision,
        LiteratureSearchProvenance,
    )

    bundle = LiteratureBundle(
        research_question="Is exposure associated with outcome in ICU patients?",
        citations=[
            CitationRecord(key="pop_2024", title="Population study", year="2024"),
            CitationRecord(key="out_2023", title="Outcome study", year="2023"),
            CitationRecord(key="meth_2022", title="Method study", year="2022"),
        ],
        search_provenance=LiteratureSearchProvenance(
            curated_seed_count=1,
            sources_enabled=["pubmed"],
            sources_returning=["pubmed"],
            search_queries={
                "pubmed": [
                    "sepsis ICU cohort mortality",
                    "ICU time zero landmark mortality",
                ]
            },
            record_queries={
                "pop_2024": ["sepsis ICU cohort mortality"],
                "out_2023": ["ICU time zero landmark mortality"],
                "meth_2022": ["sepsis ICU cohort mortality"],
            },
            search_conducted=True,
        ),
        screening_decisions=[
            LiteratureScreeningDecision(
                citation_key="pop_2024",
                source="pubmed",
                disposition="include",
                evidence_role="related_context",
                rationale="Matches population.",
                query="sepsis ICU cohort mortality",
            ),
            LiteratureScreeningDecision(
                citation_key="out_2023",
                source="pubmed",
                disposition="include",
                evidence_role="related_context",
                rationale="Matches outcome window.",
                query="ICU time zero landmark mortality",
            ),
            LiteratureScreeningDecision(
                citation_key="meth_2022",
                source="pubmed",
                disposition="include",
                evidence_role="method",
                rationale="Matches method.",
                query="sepsis ICU cohort mortality",
            ),
        ],
    )
    contract = contract_from_bundle(bundle, supports=_supports())
    assert contract.search_conducted is True
    assert {s.source for s in contract.searches} == {"pubmed"}
    assert {h.citation_key for h in contract.hits} == {
        "pop_2024",
        "out_2023",
        "meth_2022",
    }
    findings = literature_retrieval_findings(
        plan=_plan(), contract=contract, llm_choice="mock"
    )
    # Lineage plus binding is clean; without source backing the gate
    # reports warnings, never a block.
    assert findings
    assert all(f.severity == "warning" for f in findings)
    assert literature_retrieval_gate_blocks(findings) is False


"""Track 3 wiring: plan supports projection + finalization contract."""

from types import SimpleNamespace

from easyicu.research_agent.planning.literature_retrieval_contract import (
    contract_for_plan_finalization,
    supports_from_plan,
)


def _binding(key="Smith2020", application="adults admitted to ICU stay cohorts",
             elements=("population",)):
    return SimpleNamespace(
        citation_key=key, application=application, design_elements=list(elements)
    )


def _candidate(disposition="selected", dimension="study_population",
               rationale="cohort matches adult ICU population",
               keys=("Smith2020",)):
    return SimpleNamespace(
        disposition=disposition,
        literature_design_decisions=[
            SimpleNamespace(
                dimension=dimension, rationale=rationale, citation_keys=list(keys)
            )
        ],
    )


def _plan_with_bindings():
    return SimpleNamespace(
        steps=[SimpleNamespace(literature_design_bindings=[_binding()])],
        design_selection=SimpleNamespace(candidates=[_candidate()]),
    )


def _bundle(question="does X associate with Y in ICU stays?", conducted=False):
    return SimpleNamespace(
        research_question=question,
        search_provenance=SimpleNamespace(
            search_queries={},
            record_queries={},
            search_conducted=conducted,
            sources_returning=[],
        ),
        citations=[],
        screening_decisions=[],
    )


def test_supports_from_plan_projects_steps_and_candidates():
    supports = supports_from_plan(_plan_with_bindings())
    by_decision = {item.decision for item in supports}
    assert "population" in by_decision
    assert all(item.supporting_statement for item in supports)


def test_supports_from_plan_skips_malformed_bindings():
    plan = SimpleNamespace(
        steps=[
            SimpleNamespace(
                literature_design_bindings=[
                    _binding(key="bad key!!"),
                    _binding(application="too short"),
                    _binding(elements=("reporting",)),
                ]
            )
        ],
        design_selection=SimpleNamespace(
            candidates=[_candidate(disposition="dropped")]
        ),
    )
    assert supports_from_plan(plan) == []


def test_finalization_contract_without_bundle_is_unconducted():
    contract = contract_for_plan_finalization(
        plan=_plan_with_bindings(), preplan_literature=None
    )
    assert contract.search_conducted is False
    assert len(contract.research_question) >= 8


def _conducted_bundle():
    bundle = _bundle(conducted=True)
    bundle.search_provenance.search_queries = {"pubmed": ["icu stays X"]}
    bundle.search_provenance.record_queries = {"Smith2020": ["icu stays X"]}
    bundle.search_provenance.sources_returning = ["pubmed"]
    bundle.screening_decisions = [
        SimpleNamespace(citation_key="Smith2020", source="pubmed")
    ]
    return bundle


def test_finalization_contract_projects_bundle_and_supports():
    contract = contract_for_plan_finalization(
        plan=_plan_with_bindings(), preplan_literature=_conducted_bundle()
    )
    assert contract.search_conducted is True
    assert [item.decision for item in contract.supports] == ["population"]
    assert [hit.citation_key for hit in contract.hits] == ["Smith2020"]


def test_contract_from_bundle_accepts_question_override():
    contract = contract_from_bundle(_bundle(question="  "), research_question="override question here")
    assert contract.research_question == "override question here"
