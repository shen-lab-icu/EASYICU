from easyicu.research_agent.literature import (
    CitationRecord,
    LiteratureBundle,
    LiteratureScreeningDecision,
)
from easyicu.research_agent.reporting.manuscript_literature import (
    audit_manuscript_literature,
    repair_evidence_ids_mistyped_as_literature,
    repair_missing_context_section_citations,
    repair_missing_methods_method_citation,
    render_writer_literature_digest,
)
from easyicu.research_agent.schema import AnalysisPlan, AnalysisStep


def _bundle() -> LiteratureBundle:
    return LiteratureBundle(
        research_question="Question",
        citations=[
            CitationRecord(
                key="paper_2024",
                title="A relevant ICU cohort",
                year="2024",
                relevance="Source excerpt: Direct comparator.",
            ),
            CitationRecord(
                key="strobe_2007",
                title="The STROBE statement",
                year="2007",
                relevance="Methodology: observational reporting.",
            ),
        ],
        screening_decisions=[
            LiteratureScreeningDecision(
                citation_key="paper_2024",
                source="pubmed",
                disposition="include",
                evidence_role="direct_comparator",
                rationale="Exact P/E/O screen passed.",
                population_match=True,
                exposure_match=True,
                outcome_match=True,
                design_excerpt_available=True,
            )
        ],
    )


def _plan_with_reporting_binding() -> AnalysisPlan:
    return AnalysisPlan(
        research_question="Describe an ICU cohort.",
        steps=[
            AnalysisStep(
                step_id="01_primary_description",
                planned_analysis_role="primary",
                intent="Estimate the prespecified descriptive quantities.",
                expected_outputs=["table:description"],
                method="descriptive",
                literature_citation_keys=["strobe_2007"],
                literature_design_bindings=[
                    {
                        "citation_key": "strobe_2007",
                        "design_elements": ["reporting"],
                        "application": (
                            "Report the setting, analysis unit, denominator, "
                            "missing data and prespecified analyses."
                        ),
                    }
                ],
            )
        ],
    )


def test_writer_digest_exposes_exact_key_and_relevance() -> None:
    digest = render_writer_literature_digest(
        _bundle(),
        plan=_plan_with_reporting_binding(),
    )
    assert "[@paper_2024]" in digest
    assert "A relevant ICU cohort" in digest
    assert "Direct comparator" in digest
    assert "direct_comparator" in digest
    assert "method:" in digest
    assert "Run-bound typed methodology applications" in digest
    assert "step=01_primary_description" in digest
    assert "design_elements=reporting" in digest


def test_manuscript_literature_audit_rejects_aggregate_only_or_unknown() -> None:
    aggregate_only = audit_manuscript_literature(
        "Search flow {evidence:literature_prisma}.", _bundle()
    )
    assert aggregate_only.status == "blocked"
    assert not aggregate_only.exact_citations_present

    unknown = audit_manuscript_literature("Prior work [@invented].", _bundle())
    assert unknown.status == "blocked"
    assert unknown.unknown_keys == ["invented"]


def test_evidence_id_mistyped_as_literature_is_demoted_not_promoted() -> None:
    manuscript = (
        "The study used typed context {[@research_context]} "
        "{evidence:research_context}. Prior work was invented [@invented]."
    )

    repaired, repairs = repair_evidence_ids_mistyped_as_literature(
        manuscript,
        _bundle(),
        evidence_ids=("research_context",),
    )

    assert repairs == ["research_context"]
    assert "[@research_context]" not in repaired
    assert "{evidence:research_context}" in repaired
    assert "[@invented]" in repaired


def test_manuscript_literature_audit_accepts_bound_exact_key() -> None:
    manuscript = """## Introduction
Prior work defines the comparator [@paper_2024].

## Methods
### Statistical analysis
The observational reporting route followed STROBE [@strobe_2007].

## Discussion
The result is compared with the retained ICU study [@paper_2024].
"""
    audit = audit_manuscript_literature(manuscript, _bundle())
    assert audit.status == "pass"
    assert audit.cited_keys == ["paper_2024", "strobe_2007"]
    assert audit.section_cited_keys["methods"] == ["strobe_2007"]


def test_manuscript_literature_audit_accepts_grouped_pandoc_citations() -> None:
    manuscript = """## Introduction
Prior work defines the comparator [@paper_2024; @strobe_2007].

## Methods
The reporting contract followed RECORD and STROBE [@record_2015; @strobe_2007].

## Discussion
The result is compared with prior ICU work [@paper_2024; @strobe_2007].
"""
    bundle = _bundle().model_copy(
        update={
            "citations": [
                *_bundle().citations,
                CitationRecord(
                    key="record_2015",
                    title="The RECORD statement",
                    year="2015",
                    relevance="Methodology: routinely collected data reporting.",
                ),
            ]
        }
    )

    audit = audit_manuscript_literature(manuscript, bundle)

    assert audit.status == "pass"
    assert audit.section_cited_keys["methods"] == ["record_2015", "strobe_2007"]


def test_manuscript_literature_audit_rejects_unknown_key_in_grouped_citation() -> None:
    manuscript = """## Introduction
Prior work [@paper_2024; @invented_2026].

## Methods
Reporting followed STROBE [@strobe_2007].

## Discussion
Comparison used the retained study [@paper_2024].
"""

    audit = audit_manuscript_literature(manuscript, _bundle())

    assert audit.status == "blocked"
    assert audit.unknown_keys == ["invented_2026"]


def test_evidence_id_is_removed_from_grouped_literature_citation() -> None:
    manuscript = "Clinical context [@paper_2024; @research_context]."

    repaired, repairs = repair_evidence_ids_mistyped_as_literature(
        manuscript,
        _bundle(),
        evidence_ids=("research_context",),
    )

    assert repairs == ["research_context"]
    assert repaired == "Clinical context [@paper_2024]."


def test_manuscript_literature_audit_rejects_one_token_citation_theatre() -> None:
    manuscript = """## Introduction
Prior work [@paper_2024].

## Methods
No literature citation here.

## Discussion
No direct comparator citation here.
"""
    audit = audit_manuscript_literature(manuscript, _bundle())

    assert audit.status == "blocked"
    assert audit.missing_required_citation_sections == ["methods", "discussion"]
    assert audit.direct_comparator_sections_missing == ["discussion"]
    assert audit.methods_method_source_missing is True


def test_methods_cannot_use_a_comparator_as_method_authority() -> None:
    manuscript = """## Introduction
Prior work [@paper_2024].

## Methods
The method followed the comparator [@paper_2024].

## Discussion
Comparison with prior work [@paper_2024].
"""
    audit = audit_manuscript_literature(manuscript, _bundle())

    assert audit.status == "blocked"
    assert audit.missing_required_citation_sections == []
    assert audit.methods_method_source_missing is True


def test_missing_methods_citation_is_repaired_only_from_exact_plan_binding() -> None:
    manuscript = """## Introduction
Prior work defines the comparator [@paper_2024].

## Methods
The prespecified cohort and analysis are described below.

## Discussion
The result is compared with the retained ICU study [@paper_2024].
"""

    repaired, repair = repair_missing_methods_method_citation(
        manuscript,
        _bundle(),
        plan=_plan_with_reporting_binding(),
    )

    assert repair is not None
    assert repair["citation_key"] == "strobe_2007"
    assert "run-bound observational reporting guidance [@strobe_2007]" in repaired
    assert audit_manuscript_literature(repaired, _bundle()).status == "pass"


def test_missing_methods_citation_stays_blocked_without_typed_binding() -> None:
    manuscript = """## Introduction
Prior work defines the comparator [@paper_2024].

## Methods
The prespecified cohort and analysis are described below.

## Discussion
The result is compared with the retained ICU study [@paper_2024].
"""
    unbound_plan = AnalysisPlan(
        research_question="Describe an ICU cohort.",
        steps=[
            AnalysisStep(
                step_id="01_primary_description",
                planned_analysis_role="primary",
                intent="Describe the cohort.",
                expected_outputs=["table:description"],
                method="descriptive",
            )
        ],
    )

    repaired, repair = repair_missing_methods_method_citation(
        manuscript,
        _bundle(),
        plan=unbound_plan,
    )

    assert repaired == manuscript
    assert repair is None
    assert audit_manuscript_literature(repaired, _bundle()).status == "blocked"


def test_missing_context_sections_use_exact_run_bound_comparator() -> None:
    manuscript = """## Introduction
No literature citation here.

## Methods
The observational reporting route followed STROBE [@strobe_2007].

## Discussion
No literature citation here.
"""

    repaired, repairs = repair_missing_context_section_citations(
        manuscript,
        _bundle(),
    )

    assert [repair["section"] for repair in repairs] == [
        "introduction",
        "discussion",
    ]
    assert all(repair["citation_key"] == "paper_2024" for repair in repairs)
    assert audit_manuscript_literature(repaired, _bundle()).status == "pass"
