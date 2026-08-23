import hashlib
import json

import pytest

from easyicu.research_agent.literature import (
    CitationRecord,
    LiteratureBundle,
    LiteratureScreeningDecision,
    LiteratureSearchProvenance,
)
from easyicu.research_agent.reporting.scientific_maturity import (
    build_scientific_maturity_audit,
    scientific_maturity_audit_from_gates,
    scientific_maturity_readiness_gates,
)
from easyicu.research_agent.schema import (
    AnalysisPlan,
    AnalysisStep,
    CohortDescriptor,
    ConceptDescriptor,
    ResearchContext,
    UserPreferences,
)


def test_article_maturity_separates_valid_user_scope_from_publishable_upgrade(
    tmp_path,
) -> None:
    context = ResearchContext(
        research_question=(
            "Across all ICU stays, report the unadjusted association between a "
            "0-24h exposure and hospital mortality."
        ),
        cohort=CohortDescriptor(
            cohort_name="all stays",
            database="miiv",
            n_patients=90,
            n_stays=100,
            inclusion_criteria=["all ICU stays including readmissions"],
            provenance={"inclusion_criteria": ["all ICU stays including readmissions"]},
        ),
        variables=[
            ConceptDescriptor(
                name="exposure",
                role="composite_score",
                dtype="int64",
                description="registered 0-24h exposure",
                source_concept="registered_exposure",
                analysis_window="0-24h",
            ),
            ConceptDescriptor(
                name="death",
                role="outcome",
                dtype="int64",
                description="hospital mortality",
                source_concept="hospital_mortality",
            ),
        ],
        primary_exposure="exposure",
        target_outcome="death",
        user_preferences=UserPreferences(covariates=[], covariate_selection="exact"),
    )
    plan = AnalysisPlan(
        research_question=context.research_question,
        analysis_type="association",
        steps=[
            AnalysisStep(
                step_id="primary",
                planned_analysis_role="primary",
                intent="Estimate the user-authorized unadjusted association.",
                method="unadjusted logistic regression",
                expected_outputs=["table:association_estimate"],
                literature_citation_keys=["direct_2018", "strobe_2007"],
            )
        ],
    )
    literature = LiteratureBundle(
        research_question=context.research_question,
        citations=[
            CitationRecord(
                key="direct_2018",
                title="Comparable ICU association study",
                year="2018",
                relevance="Study-design excerpt: Adult ICU cohort with the same exposure and hospital mortality.",
            ),
            CitationRecord(
                key="strobe_2007",
                title="STROBE",
                year="2007",
                relevance="Observational reporting guidance.",
            ),
        ],
        search_provenance=LiteratureSearchProvenance(
            curated_seed_count=1,
            sources_enabled=["pubmed"],
            sources_returning=["pubmed"],
            search_queries={"pubmed": ["ICU AND exposure AND hospital mortality"]},
            search_conducted=True,
            searched_at="2026-08-12T00:00:00+00:00",
        ),
        screening_decisions=[
            LiteratureScreeningDecision(
                citation_key="direct_2018",
                source="pubmed",
                disposition="include",
                evidence_role="direct_comparator",
                rationale="P/E/O match from retained abstract.",
                population_match=True,
                exposure_match=True,
                outcome_match=True,
                design_excerpt_available=True,
            )
        ],
    )
    (tmp_path / "preplan_literature_bundle.json").write_text(
        literature.model_dump_json(indent=2), encoding="utf-8"
    )
    (tmp_path / "manuscript_scaffold_bound.md").write_text(
        """# Title

## Abstract
Short abstract.

## Introduction
Short introduction [@direct_2018].

## Methods
Short methods [@strobe_2007].

## Results
Short results.

## Discussion
Short discussion [@direct_2018].

## Limitations
Short limitations.

## Conclusion
Short conclusion.

## Data availability
Available under the database agreement.

## Funding
None.

## Conflicts of interest
None.
""",
        encoding="utf-8",
    )
    (tmp_path / "manuscript_literature_audit.json").write_text(
        json.dumps(
            {
                "status": "pass",
                "exact_citations_present": True,
                "section_cited_keys": {
                    "introduction": ["direct_2018"],
                    "methods": ["strobe_2007"],
                    "discussion": ["direct_2018"],
                },
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "reviewer_report.json").write_text(
        json.dumps(
            {
                "summary": {
                    "aggregated_recommendation": "accept",
                    "counts": {"major": 0, "reject": 0},
                }
            }
        ),
        encoding="utf-8",
    )

    audit = build_scientific_maturity_audit(
        context=context,
        plan=plan,
        run_dir=tmp_path,
        display_suite={"display_suite_complete": True},
        publication_bundle={
            "publication_figure_contract_ready": False,
            "publication_figure_source_data_ready": False,
            "publication_figure_visual_qa_passed": False,
        },
    )

    by_code = {finding.code: finding for finding in audit.findings}
    assert audit.status == "analysis_only"
    assert "RECENT_DIRECT_COMPARATOR_NOT_ESTABLISHED" in by_code
    assert "POST_BASELINE_EXPOSURE_TIMING_NOT_CLOSED" in by_code
    assert "REPEATED_STAY_DEPENDENCE_UNRESOLVED" in by_code
    assert "UNADJUSTED_ASSOCIATION_NOT_ARTICLE_GRADE" in by_code
    assert "ROBUSTNESS_AXES_TOO_NARROW" in by_code
    assert "MANUSCRIPT_CORE_SECTIONS_TOO_THIN" in by_code
    assert "PUBLICATION_FIGURE_CONTRACT_NOT_VERIFIED" in by_code
    assert "PUBLICATION_FIGURE_SOURCE_DATA_NOT_VERIFIED" in by_code
    assert "PUBLICATION_FIGURE_VISUAL_QA_NOT_PASSED" in by_code
    assert audit.facts["publication_figure"] == {
        "bundle_ready": False,
        "contract_ready": False,
        "source_data_ready": False,
        "visual_qa_passed": False,
        "visual_qa_errors": [],
    }
    assert by_code[
        "UNADJUSTED_ASSOCIATION_NOT_ARTICLE_GRADE"
    ].requires_user_authorization
    assert by_code["UNADJUSTED_ASSOCIATION_NOT_ARTICLE_GRADE"].authorization_question
    assert audit.facts["newest_direct_comparator_year"] == 2018
    assert audit.facts["manuscript"]["thin_sections"] == [
        "abstract",
        "introduction",
        "methods",
        "results",
        "discussion",
    ]

    readiness_gates = scientific_maturity_readiness_gates(audit)
    assert readiness_gates["scientific_maturity_article_grade"] is False
    assert scientific_maturity_audit_from_gates(readiness_gates) == audit

    incomplete = dict(readiness_gates)
    incomplete.pop("scientific_maturity_findings")
    with pytest.raises(KeyError, match="scientific_maturity_findings"):
        scientific_maturity_audit_from_gates(incomplete)


def test_primary_figure_absolute_risk_uses_shared_panel_semantics(tmp_path) -> None:
    context = ResearchContext(
        research_question="Is a measured exposure associated with mortality?",
        cohort=CohortDescriptor(
            cohort_name="demo",
            database="synthetic",
            n_patients=100,
            n_stays=100,
        ),
        variables=[],
        primary_exposure="exposure",
        target_outcome="death",
    )
    plan = AnalysisPlan(
        research_question=context.research_question,
        analysis_type="association",
        steps=[],
    )
    figure_dir = tmp_path / "publication_figures"
    figure_dir.mkdir()
    (figure_dir / "easyicu_publication_figure.figure_contract.json").write_text(
        json.dumps(
            {
                "figure_id": "easyicu_publication_figure",
                "core_claim": "Measurement state and observed outcome risk.",
                "panels": [
                    {
                        "panel_id": "A",
                        "title": "Measurement state and observed outcome risk",
                        "role": "data_quality",
                        "claim": "Observed outcome risk is shown for each state.",
                    }
                ],
                "source_data": ["risk.csv"],
            }
        ),
        encoding="utf-8",
    )

    audit = build_scientific_maturity_audit(
        context=context,
        plan=plan,
        run_dir=tmp_path,
        display_suite={"display_suite_complete": True},
        publication_bundle={
            "publication_figure_contract_ready": True,
            "publication_figure_source_data_ready": True,
            "publication_figure_visual_qa_passed": True,
        },
    )

    codes = {finding.code for finding in audit.findings}
    assert "PRIMARY_FIGURE_ABSOLUTE_RISK_CONTEXT_MISSING" not in codes
    assert audit.facts["primary_figure"]["absolute_risk_panel_present"] is True


def test_primary_figure_adjustment_label_uses_registered_runtime_receipt(
    tmp_path,
) -> None:
    context = ResearchContext(
        research_question="Is an exposure associated with mortality?",
        cohort=CohortDescriptor(
            cohort_name="demo",
            database="synthetic",
            n_patients=100,
            n_stays=100,
        ),
        variables=[],
        primary_exposure="exposure",
        target_outcome="death",
    )
    plan = AnalysisPlan(
        research_question=context.research_question,
        analysis_type="association",
        steps=[],
    )
    figure_dir = tmp_path / "publication_figures"
    figure_dir.mkdir()
    (figure_dir / "adjusted.figure_contract.json").write_text(
        json.dumps(
            {
                "figure_id": "adjusted",
                "core_claim": "Adjusted association.",
                "panels": [
                    {
                        "panel_id": "A",
                        "title": "Adjusted effect estimate",
                        "role": "effect",
                        "claim": "Adjusted estimate from the executed model.",
                        "evidence_ids": ["table_primary_effect"],
                    }
                ],
                "source_data": ["effect.csv"],
            }
        ),
        encoding="utf-8",
    )
    evidence_dir = tmp_path / "evidence"
    evidence_dir.mkdir()
    receipt_path = evidence_dir / "log_model_runtime_receipt.json"
    receipt_path.write_text(
        json.dumps(
            {
                "schema_version": "easyicu.model_runtime_receipt/1",
                "adjustment_columns": ["age", "sex"],
            }
        ),
        encoding="utf-8",
    )
    receipt_sha = hashlib.sha256(receipt_path.read_bytes()).hexdigest()
    (evidence_dir / "evidence_authority.json").write_text(
        json.dumps(
            {
                "records": [
                    {
                        "evidence_id": "table_primary_effect",
                        "kind": "table",
                        "producer": "runner",
                        "produced_by_step": "primary_model",
                        "relative_path": "evidence/table_primary_effect.csv",
                        "sha256": "not_needed_for_step_link",
                    },
                    {
                        "evidence_id": "log_model_receipt",
                        "kind": "log",
                        "producer": "runner",
                        "produced_by_step": "primary_model",
                        "relative_path": "evidence/log_model_runtime_receipt.json",
                        "sha256": receipt_sha,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    audit = build_scientific_maturity_audit(
        context=context,
        plan=plan,
        run_dir=tmp_path,
        display_suite={"display_suite_complete": True},
        publication_bundle={
            "publication_figure_contract_ready": True,
            "publication_figure_source_data_ready": True,
            "publication_figure_visual_qa_passed": True,
        },
    )

    codes = {finding.code for finding in audit.findings}
    assert "PRIMARY_FIGURE_ADJUSTMENT_LABEL_CONFLICT" not in codes
    assert audit.facts["primary_figure"]["expected_adjustment_label"] == "adjusted"
    assert audit.facts["primary_figure"]["adjustment_covariates"] == ["age", "sex"]
    assert audit.facts["primary_figure"]["adjustment_authority"] == "runtime_receipt"
    assert audit.facts["primary_covariates"] == ["age", "sex"]
    assert "ADJUSTMENT_SET_NOT_USER_CONFIRMED" in codes
    assert "UNADJUSTED_ASSOCIATION_NOT_ARTICLE_GRADE" not in codes

    receipt_path.write_text("{}", encoding="utf-8")
    tampered = build_scientific_maturity_audit(
        context=context,
        plan=plan,
        run_dir=tmp_path,
        display_suite={"display_suite_complete": True},
        publication_bundle={
            "publication_figure_contract_ready": True,
            "publication_figure_source_data_ready": True,
            "publication_figure_visual_qa_passed": True,
        },
    )
    assert tampered.facts["primary_figure"]["adjustment_authority"] == "not_established"
    assert any(
        finding.code == "PRIMARY_FIGURE_ADJUSTMENT_LABEL_CONFLICT"
        for finding in tampered.findings
    )
