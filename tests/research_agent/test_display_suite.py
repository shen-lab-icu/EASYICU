from __future__ import annotations

import json
from pathlib import Path

from easyicu.research_agent.display_suite import summarize_display_suite_status
from easyicu.research_agent.evidence import EvidenceStore


def _write_contract(
    root: Path,
    relative_dir: str,
    stem: str,
    *,
    figure_id: str,
    core_claim: str,
    panels: list[dict[str, str]],
) -> str:
    out = root / relative_dir
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{stem}.figure_contract.json").write_text(
        json.dumps(
            {
                "figure_id": figure_id,
                "core_claim": core_claim,
                "panels": panels,
            }
        ),
        encoding="utf-8",
    )
    return f"{relative_dir}/{stem}.figure_contract.json"


def _register_table_one(evidence: EvidenceStore, tmp_path: Path) -> None:
    table_path = tmp_path / "table_one.csv"
    table_path.write_text("variable,value\nage,64\n", encoding="utf-8")
    evidence.register_file(
        kind="table",
        description="Table 1 baseline cohort characteristics.",
        source_path=table_path,
        evidence_id="table_table_one",
        producer="test",
        generation_mode="system",
    )


def _association_context(ra):
    return ra.ResearchContext(
        research_question="Estimate whether an exposure is associated with mortality.",
        cohort=ra.CohortDescriptor(
            cohort_name="demo",
            database="synthetic",
            n_stays=10,
            n_patients=10,
        ),
        variables=[],
        target_outcome="death",
    )


def _association_plan(ra, context):
    return ra.AnalysisPlan(
        research_question=context.research_question,
        steps=[
            ra.AnalysisStep(
                step_id="01_table_one",
                intent="Summarise baseline characteristics.",
                expected_outputs=["table:table_one"],
            ),
            ra.AnalysisStep(
                step_id="02_model",
                intent="Fit adjusted association model.",
                expected_outputs=["table:adjusted_association"],
            ),
        ],
    )


def test_display_suite_requires_article_grade_primary_not_only_supporting(
    ra,
    tmp_path: Path,
):
    evidence = EvidenceStore(tmp_path)
    _register_table_one(evidence, tmp_path)
    context = _association_context(ra)
    plan = _association_plan(ra, context)
    primary = _write_contract(
        tmp_path,
        "publication_figures",
        "easyicu_publication_figure",
        figure_id="easyicu_publication_figure",
        core_claim="Adjusted association effect is shown.",
        panels=[
            {
                "panel_id": "A",
                "title": "Adjusted association",
                "role": "relationship",
                "chart_type": "forest",
                "claim": "The adjusted odds ratio is shown.",
            }
        ],
    )
    supporting = _write_contract(
        tmp_path,
        "steps/03_supporting/outputs",
        "supporting_context",
        figure_id="supporting_context",
        core_claim="Supporting absolute risk, data quality, and sensitivity context.",
        panels=[
            {
                "panel_id": "B",
                "title": "Exposure prevalence and absolute outcome risk",
                "role": "descriptive_result",
                "chart_type": "dot_interval_absolute_risk",
                "claim": "Exposure prevalence and absolute outcome risk are shown.",
            },
            {
                "panel_id": "C",
                "title": "Missingness and measurement availability",
                "role": "data_quality",
                "chart_type": "availability_panel",
                "claim": "Missingness and measurement availability are shown.",
            },
            {
                "panel_id": "D",
                "title": "Sensitivity and denominator audit",
                "role": "robustness",
                "chart_type": "specification_grid",
                "claim": "Sensitivity and denominator context are shown.",
            },
        ],
    )

    status = summarize_display_suite_status(
        context=context,
        plan=plan,
        evidence=evidence,
        run_dir=tmp_path,
        publication={"publication_figure_bundle_ready": True},
    )

    assert status["display_suite_complete"] is False
    assert status["display_primary_publication_contract_paths"] == [primary]
    assert status["display_supporting_figure_contract_paths"] == [supporting]
    assert status["display_contract_panel_count"] == 4
    assert status["display_primary_publication_panel_count"] == 1
    assert status["display_supporting_panel_count"] == 3
    assert status["display_absolute_risk_visual_present"] is True
    assert status["display_primary_publication_absolute_risk_visual_present"] is False
    assert status["display_supporting_absolute_risk_visual_present"] is True
    assert any(
        "Primary publication figure exposes fewer" in err
        for err in status["display_suite_errors"]
    )
    assert any(
        "Primary publication figure lacks panel-role" in err
        for err in status["display_suite_errors"]
    )
    assert any(
        "Primary association figure lacks a visual prevalence" in err
        for err in status["display_suite_errors"]
    )


def test_display_suite_accepts_complete_primary_article_display(
    ra,
    tmp_path: Path,
):
    evidence = EvidenceStore(tmp_path)
    _register_table_one(evidence, tmp_path)
    context = _association_context(ra)
    plan = _association_plan(ra, context)
    _write_contract(
        tmp_path,
        "publication_figures",
        "easyicu_publication_figure",
        figure_id="easyicu_publication_figure",
        core_claim="Absolute risk, primary effect, data quality, and sensitivity audit are shown.",
        panels=[
            {
                "panel_id": "A",
                "title": "Absolute outcome risk",
                "role": "descriptive_result",
                "chart_type": "dot_interval_absolute_risk",
                "claim": "Exposure prevalence and absolute outcome risk are shown before adjusted estimates.",
            },
            {
                "panel_id": "B",
                "title": "Adjusted odds-ratio estimate",
                "role": "relationship",
                "chart_type": "forest",
                "claim": "The primary effect estimate is drawn from source data.",
            },
            {
                "panel_id": "C",
                "title": "Missingness and measurement availability",
                "role": "data_quality",
                "chart_type": "availability_panel",
                "claim": "Missingness and measurement availability are shown.",
            },
            {
                "panel_id": "D",
                "title": "Sensitivity and denominator audit",
                "role": "robustness",
                "chart_type": "specification_grid",
                "claim": "Robustness and denominator context are shown.",
            },
        ],
    )

    status = summarize_display_suite_status(
        context=context,
        plan=plan,
        evidence=evidence,
        run_dir=tmp_path,
        publication={"publication_figure_bundle_ready": True},
    )

    assert status["display_suite_complete"] is True
    assert status["display_primary_publication_panel_count"] == 4
    assert status["display_primary_publication_absolute_risk_visual_present"] is True
    assert status["display_primary_publication_chart_types"] == [
        "availability_panel",
        "dot_interval_absolute_risk",
        "forest",
        "specification_grid",
    ]
    assert status["display_suite_errors"] == []
