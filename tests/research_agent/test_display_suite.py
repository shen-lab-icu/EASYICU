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


def test_article_audits_ignore_absent_and_superseded_supporting_contracts(
    ra,
    tmp_path: Path,
):
    from easyicu.research_agent.article_contract import (
        summarize_article_contract_coverage,
    )
    from easyicu.research_agent.figure_strategy import (
        summarize_article_figure_strategy_coverage,
    )
    from easyicu.research_agent.pipeline_report import write_readiness_artifacts

    evidence = EvidenceStore(tmp_path)
    _register_table_one(evidence, tmp_path)
    context = _association_context(ra)
    plan = _association_plan(ra, context)

    active_path = _write_contract(
        tmp_path,
        "steps/02_current/outputs",
        "current_quality",
        figure_id="current_quality",
        core_claim="Current missingness and measurement quality are shown.",
        panels=[
            {
                "panel_id": "A",
                "title": "Current measurement availability",
                "role": "data_quality",
                "chart_type": "availability_panel",
                "claim": "Current missingness and measurement availability are shown.",
            }
        ],
    )
    _write_contract(
        tmp_path,
        "steps/03_absent_downstream/outputs",
        "stale_sensitivity",
        figure_id="stale_sensitivity",
        core_claim="Stale sensitivity and robustness results are shown.",
        panels=[
            {
                "panel_id": "B",
                "title": "Stale sensitivity analysis",
                "role": "robustness",
                "chart_type": "specification_grid",
                "claim": "Stale alternative specifications are shown.",
            }
        ],
    )
    _write_contract(
        tmp_path,
        "steps/02_current/outputs",
        "superseded_same_step",
        figure_id="superseded_same_step",
        core_claim="An older contract from the same step is stale.",
        panels=[
            {
                "panel_id": "OLD",
                "title": "Superseded same-step result",
                "role": "prediction",
                "chart_type": "curve",
                "claim": "This older contract must not re-enter the current view.",
            }
        ],
    )
    _write_contract(
        tmp_path,
        "steps/04_retried/outputs",
        "superseded_prediction",
        figure_id="superseded_prediction",
        core_claim="Superseded prediction results are shown.",
        panels=[
            {
                "panel_id": "C",
                "title": "Superseded prediction",
                "role": "prediction",
                "chart_type": "curve",
                "claim": "Superseded model discrimination is shown.",
            }
        ],
    )

    def register_step_table(step_id: str, name: str, description: str) -> str:
        path = tmp_path / f"{name}.csv"
        path.write_text("x\n1\n", encoding="utf-8")
        return evidence.register_file(
            kind="table",
            description=description,
            source_path=path,
            evidence_id=name,
            produced_by_step=step_id,
            producer="test",
            generation_mode="system",
        ).evidence_id

    current_evidence_id = register_step_table(
        "02_current",
        "current_quality_table",
        "Current missingness and measurement quality table.",
    )
    absent_evidence_id = register_step_table(
        "03_absent_downstream",
        "stale_sensitivity_table",
        "Stale sensitivity and robustness table.",
    )
    stale_run_level_path = tmp_path / "stale_run_level_robustness.json"
    stale_run_level_path.write_text('{"status":"ok"}', encoding="utf-8")
    evidence.register_file(
        kind="statistic",
        description="Stale run-level sensitivity and robustness summary.",
        source_path=stale_run_level_path,
        evidence_id="stale_run_level_robustness",
        producer="pipeline",
        generation_mode="system",
    )
    old_retry_evidence_id = register_step_table(
        "04_retried",
        "superseded_prediction_table",
        "Superseded prediction table.",
    )
    records = [
        {
            "step_id": "02_current",
            "status": "ok",
            "evidence_ids": [current_evidence_id],
            "step_summary": {
                "method": "baseline_table",
                "contract_files": [Path(active_path).name],
                "notes": [
                    "Sensitivity was mentioned only as a future caveat, not run."
                ],
            },
        },
        {
            "step_id": "04_retried",
            "status": "ok",
            "evidence_ids": [old_retry_evidence_id],
        },
        {"step_id": "04_retried", "status": "contract_failed"},
    ]

    display = summarize_display_suite_status(
        context=context,
        plan=plan,
        evidence=evidence,
        run_dir=tmp_path,
        publication={"publication_figure_bundle_ready": False},
        per_step_records=records,
    )
    assert display["display_supporting_figure_contract_paths"] == [active_path]
    assert display["display_figure_contract_count"] == 1
    assert display["display_result_figure_contract_count"] == 0
    assert "sensitivity" not in display["display_categories"]
    assert "prediction" not in display["display_categories"]
    active_record_evidence_ids = {
        evidence_id
        for record in records
        if record.get("status") == "ok"
        for evidence_id in (record.get("evidence_ids") or [])
    }
    assert absent_evidence_id not in active_record_evidence_ids

    article = summarize_article_contract_coverage(
        context=context,
        plan=plan,
        evidence_records=evidence.records(),
        per_step_records=records,
        run_dir=tmp_path,
    )
    assert "robustness" not in article["article_artifact_roles"]

    strategy = summarize_article_figure_strategy_coverage(
        context=context,
        run_dir=tmp_path,
        per_step_records=records,
    )
    assert strategy["article_figure_strategy_role_panels"]["robustness"] == []
    assert all(
        "stale_sensitivity" not in panel_id
        for panel_ids in strategy["article_figure_strategy_role_panels"].values()
        for panel_id in panel_ids
    )

    manuscript_path = tmp_path / "manuscript_scaffold_bound.md"
    manuscript_path.write_text(
        "# Manuscript scaffold not generated\n",
        encoding="utf-8",
    )
    gates, _ = write_readiness_artifacts(
        context=context,
        plan=plan,
        findings=[],
        per_step_records=records,
        evidence=evidence,
        run_dir=tmp_path,
        manuscript_path=manuscript_path,
        stop_after_analysis=True,
    )
    assert gates["display_supporting_figure_contract_paths"] == [active_path]
    run_status = json.loads(
        (tmp_path / "run_status.json").read_text(encoding="utf-8")
    )
    assert run_status["gates"]["display_supporting_figure_contract_paths"] == [
        active_path
    ]
    gallery_text = (tmp_path / "figure_gallery.json").read_text(encoding="utf-8")
    assert "stale_sensitivity" not in gallery_text
    assert "superseded_prediction" not in gallery_text
