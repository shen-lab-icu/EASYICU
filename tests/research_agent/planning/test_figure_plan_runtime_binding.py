from __future__ import annotations

import json
from pathlib import Path

from easyicu.research_agent.contracts.figure_plan import PlannedFigurePanelSpec
from easyicu.research_agent.execution.figure_plan_binding import (
    validate_planned_figure_contract_bindings,
    validate_step_planned_figure_contract_binding,
)
from easyicu.research_agent.schema import AnalysisPlan, AnalysisStep


STEP_ID = "06_data_quality_figure"
FIGURE_OUTPUT = "figure:data_quality"
FIGURE_FILE = "data_quality.svg"
CONTRACT_FILE = "data_quality.figure_contract.json"
SOURCE_PRODUCT = "table:measurement_process_audit"


def _plan(*, chart_type: str = "coverage_heatmap") -> AnalysisPlan:
    return AnalysisPlan(
        research_question="Audit source coverage.",
        steps=[
            AnalysisStep(
                step_id=STEP_ID,
                planned_analysis_role="auxiliary",
                intent="Render the prespecified data-quality panel.",
                method="visualization",
                inputs=[SOURCE_PRODUCT],
                expected_outputs=[FIGURE_OUTPUT],
                figure_panels=[
                    PlannedFigurePanelSpec(
                        panel_id="measurement_coverage",
                        figure_output=FIGURE_OUTPUT,
                        article_role="data_quality",
                        chart_type=chart_type,
                        source_products=[SOURCE_PRODUCT],
                    )
                ],
            )
        ],
    )


def _runtime(
    tmp_path: Path,
    *,
    chart_type: str = "coverage_heatmap",
    source_products: list[str] | None = None,
    panel_id: str = "measurement_coverage",
) -> tuple[Path, list[dict[str, object]]]:
    run_dir = tmp_path / "run"
    out_dir = run_dir / "steps" / STEP_ID / "outputs"
    out_dir.mkdir(parents=True)
    (out_dir / FIGURE_FILE).write_text("<svg/>", encoding="utf-8")
    (out_dir / CONTRACT_FILE).write_text(
        json.dumps(
            {
                "figure_id": FIGURE_OUTPUT,
                "core_claim": "Measurement-process coverage is visible.",
                "panels": [
                    {
                        "panel_id": panel_id,
                        "title": "Coverage",
                        "role": "data_quality",
                        "claim": "Coverage across audited variables.",
                        "evidence_ids": ["data_quality_source.csv"],
                        "metadata": {
                            "chart_type": chart_type,
                            "source_products": (
                                source_products
                                if source_products is not None
                                else [SOURCE_PRODUCT]
                            ),
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    records: list[dict[str, object]] = [
        {
            "step_id": STEP_ID,
            "status": "ok",
            "step_summary": {
                "output_files": {FIGURE_OUTPUT: FIGURE_FILE},
                "contract_files": [CONTRACT_FILE],
            },
        }
    ]
    return run_dir, records


def test_exact_role_chart_and_source_products_bind_to_runtime_contract(
    tmp_path: Path,
) -> None:
    run_dir, records = _runtime(tmp_path)

    findings = validate_planned_figure_contract_bindings(
        plan=_plan(),
        run_dir=run_dir,
        per_step_records=records,
    )

    assert findings == []


def test_planned_coverage_heatmap_rejects_runtime_horizontal_bar(
    tmp_path: Path,
) -> None:
    run_dir, records = _runtime(tmp_path, chart_type="horizontal_bar")

    findings = validate_planned_figure_contract_bindings(
        plan=_plan(chart_type="coverage_heatmap"),
        run_dir=run_dir,
        per_step_records=records,
    )

    assert len(findings) == 1
    finding = findings[0]
    assert finding.severity == "error"
    assert finding.validator == "planned_figure_contract_binding"
    assert finding.detail["reason"] == "runtime_panel_contract_mismatch"
    assert finding.detail["planned_panel_signatures"][0]["chart_type"] == (
        "coverage_heatmap"
    )
    assert finding.detail["runtime_panel_signatures"][0]["chart_type"] == (
        "horizontal_bar"
    )


def test_runtime_panel_cannot_borrow_a_different_typed_source_product(
    tmp_path: Path,
) -> None:
    run_dir, records = _runtime(
        tmp_path,
        source_products=["table:missingness_measurement_audit"],
    )

    findings = validate_planned_figure_contract_bindings(
        plan=_plan(),
        run_dir=run_dir,
        per_step_records=records,
    )

    assert len(findings) == 1
    assert findings[0].detail["reason"] == "runtime_panel_contract_mismatch"


def test_panel_id_is_part_of_the_runtime_scientific_binding(tmp_path: Path) -> None:
    run_dir, records = _runtime(tmp_path, panel_id="different_panel")

    findings = validate_planned_figure_contract_bindings(
        plan=_plan(),
        run_dir=run_dir,
        per_step_records=records,
    )

    assert len(findings) == 1
    assert findings[0].detail["reason"] == "runtime_panel_contract_mismatch"
    assert findings[0].detail["planned_panel_signatures"][0]["panel_id"] == (
        "measurement_coverage"
    )
    assert findings[0].detail["runtime_panel_signatures"][0]["panel_id"] == (
        "different_panel"
    )


def test_single_step_validator_runs_before_run_level_article_audit(
    tmp_path: Path,
) -> None:
    run_dir, records = _runtime(tmp_path, chart_type="horizontal_bar")
    step = _plan().steps[0]

    findings = validate_step_planned_figure_contract_binding(
        step=step,
        out_dir=run_dir / "steps" / STEP_ID / "outputs",
        step_summary=records[0]["step_summary"],
    )

    assert len(findings) == 1
    assert findings[0].detail["reason"] == "runtime_panel_contract_mismatch"
