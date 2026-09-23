from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Sequence

import pytest

from easyicu.research_agent.contracts.figure_plan import (
    PlannedFigurePanelSpec,
    landmark_association_composite_panels,
    separable_display_panel_ids,
)
from easyicu.research_agent.execution.figure_plan_binding import (
    validate_planned_figure_contract_bindings,
    validate_step_planned_figure_contract_binding,
)
from easyicu.research_agent.planning.figure_plan_shaping import (
    apply_article_figure_strategy_placements,
)
from easyicu.research_agent.schema import AnalysisPlan, AnalysisStep


STEP_ID = "06_data_quality_figure"
FIGURE_OUTPUT = "figure:data_quality"
FIGURE_FILE = "data_quality.svg"
CONTRACT_FILE = "data_quality.figure_contract.json"
SOURCE_PRODUCT = "table:measurement_process_audit"


def _plan(
    *,
    chart_type: str = "coverage_heatmap",
    policy_alternative_chart_types: Sequence[str] = (),
) -> AnalysisPlan:
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
                        policy_alternative_chart_types=list(
                            policy_alternative_chart_types
                        ),
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


def test_a_skipped_dependency_figure_step_names_the_blocking_producer(
    tmp_path: Path,
) -> None:
    """The error stays fail-closed but carries the cause the gate recorded.

    A planned panel with no bound summary is still an error; when the step was
    skipped because its producer failed, the finding must name that producer
    and its status instead of forcing a cross-reference to the dependency-gate
    warning.
    """

    run_dir = tmp_path / "run"
    (run_dir / "steps" / STEP_ID / "outputs").mkdir(parents=True)
    records: list[dict[str, object]] = [
        {"step_id": "04_concept_audit", "status": "concept_audit_blocked"},
        {
            "step_id": STEP_ID,
            "status": "skipped_dependency_failed",
            "dependency_step_id": "04_concept_audit",
            "diagnostic_only": True,
        },
    ]

    findings = validate_planned_figure_contract_bindings(
        plan=_plan(),
        run_dir=run_dir,
        per_step_records=records,
    )

    assert len(findings) == 1
    finding = findings[0]
    assert finding.severity == "error"
    assert finding.detail["reason"] == "figure_step_skipped_dependency_failed"
    assert finding.detail["dependency_step_id"] == "04_concept_audit"
    assert finding.detail["dependency_status"] == "concept_audit_blocked"
    assert finding.detail["diagnostic_only"] is True


def test_a_failed_figure_step_keeps_the_generic_missing_summary_reason(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    (run_dir / "steps" / STEP_ID / "outputs").mkdir(parents=True)
    records: list[dict[str, object]] = [
        {"step_id": STEP_ID, "status": "execution_failed"},
    ]

    findings = validate_planned_figure_contract_bindings(
        plan=_plan(),
        run_dir=run_dir,
        per_step_records=records,
    )

    assert len(findings) == 1
    assert findings[0].detail["reason"] == (
        "figure_step_has_no_current_successful_summary"
    )


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


def test_a_declared_policy_alternative_chart_binds_but_an_undeclared_one_does_not(
    tmp_path: Path,
) -> None:
    run_dir, records = _runtime(tmp_path, chart_type="missingness_matrix")

    accepted = validate_planned_figure_contract_bindings(
        plan=_plan(
            chart_type="coverage_heatmap",
            policy_alternative_chart_types=["missingness_matrix"],
        ),
        run_dir=run_dir,
        per_step_records=records,
    )
    assert accepted == []

    rejected = validate_planned_figure_contract_bindings(
        plan=_plan(
            chart_type="coverage_heatmap",
            policy_alternative_chart_types=["leakage_audit"],
        ),
        run_dir=run_dir,
        per_step_records=records,
    )
    assert [item.detail["reason"] for item in rejected] == [
        "runtime_panel_contract_mismatch"
    ]


def test_policy_alternatives_must_be_distinct_chart_grammars() -> None:
    with pytest.raises(ValueError, match="must not repeat chart_type"):
        PlannedFigurePanelSpec(
            panel_id="a",
            figure_output=FIGURE_OUTPUT,
            article_role="data_quality",
            chart_type="coverage_heatmap",
            source_products=[SOURCE_PRODUCT],
            policy_alternative_chart_types=["coverage_heatmap"],
        )
    with pytest.raises(ValueError, match="must be unique"):
        PlannedFigurePanelSpec(
            panel_id="a",
            figure_output=FIGURE_OUTPUT,
            article_role="data_quality",
            chart_type="coverage_heatmap",
            source_products=[SOURCE_PRODUCT],
            policy_alternative_chart_types=["missingness_matrix", "missingness_matrix"],
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


@pytest.mark.parametrize("level", ["step", "final"])
def test_supplementary_placement_does_not_waive_chart_or_source_binding(
    tmp_path, level
):
    run_dir, records = _runtime(
        tmp_path, chart_type="horizontal_bar", source_products=["table:other_source"]
    )
    plan = _plan()
    plan.steps[0].figure_panels = [
        plan.steps[0].figure_panels[0].model_copy(update={"placement": "supplementary"})
    ]
    if level == "step":
        findings = validate_step_planned_figure_contract_binding(
            step=plan.steps[0],
            out_dir=run_dir / "steps" / STEP_ID / "outputs",
            step_summary=records[0]["step_summary"],
        )
    else:
        findings = validate_planned_figure_contract_bindings(
            plan=plan, run_dir=run_dir, per_step_records=records
        )
    assert [f.detail["reason"] for f in findings] == ["runtime_panel_contract_mismatch"]


# Dev9 E2 measured the cost of an unbounded placement projection.  The
# data-quality figure is one exported composite -- two planned panels, one PNG,
# one contract -- yet the article strategy kept the availability panel in the
# main article while the host audit rule moved process coverage out of it.  The
# runtime binding gate then grouped the planned panels by placement and looked
# for a second artifact that no renderer could produce, so a step that rendered
# correctly fail-closed the entire run after every scientific step had already
# spent its budget.  These tests pin the bound in the owner that creates the
# promise, and keep the gate evidence that the bound is load-bearing.

COMPOSITE_STEP_ID = "12_data_quality_figure"
COMPOSITE_OUTPUT = "figure:data_quality"
COMPOSITE_FILE = "data_quality.svg"
COMPOSITE_CONTRACT = "data_quality.figure_contract.json"
AVAILABILITY_SOURCE = "table:measurement_audit"
PROCESS_SOURCE = "table:measurement_process"
LANDMARK_AUDIT_PROFILE = (
    "table:generic_landmark_rcs_curve",
    "table:generic_adjusted_absolute_risk",
    "table:robustness_grid_exposure_contrasts",
    "table:robustness_summary",
    "table:measurement_process",
)


def _composite_panels() -> list[PlannedFigurePanelSpec]:
    return [
        PlannedFigurePanelSpec(
            panel_id="source_availability",
            figure_output=COMPOSITE_OUTPUT,
            article_role="data_quality",
            chart_type="availability_panel",
            source_products=[AVAILABILITY_SOURCE],
        ),
        PlannedFigurePanelSpec(
            panel_id="measurement_process_coverage",
            figure_output=COMPOSITE_OUTPUT,
            article_role="data_quality",
            chart_type="coverage_heatmap",
            source_products=[PROCESS_SOURCE],
        ),
    ]


def _composite_plan(placements: Sequence[str]) -> AnalysisPlan:
    step = AnalysisStep(
        step_id=COMPOSITE_STEP_ID,
        planned_analysis_role="auxiliary",
        intent="Render the prespecified data-quality composite.",
        method="visualization",
        inputs=[AVAILABILITY_SOURCE, PROCESS_SOURCE],
        expected_outputs=[COMPOSITE_OUTPUT],
        figure_panels=[
            panel.model_copy(update={"placement": placement})
            for panel, placement in zip(_composite_panels(), placements)
        ],
    )
    return AnalysisPlan(research_question="Audit source coverage.", steps=[step])


def _composite_runtime(
    tmp_path: Path,
) -> tuple[Path, list[dict[str, object]]]:
    run_dir = tmp_path / "run"
    out_dir = run_dir / "steps" / COMPOSITE_STEP_ID / "outputs"
    out_dir.mkdir(parents=True)
    (out_dir / COMPOSITE_FILE).write_text("<svg/>", encoding="utf-8")
    (out_dir / COMPOSITE_CONTRACT).write_text(
        json.dumps(
            {
                "figure_id": COMPOSITE_OUTPUT,
                "core_claim": (
                    "Source availability and measurement-process coverage are "
                    "rendered from two digest-verified parent audit tables."
                ),
                "panels": [
                    {
                        "panel_id": "source_availability",
                        "title": "Source availability",
                        "role": "data_quality",
                        "claim": "Stays with no recorded source value.",
                        "evidence_ids": ["data_quality_missingness_source_data.csv"],
                        "metadata": {
                            "chart_type": "availability_panel",
                            "source_products": [AVAILABILITY_SOURCE],
                            "placement": "supplementary",
                        },
                    },
                    {
                        "panel_id": "measurement_process_coverage",
                        "title": "Measurement-process coverage",
                        "role": "data_quality",
                        "claim": "Measured share per audited variable.",
                        "evidence_ids": [
                            "data_quality_measurement_process_source_data.csv"
                        ],
                        "metadata": {
                            "chart_type": "coverage_heatmap",
                            "source_products": [PROCESS_SOURCE],
                            "placement": "supplementary",
                        },
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    records: list[dict[str, object]] = [
        {
            "step_id": COMPOSITE_STEP_ID,
            "status": "ok",
            "step_summary": {
                "output_files": {COMPOSITE_OUTPUT: COMPOSITE_FILE},
                "contract_files": [COMPOSITE_CONTRACT],
            },
        }
    ]
    return run_dir, records


def _data_quality_main_strategy() -> SimpleNamespace:
    return SimpleNamespace(
        role_strategies=[SimpleNamespace(role="data_quality", placement="main")]
    )


def test_audit_demotion_moves_the_whole_composite_instead_of_one_panel() -> None:
    shaped = apply_article_figure_strategy_placements(
        plan=_composite_plan(["main", "main"]),
        strategy=_data_quality_main_strategy(),
    )

    assert [
        (panel.panel_id, panel.placement)
        for panel in shaped.steps[0].figure_panels
    ] == [
        ("source_availability", "supplementary"),
        ("measurement_process_coverage", "supplementary"),
    ]


def test_projected_composite_binds_to_its_single_exported_surface(
    tmp_path: Path,
) -> None:
    shaped = apply_article_figure_strategy_placements(
        plan=_composite_plan(["main", "main"]),
        strategy=_data_quality_main_strategy(),
    )
    run_dir, records = _composite_runtime(tmp_path)

    assert (
        validate_planned_figure_contract_bindings(
            plan=shaped, run_dir=run_dir, per_step_records=records
        )
        == []
    )


def test_hand_split_composite_is_the_shape_no_renderer_can_bind(
    tmp_path: Path,
) -> None:
    """Keep the gate failing on a split, so the bound above stays necessary."""

    run_dir, records = _composite_runtime(tmp_path)

    findings = validate_planned_figure_contract_bindings(
        plan=_composite_plan(["main", "supplementary"]),
        run_dir=run_dir,
        per_step_records=records,
    )

    assert sorted(str(finding.detail["reason"]) for finding in findings) == [
        "runtime_figure_output_is_unbound",
        "runtime_panel_contract_mismatch",
    ]


def test_separable_display_is_declared_by_the_renderer_contract() -> None:
    templates = landmark_association_composite_panels(LANDMARK_AUDIT_PROFILE)

    assert separable_display_panel_ids(
        source_products=LANDMARK_AUDIT_PROFILE,
        panel_ids=[template.panel_id for template in templates],
    ) == {"robustness_summary", "measurement_process"}
    assert separable_display_panel_ids(
        source_products=LANDMARK_AUDIT_PROFILE,
        panel_ids=[template.panel_id for template in templates][:3],
    ) == frozenset()
    assert separable_display_panel_ids(
        source_products=[AVAILABILITY_SOURCE, PROCESS_SOURCE],
        panel_ids=["source_availability", "measurement_process_coverage"],
    ) == frozenset()
