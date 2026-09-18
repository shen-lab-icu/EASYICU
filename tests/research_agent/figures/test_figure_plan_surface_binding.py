"""One physical display cannot satisfy two different publication placements."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from easyicu.research_agent.contracts.figure_plan import (
    PlannedFigurePanelSpec,
    landmark_association_composite_panels,
)
from easyicu.research_agent.execution.figure_plan_binding import (
    validate_step_planned_figure_contract_binding,
)
from easyicu.research_agent.execution.runners.landmark_association_figure_executor import (
    run_landmark_association_figure,
)
from easyicu.research_agent.schema import AnalysisStep


def _panel(panel_id: str, output: str, placement: str) -> PlannedFigurePanelSpec:
    return PlannedFigurePanelSpec(
        panel_id=panel_id, figure_output=output, placement=placement,
        article_role="data_quality", chart_type="availability_panel",
        source_products=["table:measurement_process"],
    )


def _step(panels: list[PlannedFigurePanelSpec]) -> AnalysisStep:
    return AnalysisStep(
        step_id="display", planned_analysis_role="auxiliary", method="visualization",
        intent="Render the declared display surfaces.",
        inputs=list(dict.fromkeys(source for panel in panels for source in panel.source_products)),
        expected_outputs=list(dict.fromkeys(panel.figure_output for panel in panels)),
        figure_panels=panels,
    )


def _surface(tmp_path: Path, panels: list[PlannedFigurePanelSpec]) -> dict:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(panels), squeeze=False)
    try:
        for ax, panel in zip(axes[0], panels):
            ax.barh([0], [1])
            ax.set_title(panel.panel_id)
        fig.savefig(tmp_path / "display.svg")
        fig.savefig(tmp_path / "display.png")
    finally:
        plt.close(fig)
    (tmp_path / "display.figure_contract.json").write_text(json.dumps({
        "figure_id": "figure:display",
        "panels": [
            {"panel_id": panel.panel_id, "role": panel.article_role,
             "metadata": {"chart_type": panel.chart_type,
                          "source_products": panel.source_products}}
            for panel in panels
        ],
    }))
    return {"contract_files": ["display.figure_contract.json"]}


@pytest.mark.parametrize("different_output", [False, True])
@pytest.mark.parametrize("supplementary_format", ["svg", "png"])
def test_same_surface_cannot_be_split_by_panel_selectors_across_placements(
    tmp_path: Path, different_output: bool, supplementary_format: str,
) -> None:
    supplementary_output = "figure:audit" if different_output else "figure:display"
    panels = [
        _panel("result", "figure:display", "main"),
        _panel("audit", supplementary_output, "supplementary"),
    ]
    summary = _surface(tmp_path, panels)
    summary.update({
        "output_files": {"figure:display": "display.svg"},
        "supplementary_output_files": {
            supplementary_output: f"display.{supplementary_format}"
        },
        "planner_product_slot_bindings": {"figure:display": {"panel_ids": ["result"]}},
        "supplementary_product_slot_bindings": {
            supplementary_output: {"panel_ids": ["audit"]}
        },
    })

    findings = validate_step_planned_figure_contract_binding(
        step=_step(panels), out_dir=tmp_path, step_summary=summary,
    )

    assert [finding.detail["reason"] for finding in findings] == [
        "runtime_figure_surface_placement_conflict"
    ]
    assert findings[0].detail["contract_file"] == "display.figure_contract.json"
    assert findings[0].detail["placements"] == ["main", "supplementary"]


def test_same_placement_allows_distinct_slots_on_one_surface(tmp_path: Path) -> None:
    panels = [
        _panel("first", "figure:first", "main"),
        _panel("second", "figure:second", "main"),
    ]
    summary = _surface(tmp_path, panels)
    summary.update({
        "output_files": {"figure:first": "display.svg", "figure:second": "display.png"},
        "planner_product_slot_bindings": {
            "figure:first": {"panel_ids": ["first"]},
            "figure:second": {"panel_ids": ["second"]},
        },
    })

    assert validate_step_planned_figure_contract_binding(
        step=_step(panels), out_dir=tmp_path, step_summary=summary,
    ) == []


@pytest.mark.parametrize("placement", ["main", "supplementary"])
def test_real_landmark_renderer_binds_split_or_all_supplementary_surfaces(
    tmp_path: Path, placement: str,
) -> None:
    coordinates = {
        "exposure": ["biomarker_mg_dl"] * 3,
        "biomarker_mg_dl": [1.0, 3.0, 5.0],
        "reference_biomarker_mg_dl": [2.0] * 3,
        "exposure_density_n": [20, 60, 20],
        "exposure_density_fraction": [0.2, 0.6, 0.2],
    }
    frames = {
        "table:biomarker_landmark_rcs_curve": pd.DataFrame({
            **coordinates, "adjusted_odds_ratio": [0.8, 1.1, 2.0],
            "ci_low": [0.7, 1.0, 1.8], "ci_high": [0.9, 1.2, 2.2],
        }),
        "table:biomarker_adjusted_absolute_risk": pd.DataFrame({
            **coordinates, "adjusted_absolute_risk": [0.1, 0.2, 0.3],
            "ci_low": [0.08, 0.18, 0.28], "ci_high": [0.12, 0.22, 0.32],
        }),
        "table:robustness_summary": pd.DataFrame({
            "axis": ["model"], "total_specs": [1], "converged_specs": [1],
        }),
        "table:measurement_process": pd.DataFrame({
            "concept": ["biomarker"], "n_total": [100], "measured_one_n": [80],
        }),
    }
    bindings = {}
    for key, frame in frames.items():
        product = key.partition(":")[2]
        path = tmp_path / f"{product}.csv"
        frame.to_csv(path, index=False)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        bindings[key] = {
            "declared_kind": "table", "evidence_kind": "table", "product": product,
            "relative_path": path.name, "sha256": digest, "evidence_id": product,
            "product_contract": {"columns": list(frame.columns), "row_count": len(frame)},
            "consumption_contract": {
                "input_key": key, "mode": "all_rows", "artifact_sha256": digest,
            },
            "identity_row": {
                "input_key": key, "declared_kind": "table", "product": product,
                "evidence_id": product, "sha256": digest,
            },
        }
    panels = [
        panel.bind(figure_output="figure:display").model_copy(update={
            "placement": "supplementary" if panel.separable_display else placement
        })
        for panel in landmark_association_composite_panels(tuple(frames))
    ]
    step = _step(panels)
    summary = run_landmark_association_figure(
        out_dir=tmp_path / "outputs", run_dir=tmp_path,
        resolved_inputs={"step_id": step.step_id, "inputs": bindings},
        step_id=step.step_id, figure_product="display", input_keys=tuple(frames),
        panel_placements={panel.panel_id: panel.placement for panel in panels},
    )
    if placement == "supplementary":
        assert summary["output_files"] == summary["supplementary_output_files"]
    else:
        assert summary["output_files"] != summary["supplementary_output_files"]
    assert validate_step_planned_figure_contract_binding(
        step=step, out_dir=tmp_path / "outputs", step_summary=summary,
    ) == []
