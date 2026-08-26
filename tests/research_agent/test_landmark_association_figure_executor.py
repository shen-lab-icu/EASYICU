from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

from easyicu.research_agent.execution.runners.landmark_association_figure_executor import (
    landmark_association_figure_executor_owns_step,
    run_landmark_association_figure,
)
from easyicu.research_agent.contracts.figure_plan import (
    landmark_association_composite_panels,
)
from easyicu.research_agent.execution.figure_plan_binding import (
    validate_step_planned_figure_contract_binding,
)
from easyicu.research_agent.execution.runners.selection import select_standard_executor
from easyicu.research_agent.planning.figure_plan_shaping import (
    close_empty_deterministic_figure_contracts,
)
from easyicu.research_agent.schema import AnalysisPlan, AnalysisStep


INPUTS = (
    "table:generic_landmark_rcs_curve",
    "table:absolute_risk_context",
    "table:robustness_summary",
    "table:measurement_process",
)


def _frames() -> dict[str, pd.DataFrame]:
    return {
        INPUTS[0]: pd.DataFrame(
            {
                "biomarker_mg_dl": [1.0, 3.0, 5.0],
                "reference_biomarker_mg_dl": [2.1, 2.1, 2.1],
                "adjusted_odds_ratio": [0.76, 1.10, 1.96],
                "ci_low": [0.72, 1.00, 1.89],
                "ci_high": [0.81, 1.21, 2.03],
            }
        ),
        INPUTS[1]: pd.DataFrame(
            {
                "label": ["Observed", "Observed", "Distribution"],
                "estimate_type": ["prevalence", "outcome_risk", "distribution"],
                "estimate": [0.54, 0.14, None],
                "ci_low": [0.53, 0.13, None],
                "ci_high": [0.55, 0.15, None],
            }
        ),
        INPUTS[2]: pd.DataFrame(
            {
                "axis": ["primary", "functional form"],
                "total_specs": [1, 1],
                "converged_specs": [1, 1],
                "range_low": [1.89, 1.24],
                "range_high": [2.03, 1.27],
            }
        ),
        INPUTS[3]: pd.DataFrame(
            {"concept": ["exposure"], "n_total": [100], "measured_one_n": [54]}
        ),
    }


def _binding(key: str, frame: pd.DataFrame, path: Path) -> dict[str, object]:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    product = key.partition(":")[2]
    return {
        "declared_kind": "table",
        "evidence_kind": "table",
        "product": product,
        "relative_path": path.name,
        "sha256": digest,
        "evidence_id": f"evidence_{product}",
        "product_contract": {"columns": list(frame.columns), "row_count": len(frame)},
        "consumption_contract": {
            "input_key": key,
            "mode": "all_rows",
            "artifact_sha256": digest,
        },
        "identity_row": {
            "input_key": key,
            "declared_kind": "table",
            "product": product,
            "evidence_id": f"evidence_{product}",
            "sha256": digest,
        },
    }


def test_empty_visualization_contract_closes_and_selects_owner(tmp_path: Path) -> None:
    draft = AnalysisStep(
        step_id="display_suite",
        planned_analysis_role="auxiliary",
        intent="Render the four typed sources.",
        method="visualization",
        inputs=list(INPUTS),
    )
    plan, findings = close_empty_deterministic_figure_contracts(
        plan=AnalysisPlan(research_question="Association?", steps=[draft])
    )
    step = plan.steps[0]
    assert findings
    assert step.expected_outputs == ["figure:display_suite"]
    assert {item.input_key for item in step.input_consumption_contracts} == set(INPUTS)

    bindings = {}
    for key, frame in _frames().items():
        path = tmp_path / f"{key.partition(':')[2]}.csv"
        frame.to_csv(path, index=False)
        bindings[key] = _binding(key, frame, path)
    assert landmark_association_figure_executor_owns_step(
        step, resolved_bindings=bindings
    )
    selection = select_standard_executor(step, plan=plan, resolved_bindings=bindings)
    assert selection is not None
    assert selection.analysis_kind == "landmark_association_composite_figure"


def test_composite_owner_accepts_a_case_neutral_curve_product(tmp_path: Path) -> None:
    generic_curve = "table:continuous_exposure_curve"
    generic_inputs = (generic_curve, *INPUTS[1:])
    draft = AnalysisStep(
        step_id="display_suite",
        planned_analysis_role="auxiliary",
        intent="Render the four typed sources.",
        method="visualization",
        inputs=list(generic_inputs),
        expected_outputs=["figure:display_suite"],
        input_consumption_contracts=[
            {"input_key": key, "mode": "all_rows"} for key in generic_inputs
        ],
    )
    frames = _frames()
    frames[generic_curve] = frames.pop(INPUTS[0])
    bindings = {}
    for key, frame in frames.items():
        path = tmp_path / f"{key.partition(':')[2]}.csv"
        frame.to_csv(path, index=False)
        bindings[key] = _binding(key, frame, path)

    assert landmark_association_figure_executor_owns_step(
        draft,
        resolved_bindings=bindings,
    )


def test_renderer_exports_four_source_bound_panels(tmp_path: Path) -> None:
    bindings = {}
    for key, frame in _frames().items():
        path = tmp_path / f"{key.partition(':')[2]}.csv"
        frame.to_csv(path, index=False)
        bindings[key] = _binding(key, frame, path)
    summary = run_landmark_association_figure(
        out_dir=tmp_path / "outputs",
        run_dir=tmp_path,
        resolved_inputs={"step_id": "display_suite", "inputs": bindings},
        step_id="display_suite",
        figure_product="display_suite",
        input_keys=INPUTS,
    )
    assert summary["status"] == "ok"
    assert len(summary["source_data_files"]) == 4
    source = pd.read_csv(tmp_path / "outputs" / summary["source_data_files"][0])
    assert source["source_row_index"].tolist() == [0, 1, 2]
    assert source["source_table"].nunique() == 1
    assert (tmp_path / "outputs" / "display_suite.figure_contract.json").is_file()
    svg = (tmp_path / "outputs" / "display_suite.svg").read_text(encoding="utf-8")
    assert "Biomarker (mg/dL; reference 2.1)" in svg
    assert "Cohort share" in svg
    assert "Observed outcome risk" in svg
    contract = pd.read_json(
        tmp_path / "outputs" / "display_suite.figure_contract.json", typ="series"
    )
    assert contract["panels"][0]["metadata"]["estimate_geometry"] == (
        "continuous_fitted_curve_with_95ci"
    )
    assert [panel["role"] for panel in contract["panels"]] == [
        "primary_estimand",
        "descriptive_result",
        "robustness",
        "data_quality",
    ]
    assert [panel["metadata"]["chart_type"] for panel in contract["panels"]] == [
        "marginal_effect_panel",
        "dot_interval_absolute_risk",
        "sensitivity_coverage_matrix",
        "availability_panel",
    ]
    robustness_panel = contract["panels"][2]
    assert robustness_panel["metadata"]["effect_comparison_authorized"] is False
    assert robustness_panel["metadata"]["reason_code"] == (
        "ROBUSTNESS_EFFECT_COMPARABILITY_UNRESOLVED"
    )
    step = AnalysisStep(
        step_id="display_suite",
        planned_analysis_role="auxiliary",
        intent="Render four typed sources.",
        inputs=list(INPUTS),
        expected_outputs=["figure:display_suite"],
        method="visualization",
        figure_panels=[
            panel.bind(figure_output="figure:display_suite")
            for panel in landmark_association_composite_panels(INPUTS)
        ],
    )
    assert (
        validate_step_planned_figure_contract_binding(
            step=step,
            out_dir=tmp_path / "outputs",
            step_summary=summary,
        )
        == []
    )


def test_renderer_moves_routine_measurement_panel_out_of_main_figure(
    tmp_path: Path,
) -> None:
    bindings = {}
    for key, frame in _frames().items():
        path = tmp_path / f"{key.partition(':')[2]}.csv"
        frame.to_csv(path, index=False)
        bindings[key] = _binding(key, frame, path)

    summary = run_landmark_association_figure(
        out_dir=tmp_path / "outputs",
        run_dir=tmp_path,
        resolved_inputs={"step_id": "display_suite", "inputs": bindings},
        step_id="display_suite",
        figure_product="display_suite",
        input_keys=INPUTS,
        panel_placements={"measurement_process": "supplementary"},
    )

    contract = pd.read_json(
        tmp_path / "outputs" / "display_suite.figure_contract.json", typ="series"
    )
    assert [panel["role"] for panel in contract["panels"]] == [
        "primary_estimand",
        "descriptive_result",
        "robustness",
    ]
    assert summary["supplementary_panel_ids"] == ["measurement_process"]
    assert len(summary["source_data_files"]) == 4
    step = AnalysisStep(
        step_id="display_suite",
        planned_analysis_role="auxiliary",
        intent="Render typed sources with routine measurement detail in supplement.",
        inputs=list(INPUTS),
        expected_outputs=["figure:display_suite"],
        method="visualization",
        figure_panels=[
            panel.bind(figure_output="figure:display_suite").model_copy(
                update={
                    "placement": (
                        "supplementary"
                        if panel.panel_id == "measurement_process"
                        else "main"
                    )
                }
            )
            for panel in landmark_association_composite_panels(INPUTS)
        ],
    )
    assert (
        validate_step_planned_figure_contract_binding(
            step=step,
            out_dir=tmp_path / "outputs",
            step_summary=summary,
        )
        == []
    )
