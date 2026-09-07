from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from easyicu.research_agent.execution.runners.landmark_association_figure_executor import (
    _continuous_exposure_label,
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
    apply_article_figure_strategy_placements,
    close_empty_deterministic_figure_contracts,
)
from easyicu.research_agent.schema import AnalysisPlan, AnalysisStep


INPUTS = (
    "table:generic_landmark_rcs_curve",
    "table:generic_adjusted_absolute_risk",
    "table:robustness_summary",
    "table:measurement_process",
)

LEGACY_INPUTS = (
    "table:absolute_risk_context",
    "table:robustness_matrix",
    "table:robustness_summary",
)


def _frames() -> dict[str, pd.DataFrame]:
    return {
        INPUTS[0]: pd.DataFrame(
            {
                "exposure": ["biomarker_mg_dl"] * 3,
                "biomarker_mg_dl": [1.0, 3.0, 5.0],
                "reference_biomarker_mg_dl": [2.1, 2.1, 2.1],
                "adjusted_odds_ratio": [0.76, 1.10, 1.96],
                "ci_low": [0.72, 1.00, 1.89],
                "ci_high": [0.81, 1.21, 2.03],
                "exposure_density_n": [20, 60, 20],
                "exposure_density_fraction": [0.2, 0.6, 0.2],
            }
        ),
        INPUTS[1]: pd.DataFrame(
            {
                "exposure": ["biomarker_mg_dl"] * 3,
                "biomarker_mg_dl": [1.0, 3.0, 5.0],
                "reference_biomarker_mg_dl": [2.1, 2.1, 2.1],
                "adjusted_absolute_risk": [0.08, 0.12, 0.23],
                "ci_low": [0.07, 0.10, 0.20],
                "ci_high": [0.09, 0.14, 0.26],
                "exposure_density_n": [20, 60, 20],
                "exposure_density_fraction": [0.2, 0.6, 0.2],
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


def test_renderer_exports_two_claim_led_panels_and_four_source_tables(
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
    )
    assert summary["status"] == "ok"
    assert len(summary["source_data_files"]) == 4
    supplementary = summary["supplementary_output_files"]["figure:display_suite"]
    assert (tmp_path / "outputs" / supplementary).is_file()
    supplementary_contract_path = (
        tmp_path / "outputs" / "display_suite_supplementary.figure_contract.json"
    )
    supplementary_contract = json.loads(supplementary_contract_path.read_text())
    assert [panel["panel_id"] for panel in supplementary_contract["panels"]] == [
        "robustness_summary",
        "measurement_process",
    ]
    source = pd.read_csv(tmp_path / "outputs" / summary["source_data_files"][0])
    assert source["source_row_index"].tolist() == [0, 1, 2]
    assert source["source_table"].nunique() == 1
    assert (tmp_path / "outputs" / "display_suite.figure_contract.json").is_file()
    svg = (tmp_path / "outputs" / "display_suite.svg").read_text(encoding="utf-8")
    assert "Biomarker (mg/dL)" in svg
    assert "Reference 2.1" in svg
    assert "Exposure distribution" in svg
    assert "Model-standardised outcome risk (%)" in svg
    assert "Absolute risk" in svg
    contract = pd.read_json(
        tmp_path / "outputs" / "display_suite.figure_contract.json", typ="series"
    )
    assert contract["panels"][0]["metadata"]["estimate_geometry"] == (
        "continuous_fitted_curve_with_95ci"
    )
    assert [panel["role"] for panel in contract["panels"]] == [
        "primary_estimand",
        "descriptive_result",
    ]
    assert [panel["metadata"]["chart_type"] for panel in contract["panels"]] == [
        "marginal_effect_panel",
        "absolute_risk_curve",
    ]
    assert summary["supplementary_panel_ids"] == [
        "measurement_process",
        "robustness_summary",
    ]
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

    supplementary_contract["panels"][1]["metadata"]["chart_type"] = "table"
    supplementary_contract_path.write_text(json.dumps(supplementary_contract))
    findings = validate_step_planned_figure_contract_binding(
        step=step, out_dir=tmp_path / "outputs", step_summary=summary
    )
    assert [f.detail["reason"] for f in findings] == ["runtime_panel_contract_mismatch"]
    supplementary_contract_path.unlink()
    assert validate_step_planned_figure_contract_binding(
        step=step, out_dir=tmp_path / "outputs", step_summary=summary
    )


def test_continuous_exposure_label_preserves_summary_and_clinical_unit() -> None:
    assert _continuous_exposure_label("lact_max") == "Maximum lactate (mmol/L)"
    assert _continuous_exposure_label("biomarker_mg_dl") == "Biomarker (mg/dL)"


def test_approved_legacy_article_step_uses_sealed_renderer_without_plan_rewrite(
    tmp_path: Path,
) -> None:
    frames = {
        LEGACY_INPUTS[0]: pd.DataFrame(
            {
                "label": ["Observed", "Observed", "Not measured", "Not measured"],
                "group_value": ["observed", "observed", "no_source", "no_source"],
                "estimate_type": [
                    "prevalence",
                    "outcome_risk",
                    "prevalence",
                    "outcome_risk",
                ],
                "estimate": [0.54, 0.14, 0.46, 0.09],
                # The first lower bound is a few ulps inside its estimate.  A
                # rendering-only error bar must clip that distance to zero.
                "ci_low": [0.5400000000000001, 0.13, 0.45, 0.08],
                "ci_high": [0.55, 0.15, 0.47, 0.10],
            }
        ),
        LEGACY_INPUTS[1]: pd.DataFrame(
            {
                "spec_id": ["primary", "linear_sensitivity"],
                "axis": ["primary", "functional_form"],
                "converged": [True, True],
            }
        ),
        LEGACY_INPUTS[2]: pd.DataFrame(
            {
                "axis": ["primary", "functional_form"],
                "total_specs": [1, 1],
                "converged_specs": [1, 1],
                "range_low": [1.89, 1.24],
                "range_high": [2.03, 1.27],
            }
        ),
    }
    bindings = {}
    for key, frame in frames.items():
        path = tmp_path / f"{key.partition(':')[2]}.csv"
        frame.to_csv(path, index=False)
        bindings[key] = _binding(key, frame, path)
    step = AnalysisStep(
        step_id="assemble_article_displays",
        planned_analysis_role="auxiliary",
        intent="Assemble the already-reviewed article display.",
        method="visualization",
        inputs=list(LEGACY_INPUTS),
        expected_outputs=["figure:assemble_article_displays"],
        input_consumption_contracts=[
            {"input_key": key, "mode": "all_rows"} for key in LEGACY_INPUTS
        ],
    )
    plan = AnalysisPlan(research_question="Association?", steps=[step])

    assert landmark_association_figure_executor_owns_step(
        step, resolved_bindings=bindings
    )
    selection = select_standard_executor(step, plan=plan, resolved_bindings=bindings)
    assert selection is not None
    assert selection.analysis_kind == "landmark_association_composite_figure"
    summary = run_landmark_association_figure(
        out_dir=tmp_path / "outputs",
        run_dir=tmp_path,
        resolved_inputs={"step_id": step.step_id, "inputs": bindings},
        step_id=step.step_id,
        figure_product="assemble_article_displays",
        input_keys=LEGACY_INPUTS,
    )
    assert summary["status"] == "ok"
    assert summary["method"] == "deterministic_legacy_landmark_article_figure"
    assert len(summary["source_data_files"]) == 3
    assert (tmp_path / "outputs" / "assemble_article_displays.png").is_file()


def test_renderer_keeps_routine_audit_panels_out_of_main_figure(
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
    ]
    assert summary["supplementary_panel_ids"] == [
        "measurement_process",
        "robustness_summary",
    ]
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
                        "main"
                        if panel.panel_id
                        in {"association_curve", "absolute_risk_curve"}
                        else "supplementary"
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


def test_audit_only_coverage_is_supplementary_even_when_robustness_is_main() -> None:
    step = AnalysisStep(
        step_id="display_suite",
        planned_analysis_role="auxiliary",
        intent="Render the typed article display suite.",
        inputs=list(INPUTS),
        expected_outputs=["figure:display_suite"],
        method="visualization",
        figure_panels=[
            panel.bind(figure_output="figure:display_suite")
            for panel in landmark_association_composite_panels(INPUTS)
        ],
    )
    strategy = SimpleNamespace(
        role_strategies=[
            SimpleNamespace(role="primary_estimand", placement="main"),
            SimpleNamespace(role="descriptive_result", placement="main"),
            SimpleNamespace(role="robustness", placement="main"),
            SimpleNamespace(role="data_quality", placement="supplementary"),
        ]
    )

    shaped = apply_article_figure_strategy_placements(
        plan=AnalysisPlan(research_question="Association?", steps=[step]),
        strategy=strategy,
    )

    placements = {
        panel.panel_id: panel.placement for panel in shaped.steps[0].figure_panels
    }
    assert placements == {
        "association_curve": "main",
        "absolute_risk_curve": "main",
        "robustness_summary": "supplementary",
        "measurement_process": "supplementary",
    }


def test_main_landmark_figure_keeps_audit_panels_in_supplement(
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
        panel_placements={
            "robustness_summary": "supplementary",
            "measurement_process": "supplementary",
        },
    )

    contract = pd.read_json(
        tmp_path / "outputs" / "display_suite.figure_contract.json", typ="series"
    )
    assert [panel["role"] for panel in contract["panels"]] == [
        "primary_estimand",
        "descriptive_result",
    ]
    assert contract["height_mm"] == 78.0
    assert summary["supplementary_panel_ids"] == [
        "measurement_process",
        "robustness_summary",
    ]
    assert len(summary["source_data_files"]) == 4
    svg = (tmp_path / "outputs" / "display_suite.svg").read_text(encoding="utf-8")
    assert "Sensitivity-analysis coverage" not in svg
    assert "Measurement availability" not in svg
