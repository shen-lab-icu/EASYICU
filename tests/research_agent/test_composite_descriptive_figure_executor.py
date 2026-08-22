from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from easyicu.research_agent.execution.runners.composite_descriptive_figure_executor import (
    COMPOSITE_ASSOCIATION_PUBLICATION_FIGURE_INPUTS,
    COMPOSITE_ASSOCIATION_SUMMARY_PUBLICATION_FIGURE_INPUTS,
    COMPOSITE_SOURCE_AWARE_ASSOCIATION_FIGURE_INPUTS,
    COMPOSITE_DESCRIPTIVE_FIGURE_INPUTS,
    COMPOSITE_DESCRIPTIVE_ROBUSTNESS_FIGURE_INPUTS,
    composite_descriptive_figure_executor_owns_step,
    run_composite_descriptive_figure,
)
from easyicu.research_agent.execution.runners.selection import select_standard_executor
from easyicu.research_agent.schema import AnalysisPlan, AnalysisStep


def _step() -> AnalysisStep:
    return AnalysisStep(
        step_id="08_primary_figure",
        planned_analysis_role="auxiliary",
        intent="Render the four declared descriptive sources.",
        method="visualization",
        inputs=list(COMPOSITE_DESCRIPTIVE_FIGURE_INPUTS),
        expected_outputs=["figure:primary_publication_figure"],
        input_consumption_contracts=[
            {"input_key": key, "mode": "all_rows"}
            for key in COMPOSITE_DESCRIPTIVE_FIGURE_INPUTS
        ],
    )


def _frames() -> dict[str, pd.DataFrame]:
    return {
        "table:cohort_flow": pd.DataFrame(
            {"concept_id": ["analysis cohort"], "n_remaining": [100]}
        ),
        "table:exposure_outcome_distribution": pd.DataFrame(
            {
                "row_role": ["exposure_level", "exposure_level", "overall"],
                "exposure_level": [0, 1, -1],
                "n_rows": [60, 40, 100],
                "exposure_denominator": [100, 100, 100],
                "exposure_pct": [60.0, 40.0, 100.0],
                "outcome_events": [6, 8, 14],
                "outcome_denominator": [60, 40, 100],
                "outcome_rate_pct": [10.0, 20.0, 14.0],
            }
        ),
        "table:missingness_measurement_audit": pd.DataFrame(
            {
                "variable": ["age", "lactate"],
                "label": ["Age", "Lactate"],
                "n_total": [100, 100],
                "missing_n": [0, 20],
                "missing_pct": [0.0, 20.0],
            }
        ),
        "table:measurement_process_audit": pd.DataFrame(
            {
                "concept": ["age", "lactate"],
                "n_total": [100, 100],
                "measured_one_n": [100, 80],
                "eligible_n": [100, 100],
            }
        ),
    }


def _robustness_frames() -> dict[str, pd.DataFrame]:
    frames = _frames()
    frames.pop("table:measurement_process_audit")
    frames["table:robustness_summary"] = pd.DataFrame(
        {
            "axis": ["primary", "missing"],
            "total_specs": [1, 2],
            "converged_specs": [1, 2],
            "non_independent_specs": [0, 0],
            "range_low": [1.2, 1.1],
            "range_high": [1.5, 1.6],
        }
    )
    return frames


def _association_frames() -> dict[str, pd.DataFrame]:
    return {
        "table:exposure_outcome_distribution": pd.DataFrame(
            {
                "row_role": ["exposure_level", "exposure_level", "overall"],
                "exposure_level": [0, 1, -1],
                "exposure_column": ["exposure"] * 3,
                "n_rows": [60, 40, 100],
                "exposure_denominator": [100, 100, 100],
                "exposure_pct": [60.0, 40.0, 100.0],
                "outcome_events": [6, 8, 14],
                "outcome_denominator": [60, 40, 100],
                "outcome_rate_pct": [10.0, 20.0, 14.0],
                "ci_low_pct": [5.0, 12.0, 8.0],
                "ci_high_pct": [15.0, 28.0, 20.0],
            }
        ),
        "table:adjusted_association_estimates": pd.DataFrame(
            {
                "fit_status": ["fitted"],
                "estimate": [1.4],
                "ci_low": [1.1],
                "ci_high": [1.8],
                "effect_scale": ["odds_ratio"],
                "model_id": ["primary_adjusted"],
                "contrast": [None],
            }
        ),
        "table:robustness_matrix": pd.DataFrame(
            {
                "spec_id": ["primary", "complete_case"],
                "point_estimate": [1.4, 1.3],
                "ci_low": [1.1, 1.0],
                "ci_high": [1.8, 1.7],
                "effect_scale": ["OR", "OR"],
                "converged": [True, True],
            }
        ),
        "table:measurement_missingness": pd.DataFrame(
            {
                "variable": ["age", "lactate"],
                "label": ["Age", "Lactate"],
                "n_total": [100, 100],
                "missing_n": [0, 20],
                "missing_pct": [0.0, 20.0],
            }
        ),
    }


def _association_summary_frames() -> dict[str, pd.DataFrame]:
    frames = _association_frames()
    frames.pop("table:robustness_matrix")
    frames["table:robustness_summary"] = pd.DataFrame(
        {
            "axis": ["primary", "missingness"],
            "total_specs": [1, 2],
            "converged_specs": [1, 2],
            "non_independent_specs": [0, 0],
            "range_low": [1.1, 1.0],
            "range_high": [1.8, 1.7],
        }
    )
    return frames


def _source_aware_association_frames() -> dict[str, pd.DataFrame]:
    frames = _association_summary_frames()
    frames.pop("table:exposure_outcome_distribution")
    frames.pop("table:measurement_missingness")
    frames["table:absolute_risk_context"] = pd.DataFrame(
        {
            "estimate_type": ["outcome_risk", "outcome_risk", "prevalence"],
            "label": ["Observed", "No source", "Observed"],
            "n": [60, 40, 60],
            "event_n": [12, 4, None],
            "estimate": [0.2, 0.1, 0.6],
            "ci_low": [0.12, 0.03, 0.5],
            "ci_high": [0.28, 0.17, 0.7],
        }
    )
    frames["table:measurement_process_audit"] = pd.DataFrame(
        {
            "concept": ["exposure", "outcome_time"],
            "n_total": [100, 100],
            "measured_one_n": [60, 20],
            "eligible_n": [100, 20],
        }
    )
    return frames


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


def test_exact_four_table_contract_selects_composite_owner(tmp_path: Path) -> None:
    bindings = {}
    for key, frame in _frames().items():
        path = tmp_path / f"{key.partition(':')[2]}.csv"
        frame.to_csv(path, index=False)
        bindings[key] = _binding(key, frame, path)
    step = _step()

    assert composite_descriptive_figure_executor_owns_step(
        step, resolved_bindings=bindings
    )
    selection = select_standard_executor(
        step,
        plan=AnalysisPlan(research_question="Describe the cohort.", steps=[step]),
        resolved_bindings=bindings,
    )
    assert selection is not None
    assert selection.analysis_kind == "composite_descriptive_figure"
    assert selection.host_sealed_renderer is True
    assert selection.consumed_input_keys == COMPOSITE_DESCRIPTIVE_FIGURE_INPUTS


def test_owner_refuses_widened_or_incomplete_contract(tmp_path: Path) -> None:
    frames = _frames()
    bindings = {}
    for key, frame in frames.items():
        path = tmp_path / f"{key.partition(':')[2]}.csv"
        frame.to_csv(path, index=False)
        bindings[key] = _binding(key, frame, path)

    widened = _step().model_copy(update={"inputs": [*_step().inputs, "table:extra"]})
    assert not composite_descriptive_figure_executor_owns_step(
        widened, resolved_bindings={**bindings, "table:extra": {}}
    )
    incomplete = dict(bindings)
    incomplete["table:measurement_process_audit"] = {
        **incomplete["table:measurement_process_audit"],
        "product_contract": {"columns": ["concept"], "row_count": 2},
    }
    assert not composite_descriptive_figure_executor_owns_step(
        _step(), resolved_bindings=incomplete
    )


def test_robustness_four_table_contract_selects_and_renders(tmp_path: Path) -> None:
    frames = _robustness_frames()
    bindings = {}
    for key, frame in frames.items():
        path = tmp_path / f"{key.partition(':')[2]}.csv"
        frame.to_csv(path, index=False)
        bindings[key] = _binding(key, frame, path)
    step = AnalysisStep.model_validate(
        {
            **_step().model_dump(mode="json"),
            "inputs": list(COMPOSITE_DESCRIPTIVE_ROBUSTNESS_FIGURE_INPUTS),
            "input_consumption_contracts": [
                {"input_key": key, "mode": "all_rows"}
                for key in COMPOSITE_DESCRIPTIVE_ROBUSTNESS_FIGURE_INPUTS
            ],
        }
    )

    selection = select_standard_executor(
        step,
        plan=AnalysisPlan(
            research_question="Describe the cohort.",
            steps=[step],
            display_labels={"exposure=0": "Absent", "exposure=1": "Present"},
        ),
        resolved_bindings=bindings,
    )
    assert selection is not None
    assert selection.consumed_input_keys == (
        COMPOSITE_DESCRIPTIVE_ROBUSTNESS_FIGURE_INPUTS
    )

    summary = run_composite_descriptive_figure(
        out_dir=tmp_path / "outputs",
        run_dir=tmp_path,
        resolved_inputs={"step_id": step.step_id, "inputs": bindings},
        step_id=step.step_id,
        figure_product="primary_publication_figure",
        input_keys=COMPOSITE_DESCRIPTIVE_ROBUSTNESS_FIGURE_INPUTS,
        display_labels={"exposure=0": "Absent", "exposure=1": "Present"},
    )
    assert summary["status"] == "ok"
    assert summary["source_inputs"] == list(
        COMPOSITE_DESCRIPTIVE_ROBUSTNESS_FIGURE_INPUTS
    )


def test_association_four_table_contract_selects_and_renders(tmp_path: Path) -> None:
    frames = _association_frames()
    bindings = {}
    for key, frame in frames.items():
        path = tmp_path / f"{key.partition(':')[2]}.csv"
        frame.to_csv(path, index=False)
        bindings[key] = _binding(key, frame, path)
    step = AnalysisStep.model_validate(
        {
            **_step().model_dump(mode="json"),
            "step_id": "publication_figure_suite",
            "inputs": list(COMPOSITE_ASSOCIATION_PUBLICATION_FIGURE_INPUTS),
            "expected_outputs": ["figure:publication_figure_suite"],
            "input_consumption_contracts": [
                {"input_key": key, "mode": "all_rows"}
                for key in COMPOSITE_ASSOCIATION_PUBLICATION_FIGURE_INPUTS
            ],
        }
    )

    selection = select_standard_executor(
        step,
        plan=AnalysisPlan(research_question="Estimate an association.", steps=[step]),
        resolved_bindings=bindings,
    )
    assert selection is not None
    assert selection.host_sealed_renderer is True
    assert (
        selection.consumed_input_keys == COMPOSITE_ASSOCIATION_PUBLICATION_FIGURE_INPUTS
    )

    out_dir = tmp_path / "outputs"
    summary = run_composite_descriptive_figure(
        out_dir=out_dir,
        run_dir=tmp_path,
        resolved_inputs={"step_id": step.step_id, "inputs": bindings},
        step_id=step.step_id,
        figure_product="publication_figure_suite",
        input_keys=COMPOSITE_ASSOCIATION_PUBLICATION_FIGURE_INPUTS,
    )
    assert summary["status"] == "ok"
    assert summary["deterministic_standard_analysis"] == "composite_association_figure"
    assert summary["source_inputs"] == list(
        COMPOSITE_ASSOCIATION_PUBLICATION_FIGURE_INPUTS
    )
    for suffix in ("png", "svg", "pdf", "tiff", "figure_contract.json"):
        assert (out_dir / f"publication_figure_suite.{suffix}").is_file()


def test_association_summary_contract_renders_ranges_without_point_estimates(
    tmp_path: Path,
) -> None:
    frames = _association_summary_frames()
    bindings = {}
    for key, frame in frames.items():
        path = tmp_path / f"{key.partition(':')[2]}.csv"
        frame.to_csv(path, index=False)
        bindings[key] = _binding(key, frame, path)
    step = AnalysisStep.model_validate(
        {
            **_step().model_dump(mode="json"),
            "step_id": "publication_figure_suite",
            "inputs": list(COMPOSITE_ASSOCIATION_SUMMARY_PUBLICATION_FIGURE_INPUTS),
            "expected_outputs": ["figure:publication_figure_suite"],
            "input_consumption_contracts": [
                {"input_key": key, "mode": "all_rows"}
                for key in COMPOSITE_ASSOCIATION_SUMMARY_PUBLICATION_FIGURE_INPUTS
            ],
        }
    )

    selection = select_standard_executor(
        step,
        plan=AnalysisPlan(research_question="Estimate an association.", steps=[step]),
        resolved_bindings=bindings,
    )
    assert selection is not None
    assert selection.host_sealed_renderer is True
    assert selection.consumed_input_keys == (
        COMPOSITE_ASSOCIATION_SUMMARY_PUBLICATION_FIGURE_INPUTS
    )

    out_dir = tmp_path / "outputs"
    summary = run_composite_descriptive_figure(
        out_dir=out_dir,
        run_dir=tmp_path,
        resolved_inputs={"step_id": step.step_id, "inputs": bindings},
        step_id=step.step_id,
        figure_product="publication_figure_suite",
        input_keys=COMPOSITE_ASSOCIATION_SUMMARY_PUBLICATION_FIGURE_INPUTS,
    )
    assert summary["status"] == "ok"
    contract = json.loads(
        (out_dir / "publication_figure_suite.figure_contract.json").read_text()
    )
    panel_c = next(panel for panel in contract["panels"] if panel["panel_id"] == "C")
    assert panel_c["title"] == "Robustness ranges"
    assert panel_c["role"] == "robustness"
    assert panel_c["metadata"]["source_products"] == ["table:robustness_summary"]


def test_source_aware_association_contract_uses_eligible_availability(
    tmp_path: Path,
) -> None:
    frames = _source_aware_association_frames()
    bindings = {}
    for key, frame in frames.items():
        path = tmp_path / f"{key.partition(':')[2]}.csv"
        frame.to_csv(path, index=False)
        bindings[key] = _binding(key, frame, path)
    step = AnalysisStep.model_validate(
        {
            **_step().model_dump(mode="json"),
            "step_id": "publication_figure",
            "inputs": list(COMPOSITE_SOURCE_AWARE_ASSOCIATION_FIGURE_INPUTS),
            "expected_outputs": ["figure:publication_figure"],
            "input_consumption_contracts": [
                {"input_key": key, "mode": "all_rows"}
                for key in COMPOSITE_SOURCE_AWARE_ASSOCIATION_FIGURE_INPUTS
            ],
        }
    )

    selection = select_standard_executor(
        step,
        plan=AnalysisPlan(research_question="Estimate an association.", steps=[step]),
        resolved_bindings=bindings,
    )
    assert selection is not None
    assert selection.host_sealed_renderer is True
    assert (
        selection.consumed_input_keys
        == COMPOSITE_SOURCE_AWARE_ASSOCIATION_FIGURE_INPUTS
    )

    out_dir = tmp_path / "outputs"
    summary = run_composite_descriptive_figure(
        out_dir=out_dir,
        run_dir=tmp_path,
        resolved_inputs={"step_id": step.step_id, "inputs": bindings},
        step_id=step.step_id,
        figure_product="publication_figure",
        input_keys=COMPOSITE_SOURCE_AWARE_ASSOCIATION_FIGURE_INPUTS,
    )
    assert summary["status"] == "ok"
    contract = json.loads(
        (out_dir / "publication_figure.figure_contract.json").read_text()
    )
    assert [panel["role"] for panel in contract["panels"]] == [
        "descriptive_result",
        "primary_estimand",
        "robustness",
        "data_quality",
    ]
    panel_d = next(panel for panel in contract["panels"] if panel["panel_id"] == "D")
    assert panel_d["title"] == "Measurement availability"
    assert panel_d["metadata"]["source_products"] == ["table:measurement_process_audit"]


def test_renderer_preserves_exact_source_rows_and_exports_figure(
    tmp_path: Path,
) -> None:
    frames = _frames()
    bindings = {}
    for key, frame in frames.items():
        path = tmp_path / f"{key.partition(':')[2]}.csv"
        frame.to_csv(path, index=False)
        bindings[key] = _binding(key, frame, path)
    manifest = {"step_id": _step().step_id, "inputs": bindings}
    out_dir = tmp_path / "outputs"

    summary = run_composite_descriptive_figure(
        out_dir=out_dir,
        run_dir=tmp_path,
        resolved_inputs=manifest,
        step_id=_step().step_id,
        figure_product="primary_publication_figure",
    )

    assert summary["status"] == "ok"
    assert summary["deterministic_standard_analysis"] == (
        "composite_descriptive_figure"
    )
    for key, frame in frames.items():
        product = key.partition(":")[2]
        source = pd.read_csv(out_dir / f"{product}_source_data.csv")
        assert source["source_row_index"].tolist() == list(range(len(frame)))
        assert source.drop(columns=["source_row_index", "source_table"]).equals(frame)
    for suffix in ("png", "svg", "pdf", "tiff", "figure_contract.json"):
        assert (out_dir / f"primary_publication_figure.{suffix}").is_file()
    stored = json.loads((out_dir / "step_summary.json").read_text())
    assert set(item["input_key"] for item in stored["input_bindings"]) == set(
        COMPOSITE_DESCRIPTIVE_FIGURE_INPUTS
    )


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("n_rows", 60.5, "integer-like"),
        ("exposure_pct", 61.0, "does not reconcile"),
    ],
)
def test_renderer_fails_closed_on_inconsistent_display_values(
    tmp_path: Path,
    column: str,
    value: float,
    message: str,
) -> None:
    frames = _frames()
    frames["table:exposure_outcome_distribution"][column] = frames[
        "table:exposure_outcome_distribution"
    ][column].astype(float)
    frames["table:exposure_outcome_distribution"].loc[0, column] = value
    bindings = {}
    for key, frame in frames.items():
        path = tmp_path / f"{key.partition(':')[2]}.csv"
        frame.to_csv(path, index=False)
        bindings[key] = _binding(key, frame, path)

    with pytest.raises(ValueError, match=message):
        run_composite_descriptive_figure(
            out_dir=tmp_path / "outputs",
            run_dir=tmp_path,
            resolved_inputs={"step_id": _step().step_id, "inputs": bindings},
            step_id=_step().step_id,
            figure_product="primary_publication_figure",
        )
