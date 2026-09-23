from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from easyicu.research_agent.audits.figures import FigureSourceDataValidator
from easyicu.research_agent.execution.runners.trajectory_selection_figure_executor import (
    run_trajectory_selection_figure,
)


def _binding(
    run_dir: Path,
    *,
    step_id: str,
    input_key: str,
    frame: pd.DataFrame,
) -> dict:
    product = input_key.split(":", 1)[1]
    path = run_dir / f"{product}.csv"
    frame.to_csv(path, index=False)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    binding = {
        "relative_path": path.name,
        "sha256": digest,
        "evidence_id": f"table_{product}",
        "declared_kind": "table",
        "evidence_kind": "table",
        "product": product,
        "produced_by_step": f"producer_{product}",
        "product_contract": {
            "columns": list(frame.columns),
            "row_count": len(frame),
        },
        "consumption_contract": {
            "input_key": input_key,
            "mode": "all_rows",
            "artifact_sha256": digest,
        },
    }
    binding["identity_row"] = {
        "input_key": input_key,
        "declared_kind": "table",
        "product": product,
        "evidence_id": binding["evidence_id"],
        "sha256": digest,
    }
    return binding


def _selection() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "n_clusters": [2, 3, 4],
            "bic": [300.0, 200.0, 100.0],
            "aic": [280.0, 180.0, 80.0],
            "final_log_likelihood": [-120.0, -70.0, -20.0],
            "parameter_count": [20, 20, 20],
            "selected": [False, False, True],
            "aic_minimum": [False, False, True],
            "upper_boundary": [False, False, True],
            "scientific_status": ["failed_closed"] * 3,
            "reason_code": ["NO_INTERIOR_OPTIMUM"] * 3,
            "reportable_result": [
                "no_interior_solution_in_prespecified_candidate_range"
            ]
            * 3,
        }
    )


def _availability() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "feature": ["resp__h0_12", "resp__h12_24", "lact__h0_12", "lact__h12_24"],
            "observed_n": [80, 60, 90, 70],
            "missing_n": [20, 40, 10, 30],
            "missing_fraction": [0.2, 0.4, 0.1, 0.3],
        }
    )


def _profiles(*, empty: bool = False) -> pd.DataFrame:
    columns = [
        "cluster",
        "source_column",
        "window_start_hours",
        "window_end_hours",
        "summary_statistic",
        "value",
        "n_observed",
    ]
    if empty:
        return pd.DataFrame(columns=columns)
    rows = []
    for cluster in (0, 1):
        for concept in ("resp", "lact"):
            for start in (0, 12):
                rows.append(
                    {
                        "cluster": cluster,
                        "source_column": f"{concept}__h{start}_{start + 12}",
                        "window_start_hours": start,
                        "window_end_hours": start + 12,
                        "summary_statistic": "mean",
                        "value": 0.5 * cluster + 0.1 * start,
                        "n_observed": 40,
                    }
                )
    return pd.DataFrame(rows, columns=columns)


def _cluster_sizes(*, empty: bool = False) -> pd.DataFrame:
    columns = ["cluster", "n"]
    if empty:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame({"cluster": [0, 1], "n": [55, 45]}, columns=columns)


def _cluster_stability(*, empty: bool = False) -> pd.DataFrame:
    columns = ["resample_id", "n_overlap", "adjusted_rand_index", "selected_n_clusters"]
    if empty:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(
        {
            "resample_id": list(range(12)),
            "n_overlap": [80] * 12,
            "adjusted_rand_index": [0.70 + 0.01 * index for index in range(12)],
            "selected_n_clusters": [2] * 12,
        },
        columns=columns,
    )


def _characterization_bindings(
    tmp_path: Path, *, step_id: str, empty: bool
) -> dict:
    return {
        key: _binding(tmp_path, step_id=step_id, input_key=key, frame=frame)
        for key, frame in (
            ("table:trajectory_profiles", _profiles(empty=empty)),
            ("table:cluster_sizes", _cluster_sizes(empty=empty)),
            ("table:cluster_stability", _cluster_stability(empty=empty)),
        )
    }


def test_failed_closed_selection_renders_a_bound_diagnostic_without_labels(
    tmp_path: Path,
) -> None:
    step_id = "trajectory_selection_figure"
    bindings = {
        "table:trajectory_candidate_selection": _binding(
            tmp_path,
            step_id=step_id,
            input_key="table:trajectory_candidate_selection",
            frame=_selection(),
        ),
        "table:feature_availability": _binding(
            tmp_path,
            step_id=step_id,
            input_key="table:feature_availability",
            frame=_availability(),
        ),
        **_characterization_bindings(tmp_path, step_id=step_id, empty=True),
    }

    summary = run_trajectory_selection_figure(
        out_dir=tmp_path / "figure",
        run_dir=tmp_path,
        resolved_inputs={"step_id": step_id, "inputs": bindings},
        step_id=step_id,
    )

    assert summary["status"] == "ok"
    assert summary["scientific_status"] == "failed_closed"
    assert summary["reason_code"] == "NO_INTERIOR_OPTIMUM"
    selection_source = pd.read_csv(
        tmp_path / "figure" / "trajectory_selection_bic_source_data.csv"
    )
    assert selection_source["source_row_index"].tolist() == [0, 1, 2]
    trace = FigureSourceDataValidator._compare_source_to_upstream(
        source_df=selection_source,
        source_path=(
            tmp_path / "figure" / "trajectory_selection_bic_source_data.csv"
        ),
        upstream_path=tmp_path / "trajectory_candidate_selection.csv",
    )
    assert trace["ok"], trace
    availability_source = pd.read_csv(
        tmp_path
        / "figure"
        / "trajectory_selection_availability_source_data.csv"
    )
    assert list(availability_source.columns) == [
        *_availability().columns,
        "source_table",
        "source_step_id",
    ]
    assert set(availability_source["source_table"]) == {"feature_availability.csv"}
    assert set(availability_source["source_step_id"]) == {
        "producer_feature_availability"
    }
    for suffix in ("png", "svg", "pdf", "tiff"):
        assert (
            tmp_path / "figure" / f"trajectory_selection_diagnostics.{suffix}"
        ).exists()
    contract = json.loads(
        (
            tmp_path
            / "figure"
            / "trajectory_selection_diagnostics.figure_contract.json"
        ).read_text("utf-8")
    )
    # A criterion-versus-k curve is a selection diagnostic. It never promises
    # that the groups are separable, so it carries the diagnostics rendered
    # role and the typed ``cluster_selection`` article role.
    assert [panel["role"] for panel in contract["panels"]] == [
        "diagnostics",
        "data_quality",
    ]
    assert [
        panel["metadata"]["article_role"] for panel in contract["panels"]
    ] == ["cluster_selection", "data_quality"]
    assert "no interior solution" in contract["panels"][0]["claim"].lower()
    assert "aic" in contract["panels"][0]["claim"].lower()
    assert "candidate labels are not displayed" in contract["statistics_note"].lower()
    assert "Fail closed: the minimum occurred" in contract["reader_caption"]
    svg = (tmp_path / "figure" / "trajectory_selection_diagnostics.svg").read_text("utf-8")
    assert "Fail closed: the minimum occurred" not in svg
    assert "No interior optimum" in svg

    # The sealed authority's plan-time panel promise binds to this exact
    # rendered contract (role, chart grammar and typed sources per panel).
    from easyicu.research_agent.execution.figure_plan_binding import (
        validate_step_planned_figure_contract_binding,
    )
    from easyicu.research_agent.schema import AnalysisStep
    from easyicu.research_agent.trajectory.scientific_runtime_authority import (
        TRAJECTORY_SELECTION_FIGURE_PANELS,
    )

    planned = AnalysisStep(
        step_id=step_id,
        planned_analysis_role="auxiliary",
        intent="Render the signed candidate-grid decision and coordinate availability.",
        inputs=["table:trajectory_candidate_selection", "table:feature_availability"],
        expected_outputs=["figure:trajectory_selection_diagnostics"],
        method="signed_trajectory_selection_diagnostic_figure",
        input_consumption_contracts=[
            {"input_key": "table:trajectory_candidate_selection", "mode": "all_rows"},
            {"input_key": "table:feature_availability", "mode": "all_rows"},
        ],
        figure_panels=[
            panel.bind(figure_output="figure:trajectory_selection_diagnostics")
            for panel in TRAJECTORY_SELECTION_FIGURE_PANELS
        ],
    )
    assert (
        validate_step_planned_figure_contract_binding(
            step=planned, out_dir=tmp_path / "figure", step_summary=summary
        )
        == []
    )


def test_availability_arithmetic_drift_fails_closed(tmp_path: Path) -> None:
    step_id = "trajectory_selection_figure"
    availability = _availability()
    availability.loc[0, "missing_fraction"] = 0.9
    bindings = {
        "table:trajectory_candidate_selection": _binding(
            tmp_path,
            step_id=step_id,
            input_key="table:trajectory_candidate_selection",
            frame=_selection(),
        ),
        "table:feature_availability": _binding(
            tmp_path,
            step_id=step_id,
            input_key="table:feature_availability",
            frame=availability,
        ),
        **_characterization_bindings(tmp_path, step_id=step_id, empty=True),
    }

    with pytest.raises(ValueError, match="missing fractions"):
        run_trajectory_selection_figure(
            out_dir=tmp_path / "figure",
            run_dir=tmp_path,
            resolved_inputs={"step_id": step_id, "inputs": bindings},
            step_id=step_id,
        )


def _selected() -> pd.DataFrame:
    frame = _selection()
    frame["selected"] = [False, True, False]
    frame["aic_minimum"] = [False, True, False]
    frame["upper_boundary"] = [False, False, True]
    frame["bic"] = [300.0, 100.0, 200.0]
    frame["aic"] = [280.0, 80.0, 180.0]
    frame["scientific_status"] = ["selected"] * 3
    frame["reason_code"] = ["INTERIOR_OPTIMUM"] * 3
    frame["reportable_result"] = ["interior_solution"] * 3
    return frame


def _run_characterization(tmp_path: Path, *, empty: bool, selection: pd.DataFrame):
    step_id = "trajectory_selection_figure"
    bindings = {
        "table:trajectory_candidate_selection": _binding(
            tmp_path,
            step_id=step_id,
            input_key="table:trajectory_candidate_selection",
            frame=selection,
        ),
        "table:feature_availability": _binding(
            tmp_path,
            step_id=step_id,
            input_key="table:feature_availability",
            frame=_availability(),
        ),
        **_characterization_bindings(tmp_path, step_id=step_id, empty=empty),
    }
    summary = run_trajectory_selection_figure(
        out_dir=tmp_path / "figure",
        run_dir=tmp_path,
        resolved_inputs={"step_id": step_id, "inputs": bindings},
        step_id=step_id,
    )
    return step_id, summary


def _signed_figure_step(step_id: str):
    from easyicu.research_agent.schema import AnalysisStep
    from easyicu.research_agent.trajectory.scientific_runtime_authority import (
        TRAJECTORY_CHARACTERIZATION_FIGURE_PANELS,
        TRAJECTORY_SELECTION_FIGURE_PANELS,
    )
    from easyicu.research_agent.execution.runners.trajectory_selection_figure_executor import (  # noqa: E501
        TRAJECTORY_SELECTION_FIGURE_INPUTS,
        TRAJECTORY_SELECTION_FIGURE_OUTPUTS,
    )

    return AnalysisStep(
        step_id=step_id,
        planned_analysis_role="auxiliary",
        intent="Render the signed candidate-grid decision and the class characterization.",
        inputs=list(TRAJECTORY_SELECTION_FIGURE_INPUTS),
        expected_outputs=list(TRAJECTORY_SELECTION_FIGURE_OUTPUTS),
        method="signed_trajectory_selection_diagnostic_figure",
        input_consumption_contracts=[
            {"input_key": key, "mode": "all_rows"}
            for key in TRAJECTORY_SELECTION_FIGURE_INPUTS
        ],
        figure_panels=[
            *(
                panel.bind(figure_output="figure:trajectory_selection_diagnostics")
                for panel in TRAJECTORY_SELECTION_FIGURE_PANELS
            ),
            *(
                panel.bind(
                    figure_output="figure:trajectory_phenotype_characterization"
                )
                for panel in TRAJECTORY_CHARACTERIZATION_FIGURE_PANELS
            ),
        ],
    )


def test_characterization_surface_renders_and_binds_its_planned_panels(
    tmp_path: Path,
) -> None:
    from easyicu.research_agent.execution.figure_plan_binding import (
        validate_step_planned_figure_contract_binding,
    )

    step_id, summary = _run_characterization(
        tmp_path, empty=False, selection=_selected()
    )
    assert summary["reportable_phenotype_solution"] is True
    assert set(summary["output_files"]) == {
        "figure:trajectory_selection_diagnostics",
        "figure:trajectory_phenotype_characterization",
    }
    contract = json.loads(
        (
            tmp_path
            / "figure"
            / "trajectory_phenotype_characterization.figure_contract.json"
        ).read_text("utf-8")
    )
    assert [panel["role"] for panel in contract["panels"]] == [
        "phenotype_profile",
        "phenotype_structure",
        "stability",
    ]
    assert [panel["metadata"]["chart_type"] for panel in contract["panels"]] == [
        "profile_heatmap",
        "cluster_heatmap",
        "subsampling_ari",
    ]
    # Every drawn value is a projection of the sealed parent tables.
    profiles = pd.read_csv(tmp_path / "figure" / "trajectory_profiles_source_data.csv")
    assert profiles["source_table"].eq("trajectory_profiles.csv").all()
    assert len(profiles) == len(_profiles())

    assert (
        validate_step_planned_figure_contract_binding(
            step=_signed_figure_step(step_id),
            out_dir=tmp_path / "figure",
            step_summary=summary,
        )
        == []
    )


def test_characterization_states_the_sealed_decision_when_no_solution_is_reportable(
    tmp_path: Path,
) -> None:
    from easyicu.research_agent.execution.figure_plan_binding import (
        validate_step_planned_figure_contract_binding,
    )

    step_id, summary = _run_characterization(
        tmp_path, empty=True, selection=_selection()
    )
    assert summary["reportable_phenotype_solution"] is False
    contract = json.loads(
        (
            tmp_path
            / "figure"
            / "trajectory_phenotype_characterization.figure_contract.json"
        ).read_text("utf-8")
    )
    assert "no stable phenotype solution" in contract["core_claim"].lower()
    assert "no stable phenotype solution" in contract["panels"][0]["claim"].lower()
    svg = (
        tmp_path / "figure" / "trajectory_phenotype_characterization.svg"
    ).read_text("utf-8")
    assert "No stable phenotype solution" in svg
    # The promise still binds: reporting that there is no phenotype is a
    # truthful outcome of the phenotype-profile role.
    assert (
        validate_step_planned_figure_contract_binding(
            step=_signed_figure_step(step_id),
            out_dir=tmp_path / "figure",
            step_summary=summary,
        )
        == []
    )
