"""Synthetic frozen trajectory products exercise subprocess parity."""

import json

import pandas as pd
import pytest

from tests.support.host_runner_parity import (
    select_for_parity as select_standard_executor,
)
from easyicu.research_agent.schema import AnalysisPlan, AnalysisStep
from easyicu.research_agent.trajectory.plan_contract import (
    STABILITY_EXECUTOR_INPUTS,
    STABILITY_EXECUTOR_OUTPUTS,
)
from tests.support import host_runner_figure_cases as cases
from tests.support.host_runner_parity import assert_parity, table_binding
from tests.support.trajectory_stability_bundle import (
    stability_spec,
    write_upstream_bundle,
)


@pytest.mark.parametrize("family", ["stability", "figure"])
def test_trajectory_products_match(tmp_path, family):
    if family == "stability":
        resolved, frame, _ = write_upstream_bundle(
            tmp_path,
            n_clusters=2,
            id_column="unit_id",
            representation_columns=("signal_a", "signal_b"),
            assignment_column="class_id",
        )
        step = AnalysisStep(
            step_id="stability",
            intent="Validate frozen synthetic classes",
            method="trajectory_cluster_stability",
            inputs=sorted(STABILITY_EXECUTOR_INPUTS),
            expected_outputs=sorted(STABILITY_EXECUTOR_OUTPUTS),
            trajectory_stability_spec=stability_spec(),
        )
        candidate = AnalysisStep(
            step_id="candidate",
            intent="Select synthetic classes",
            method="model_based_clustering",
            inputs=[
                "artifact:trajectory_representation",
                "manifest:trajectory_representation_schema",
            ],
            expected_outputs=[
                "artifact:candidate_cluster_models",
                "artifact:candidate_cluster_assignments",
                "manifest:cluster_selection",
                "manifest:candidate_cluster_solution_schema",
            ],
        )
        plan = AnalysisPlan(
            research_question="Synthetic trajectory classes",
            analysis_type="trajectory_clustering",
            steps=[candidate, step],
        )
    else:
        step = cases.trajectory_signed_figure_step("trajectory_figure")
        frames = {
            "table:trajectory_candidate_selection": cases.trajectory_selected(),
            "table:feature_availability": cases.trajectory_availability(),
            "table:trajectory_profiles": cases.trajectory_profiles(),
            "table:cluster_sizes": cases.trajectory_cluster_sizes(),
            "table:cluster_stability": cases.trajectory_cluster_stability(),
        }
        resolved = {
            "inputs": {
                key: table_binding(tmp_path, key, frame)
                for key, frame in frames.items()
            }
        }
        frame = pd.DataFrame({"measurement": [1.0, 2.0]})
        plan = AnalysisPlan(
            research_question="Synthetic trajectory classes",
            analysis_type="trajectory_clustering",
            steps=[step],
        )
    cohort = tmp_path / "cohort.parquet"
    frame.to_parquet(cohort, index=False)
    selected = select_standard_executor(
        step, plan=plan, resolved_bindings=resolved["inputs"]
    )
    assert selected is not None, step
    manifest = tmp_path / "resolved_inputs.json"
    manifest.write_text(json.dumps({**resolved, "step_id": step.step_id}))
    assert_parity(
        tmp_path,
        cohort,
        step,
        selected.code,
        manifest,
        {"SOURCE_DATE_EPOCH": "1700000000"},
    )
