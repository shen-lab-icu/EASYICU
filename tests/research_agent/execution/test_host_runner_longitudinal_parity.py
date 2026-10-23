"""Longitudinal owners use only invented observations and local runtimes."""

import hashlib
import json
import shutil

import numpy as np
import pandas as pd
import pytest

from easyicu.research_agent.authority.current_case_scientific_runtime import (
    load_current_case_scientific_runtime_authority,
)
from tests.support.host_runner_parity import (
    select_for_parity as select_standard_executor,
)
from tests.support.host_runner_signed_cases import trajectory_authority
from tests.support.host_runner_time_varying import (
    time_varyingsource,
    time_varying_materialize,
    time_varying_projection,
)
from tests.support.host_runner_parity import assert_parity, table_binding


@pytest.mark.parametrize("family", ["time_varying", "trajectory"])
def test_longitudinal_executor_products_match(tmp_path, family):
    extra_env = {"SOURCE_DATE_EPOCH": "1700000000"}
    if family == "time_varying":
        if shutil.which("Rscript") is None:
            pytest.skip("This typed executor requires the optional native R runtime")
        acquired = time_varying_materialize(time_varyingsource(tmp_path))
        projection = time_varying_projection(acquired.universe_path)
        authority = load_current_case_scientific_runtime_authority(projection.authority)
        plan = authority.development_execution_only_plan(
            research_question="Generated measurements and survival"
        )
        cohort = acquired.universe_path
        extra_env["COHORT_TRAJECTORY_PARQUET"] = str(acquired.trajectory_path)
        kwargs = {
            "current_case_scientific_runtime_authority": authority,
            "scientific_runtime_projection_sha256": projection.projection_sha256,
        }
        steps = plan.steps[1:]
    else:
        authority = trajectory_authority()
        plan = authority.development_execution_only_plan(
            research_question="Generated longitudinal clusters"
        )
        cohort = tmp_path / "cohort.parquet"
        pd.DataFrame({"stay_id": np.arange(1, 121)}).to_parquet(cohort, index=False)
        rng = np.random.default_rng(44)
        rows = []
        for index in range(120):
            for window in (0, 12):
                for coordinate in authority.coordinate_concepts:
                    value = float(
                        (index // 40) * 4 + 1 + window / 24 + rng.normal(0, 0.1)
                    )
                    rows.append(
                        {
                            "stay_id": index + 1,
                            "charttime": float(window),
                            "concept": coordinate,
                            "value_num": value,
                            "value_str": str(value),
                            "evidence_state": "direct_observed",
                            "owner_observed": 1,
                            "owner_available": 1,
                        }
                    )
        trajectory = tmp_path / "trajectory.parquet"
        pd.DataFrame(rows).to_parquet(trajectory, index=False)
        extra_env["TRAJECTORY_PARQUET"] = str(trajectory)
        kwargs = {
            "trajectory_scientific_runtime_authority": authority,
            "scientific_runtime_projection_sha256": "2" * 64,
        }
        steps = plan.steps[:2]
    frame = pd.read_parquet(cohort)
    products = {
        "table:analysis_cohort": {
            "relative_path": cohort.relative_to(tmp_path).as_posix(),
            "sha256": hashlib.sha256(cohort.read_bytes()).hexdigest(),
            "declared_kind": "table",
            "evidence_kind": "table",
            "product": "analysis_cohort",
            "evidence_id": "synthetic-cohort",
            "produced_by_step": "source",
            "product_contract": {
                "columns": list(frame.columns),
                "row_count": len(frame),
            },
        }
    }
    for step in steps:
        bindings = {key: products[key] for key in step.inputs if ":" in key}
        selected = select_standard_executor(
            step, plan=plan, resolved_bindings=bindings, **kwargs
        )
        assert selected is not None, step
        manifest = tmp_path / "resolved_inputs.json"
        manifest.write_text(json.dumps({"step_id": step.step_id, "inputs": bindings}))
        assert_parity(tmp_path, cohort, step, selected.code, manifest, extra_env)
        output = tmp_path / "steps" / step.step_id / "outputs"
        summary = json.loads((output / "step_summary.json").read_text())
        for key, filename in summary.get("output_files", {}).items():
            path = output / filename
            products[key] = {
                "relative_path": path.relative_to(tmp_path).as_posix(),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "evidence_id": key,
                "produced_by_step": step.step_id,
            }
