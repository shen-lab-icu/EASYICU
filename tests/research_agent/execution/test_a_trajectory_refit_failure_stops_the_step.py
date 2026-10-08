"""A stability refit that fails ends its step with a stop the host can read.

The trajectory stability owner reports every outcome in its step summary and
returns, so its step process exited 0 even when a planned refit had failed for
a reason other than the class model's own result on its subsample.  The host
read the summary's failed status, but nothing carried the failure's code on:
the run's report, its retry and its message fell back to a generic failure.
The stability rule needs every planned refit, so such a failure leaves it
without a result, and the plan, the data and the seeds decide it.  The step
process now ends with the owner's registered stop after the summary is
written.  The rule's own rejection stays a reportable result, and an input or
contract failure keeps the failure it had.  Synthetic, opaque bundles only.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import easyicu
from easyicu.research_agent.contracts.executor_stop import (
    EXECUTOR_STOP_RECORD_NAME,
    EXECUTOR_STOP_REASONS,
    ExecutorStop,
    executor_stop_codes,
)
from easyicu.research_agent.execution.executor_stop_record import (
    read_executor_stop_record,
)
from easyicu.research_agent.execution.retry_basis import load_failed_step_retry_basis
from easyicu.research_agent.execution.runners import (
    trajectory_stability_executor as owner,
)
from easyicu.research_agent.execution.runners.trajectory_stability_executor import (
    run_trajectory_stability,
    stop_on_refit_failure,
    trajectory_stability_executor_code,
)
from easyicu.research_agent.schema import AnalysisPlan
from easyicu.research_agent.trajectory.mixed_mode_latent_class import (
    ClassModelFitNotRealized,
)
from easyicu.research_agent.trajectory.plan_contract import (
    STABILITY_EXECUTOR_INPUTS,
    STABILITY_EXECUTOR_OUTPUTS,
)

from tests.support.trajectory_stability_bundle import (
    sha256_file,
    stability_spec,
    write_upstream_bundle,
)

_OWNER = "trajectory_cluster_stability"
_STOP = "trajectory_stability_refit_failed"


def _bundle(tmp_path: Path, *, one_reference_singleton: bool = False):
    resolved, representation, assignments = write_upstream_bundle(
        tmp_path,
        n_clusters=2,
        id_column="opaque_unit",
        representation_columns=("coordinate_a", "coordinate_b"),
        assignment_column="candidate_label",
    )
    if one_reference_singleton:
        # One reference class holds a single unit, so a subsample without it
        # has one reference class, and its refit cannot be compared with it.
        assignments["candidate_label"] = "group::100"
        last_unit = representation["opaque_unit"].iloc[-1]
        assignments.loc[assignments["opaque_unit"] == last_unit, "candidate_label"] = (
            "group::200"
        )
        upstream = tmp_path / "upstream"
        assignment_path = upstream / "opaque_candidate_labels.csv"
        assignments.to_csv(assignment_path, index=False)
        inputs = resolved["inputs"]
        inputs["artifact:candidate_cluster_assignments"]["sha256"] = sha256_file(
            assignment_path
        )
        solution_path = upstream / "opaque_solution_schema.json"
        solution = json.loads(solution_path.read_text(encoding="utf-8"))
        solution["candidate_assignments_sha256"] = sha256_file(assignment_path)
        solution_path.write_text(json.dumps(solution), encoding="utf-8")
        inputs["manifest:candidate_cluster_solution_schema"]["sha256"] = sha256_file(
            solution_path
        )
    return resolved


def _plan(spec) -> AnalysisPlan:
    """Representation, candidate selection, then the stability owner's step."""

    return AnalysisPlan.model_validate(
        {
            "research_question": "Assess fixed-window trajectory phenotypes.",
            "analysis_type": "trajectory_clustering",
            "steps": [
                {
                    "step_id": "01_representation",
                    "planned_analysis_role": "auxiliary",
                    "intent": "Build the trajectory representation.",
                    "inputs": ["coordinate_a", "coordinate_b"],
                    "expected_outputs": [
                        "artifact:trajectory_representation",
                        "table:trajectory_membership",
                        "manifest:trajectory_representation_schema",
                    ],
                    "method": "missingness_aware_trajectory_representation",
                },
                {
                    "step_id": "02_candidates",
                    "planned_analysis_role": "primary",
                    "intent": "Fit and select the candidate solution.",
                    "inputs": [
                        "artifact:trajectory_representation",
                        "manifest:trajectory_representation_schema",
                    ],
                    "expected_outputs": [
                        "artifact:candidate_cluster_models",
                        "artifact:candidate_cluster_assignments",
                        "manifest:cluster_selection",
                        "manifest:candidate_cluster_solution_schema",
                    ],
                    "method": "latent_class_trajectory_clustering",
                    "scientific_action_id": "phenotyping.trajectory_feature_clustering",
                },
                {
                    "step_id": "03_stability",
                    "planned_analysis_role": "sensitivity",
                    "intent": "Execute the planned stability design.",
                    "inputs": sorted(STABILITY_EXECUTOR_INPUTS),
                    "expected_outputs": sorted(STABILITY_EXECUTOR_OUTPUTS),
                    "method": "trajectory_cluster_stability",
                    "scientific_action_id": "phenotyping.trajectory_cluster_stability",
                    "trajectory_stability_spec": spec.model_dump(mode="json"),
                },
            ],
            "rationale": "Exercise the stability owner's step process.",
        }
    )


def _run_step_process(tmp_path: Path, resolved, spec) -> subprocess.CompletedProcess:
    """Run the step's generated runner as the step process runs it."""

    plan = _plan(spec)
    [step] = [item for item in plan.steps if item.step_id == "03_stability"]
    code = trajectory_stability_executor_code(step, plan=plan)
    inputs_path = tmp_path / "resolved_inputs.json"
    inputs_path.write_text(json.dumps(resolved), encoding="utf-8")
    package_root = str(Path(easyicu.__file__).resolve().parents[1])
    env = {
        **os.environ,
        "STEP_OUT_DIR": str(tmp_path / "step_outputs"),
        "EASYICU_RUN_DIR": str(tmp_path),
        "EASYICU_RESOLVED_INPUTS_JSON": str(inputs_path),
        "EASYICU_HOME": str(tmp_path / "home"),
        "PYTHONPATH": os.pathsep.join(
            item for item in (package_root, os.environ.get("PYTHONPATH")) if item
        ),
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    return subprocess.run(
        [sys.executable, "-B", "-c", code],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )


def _summary(out_dir: Path) -> dict:
    return json.loads((out_dir / "step_summary.json").read_text(encoding="utf-8"))


def test_the_stop_is_the_stability_owners_and_repeats_on_an_unchanged_retry():
    reason = EXECUTOR_STOP_REASONS[_STOP]

    assert reason.owner == _OWNER
    # The failed refits' errors are free text; the stop names no cause.
    assert reason.cause_codes == frozenset()
    # Every refit's subsample and seed follow from the plan's design.
    assert reason.repeats_on_unchanged_retry is True


def test_a_refit_that_fails_ends_the_step_process_with_the_stop(tmp_path: Path):
    resolved = _bundle(tmp_path, one_reference_singleton=True)

    completed = _run_step_process(
        tmp_path, resolved, stability_spec().model_copy(update={"base_seed": 9})
    )

    assert completed.returncode != 0
    assert f"ExecutorStop: {_STOP}" in completed.stderr
    out_dir = tmp_path / "step_outputs"
    # The summary is written before the stop and keeps its own codes.
    summary = _summary(out_dir)
    assert (summary["status"], summary["failure_class"], summary["reason_code"]) == (
        "failed_closed",
        "numerical_engine_failure",
        "TRAJECTORY_REFIT_ENGINE_FAILURE",
    )
    stop, rejection = read_executor_stop_record(out_dir, expected_owner=_OWNER)
    assert rejection is None
    assert (stop.reason_code, stop.cause_code) == (_STOP, None)


def test_a_completed_design_ends_the_step_process_as_before(tmp_path: Path):
    resolved = _bundle(tmp_path)

    completed = _run_step_process(tmp_path, resolved, stability_spec())

    assert completed.returncode == 0, completed.stderr
    out_dir = tmp_path / "step_outputs"
    assert _summary(out_dir)["status"] == "ok"
    assert not (out_dir / EXECUTOR_STOP_RECORD_NAME).exists()


def _refits_raise(*errors):
    """A refit that raises each error in turn; ``None`` fits the subsample."""

    fit = owner._fit_observed_data_diag_gmm
    planned = list(errors)

    def refit(x, **kwargs):
        error = planned.pop(0)
        if error is not None:
            raise error
        return fit(x, **kwargs)

    return refit


def _summary_after(tmp_path: Path, monkeypatch, *errors) -> tuple[dict, Path]:
    resolved = _bundle(tmp_path)
    monkeypatch.setattr(owner, "_fit_observed_data_diag_gmm", _refits_raise(*errors))
    out_dir = tmp_path / "step_outputs"
    summary = run_trajectory_stability(
        spec=stability_spec(), out_dir=out_dir, run_dir=tmp_path, resolved_inputs=resolved
    )
    return summary, out_dir


@pytest.mark.parametrize(
    "error",
    [
        ValueError("a refit coordinate has no observed values"),
        np.linalg.LinAlgError("singular matrix"),
        FloatingPointError("overflow encountered"),
    ],
    ids=["value_error", "linear_algebra_error", "floating_point_error"],
)
def test_a_refit_that_fails_on_a_condition_of_its_subsample_is_the_stop(
    tmp_path: Path, monkeypatch, error: Exception
):
    summary, out_dir = _summary_after(tmp_path, monkeypatch, None, error)

    assert summary["reason_code"] == "TRAJECTORY_REFIT_ENGINE_FAILURE"
    with pytest.raises(ExecutorStop) as raised:
        stop_on_refit_failure(summary, out_dir=out_dir)

    assert (raised.value.owner, raised.value.reason_code) == (_OWNER, _STOP)
    assert str(raised.value).startswith(f"{_STOP}: ")
    stop, _rejection = read_executor_stop_record(out_dir, expected_owner=_OWNER)
    assert stop is not None and stop.reason_code == _STOP


@pytest.mark.parametrize(
    "errors",
    [
        (None, RuntimeError("engine defect")),
        (None, TypeError("unsupported operand")),
        (None, OSError("no space left on device")),
        # One such error among conditions of the subsample is enough, in
        # either order.
        (ValueError("a refit coordinate has no observed values"), KeyError("label")),
        (KeyError("label"), ValueError("a refit coordinate has no observed values")),
    ],
    ids=["runtime_error", "type_error", "os_error", "before_a_condition", "after_a_condition"],
)
def test_an_internal_or_environment_error_of_a_refit_is_no_stop(
    tmp_path: Path, monkeypatch, errors
):
    summary, out_dir = _summary_after(tmp_path, monkeypatch, *errors)

    # No revision of the plan removes it, and a retry may not repeat it.
    assert (summary["status"], summary["failure_class"], summary["reason_code"]) == (
        "failed_closed",
        "internal_or_environment_failure",
        "TRAJECTORY_REFIT_INTERNAL_FAILURE",
    )
    assert stop_on_refit_failure(summary, out_dir=out_dir) is None
    assert not (out_dir / EXECUTOR_STOP_RECORD_NAME).exists()


def test_refits_that_do_not_reach_the_solution_are_the_rules_result_not_a_stop(
    tmp_path: Path, monkeypatch
):
    summary, out_dir = _summary_after(
        tmp_path,
        monkeypatch,
        None,
        ClassModelFitNotRealized("observed-data refit did not converge"),
    )

    assert summary["status"] == "ok"
    assert summary["reason_code"] == "TRAJECTORY_STABILITY_REFITS_BELOW_MINIMUM"
    [outcome] = summary["reportable_rule_outcomes"]
    assert outcome["disposition"] == "too_few_successful_refits"
    assert stop_on_refit_failure(summary, out_dir=out_dir) is None
    assert not (out_dir / EXECUTOR_STOP_RECORD_NAME).exists()


@pytest.mark.parametrize(
    ("arguments", "reason_code"),
    [
        ({"scientific_runtime_authority": {"schema_version": "unknown"}}, "TRAJECTORY_SCIENTIFIC_AUTHORITY_INVALID"),
        ({"spec": {"n_resamples": 1}}, "TRAJECTORY_STABILITY_SPEC_INVALID"),
        ({"resolved_inputs": {"inputs": {}}}, "TRAJECTORY_STABILITY_CONTRACT_INVALID"),
    ],
    ids=["authority", "spec", "contract"],
)
def test_an_input_or_contract_failure_keeps_the_failure_it_had(
    tmp_path: Path, arguments, reason_code: str
):
    out_dir = tmp_path / "step_outputs"
    summary = run_trajectory_stability(
        **{
            "spec": stability_spec(),
            "out_dir": out_dir,
            "run_dir": tmp_path,
            "resolved_inputs": _bundle(tmp_path),
            **arguments,
        }
    )

    assert (summary["status"], summary["reason_code"]) == ("failed_closed", reason_code)
    assert stop_on_refit_failure(summary, out_dir=out_dir) is None
    assert not (out_dir / EXECUTOR_STOP_RECORD_NAME).exists()


def _failed_stability_step(**fields) -> dict:
    return {
        "step_id": "03_stability",
        "status": "deterministic_standard_blocked",
        "deterministic_standard_analysis": _OWNER,
        "standard_executor_terminal_reason": "executor_typed_stop",
        "executor_stop_reason_code": _STOP,
        "step_llm_repair_attempts": 0,
        "step_llm_repair_budget": 2,
        "step_provider_call_attempts": 0,
        "step_provider_call_budget": 9,
        "step_provider_call_remaining": 9,
        **fields,
    }


def test_a_stopped_stability_step_is_a_stop_its_retry_would_repeat(tmp_path: Path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "manifest.json").write_text(
        json.dumps({"per_step_records": [_failed_stability_step()]}), encoding="utf-8"
    )

    basis = load_failed_step_retry_basis(run_dir, "03_stability")

    assert basis is not None
    assert basis.failure_class == "typed_stop"
    assert basis.stop_reason_code == _STOP
    assert basis.stop_repeats_on_unchanged_retry is True
    assert executor_stop_codes(_failed_stability_step()) == {"reason_code": _STOP}
    # Another executor's step cannot carry the stability owner's stop.
    assert (
        executor_stop_codes(
            _failed_stability_step(
                deterministic_standard_analysis="signed_landmark_continuous_survival_suite"
            )
        )
        == {}
    )
