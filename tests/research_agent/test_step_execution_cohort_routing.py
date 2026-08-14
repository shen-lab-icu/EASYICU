from __future__ import annotations

import ast
import hashlib
import inspect
from pathlib import Path

import pandas as pd
import pytest

from easyicu.research_agent.schema import (
    AnalysisPlan,
    AnalysisStep,
    RobustnessReplaySpec,
)
from easyicu.research_agent.execution.cohort_routing import (
    StepExecutionCohortRoutingError,
    bind_step_execution_cohort,
    bound_step_execution_cohort_path,
    step_may_access_preselection_universe,
)
from easyicu.research_agent.execution.envelope_sealing import (
    compile_sealed_step_result_shadow,
)
from easyicu.research_agent.execution.phase import (
    _evaluate_final_deterministic_gates,
    run_execute_phase,
)


def _binding(run_dir: Path, relative_path: str) -> dict[str, str]:
    path = (run_dir / relative_path).resolve()
    return {
        "product": "analysis_cohort",
        "relative_path": relative_path,
        "absolute_path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def test_bound_analysis_cohort_replaces_the_run_level_fallback(tmp_path: Path) -> None:
    fallback = tmp_path / "development_sample.parquet"
    fallback.write_bytes(b"untransformed sample")
    child = tmp_path / "evidence" / "analysis_cohort.parquet"
    child.parent.mkdir()
    child.write_bytes(b"step-owned transformed sample")

    selected = bound_step_execution_cohort_path(
        run_dir=tmp_path,
        fallback_path=fallback,
        resolved_input_bindings={
            "artifact:analysis_cohort": _binding(
                tmp_path, "evidence/analysis_cohort.parquet"
            )
        },
    )

    assert selected == child.resolve()


def test_planner_spelled_cohort_product_replaces_the_run_level_fallback(
    tmp_path: Path,
) -> None:
    """``cohort:analysis_set`` names the same reserved population.

    Matching only the literal ``analysis_cohort`` left the runner surface on
    the run-level fallback while the step's own typed input pointed somewhere
    else -- two populations under two names, which is what this boundary
    exists to prevent.
    """

    fallback = tmp_path / "development_sample.parquet"
    fallback.write_bytes(b"run-level sample")
    child = tmp_path / "evidence" / "analysis_set.parquet"
    child.parent.mkdir()
    child.write_bytes(b"step-emitted primary cohort")
    binding = _binding(tmp_path, "evidence/analysis_set.parquet")
    binding["product"] = "analysis_set"

    selected = bound_step_execution_cohort_path(
        run_dir=tmp_path,
        fallback_path=fallback,
        resolved_input_bindings={"cohort:analysis_set": binding},
    )

    assert selected == child.resolve()


def test_unreserved_analysis_set_spellings_stay_on_the_fallback(
    tmp_path: Path,
) -> None:
    """Only the reserved ``cohort:`` spelling claims the primary population."""

    fallback = tmp_path / "development_sample.parquet"
    fallback.write_bytes(b"run-level sample")
    other = tmp_path / "evidence" / "analysis_set.parquet"
    other.parent.mkdir()
    other.write_bytes(b"model-specific analysis set")
    binding = _binding(tmp_path, "evidence/analysis_set.parquet")
    binding["product"] = "analysis_set"

    selected = bound_step_execution_cohort_path(
        run_dir=tmp_path,
        fallback_path=fallback,
        resolved_input_bindings={"table:analysis_set": binding},
    )

    assert selected == fallback.resolve()


def test_non_cohort_typed_inputs_do_not_change_the_runner_surface(
    tmp_path: Path,
) -> None:
    fallback = tmp_path / "development_sample.parquet"
    fallback.write_bytes(b"sample")
    table = tmp_path / "evidence" / "table.csv"
    table.parent.mkdir()
    table.write_bytes(b"value\n1\n")
    binding = _binding(tmp_path, "evidence/table.csv")
    binding["product"] = "outcome_incidence"

    selected = bound_step_execution_cohort_path(
        run_dir=tmp_path,
        fallback_path=fallback,
        resolved_input_bindings={"table:outcome_incidence": binding},
    )

    assert selected == fallback.resolve()


def test_run_level_fallback_is_digest_bound_for_the_result_envelope(
    tmp_path: Path,
) -> None:
    fallback = tmp_path / "cohort_analysis.parquet"
    fallback.write_bytes(b"sealed cohort bytes")
    output_dir = tmp_path / "steps" / "02_summary" / "outputs"
    output_dir.mkdir(parents=True)
    step_record: dict[str, object] = {}

    selected = bind_step_execution_cohort(
        tmp_path,
        fallback,
        {},
        step_record,
    )
    step = AnalysisStep(
        step_id="02_summary",
        intent="Summarize the locked cohort.",
    )
    snapshot = compile_sealed_step_result_shadow(
        step=step,
        step_summary={
            "status": "ok",
            "cohort_path": "/easyicu-run/cohort_analysis.parquet",
        },
        output_dir=output_dir,
        run_dir=tmp_path,
        execution_cohort_path=selected,
        execution_cohort_sha256=str(step_record["execution_cohort_sha256"]),
        current_status="ok",
    )

    assert selected == fallback.resolve()
    assert step_record["execution_cohort_role"] == "run_level_execution_cohort"
    assert (
        step_record["execution_cohort_sha256"]
        == hashlib.sha256(fallback.read_bytes()).hexdigest()
    )
    assert snapshot.ready is True
    assert not [
        issue
        for issue in snapshot.envelope.normalization_issues
        if issue.code == "absolute_unbound_path"
    ]


@pytest.mark.parametrize("mutation", ["absolute_path", "sha256", "relative_path"])
def test_changed_typed_cohort_authority_fails_closed(
    tmp_path: Path, mutation: str
) -> None:
    fallback = tmp_path / "development_sample.parquet"
    fallback.write_bytes(b"sample")
    child = tmp_path / "evidence" / "analysis_cohort.parquet"
    child.parent.mkdir()
    child.write_bytes(b"step-owned transformed sample")
    binding = _binding(tmp_path, "evidence/analysis_cohort.parquet")
    if mutation == "absolute_path":
        other = tmp_path / "evidence" / "other.parquet"
        other.write_bytes(child.read_bytes())
        binding[mutation] = str(other)
    elif mutation == "sha256":
        binding[mutation] = "0" * 64
    else:
        binding[mutation] = "../outside.parquet"

    with pytest.raises(StepExecutionCohortRoutingError):
        bound_step_execution_cohort_path(
            run_dir=tmp_path,
            fallback_path=fallback,
            resolved_input_bindings={"artifact:analysis_cohort": binding},
        )


def test_multiple_distinct_analysis_cohorts_fail_closed(tmp_path: Path) -> None:
    fallback = tmp_path / "development_sample.parquet"
    fallback.write_bytes(b"sample")
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    first = evidence / "first.parquet"
    second = evidence / "second.parquet"
    first.write_bytes(b"first")
    second.write_bytes(b"second")

    with pytest.raises(
        StepExecutionCohortRoutingError,
        match="Multiple distinct typed analysis_cohort inputs",
    ):
        bound_step_execution_cohort_path(
            run_dir=tmp_path,
            fallback_path=fallback,
            resolved_input_bindings={
                "artifact:analysis_cohort": _binding(
                    tmp_path, "evidence/first.parquet"
                ),
                "cohort:analysis_cohort": _binding(tmp_path, "evidence/second.parquet"),
            },
        )


def test_execute_phase_routes_runner_and_gates_to_the_bound_step_cohort() -> None:
    tree = ast.parse(inspect.getsource(run_execute_phase))
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    assert any(
        isinstance(call.func, ast.Name)
        and call.func.id == "_bind_step_execution_cohort"
        for call in calls
    )
    # The per-step runner build moved to the candidate loop (1e5182a): the
    # execute phase binds the cohort, the candidate transition rebuilds the
    # runner on it whenever the bound path diverges from the run cohort.
    from easyicu.research_agent.execution.candidate_loop import (
        _candidate_execute_transition,
    )

    candidate_calls = [
        node
        for node in ast.walk(
            ast.parse(inspect.getsource(_candidate_execute_transition))
        )
        if isinstance(node, ast.Call)
    ]
    runner_builds = [
        call
        for call in candidate_calls
        if isinstance(call.func, ast.Attribute)
        and call.func.attr == "_build_runner"
        and any(
            keyword.arg == "cohort_path"
            and isinstance(keyword.value, ast.Attribute)
            and keyword.value.attr == "step_execution_cohort_path"
            for keyword in call.keywords
        )
    ]
    assert runner_builds
    final_gate_tree = ast.parse(inspect.getsource(_evaluate_final_deterministic_gates))
    assert any(
        isinstance(call.func, ast.Name)
        and call.func.id == "_bound_step_execution_cohort_path"
        for call in ast.walk(final_gate_tree)
        if isinstance(call, ast.Call)
    )


def _run_universe_environment_probe(
    *, ra, tmp_path: Path, step: AnalysisStep, plan: AnalysisPlan
) -> tuple[object, str]:
    cohort_path = tmp_path / f"{step.step_id}_cohort.parquet"
    universe_path = tmp_path / f"{step.step_id}_universe.parquet"
    pd.DataFrame({"stay_id": [1]}).to_parquet(cohort_path, index=False)
    pd.DataFrame({"stay_id": [1, 2]}).to_parquet(universe_path, index=False)
    pipeline = ra.ResearchAgentPipeline(
        workdir=tmp_path / f"work_{step.step_id}",
        enable_memory=False,
        runner_kind="subprocess",
        runner_kwargs={"allow_unsafe_host_fallback": True},
    )
    runner = pipeline._build_runner(
        run_dir=tmp_path / f"run_{step.step_id}",
        cohort_path=cohort_path,
        universe_path=universe_path,
        preselection_universe_authorized=step_may_access_preselection_universe(
            step=step,
            plan=plan,
        ),
    )
    result = runner.run(
        step_id=step.step_id,
        code=(
            "import os\n"
            "from pathlib import Path\n"
            "Path(os.environ['STEP_OUT_DIR'], 'universe_env.txt').write_text("
            "os.environ.get('EASYICU_UNIVERSE_PARQUET', '<absent>'))\n"
        ),
    )
    assert result.succeeded, result.stderr
    observed = (result.out_dir / "universe_env.txt").read_text(encoding="utf-8")
    return runner, observed


def test_ordinary_primary_script_cannot_access_preselection_universe(
    ra, tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("EASYICU_UNIVERSE_PARQUET", "/ambient/forged.parquet")
    step = AnalysisStep(
        step_id="01_primary",
        planned_analysis_role="primary",
        intent="Estimate the primary association on the locked cohort.",
        method="adjusted_association",
        expected_outputs=["table:primary_estimate"],
    )
    plan = AnalysisPlan(research_question="Q", steps=[step])

    runner, observed = _run_universe_environment_probe(
        ra=ra,
        tmp_path=tmp_path,
        step=step,
        plan=plan,
    )

    assert step_may_access_preselection_universe(step=step, plan=plan) is False
    assert "EASYICU_UNIVERSE_PARQUET" not in runner.extra_env
    assert observed == "<absent>"


def test_typed_robustness_script_receives_digest_bound_preselection_universe(
    ra, tmp_path: Path
) -> None:
    step = AnalysisStep(
        step_id="02_robustness",
        intent="Replay the locked robustness grid.",
        method="arbitrary_label_does_not_grant_authority",
        expected_outputs=["table:robustness_summary"],
        robustness_replay_spec=RobustnessReplaySpec(
            products=[
                {
                    "product_id": "robustness_summary",
                    "output": "robustness_summary",
                }
            ]
        ),
    )
    plan = AnalysisPlan(research_question="Q", steps=[step])

    runner, observed = _run_universe_environment_probe(
        ra=ra,
        tmp_path=tmp_path,
        step=step,
        plan=plan,
    )
    without_universe = runner.__class__(
        workdir=tmp_path / "run_without_universe",
        cohort_parquet=runner.cohort_parquet,
        python_executable=runner.python_executable,
        allow_unsafe_host_fallback=True,
    )

    assert step_may_access_preselection_universe(step=step, plan=plan) is True
    assert observed == runner.extra_env["EASYICU_UNIVERSE_PARQUET"]
    assert runner.authority_identity_sha256 != without_universe.authority_identity_sha256
