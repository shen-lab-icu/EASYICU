from __future__ import annotations

import ast
import hashlib
import inspect
from pathlib import Path

import pytest

from easyicu.research_agent.schema import AnalysisStep
from easyicu.research_agent.execution.cohort_routing import (
    StepExecutionCohortRoutingError,
    bind_step_execution_cohort,
    bound_step_execution_cohort_path,
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
    assert step_record["execution_cohort_sha256"] == hashlib.sha256(
        fallback.read_bytes()
    ).hexdigest()
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
    runner_builds = [
        call
        for call in calls
        if isinstance(call.func, ast.Attribute)
        and call.func.attr == "_build_runner"
        and any(
            keyword.arg == "cohort_path"
            and isinstance(keyword.value, ast.Name)
            and keyword.value.id == "step_execution_cohort_path"
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
