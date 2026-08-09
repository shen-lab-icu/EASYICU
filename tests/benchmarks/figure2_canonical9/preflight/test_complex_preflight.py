"""Zero-Provider preflight coverage for M2, H1, H2, and H3.

These are diagnostic fixtures, never paper results.  Structural and reviewed-
code checks always run; the full real-pipeline graph runs only when the exact
CodeRunner isolation capability is available.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

from benchmarks.figure2_canonical9.preflight.fixtures import (
    COMPLEX_CASES,
    H1,
    H2,
    H3,
    M2,
    PreflightCase,
)
from benchmarks.figure2_canonical9.preflight.harness import (
    PreflightRun,
    ScriptedPreflightLLM,
    paper_acceptance_verdict,
    preflight_runtime_manifest,
    provider_transport_spy,
    run_preflight,
)
from easyicu.research_agent.agents.core import PlannerAgent
from easyicu.research_agent.execution.runner import CodeRunner
from easyicu.research_agent.execution.runners.selection import (
    select_standard_executor,
)
from easyicu.research_agent.providers.protocol import LLMMessage
from easyicu.research_agent.schema import CohortDescriptor, ResearchContext

CASES = list(COMPLEX_CASES.values())


@pytest.fixture(params=CASES, ids=[case.task_id for case in CASES])
def case(request) -> PreflightCase:
    return request.param


def _reviewed_primary_code(case: PreflightCase) -> tuple[str, ScriptedPreflightLLM]:
    controller = ScriptedPreflightLLM(case)
    code = controller.client.complete(
        [
            LLMMessage(
                role="user",
                content=f"WRITE THE PYTHON CODE {case.primary_step_id}",
            )
        ]
    )
    return code, controller


def test_complex_registry_is_exact_minimum_family_set() -> None:
    assert set(COMPLEX_CASES) == {
        "m2_mortality_prediction",
        "h1_ventilation_survival",
        "h2_vasopressor_causal",
        "h3_trajectory_clustering",
    }


def test_generated_blueprint_notes_cannot_widen_single_database_contract() -> None:
    """Generic prompt examples must not create a cross-database obligation."""

    authority_context = ResearchContext(
        research_question=M2.question,
        cohort=CohortDescriptor(
            cohort_name="m2_preflight",
            database=M2.database,
            n_stays=160,
            n_patients=80,
        ),
        variables=[],
        target_outcome=M2.target_outcome,
    )
    prompt_context = authority_context.model_copy(
        update={
            "notes": (
                "ANALYSIS BLUEPRINT: generic examples may discuss external "
                "validation and transportability, without requesting either."
            )
        }
    )

    parsed = PlannerAgent(object())._parse(
        M2.build_plan().model_dump_json(),
        prompt_context,
        enforce_article_contract=True,
        article_contract_context=authority_context,
    )

    assert parsed.analysis_type == "prediction_model"


def test_complex_plan_contract_and_typed_inputs(case: PreflightCase) -> None:
    plan = case.build_plan()
    cohort = case.build_cohort(160)

    assert plan.analysis_type == case.analysis_type
    assert [step.step_id for step in plan.steps].count(case.primary_step_id) == 1
    assert (
        next(
            step
            for step in plan.steps
            if step.step_id == case.primary_step_id
        ).planned_analysis_role
        == "primary"
    )
    missing_raw_inputs = {
        input_name
        for step in plan.steps
        for input_name in step.inputs
        if ":" not in input_name and input_name not in cohort.columns
    }
    assert missing_raw_inputs == set()

    table_one = next(
        step for step in plan.steps if step.step_id == case.deterministic_step_id
    )
    selection = select_standard_executor(table_one, plan=plan)
    assert selection is not None
    assert selection.analysis_kind == "grouped_table_one"


def test_complex_guardrails_are_live_and_structurally_covered(
    case: PreflightCase,
) -> None:
    assert len(case.guardrail_checks) == len(case.semantic_guardrails)
    assert sorted(check.guardrail_index for check in case.guardrail_checks) == list(
        range(len(case.semantic_guardrails))
    )
    failures = [
        (check.key, case.semantic_guardrails[check.guardrail_index])
        for check in case.guardrail_checks
        if not check.holds(case)
    ]
    assert failures == []


def test_complex_product_map_covers_live_suite_outputs(case: PreflightCase) -> None:
    mapped = case.product_mapping()
    assert [product for product, _mapping in mapped] == list(case.expected_products)


def test_complex_coder_dependencies_are_importable(case: PreflightCase) -> None:
    missing = [
        package
        for package in case.required_imports
        if importlib.util.find_spec(package) is None
    ]
    assert missing == []


def test_h3_has_independent_deterministic_stability_owner() -> None:
    plan = H3.build_plan()
    stability = next(step for step in plan.steps if step.step_id == "04_stability_freeze")
    selection = select_standard_executor(stability, plan=plan)

    assert selection is not None
    assert selection.analysis_kind == "trajectory_cluster_stability"
    assert "run_trajectory_stability" in selection.code


@pytest.mark.parametrize(
    ("case", "required_files"),
    [
        (
            M2,
            {
                "model_performance_train_test.csv",
                "risk_predictions_test.csv",
                "calibration_curve.csv",
                "decision_curve.csv",
                "split_definition.csv",
            },
        ),
        (
            H1,
            {"cox_summary.csv", "survival_curve.csv", "ph_diagnostics.csv"},
        ),
        (
            H2,
            {
                "causal_effect.csv",
                "covariate_balance.csv",
                "positivity_diagnostics.csv",
                "assignment_model.csv",
            },
        ),
        (
            H3,
            {
                "cluster_assignments.csv",
                "cluster_outcomes.csv",
                "cluster_trajectory_means.csv",
            },
        ),
    ],
    ids=["m2_prediction", "h1_survival", "h2_causal", "h3_trajectory"],
)
def test_reviewed_family_code_executes_with_zero_provider_calls(
    case: PreflightCase,
    required_files: set[str],
    tmp_path: Path,
) -> None:
    """Development smoke for static reviewed code, not a paper-authority run."""

    cohort_path = tmp_path / "cohort.parquet"
    case.build_cohort(160).to_parquet(cohort_path, index=False)
    with provider_transport_spy() as spy:
        code, _controller = _reviewed_primary_code(case)
        runner = CodeRunner(
            workdir=tmp_path / "run",
            cohort_parquet=cohort_path,
            timeout_seconds=60.0,
            network_policy="none",
            # Explicit development-only fallback: the formal graph test below
            # still requires exact non-degraded isolation.
            allow_unsafe_host_fallback=True,
        )
        result = runner.run(step_id=case.primary_step_id, code=code)

    assert spy.calls == 0
    assert result.succeeded, result.stderr
    assert required_files.issubset({path.name for path in result.artefacts})
    summary = json.loads(
        (result.out_dir / "step_summary.json").read_text(encoding="utf-8")
    )
    assert summary.get("status") == "ok" or summary.get("method")
    if case is M2:
        split_frame = pd.read_csv(result.out_dir / "split_definition.csv")
        assert int(split_frame.loc[0, "patient_overlap_n"]) == 0
        metrics = pd.read_csv(result.out_dir / "model_performance_train_test.csv")
        assert {
            "auroc",
            "average_precision",
            "recall",
            "f1",
            "brier",
            "calibration_slope",
        }.issubset(metrics.columns)


def test_production_evaluator_rejects_each_complex_mock_run(
    case: PreflightCase,
    tmp_path: Path,
) -> None:
    _code, controller = _reviewed_primary_code(case)
    run = PreflightRun(
        case=case,
        run_dir=tmp_path,
        run_id="diagnostic-only",
        manifest={},
        llm=controller,
    )
    verdict = paper_acceptance_verdict(run)

    assert verdict.status == "invalid"
    assert verdict.issues


@pytest.fixture(scope="module", params=CASES, ids=[case.task_id for case in CASES])
def complex_graph_run(request, tmp_path_factory) -> PreflightRun:
    manifest = preflight_runtime_manifest()
    if not manifest.integration_ready:
        pytest.skip(manifest.blocked_reason or "integration_not_ready")
    case = request.param
    return run_preflight(
        case,
        workdir=tmp_path_factory.mktemp(f"complex_{case.task_id}"),
        n_rows=160,
    )


def test_complex_real_graph_is_zero_provider_and_finalises(
    complex_graph_run: PreflightRun,
) -> None:
    run = complex_graph_run
    assert run.pipeline_ran is True
    assert run.external_provider_calls == 0
    assert run.raised is None
    assert run.record(run.case.deterministic_step_id)["status"] == "ok"
    assert run.record(run.case.primary_step_id)
    assert run.manifest.get("readiness", {}).get("paper_authorized") is not True
    assert paper_acceptance_verdict(run).status == "invalid"
