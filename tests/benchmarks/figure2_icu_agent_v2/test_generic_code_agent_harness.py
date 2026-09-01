from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from benchmarks.figure2_icu_agent_v2.formal_generic_runner import (
    FormalGenericModelGateway,
)
from benchmarks.figure2_icu_agent_v2.design_v2_1 import DesignContractError
from benchmarks.figure2_icu_agent_v2.generic_code_agent_harness import (
    CANONICAL_FILES,
    DockerRunnerBackend,
    GenericCodeAgentHarness,
    GenericExecutionResult,
    PlanReviewDecision,
)
from benchmarks.figure2_icu_agent_v2.formal_provider_gate import FormalCallCoordinate
from benchmarks.figure2_icu_agent_v2.review_bundle_normalizer import (
    normalize_review_bundle,
)
from easyicu.research_agent.providers.protocol import LLMMessage


PLAN = {
    "population": "adult ICU stays",
    "eligibility": "first eligible stay",
    "exposure_or_index": "baseline exposure",
    "outcome": "in-hospital mortality",
    "time_origin": "ICU admission",
    "estimand": "risk difference",
    "method": "binomial regression",
    "missing_data": "complete case with missingness report",
    "diagnostics": ["calibration"],
    "artifacts": ["cohort table", "result table"],
    "limitations": ["observational design"],
}


class _OfflineModel:
    def __init__(self, responses: list[dict]) -> None:
        self.responses = list(responses)
        self.phases: list[str] = []

    def complete(self, *, phase, messages):
        assert messages[0].role == "system"
        self.phases.append(phase)
        return json.dumps(self.responses.pop(0))


class _OfflineExecutor:
    def __init__(self, *, timed_out: bool = False) -> None:
        self.calls = []
        self.timed_out = timed_out

    def execute(self, *, action_id, language, code):
        self.calls.append((action_id, language, code))
        return GenericExecutionResult(
            returncode=-1 if self.timed_out else 0,
            stdout="ok",
            stderr="",
            timed_out=self.timed_out,
            duration_seconds=0.01,
            artifact_paths=(Path("table.csv"),),
        )


def _finalize_action():
    return {
        "action": "finalize",
        "cohort": {"denominator": 42},
        "results": {"estimate": 0.12},
        "diagnostics": {"calibration": "acceptable"},
        "report": "The estimated risk difference was 0.12.",
        "headline_evidence": [
            {"claim": "risk difference", "artifact": "03_results.json"}
        ],
    }


def test_offline_generic_harness_executes_and_writes_complete_bundle(tmp_path: Path):
    model = _OfflineModel(
        [
            PLAN,
            {"action": "execute", "language": "python", "code": "print('ok')"},
            _finalize_action(),
        ]
    )
    executor = _OfflineExecutor()
    harness = GenericCodeAgentHarness(
        model=model,
        executor=executor,
        resource_snapshot=lambda: {
            "within_frozen_budget": True,
            "provider_tokens": 120,
            "billed_cost": 0.01,
        },
    )

    result = harness.run(
        task_prompt="Estimate the association.",
        neutral_input_description="cohort.parquet and dictionary.json",
        output_dir=tmp_path / "bundle",
        review_plan=lambda plan: PlanReviewDecision("approve"),
    )

    assert result.terminal_status == "completed"
    assert sorted(path.name for path in result.output_dir.iterdir()) == sorted(
        CANONICAL_FILES
    )
    assert executor.calls == [("generic_0001", "python", "print('ok')")]
    receipt = json.loads((result.output_dir / "07_run_receipt.json").read_text())
    assert receipt["model_turns"] == 3
    assert receipt["tool_calls"] == 1
    assert receipt["within_frozen_budget"] is True
    manifest = json.loads(
        (result.output_dir / "05_evidence_manifest.json").read_text()
    )
    assert set(manifest["files"]) == {
        "01_plan.json",
        "02_cohort.json",
        "03_results.json",
        "04_diagnostics.json",
        "06_report.md",
    }
    normalized = normalize_review_bundle(result.output_dir)
    assert tuple(normalized.files) == CANONICAL_FILES


def test_one_plan_revision_then_approval(tmp_path: Path):
    revised = {**PLAN, "method": "adjusted binomial regression"}
    model = _OfflineModel([PLAN, revised, _finalize_action()])
    review_calls = []
    harness = GenericCodeAgentHarness(model=model, executor=_OfflineExecutor())

    def review_once(plan):
        review_calls.append(plan)
        return PlanReviewDecision("revise", "adjust for the frozen covariates")

    result = harness.run(
        task_prompt="Estimate the association.",
        neutral_input_description="neutral input package",
        output_dir=tmp_path / "bundle",
        review_plan=review_once,
    )

    assert result.terminal_status == "completed"
    plan = json.loads((result.output_dir / "01_plan.json").read_text())
    assert plan["method"] == "adjusted binomial regression"
    assert model.phases == ["plan", "plan_revision", "execute_or_finalize"]
    assert len(review_calls) == 1


def test_execution_timeout_is_terminal_without_model_retry(tmp_path: Path):
    model = _OfflineModel(
        [PLAN, {"action": "execute", "language": "r", "code": "print(1)"}]
    )
    harness = GenericCodeAgentHarness(
        model=model,
        executor=_OfflineExecutor(timed_out=True),
    )

    result = harness.run(
        task_prompt="Estimate the association.",
        neutral_input_description="neutral input package",
        output_dir=tmp_path / "bundle",
        review_plan=lambda plan: PlanReviewDecision("approve"),
    )

    assert result.failure_category == "execution_timeout"
    assert model.responses == []
    assert sorted(path.name for path in result.output_dir.iterdir()) == sorted(
        CANONICAL_FILES
    )
    receipt = json.loads((result.output_dir / "07_run_receipt.json").read_text())
    assert receipt["failure_category"] == "execution_timeout"
    normalized = normalize_review_bundle(result.output_dir)
    assert tuple(normalized.files) == CANONICAL_FILES


def test_nonfinite_or_extra_action_fields_produce_terminal_bundle(tmp_path: Path):
    model = _OfflineModel([PLAN, {**_finalize_action(), "unexpected": 1}])
    harness = GenericCodeAgentHarness(model=model, executor=_OfflineExecutor())

    result = harness.run(
        task_prompt="Estimate.",
        neutral_input_description="neutral package",
        output_dir=tmp_path / "bundle",
        review_plan=lambda plan: PlanReviewDecision("approve"),
    )

    assert result.failure_category == "agent_output_contract_error"
    assert sorted(path.name for path in result.output_dir.iterdir()) == sorted(
        CANONICAL_FILES
    )


def test_invalid_initial_plan_produces_terminal_bundle(tmp_path: Path):
    model = _OfflineModel([{"population": "missing all other fields"}])
    harness = GenericCodeAgentHarness(model=model, executor=_OfflineExecutor())

    result = harness.run(
        task_prompt="Estimate.",
        neutral_input_description="neutral package",
        output_dir=tmp_path / "bundle",
        review_plan=lambda plan: PlanReviewDecision("approve"),
    )

    assert result.failure_category == "agent_output_contract_error"
    assert sorted(path.name for path in result.output_dir.iterdir()) == sorted(
        CANONICAL_FILES
    )


class _RunnerResult:
    returncode = 0
    stdout = "ok"
    stderr = ""
    timed_out = False
    duration_seconds = 0.1
    artefacts = []


class _RunnerSpy:
    def __init__(self):
        self.calls = []

    def run(self, *, step_id, code):
        self.calls.append((step_id, code))
        return _RunnerResult()


@pytest.mark.parametrize(
    ("language", "interpreter"), [("r", "Rscript"), ("shell", "bash")]
)
def test_docker_adapter_uses_fixed_interpreter_without_host_shell(
    language: str,
    interpreter: str,
):
    runner = _RunnerSpy()
    backend = DockerRunnerBackend(runner)

    backend.execute(action_id="generic_0001", language=language, code="echo $HOME")

    wrapper = runner.calls[0][1]
    assert f"subprocess.run([{interpreter!r}, str(script_path)]" in wrapper
    assert "shell=True" not in wrapper


def test_formal_gateway_denies_before_transport():
    class TransportSpy:
        name = "transport-spy"

        def __init__(self):
            self.called = False

        def complete(self, messages, **kwargs):
            del messages, kwargs
            self.called = True
            return "unexpected"

    transport = TransportSpy()
    gateway = FormalGenericModelGateway(
        client=transport,
        receipts={},
        scope="qualification12",
        task_id="qualification12_a_01",
        max_tokens=100,
        temperature=0.0,
    )

    with pytest.raises(DesignContractError) as exc_info:
        gateway.complete(
            phase="plan",
            messages=[LLMMessage(role="user", content="do not send")],
        )

    assert getattr(exc_info.value, "reason_code", None) == (
        "FORMAL_PROVIDER_CALL_NOT_AUTHORIZED"
    )
    assert transport.called is False


def test_formal_runner_source_has_no_provider_transport_bypass():
    source_path = Path(__file__).parents[3] / (
        "benchmarks/figure2_icu_agent_v2/formal_generic_runner.py"
    )
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    imported = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    }
    called_names = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }

    assert "easyicu.research_agent.providers.client_trust" not in imported
    assert "authorized_complete" not in called_names
    assert "complete_formal_provider_call" in called_names
    assert not any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "complete"
        for node in ast.walk(tree)
    )


def test_formal_coordinate_remains_generic_arm_owned():
    coordinate = FormalCallCoordinate(
        scope="core_wp2_wp3",
        task_id="icu27_t01",
        arm="generic_code_agent",
        call_id="generic_0001",
    )
    assert coordinate.arm == "generic_code_agent"
