from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.figure2_icu_agent_v2.formal_generic_runner import (
    FormalGenericModelGateway,
)
from benchmarks.figure2_icu_agent_v2.design_v2_1 import DesignContractError
from benchmarks.figure2_icu_agent_v2.generic_code_agent_harness import (
    DockerRunnerBackend,
    GenericCodeAgentHarness,
    GenericExecutionResult,
    GenericBudgetExhausted,
    PlanReviewDecision,
)
from benchmarks.figure2_icu_agent_v2.review_bundle_semantics import CANONICAL_FILES
from benchmarks.figure2_icu_agent_v2.formal_provider_gate import (
    FormalCallCoordinate,
    FormalProviderSession,
)
from benchmarks.figure2_icu_agent_v2.review_bundle_normalizer import (
    ReviewBlindingContext,
    normalize_review_bundle,
)


_BLINDING_CONTEXT = ReviewBlindingContext(
    host_markers=("fig2-server-01", "fig2-laptop-01"),
    output_roots=("/formal/server", "/formal/laptop"),
)


def _normalize(source_dir: Path):
    return normalize_review_bundle(
        source_dir,
        blinding_context=_BLINDING_CONTEXT,
    )
from easyicu.research_agent.providers.protocol import LLMMessage
from easyicu.research_agent.execution.runner import DockerRunner
from easyicu.research_agent.authority.provider_hard_stop import (
    ProviderHardStopLedger,
    ProviderHardStopLimits,
)


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
MANDATORY_ARTIFACTS = ("cohort flow", "result table", "core diagnostic")


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
        "artifact_inventory": {
            "cohort flow": ["02_cohort.json"],
            "result table": ["03_results.json"],
            "core diagnostic": ["04_diagnostics.json"],
        },
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
        mandatory_artifacts=MANDATORY_ARTIFACTS,
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
    assert set(manifest["harness_computed_file_digests"]) == {
        "01_plan.json",
        "02_cohort.json",
        "03_results.json",
        "04_diagnostics.json",
        "06_report.md",
    }
    assert manifest["agent_asserted_headline_evidence"] == [
        {"claim": "risk difference", "artifact": "03_results.json"}
    ]
    assert receipt["agent_asserted_mandatory_artifact_presence"] == {
        "cohort flow": True,
        "result table": True,
        "core diagnostic": True,
    }
    assert receipt["substantive_output_files"] == {
        "02_cohort.json": True,
        "03_results.json": True,
        "04_diagnostics.json": True,
        "06_report.md": True,
    }
    normalized = _normalize(result.output_dir)
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
        mandatory_artifacts=MANDATORY_ARTIFACTS,
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
        [PLAN, {"action": "execute", "language": "shell", "code": "echo 1"}]
    )
    harness = GenericCodeAgentHarness(
        model=model,
        executor=_OfflineExecutor(timed_out=True),
    )

    result = harness.run(
        task_prompt="Estimate the association.",
        neutral_input_description="neutral input package",
        mandatory_artifacts=MANDATORY_ARTIFACTS,
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
    assert receipt["agent_asserted_mandatory_artifact_presence"] == {
        label: False for label in MANDATORY_ARTIFACTS
    }
    normalized = _normalize(result.output_dir)
    assert tuple(normalized.files) == CANONICAL_FILES


def test_nonfinite_or_extra_action_fields_produce_terminal_bundle(tmp_path: Path):
    model = _OfflineModel([PLAN, {**_finalize_action(), "unexpected": 1}])
    harness = GenericCodeAgentHarness(model=model, executor=_OfflineExecutor())

    result = harness.run(
        task_prompt="Estimate.",
        neutral_input_description="neutral package",
        mandatory_artifacts=MANDATORY_ARTIFACTS,
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
        mandatory_artifacts=MANDATORY_ARTIFACTS,
        output_dir=tmp_path / "bundle",
        review_plan=lambda plan: PlanReviewDecision("approve"),
    )

    assert result.failure_category == "agent_output_contract_error"
    assert sorted(path.name for path in result.output_dir.iterdir()) == sorted(
        CANONICAL_FILES
    )


def test_shared_budget_exhaustion_produces_terminal_bundle(tmp_path: Path):
    class ExhaustedModel:
        def complete(self, *, phase, messages):
            del phase, messages
            raise GenericBudgetExhausted

    harness = GenericCodeAgentHarness(
        model=ExhaustedModel(),
        executor=_OfflineExecutor(),
        resource_snapshot=lambda: {"within_frozen_budget": False},
    )

    result = harness.run(
        task_prompt="Estimate.",
        neutral_input_description="neutral package",
        mandatory_artifacts=MANDATORY_ARTIFACTS,
        output_dir=tmp_path / "bundle",
        review_plan=lambda plan: PlanReviewDecision("approve"),
    )

    assert result.failure_category == "budget_exhausted"
    receipt = json.loads((result.output_dir / "07_run_receipt.json").read_text())
    assert receipt["within_frozen_budget"] is False
    assert all(
        not value
        for value in receipt[
            "agent_asserted_mandatory_artifact_presence"
        ].values()
    )


def test_empty_referenced_result_is_reported_absent(tmp_path: Path):
    action = _finalize_action()
    action["results"] = {}
    harness = GenericCodeAgentHarness(
        model=_OfflineModel([PLAN, action]),
        executor=_OfflineExecutor(),
    )

    result = harness.run(
        task_prompt="Estimate.",
        neutral_input_description="neutral package",
        mandatory_artifacts=MANDATORY_ARTIFACTS,
        output_dir=tmp_path / "bundle",
        review_plan=lambda plan: PlanReviewDecision("approve"),
    )

    receipt = json.loads((result.output_dir / "07_run_receipt.json").read_text())
    assert result.terminal_status == "completed"
    assert receipt["agent_asserted_mandatory_artifact_presence"][
        "result table"
    ] is False
    assert receipt["substantive_output_files"]["03_results.json"] is False


class _RunnerResult:
    returncode = 0
    stdout = "ok"
    stderr = ""
    timed_out = False
    duration_seconds = 0.1
    artefacts = []


class _RunnerSpy(DockerRunner):
    def __init__(self):
        self.calls = []
        self.network = "none"
        self.cpu_limit = "4"
        self.memory_limit = "8g"
        self.pids_limit = 256
        self.open_files_limit = 4096
        self.timeout_seconds = 300.0

    def run(self, *, step_id, code):
        self.calls.append((step_id, code))
        return _RunnerResult()


def test_docker_adapter_uses_fixed_interpreter_without_host_shell(
):
    runner = _RunnerSpy()
    backend = DockerRunnerBackend(runner)

    backend.execute(
        action_id="generic_0001", language="shell", code="echo $HOME"
    )

    wrapper = runner.calls[0][1]
    assert "subprocess.run(['bash', str(script_path)]" in wrapper
    assert "shell=True" not in wrapper


def test_docker_adapter_rejects_non_docker_backend():
    with pytest.raises(TypeError, match="requires a DockerRunner"):
        DockerRunnerBackend(object())


def test_formal_gateway_denies_before_transport(tmp_path: Path):
    class TransportSpy:
        name = "transport-spy"

        def __init__(self):
            self.called = False

        def complete(self, messages, **kwargs):
            del messages, kwargs
            self.called = True
            return "unexpected"

    transport = TransportSpy()
    hard_stop = ProviderHardStopLedger(
        path=tmp_path / "formal-gateway-hard-stop.json",
        task_ids=("qualification12_a_01",),
        limits=ProviderHardStopLimits(
            max_provider_attempts_per_run=1,
            max_provider_attempts_per_batch=1,
            max_total_tokens_per_run=100,
            max_total_tokens_per_batch=100,
            max_estimated_cost_usd_per_batch=1.0,
            max_wall_clock_seconds_per_task=60.0,
            input_cost_usd_per_million_tokens=0.1,
            output_cost_usd_per_million_tokens=0.1,
        ),
        batch_id="formal-gateway-test",
    ).start_task("qualification12_a_01")
    gateway = FormalGenericModelGateway(
        client=transport,
        session=FormalProviderSession(
            receipts={},
            scope="qualification12",
            task_id="qualification12_a_01",
            arm="generic_code_agent",
            execution_site="server",
            provider_hard_stop=hard_stop,
        ),
        max_tokens=100,
        temperature=0.0,
    )

    with pytest.raises(DesignContractError) as exc_info:
        gateway.complete(
            phase="plan",
            messages=[LLMMessage(role="user", content="do not send")],
        )

    assert getattr(exc_info.value, "reason_code", None) == (
        "FORMAL_AUTHORITY_SIGNER_NOT_REGISTERED"
    )
    assert transport.called is False


def test_formal_gateway_requires_governed_provider_session():
    with pytest.raises(TypeError, match="FormalProviderSession"):
        FormalGenericModelGateway(
            client=object(),
            session=object(),  # type: ignore[arg-type]
            max_tokens=100,
            temperature=0.0,
        )


def test_formal_coordinate_remains_generic_arm_owned():
    coordinate = FormalCallCoordinate(
        scope="core_wp2_wp3",
        task_id="icu27_t01",
        arm="generic_code_agent",
        execution_site="server",
        call_id="generic_0001",
    )
    assert coordinate.arm == "generic_code_agent"
