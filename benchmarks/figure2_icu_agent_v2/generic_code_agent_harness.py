"""Generic coding-agent baseline for the Figure 2 v2.1 experiment.

The harness deliberately owns orchestration, not scientific governance.  It
offers a neutral plan/execute/inspect/repair/finalize loop and produces the
shared review bundle.  Formal Provider access is owned by
``formal_generic_runner``; tests and Dev9 may inject an offline model gateway.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence

from easyicu.research_agent.execution.runner import DockerRunner
from easyicu.research_agent.authority.provider_hard_stop import (
    ProviderHardStopExceeded,
    TaskProviderHardStop,
)
from easyicu.research_agent.providers.protocol import LLMMessage

from .review_bundle_writer import (
    ReviewBundleMaterial,
    ReviewBundleWriteError,
    ReviewBundleWriter,
    terminal_failure_material,
    write_review_bundle,
)
from .review_bundle_semantics import (
    FailureCategory,
    ReviewResourceReceipt,
    TerminalOutcome,
    TerminalStatus,
    validate_review_task_id,
)


SPEC_PATH = Path(__file__).with_name("generic_code_agent_spec_v1.json")
PLAN_FIELDS = (
    "population",
    "eligibility",
    "exposure_or_index",
    "outcome",
    "time_origin",
    "estimand",
    "method",
    "missing_data",
    "diagnostics",
    "artifacts",
    "limitations",
)
LIST_PLAN_FIELDS = frozenset({"diagnostics", "artifacts", "limitations"})
ALLOWED_LANGUAGES = frozenset({"python", "shell"})


class GenericHarnessError(RuntimeError):
    """Typed, arm-neutral failure at the generic harness boundary."""

    owner = "figure2.generic_code_agent_harness_v1"

    def __init__(self, reason_code: str, message: str) -> None:
        self.reason_code = reason_code
        self.easyicu_safe_diagnostic = {
            "owner": self.owner,
            "reason_code": reason_code,
        }
        super().__init__(f"{reason_code}: {message}")


class GenericBudgetExhausted(RuntimeError):
    """The shared task budget denied further model or execution work."""

    reason_code = "GENERIC_BUDGET_EXHAUSTED"


@dataclass(frozen=True)
class PlanReviewDecision:
    """One review opportunity; ``revise`` is conditional approval after one edit."""

    disposition: str
    feedback: str = ""

    def __post_init__(self) -> None:
        if self.disposition not in {"approve", "revise", "reject"}:
            raise ValueError("plan disposition must be approve, revise, or reject")
        if self.disposition == "revise" and not self.feedback.strip():
            raise ValueError("a revision decision requires feedback")


@dataclass(frozen=True)
class GenericExecutionResult:
    returncode: int
    stdout: str
    stderr: str
    timed_out: bool
    duration_seconds: float
    artifact_paths: tuple[Path, ...] = ()


class GenericModelGateway(Protocol):
    """One model turn; formal implementations must use the formal gate."""

    def complete(self, *, phase: str, messages: Sequence[LLMMessage]) -> str: ...


class GenericExecutionBackend(Protocol):
    """Language-neutral execution boundary for the generic baseline."""

    def execute(
        self,
        *,
        action_id: str,
        language: str,
        code: str,
    ) -> GenericExecutionResult: ...


class DockerRunnerBackend:
    """Adapt the isolated Docker runner to Python and in-container shell."""

    def __init__(
        self,
        runner: DockerRunner,
        *,
        task_hard_stop: TaskProviderHardStop | None = None,
    ) -> None:
        if not isinstance(runner, DockerRunner):
            raise TypeError("DockerRunnerBackend requires a DockerRunner instance")
        if runner.network != "none":
            raise ValueError("generic formal execution requires Docker network='none'")
        if not all(
            (
                runner.cpu_limit,
                runner.memory_limit,
                runner.pids_limit,
                runner.open_files_limit,
            )
        ):
            raise ValueError("generic formal execution requires every resource ceiling")
        self._runner = runner
        self._task_hard_stop = task_hard_stop

    @staticmethod
    def _interpreter_wrapper(language: str, code: str) -> str:
        if language != "shell":
            raise ValueError(f"unsupported wrapper language: {language}")
        return (
            "from pathlib import Path\n"
            "import subprocess\n"
            "import sys\n\n"
            "script_path = Path('/tmp/generic-agent-action.sh')\n"
            f"script_path.write_text({code!r}, encoding='utf-8')\n"
            "completed = subprocess.run(['bash', str(script_path)], check=False)\n"
            "raise SystemExit(completed.returncode)\n"
        )

    def execute(
        self,
        *,
        action_id: str,
        language: str,
        code: str,
    ) -> GenericExecutionResult:
        normalized_language = str(language).strip().lower()
        if normalized_language not in ALLOWED_LANGUAGES:
            raise GenericHarnessError(
                "GENERIC_UNSUPPORTED_LANGUAGE",
                f"unsupported execution language: {language!r}",
            )
        runner_code = (
            code
            if normalized_language == "python"
            else self._interpreter_wrapper(normalized_language, code)
        )
        original_timeout = self._runner.timeout_seconds
        budget_limited = False
        if self._task_hard_stop is not None:
            try:
                bounded_timeout = self._task_hard_stop.cap_timeout(original_timeout)
            except ProviderHardStopExceeded as exc:
                raise GenericBudgetExhausted from exc
            budget_limited = bounded_timeout < original_timeout
            self._runner.timeout_seconds = bounded_timeout
        try:
            result = self._runner.run(step_id=action_id, code=runner_code)
        finally:
            self._runner.timeout_seconds = original_timeout
        if self._task_hard_stop is not None:
            if result.timed_out and budget_limited:
                raise GenericBudgetExhausted
            try:
                self._task_hard_stop.assert_active()
            except ProviderHardStopExceeded as exc:
                raise GenericBudgetExhausted from exc
        return GenericExecutionResult(
            returncode=int(result.returncode),
            stdout=str(result.stdout),
            stderr=str(result.stderr),
            timed_out=bool(result.timed_out),
            duration_seconds=float(result.duration_seconds),
            artifact_paths=tuple(Path(path) for path in result.artefacts),
        )


@dataclass(frozen=True)
class GenericHarnessResult:
    terminal_status: TerminalStatus
    failure_category: FailureCategory | None
    output_dir: Path
    model_turns: int
    tool_calls: int


def _load_json_object(raw: str, *, reason_code: str) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    try:
        value = json.loads(
            raw,
            object_pairs_hook=reject_duplicates,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant: {token}")
            ),
        )
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        raise GenericHarnessError(reason_code, "model response is not strict JSON") from exc
    if not isinstance(value, dict):
        raise GenericHarnessError(reason_code, "model response must be a JSON object")
    _reject_nonfinite(value, reason_code=reason_code)
    return value


def _reject_nonfinite(value: Any, *, reason_code: str) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise GenericHarnessError(reason_code, "non-finite numeric value")
    if isinstance(value, dict):
        for nested in value.values():
            _reject_nonfinite(nested, reason_code=reason_code)
    elif isinstance(value, list):
        for nested in value:
            _reject_nonfinite(nested, reason_code=reason_code)


def _validate_plan(value: Mapping[str, Any]) -> dict[str, Any]:
    if set(value) != set(PLAN_FIELDS):
        raise GenericHarnessError(
            "GENERIC_PLAN_CONTRACT_INVALID",
            "plan fields do not match the frozen neutral plan contract",
        )
    normalized: dict[str, Any] = {}
    for field in PLAN_FIELDS:
        item = value[field]
        if field in LIST_PLAN_FIELDS:
            if not isinstance(item, list) or not all(
                isinstance(entry, str) and entry.strip() for entry in item
            ):
                raise GenericHarnessError(
                    "GENERIC_PLAN_CONTRACT_INVALID",
                    f"{field} must be a list of non-empty strings",
                )
            normalized[field] = list(item)
        elif not isinstance(item, str) or not item.strip():
            raise GenericHarnessError(
                "GENERIC_PLAN_CONTRACT_INVALID",
                f"{field} must be a non-empty string",
            )
        else:
            normalized[field] = item
    return normalized


def _validate_mandatory_artifacts(values: Sequence[str]) -> tuple[str, ...]:
    normalized = tuple(str(value).strip() for value in values)
    if not normalized or any(not value for value in normalized):
        raise ValueError("mandatory artifacts must be non-empty strings")
    if len(set(normalized)) != len(normalized):
        raise ValueError("mandatory artifacts must be unique")
    return normalized


class GenericCodeAgentHarness:
    """Run one fresh generic-agent task under externally enforced budgets."""

    def __init__(
        self,
        *,
        task_id: str,
        model: GenericModelGateway,
        executor: GenericExecutionBackend,
        resource_snapshot: Callable[[], Mapping[str, Any]] | None = None,
    ) -> None:
        spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
        self._task_id = validate_review_task_id(task_id)
        self._system_prompt = str(spec["system_prompt"])
        self._model = model
        self._executor = executor
        self._resource_snapshot = resource_snapshot or (
            lambda: {"within_frozen_budget": False}
        )

    def _model_json(
        self,
        *,
        phase: str,
        messages: list[LLMMessage],
        reason_code: str,
    ) -> dict[str, Any]:
        raw = self._model.complete(phase=phase, messages=tuple(messages))
        messages.append(LLMMessage(role="assistant", content=raw))
        return _load_json_object(raw, reason_code=reason_code)

    def run(
        self,
        *,
        task_prompt: str,
        neutral_input_description: str,
        mandatory_artifacts: Sequence[str],
        output_dir: Path,
        review_plan: Callable[[Mapping[str, Any]], PlanReviewDecision],
    ) -> GenericHarnessResult:
        if not task_prompt.strip() or not neutral_input_description.strip():
            raise ValueError("task prompt and neutral input description are required")
        required_artifacts = _validate_mandatory_artifacts(mandatory_artifacts)
        try:
            destination = ReviewBundleWriter(Path(output_dir)).output_dir
        except ReviewBundleWriteError as exc:
            raise GenericHarnessError(
                "GENERIC_OUTPUT_DIRECTORY_UNSAFE",
                str(exc),
            ) from exc
        started = time.monotonic()
        model_turns = 0
        tool_calls = 0
        messages = [
            LLMMessage(role="system", content=self._system_prompt),
            LLMMessage(
                role="user",
                content=(
                    f"Research task:\n{task_prompt.strip()}\n\n"
                    f"Neutral input package:\n{neutral_input_description.strip()}\n\n"
                    "Return only the frozen plan JSON object."
                ),
            ),
        ]

        try:
            plan = _validate_plan(
                self._model_json(
                    phase="plan",
                    messages=messages,
                    reason_code="GENERIC_PLAN_RESPONSE_INVALID",
                )
            )
        except GenericBudgetExhausted:
            return self._write_terminal_failure(
                destination=destination,
                plan=self._unavailable_plan(),
                category=FailureCategory.BUDGET_EXHAUSTED,
                mandatory_artifacts=required_artifacts,
                model_turns=1,
                tool_calls=tool_calls,
                started=started,
            )
        except GenericHarnessError:
            return self._write_terminal_failure(
                destination=destination,
                plan=self._unavailable_plan(),
                category=FailureCategory.AGENT_OUTPUT_CONTRACT_ERROR,
                mandatory_artifacts=required_artifacts,
                model_turns=1,
                tool_calls=tool_calls,
                started=started,
            )
        model_turns += 1
        decision = review_plan(plan)
        if decision.disposition == "reject":
            return self._write_terminal_failure(
                destination=destination,
                plan=plan,
                category=FailureCategory.PLAN_REJECTED,
                mandatory_artifacts=required_artifacts,
                model_turns=model_turns,
                tool_calls=tool_calls,
                started=started,
            )
        if decision.disposition == "revise":
            messages.append(
                LLMMessage(
                    role="user",
                    content=(
                        "Revise the plan once using this neutral human feedback. "
                        "Return only the complete frozen plan JSON object:\n"
                        + decision.feedback.strip()
                    ),
                )
            )
            try:
                plan = _validate_plan(
                    self._model_json(
                        phase="plan_revision",
                        messages=messages,
                        reason_code="GENERIC_PLAN_RESPONSE_INVALID",
                    )
                )
            except GenericBudgetExhausted:
                return self._write_terminal_failure(
                    destination=destination,
                    plan=plan,
                    category=FailureCategory.BUDGET_EXHAUSTED,
                    mandatory_artifacts=required_artifacts,
                    model_turns=model_turns + 1,
                    tool_calls=tool_calls,
                    started=started,
                )
            except GenericHarnessError:
                return self._write_terminal_failure(
                    destination=destination,
                    plan=plan,
                    category=FailureCategory.AGENT_OUTPUT_CONTRACT_ERROR,
                    mandatory_artifacts=required_artifacts,
                    model_turns=model_turns + 1,
                    tool_calls=tool_calls,
                    started=started,
                )
            model_turns += 1

        messages.append(
            LLMMessage(
                role="user",
                content=(
                    "The plan is approved and locked. Continue with one JSON action. "
                    "Use {\"action\":\"execute\",\"language\":\"python|shell\","
                    "\"code\":\"...\"} or finalize with "
                    "{\"action\":\"finalize\",\"cohort\":{},\"results\":{},"
                    "\"diagnostics\":{},\"report\":\"...\","
                    "\"headline_evidence\":[],\"artifact_inventory\":{}}. "
                    "artifact_inventory must map every required artifact label to "
                    "one or more canonical review files. Required labels:\n"
                    + json.dumps(required_artifacts, ensure_ascii=False)
                ),
            )
        )

        while True:
            try:
                action = self._model_json(
                    phase="execute_or_finalize",
                    messages=messages,
                    reason_code="GENERIC_ACTION_RESPONSE_INVALID",
                )
            except GenericBudgetExhausted:
                return self._write_terminal_failure(
                    destination=destination,
                    plan=plan,
                    category=FailureCategory.BUDGET_EXHAUSTED,
                    mandatory_artifacts=required_artifacts,
                    model_turns=model_turns + 1,
                    tool_calls=tool_calls,
                    started=started,
                )
            except GenericHarnessError:
                return self._write_terminal_failure(
                    destination=destination,
                    plan=plan,
                    category=FailureCategory.AGENT_OUTPUT_CONTRACT_ERROR,
                    mandatory_artifacts=required_artifacts,
                    model_turns=model_turns + 1,
                    tool_calls=tool_calls,
                    started=started,
                )
            model_turns += 1
            action_name = action.get("action")
            if action_name == "execute":
                if set(action) != {"action", "language", "code"}:
                    return self._write_terminal_failure(
                        destination=destination,
                        plan=plan,
                        category=FailureCategory.AGENT_OUTPUT_CONTRACT_ERROR,
                        mandatory_artifacts=required_artifacts,
                        model_turns=model_turns,
                        tool_calls=tool_calls,
                        started=started,
                    )
                language = str(action["language"]).strip().lower()
                code = action["code"]
                if language not in ALLOWED_LANGUAGES or not isinstance(code, str) or not code:
                    return self._write_terminal_failure(
                        destination=destination,
                        plan=plan,
                        category=FailureCategory.AGENT_OUTPUT_CONTRACT_ERROR,
                        mandatory_artifacts=required_artifacts,
                        model_turns=model_turns,
                        tool_calls=tool_calls,
                        started=started,
                    )
                tool_calls += 1
                try:
                    observation = self._executor.execute(
                        action_id=f"generic_{tool_calls:04d}",
                        language=language,
                        code=code,
                    )
                except GenericBudgetExhausted:
                    return self._write_terminal_failure(
                        destination=destination,
                        plan=plan,
                        category=FailureCategory.BUDGET_EXHAUSTED,
                        mandatory_artifacts=required_artifacts,
                        model_turns=model_turns,
                        tool_calls=tool_calls,
                        started=started,
                    )
                if observation.timed_out:
                    return self._write_terminal_failure(
                        destination=destination,
                        plan=plan,
                        category=FailureCategory.EXECUTION_TIMEOUT,
                        mandatory_artifacts=required_artifacts,
                        model_turns=model_turns,
                        tool_calls=tool_calls,
                        started=started,
                    )
                messages.append(
                    LLMMessage(
                        role="user",
                        content=(
                            "Execution observation (inspect, then repair or finalize):\n"
                            + json.dumps(
                                {
                                    "returncode": observation.returncode,
                                    "stdout": observation.stdout,
                                    "stderr": observation.stderr,
                                    "timed_out": observation.timed_out,
                                    "artifacts": [
                                        path.name for path in observation.artifact_paths
                                    ],
                                },
                                ensure_ascii=False,
                                allow_nan=False,
                            )
                        ),
                    )
                )
                continue
            if action_name == "finalize":
                try:
                    return self._write_success(
                        destination=destination,
                        plan=plan,
                        action=action,
                        mandatory_artifacts=required_artifacts,
                        model_turns=model_turns,
                        tool_calls=tool_calls,
                        started=started,
                    )
                except GenericHarnessError:
                    return self._write_terminal_failure(
                        destination=destination,
                        plan=plan,
                        category=FailureCategory.AGENT_OUTPUT_CONTRACT_ERROR,
                        mandatory_artifacts=required_artifacts,
                        model_turns=model_turns,
                        tool_calls=tool_calls,
                        started=started,
                    )
            return self._write_terminal_failure(
                destination=destination,
                plan=plan,
                category=FailureCategory.AGENT_OUTPUT_CONTRACT_ERROR,
                mandatory_artifacts=required_artifacts,
                model_turns=model_turns,
                tool_calls=tool_calls,
                started=started,
            )

    @staticmethod
    def _unavailable_plan() -> dict[str, Any]:
        unavailable = "not available because the agent plan contract failed"
        return {
            field: [] if field in LIST_PLAN_FIELDS else unavailable
            for field in PLAN_FIELDS
        }

    def _write_success(
        self,
        *,
        destination: Path,
        plan: Mapping[str, Any],
        action: Mapping[str, Any],
        mandatory_artifacts: tuple[str, ...],
        model_turns: int,
        tool_calls: int,
        started: float,
    ) -> GenericHarnessResult:
        expected = {
            "action",
            "cohort",
            "results",
            "diagnostics",
            "report",
            "headline_evidence",
            "artifact_inventory",
        }
        if set(action) != expected:
            raise GenericHarnessError(
                "GENERIC_FINAL_BUNDLE_INVALID",
                "finalize fields do not match the frozen contract",
            )
        if not all(isinstance(action[name], dict) for name in ("cohort", "results", "diagnostics")):
            raise GenericHarnessError(
                "GENERIC_FINAL_BUNDLE_INVALID",
                "cohort, results, and diagnostics must be JSON objects",
            )
        if not isinstance(action["report"], str) or not action["report"].strip():
            raise GenericHarnessError(
                "GENERIC_FINAL_BUNDLE_INVALID",
                "report must be non-empty Markdown",
            )
        if not isinstance(action["headline_evidence"], list):
            raise GenericHarnessError(
                "GENERIC_FINAL_BUNDLE_INVALID",
                "headline_evidence must be a JSON list",
            )
        inventory = action["artifact_inventory"]
        if not isinstance(inventory, dict) or set(inventory) != set(
            mandatory_artifacts
        ):
            raise GenericHarnessError(
                "GENERIC_FINAL_BUNDLE_INVALID",
                "artifact_inventory must map every frozen mandatory artifact",
            )
        snapshot = dict(self._resource_snapshot())
        receipt = ReviewResourceReceipt.from_snapshot(
            snapshot,
            model_turns=model_turns,
            tool_calls=tool_calls,
            wall_seconds=round(max(0.0, time.monotonic() - started), 6),
        )
        material = ReviewBundleMaterial(
            plan=plan,
            cohort=action["cohort"],
            results=action["results"],
            diagnostics=action["diagnostics"],
            report=action["report"],
            headline_evidence=action["headline_evidence"],
            artifact_inventory=inventory,
        )
        try:
            write_review_bundle(
                material,
                output_dir=destination,
                task_id=self._task_id,
                mandatory_artifacts=mandatory_artifacts,
                resource_receipt=receipt,
            )
        except ReviewBundleWriteError as exc:
            raise GenericHarnessError(
                "GENERIC_FINAL_BUNDLE_INVALID",
                str(exc),
            ) from exc
        return GenericHarnessResult(
            terminal_status=TerminalStatus.COMPLETED,
            failure_category=None,
            output_dir=destination,
            model_turns=model_turns,
            tool_calls=tool_calls,
        )

    def _write_terminal_failure(
        self,
        *,
        destination: Path,
        plan: Mapping[str, Any],
        category: FailureCategory,
        mandatory_artifacts: tuple[str, ...],
        model_turns: int,
        tool_calls: int,
        started: float,
    ) -> GenericHarnessResult:
        snapshot = dict(self._resource_snapshot())
        receipt = ReviewResourceReceipt.from_snapshot(
            snapshot,
            model_turns=model_turns,
            tool_calls=tool_calls,
            wall_seconds=round(max(0.0, time.monotonic() - started), 6),
        )
        material = terminal_failure_material(
            plan=plan,
            failure_category=category,
            mandatory_artifacts=mandatory_artifacts,
        )
        try:
            write_review_bundle(
                material,
                output_dir=destination,
                task_id=self._task_id,
                mandatory_artifacts=mandatory_artifacts,
                resource_receipt=receipt,
                outcome=TerminalOutcome.failed(category),
            )
        except ReviewBundleWriteError as exc:
            raise GenericHarnessError(
                "GENERIC_FINAL_BUNDLE_INVALID",
                str(exc),
            ) from exc
        return GenericHarnessResult(
            terminal_status=TerminalStatus.FAILED,
            failure_category=category,
            output_dir=destination,
            model_turns=model_turns,
            tool_calls=tool_calls,
        )


__all__ = [
    "DockerRunnerBackend",
    "GenericCodeAgentHarness",
    "GenericBudgetExhausted",
    "GenericExecutionBackend",
    "GenericExecutionResult",
    "GenericHarnessError",
    "GenericHarnessResult",
    "GenericModelGateway",
    "PlanReviewDecision",
]
