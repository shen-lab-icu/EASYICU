"""Only formal Provider-backed entry point for the Figure 2 generic arm."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from easyicu.research_agent.authority.provider_hard_stop import (
    ProviderHardStopExceeded,
    TaskProviderHardStop,
)
from easyicu.research_agent.execution.runner import DockerRunner
from easyicu.research_agent.providers.protocol import LLMMessage

from .formal_provider_gate import (
    FormalProviderSession,
    complete_formal_provider_call,
)
from .formal_trajectory_lifecycle import FormalExecutionSession
from .generic_code_agent_harness import (
    DockerRunnerBackend,
    GenericCodeAgentHarness,
    GenericBudgetExhausted,
    GenericExecutionBackend,
    GenericHarnessResult,
    PlanReviewDecision,
)
from .review_bundle_semantics import ReviewResourceReceipt


class FormalGenericResourceReceiptError(RuntimeError):
    """Formal budget/cost receipt is incomplete and cannot be normalized away."""

    reason_code = "FORMAL_GENERIC_RESOURCE_RECEIPT_INCOMPLETE"
    owner = "figure2.formal_generic_runner_v1"

    def __init__(self, missing: Sequence[str]) -> None:
        self.missing = tuple(missing)
        self.easyicu_safe_diagnostic = {
            "owner": self.owner,
            "reason_code": self.reason_code,
            "missing_fields": list(self.missing),
        }
        super().__init__(
            f"{self.reason_code}: missing formal resource fields: "
            + ", ".join(self.missing)
        )


@dataclass
class FormalGenericModelGateway:
    """Route every formal generic-arm model turn through the receipt gate."""

    client: Any
    session: FormalProviderSession
    max_tokens: int
    temperature: float

    def __post_init__(self) -> None:
        if not isinstance(self.session, FormalProviderSession):
            raise TypeError("session must be FormalProviderSession")
        if self.session.arm != "generic_code_agent":
            raise ValueError(
                "formal generic gateway requires the generic_code_agent arm"
            )

    def complete(self, *, phase: str, messages: Sequence[LLMMessage]) -> str:
        del phase
        try:
            return complete_formal_provider_call(
                self.client,
                messages,
                session=self.session,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )
        except ProviderHardStopExceeded as exc:
            raise GenericBudgetExhausted from exc


class FormalGenericCodeAgentRunner:
    """Bind the generic harness to the mandatory formal Provider boundary."""

    def __init__(
        self,
        *,
        client: Any,
        receipts: Mapping[str, Any],
        scope: str,
        task_id: str,
        execution_site: str,
        trajectory_lease_path: Path,
        max_tokens: int,
        temperature: float,
        docker_runner_factory: Callable[[Path], DockerRunner],
        provider_hard_stop: TaskProviderHardStop,
        resource_snapshot: Callable[[], Mapping[str, Any]],
    ) -> None:
        session = FormalExecutionSession(
            lease_path=trajectory_lease_path,
            receipts=receipts,
            scope=scope,
            task_id=task_id,
            arm="generic_code_agent",
            execution_site=execution_site,
            provider_hard_stop=provider_hard_stop,
        )
        gateway = FormalGenericModelGateway(
            client=client,
            session=session.provider,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        def validated_resource_snapshot() -> Mapping[str, Any]:
            snapshot = dict(resource_snapshot())
            required = {"within_frozen_budget", "billed_cost"}
            missing = sorted(required.difference(snapshot))
            if missing:
                raise FormalGenericResourceReceiptError(missing)
            if not isinstance(snapshot["within_frozen_budget"], bool):
                raise FormalGenericResourceReceiptError(
                    ("within_frozen_budget",)
                )
            try:
                provider_hard_stop.assert_active()
                hard_stop_active = True
            except ProviderHardStopExceeded:
                hard_stop_active = False
            within_frozen_budget = (
                snapshot["within_frozen_budget"] and hard_stop_active
            )
            accounting = provider_hard_stop.accounting_summary()
            try:
                receipt = ReviewResourceReceipt.from_provider_accounting(
                    accounting,
                    within_frozen_budget=within_frozen_budget,
                    reported_billed_cost_usd=snapshot["billed_cost"],
                )
            except ValueError as exc:
                raise FormalGenericResourceReceiptError(
                    ("resource_receipt",)
                ) from exc
            return receipt.as_dict()

        def build_harness() -> GenericCodeAgentHarness:
            docker_runner = docker_runner_factory(session.workdir)
            session.require_workdir(docker_runner.workdir)
            executor: GenericExecutionBackend = DockerRunnerBackend(
                docker_runner,
                task_hard_stop=provider_hard_stop,
            )
            return GenericCodeAgentHarness(
                task_id=task_id,
                model=gateway,
                executor=executor,
                resource_snapshot=validated_resource_snapshot,
            )

        self._harness = session.initialize(
            factory=build_harness,
        )
        self._trajectory = session
        self._provider_hard_stop = provider_hard_stop

    def run(
        self,
        *,
        task_prompt: str,
        neutral_input_description: str,
        mandatory_artifacts: Sequence[str],
        output_dir: Path,
        review_plan: Callable[[Mapping[str, Any]], PlanReviewDecision],
    ) -> GenericHarnessResult:
        def review_without_charging_human_wait(
            plan: Mapping[str, Any],
        ) -> PlanReviewDecision:
            self._provider_hard_stop.pause()
            try:
                return review_plan(plan)
            finally:
                self._provider_hard_stop.resume()

        required_artifacts = tuple(mandatory_artifacts)
        return self._trajectory.run_to_terminal(
            operation=lambda: self._harness.run(
                task_prompt=task_prompt,
                neutral_input_description=neutral_input_description,
                mandatory_artifacts=required_artifacts,
                output_dir=output_dir,
                review_plan=review_without_charging_human_wait,
            ),
            output_dir=output_dir,
            mandatory_artifacts=required_artifacts,
        )


__all__ = [
    "FormalGenericCodeAgentRunner",
    "FormalGenericModelGateway",
    "FormalGenericResourceReceiptError",
]
