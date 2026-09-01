"""Only formal Provider-backed entry point for the Figure 2 generic arm."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from easyicu.research_agent.providers.protocol import LLMMessage

from .formal_provider_gate import (
    FormalCallCoordinate,
    complete_formal_provider_call,
)
from .generic_code_agent_harness import (
    GenericCodeAgentHarness,
    GenericExecutionBackend,
    GenericHarnessResult,
    PlanReviewDecision,
)


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
    receipts: Mapping[str, Any]
    scope: str
    task_id: str
    max_tokens: int
    temperature: float
    _call_number: int = 0

    def complete(self, *, phase: str, messages: Sequence[LLMMessage]) -> str:
        del phase
        self._call_number += 1
        coordinate = FormalCallCoordinate(
            scope=self.scope,
            task_id=self.task_id,
            arm="generic_code_agent",
            call_id=f"generic_{self._call_number:04d}",
        )
        return complete_formal_provider_call(
            self.client,
            messages,
            receipts=self.receipts,
            coordinate=coordinate,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )


class FormalGenericCodeAgentRunner:
    """Bind the generic harness to the mandatory formal Provider boundary."""

    def __init__(
        self,
        *,
        client: Any,
        receipts: Mapping[str, Any],
        scope: str,
        task_id: str,
        max_tokens: int,
        temperature: float,
        executor: GenericExecutionBackend,
        resource_snapshot: Callable[[], Mapping[str, Any]],
    ) -> None:
        gateway = FormalGenericModelGateway(
            client=client,
            receipts=receipts,
            scope=scope,
            task_id=task_id,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        def validated_resource_snapshot() -> Mapping[str, Any]:
            snapshot = dict(resource_snapshot())
            required = {
                "within_frozen_budget",
                "provider_tokens",
                "provider_calls",
                "billed_cost",
            }
            missing = sorted(required.difference(snapshot))
            if missing:
                raise FormalGenericResourceReceiptError(missing)
            return snapshot

        self._harness = GenericCodeAgentHarness(
            model=gateway,
            executor=executor,
            resource_snapshot=validated_resource_snapshot,
        )

    def run(
        self,
        *,
        task_prompt: str,
        neutral_input_description: str,
        output_dir: Path,
        review_plan: Callable[[Mapping[str, Any]], PlanReviewDecision],
    ) -> GenericHarnessResult:
        return self._harness.run(
            task_prompt=task_prompt,
            neutral_input_description=neutral_input_description,
            output_dir=output_dir,
            review_plan=review_plan,
        )


__all__ = [
    "FormalGenericCodeAgentRunner",
    "FormalGenericModelGateway",
    "FormalGenericResourceReceiptError",
]
