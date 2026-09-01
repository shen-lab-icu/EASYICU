"""Formal Figure 2 Provider boundary.

The experiment-specific receipt gate runs before the production provider
trust check and before any prompt can reach a transport.  The current design
candidate authorizer is executable but deliberately fail-closed because no
trusted external signer key is registered.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

from easyicu.research_agent.providers.client_trust import authorized_complete
from easyicu.research_agent.providers.hard_stop import HardStopClient
from easyicu.research_agent.providers.protocol import LLMMessage
from easyicu.research_agent.authority.provider_hard_stop import TaskProviderHardStop

from .formal_authority import authorize_formal_provider_call


_FORMAL_SCOPES = frozenset(
    {"qualification12", "core_wp2_wp3", "wp5_phase_a", "wp5_phase_b_showcase"}
)
_FORMAL_ARMS = frozenset({"easyicu_full", "generic_code_agent"})


@dataclass(frozen=True)
class FormalCallCoordinate:
    scope: str
    task_id: str
    arm: str
    call_id: str

    def __post_init__(self) -> None:
        if self.scope not in _FORMAL_SCOPES:
            raise ValueError(f"unsupported formal scope: {self.scope}")
        if self.arm not in _FORMAL_ARMS:
            raise ValueError(f"unsupported formal arm: {self.arm}")
        for field_name in ("task_id", "call_id"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be a non-empty string")


class FormalProviderBudgetMissingError(RuntimeError):
    reason_code = "FORMAL_PROVIDER_BUDGET_MISSING"


def complete_formal_provider_call(
    client: Any,
    messages: Sequence[LLMMessage],
    *,
    receipts: Mapping[str, Any],
    coordinate: FormalCallCoordinate,
    provider_hard_stop: TaskProviderHardStop | None = None,
    **kwargs: Any,
) -> str:
    """Authorize one exact experiment call before trusted transport dispatch."""

    authority_payload = {
        "receipts": dict(receipts),
        "call_coordinate": asdict(coordinate),
    }
    authorize_formal_provider_call(authority_payload)
    if provider_hard_stop is None:
        raise FormalProviderBudgetMissingError(
            "Formal Provider transport requires the shared durable hard-stop budget"
        )
    budgeted_client = HardStopClient(
        client,
        role=coordinate.arm,
        task=provider_hard_stop,
    )
    return authorized_complete(budgeted_client, messages, **kwargs)


__all__ = [
    "FormalCallCoordinate",
    "FormalProviderBudgetMissingError",
    "complete_formal_provider_call",
]
