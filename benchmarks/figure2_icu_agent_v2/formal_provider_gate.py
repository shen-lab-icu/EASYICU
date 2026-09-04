"""Formal Figure 2 Provider boundary.

The experiment-specific receipt gate runs before the production provider
trust check and before any prompt can reach a transport.  The current design
candidate authorizer is executable but deliberately fail-closed because no
trusted external signer key is registered.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
from threading import Lock
from typing import Any, Mapping, Sequence

from easyicu.research_agent.providers.client_trust import (
    authorized_complete,
    require_provider_client_authorization,
)
from easyicu.research_agent.providers.hard_stop import HardStopClient
from easyicu.research_agent.providers.protocol import LLMMessage
from easyicu.research_agent.authority.provider_hard_stop import TaskProviderHardStop

from .formal_authority import authorize_formal_provider_call


_FORMAL_SCOPES = frozenset(
    {"qualification12", "core_wp2_wp3", "wp5_phase_a", "wp5_phase_b_showcase"}
)
_FORMAL_ARMS = frozenset({"easyicu_full", "generic_code_agent"})
_FORMAL_SITES = frozenset({"server", "laptop"})


@dataclass(frozen=True)
class FormalCallCoordinate:
    scope: str
    task_id: str
    arm: str
    execution_site: str
    call_id: str

    def __post_init__(self) -> None:
        if self.scope not in _FORMAL_SCOPES:
            raise ValueError(f"unsupported formal scope: {self.scope}")
        if self.arm not in _FORMAL_ARMS:
            raise ValueError(f"unsupported formal arm: {self.arm}")
        if self.execution_site not in _FORMAL_SITES:
            raise ValueError(f"unsupported execution site: {self.execution_site}")
        for field_name in ("task_id", "call_id"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be a non-empty string")


class FormalProviderBudgetMissingError(RuntimeError):
    reason_code = "FORMAL_PROVIDER_BUDGET_MISSING"


def _authorize_coordinate(
    *,
    receipts: Mapping[str, Any],
    coordinate: FormalCallCoordinate,
    consume: bool = True,
) -> None:
    authorize_formal_provider_call(
        {
            "receipts": deepcopy(dict(receipts)),
            "call_coordinate": asdict(coordinate),
        },
        consume=consume,
    )


class FormalProviderSession:
    """Own one arm's governed call sequence, receipts, and durable budget."""

    def __init__(
        self,
        *,
        receipts: Mapping[str, Any],
        scope: str,
        task_id: str,
        arm: str,
        execution_site: str,
        provider_hard_stop: TaskProviderHardStop,
    ) -> None:
        if not isinstance(provider_hard_stop, TaskProviderHardStop):
            raise FormalProviderBudgetMissingError(
                "Formal Provider transport requires the shared durable hard-stop budget"
            )
        # Validate the static coordinate shape before any collaborator is wrapped.
        FormalCallCoordinate(
            scope=scope,
            task_id=task_id,
            arm=arm,
            execution_site=execution_site,
            call_id="preflight_0000",
        )
        self._receipts = deepcopy(dict(receipts))
        self._scope = scope
        self._task_id = task_id
        self._arm = arm
        self._execution_site = execution_site
        self._provider_hard_stop = provider_hard_stop
        self._call_number = 0
        self._lock = Lock()

    @property
    def arm(self) -> str:
        return self._arm

    @property
    def provider_hard_stop(self) -> TaskProviderHardStop:
        return self._provider_hard_stop

    def authorize_next_call(self) -> FormalCallCoordinate:
        """Authorize and durably consume the next exact signed coordinate."""

        with self._lock:
            number = self._call_number + 1
            prefix = "easyicu" if self._arm == "easyicu_full" else "generic"
            coordinate = FormalCallCoordinate(
                scope=self._scope,
                task_id=self._task_id,
                arm=self._arm,
                execution_site=self._execution_site,
                call_id=f"{prefix}_{number:04d}",
            )
            _authorize_coordinate(receipts=self._receipts, coordinate=coordinate)
            self._call_number = number
            return coordinate

    def validate_next_call(self) -> FormalCallCoordinate:
        """Validate the next coordinate without consuming it."""

        with self._lock:
            number = self._call_number + 1
            prefix = "easyicu" if self._arm == "easyicu_full" else "generic"
            coordinate = FormalCallCoordinate(
                scope=self._scope,
                task_id=self._task_id,
                arm=self._arm,
                execution_site=self._execution_site,
                call_id=f"{prefix}_{number:04d}",
            )
            _authorize_coordinate(
                receipts=self._receipts,
                coordinate=coordinate,
                consume=False,
            )
            return coordinate


class FormalAuthorizedHardStopClient(HardStopClient):
    """Pipeline-compatible client enforcing authority before every call."""

    name = "formal_authorized_provider_hard_stop"

    def __init__(
        self,
        inner: Any,
        *,
        role: str,
        session: FormalProviderSession,
    ) -> None:
        if not isinstance(session, FormalProviderSession):
            raise TypeError("session must be FormalProviderSession")
        super().__init__(inner, role=role, task=session.provider_hard_stop)
        self._session = session

    def _authorize_next_call(self) -> None:
        self._session.validate_next_call()
        require_provider_client_authorization(self._inner)
        self._session.authorize_next_call()

    def complete_with_usage(
        self,
        messages: Sequence[Any],
        *,
        max_tokens: int = 2048,
        temperature: float = 0.2,
        structured_output: Any = None,
    ) -> tuple[str, Mapping[str, Any] | None]:
        self._authorize_next_call()
        return super().complete_with_usage(
            messages,
            max_tokens=max_tokens,
            temperature=temperature,
            structured_output=structured_output,
        )

    def complete_with_images(
        self,
        *,
        prompt: str,
        image_paths: Sequence[Any],
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> str:
        self._authorize_next_call()
        return super().complete_with_images(
            prompt=prompt,
            image_paths=image_paths,
            max_tokens=max_tokens,
            temperature=temperature,
        )


def complete_formal_provider_call(
    client: Any,
    messages: Sequence[LLMMessage],
    *,
    session: FormalProviderSession,
    **kwargs: Any,
) -> str:
    """Authorize one exact experiment call before trusted transport dispatch."""

    if not isinstance(session, FormalProviderSession):
        raise TypeError("session must be FormalProviderSession")
    session.validate_next_call()
    require_provider_client_authorization(client)
    coordinate = session.authorize_next_call()
    budgeted_client = HardStopClient(
        client,
        role=coordinate.arm,
        task=session.provider_hard_stop,
    )
    return authorized_complete(budgeted_client, messages, **kwargs)


__all__ = [
    "FormalCallCoordinate",
    "FormalAuthorizedHardStopClient",
    "FormalProviderBudgetMissingError",
    "FormalProviderSession",
    "complete_formal_provider_call",
]
