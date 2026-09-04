"""Formal projection for every Provider-bearing EasyICU collaborator."""

from __future__ import annotations

from threading import Lock
from typing import Any, Iterator, Mapping

from easyicu.research_agent.authority.provider_hard_stop import TaskProviderHardStop
from easyicu.research_agent.orchestration.services import PipelineServices

from .formal_provider_gate import (
    FormalAuthorizedHardStopClient,
    FormalCallCoordinate,
)


class _FormalCallSequence:
    def __init__(self, *, scope: str, task_id: str, execution_site: str) -> None:
        self._scope = scope
        self._task_id = task_id
        self._execution_site = execution_site
        self._number = 0
        self._lock = Lock()

    def next(self) -> FormalCallCoordinate:
        with self._lock:
            self._number += 1
            number = self._number
        return FormalCallCoordinate(
            scope=self._scope,
            task_id=self._task_id,
            arm="easyicu_full",
            execution_site=self._execution_site,
            call_id=f"easyicu_{number:04d}",
        )


class FormalEasyICUModelRouter:
    """Wrap every EasyICU role client in the same formal gate and ledger."""

    name = "formal_easyicu_model_router"

    def __init__(
        self,
        inner: Any,
        *,
        receipts: Mapping[str, Any],
        scope: str,
        task_id: str,
        execution_site: str,
        provider_hard_stop: TaskProviderHardStop,
    ) -> None:
        if inner is None:
            raise ValueError("formal EasyICU execution requires an LLM client")
        if not isinstance(provider_hard_stop, TaskProviderHardStop):
            raise TypeError("provider_hard_stop must be TaskProviderHardStop")
        self._inner = inner
        self._receipts = dict(receipts)
        self._hard_stop = provider_hard_stop
        self._sequence = _FormalCallSequence(
            scope=scope,
            task_id=task_id,
            execution_site=execution_site,
        )
        self._role_clients: dict[str, FormalAuthorizedHardStopClient] = {}
        self._iter_clients: list[FormalAuthorizedHardStopClient] | None = None

    def _resolve_inner(self, role: str) -> Any:
        if hasattr(self._inner, "for_role"):
            return self._inner.for_role(role)
        return self._inner

    def wrap_client(self, client: Any, *, role: str) -> Any:
        if client is None:
            return None
        return FormalAuthorizedHardStopClient(
            client,
            role=role,
            task=self._hard_stop,
            receipts=self._receipts,
            coordinate_factory=self._sequence.next,
        )

    def for_role(self, role: str) -> Any:
        normalized_role = str(role).strip()
        if not normalized_role:
            raise ValueError("formal EasyICU role must be non-empty")
        if normalized_role not in self._role_clients:
            wrapped = self.wrap_client(
                self._resolve_inner(normalized_role),
                role=normalized_role,
            )
            if wrapped is None:
                raise ValueError(f"formal EasyICU role has no client: {normalized_role}")
            self._role_clients[normalized_role] = wrapped
        return self._role_clients[normalized_role]

    def iter_clients(self) -> Iterator[Any]:
        if self._iter_clients is None:
            raw_clients = (
                tuple(self._inner.iter_clients())
                if hasattr(self._inner, "iter_clients")
                else (self._inner,)
            )
            self._iter_clients = [
                self.wrap_client(client, role=f"router_client_{index:04d}")
                for index, client in enumerate(raw_clients, start=1)
                if client is not None
            ]
        yield from self._iter_clients


class FormalEasyICUCollaboratorAdapter:
    """Apply formal authority to the complete production-owned Provider surface."""

    _DIRECT_CLIENT_ROLES = {
        "vlm_client": "vlm_visual_qa",
        "llm_concept_auditor_client": "concept_auditor",
    }

    def __init__(
        self,
        services: PipelineServices,
        *,
        receipts: Mapping[str, Any],
        scope: str,
        task_id: str,
        execution_site: str,
        provider_hard_stop: TaskProviderHardStop,
    ) -> None:
        if services.provider_hard_stop is not provider_hard_stop:
            raise ValueError(
                "PipelineServices must carry the same formal provider hard stop"
            )
        self._services = services
        self._router = FormalEasyICUModelRouter(
            services.llm,
            receipts=receipts,
            scope=scope,
            task_id=task_id,
            execution_site=execution_site,
            provider_hard_stop=provider_hard_stop,
        )

    def _project_collaborator(self, name: str, collaborator: Any) -> Any:
        if name == "llm":
            return self._router
        if name == "visual_qa_adapter":
            if collaborator is not None:
                raise ValueError(
                    "formal EasyICU execution forbids an opaque visual QA adapter; "
                    "supply its Provider client through vlm_client"
                )
            return None
        try:
            role = self._DIRECT_CLIENT_ROLES[name]
        except KeyError as exc:
            raise ValueError(
                f"unsupported Provider-bearing PipelineServices field: {name}"
            ) from exc
        return self._router.wrap_client(collaborator, role=role)

    def project(self) -> PipelineServices:
        """Return services with no ungoverned Provider collaborator remaining."""

        return self._services.map_provider_collaborators(
            self._project_collaborator
        )


__all__ = [
    "FormalEasyICUCollaboratorAdapter",
    "FormalEasyICUModelRouter",
]
