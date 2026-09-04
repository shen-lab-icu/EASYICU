"""Formal Figure 2 entry point for the complete EasyICU arm."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from threading import Lock
from typing import Any, Iterator, Mapping

from easyicu.research_agent.authority.provider_hard_stop import TaskProviderHardStop
from easyicu.research_agent.orchestration.config import PipelineConfig
from easyicu.research_agent.orchestration.services import PipelineServices
from easyicu.research_agent.pipeline import ResearchAgentPipeline
from easyicu.research_agent.schema import PipelineResult

from .easyicu_review_bundle_adapter import (
    EasyICUReviewMaterial,
    write_easyicu_review_bundle,
)
from .review_bundle_writer import terminal_failure_material

from .formal_provider_gate import (
    FormalAuthorizedHardStopClient,
    FormalCallCoordinate,
)
from .formal_scheduler import (
    consume_trajectory_lease,
    signed_output_root,
    signed_site_assignment,
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


class FormalEasyICURunner:
    """Construct one non-resumable EasyICU pipeline under formal authority."""

    def __init__(
        self,
        *,
        config: PipelineConfig,
        services: PipelineServices,
        receipts: Mapping[str, Any],
        scope: str,
        task_id: str,
        execution_site: str,
        trajectory_lease_path: Path,
        provider_hard_stop: TaskProviderHardStop,
    ) -> None:
        if services.provider_hard_stop is not provider_hard_stop:
            raise ValueError(
                "PipelineServices must carry the same formal provider hard stop"
            )
        lease = consume_trajectory_lease(
            trajectory_lease_path,
            scope=scope,
            task_id=task_id,
            arm="easyicu_full",
            execution_site=execution_site,
            site_assignment=signed_site_assignment(receipts, scope=scope),
            expected_output_root=signed_output_root(
                receipts,
                execution_site=execution_site,
            ),
        )
        self._leased_output_dir = Path(lease["output_dir"]).resolve()
        router = FormalEasyICUModelRouter(
            services.llm,
            receipts=receipts,
            scope=scope,
            task_id=task_id,
            execution_site=execution_site,
            provider_hard_stop=provider_hard_stop,
        )
        formal_services = replace(
            services,
            llm=router,
            vlm_client=router.wrap_client(
                services.vlm_client,
                role="vlm_visual_qa",
            ),
            llm_concept_auditor_client=router.wrap_client(
                services.llm_concept_auditor_client,
                role="concept_auditor",
            ),
        )
        self._pipeline = ResearchAgentPipeline(
            config=config,
            services=formal_services,
        )
        self._provider_hard_stop = provider_hard_stop

    def run(self, **kwargs: Any) -> Any:
        if kwargs.get("resume_run_id") is not None or kwargs.get(
            "resume_from_step_id"
        ) is not None:
            raise ValueError("formal Figure 2 runs cannot resume")
        return self._pipeline.run(**kwargs)

    def _write_terminal_failure_bundle(
        self,
        *,
        output_dir: Path,
        mandatory_artifacts: tuple[str, ...],
    ) -> None:
        accounting = self._provider_hard_stop.accounting_summary()
        failed = terminal_failure_material(
            plan={"available": False, "failure_category": "execution_failure"},
            failure_category="execution_failure",
            mandatory_artifacts=mandatory_artifacts,
        )
        write_easyicu_review_bundle(
            failed,
            output_dir=output_dir,
            mandatory_artifacts=mandatory_artifacts,
            resource_receipt={
                "within_frozen_budget": False,
                "provider_accounting": accounting,
            },
            terminal_status="failed",
            failure_category="execution_failure",
        )

    def run_and_write_review_bundle(
        self,
        *,
        output_dir: Path,
        mandatory_artifacts: tuple[str, ...],
        artifact_inventory: Mapping[str, Any],
        **run_kwargs: Any,
    ) -> PipelineResult:
        """Run once and immediately project fixed native outputs for review."""

        if Path(output_dir).resolve() != self._leased_output_dir:
            raise ValueError("formal output directory does not match the consumed lease")
        try:
            result = self.run(**run_kwargs)
            if not isinstance(result, PipelineResult):
                raise ValueError(
                    "formal review projection requires a terminal PipelineResult"
                )
            material = EasyICUReviewMaterial.from_pipeline_result(
                result,
                artifact_inventory=artifact_inventory,
            )
            self._provider_hard_stop.assert_active()
            accounting = self._provider_hard_stop.accounting_summary()
        except Exception:
            self._write_terminal_failure_bundle(
                output_dir=output_dir,
                mandatory_artifacts=mandatory_artifacts,
            )
            raise
        write_easyicu_review_bundle(
            material,
            output_dir=output_dir,
            mandatory_artifacts=mandatory_artifacts,
            resource_receipt={
                "within_frozen_budget": True,
                "provider_accounting": accounting,
            },
        )
        return result


__all__ = ["FormalEasyICUModelRouter", "FormalEasyICURunner"]
