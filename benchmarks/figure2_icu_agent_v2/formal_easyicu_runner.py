"""Formal Figure 2 entry point for the complete EasyICU arm."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

from easyicu.research_agent.authority.provider_hard_stop import TaskProviderHardStop
from easyicu.research_agent.orchestration.config import PipelineConfig
from easyicu.research_agent.orchestration.services import PipelineServices
from easyicu.research_agent.pipeline import ResearchAgentPipeline
from easyicu.research_agent.schema import PipelineResult

from .easyicu_review_bundle_adapter import (
    EasyICUReviewMaterial,
    write_easyicu_review_bundle,
)
from .formal_collaborator_adapter import (
    FormalEasyICUCollaboratorAdapter,
    FormalEasyICUModelRouter,
)
from .review_bundle_writer import terminal_failure_material
from .formal_trajectory_lifecycle import FormalExecutionSession


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
        session = FormalExecutionSession(
            lease_path=trajectory_lease_path,
            receipts=receipts,
            scope=scope,
            task_id=task_id,
            arm="easyicu_full",
            execution_site=execution_site,
            provider_hard_stop=provider_hard_stop,
        )
        formal_services = FormalEasyICUCollaboratorAdapter(
            services,
            session=session.provider,
        ).project()
        formal_config = replace(
            config,
            workdir=session.workdir,
            cache_dir=session.workdir / ".cache",
        )
        self._pipeline = session.initialize(
            factory=lambda: ResearchAgentPipeline(
                config=formal_config,
                services=formal_services,
            ),
        )
        self._trajectory = session
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

        self._trajectory.require_output_dir(output_dir)
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
