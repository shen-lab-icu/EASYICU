"""A recovery seed bound to superseded code/image must fail closed.

Real signature 2026-09-12: job ``31236322202f`` tried to resume an approved
execution whose recovery seed still bound the old code/image coordinates
(``9a1fcfe3d``/``d90121e69b0c``) while the selected checkpoint belonged to the
current ones (``547bd2219``/``25a3fafae121``); the retry was legitimately
invalidated as ``research_pipeline_execution_retry_recovery_seed_superseded``
instead of silently restoring a configuration the checkpoint no longer owns.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from easyicu.research_agent.orchestration.config import PipelineConfig
from easyicu.webserver import agent_pipeline_runs
from easyicu.webserver.agent_review_recovery import WebReviewRecoverySeed


def test_recovery_seed_bound_to_superseded_configuration_fails_closed(
    tmp_path: Path,
) -> None:
    wrapper = tmp_path / "projects" / "study" / "run-wrapper"
    superseded_config = PipelineConfig(workdir=wrapper / "pipeline")
    current_config = PipelineConfig(
        workdir=wrapper / "pipeline",
        enable_pubmed=True,
    )
    prepared_package_binding = {"package_sha256": "d" * 64}
    seed = WebReviewRecoverySeed.create(
        wrapper_dir=str(wrapper.resolve()),
        study={"id": "study"},
        scientific_configuration_sha256="a" * 64,
        provider_meta={"provider": "openai"},
        provider_public={"provider": "openai", "model": "model-a"},
        credential_source="pi_verified",
        budget_mode="full_reviewed",
        prepared_package_binding=prepared_package_binding,
        pipeline_config=superseded_config.recovery_payload(),
        pipeline_config_sha256=superseded_config.canonical_digest(),
        acquisition_projection={},
        hard_stop_ledger_path=str(wrapper / ".runtime" / "ledger.json"),
        hard_stop_task_id="web-job-a",
        hard_stop_declaration_sha256="b" * 64,
        created_at=1.0,
    )
    target = agent_pipeline_runs._ExecutionResumeTarget(
        wrapper_dir=wrapper.resolve(),
        pipeline_run_id="run-analysis",
        pipeline_config_sha256=current_config.canonical_digest(),
    )

    with pytest.raises(agent_pipeline_runs.ResearchPipelineRunError) as raised:
        agent_pipeline_runs._validated_execution_retry_config(
            current_config=current_config,
            target=target,
            recovery_seed=seed,
            current_scientific_digest="a" * 64,
            prepared_package_binding=prepared_package_binding,
        )

    assert raised.value.code == (
        "research_pipeline_execution_retry_recovery_seed_superseded"
    )
