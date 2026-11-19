"""A synthetic plan run paused at its review, registered as a live review.

Shared by tests that approve, reject or resume one paused plan. Synthetic
study, plan and export; no benchmark item.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from easyicu.research_agent.authority.plan_review import PlanReviewAuthority
from easyicu.research_agent.orchestration.workflow import (
    HumanReviewPending,
    HumanReviewRequest,
)
from easyicu.research_agent.schema import AnalysisPlan, AnalysisStep
from easyicu.webserver import agent_pipeline_runs, dataio
from easyicu.webserver import study_contexts as study_context_owner
from tests.webserver.copilot.research_workflow_fixtures import (
    _acquisition_receipt,
    complete_study,
)

_RUN_ID = "run-approved-then-failed"
_WRAPPER = "run_web_approved_then_failed"


class _FailingPipeline:
    """A pipeline whose approved resume raises; ``resumable`` is its state after."""

    def __init__(self, error: Exception, *, resumable: bool) -> None:
        self._error = error
        self.has_resumable_human_review = resumable

    def resume_human_review(self, decisions, *, run_id, progress_callback=None):
        raise self._error


def _write_export(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"stay_id": [1], "age": [65]}).to_parquet(
        root / "demographics.parquet", index=False
    )
    (root / "_manifest.json").write_text(
        json.dumps(
            {
                "database": "miiv",
                "format": "parquet",
                "concept_selection": {
                    "mode": "explicit",
                    "modules": {"demographics": ["age"]},
                },
                "feature_definitions": {"included": False},
                "files": [
                    {
                        "file": "demographics.parquet",
                        "module": "demographics",
                        "concepts": 1,
                        "concept_ids": ["age"],
                        "rows": 1,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return root


def _write_paused_run(run_dir: Path) -> None:
    """A pipeline run paused at plan review: a plan, no analysis yet."""

    run_dir.mkdir(parents=True)
    (run_dir / "run_status.json").write_text("{}", encoding="utf-8")
    plan_bytes = json.dumps(
        {"steps": [{"id": "model", "title": "Fit the specified model"}]}
    ).encode("utf-8")
    (run_dir / "analysis_plan.json").write_bytes(plan_bytes)
    (run_dir / "manifest.json").write_text(
        json.dumps(
            {
                "readiness": {
                    "execution_complete": False,
                    "analysis_validated": False,
                    "evidence_complete": False,
                    "numeric_verified": False,
                    "manuscript_ready": False,
                },
                "current_plan_authority": {
                    "relative_path": "analysis_plan.json",
                    "sha256": hashlib.sha256(plan_bytes).hexdigest(),
                },
            }
        ),
        encoding="utf-8",
    )


def _paused(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    pipeline: _FailingPipeline,
) -> tuple[Path, Path, dict[str, Any]]:
    """Pause one approvable plan and register its live review."""

    monkeypatch.setattr(
        agent_pipeline_runs,
        "_load_pending_scientific_review",
        lambda *_args, **_kwargs: {
            "schema_version": agent_pipeline_runs.CURRENT_SCIENTIFIC_REVIEW_SCHEMA_VERSION,
            "approval_allowed": True,
        },
    )
    export = _write_export(tmp_path / "export")
    study = {
        **complete_study(),
        "data_source": {"path": str(export), "database": "miiv"},
    }
    binding = dict(
        dataio.validate_research_pipeline_source(str(export), database="miiv")[
            "binding"
        ]
    )
    run_dir = tmp_path / "pipeline" / _RUN_ID
    _write_paused_run(run_dir)
    authority = PlanReviewAuthority.create(
        plan=AnalysisPlan(
            research_question="Is an ICU exposure associated with mortality?",
            steps=[
                AnalysisStep(
                    step_id="model",
                    intent="Estimate the association",
                    method="descriptive",
                    inputs=[],
                    expected_outputs=["table:model"],
                )
            ],
        )
    )
    request = HumanReviewRequest.create(
        kind="scientific_stop",
        summary="Review the digest-bound plan before analysis.",
        authority_sha256="a" * 64,
        payload={
            "reason": "operator_plan_approval_required",
            "plan_review_authority": authority.model_dump(mode="json"),
        },
    )
    pending = HumanReviewPending(
        run_id=_RUN_ID,
        thread_id="thread-approved-then-failed",
        run_dir=str(run_dir),
        requests=(request,),
    )
    project_root = tmp_path / "projects"
    wrapper = project_root / str(study["id"]) / _WRAPPER
    agent_pipeline_runs._write_projection(
        wrapper_dir=wrapper,
        study=study,
        provider={"provider": "openai", "model": "test-model"},
        acquisition=_acquisition_receipt(),
        run_dir=run_dir,
        pending=pending,
    )
    hard_stop = agent_pipeline_runs._start_web_provider_hard_stop(
        wrapper_dir=wrapper,
        job_id="approved-then-failed",
        declaration_sha256=study_context_owner.scientific_configuration_sha256(study),
    )
    hard_stop.pause()
    entry = agent_pipeline_runs._PendingRun(
        pipeline=pipeline,
        pending=pending,
        wrapper_dir=wrapper,
        study=study,
        provider={},
        acquisition=_acquisition_receipt(),
        created_at=1.0,
        prepared_package_binding=binding,
        provider_hard_stop=hard_stop,
    )
    registry = agent_pipeline_runs.PendingReviewRegistry()
    monkeypatch.setattr(agent_pipeline_runs, "_PENDING_REVIEWS", registry)
    registry.register(entry)
    return project_root, wrapper, study
