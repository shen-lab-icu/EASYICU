from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from easyicu.webserver import agent_pipeline_runs


def test_completed_approved_run_can_retry_post_execution_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "projects"
    wrapper = root / "study" / "run-wrapper"
    run_dir = wrapper / "pipeline" / "run-analysis"
    run_dir.mkdir(parents=True)
    (run_dir / "human_review_checkpoint.json").write_text("{}", encoding="utf-8")
    (run_dir / "run_status.json").write_text(
        json.dumps(
            {
                "gates": {
                    "execution_complete": True,
                    "evidence_complete": True,
                    "numeric_verified": True,
                    "analysis_validated": False,
                    "failed_steps": [],
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        agent_pipeline_runs.agent_runs,
        "list_run_history",
        lambda **_kwargs: {
            "runs": [
                {
                    "run_id": "run-analysis",
                    "scientific_configuration_sha256": "a" * 64,
                    "gate_reason": "research_agent_pipeline_failed_closed",
                    "run_status": "blocked",
                    "project_dir": str(wrapper),
                }
            ]
        },
    )
    monkeypatch.setattr(
        agent_pipeline_runs.study_context_owner,
        "scientific_configuration_sha256",
        lambda _study: "a" * 64,
    )
    from easyicu.research_agent.orchestration import human_review_checkpoint

    monkeypatch.setattr(
        human_review_checkpoint,
        "load_checkpoint",
        lambda *_args, **_kwargs: SimpleNamespace(
            state="completed",
            approved_decisions=[{"decision": "approved"}],
        ),
    )

    target = agent_pipeline_runs._resolve_execution_resume_wrapper(
        study={"id": "study"},
        project_root=str(root),
        source_run_id="run-analysis",
    )

    assert target.wrapper_dir == wrapper.resolve()
    assert target.pipeline_run_id == "run-analysis"
