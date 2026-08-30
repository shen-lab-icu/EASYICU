from __future__ import annotations

from pathlib import Path

import pytest

from easyicu.webserver import study_contexts
from easyicu.webserver.pi_copilot import contracts, run_authority
from easyicu.webserver.pi_copilot import tools as tool_module


def test_cancelled_retry_does_not_hide_unchanged_planner_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    study = {
        "id": "study-resume-history",
        "revision": 1,
        "question": "Does an ICU measurement relate to in-hospital death?",
        "database": "miiv",
    }
    digest = study_contexts.scientific_configuration_sha256(study)
    root = tmp_path / "project-root"
    checkpoint_dir = root / study["id"] / "run_valid-prefix"
    checkpoint_dir.mkdir(parents=True)
    rows = [
        {
            "run_id": "run_user-stopped",
            "study_id": study["id"],
            "run_status": "failed",
            "gate_reason": "research_pipeline_cancelled",
            "scientific_configuration_sha256": digest,
            "development_planner_checkpoint_available": False,
        },
        {
            "run_id": "run_valid-prefix",
            "study_id": study["id"],
            "run_status": "failed",
            "gate_reason": "research_pipeline_progressive_compile_failed",
            "scientific_configuration_sha256": digest,
            "project_dir": str(checkpoint_dir),
            "development_planner_checkpoint_available": True,
        },
    ]
    monkeypatch.setattr(tool_module, "_run_rows", lambda _context: rows)
    monkeypatch.setattr(
        tool_module,
        "research_pipeline_project_root",
        lambda _study_id: root,
    )

    assert (
        tool_module._development_resume_source_job_id(object(), study)
        == "valid-prefix"
    )

    assert (
        run_authority.resumable_planner_checkpoint_job_id(
            study=study,
            rows=rows,
            project_root=root,
        )
        == "valid-prefix"
    )


def test_newer_nontransparent_failure_still_blocks_older_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    study = {"id": "study-resume-history", "revision": 1, "question": "Q"}
    digest = study_contexts.scientific_configuration_sha256(study)
    monkeypatch.setattr(
        tool_module,
        "_run_rows",
        lambda _context: [
            {
                "run_id": "run_newer-scientific-failure",
                "study_id": study["id"],
                "run_status": "failed",
                "gate_reason": "research_pipeline_execution_failed",
                "scientific_configuration_sha256": digest,
                "development_planner_checkpoint_available": False,
            },
            {
                "run_id": "run_old-checkpoint",
                "study_id": study["id"],
                "run_status": "failed",
                "gate_reason": "research_pipeline_progressive_compile_failed",
                "scientific_configuration_sha256": digest,
                "development_planner_checkpoint_available": True,
            },
        ],
    )

    assert tool_module._development_resume_source_job_id(object(), study) == ""


def test_checkpoint_seeding_is_wider_than_the_plan_resume_offer() -> None:
    """These two sets differ on purpose; do not collapse them.

    A preserved prefix may seed a *fresh* planning run for any bounded Planner
    failure, because re-deriving already-validated steps only burns provider
    budget. Offering the researcher "resume this plan" is narrower: only a
    budget-exhausted plan is itself still intact. Contract exhaustion and
    compile-gate failure had their planning state rejected, so their governed
    next action stays ``failed_pipeline_requires_fresh_plan`` even though that
    fresh plan may still be seeded from the preserved prefix.
    """

    assert contracts.PLAN_RESUME_OFFER_GATE_REASONS == {
        "research_pipeline_planner_efficiency_budget_exhausted"
    }
    assert (
        contracts.PLAN_RESUME_OFFER_GATE_REASONS
        < contracts.PLANNER_CHECKPOINT_GATE_REASONS
    )
    assert contracts.PLANNER_CHECKPOINT_GATE_REASONS == {
        "research_pipeline_planner_efficiency_budget_exhausted",
        "research_pipeline_plan_contract_exhausted",
        "research_pipeline_progressive_compile_failed",
    }
