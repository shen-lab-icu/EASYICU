from __future__ import annotations

from pathlib import Path

from easyicu.webserver import study_contexts
from easyicu.webserver.pi_copilot import contracts, run_authority


def test_cancelled_retry_does_not_hide_unchanged_planner_checkpoint(
    tmp_path: Path,
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
    assert (
        run_authority.resumable_planner_checkpoint_job_id(
            study=study,
            rows=rows,
            project_root=root,
        )
        == "valid-prefix"
    )


def test_newer_nontransparent_failure_still_blocks_older_checkpoint(
    tmp_path: Path,
) -> None:
    study = {"id": "study-resume-history", "revision": 1, "question": "Q"}
    digest = study_contexts.scientific_configuration_sha256(study)
    rows = [
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
        ]

    assert run_authority.resumable_planner_checkpoint_job_id(
        study=study,
        rows=rows,
        project_root=tmp_path,
    ) == ""


def test_failed_preparation_keeps_unchanged_candidate_plan_authoritative() -> None:
    digest = "a" * 64
    candidate = {
        "run_id": "run_candidate",
        "run_type": "full",
        "run_status": "human_review_pending",
        "gate_reason": "human_plan_review_required",
        "scientific_configuration_sha256": digest,
        "pending_review_reason_codes": ["operator_plan_approval_required"],
        "artifact_names": ["agent_plan.json", "source_run_manifest.json"],
    }
    failed_preparation = {
        "run_id": "run_failed-preparation",
        "run_type": "full",
        "run_status": "failed",
        "gate_reason": "research_pipeline_data_foundation_failed",
        "scientific_configuration_sha256": digest,
        "artifact_names": ["source_run_manifest.json", "evidence_ledger.json"],
    }

    assert (
        run_authority.workflow_authoritative_run([failed_preparation, candidate])
        is candidate
    )


def test_cancelled_duplicate_does_not_hide_scientific_revision_candidate() -> None:
    digest = "a" * 64
    candidate = {
        "run_id": "run_scientific-candidate",
        "run_type": "full",
        "run_status": "human_review_pending",
        "gate_reason": "human_plan_review_required",
        "scientific_configuration_sha256": digest,
        "pending_review_reason_codes": ["plan_scientific_changes_required"],
        "artifact_names": ["agent_plan.json", "source_run_manifest.json"],
    }
    cancelled_duplicate = {
        "run_id": "run_cancelled-duplicate",
        "run_type": "full",
        "run_status": "failed",
        "gate_reason": "research_pipeline_cancelled",
        "scientific_configuration_sha256": digest,
        "artifact_names": ["source_run_manifest.json", "evidence_ledger.json"],
    }

    assert (
        run_authority.workflow_authoritative_run([cancelled_duplicate, candidate])
        is candidate
    )


def test_failed_preparation_does_not_restore_stale_candidate_plan() -> None:
    failed_preparation = {
        "run_id": "run_failed-preparation",
        "run_type": "full",
        "run_status": "failed",
        "scientific_configuration_sha256": "b" * 64,
        "artifact_names": ["source_run_manifest.json"],
    }
    stale_candidate = {
        "run_id": "run_stale-candidate",
        "run_type": "full",
        "run_status": "human_review_pending",
        "gate_reason": "human_plan_review_required",
        "scientific_configuration_sha256": "a" * 64,
        "pending_review_reason_codes": ["operator_plan_approval_required"],
        "artifact_names": ["agent_plan.json"],
    }

    assert (
        run_authority.workflow_authoritative_run(
            [failed_preparation, stale_candidate]
        )
        is failed_preparation
    )


def test_consecutive_failed_preparations_keep_candidate_plan_authoritative() -> None:
    digest = "a" * 64
    failed_attempts = [
        {
            "run_id": f"run_failed-{index}",
            "run_type": "full",
            "run_status": "failed",
            "scientific_configuration_sha256": digest,
            "artifact_names": ["source_run_manifest.json"],
        }
        for index in range(2)
    ]
    candidate = {
        "run_id": "run_candidate",
        "run_type": "full",
        "run_status": "human_review_pending",
        "gate_reason": "human_plan_review_required",
        "scientific_configuration_sha256": digest,
        "pending_review_reason_codes": ["operator_plan_approval_required"],
        "artifact_names": ["agent_plan.json"],
    }

    assert (
        run_authority.workflow_authoritative_run([*failed_attempts, candidate])
        is candidate
    )


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
        "research_pipeline_planner_efficiency_budget_exhausted",
        "research_pipeline_planner_provider_unavailable",
    }
    assert (
        contracts.PLAN_RESUME_OFFER_GATE_REASONS
        < contracts.PLANNER_CHECKPOINT_GATE_REASONS
    )
    assert contracts.PLANNER_CHECKPOINT_GATE_REASONS == {
        "research_pipeline_planner_efficiency_budget_exhausted",
        "research_pipeline_plan_contract_exhausted",
        "research_pipeline_progressive_compile_failed",
        "research_pipeline_planner_provider_unavailable",
    }


def test_legacy_planner_provider_http_projection_recovers_resume_reason(
    tmp_path: Path,
) -> None:
    project_dir = tmp_path / "run-provider-http"
    project_dir.mkdir()
    (project_dir / "source_run_manifest.json").write_text(
        """{
  "analysis_started": false,
  "failure_code": "research_pipeline_execution_failed",
  "failure_type": "provider_http",
  "schema_version": "easyicu.web-research-pipeline-projection/1",
  "status": "failed"
}""",
        encoding="utf-8",
    )

    assert run_authority._normalized_planner_gate_reason(
        {
            "gate_reason": "research_pipeline_execution_failed",
            "project_dir": str(project_dir),
        }
    ) == "research_pipeline_planner_provider_unavailable"


def test_legacy_execution_failure_is_not_upgraded_after_analysis_started(
    tmp_path: Path,
) -> None:
    project_dir = tmp_path / "run-provider-http"
    project_dir.mkdir()
    (project_dir / "source_run_manifest.json").write_text(
        """{
  "analysis_started": true,
  "failure_code": "research_pipeline_execution_failed",
  "failure_type": "provider_http",
  "schema_version": "easyicu.web-research-pipeline-projection/1",
  "status": "failed"
}""",
        encoding="utf-8",
    )

    assert run_authority._normalized_planner_gate_reason(
        {
            "gate_reason": "research_pipeline_execution_failed",
            "project_dir": str(project_dir),
        }
    ) == "research_pipeline_execution_failed"
