"""The recorded profile, rather than current Docker availability, labels a plan."""

from types import SimpleNamespace
import json

import pytest

from easyicu.research_agent.authority.plan_review import PlanReviewAuthority, ReviewExecutionAuthority
from easyicu.research_agent.orchestration.profiles import DESKTOP_FIRST_RESULT, DESKTOP_FIRST_RESULT_LIVE_PUBMED


def _requests(ref):
    authority = PlanReviewAuthority.create(
        plan={"research_question": "Describe synthetic values", "steps": []}, evidence_sha256={},
        execution=ReviewExecutionAuthority(
            pipeline_config_sha256="a" * 64,
            submission_profile_ref=ref,
            capability_activation_sha256="b" * 64,
            run_input_capsule_sha256="c" * 64,
        ),
    )
    return [SimpleNamespace(
        payload={"plan_review_authority": authority.model_dump(mode="json")},
        review_id="synthetic_review", kind="scientific_stop", summary="Review the plan",
        authority_sha256="d" * 64,
    )]


def test_recorded_desktop_profile_is_projected_from_the_run_and_review_authority():
    from easyicu.webserver.research_launch_runtime import recorded_submission_profile_ref

    profile = DESKTOP_FIRST_RESULT
    manifest = {
        "submission_profile_name": profile.name,
        "submission_profile_version": profile.version,
    }
    assert recorded_submission_profile_ref(manifest=manifest) == profile.ref
    assert recorded_submission_profile_ref(requests=_requests(profile.ref)) == profile.ref


@pytest.mark.parametrize("source", ["missing", "unknown", "mismatch", "corrupt"])
def test_unverifiable_profile_is_not_invented(source):
    from easyicu.webserver.research_launch_runtime import recorded_submission_profile_ref

    requests = [] if source == "missing" else _requests(DESKTOP_FIRST_RESULT.ref)
    if source == "unknown":
        requests = _requests("unknown/1.0.0")
    if source == "mismatch":
        requests += _requests(DESKTOP_FIRST_RESULT_LIVE_PUBMED.ref)
    if source == "corrupt":
        requests[0].payload["plan_review_authority"]["plan_sha256"] = "0" * 64
    assert recorded_submission_profile_ref(requests=requests) is None


def test_workflow_carries_the_recorded_profile_to_the_plan_card():
    from easyicu.webserver import study_contexts
    from easyicu.webserver.pi_copilot.workflow import build_research_workflow_snapshot
    from tests.webserver.copilot.research_workflow_fixtures import complete_study

    study = complete_study()
    snapshot = build_research_workflow_snapshot(
        study=study, active_export_present=True, active_job=None,
        latest_run={
            "run_id": "synthetic", "run_type": "full",
            "engine": "easyicu.research_agent.pipeline",
            "run_status": "human_review_pending", "gate_status": "blocked",
            "gate_reason": "human_plan_review_required",
            "pending_review_reason_codes": ["operator_plan_approval_required"],
            "artifact_names": ["agent_plan.json", "source_run_manifest.json"],
        },
        plan_review_authority={
            "run_id": "synthetic", "resumable_here": True,
            "plan_approval_allowed": True, "research_input_state": "prepared",
            "scientific_plan_review": {"status": "ready"},
            "submission_profile_ref": DESKTOP_FIRST_RESULT.ref,
            "scientific_configuration_sha256": study_contexts.scientific_configuration_sha256(study),
        },
    )
    assert snapshot.plan_review_summary["submission_profile_ref"] == DESKTOP_FIRST_RESULT.ref


def test_web_run_record_retains_the_profile_that_produced_the_results(tmp_path):
    from easyicu.webserver import agent_pipeline_runs
    from tests.webserver.copilot.research_workflow_fixtures import _acquisition_receipt, complete_study

    run = tmp_path / "synthetic_run"
    run.mkdir()
    (run / "manifest.json").write_text(json.dumps({
        "submission_profile_name": DESKTOP_FIRST_RESULT.name,
        "submission_profile_version": DESKTOP_FIRST_RESULT.version,
    }))
    wrapper = tmp_path / "wrapper"
    agent_pipeline_runs._write_projection(
        wrapper_dir=wrapper, study=complete_study(), provider={},
        acquisition=_acquisition_receipt(), run_dir=run,
    )
    record = json.loads((wrapper / "source_run_manifest.json").read_text())
    assert record["submission_profile_ref"] == DESKTOP_FIRST_RESULT.ref


def test_pending_review_projects_its_bound_profile(tmp_path, monkeypatch):
    from easyicu.webserver import agent_pipeline_runs
    from tests.webserver.copilot.research_workflow_fixtures import complete_study

    pending = SimpleNamespace(
        run_id="synthetic", run_dir=tmp_path, resume_scope="same_process",
        resumable_here=True, requests=_requests(DESKTOP_FIRST_RESULT.ref),
    )
    entry = SimpleNamespace(
        pending=pending, study=complete_study(), credential_source="pi_verified",
        budget_mode="full_reviewed", provider={}, wrapper_dir=tmp_path,
    )
    monkeypatch.setattr(agent_pipeline_runs, "_PENDING_REVIEWS", {"synthetic": entry})
    assert agent_pipeline_runs.pending_review("synthetic")["submission_profile_ref"] == DESKTOP_FIRST_RESULT.ref
