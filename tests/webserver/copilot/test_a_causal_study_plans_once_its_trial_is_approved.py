"""A causal study states and approves its target trial before it plans.

While a causal study's plan step is ready, the workflow names the trial's
next action in place of the plan's: a statement first, then the card to
review -- while the compile runs, after it stopped, or once it wrote a record.
No plan transition is offered until the researcher approved the record the
study names; then the study plans on its data, where the run verifies the
approved record, never as a metadata-only candidate.  The snapshot the model reads
carries the latest compile job, and the projection carries the card the
browser shows.  A study of another family plans as before.  Synthetic
manifests and records only.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import pytest

from easyicu.webserver import dataio, target_trial_records
from easyicu.webserver.pi_copilot import workflow as workflow_owner
from easyicu.webserver.pi_copilot.extraction_handoff import compile_study_cohort
from tests.support.target_trial import kept_target_trial_record, target_trial_design

STUDY_ID = "study-trial-plan"
_CAUSAL = {
    "analysis_family": "causal_inference",
    "analysis_unit": "icu_stay",
    "variance_estimator": "model_based",
}


@pytest.fixture(autouse=True)
def _isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    monkeypatch.setenv("EASYICU_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(workflow_owner.sources, "load_registry", lambda: {})
    monkeypatch.setattr(workflow_owner, "list_bound_run_history", lambda **_kwargs: [])
    latest: dict[str, Any] = {"value": None}
    monkeypatch.setattr(
        workflow_owner, "latest_target_trial_compile", lambda _study_id: latest["value"]
    )
    return latest


def _study(tmp_path: Path, **fields: Any) -> dict[str, Any]:
    """A question and a current prepared package: a study ready to plan."""

    export = tmp_path / "export"
    study: dict[str, Any] = {
        "id": STUDY_ID,
        "revision": 2,
        "question": "Does starting a vasopressor early change death by day 28?",
        "data_source": {"path": str(export), "database": "miiv"},
        **fields,
    }
    if not export.exists():
        export.mkdir(parents=True)
        contract = compile_study_cohort(study)
        manifest = {
            "schema_version": "easyicu_native_export_v2",
            "database": "miiv",
            "data_path": str(tmp_path / "raw"),
            "format": "parquet",
            "files": [{"file": "demographics.parquet", "module": "demographics"}],
            "cohort_contract": contract,
            "cohort_execution": dataio.export_cohort_execution(contract),
        }
        (export / "_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return study


def _project(study: dict[str, Any]) -> Any:
    return workflow_owner.build_project_workflow_projection(
        study_context_id=study["id"], study_override=study
    )


def _latest(status: str, reason: Optional[str] = None, digest: Optional[str] = None):
    return {
        "job_id": "job_a",
        "status": status,
        "reason_code": reason,
        "compile_sha256": digest,
        "detail": "synthetic",
    }


def test_a_study_of_another_family_plans_as_before(tmp_path: Path) -> None:
    projection = _project(_study(tmp_path))

    assert projection.workflow.next_action_code == "provider_ready_to_generate_plan"
    assert [
        stage.reason_code for stage in projection.workflow.stages if stage.id == "plan"
    ] == ["provider_ready_to_generate_plan"]
    assert projection.host_decisions.plan_transition is not None
    assert projection.workflow.target_trial_compile is None
    assert projection.target_trial_card is None


@pytest.mark.parametrize(
    ("latest", "next_action", "card_state"),
    [
        (None, "target_trial_statement_needed", None),
        (_latest("running"), "target_trial_review", "compiling"),
        (
            _latest("stopped", "target_trial_data_unavailable"),
            "target_trial_review",
            "stopped",
        ),
    ],
    ids=["unstated", "compiling", "data_unavailable"],
)
def test_a_causal_study_states_its_trial_before_a_plan(
    tmp_path: Path,
    _isolated: dict[str, Any],
    latest: Optional[dict],
    next_action: str,
    card_state: Optional[str],
) -> None:
    _isolated["value"] = latest

    projection = _project(_study(tmp_path, analysis_design=dict(_CAUSAL)))

    workflow = projection.workflow
    assert (workflow.current_stage, workflow.next_action_code) == ("plan", next_action)
    # The plan step gives the same reason as the next action.
    (plan,) = [stage for stage in workflow.stages if stage.id == "plan"]
    assert plan.reason_code == next_action
    assert workflow.target_trial_compile == latest
    assert projection.host_decisions.plan_transition is None
    card = projection.target_trial_card
    assert (card["state"] if card is not None else None) == card_state


def test_a_compiled_record_is_reviewed_and_an_approved_one_plans(
    tmp_path: Path, _isolated: dict[str, Any]
) -> None:
    kept = kept_target_trial_record()
    target_trial_records.keep_target_trial_record(STUDY_ID, kept)
    _isolated["value"] = _latest("compiled", digest=kept.compile_sha256)

    stated = _project(
        _study(
            tmp_path,
            analysis_design=dict(_CAUSAL),
            target_trial_design=target_trial_design(
                kept=kept, study_id=STUDY_ID, approved=False
            ),
        )
    )
    assert stated.workflow.next_action_code == "target_trial_review"
    assert stated.target_trial_card["state"] == "approvable"
    assert stated.host_decisions.plan_transition is None

    approved = _project(
        _study(
            tmp_path,
            analysis_design=dict(_CAUSAL),
            target_trial_design=target_trial_design(kept=kept, study_id=STUDY_ID),
        )
    )
    assert approved.workflow.next_action_code == "target_trial_plan_ready"
    assert [
        stage.reason_code for stage in approved.workflow.stages if stage.id == "plan"
    ] == ["target_trial_plan_ready"]
    assert approved.target_trial_card["state"] == "approved"
    offered = approved.host_decisions.plan_transition
    assert offered is not None
    assert offered.next_action_code == "target_trial_plan_ready"

    # A newer statement holds the approved record until it compiles.
    _isolated["value"] = _latest("running")
    restating = _project(
        _study(
            tmp_path,
            analysis_design=dict(_CAUSAL),
            target_trial_design=target_trial_design(kept=kept, study_id=STUDY_ID),
        )
    )
    assert restating.workflow.next_action_code == "target_trial_review"
    assert restating.host_decisions.plan_transition is None


def test_a_running_plan_keeps_its_own_next_action(tmp_path: Path) -> None:
    snapshot = workflow_owner.build_research_workflow_snapshot(
        study=_study(tmp_path, analysis_design=dict(_CAUSAL)),
        active_export_present=True,
        active_job={"kind": "agent-run", "status": "running", "events": []},
        latest_run=None,
        target_trial_compile=None,
    )

    assert snapshot.next_action_code == "research_planning_running"


def test_an_approved_trial_is_planned_on_the_studys_data(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from easyicu.webserver.routes import agent as agent_routes

    def projection(next_action: str) -> Any:
        workflow = type("Workflow", (), {"next_action_code": next_action})()
        return type("Projection", (), {"workflow": workflow})()

    for next_action, candidate_only in (
        ("provider_ready_to_generate_plan", True),
        ("target_trial_plan_ready", False),
    ):
        monkeypatch.setattr(
            agent_routes,
            "build_project_workflow_projection",
            lambda **_kwargs: projection(next_action),
        )
        body = {"planner_start_mode": "fresh", "study_context_id": STUDY_ID}
        # A candidate plan reads metadata only; the trial's plan reads the
        # rows its approved record was compiled on.
        assert agent_routes._candidate_plan_only_authorized(body) is candidate_only


def test_every_plan_of_an_approved_trial_is_planned_on_its_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Whichever adapter asks for a candidate -- a stopped plan's fresh plan in
    the browser, the conversation's replan -- the submission plans the study
    on its data once its trial is approved."""

    from easyicu.webserver import research_run_submission
    from easyicu.webserver import study_contexts as context_store

    monkeypatch.setattr(
        context_store, "_CONFIG_PATH", tmp_path / "cfg" / "study-contexts.json"
    )

    def candidate(intent: str = "candidate_plan") -> bool:
        return research_run_submission.candidate_plan_authorized(
            context_store.get_context(STUDY_ID), intent=intent
        )

    created = context_store.upsert_context(
        {"id": STUDY_ID, "question": "q", "analysis_design": dict(_CAUSAL)}
    )
    kept = kept_target_trial_record()
    target_trial_records.keep_target_trial_record(STUDY_ID, kept)
    stated = context_store.bind_target_trial_design(
        STUDY_ID, kept.design(), expected_revision=created["revision"]
    )
    # Before the approval a candidate is a candidate, as for any study.
    assert candidate() is True

    context_store.record_target_trial_approval(
        STUDY_ID,
        confirmed_compile_sha256=kept.compile_sha256,
        n_lines_confirmed=kept.confirmation_lines,
        expected_revision=stated["revision"],
    )

    # Approved, the study plans on its data: a candidate could not verify the
    # record its run compiles again.
    assert candidate() is False
    assert candidate("reviewed_analysis") is False
    # A study of another family is a candidate when asked, as before.
    context_store.upsert_context({"id": "study-other", "question": "q"})
    assert (
        research_run_submission.candidate_plan_authorized(
            context_store.get_context("study-other"), intent="candidate_plan"
        )
        is True
    )


@pytest.mark.parametrize(
    "case",
    ["unapproved", "other_record", "record_gone", "restating", "restatement_stopped", "failed"],
)
def test_no_trial_but_the_approved_one_it_keeps_is_planned_on_data(
    tmp_path: Path, _isolated: dict[str, Any], case: str
) -> None:
    kept = kept_target_trial_record()
    target_trial_records.keep_target_trial_record(STUDY_ID, kept)
    _isolated["value"] = {
        "unapproved": _latest("compiled", digest=kept.compile_sha256),
        "other_record": _latest("compiled", digest="e" * 64),
        "record_gone": _latest("compiled", digest=kept.compile_sha256),
        "restating": _latest("running"),
        "restatement_stopped": _latest("stopped", "target_trial_data_unavailable"),
        "failed": _latest("failed", "target_trial_compile_failed"),
    }[case]
    if case == "record_gone":
        for path in target_trial_records.records_root().rglob("records/*.json"):
            path.unlink()

    projection = _project(
        _study(
            tmp_path,
            analysis_design=dict(_CAUSAL),
            target_trial_design=target_trial_design(
                kept=kept, study_id=STUDY_ID, approved=case != "unapproved"
            ),
        )
    )

    assert projection.workflow.next_action_code == "target_trial_review"
    assert projection.host_decisions.plan_transition is None
