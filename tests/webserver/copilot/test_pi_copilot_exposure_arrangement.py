"""A current, verified decision is proposed and the click uses the revision owner."""

from types import SimpleNamespace

import pytest

from easyicu.webserver.pi_copilot.plan_revision_intent import requested_candidate_run
from easyicu.webserver.pi_copilot.question_requirement_notes import (
    exposure_arrangement_offer,
)


pytest_plugins = ["tests.webserver.copilot.pi_copilot_contract_fixtures"]


@pytest.fixture
def current_review() -> dict:
    return {
        "run_id": "synthetic_candidate",
        "question_requirements": {
            "judged_on_plan_under_review": True,
            "items": [
                {
                    "disposition": "not_covered",
                    "source": "host_verified",
                    "quote": "Study the two monitored signals with the first as primary.",
                    "exposure_choice": {
                        "primary_exposure": "first_signal",
                        "additional_exposures": ["second_signal"],
                    },
                }
            ],
        },
    }


@pytest.mark.parametrize("language", ["en", "zh"])
def test_default_and_alternatives_require_a_click_to_revise(current_review, language):
    code = "question_requirement_not_covered"
    offer = exposure_arrangement_offer(code, current_review)
    assert offer["default_option"] == "primary_and_secondary"
    assert set(offer["options"]) == {
        "analyse_both",
        "primary_and_secondary",
        "only_primary",
    }
    projection = SimpleNamespace(
        workflow=SimpleNamespace(
            next_action_code=code,
            plan_review_summary={
                **current_review,
                "exposure_arrangement": offer,
            },
        ),
        latest_run={"run_id": "synthetic_candidate"},
        active_job={"present": False},
        runs=[
            {
                "run_id": "synthetic_candidate",
                "authoritative": True,
                "run_status": "human_review_pending",
                "artifact_names": ["agent_plan.json"],
            }
        ],
    )
    assert (
        requested_candidate_run("", message_origin="host_prompt", projection=projection)
        == ""
    )
    for option in offer["options"].values():
        text = option[language]
        assert "first_signal" in text and "second_signal" in text
        assert (
            requested_candidate_run(
                text, message_origin="model_option", projection=projection
            )
            == ""
        )
        assert (
            requested_candidate_run(
                text, message_origin="host_prompt", projection=projection
            )
            == "synthetic_candidate"
        )
    projection.latest_run = {"run_id": "newer_candidate"}
    assert (
        requested_candidate_run(
            text, message_origin="host_prompt", projection=projection
        )
        == ""
    )


def test_no_choice_from_an_approval_gate_or_obsolete_judgment(current_review):
    assert (
        exposure_arrangement_offer("operator_plan_approval_required", current_review)
        is None
    )
    current_review["question_requirements"]["judged_on_plan_under_review"] = False
    assert (
        exposure_arrangement_offer("question_requirement_not_covered", current_review)
        is None
    )


def test_selected_arrangement_reaches_revision_turn(
    tmp_path, study_state, monkeypatch, current_review
):
    import time

    from easyicu.webserver.pi_copilot import service as service_module
    from easyicu.webserver.pi_copilot.service import PiCopilotError, PiCopilotService
    from tests.webserver.copilot.pi_copilot_contract_fixtures import FakeGateway

    class RevisionGateway(FakeGateway):
        def request(self, method, params, **kwargs):
            result = super().request(method, params, **kwargs)
            if method == "session.prompt":
                kwargs["event_sink"](
                    {
                        "type": "tool_end",
                        "tool_name": "easyicu_request_replan",
                        "code": "easyicu_full_run_submitted",
                        "job_id": "synthetic_plan_job",
                        "status": "ok",
                    }
                )
            return result

    gateway = RevisionGateway()
    service = PiCopilotService(store_path=tmp_path / "sessions.json", gateway=gateway)
    session = service.create_session(
        project_id="synthetic-project", external_llm_opt_in=True
    )["session"]["session_id"]
    code = "question_requirement_not_covered"
    offer = exposure_arrangement_offer(code, current_review)
    review = SimpleNamespace(
        next_action_code=code,
        plan_review_summary={
            **current_review,
            "exposure_arrangement": offer,
        },
    )
    run = current_review["run_id"]
    projected = SimpleNamespace(
        workflow=review,
        latest_run={"run_id": run},
        active_job={"present": False},
        runs=[
            {
                "run_id": run,
                "authoritative": True,
                "run_status": "human_review_pending",
                "artifact_names": ["agent_plan.json"],
            }
        ],
    )
    monkeypatch.setattr(
        service_module, "build_project_workflow_projection", lambda **_: projected
    )
    monkeypatch.setattr(service, "_watch_child_job_for_replay", lambda **_: None)

    def click(message):
        return service.send_message(
            session,
            project_id="synthetic-project",
            message=message,
            message_origin="host_prompt",
            message_intent="confirm_exposure_arrangement",
        )

    with pytest.raises(PiCopilotError, match="exposure arrangement changed"):
        click("Please revise the research plan for review. A different choice.")
    assert not any(method == "session.prompt" for method, *_ in gateway.calls)
    selection = offer["options"][offer["default_option"]]["zh"]
    receipt = click(selection)
    job = service_module.jobs.MANAGER.get(receipt["job_id"])
    deadline = time.monotonic() + 5
    while job.status == "running" and time.monotonic() < deadline:
        time.sleep(0.01)
    assert job.status == "done", job.error
    ((_, delivery, context),) = [
        call for call in gateway.calls if call[0] == "session.prompt"
    ]
    assert delivery["intent"] == "request_candidate_plan_revision"
    assert context.user_message == selection
    assert context.expected_replan_source_run_id == run
