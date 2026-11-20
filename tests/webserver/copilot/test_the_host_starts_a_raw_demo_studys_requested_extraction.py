"""The host starts the extraction a raw Demo study's researcher asked for.

B19 routing: on the raw-source plan card (B18) the researcher sends the card's
drafted request, the turn is granted ``extract``, and the model still wandered
through read tools and never started the extraction. Now, only when the
researcher's own sent turn is granted ``extract`` by the turn inference and the
workflow is exactly in that state for a verified official Demo raw source, the
host runs the extraction tool once before the model turn and hands the model
its receipt with no tools. Any missing condition leaves the turn to the model;
a second turn does not start it again; a negated request never fires. Fakes
only: no provider, no data.
"""

from __future__ import annotations

import time
from types import SimpleNamespace

import pytest

from easyicu.webserver.pi_copilot import host_study_extraction as host
from easyicu.webserver.pi_copilot import tools as tool_module
from easyicu.webserver.pi_copilot.contracts import PiCopilotError
from easyicu.webserver.pi_copilot.turn_authority import infer_explicit_turn_actions

pytest_plugins = ["tests.webserver.copilot.pi_copilot_contract_fixtures"]

# The raw-source plan card's drafted requests (6b93f2c69).
SENT_ZH = "我确认：按本研究现有的人群条件进行数据提取，生成本研究的数据包。"
SENT_EN = (
    "I confirm data extraction for this study’s current cohort to build this "
    "study’s data package."
)
NEGATED = "不要把这句话当成我确认进行数据提取。"
NOT_EXTRACTION = "我确认这个研究问题。"
STUDY = {
    "id": "study-test",
    "data_source": {"database": "mimic_demo", "path": "/demo/raw"},
    "modules": [],
}


def _projection(
    *,
    code: str = "plan_execution_upgrade_required",
    input_available: bool = False,
    job_status: str = "",
):
    return SimpleNamespace(
        workflow=SimpleNamespace(
            next_action_code=code,
            execution_input_available=input_available,
            current_stage="plan",
        ),
        active_job={"present": bool(job_status), "status": job_status}
        if job_status
        else {"present": False},
    )


@pytest.fixture
def verified_demo(monkeypatch):
    seen: list = []
    monkeypatch.setattr(
        host,
        "official_demo_raw_source",
        lambda study: (
            seen.append(study)
            or study.get("data_source", {}).get("path") == "/demo/raw"
        ),
    )
    return seen


def _due(text: str, *, origin: str = "typed", projection=None, study=STUDY) -> bool:
    return host.raw_demo_extraction_due(
        granted_actions=infer_explicit_turn_actions(text),
        message_origin=origin,
        workflow=projection or _projection(),
        study=study,
    )


@pytest.mark.parametrize("text", [SENT_ZH, SENT_EN])
def test_the_cards_request_in_the_raw_demo_state_is_due(verified_demo, text) -> None:
    assert _due(text)
    assert verified_demo == [STUDY]


@pytest.mark.parametrize(
    ("label", "kwargs"),
    [
        ("negated request", {"text": NEGATED}),
        ("no extraction grant", {"text": NOT_EXTRACTION}),
        ("model-written option", {"text": SENT_ZH, "origin": "model_option"}),
        ("host prompt", {"text": SENT_ZH, "origin": "host_prompt"}),
        (
            "another plan state",
            {
                "text": SENT_ZH,
                "projection": _projection(code="operator_plan_approval_required"),
            },
        ),
        (
            "source already readable",
            {"text": SENT_ZH, "projection": _projection(input_available=True)},
        ),
        (
            "extraction already running",
            {"text": SENT_ZH, "projection": _projection(job_status="running")},
        ),
        (
            "not a verified Demo raw",
            {
                "text": SENT_ZH,
                "study": {
                    **STUDY,
                    "data_source": {"database": "mimic_demo", "path": "/copy"},
                },
            },
        ),
    ],
)
def test_any_missing_condition_leaves_the_turn_to_the_model(
    verified_demo, label, kwargs
) -> None:
    text = kwargs.pop("text")
    assert not _due(text, **kwargs), label


def test_an_unverified_raw_folder_is_not_due_without_the_demo_owner(tmp_path) -> None:
    # The real Demo cache owner: a folder it did not verify is not its raw data.
    study = {**STUDY, "data_source": {"database": "mimic_demo", "path": str(tmp_path)}}
    assert not _due(SENT_ZH, study=study)


def test_the_host_submission_reports_the_tool_receipt_through_the_turn_events(
    monkeypatch,
) -> None:
    calls: list = []
    receipt = {
        "status": "ok",
        "code": "easyicu_extraction_submitted",
        "summary": "Submitted EasyICU feature extraction job extract-1.",
        "owner": "easyicu.webserver.routes.jobs",
        "details": {"job_id": "extract-1", "resource": {"kind": "extraction"}},
        "authority": {},
    }
    monkeypatch.setattr(
        tool_module,
        "execute_tool",
        lambda name, args, context: (
            calls.append((name, dict(args), context)) or receipt
        ),
    )
    events: list = []
    context = object()

    result = host.submit_host_study_extraction(
        context, call_id="host-1", emit=events.append
    )

    assert result is receipt
    assert calls == [("easyicu_start_extraction", {}, context)]
    assert [event["type"] for event in events] == ["tool_start", "tool_end"]
    end = events[-1]
    assert (end["tool_call_id"], end["tool_name"], end["code"], end["job_id"]) == (
        "host-1",
        "easyicu_start_extraction",
        "easyicu_extraction_submitted",
        "extract-1",
    )
    assert end["is_error"] is False
    assert end["resource"] == {"kind": "extraction"}


def test_a_refused_host_submission_is_a_blocked_receipt(monkeypatch) -> None:
    def refuse(name, args, context):
        raise PiCopilotError("pi_session_authority_stale", "authority changed")

    monkeypatch.setattr(tool_module, "execute_tool", refuse)
    events: list = []

    result = host.submit_host_study_extraction(
        object(), call_id="host-2", emit=events.append
    )

    assert (result["status"], result["code"]) == (
        "blocked",
        "pi_session_authority_stale",
    )
    assert events[-1]["is_error"] is True
    assert "job_id" not in events[-1]


# --- The service turn ---------------------------------------------------------


@pytest.fixture
def turn(tmp_path, study_state, monkeypatch, verified_demo):
    from easyicu.webserver.pi_copilot import service as service_module
    from easyicu.webserver.pi_copilot.service import PiCopilotService
    from tests.webserver.copilot.pi_copilot_contract_fixtures import FakeGateway

    order: list[str] = []

    class Gateway(FakeGateway):
        def request(self, method, params, **kwargs):
            if method == "session.prompt":
                order.append("model")
            return super().request(method, params, **kwargs)

    gateway = Gateway()
    service = PiCopilotService(store_path=tmp_path / "sessions.json", gateway=gateway)
    session_id = service.create_session(
        project_id="project-raw-demo", external_llm_opt_in=True
    )["session"]["session_id"]
    state = SimpleNamespace(projection=_projection(), study=dict(STUDY))
    monkeypatch.setattr(
        service_module,
        "build_project_workflow_projection",
        lambda **_: state.projection,
    )
    monkeypatch.setattr(service, "_bound_study_context", lambda record: state.study)
    monkeypatch.setattr(service, "_watch_child_job_for_replay", lambda **_: None)
    extractions: list = []

    def execute(name, args, context):
        order.append("extraction")
        extractions.append((name, context.user_message))
        context.invalidate_authority("easyicu_extraction_submitted")
        # The submission starts this study's extraction job.
        state.projection = _projection(job_status="running")
        return {
            "status": "ok",
            "code": "easyicu_extraction_submitted",
            "summary": "Submitted EasyICU feature extraction job extract-1.",
            "owner": "easyicu.webserver.routes.jobs",
            "details": {"job_id": "extract-1"},
            "authority": {},
        }

    monkeypatch.setattr(tool_module, "execute_tool", execute)

    def send(text: str, origin: str = "typed"):
        submitted = service.send_message(
            session_id,
            project_id="project-raw-demo",
            message=text,
            message_origin=origin,
        )
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            job = service_module.jobs.MANAGER.get(submitted["job_id"])
            if job and job.status != "running":
                break
            time.sleep(0.01)
        assert job.status == "done", job.error
        return [call for call in gateway.calls if call[0] == "session.prompt"][-1][1]

    return SimpleNamespace(
        send=send,
        state=state,
        extractions=extractions,
        order=order,
        service=service,
        session_id=session_id,
    )


def test_the_service_starts_the_extraction_once_before_the_model_turn(turn) -> None:
    params = turn.send(SENT_ZH)

    assert turn.extractions == [("easyicu_start_extraction", SENT_ZH)]
    assert turn.order == ["extraction", "model"]
    assert params["intent"] == host.HOST_STUDY_EXTRACTION_INTENT
    assert params["host_receipt"]["code"] == "easyicu_extraction_submitted"
    assert params["message"] == SENT_ZH

    # The extraction is now running: the same request again goes to the model.
    params = turn.send(SENT_ZH)
    assert len(turn.extractions) == 1
    assert "intent" not in params and "host_receipt" not in params


@pytest.mark.parametrize(
    ("text", "origin", "state"),
    [
        (NEGATED, "typed", {}),
        (NOT_EXTRACTION, "typed", {}),
        (SENT_ZH, "model_option", {}),
        (SENT_ZH, "typed", {"projection": _projection(input_available=True)}),
        (
            SENT_ZH,
            "typed",
            {
                "study": {
                    **STUDY,
                    "data_source": {"database": "mimic_demo", "path": "/copy"},
                }
            },
        ),
    ],
)
def test_the_service_leaves_any_other_turn_to_the_model(
    turn, text, origin, state
) -> None:
    for key, value in state.items():
        setattr(turn.state, key, value)

    params = turn.send(text, origin)

    assert turn.extractions == []
    assert turn.order == ["model"]
    assert params.get("intent") != host.HOST_STUDY_EXTRACTION_INTENT
    assert "host_receipt" not in params
