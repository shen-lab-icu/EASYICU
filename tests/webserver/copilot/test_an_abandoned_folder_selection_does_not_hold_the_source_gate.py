"""An abandoned local folder selection does not hold a conversation's data gate.

Opening the folder panel holds the session in ``selection_in_progress`` so the
panel can confirm what it binds.  A researcher who leaves the panel and picks a
registered export in the conversation, or who leaves it without choosing, must
reach the source confirmation a new session would show, not a card whose only
way out is the folder panel.  Neither path confirms the source for them.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from easyicu.webserver.pi_copilot import service as service_module
from easyicu.webserver.pi_copilot.contracts import PiCopilotError
from easyicu.webserver.pi_copilot.service import PiCopilotService
from easyicu.webserver.routes.pi_copilot import PiDataSourceAuthorizationRequest
from tests.webserver.copilot.pi_copilot_contract_fixtures import (  # noqa: F401 - fixture
    FakeGateway,
    study_state,
)

PROJECT = "project-abandoned-folder-selection"
REGISTERED = "/private/registered-eicu-export"


class BindingGateway(FakeGateway):
    """A turn whose study tool binds the study to ``bind_path``."""

    def __init__(self, study: dict[str, Any], bind_path: str | None) -> None:
        super().__init__()
        self.study = study
        self.bind_path = bind_path
        self.authorization_during_turn: list[str] = []

    def request(self, method: str, params: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        if method == "session.prompt":
            context = kwargs.get("tool_context")
            if context is not None:
                self.authorization_during_turn.append(
                    context.session.data_source_authorization.status
                )
            if self.bind_path is not None:
                self.study["data_source"] = {
                    "database": "eicu",
                    "label": "eICU v2.0",
                    "path": self.bind_path,
                }
        return super().request(method, params, **kwargs)


@pytest.fixture
def registry(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        service_module.sources,
        "load_registry",
        lambda: {
            "sources": [
                {"id": "src_eicu", "path": REGISTERED, "database": "eicu",
                 "label": "eICU v2.0", "ok": True},
                {"id": "src_broken", "path": "/private/unvalidated", "database": "eicu",
                 "label": "eICU v2.0", "ok": False},
            ]
        },
    )


def _service(tmp_path: Path, gateway: FakeGateway) -> PiCopilotService:
    return PiCopilotService(store_path=tmp_path / "sessions.json", gateway=gateway)


def _selecting(service: PiCopilotService) -> str:
    session_id = service.create_session(project_id=PROJECT, external_llm_opt_in=True)[
        "session"]["session_id"]
    opened = service.authorize_data_source(
        session_id, project_id=PROJECT, action="begin_local_selection", database="eicu",
    )
    assert opened["session"]["data_source_authorization"]["status"] == "selection_in_progress"
    return session_id


def _turn(service: PiCopilotService, session_id: str, message: str) -> dict[str, Any]:
    submitted = service.send_message(session_id, project_id=PROJECT, message=message)
    deadline = time.monotonic() + 3
    job = None
    while time.monotonic() < deadline:
        job = service_module.jobs.MANAGER.get(submitted["job_id"])
        if job and job.status != "running":
            break
        time.sleep(0.01)
    assert job is not None and job.status == "done"
    return service._get_record(session_id).data_source_authorization.model_dump(mode="json")


def test_a_turn_that_binds_a_registered_export_ends_the_selection_unconfirmed(
    tmp_path: Path, study_state: dict[str, Any], registry: None,
) -> None:
    study_state["data_source"] = {}
    gateway = BindingGateway(study_state, REGISTERED)
    service = _service(tmp_path, gateway)
    session_id = _selecting(service)

    authorization = _turn(service, session_id, "就用你推荐的那一份")

    # The conversation's choice is not a confirmation: the researcher still
    # confirms this source on the host's own card.
    assert authorization["status"] == "pending"
    assert authorization["status"] != "confirmed"
    assert authorization["reason"] == "project_source_confirmation_required"
    assert authorization["confirmed_at"] is None
    assert authorization["source"]["database"] == "eicu"
    assert gateway.authorization_during_turn == ["selection_in_progress"]
    # A later rebind keeps the gate, as for any session with a bound source.
    rebound = service.rebind_session(session_id, project_id=PROJECT)
    assert rebound["session"]["data_source_authorization"]["status"] == "pending"


def test_an_explicit_choice_of_the_registered_export_still_confirms_it(
    tmp_path: Path, study_state: dict[str, Any], registry: None,
) -> None:
    study_state["data_source"] = {}
    service = _service(tmp_path, BindingGateway(study_state, REGISTERED))
    session_id = _selecting(service)

    authorization = _turn(
        service, session_id, "确认使用 EasyICU 中已准备好的 eICU v2.0 数据导出",
    )

    assert authorization["status"] == "confirmed"
    assert authorization["confirmation_mode"] == "reuse_project_source"


@pytest.mark.parametrize(
    ("before", "bound"),
    [
        # The panel is open on a study whose registered source the turn left as
        # it was: chatting while choosing a folder does not end the choice.
        ({"database": "eicu", "label": "eICU v2.0", "path": REGISTERED}, None),
        # A path that is not a validated registered export is not a source the
        # conversation can stand behind.
        ({}, "/private/unvalidated"),
        ({}, "/private/not-registered"),
        ({}, None),
    ],
)
def test_a_turn_that_binds_no_new_registered_export_keeps_the_selection(
    tmp_path: Path, study_state: dict[str, Any], registry: None,
    before: dict[str, Any], bound: str | None,
) -> None:
    study_state["data_source"] = dict(before)
    service = _service(tmp_path, BindingGateway(study_state, bound))
    session_id = _selecting(service)

    authorization = _turn(service, session_id, "这份数据有多少例？")

    assert authorization["status"] == "selection_in_progress"


def test_a_session_without_an_open_selection_keeps_its_own_gate(
    tmp_path: Path, study_state: dict[str, Any], registry: None,
) -> None:
    # Only an open selection is the reconciler's: a session still waiting for
    # its first source keeps the gate its rebind recomputes.
    study_state["data_source"] = {}
    service = _service(tmp_path, BindingGateway(study_state, REGISTERED))
    session_id = service.create_session(project_id=PROJECT, external_llm_opt_in=True)[
        "session"]["session_id"]
    before = service._get_record(session_id).data_source_authorization.model_dump(mode="json")

    authorization = _turn(service, session_id, "就用你推荐的那一份")

    assert before["status"] == "pending"
    assert authorization == before


@pytest.mark.parametrize(
    ("source", "reason"),
    [
        ({}, "local_data_selection_required"),
        ({"database": "eicu", "label": "eICU v2.0", "path": REGISTERED},
         "project_source_confirmation_required"),
    ],
)
def test_leaving_the_folder_panel_returns_the_session_to_a_new_sessions_gate(
    tmp_path: Path, study_state: dict[str, Any], registry: None,
    source: dict[str, Any], reason: str,
) -> None:
    study_state["data_source"] = dict(source)
    service = _service(tmp_path, FakeGateway())
    session_id = _selecting(service)

    left = service.authorize_data_source(
        session_id, project_id=PROJECT, action="cancel_local_selection",
    )

    authorization = left["session"]["data_source_authorization"]
    assert left["resource"] is None
    assert authorization["status"] == "pending"
    assert authorization["reason"] == reason
    assert authorization["confirmed_at"] is None
    assert service._get_record(session_id).data_source_authorization.status == "pending"


def test_leaving_needs_an_open_selection(
    tmp_path: Path, study_state: dict[str, Any], registry: None,
) -> None:
    service = _service(tmp_path, FakeGateway())
    session_id = service.create_session(project_id=PROJECT, external_llm_opt_in=True)[
        "session"]["session_id"]

    with pytest.raises(PiCopilotError) as raised:
        service.authorize_data_source(
            session_id, project_id=PROJECT, action="cancel_local_selection",
        )

    assert raised.value.code == "pi_session_local_selection_not_started"
    assert raised.value.status_code == 409


def test_a_running_data_job_keeps_its_selection_open(
    tmp_path: Path, study_state: dict[str, Any], registry: None,
) -> None:
    service = _service(tmp_path, FakeGateway())
    session_id = _selecting(service)
    study_state["active_job_id"] = "job-extraction"

    with pytest.raises(PiCopilotError) as raised:
        service.authorize_data_source(
            session_id, project_id=PROJECT, action="cancel_local_selection",
        )

    assert raised.value.code == "pi_session_local_selection_job_active"
    assert raised.value.status_code == 409
    assert (
        service._get_record(session_id).data_source_authorization.status
        == "selection_in_progress"
    )


def test_the_route_accepts_leaving_and_nothing_else_new() -> None:
    assert PiDataSourceAuthorizationRequest(
        project_id=PROJECT, action="cancel_local_selection",
    ).action == "cancel_local_selection"
    with pytest.raises(ValidationError):
        PiDataSourceAuthorizationRequest(project_id=PROJECT, action="abandon_everything")
