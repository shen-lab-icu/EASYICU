"""Scripted Copilot model and registered synthetic source for quick answers."""

from __future__ import annotations

import time
from typing import Any

import pandas as pd
import pytest

from easyicu.extensions import ExtensionRegistry
from easyicu.webserver import settings, sources, study_contexts
from easyicu.webserver.pi_copilot import service as service_module
from easyicu.webserver.pi_copilot.contracts import (
    AuthorityBinding, PiCopilotError, PiSessionDataSourceAuthorization, PiSessionRecord,
)
from easyicu.webserver.pi_copilot.service import PiCopilotService
from easyicu.webserver.pi_copilot.tools import execute_tool
from tests.support.native_outcome_export import typed_native_export
from tests.webserver.copilot.pi_copilot_contract_fixtures import FakeGateway


QUESTION = "在 eICU demo 中，入 ICU 第一天最高乳酸的中位数是多少？"


TRANSITION = "EasyICU 已确认所选数据来源，请在当前对话中继续。"


def reading(question: str = QUESTION) -> dict[str, Any]:
    return {"question": question, "intent": "describe", "population": {"criteria": []},
            "measure": {"concept": "lact", "summary": "max", "statistic": "median",
                        "window": {"start_hours": 0, "end_hours": 24}}}


class ModelGateway(FakeGateway):
    """The conversation's model: submits a reading of the offered question."""

    def __init__(self) -> None:
        super().__init__()
        self.submitted: list[dict[str, Any]] = []
        self.call_quick_answer_anyway = False
        self.declines = False

    def request(self, method: str, params: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        response = super().request(method, params, **kwargs)
        context = kwargs.get("tool_context")
        offered = params.get("quick_question")
        if self.declines:
            return response
        if method == "session.prompt" and context is not None and (offered or self.call_quick_answer_anyway):
            # The sidecar hands a host refusal back to the model as a tool error.
            try:
                self.submitted.append(execute_tool(
                    "easyicu_quick_answer", {"reading": reading(offered or QUESTION)}, context,
                ))
            except PiCopilotError as exc:
                self.submitted.append({"status": "blocked", "code": exc.code})
        return response


@pytest.fixture
def conversation(tmp_path, monkeypatch):
    export = typed_native_export(
        tmp_path / "synthetic", database="eicu_demo",
        longitudinal=pd.DataFrame({"stay_id": range(60), "charttime": [1.0] * 60,
                                   "lact": [float(i % 15 + 1) / 2 for i in range(60)]}),
        longitudinal_concepts=["lact"],
        statics=pd.DataFrame({"stay_id": range(60), "age": [45] * 60, "sex": ["Male"] * 30 + ["Female"] * 30}),
        static_concepts=["age", "sex"],
        outcome=pd.DataFrame({"stay_id": range(60), "charttime": [0.0] * 60, "icu_death": [False] * 30 + [True] * 30}),
        outcome_concepts=["icu_death"],
    )
    registry = {"sources": [{"id": "src_synthetic", "database": "eicu_demo", "path": str(export), "ok": True}]}
    monkeypatch.setattr(sources, "load_registry", lambda **_kwargs: registry)
    monkeypatch.setattr(study_contexts, "_CONFIG_PATH", tmp_path / "studies.json")
    monkeypatch.setattr(settings, "load_settings", lambda: {"ai_enabled": True, "language": "zh"})
    study = study_contexts.upsert_context({"id": "study-transition", "question": QUESTION,
                                         "data_source": {"database": "eicu_demo", "path": str(export)}})
    gateway = ModelGateway()
    service = PiCopilotService(store_path=tmp_path / "sessions.json", gateway=gateway,
                              extension_registry=ExtensionRegistry(tmp_path / "extensions"))
    service.project_store.bind("project-transition", study["id"])
    record = PiSessionRecord(
        session_id="pi-transition", project_id="project-transition", external_llm_opt_in=True,
        credentialed_data_in_history=False,
        binding=AuthorityBinding(study_context_id=study["id"], study_revision=study["revision"]),
        data_source_authorization=PiSessionDataSourceAuthorization(
            status="confirmed", confirmation_mode="select_registered_source",
        ),
    )
    service._save_record(record)
    yield service, record, gateway
    service.close()


def typed_turn(service, record, text: str, *, job_id: str = "job-question", origin: str = "typed") -> None:
    service.replay_store.start_turn(session_id=record.session_id, project_id=record.project_id,
                                    job_id=job_id, allowed_actions=["configure"],
                                    user_message=text, message_origin=origin)
    service.replay_store.finish_turn(session_id=record.session_id, project_id=record.project_id,
                                     job_id=job_id, status="done", assistant_reply="研究问题已保存。")


def confirm_source_and_continue(service, record) -> str:
    submitted = service.send_message(
        record.session_id, project_id=record.project_id, message=TRANSITION,
        message_intent="advance_after_data_source_confirmation", message_origin="host_prompt",
    )
    deadline = time.monotonic() + 30
    job = None
    while time.monotonic() < deadline:
        job = service_module.jobs.MANAGER.get(submitted["job_id"])
        if job and job.status != "running" and job.status != "queued":
            break
        time.sleep(0.02)
    assert job is not None and job.status == "done", job and (job.status, getattr(job, "error", None))
    return submitted["job_id"]


def prompts(gateway) -> list[dict[str, Any]]:
    return [params for method, params, _ in gateway.calls if method == "session.prompt"]
