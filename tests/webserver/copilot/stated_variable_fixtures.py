"""Synthetic save-tool context shared by structured question-reading tests."""

from types import SimpleNamespace

from easyicu.webserver.pi_copilot import tools as tool_module
from easyicu.webserver.pi_copilot.contracts import (
    AuthorityBinding, PiSessionRecord, ToolExecutionContext,
)


def _update(monkeypatch, *, words="在控制年龄之后", variable="年龄", concepts=("age",),
            statements=None, current=None, question=None, field="stated_adjustments"):
    question = question or f"{words}，指标与结局的关系如何？"
    current = current or {
        "id": "adjustment-study", "revision": 1, "question": question,
        "data_source": {"path": "/synthetic/source", "database": "miiv"},
    }
    writes = []
    monkeypatch.setattr(tool_module, "_bound_context", lambda _binding: dict(current))
    monkeypatch.setattr(tool_module, "_workflow_snapshot", lambda *args, **kwargs: {})
    monkeypatch.setattr(tool_module.study_contexts, "upsert_context",
                        lambda raw, **kwargs: writes.append(raw) or {**current, **raw, "revision": 2})
    monkeypatch.setattr(
        "easyicu.research_agent.acquisition.catalog.build_available_catalog",
        lambda _path: SimpleNamespace(concepts=[SimpleNamespace(concept_id=name) for name in concepts]),
    )
    context = ToolExecutionContext(data_handling="zero_retention_declared",
        session=PiSessionRecord(session_id="adjustment-session", binding=AuthorityBinding(
            study_context_id=current["id"], study_revision=current["revision"],
        )), user_message=question, allowed_actions={"configure"},
    )
    params = {"question": question, field: (
        [{"variable": variable, "words": words}] if statements is None else statements
    )}
    return tool_module.execute_tool("easyicu_update_study_context", params, context), writes


