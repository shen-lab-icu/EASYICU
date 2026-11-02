"""Synthetic provider and evidence fixtures for progressive retry tests."""
from __future__ import annotations

import hashlib
from pathlib import Path

from easyicu.research_agent.authority.provider_hard_stop import ProviderHardStopLedger
from easyicu.research_agent.planning import outline_retry_journal as owner
from easyicu.webserver import provider_adapter


def _setup(tmp_path, monkeypatch):
    import easyicu.research_agent.providers.llm  # noqa: F401
    monkeypatch.setattr(owner.state_paths, "state_root", lambda: tmp_path / "state")
    env = {"OPENAI_API_KEY": "synthetic-secret", "OPENAI_BASE_URL": "http://127.0.0.1:1/v1",
           "OPENAI_MODEL": "synthetic", "EASYICU_DISABLE_PROVIDER_ENV_FILE": "1",
           "EASYICU_PROVIDER_DATA_HANDLING": "user_accepted_external"}
    inner, _ = provider_adapter.build_research_agent_provider_client({"provider": "openai", "external": True}, environ=env)
    book = ProviderHardStopLedger(path=tmp_path / "budget.json", task_ids=("task",),
        limits=provider_adapter.web_research_agent_hard_stop_limits("planner_canary"),
        batch_id="task", declaration_sha256="a" * 64)
    return inner, book, book.start_task("task")


class _RecordingEvidence:
    def __init__(self) -> None:
        self.records: dict[str, dict[str, object]] = {}

    def get(self, evidence_id_or_alias: str) -> object | None:
        return self.records.get(evidence_id_or_alias)

    def register_file(self, **kwargs: object) -> object:
        evidence_id = str(kwargs["evidence_id"])
        source_path = Path(str(kwargs["source_path"]))
        self.records[evidence_id] = {
            **dict(kwargs),
            "sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        }
        return self.records[evidence_id]
