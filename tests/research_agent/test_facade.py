"""Governed one-shot facade: Biomni-style entry, EasyICU fail-closed rules."""

from __future__ import annotations

import ast
import inspect

from easyicu.research_agent import facade
from easyicu.research_agent.providers.mocks import MockLLMClient


def test_go_defaults_to_offline_mock_and_delegates_to_pipeline(monkeypatch) -> None:
    sentinel = object()

    def fake_run(self, **kwargs):
        assert kwargs["question"] == "describe this cohort"
        assert kwargs["cohort"] == "cohort.csv"
        assert kwargs["stop_after_analysis"] is True
        return sentinel

    captured = {}

    def fake_init(self, **kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(facade.ResearchAgentPipeline, "run", fake_run)
    monkeypatch.setattr(facade.ResearchAgentPipeline, "__init__", fake_init)

    result = facade.go("describe this cohort", cohort="cohort.csv")

    assert result is sentinel
    assert isinstance(captured["llm"], MockLLMClient)
    assert captured["workdir"] is None


def test_go_does_not_expose_publication_authority_toggles() -> None:
    source = inspect.getsource(facade.go)
    for forbidden in (
        "formal_mode",
        "paper_authority",
        "publication_authority",
        "manuscript_authority",
        "max_step_provider_calls",
        "provider_budget",
    ):
        assert forbidden not in source

    tree = ast.parse(source)
    params = {node.arg for node in ast.walk(tree) if isinstance(node, ast.arg)}
    assert "question" in params  # the one Biomni-style positional request
    assert "llm" in params
    assert "cohort" in params
