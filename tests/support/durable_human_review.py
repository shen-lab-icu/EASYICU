"""Shared synthetic durable human-review fixtures."""
from pathlib import Path

import pandas as pd

from easyicu.research_agent.pipeline import ResearchAgentPipeline
from easyicu.research_agent.providers.mocks import MockLLMClient


def _pipeline(root: Path, **overrides) -> ResearchAgentPipeline:
    options = {
        "workdir": root,
        "llm": MockLLMClient(),
        "require_human_plan_review": True,
        "enable_visual_qa": False,
        "enable_publication_figure_skill": False,
        "enable_nature_writing_skill": False,
    }
    options.update(overrides)
    return ResearchAgentPipeline(**options)


def _force_approvable_plan_review(monkeypatch) -> None:
    """Remove unrelated Mock-LLM findings while keeping real review authority."""

    import easyicu.research_agent.orchestration.workflow as workflow_module

    real = workflow_module.human_review_requests_for_plan

    def plan_only(**kwargs):
        request = dict(kwargs)
        request["findings"] = []
        request["require_plan_review"] = True
        return real(**request)

    monkeypatch.setattr(
        workflow_module,
        "human_review_requests_for_plan",
        plan_only,
    )


def _cohort() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "age": [45, 60, 75] * 10,
            "death": [0, 1, 0] * 10,
        }
    )

