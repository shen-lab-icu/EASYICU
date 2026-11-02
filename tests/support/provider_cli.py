"""Shared synthetic fixtures for provider and resume contracts."""
from __future__ import annotations


def _client(backend="codex", model=None):
    from easyicu.research_agent.providers.llm import CLIAgentLLMClient

    return CLIAgentLLMClient(backend=backend, model=model)



def _authorized_client(backend="codex", model=None):
    from easyicu.research_agent.providers.factory import authorize_provider_client

    client = _client(backend, model=model)
    return authorize_provider_client(
        client,
        provider=f"{backend}-cli",
        model=model or "cli-default",
        base_url=f"cli://{backend}",
        destination="external",
        environment={"EASYICU_ALLOW_EXTERNAL_LLM": "1"},
    )

