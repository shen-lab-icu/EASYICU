"""Cancellation, timeout and provider identity contracts for the Web bridge."""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from easyicu.webserver import agent_pipeline_runs, provider_adapter


def test_web_cancellation_is_a_typed_progress_control_signal() -> None:
    from easyicu.research_agent.orchestration.progress import ProgressControlSignal

    assert issubclass(agent_pipeline_runs.ResearchPipelineRunError, ProgressControlSignal)
    job = SimpleNamespace(cancel_requested=True, emit=lambda _event: None)

    with pytest.raises(ProgressControlSignal) as raised:
        agent_pipeline_runs._progress(job, step="planning", label="Planning")

    assert raised.value.code == "research_pipeline_cancelled"


@pytest.mark.parametrize("cancel_before_call", [True, False])
def test_web_cancellation_crosses_the_structured_retry_boundary(cancel_before_call):
    from easyicu.research_agent.orchestration.progress import (
        ResumableProgressChannel,
        planner_retry_progress_callback,
    )
    from easyicu.research_agent.providers.llm import LLMMessage
    from easyicu.research_agent.providers.mocks import ScriptedMockLLMClient
    from easyicu.research_agent.providers.structured_retry import call_llm_with_structured_retry

    job = SimpleNamespace(cancel_requested=cancel_before_call, emit=lambda _event: None)
    channel = ResumableProgressChannel(lambda event: agent_pipeline_runs._pipeline_progress(job, event))
    client = ScriptedMockLLMClient(["not-json", "not-json"])

    def parser(_raw):
        job.cancel_requested = True
        raise ValueError("invalid response")

    with pytest.raises(agent_pipeline_runs.ResearchPipelineRunError) as raised:
        call_llm_with_structured_retry(
            client, [LLMMessage(role="user", content="give json")],
            parser=parser, max_retries=1,
            progress_callback=planner_retry_progress_callback(channel.emit, run_id="cancel-check"),
        )

    assert raised.value.code == "research_pipeline_cancelled"
    assert len(client.calls) == (0 if cancel_before_call else 1)


@pytest.mark.parametrize(
    ("budget_mode", "expected"),
    [
        ("planner_canary", (240.0, 480.0)),
        ("candidate_plan", (240.0, 480.0)),
        ("full_reviewed", (None, None)),
    ],
)
def test_provider_request_timeouts_preserve_a_separate_hard_stop(
    budget_mode: str,
    expected: tuple[float | None, float | None],
) -> None:
    assert agent_pipeline_runs._provider_request_timeouts_for_budget(budget_mode) == expected


def test_provider_public_identity_binds_endpoint_without_disclosing_it() -> None:
    common = {
        "provider": "openai",
        "api_key": "test-key",
        "api_key_env": "OPENAI_API_KEY",
        "base_url_env": "OPENAI_BASE_URL",
        "model": "test-model",
        "model_env": "OPENAI_MODEL",
        "auth_header": "authorization",
    }

    first = provider_adapter._credential_public_metadata(
        {**common, "base_url": "https://one.example/v1/chat/completions"}
    )
    second = provider_adapter._credential_public_metadata(
        {**common, "base_url": "https://two.example/v1/chat/completions"}
    )

    assert first["endpoint_fingerprint"] != second["endpoint_fingerprint"]
    assert "one.example" not in json.dumps(first)
