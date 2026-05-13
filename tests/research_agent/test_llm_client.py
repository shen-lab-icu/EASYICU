from __future__ import annotations

from types import SimpleNamespace


def test_openai_client_strips_reasoning_blocks_from_content(ra):
    from easyicu.research_agent.llm import LLMMessage, OpenAIClient

    class _Completions:
        def create(self, **kwargs):
            msg = SimpleNamespace(
                content="<think>private chain of thought</think>\n{\"ok\": true}"
            )
            choice = SimpleNamespace(message=msg, finish_reason="stop")
            return SimpleNamespace(choices=[choice], usage=None)

    client = OpenAIClient.__new__(OpenAIClient)
    client._client = SimpleNamespace(chat=SimpleNamespace(completions=_Completions()))
    client._model = "qwen3-8b"
    client._timeout = 120.0
    client._extra_body = {}

    out = client.complete([LLMMessage(role="user", content="return json")])

    assert out == '{"ok": true}'


def test_openai_client_recovers_unclosed_reasoning_prefix_for_debuggability(ra):
    from easyicu.research_agent.llm import LLMMessage, OpenAIClient

    class _Completions:
        def create(self, **kwargs):
            msg = SimpleNamespace(content="<think>still thinking")
            choice = SimpleNamespace(message=msg, finish_reason="length")
            return SimpleNamespace(choices=[choice], usage=None)

    client = OpenAIClient.__new__(OpenAIClient)
    client._client = SimpleNamespace(chat=SimpleNamespace(completions=_Completions()))
    client._model = "qwen3-8b"
    client._timeout = 120.0
    client._extra_body = {}

    out = client.complete([LLMMessage(role="user", content="return json")])

    assert out == "still thinking"
