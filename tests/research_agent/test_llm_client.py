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


def test_openai_client_supports_local_noauth_proxy_mode(ra):
    from easyicu.research_agent.llm import LLMMessage, OpenAIClient

    class _Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": "OK", "role": "assistant"},
                    }
                ],
                "usage": {
                    "prompt_tokens": 12,
                    "completion_tokens": 4,
                    "total_tokens": 16,
                },
            }

    class _HttpClient:
        def __init__(self):
            self.calls = []

        def post(self, path, json):
            self.calls.append((path, json))
            return _Response()

    http_client = _HttpClient()
    client = OpenAIClient.__new__(OpenAIClient)
    client._client = None
    client._model = "gpt-5.4"
    client._timeout = 120.0
    client._extra_body = {}
    client._local_noauth_mode = True
    client._local_http_client = http_client

    out = client.complete(
        [LLMMessage(role="user", content="Reply with the single word OK.")],
        seed=7,
    )

    assert out == "OK"
    assert http_client.calls[0][0] == "/chat/completions"
    assert http_client.calls[0][1]["model"] == "gpt-5.4"
    assert http_client.calls[0][1]["seed"] == 7
    assert client.last_usage == {
        "prompt_tokens": 12,
        "completion_tokens": 4,
        "total_tokens": 16,
    }
