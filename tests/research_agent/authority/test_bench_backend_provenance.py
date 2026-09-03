"""Freeze the call-path provenance the bench records per run.

A frozen (定稿) canonical run must be unambiguous about *where* the model was
served, not just which model string was passed: ``--provider openai --model
gpt-5.5`` can route to the local Codex Tools proxy (127.0.0.1:8787) or to
api.openai.com depending on ``OPENAI_BASE_URL`` — different serving paths with
different latency / concurrency / rate-limit behaviour. These tests lock that
the resolved backend base-URL is captured in both the JSON payload key and the
human-readable run registry.
"""

from __future__ import annotations

import os
from unittest import mock

from tools.run_research_agent_bench import (
    _render_run_registry,
    _resolve_backend_base_url,
)


def test_resolve_backend_openai_prefers_env_base_url():
    with mock.patch.dict(os.environ, {"OPENAI_BASE_URL": "http://127.0.0.1:8787/v1"}):
        assert _resolve_backend_base_url("openai") == "http://127.0.0.1:8787/v1"


def test_resolve_backend_openai_defaults_to_public_api_when_unset():
    env = {k: v for k, v in os.environ.items() if k != "OPENAI_BASE_URL"}
    with mock.patch.dict(os.environ, env, clear=True):
        assert _resolve_backend_base_url("openai") == "https://api.openai.com/v1"


def test_resolve_backend_openrouter_and_mock():
    env = {k: v for k, v in os.environ.items() if k != "OPENROUTER_BASE_URL"}
    with mock.patch.dict(os.environ, env, clear=True):
        assert _resolve_backend_base_url("openrouter") == "https://openrouter.ai/api/v1"
    assert _resolve_backend_base_url("mock") == "mock://deterministic"


def test_run_registry_records_backend_line():
    payload = {
        "generated_at": "2026-07-07T00:00:00+00:00",
        "provider": "openai",
        "model": "gpt-5.5",
        "backend_base_url": "http://127.0.0.1:8787/v1",
        "git_sha": "abc1234",
        "seed": 7,
        "arms": ["aware"],
        "scores": [],
    }
    md = _render_run_registry(payload)
    # backend must be visible alongside provider/model so a frozen batch is
    # traceable to its serving path without opening each run folder.
    assert "backend: `http://127.0.0.1:8787/v1`" in md
    assert "provider/model: `openai` / `gpt-5.5`" in md
