from __future__ import annotations

import json
import stat
from pathlib import Path

import pytest

from tools.configure_webserver_provider_env import write_provider_env
from tools.run_webserver_provider_point_fire import safe_summary, safety_failures


def test_configure_provider_env_writes_0600_without_returning_secret(tmp_path: Path) -> None:
    target = tmp_path / "provider.env"

    meta = write_provider_env(
        target,
        {
            "OPENAI_API_KEY": "sk-test-provider-tool",
            "OPENAI_BASE_URL": "http://127.0.0.1:8787/v1",
            "OPENAI_MODEL": "gpt-private",
            "EASYICU_LLM_MAX_TOKENS": "640",
        },
    )

    mode = stat.S_IMODE(target.stat().st_mode)
    assert mode == 0o600
    assert meta["mode"] == "0600"
    assert meta["secrets_returned"] is False
    assert "OPENAI_API_KEY" in meta["keys"]
    assert "sk-test-provider-tool" not in json.dumps(meta)
    text = target.read_text(encoding="utf-8")
    assert "OPENAI_API_KEY=sk-test-provider-tool" in text


def test_configure_provider_env_refuses_existing_without_force(tmp_path: Path) -> None:
    target = tmp_path / "provider.env"
    write_provider_env(target, {"OPENAI_API_KEY": "first"})

    with pytest.raises(FileExistsError):
        write_provider_env(target, {"OPENAI_API_KEY": "second"})

    meta = write_provider_env(target, {"OPENAI_API_KEY": "second"}, force=True)
    assert meta["secrets_returned"] is False
    assert "second" in target.read_text(encoding="utf-8")


def test_point_fire_safe_summary_strips_provider_values_and_checks_safety() -> None:
    snapshot = {
        "id": "job1",
        "status": "done",
        "result": {
            "run_id": "run_job1",
            "run_type": "full",
            "project_dir": "/tmp/easyicu/run_job1",
            "uploads": 0,
            "tokens": 0,
            "provider": {
                "provider": "openai",
                "external": True,
                "credentials_loaded": True,
                "credential_source": "OPENAI_API_KEY",
                "credential_fingerprint": "abc123",
                "base_url": "http://127.0.0.1:8787/v1/chat/completions",
                "base_url_source": "OPENAI_BASE_URL",
                "base_url_endpoint": "chat_completions",
                "model": "gpt-private",
                "model_source": "OPENAI_MODEL",
                "client_constructed": True,
                "external_calls": 1,
                "usage": {"total_tokens": 12},
            },
            "gate": {
                "status": "analysis_only",
                "reason": "full_agent_complete_human_signoff_required",
                "reportable": False,
                "draft_unlocked": False,
                "checks": [
                    {"id": "source_valid", "passed": True},
                    {
                        "id": "no_patient_rows_persisted",
                        "passed": True,
                        "scanned_artifacts": 6,
                        "row_level_markers": [],
                    },
                ],
            },
            "strict_evidence_audit": {
                "claims_passed": True,
                "sentences_passed": True,
            },
        },
    }

    summary = safe_summary(snapshot, review={"readiness": {"status": "awaiting_human_signoff"}})

    serialized = json.dumps(summary)
    assert "http://127.0.0.1:8787" not in serialized
    assert "gpt-private" not in serialized
    assert summary["provider"]["base_url_source"] == "OPENAI_BASE_URL"
    assert summary["provider"]["model_source"] == "OPENAI_MODEL"
    assert summary["provider"]["external_calls"] == 1
    assert safety_failures(summary) == []


def test_point_fire_safety_failures_keep_draft_locked() -> None:
    summary = {
        "provider": {"external_calls": 1, "client_constructed": True},
        "gate": {"reportable": False, "draft_unlocked": True},
        "privacy": {"passed": True},
        "uploads": 0,
        "tokens": 0,
    }

    assert "gate.draft_unlocked must remain false" in safety_failures(summary)
