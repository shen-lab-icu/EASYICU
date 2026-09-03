"""Credential-safe provenance and opt-in LLM parse diagnostics."""

from __future__ import annotations

import json
import os
import stat

import pytest


def test_config_provenance_hashes_secret_keys_and_credential_values(tmp_path):
    from easyicu.research_agent.orchestration.config import PipelineConfig

    secrets = {
        "authorization": "Bearer auth-secret-123456789",
        "proxy_authorization": "Basic proxy-secret-123456789",
        "cookie": "session=browser-cookie-secret",
        "set_cookie": "session=server-cookie-secret; HttpOnly",
        "session_id": "session-secret-123",
        "dsn": "postgresql://db-user:db-password@db.example/easyicu",
        "connection_string": "Server=db;Password=connection-secret",
        "database_url": "postgresql://url-user:url-password@db.example/easyicu",
        "embedded_url": "mysql://mysql-user:mysql-password@db.example/easyicu",
        "token_value": "Bearer unfamiliar-token-123456",
        "private_key": "-----BEGIN PRIVATE KEY-----private-material-----END PRIVATE KEY-----",
    }
    config = PipelineConfig(
        workdir=tmp_path,
        runner_kwargs={
            "headers": {
                "Authorization": secrets["authorization"],
                "Proxy-Authorization": secrets["proxy_authorization"],
                "Cookie": secrets["cookie"],
                "Set-Cookie": secrets["set_cookie"],
            },
            "session_id": secrets["session_id"],
            "dsn": secrets["dsn"],
            "connection_string": secrets["connection_string"],
            "database_url": secrets["database_url"],
            "private_key": secrets["private_key"],
            "transport": {
                "endpoint": secrets["embedded_url"],
                "public_endpoint": "https://db.example/easyicu",
                "description": secrets["token_value"],
            },
        },
    )

    payload = config.canonical_payload()
    encoded = json.dumps(payload, sort_keys=True)
    for secret in secrets.values():
        assert secret not in encoded
    runner = payload["runner_kwargs"]
    assert runner["headers"]["Authorization"].startswith("sha256:")
    assert runner["headers"]["Proxy-Authorization"].startswith("sha256:")
    assert runner["headers"]["Cookie"].startswith("sha256:")
    assert runner["headers"]["Set-Cookie"].startswith("sha256:")
    assert runner["session_id"].startswith("sha256:")
    assert runner["dsn"].startswith("sha256:")
    assert runner["connection_string"].startswith("sha256:")
    assert runner["database_url"].startswith("sha256:")
    assert runner["private_key"].startswith("sha256:")
    assert runner["transport"]["endpoint"].startswith("sha256:")
    assert runner["transport"]["description"].startswith("sha256:")
    assert runner["transport"]["public_endpoint"] == "https://db.example/easyicu"

    rotated = config.with_overrides(
        runner_kwargs={
            **dict(config.runner_kwargs or {}),
            "dsn": "postgresql://db-user:rotated@db.example/easyicu",
        }
    )
    assert rotated.canonical_digest() != config.canonical_digest()


def test_config_recovery_preserves_digest_bound_literature_keys(tmp_path):
    from easyicu.research_agent.orchestration.config import PipelineConfig

    citation_key = "pubmed_41159833"
    config = PipelineConfig(
        workdir=tmp_path / "pipeline",
        bound_preplan_literature={
            "research_question": "SOFA-2 and in-hospital mortality",
            "citations": [
                {
                    "key": citation_key,
                    "title": "Development and Validation of the SOFA-2 Score",
                    "pmid": "41159833",
                }
            ],
            "screening_decisions": [],
        },
    )

    payload = config.recovery_payload()
    assert payload["bound_preplan_literature"]["citations"][0]["key"] == (
        citation_key
    )
    restored = PipelineConfig.from_recovery_payload(
        payload,
        expected_digest=config.canonical_digest(),
    )
    assert restored.canonical_digest() == config.canonical_digest()
    assert restored.bound_preplan_literature["citations"][0]["key"] == citation_key


def test_config_recovery_refuses_credentials_and_opaque_runner_state(tmp_path):
    from easyicu.research_agent.orchestration.config import (
        PipelineConfig,
        PipelineConfigRecoveryError,
    )

    with pytest.raises(
        PipelineConfigRecoveryError,
        match="pipeline_config_recovery_field_not_persistable:pubmed_api_key",
    ):
        PipelineConfig(
            workdir=tmp_path / "with-key",
            pubmed_api_key="provider-secret-123456789",
        ).recovery_payload()

    with pytest.raises(
        PipelineConfigRecoveryError,
        match="pipeline_config_recovery_field_not_persistable:runner_kwargs",
    ):
        PipelineConfig(
            workdir=tmp_path / "with-runner",
            runner_kwargs={"headers": {"Authorization": "Bearer secret"}},
        ).recovery_payload()


def test_config_recovery_rejects_payload_drift(tmp_path):
    from easyicu.research_agent.orchestration.config import (
        PipelineConfig,
        PipelineConfigRecoveryError,
    )

    config = PipelineConfig(workdir=tmp_path / "pipeline")
    payload = config.recovery_payload()
    payload["enable_literature"] = False

    with pytest.raises(
        PipelineConfigRecoveryError,
        match="pipeline_config_recovery_digest_mismatch",
    ):
        PipelineConfig.from_recovery_payload(
            payload,
            expected_digest=config.canonical_digest(),
        )


def test_raw_parse_dump_is_off_without_explicit_flag_and_directory(
    tmp_path, monkeypatch
):
    from easyicu.research_agent.agents.core import _dump_raw

    debug_dir = tmp_path / "debug"
    monkeypatch.setenv("EASYICU_LLM_DEBUG", "0")
    monkeypatch.setenv("EASYICU_LLM_DEBUG_DIR", str(debug_dir))
    assert _dump_raw("secret response", "planner") is None
    assert not debug_dir.exists()

    monkeypatch.setenv("EASYICU_LLM_DEBUG", "1")
    monkeypatch.delenv("EASYICU_LLM_DEBUG_DIR", raising=False)
    assert _dump_raw("secret response", "planner") is None


@pytest.mark.skipif(os.name != "posix", reason="POSIX file modes only")
def test_raw_parse_dump_is_bounded_redacted_and_owner_only(tmp_path, monkeypatch):
    from easyicu.research_agent.agents.core import (
        LLM_PARSE_DEBUG_CHARS,
        _dump_raw,
    )

    debug_dir = tmp_path / "run" / "llm_debug"
    monkeypatch.setenv("EASYICU_LLM_DEBUG", "1")
    monkeypatch.setenv("EASYICU_LLM_DEBUG_DIR", str(debug_dir))
    raw = (
        "Authorization: Bearer auth-secret-123456789\n"
        "Cookie: session=cookie-secret-123456\n"
        "dsn=postgresql://db-user:db-password@db.example/easyicu\n"
        + ("x" * (LLM_PARSE_DEBUG_CHARS * 2))
    )

    path = _dump_raw(raw, "../../planner")

    assert path is not None
    assert path.parent == debug_dir
    assert ".." not in path.name
    assert stat.S_IMODE(debug_dir.stat().st_mode) == 0o700
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    payload = json.loads(path.read_text(encoding="utf-8"))
    encoded = json.dumps(payload)
    assert payload["schema_version"] == "easyicu.llm_parse_debug/1"
    assert payload["response_chars"] == len(raw)
    assert payload["truncated"] is True
    assert len(payload["response_head"]) <= LLM_PARSE_DEBUG_CHARS
    assert "auth-secret" not in encoded
    assert "cookie-secret" not in encoded
    assert "db-password" not in encoded
    assert "[REDACTED]" in encoded
