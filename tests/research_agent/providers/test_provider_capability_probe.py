from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from tools import probe_research_agent_provider_capabilities as probe


class _SyntheticClient:
    def __init__(self, *, strict: bool, strict_supported: bool) -> None:
        self.strict = strict
        self.strict_supported = strict_supported
        self.last_finish_reason = "stop"
        self.last_transport_attempts = 1

    def complete_with_usage(self, *_args: Any, **_kwargs: Any):
        if self.strict and not self.strict_supported:
            raise RuntimeError("synthetic strict transport unavailable")
        return (
            '{"status":"ready","value":7}',
            {
                "prompt_tokens": 11,
                "completion_tokens": 9,
                "total_tokens": 20,
                "actual_model": "deepseek-v4-flash-actual",
            },
        )


def _client_factory(*, strict_supported: bool):
    def build(**kwargs: Any) -> _SyntheticClient:
        assert kwargs["provider"] == "deepseek"
        assert kwargs["model"] == "deepseek-v4-flash"
        assert kwargs["max_retries"] == 0
        assert kwargs["stream_enabled"] is False
        if kwargs["supports_strict_json_schema"]:
            assert kwargs["extra_body"] is None
        else:
            assert kwargs["extra_body"] == {"response_format": {"type": "json_object"}}
        return _SyntheticClient(
            strict=bool(kwargs["supports_strict_json_schema"]),
            strict_supported=strict_supported,
        )

    return build


@pytest.mark.parametrize("strict_supported", [False, True])
def test_capability_probe_stores_only_bounded_metadata(
    tmp_path: Path,
    strict_supported: bool,
) -> None:
    output = tmp_path / "report.json"

    report = probe.run_capability_probe(
        provider="deepseek",
        model="deepseek-v4-flash",
        environment={"DEEPSEEK_API_KEY": "test-private-key"},
        output_path=output,
        external_llm_opt_in=True,
        client_factory=_client_factory(strict_supported=strict_supported),
    )

    assert report["status"] == "usable"
    assert report["capabilities"] == {
        "host_validated_json": True,
        "json_object_mode": True,
        "strict_json_schema": strict_supported,
    }
    assert report["transport_attempt_cap"] == 2
    assert report["host_validated_json"]["actual_model"] == ("deepseek-v4-flash-actual")
    assert report["host_validated_json"]["transport_attempts"] == 1
    assert report["host_validated_json"]["transport_mode"] == "json_object"
    assert report["stores_prompt_or_response_text"] is False
    serialized = output.read_text(encoding="utf-8")
    assert "test-private-key" not in serialized
    assert '{"status":"ready","value":7}' not in serialized
    assert (output.stat().st_mode & 0o777) == 0o600


def test_account_capability_probe_uses_login_transport_without_api_key(
    tmp_path: Path,
) -> None:
    from easyicu.research_agent.providers.mocks import ExternalCaptureMockLLMClient

    calls: list[dict[str, Any]] = []

    def account_builder(**kwargs: Any) -> object:
        calls.append(kwargs)
        return SimpleNamespace(
            client=ExternalCaptureMockLLMClient(
                ['{"status":"ready","value":7}'],
            )
        )

    output = tmp_path / "codex-account.json"
    report = probe.run_capability_probe(
        provider="codex",
        model=None,
        environment={"PATH": os.environ.get("PATH", "")},
        output_path=output,
        external_llm_opt_in=True,
        account_client_factory=account_builder,
    )

    assert report["status"] == "usable"
    assert report["model"] == "cli-default"
    assert report["transport_family"] == "account_session"
    assert report["host_validated_json"]["transport_mode"] == "prompted_json"
    assert report["capabilities"] == {
        "host_validated_json": True,
        "json_object_mode": False,
        "strict_json_schema": True,
    }
    assert len(calls) == 2
    assert all(call["prefer"] == "codex" for call in calls)
    assert all(call["model"] is None for call in calls)
    assert all(call["allow_mock"] is False for call in calls)
    assert all(call["ladder"] == ["codex"] for call in calls)
    serialized = output.read_text(encoding="utf-8")
    assert "OPENAI_API_KEY" not in serialized


def test_anthropic_capability_probe_uses_native_transport_not_json_object(
    tmp_path: Path,
) -> None:
    calls: list[dict[str, Any]] = []

    def build(**kwargs: Any) -> _SyntheticClient:
        calls.append(kwargs)
        assert kwargs["provider"] == "anthropic"
        assert kwargs["extra_body"] is None
        return _SyntheticClient(
            strict=bool(kwargs["supports_strict_json_schema"]),
            strict_supported=True,
        )

    report = probe.run_capability_probe(
        provider="anthropic",
        model="claude-sonnet-4-5",
        environment={"ANTHROPIC_API_KEY": "test-private-key"},
        output_path=tmp_path / "anthropic.json",
        external_llm_opt_in=True,
        client_factory=build,
    )

    assert report["status"] == "usable"
    assert report["transport_family"] == "anthropic_messages"
    assert report["host_validated_json"]["transport_mode"] == "prompted_json"
    assert report["capabilities"] == {
        "host_validated_json": True,
        "json_object_mode": False,
        "strict_json_schema": True,
    }
    assert len(calls) == 2


def test_capability_probe_rejects_readable_env_file(tmp_path: Path) -> None:
    env_file = tmp_path / "provider.env"
    env_file.write_text("DEEPSEEK_API_KEY=test-private-key\n", encoding="utf-8")
    os.chmod(env_file, 0o644)

    with pytest.raises(ValueError, match="permissions"):
        probe._parse_env_file(env_file)


def test_capability_probe_rejects_duplicate_env_keys(tmp_path: Path) -> None:
    env_file = tmp_path / "provider.env"
    env_file.write_text(
        "DEEPSEEK_API_KEY=first\nDEEPSEEK_API_KEY=second\n",
        encoding="utf-8",
    )
    os.chmod(env_file, 0o600)

    with pytest.raises(ValueError, match="duplicate env key"):
        probe._parse_env_file(env_file)


def test_capability_probe_refuses_symlink_report_destination(tmp_path: Path) -> None:
    protected = tmp_path / "protected.json"
    protected.write_text('{"preserved":true}\n', encoding="utf-8")
    output = tmp_path / "report.json"
    output.symlink_to(protected)

    with pytest.raises(ValueError, match="unsafe capability report destination"):
        probe.run_capability_probe(
            provider="deepseek",
            model="deepseek-v4-flash",
            environment={"DEEPSEEK_API_KEY": "test-private-key"},
            output_path=output,
            external_llm_opt_in=True,
            client_factory=_client_factory(strict_supported=True),
        )

    assert json.loads(protected.read_text(encoding="utf-8")) == {"preserved": True}


def test_capability_probe_requires_explicit_external_opt_in(tmp_path: Path) -> None:
    env_file = tmp_path / "provider.env"
    env_file.write_text("DEEPSEEK_API_KEY=test-private-key\n", encoding="utf-8")
    os.chmod(env_file, 0o600)

    with pytest.raises(SystemExit, match="external-llm-opt-in"):
        probe.main(
            [
                "--provider",
                "deepseek",
                "--model",
                "deepseek-v4-flash",
                "--env-file",
                str(env_file),
                "--out",
                str(tmp_path / "report.json"),
            ]
        )


def test_capability_probe_library_gate_runs_before_client_construction(
    tmp_path: Path,
) -> None:
    from easyicu.ai_optin import AIOptInError

    constructed = []

    def forbidden_builder(**_kwargs: Any) -> object:
        constructed.append(True)
        raise AssertionError("provider client construction must be unreachable")

    with pytest.raises(AIOptInError, match="AI features are disabled"):
        probe.run_capability_probe(
            provider="deepseek",
            model="deepseek-v4-flash",
            environment={"DEEPSEEK_API_KEY": "test-private-key"},
            output_path=tmp_path / "report.json",
            client_factory=forbidden_builder,
        )

    assert constructed == []


def test_capability_probe_report_is_valid_json(tmp_path: Path) -> None:
    output = tmp_path / "report.json"
    probe.run_capability_probe(
        provider="deepseek",
        model="deepseek-v4-flash",
        environment={"DEEPSEEK_API_KEY": "test-private-key"},
        output_path=output,
        external_llm_opt_in=True,
        client_factory=_client_factory(strict_supported=True),
    )
    assert json.loads(output.read_text(encoding="utf-8"))["schema_version"] == (
        probe.REPORT_SCHEMA
    )
