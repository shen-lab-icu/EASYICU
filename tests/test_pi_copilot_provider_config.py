"""Private credential and first-use gate tests for Pi Copilot."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import pytest

from easyicu.webserver.pi_copilot.contracts import PiCopilotError
from easyicu.webserver.pi_copilot.provider_config import (
    DEFAULT_MODEL,
    PiProviderConfigStore,
)


def test_pi_provider_owner_uses_shared_url_security_not_scientific_adapter() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "src/easyicu/webserver/pi_copilot/provider_config.py"
    ).read_text(encoding="utf-8")
    assert "provider_url_security" in source
    assert "provider_adapter" not in source


def test_cliproxyapi_default_uses_the_reported_model_identifier() -> None:
    assert DEFAULT_MODEL == "gpt-5.6-luna"


def _models_ok(
    url: str,
    headers: Mapping[str, str],
    timeout: float,
) -> tuple[int, Any]:
    assert url == "http://127.0.0.1:8317/v1/models"
    assert headers["Authorization"] == "Bearer test-private-key"
    assert timeout > 0
    return 200, {"data": [{"id": "gpt5.6 luna"}]}


def test_verified_provider_config_is_private_and_never_returns_secret(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "pi-provider.env"
    receipt_path = tmp_path / "pi-provider-verification.json"
    store = PiProviderConfigStore(
        config_path=config_path,
        receipt_path=receipt_path,
    )

    config, public = store.verify_and_save(
        provider="easyicu-local",
        api_key="test-private-key",
        base_url="http://127.0.0.1:8317/v1",
        model="gpt5.6 luna",
        api_transport="openai-completions",
        verifier=_models_ok,
    )

    assert config.api_key == "test-private-key"
    assert public["credential_present"] is True
    assert public["connection_verified"] is True
    assert public["model_available"] is True
    assert public["secrets_returned"] is False
    assert "test-private-key" not in json.dumps(public)
    assert (config_path.stat().st_mode & 0o777) == 0o600
    assert (receipt_path.stat().st_mode & 0o777) == 0o600
    assert "test-private-key" in config_path.read_text(encoding="utf-8")
    assert "test-private-key" not in receipt_path.read_text(encoding="utf-8")

    restored = store.public_status()
    assert restored["connection_verified"] is True
    assert restored["credential_present"] is True
    assert "test-private-key" not in json.dumps(restored)


def test_failed_verification_does_not_persist_credentials(tmp_path: Path) -> None:
    store = PiProviderConfigStore(
        config_path=tmp_path / "pi-provider.env",
        receipt_path=tmp_path / "pi-provider-verification.json",
    )

    def rejected(
        url: str,
        headers: Mapping[str, str],
        timeout: float,
    ) -> tuple[int, Any]:
        return 401, {"error": "invalid key"}

    with pytest.raises(PiCopilotError) as caught:
        store.verify_and_save(
            provider="easyicu-local",
            api_key="must-not-be-written",
            base_url="http://127.0.0.1:8317/v1",
            model="gpt5.6 luna",
            api_transport="openai-completions",
            verifier=rejected,
        )

    assert caught.value.code == "pi_provider_auth_failed"
    assert not store.config_path.exists()
    assert not store.receipt_path.exists()
    assert "must-not-be-written" not in json.dumps(caught.value.detail)


def test_model_must_be_reported_by_verified_endpoint(tmp_path: Path) -> None:
    store = PiProviderConfigStore(
        config_path=tmp_path / "pi-provider.env",
        receipt_path=tmp_path / "pi-provider-verification.json",
    )

    with pytest.raises(PiCopilotError) as caught:
        store.verify_and_save(
            provider="easyicu-local",
            api_key="must-not-be-written",
            base_url="http://127.0.0.1:8317/v1",
            model="missing-model",
            api_transport="openai-completions",
            verifier=lambda *_: (
                200,
                {
                    "data": [
                        {"id": "served-model"},
                        {"id": "reflected-must-not-be-written"},
                    ]
                },
            ),
        )

    assert caught.value.code == "pi_provider_model_unavailable"
    assert caught.value.details == {
        "available_models": ["served-model"],
        "models_reported": 1,
    }
    assert not store.config_path.exists()


@pytest.mark.parametrize(
    ("transport", "base_url", "payload", "model", "expected_headers"),
    [
        (
            "anthropic-messages",
            "https://api.anthropic.com/v1",
            {"data": [{"id": "claude-sonnet-4-6"}]},
            "claude-sonnet-4-6",
            {"x-api-key": "test-private-key", "anthropic-version": "2023-06-01"},
        ),
        (
            "google-generative-ai",
            "https://generativelanguage.googleapis.com/v1beta",
            {"models": [{"name": "models/gemini-3.5-flash"}]},
            "gemini-3.5-flash",
            {"x-goog-api-key": "test-private-key"},
        ),
    ],
)
def test_native_provider_protocols_use_their_own_auth_and_catalog_shape(
    tmp_path: Path,
    transport: str,
    base_url: str,
    payload: dict[str, Any],
    model: str,
    expected_headers: dict[str, str],
) -> None:
    store = PiProviderConfigStore(
        config_path=tmp_path / "pi-provider.env",
        receipt_path=tmp_path / "receipt.json",
    )

    def verify(
        url: str,
        headers: Mapping[str, str],
        timeout: float,
    ) -> tuple[int, Any]:
        assert url.endswith("/models")
        assert timeout > 0
        assert "Authorization" not in headers
        for key, value in expected_headers.items():
            assert headers[key] == value
        return 200, payload

    _config, public = store.verify_and_save(
        provider="native-provider",
        api_key="test-private-key",
        base_url=base_url,
        model=model,
        api_transport=transport,
        verifier=verify,
    )

    assert public["connection_verified"] is True


def test_insecure_config_file_is_not_loaded(tmp_path: Path) -> None:
    config_path = tmp_path / "pi-provider.env"
    config_path.write_text("EASYICU_PI_API_KEY=exposed\n", encoding="utf-8")
    config_path.chmod(0o644)
    store = PiProviderConfigStore(
        config_path=config_path,
        receipt_path=tmp_path / "receipt.json",
    )

    status = store.public_status(environ={})

    assert status["credential_present"] is False
    assert status["config_file_status"] == "insecure_permissions"
    assert status["connection_verified"] is False


def test_verification_receipt_is_invalidated_when_configuration_changes(
    tmp_path: Path,
) -> None:
    store = PiProviderConfigStore(
        config_path=tmp_path / "pi-provider.env",
        receipt_path=tmp_path / "receipt.json",
    )
    store.verify_and_save(
        provider="easyicu-local",
        api_key="test-private-key",
        base_url="http://127.0.0.1:8317/v1",
        model="gpt5.6 luna",
        api_transport="openai-completions",
        verifier=_models_ok,
    )

    changed = store.public_status(
        environ={
            "EASYICU_PI_PROVIDER": "easyicu-local",
            "EASYICU_PI_API_KEY": "different-key",
            "EASYICU_PI_BASE_URL": "http://127.0.0.1:8317/v1",
            "EASYICU_PI_MODEL": "gpt5.6 luna",
            "EASYICU_PI_API": "openai-completions",
        },
        include_file=False,
    )

    assert changed["credential_present"] is True
    assert changed["connection_verified"] is False


def test_rejected_service_address_never_reaches_verifier(tmp_path: Path) -> None:
    store = PiProviderConfigStore(
        config_path=tmp_path / "pi-provider.env",
        receipt_path=tmp_path / "receipt.json",
    )
    calls: list[str] = []

    with pytest.raises(PiCopilotError) as caught:
        store.verify_and_save(
            provider="easyicu-local",
            api_key="must-not-be-written",
            base_url="file:///tmp/provider",
            model="gpt5.6 luna",
            api_transport="openai-completions",
            verifier=lambda url, *_: (calls.append(url), (200, {}))[1],
        )

    assert caught.value.code == "pi_provider_base_url_rejected"
    assert calls == []
    assert not store.config_path.exists()
