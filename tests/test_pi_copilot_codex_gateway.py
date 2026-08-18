"""Focused contracts for the browser-account to Pi gateway owner."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from easyicu.webserver import codex_account_sessions
from easyicu.webserver.pi_copilot.codex_gateway import CodexPiGatewayPool
from easyicu.webserver.pi_copilot.contracts import (
    PiCopilotError,
    ResearchProviderBinding,
)


class _TemplateGateway:
    def __init__(self, root: Path) -> None:
        self.app_dir = root / "app"
        self.declared_session_dir = root / "sessions"
        self.declared_cwd = root / "workspace"


class _AccountGateway:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.closed = False

    def close(self) -> None:
        self.closed = True


def _binding() -> ResearchProviderBinding:
    return ResearchProviderBinding(
        provider="codex",
        credential_source="codex_user_auth",
        authentication_mode="chatgpt_account",
        model="gpt-5.6-luna",
        account_session_sha256="a" * 64,
    )


def test_pool_compiles_one_secret_free_account_gateway_and_reuses_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    auth_file = tmp_path / "codex" / "auth.json"
    auth_file.parent.mkdir()
    auth_file.write_text("{}", encoding="utf-8")
    auth_file.chmod(0o600)
    projections: list[tuple[str, str]] = []

    def project(binding_sha256: str, *, model: str) -> dict[str, str]:
        projections.append((binding_sha256, model))
        return {
            "EASYICU_PI_PROVIDER": "openai-codex",
            "EASYICU_PI_API": "openai-codex-responses",
            "EASYICU_PI_MODEL": model,
            "EASYICU_PI_CODEX_AUTH_FILE": str(auth_file),
            "EASYICU_PI_CODEX_SESSION_SHA256": binding_sha256,
            "EASYICU_PI_API_KEY": "ambient-key-must-not-cross",
        }

    monkeypatch.setattr(
        codex_account_sessions,
        "pi_conversation_environment_for_binding",
        project,
    )
    created: list[_AccountGateway] = []

    def factory(**kwargs: Any) -> _AccountGateway:
        gateway = _AccountGateway(**kwargs)
        created.append(gateway)
        return gateway

    pool = CodexPiGatewayPool(
        template_gateway=_TemplateGateway(tmp_path),
        gateway_factory=factory,  # type: ignore[arg-type]
    )

    first = pool.gateway_for(_binding(), refresh_account=True)
    repeated = pool.gateway_for(_binding())
    refreshed = pool.gateway_for(_binding(), refresh_account=True)

    assert first is repeated is refreshed
    assert len(created) == 1
    assert projections == [
        ("a" * 64, "gpt-5.6-luna"),
        ("a" * 64, "gpt-5.6-luna"),
    ]
    assert "EASYICU_PI_API_KEY" not in created[0].kwargs["environ"]
    assert created[0].kwargs["account_binding_sha256"] == "a" * 64
    assert created[0].kwargs["app_dir"] == tmp_path / "app"

    pool.close()
    assert created[0].closed is True


def test_pool_preserves_the_account_owner_error_code(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def blocked(*_args: Any, **_kwargs: Any) -> dict[str, str]:
        raise codex_account_sessions.CodexAccountSessionError(
            "codex_auth_login_required"
        )

    monkeypatch.setattr(
        codex_account_sessions,
        "pi_conversation_environment_for_binding",
        blocked,
    )
    pool = CodexPiGatewayPool(template_gateway=_TemplateGateway(tmp_path))

    with pytest.raises(PiCopilotError) as failure:
        pool.gateway_for(_binding(), refresh_account=True)

    assert failure.value.code == "codex_auth_login_required"
    assert failure.value.status_code == 409
