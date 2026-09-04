"""Browser-bound Codex account projection for Pi Copilot conversations.

This owner compiles one immutable account/model binding into an isolated Pi
gateway.  Login, refresh, and credential-file validation remain owned by
``codex_account_sessions``; Pi receives no browser cookie and this pool never
returns or persists a credential value.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any

from easyicu.webserver import codex_account_sessions

from .contracts import PiCopilotError, ResearchProviderBinding
from .gateway import PiGatewayClient


GatewayFactory = Callable[..., PiGatewayClient]


class CodexPiGatewayPool:
    """Reuse one Pi sidecar per immutable browser-account/model coordinate."""

    def __init__(
        self,
        *,
        template_gateway: Any,
        gateway_factory: GatewayFactory = PiGatewayClient,
    ) -> None:
        self._template_gateway = template_gateway
        self._gateway_factory = gateway_factory
        self._gateways: dict[tuple[str, str], PiGatewayClient] = {}
        self._lock = threading.RLock()

    @staticmethod
    def _coordinates(
        binding: ResearchProviderBinding,
    ) -> tuple[str, str]:
        if binding.provider != "codex" or not binding.account_session_sha256:
            raise PiCopilotError(
                "pi_codex_conversation_binding_required",
                "A verified Codex account binding is required for this conversation.",
                status_code=409,
            )
        return binding.account_session_sha256, binding.model

    def gateway_for(
        self,
        binding: ResearchProviderBinding,
        *,
        refresh_account: bool = False,
    ) -> PiGatewayClient:
        account_sha256, model = self._coordinates(binding)
        key = (account_sha256, model)
        with self._lock:
            existing = self._gateways.get(key)
        if existing is not None and not refresh_account:
            return existing
        try:
            environment = (
                codex_account_sessions.pi_conversation_environment_for_binding(
                    account_sha256,
                    model=model,
                )
            )
        except codex_account_sessions.CodexAccountSessionError as exc:
            raise PiCopilotError(
                exc.code,
                "The browser-bound Codex account is not ready for this conversation.",
                status_code=409,
            ) from exc
        # Never let an ambient or previously configured API key coexist with
        # this account authority in the child process.
        environment.pop("EASYICU_PI_API_KEY", None)
        with self._lock:
            existing = self._gateways.get(key)
            if existing is not None:
                return existing
            kwargs: dict[str, Any] = {
                "environ": environment,
                "account_binding_sha256": account_sha256,
            }
            for name, public_name in (
                ("app_dir", "app_dir"),
                ("declared_session_dir", "session_dir"),
                ("declared_cwd", "cwd"),
            ):
                value = getattr(self._template_gateway, name, None)
                if value is not None:
                    kwargs[public_name] = value
            gateway = self._gateway_factory(**kwargs)
            self._gateways[key] = gateway
            return gateway

    def close(self) -> None:
        with self._lock:
            gateways = list(self._gateways.values())
            self._gateways.clear()
        for gateway in gateways:
            gateway.close()

    def memory_statuses(self) -> list[dict[str, Any]]:
        with self._lock:
            gateways = list(self._gateways.values())
        return [gateway.memory_status() for gateway in gateways]

    def maintain_sessions(self, *, exclude_session_id: str = "") -> None:
        with self._lock:
            gateways = list(self._gateways.values())
        for gateway in gateways:
            gateway.maintain_sessions(exclude_session_id=exclude_session_id)


__all__ = ["CodexPiGatewayPool"]
