"""Private, verified model-service configuration for Pi Copilot.

This owner is deliberately separate from the scientific-run provider adapter.
Pi credentials unlock only the conversational shell.  Secret values may cross
the local browser-to-FastAPI setup request once, but are never returned to the
browser, copied into session metadata, or exposed to a Pi tool result.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional

from easyicu.webserver.provider_url_security import (
    ProviderUrlSecurityError,
    validate_credential_endpoint,
)

from .contracts import PiCopilotError, utc_now

DEFAULT_PROVIDER = "easyicu-local"
DEFAULT_BASE_URL = "http://127.0.0.1:8317/v1"
DEFAULT_MODEL = "gpt5.6 luna"
DEFAULT_API_TRANSPORT = "openai-completions"
SUPPORTED_API_TRANSPORTS = frozenset(
    {
        "anthropic-messages",
        "google-generative-ai",
        "openai-completions",
        "openai-responses",
    }
)
_DEFAULT_CONFIG_PATH = Path.home() / ".easyicu" / "pi-provider.env"
_DEFAULT_RECEIPT_PATH = (
    Path.home() / ".easyicu" / "pi-provider-verification.json"
)
_CONFIG_KEYS = frozenset(
    {
        "EASYICU_PI_API_KEY",
        "EASYICU_PI_PROVIDER",
        "EASYICU_PI_MODEL",
        "EASYICU_PI_BASE_URL",
        "EASYICU_PI_API",
    }
)
_MAX_MODELS_RESPONSE_BYTES = 1024 * 1024

Verifier = Callable[[str, Mapping[str, str], float], tuple[int, Any]]


@dataclass(frozen=True)
class PiProviderConfig:
    """Validated private inputs consumed only by the Pi gateway."""

    provider: str
    api_key: str
    base_url: str
    model: str
    api_transport: str

    def as_environment(self) -> Dict[str, str]:
        return {
            "EASYICU_PI_PROVIDER": self.provider,
            "EASYICU_PI_API_KEY": self.api_key,
            "EASYICU_PI_BASE_URL": self.base_url,
            "EASYICU_PI_MODEL": self.model,
            "EASYICU_PI_API": self.api_transport,
        }

    def fingerprint(self) -> str:
        canonical = json.dumps(
            self.as_environment(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class PiProviderConfigStore:
    """Own the private credential file and matching verification receipt."""

    def __init__(
        self,
        *,
        config_path: Optional[Path] = None,
        receipt_path: Optional[Path] = None,
    ) -> None:
        self.config_path = Path(config_path or _DEFAULT_CONFIG_PATH)
        self.receipt_path = Path(receipt_path or _DEFAULT_RECEIPT_PATH)
        self._lock = threading.RLock()

    @staticmethod
    def _clean(value: Any, *, field: str, limit: int) -> str:
        text = str(value or "").strip()
        if not text:
            raise PiCopilotError(
                f"pi_provider_{field}_required",
                f"Enter a {field.replace('_', ' ')} before continuing.",
            )
        if len(text) > limit or "\n" in text or "\r" in text or "\0" in text:
            raise PiCopilotError(
                f"pi_provider_{field}_invalid",
                f"The {field.replace('_', ' ')} is not valid.",
            )
        return text

    @classmethod
    def make_config(
        cls,
        *,
        provider: str,
        api_key: str,
        base_url: str,
        model: str,
        api_transport: str,
    ) -> PiProviderConfig:
        provider_text = cls._clean(provider, field="provider", limit=80)
        api_key_text = cls._clean(api_key, field="api_key", limit=8192)
        base_url_text = cls._clean(base_url, field="base_url", limit=2048)
        model_text = cls._clean(model, field="model", limit=256)
        transport_text = cls._clean(
            api_transport,
            field="api_transport",
            limit=64,
        ).lower()
        if transport_text not in SUPPORTED_API_TRANSPORTS:
            raise PiCopilotError(
                "pi_provider_api_transport_unsupported",
                "Choose one of Pi's supported custom-provider API transports.",
            )
        try:
            validate_credential_endpoint(base_url_text)
        except ProviderUrlSecurityError as exc:
            raise PiCopilotError(
                "pi_provider_base_url_rejected",
                "The model-service address is not allowed by the local security policy.",
                details={
                    "reason": exc.reason
                },
            ) from exc
        return PiProviderConfig(
            provider=provider_text,
            api_key=api_key_text,
            base_url=base_url_text.rstrip("/"),
            model=model_text,
            api_transport=transport_text,
        )

    def verify_and_save(
        self,
        *,
        provider: str,
        api_key: str,
        base_url: str,
        model: str,
        api_transport: str,
        verifier: Optional[Verifier] = None,
    ) -> tuple[PiProviderConfig, Dict[str, Any]]:
        """Verify first, then atomically persist the credential and receipt."""

        config = self.make_config(
            provider=provider,
            api_key=api_key,
            base_url=base_url,
            model=model,
            api_transport=api_transport,
        )
        verification = verify_provider_connection(config, transport=verifier)
        with self._lock:
            self._write_config(config)
            self._write_receipt(config, verification)
        return config, self.public_status(environ=config.as_environment())

    def environment(
        self,
        *,
        environ: Optional[Mapping[str, str]] = None,
        include_file: bool = True,
    ) -> Dict[str, str]:
        """Return process configuration with secure file values as fallback."""

        source = os.environ if environ is None else environ
        file_values, _status = (
            self._read_config_file() if include_file else ({}, "not_loaded")
        )
        merged = dict(file_values)
        for key, value in source.items():
            if key in _CONFIG_KEYS:
                merged[key] = str(value)
        return merged

    def resolved_config(
        self,
        *,
        environ: Optional[Mapping[str, str]] = None,
        include_file: bool = True,
    ) -> Optional[PiProviderConfig]:
        values = self.environment(
            environ=environ,
            include_file=include_file,
        )
        api_key = str(values.get("EASYICU_PI_API_KEY") or "").strip()
        if not api_key:
            return None
        try:
            return self.make_config(
                provider=str(
                    values.get("EASYICU_PI_PROVIDER") or DEFAULT_PROVIDER
                ),
                api_key=api_key,
                base_url=str(
                    values.get("EASYICU_PI_BASE_URL") or DEFAULT_BASE_URL
                ),
                model=str(values.get("EASYICU_PI_MODEL") or DEFAULT_MODEL),
                api_transport=str(
                    values.get("EASYICU_PI_API") or DEFAULT_API_TRANSPORT
                ),
            )
        except PiCopilotError:
            return None

    def public_status(
        self,
        *,
        environ: Optional[Mapping[str, str]] = None,
        include_file: bool = True,
    ) -> Dict[str, Any]:
        """Return setup state and safe editable defaults, never a credential."""

        _file_values, file_status = self._read_config_file()
        values = self.environment(
            environ=environ,
            include_file=include_file,
        )
        config = self.resolved_config(
            environ=environ,
            include_file=include_file,
        )
        receipt = self._read_receipt()
        verified = bool(
            config
            and receipt
            and receipt.get("fingerprint") == config.fingerprint()
        )
        return {
            "provider": str(
                values.get("EASYICU_PI_PROVIDER") or DEFAULT_PROVIDER
            ),
            "base_url": str(
                values.get("EASYICU_PI_BASE_URL") or DEFAULT_BASE_URL
            ),
            "model": str(values.get("EASYICU_PI_MODEL") or DEFAULT_MODEL),
            "api_transport": str(
                values.get("EASYICU_PI_API") or DEFAULT_API_TRANSPORT
            ),
            "credential_present": bool(config),
            "connection_verified": verified,
            "verified_at": receipt.get("verified_at") if verified else None,
            "model_available": bool(
                verified and receipt.get("model_available") is True
            ),
            "config_file_status": file_status,
            "credential_storage": "private_local_file_0600",
            "secrets_returned": False,
        }

    def _read_config_file(self) -> tuple[Dict[str, str], str]:
        try:
            mode = self.config_path.stat().st_mode
        except FileNotFoundError:
            return {}, "missing"
        except OSError:
            return {}, "unreadable"
        if not self.config_path.is_file():
            return {}, "not_file"
        if mode & (stat.S_IRWXG | stat.S_IRWXO):
            return {}, "insecure_permissions"
        try:
            lines = self.config_path.read_text(encoding="utf-8").splitlines()
        except OSError:
            return {}, "unreadable"
        values: Dict[str, str] = {}
        for raw in lines:
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, encoded = line.split("=", 1)
            key = key.strip()
            if key not in _CONFIG_KEYS:
                continue
            try:
                value = json.loads(encoded)
            except json.JSONDecodeError:
                continue
            if isinstance(value, str):
                values[key] = value
        return values, "loaded"

    def _read_receipt(self) -> Dict[str, Any]:
        try:
            mode = self.receipt_path.stat().st_mode
            if mode & (stat.S_IRWXG | stat.S_IRWXO):
                return {}
            payload = json.loads(self.receipt_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, OSError, json.JSONDecodeError):
            return {}
        if not isinstance(payload, dict):
            return {}
        if payload.get("schema_version") != "easyicu.pi-provider-verification/1":
            return {}
        return payload

    def _write_config(self, config: PiProviderConfig) -> None:
        lines = [
            "# EasyICU private Pi Copilot model-service configuration",
            "# Created by the local first-use setup. Do not commit this file.",
        ]
        for key, value in config.as_environment().items():
            lines.append(f"{key}={json.dumps(value, ensure_ascii=False)}")
        self._atomic_private_write(
            self.config_path,
            "\n".join(lines) + "\n",
        )

    def _write_receipt(
        self,
        config: PiProviderConfig,
        verification: Mapping[str, Any],
    ) -> None:
        payload = {
            "schema_version": "easyicu.pi-provider-verification/1",
            "fingerprint": config.fingerprint(),
            "verified_at": utc_now(),
            "model_available": bool(verification.get("model_available")),
        }
        self._atomic_private_write(
            self.receipt_path,
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        )

    @staticmethod
    def _atomic_private_write(path: Path, content: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        handle = tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=str(path.parent),
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        )
        temporary = Path(handle.name)
        try:
            with handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            temporary.chmod(0o600)
            temporary.replace(path)
            path.chmod(0o600)
        finally:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass


def _default_models_transport(
    url: str,
    headers: Mapping[str, str],
    timeout: float,
) -> tuple[int, Any]:
    import requests

    try:
        response = requests.get(
            url,
            headers=dict(headers),
            timeout=timeout,
            allow_redirects=False,
            stream=True,
        )
        try:
            if response.is_redirect or response.is_permanent_redirect:
                return response.status_code, {"redirect_refused": True}
            chunks = []
            total = 0
            for chunk in response.iter_content(chunk_size=32 * 1024):
                total += len(chunk)
                if total > _MAX_MODELS_RESPONSE_BYTES:
                    raise PiCopilotError(
                        "pi_provider_response_too_large",
                        "The model-service verification response was too large.",
                    )
                chunks.append(chunk)
            raw = b"".join(chunks)
        finally:
            response.close()
    except PiCopilotError:
        raise
    except requests.RequestException as exc:
        raise PiCopilotError(
            "pi_provider_connection_failed",
            "EasyICU could not reach the model service.",
        ) from exc
    try:
        payload = json.loads(raw.decode("utf-8")) if raw else {}
    except (UnicodeDecodeError, json.JSONDecodeError):
        payload = {}
    return response.status_code, payload


def verify_provider_connection(
    config: PiProviderConfig,
    *,
    transport: Optional[Verifier] = None,
    timeout: float = 10.0,
) -> Dict[str, Any]:
    """Verify authentication and exact model availability via ``/models``.

    Pi supports many provider brands, but custom providers converge on four
    wire protocols.  Keep discovery protocol-aware here so a native Anthropic
    or Google endpoint is not incorrectly treated as OpenAI-compatible.
    """

    # Validate immediately before the request as well as during input parsing.
    try:
        validate_credential_endpoint(config.base_url)
    except ProviderUrlSecurityError as exc:
        raise PiCopilotError(
            "pi_provider_base_url_rejected",
            "The model-service address is not allowed by the local security policy.",
        ) from exc
    url = f"{config.base_url.rstrip('/')}/models"
    headers = {"Accept": "application/json"}
    if config.api_transport == "anthropic-messages":
        headers.update(
            {
                "x-api-key": config.api_key,
                "anthropic-version": "2023-06-01",
            }
        )
    elif config.api_transport == "google-generative-ai":
        headers["x-goog-api-key"] = config.api_key
    else:
        headers["Authorization"] = f"Bearer {config.api_key}"
    status_code, payload = (transport or _default_models_transport)(
        url,
        headers,
        timeout,
    )
    if status_code in {401, 403}:
        raise PiCopilotError(
            "pi_provider_auth_failed",
            "The model service rejected this API credential.",
        )
    if status_code < 200 or status_code >= 300:
        raise PiCopilotError(
            "pi_provider_verification_failed",
            "The model service did not accept the verification request.",
            details={"status_code": int(status_code)},
        )
    if config.api_transport == "google-generative-ai":
        rows = payload.get("models") if isinstance(payload, Mapping) else None
        identifier_key = "name"
    else:
        rows = payload.get("data") if isinstance(payload, Mapping) else None
        identifier_key = "id"
    if not isinstance(rows, list):
        raise PiCopilotError(
            "pi_provider_models_response_invalid",
            "The model service returned an invalid model list.",
        )
    identifiers = {
        identifier
        for row in rows
        if isinstance(row, Mapping) and row.get(identifier_key)
        for identifier in [str(row.get(identifier_key) or "").strip()]
        if 0 < len(identifier) <= 256
        and "\n" not in identifier
        and "\r" not in identifier
        and "\0" not in identifier
        and config.api_key not in identifier
    }
    if config.api_transport == "google-generative-ai":
        identifiers = {
            identifier.removeprefix("models/") for identifier in identifiers
        }
    if config.model not in identifiers:
        raise PiCopilotError(
            "pi_provider_model_unavailable",
            "The selected model was not reported by this model service.",
            details={
                "available_models": sorted(identifiers)[:100],
                "models_reported": len(identifiers),
            },
        )
    return {
        "connection_verified": True,
        "model_available": True,
        "models_reported": len(identifiers),
        "secrets_returned": False,
    }


__all__ = [
    "DEFAULT_API_TRANSPORT",
    "DEFAULT_BASE_URL",
    "DEFAULT_MODEL",
    "DEFAULT_PROVIDER",
    "PiProviderConfig",
    "PiProviderConfigStore",
    "SUPPORTED_API_TRANSPORTS",
    "verify_provider_connection",
]
