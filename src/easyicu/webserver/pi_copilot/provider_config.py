"""Private, verified model-service configuration for Pi Copilot.

This owner is deliberately separate from the scientific-run provider adapter.
Pi credentials unlock only the conversational shell by default.  A separately
authorized full scientific run may request a verified in-memory projection for
the Research Agent provider; that projection is never persisted to the project,
returned to the browser, copied into session metadata, or exposed to a Pi tool
result.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
import tempfile
import threading
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Dict, Literal, Mapping, Optional

from pydantic import BaseModel, ConfigDict

from easyicu.research_agent.providers.client_trust import (
    DataHandling, InferenceLocation, PROVIDER_POLICY_NOTICE_KEY,
)
from easyicu.provider_auth import (
    OPENAI_AUTH_HEADER_ENV,
    OpenAIAuthHeader,
    credential_headers,
    pi_openai_auth_header,
)
from easyicu.webserver import hosted, state_paths
from easyicu.webserver.provider_url_security import (
    ProviderUrlSecurityError,
    validate_credential_endpoint,
)

from .contracts import PiCopilotError, utc_now

DEFAULT_PROVIDER = "easyicu-local"
DEFAULT_BASE_URL = "http://127.0.0.1:8317/v1"
DEFAULT_MODEL = "gpt-5.6-luna"
DEFAULT_API_TRANSPORT = "openai-completions"
# Recommendation defaults from the connection form's provider presets. A model
# is recommended only when the endpoint actually reports this identifier.
PROVIDER_DEFAULT_MODELS = {
    DEFAULT_PROVIDER: DEFAULT_MODEL,
    "openai": DEFAULT_MODEL,
    "deepseek": "deepseek-chat",
    "anthropic": "claude-sonnet-4-6",
    "google": "gemini-3.5-flash",
}
SUPPORTED_API_TRANSPORTS = frozenset(
    {
        "anthropic-messages",
        "google-generative-ai",
        "openai-completions",
        "openai-responses",
    }
)
_DEFAULT_CONFIG_PATH = state_paths.state_root() / "pi-provider.env"
_DEFAULT_RECEIPT_PATH = state_paths.state_root() / "pi-provider-verification.json"
_CONFIG_KEYS = frozenset(
    {
        "EASYICU_PI_API_KEY",
        "EASYICU_PI_PROVIDER",
        "EASYICU_PI_MODEL",
        "EASYICU_PI_BASE_URL",
        "EASYICU_PI_API",
        "EASYICU_PI_DATA_HANDLING",
        "EASYICU_PI_INFERENCE_LOCATION",
    }
)
_MAX_VERIFICATION_RESPONSE_BYTES = 1024 * 1024
DECLARATION_NOTICE_VERSION = "1"
_POLICY_ENV_KEYS = frozenset({"EASYICU_PI_DATA_HANDLING", "EASYICU_PI_INFERENCE_LOCATION"})
_DECLARATION_MESSAGES = {
    "settings_only": "The data-handling declaration can be changed only in the model settings.",
    "admin_required": (
        "On a shared server, only the administrator can set the data-handling declaration, "
        "in the server configuration."
    ),
    "notice_mismatch": "The declaration notice changed after this page loaded. Read it again, then save.",
    "value_invalid": "Choose one of the listed options.",
    "administrator_configuration": (
        "The administrator set this declaration in the configuration, so it cannot be changed here."
    ),
    "connection_required": "Connect a model first, then declare how it handles data.",
    "connection_changed": (
        "The model connection changed while this form was open. Check the current connection, then save again."
    ),
    "record_unavailable": (
        "EasyICU could not read the declaration record, so this model is treated as undeclared. "
        "Check that the EasyICU data folder is readable, then try again."
    ),
    "busy": "A conversation turn is running. Change the declaration after it finishes.",
    "gateway_unavailable": (
        "EasyICU could not apply the change to the model connection. "
        "Try again; if it fails again, restart EasyICU."
    ),
}


def declaration_error(reason: str, status_code: int = 403) -> PiCopilotError:
    code = f"pi_provider_declaration_{reason}"
    return PiCopilotError(code, _DECLARATION_MESSAGES[reason], status_code=status_code)


class PiProviderConnection(BaseModel):
    """The exact connection whose notice the settings page displayed."""

    model_config = ConfigDict(extra="forbid", strict=True)
    provider: str
    base_url: str
    model: str


class PiProviderDeclarationRecord(BaseModel):
    """Append-only snapshots; a lapse preserves the original human receipt."""

    model_config = ConfigDict(extra="forbid")
    schema_version: Literal["easyicu.pi-provider-declaration/1"] = "easyicu.pi-provider-declaration/1"
    actor: dict[str, str]
    recorded_at: str
    connection: PiProviderConnection
    value: DataHandling
    notice_key: Literal["physionet_llm_data_handling_declaration"]
    notice_version: Literal["1"]
    lapsed_at: str | None = None


class PiProviderDeclarationState(BaseModel):
    """Settings read model; no credentials or model-controlled authority."""

    model_config = ConfigDict(extra="forbid")
    connection: PiProviderConnection | None
    value: DataHandling
    recorded_at: str | None
    actor: dict[str, str] | None
    lapsed: bool
    may_change: bool
    change_authority: Literal["local_user", "administrator_configuration"]
    source: Literal["settings_ui", "administrator_configuration", "unspecified"]
    status_code: str
    notice_key: str = PROVIDER_POLICY_NOTICE_KEY
    notice_version: str = DECLARATION_NOTICE_VERSION

Verifier = Callable[
    [str, str, Mapping[str, str], Optional[Mapping[str, Any]], float],
    tuple[int, Any],
]


@dataclass(frozen=True)
class PiProviderConfig:
    """Validated private inputs consumed only by the Pi gateway."""

    provider: str
    api_key: str
    base_url: str
    model: str
    api_transport: str
    data_handling: DataHandling = DataHandling.UNSPECIFIED
    inference_location: InferenceLocation = InferenceLocation.UNSPECIFIED

    def __post_init__(self) -> None:
        handling = DataHandling(self.data_handling)
        location = InferenceLocation(self.inference_location)
        if handling is DataHandling.ON_DEVICE or location is InferenceLocation.ON_DEVICE:
            handling = DataHandling.ON_DEVICE
            location = InferenceLocation.ON_DEVICE
        object.__setattr__(self, "data_handling", handling)
        object.__setattr__(self, "inference_location", location)

    @property
    def openai_auth_header(self) -> OpenAIAuthHeader:
        """Return the non-secret wire-auth mode compiled from this preset."""

        return pi_openai_auth_header(
            provider=self.provider,
            api_transport=self.api_transport,
        )

    def as_environment(self) -> Dict[str, str]:
        return {
            "EASYICU_PI_PROVIDER": self.provider,
            "EASYICU_PI_API_KEY": self.api_key,
            "EASYICU_PI_BASE_URL": self.base_url,
            "EASYICU_PI_MODEL": self.model,
            "EASYICU_PI_API": self.api_transport,
            "EASYICU_PI_DATA_HANDLING": self.data_handling.value,
            "EASYICU_PI_INFERENCE_LOCATION": self.inference_location.value,
        }

    def fingerprint(self) -> str:
        environment = self.as_environment()
        if (
            self.data_handling is DataHandling.UNSPECIFIED
            and self.inference_location is InferenceLocation.UNSPECIFIED
        ):
            # An undeclared connection keeps the fingerprint it had before the
            # declaration keys existed, so its verification receipt stays valid.
            # Any declaration changes the fingerprint and needs a new verification.
            environment.pop("EASYICU_PI_DATA_HANDLING")
            environment.pop("EASYICU_PI_INFERENCE_LOCATION")
        canonical = json.dumps(
            {
                "schema": "easyicu.pi-provider-config-fingerprint/2",
                "environment": environment,
                "openai_auth_header": self.openai_auth_header.value,
                "verification_contract": "model-catalog+inference-probe/1",
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def connection(self) -> PiProviderConnection:
        return PiProviderConnection(provider=self.provider, base_url=self.base_url, model=self.model)


class PiProviderConfigStore:
    """Own the private credential file and matching verification receipt."""

    def __init__(
        self,
        *,
        config_path: Optional[Path] = None,
        receipt_path: Optional[Path] = None,
        declaration_path: Optional[Path] = None,
    ) -> None:
        self.config_path = Path(config_path or _DEFAULT_CONFIG_PATH)
        self.receipt_path = Path(receipt_path or _DEFAULT_RECEIPT_PATH)
        self.declaration_path = Path(declaration_path) if declaration_path else (
            self.receipt_path.with_name(self.receipt_path.stem + "-declarations.jsonl")
        )
        self._lock = threading.RLock()

    def _declaration_record(self, connection: PiProviderConnection | None) -> PiProviderDeclarationRecord | None:
        """Persist an observed lapse so returning to an old connection cannot revive it."""
        with self._lock:
            record = None
            try:
                with self.declaration_path.open(encoding="utf-8") as stream:
                    if os.fstat(stream.fileno()).st_mode & (stat.S_IRWXG | stat.S_IRWXO):
                        raise declaration_error("record_unavailable", 409)
                    for line in stream:
                        record = PiProviderDeclarationRecord.model_validate_json(line)
            except FileNotFoundError:
                return None
            except (OSError, ValueError) as exc:
                raise declaration_error("record_unavailable", 409) from exc
            if record and record.lapsed_at is None and record.connection != connection:
                record = record.model_copy(update={"lapsed_at": utc_now()})
                self._append_declaration(record)
            return record

    def _append_declaration(self, record: PiProviderDeclarationRecord) -> None:
        try:
            self.declaration_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            flags = os.O_WRONLY | os.O_APPEND | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
            with os.fdopen(os.open(self.declaration_path, flags, 0o600), "a", encoding="utf-8") as stream:
                mode = os.fstat(stream.fileno()).st_mode
                if not stat.S_ISREG(mode) or mode & (stat.S_IRWXG | stat.S_IRWXO):
                    raise declaration_error("record_unavailable", 409)
                stream.write(record.model_dump_json() + "\n")
                stream.flush()
                os.fsync(stream.fileno())
        except OSError as exc:
            raise declaration_error("record_unavailable", 409) from exc

    def _administrator_policy(self, *, environ=None, include_file=True) -> bool:
        """Precedence: process policy variables (even unspecified) win as a pair.

        Non-default policy saved by the administrator's legacy config writer
        also wins. Both are reported as administrator_configuration (lead copy:
        "set by the administrator's configuration"). Otherwise only the latest
        matching, unlapsed settings record applies. No endpoint/provider/port
        heuristics declare on-device inference. UI writes never edit env policy.
        """
        source = os.environ if environ is None else environ
        if _POLICY_ENV_KEYS.intersection(source):
            return True
        values = self.environment(environ=environ, include_file=include_file)
        return any(values.get(key, "unspecified") != "unspecified" for key in _POLICY_ENV_KEYS)

    def declaration_status(self, *, environ=None, include_file=True) -> Dict[str, Any]:
        with self._lock:
            config = self._configured_connection(environ=environ, include_file=include_file)
            connection = config.connection() if config else None
            record = self._declaration_record(connection)
            administrator = self._administrator_policy(environ=environ, include_file=include_file)
            lapsed = bool(record and record.lapsed_at)
            value = (config.data_handling if config else DataHandling.UNSPECIFIED) if administrator else (
                record.value if record and not lapsed else DataHandling.UNSPECIFIED
            )
            source = "administrator_configuration" if administrator else "settings_ui" if record else "unspecified"
            reason = "administrator_configuration" if administrator else "admin_required" if hosted.is_hosted() else (
                "connection_required" if config is None else "lapsed" if lapsed else "recorded" if record else "unspecified"
            )
            return PiProviderDeclarationState(
                connection=connection, value=value, lapsed=lapsed,
                recorded_at=record.recorded_at if record and not administrator else None,
                actor=record.actor if record and not administrator else None,
                may_change=bool(config and not administrator and not hosted.is_hosted()),
                change_authority="administrator_configuration" if administrator or hosted.is_hosted() else "local_user",
                source=source, status_code=f"pi_provider_declaration_{reason}",
            ).model_dump(mode="json")

    def declare_data_handling(
        self, *, value: str, connection: Mapping[str, str], notice_key: str,
        notice_version: str, channel: str,
    ) -> Dict[str, Any]:
        """Only the settings adapter calls this; never expose it as a Pi tool."""
        if channel != "settings_ui":
            raise declaration_error("settings_only")
        if hosted.is_hosted():
            # The existing hosted guard grants no browser account admin rights.
            raise declaration_error("admin_required")
        if notice_key != PROVIDER_POLICY_NOTICE_KEY or notice_version != DECLARATION_NOTICE_VERSION:
            raise declaration_error("notice_mismatch", 409)
        try:
            declared = DataHandling(value)
        except ValueError as exc:
            raise declaration_error("value_invalid", 422) from exc
        with self._lock:
            if self._administrator_policy():
                raise declaration_error("administrator_configuration", 409)
            config = self._configured_connection()
            if config is None:
                raise declaration_error("connection_required", 409)
            if dict(connection) != config.connection().model_dump():
                raise declaration_error("connection_changed", 409)
            # Validate the existing journal before appending, even on a clear.
            self._declaration_record(config.connection())
            self._append_declaration(PiProviderDeclarationRecord(
                actor={"kind": "local_user", "id": "local_user"}, recorded_at=utc_now(),
                connection=config.connection(), value=declared,
                notice_key=notice_key, notice_version=notice_version,
            ))
            return self.declaration_status()

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
        data_handling: DataHandling = DataHandling.UNSPECIFIED,
        inference_location: InferenceLocation = InferenceLocation.UNSPECIFIED,
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
                details={"reason": exc.reason},
            ) from exc
        return PiProviderConfig(
            provider=provider_text,
            api_key=api_key_text,
            base_url=base_url_text.rstrip("/"),
            model=model_text,
            api_transport=transport_text,
            data_handling=data_handling,
            inference_location=inference_location,
        )

    def verify_and_save(
        self,
        *,
        provider: str,
        api_key: str,
        base_url: str,
        model: str,
        api_transport: str,
        data_handling: DataHandling = DataHandling.UNSPECIFIED,
        inference_location: InferenceLocation = InferenceLocation.UNSPECIFIED,
        verifier: Optional[Verifier] = None,
        environ: Optional[Mapping[str, str]] = None,
    ) -> tuple[PiProviderConfig, Dict[str, Any]]:
        """Verify first, then atomically persist the credential and receipt."""

        config = self.make_config(
            provider=provider,
            api_key=api_key,
            base_url=base_url,
            model=model,
            api_transport=api_transport,
            data_handling=data_handling,
            inference_location=inference_location,
        )
        verification = verify_provider_connection(config, transport=verifier)
        source = os.environ if environ is None else environ
        policy = {key: source[key] for key in _POLICY_ENV_KEYS if key in source}
        with self._lock:
            self._declaration_record(config.connection())
            self._write_config(config)
            # Verify the effective policy without copying a human declaration
            # into administrator configuration. A new choice needs a new receipt.
            effective = self._require_stored_config(environ=policy)
            self._write_receipt(effective, verification)
            return effective, self.public_status(environ=policy)

    def _require_stored_config(
        self,
        *,
        environ: Optional[Mapping[str, str]] = None,
    ) -> PiProviderConfig:
        stored = self.resolved_config(environ=environ)
        if stored is None:
            raise PiCopilotError(
                "pi_provider_connection_not_configured",
                "Connect a model service before choosing a model.",
            )
        return stored

    def discover_models(
        self,
        *,
        verifier: Optional[Verifier] = None,
    ) -> list[str]:
        """List the models the stored connection reports, without saving."""

        return discover_provider_models(
            self._require_stored_config(),
            transport=verifier,
        )

    def switch_model(
        self,
        model: str,
        *,
        verifier: Optional[Verifier] = None,
        environ: Optional[Mapping[str, str]] = None,
    ) -> tuple[PiProviderConfig, Dict[str, Any]]:
        """Re-verify the stored connection on one different model.

        The browser never receives the credential, so a model change has to
        reuse the private stored config instead of round-tripping a key
        through the UI.  Delegating to ``verify_and_save`` keeps the endpoint
        security check, the catalog membership rule and the verification
        receipt on exactly the same path a first-time connection takes.
        An administrator tool passes ``environ={}`` so its own shell cannot
        override the instance's stored connection.
        """

        # A human choice belongs to the old model; never persist its effective
        # value as administrator policy on the replacement connection.
        stored = self._configured_connection(environ=environ)
        if stored is None:
            self._require_stored_config(environ=environ)
        return self.verify_and_save(
            provider=stored.provider,
            api_key=stored.api_key,
            base_url=stored.base_url,
            model=model,
            api_transport=stored.api_transport,
            data_handling=stored.data_handling,
            inference_location=stored.inference_location,
            verifier=verifier,
            environ=environ,
        )

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
        if _POLICY_ENV_KEYS.intersection(source):
            # Do not let an inherited inference_location undo an explicit env
            # clear of data_handling (or the converse).
            for key in _POLICY_ENV_KEYS:
                merged.pop(key, None)
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
        with self._lock:
            config = self._configured_connection(environ=environ, include_file=include_file)
            record = self._declaration_record(config.connection() if config else None)
            if config and record and not record.lapsed_at and not self._administrator_policy(
                environ=environ, include_file=include_file,
            ):
                config = replace(config, data_handling=record.value,
                                 inference_location=InferenceLocation.UNSPECIFIED)
            return config

    def _configured_connection(
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
                provider=str(values.get("EASYICU_PI_PROVIDER") or DEFAULT_PROVIDER),
                api_key=api_key,
                base_url=str(values.get("EASYICU_PI_BASE_URL") or DEFAULT_BASE_URL),
                model=str(values.get("EASYICU_PI_MODEL") or DEFAULT_MODEL),
                api_transport=str(
                    values.get("EASYICU_PI_API") or DEFAULT_API_TRANSPORT
                ),
                data_handling=DataHandling(values.get("EASYICU_PI_DATA_HANDLING", "unspecified")),
                inference_location=InferenceLocation(values.get("EASYICU_PI_INFERENCE_LOCATION", "unspecified")),
            )
        except (PiCopilotError, ValueError):
            return None

    def research_agent_environment(
        self,
        *,
        external_llm_opt_in: bool,
        environ: Optional[Mapping[str, str]] = None,
        include_file: bool = True,
    ) -> Dict[str, str]:
        """Project one verified OpenAI-compatible config into run memory.

        This method does not itself authorize a scientific run.  The caller
        must already hold the distinct, one-turn full-analysis grant and pass
        its explicit external-model opt-in.  The returned mapping is consumed
        only by the provider factory and disables the separate legacy provider
        env file so one run cannot silently mix two credential authorities.
        """

        if not external_llm_opt_in:
            raise PiCopilotError(
                "pi_provider_research_opt_in_required",
                "A full Research Agent run requires explicit external-model authorization.",
            )
        config = self.resolved_config(
            environ=environ,
            include_file=include_file,
        )
        status = self.public_status(
            environ=environ,
            include_file=include_file,
        )
        if config is None or not (
            status.get("connection_verified") and status.get("model_available")
        ):
            raise PiCopilotError(
                "pi_provider_research_configuration_unverified",
                "Verify the Pi model service before using it for a full Research Agent run.",
            )
        if config.api_transport != "openai-completions":
            raise PiCopilotError(
                "pi_provider_research_transport_unsupported",
                "The Research Agent currently requires an OpenAI Chat Completions compatible Pi provider.",
                details={"api_transport": config.api_transport},
            )
        if not status.get("inference_verified"):
            raise PiCopilotError(
                "pi_provider_research_inference_unverified",
                "Verify one bounded model response before using this service for a full Research Agent run.",
            )
        return {
            "OPENAI_API_KEY": config.api_key,
            "OPENAI_BASE_URL": config.base_url,
            "OPENAI_MODEL": config.model,
            OPENAI_AUTH_HEADER_ENV: config.openai_auth_header.value,
            "EASYICU_DISABLE_PROVIDER_ENV_FILE": "1",
            "EASYICU_PROVIDER_DATA_HANDLING": config.data_handling.value,
            "EASYICU_PROVIDER_INFERENCE_LOCATION": config.inference_location.value,
        }

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
            config and receipt and receipt.get("fingerprint") == config.fingerprint()
        )
        return {
            "provider": str(values.get("EASYICU_PI_PROVIDER") or DEFAULT_PROVIDER),
            "base_url": str(values.get("EASYICU_PI_BASE_URL") or DEFAULT_BASE_URL),
            "model": str(values.get("EASYICU_PI_MODEL") or DEFAULT_MODEL),
            "api_transport": str(values.get("EASYICU_PI_API") or DEFAULT_API_TRANSPORT),
            "data_handling": config.data_handling.value if config else "unspecified",
            "inference_location": config.inference_location.value if config else "unspecified",
            "data_handling_notice_key": PROVIDER_POLICY_NOTICE_KEY,
            "credential_present": bool(config),
            "connection_verified": verified,
            "verified_at": receipt.get("verified_at") if verified else None,
            "model_available": bool(
                verified and receipt.get("model_available") is True
            ),
            "inference_verified": bool(
                verified and receipt.get("inference_verified") is True
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
        if payload.get("schema_version") != "easyicu.pi-provider-verification/2":
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
            "schema_version": "easyicu.pi-provider-verification/2",
            "fingerprint": config.fingerprint(),
            "verified_at": utc_now(),
            "model_available": bool(verification.get("model_available")),
            "inference_verified": bool(verification.get("inference_verified")),
            "verification_contract": "model-catalog+inference-probe/1",
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


def _default_verification_transport(
    method: str,
    url: str,
    headers: Mapping[str, str],
    body: Optional[Mapping[str, Any]],
    timeout: float,
) -> tuple[int, Any]:
    import requests

    try:
        response = requests.request(
            method,
            url,
            headers=dict(headers),
            json=dict(body) if body is not None else None,
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
                if total > _MAX_VERIFICATION_RESPONSE_BYTES:
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


def _discovery_request(config: PiProviderConfig) -> tuple[str, Dict[str, str]]:
    """Build the authenticated model-catalog request for this wire protocol.

    Pi supports many provider brands, but custom providers converge on four
    wire protocols.  Keep discovery protocol-aware here so a native Anthropic
    or Google endpoint is not incorrectly treated as OpenAI-compatible.
    """

    headers: Dict[str, str] = {"Accept": "application/json"}
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
        headers.update(
            credential_headers(
                config.api_key,
                mode=config.openai_auth_header,
            )
        )
    return f"{config.base_url.rstrip('/')}/models", headers


def _request_model_identities(
    config: PiProviderConfig,
    *,
    verifier: Verifier,
    timeout: float,
) -> list[str]:
    """Return the model identifiers this endpoint reports, or fail closed."""

    # Validate immediately before the request as well as during input parsing.
    try:
        validate_credential_endpoint(config.base_url)
    except ProviderUrlSecurityError as exc:
        raise PiCopilotError(
            "pi_provider_base_url_rejected",
            "The model-service address is not allowed by the local security policy.",
        ) from exc
    url, headers = _discovery_request(config)
    status_code, payload = verifier(
        "GET",
        url,
        headers,
        None,
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
        identifiers = {identifier.removeprefix("models/") for identifier in identifiers}
    return sorted(identifiers)


def discover_provider_models(
    config: PiProviderConfig,
    *,
    transport: Optional[Verifier] = None,
    timeout: float = 10.0,
) -> list[str]:
    """List the models a configured endpoint reports, without requiring a save.

    Verification needs the selected model to appear in this catalog; a browser
    choosing the next model needs only the catalog itself, so it must not have
    to fail a save first to learn what is available.
    """

    return _request_model_identities(
        config,
        verifier=transport or _default_verification_transport,
        timeout=timeout,
    )


def verify_provider_connection(
    config: PiProviderConfig,
    *,
    transport: Optional[Verifier] = None,
    timeout: float = 10.0,
) -> Dict[str, Any]:
    """Verify model discovery and one bounded inference on the selected wire.

    Pi supports many provider brands, but custom providers converge on four
    wire protocols.  Keep discovery protocol-aware here so a native Anthropic
    or Google endpoint is not incorrectly treated as OpenAI-compatible.
    """

    verifier = transport or _default_verification_transport
    identifiers = _request_model_identities(
        config,
        verifier=verifier,
        timeout=timeout,
    )
    # The bounded inference probe below authenticates the same way discovery
    # does, so rebuild the wire headers rather than carry them out of the
    # catalog request.
    _, headers = _discovery_request(config)
    if config.model not in identifiers:
        raise PiCopilotError(
            "pi_provider_model_unavailable",
            "The selected model was not reported by this model service.",
            details={
                "available_models": sorted(identifiers)[:100],
                "models_reported": len(identifiers),
            },
        )
    inference_verified = False
    if config.api_transport == "openai-completions":
        inference_url = f"{config.base_url.rstrip('/')}/chat/completions"
        inference_headers = {
            **headers,
            "Content-Type": "application/json",
        }
        inference_status, inference_payload = verifier(
            "POST",
            inference_url,
            inference_headers,
            {
                "model": config.model,
                "messages": [
                    {
                        "role": "user",
                        "content": "Reply with READY.",
                    }
                ],
                "max_tokens": 8,
                "stream": False,
            },
            timeout,
        )
        if inference_status in {401, 403}:
            raise PiCopilotError(
                "pi_provider_auth_failed",
                "The model service rejected this API credential for inference.",
            )
        if inference_status < 200 or inference_status >= 300:
            raise PiCopilotError(
                "pi_provider_inference_verification_failed",
                "The selected model did not accept a bounded verification request.",
                details={"status_code": int(inference_status)},
            )
        choices = (
            inference_payload.get("choices")
            if isinstance(inference_payload, Mapping)
            else None
        )
        if (
            not isinstance(choices, list)
            or not choices
            or not isinstance(choices[0], Mapping)
        ):
            raise PiCopilotError(
                "pi_provider_inference_response_invalid",
                "The model service returned an invalid Chat Completions response.",
            )
        inference_verified = True
    return {
        "connection_verified": True,
        "model_available": True,
        "inference_verified": inference_verified,
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
