"""Single content-addressed execution identity for runs and reuse gates."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Literal, Mapping

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..providers.factory import provider_authorization_manifest
from ..providers.prompts import PROMPT_PACK_VERSION, prompt_pack_files
from .runtime_artifacts import capture_code_version

EXECUTION_IDENTITY_SCHEMA = "easyicu.execution_identity/2"
EXPECTED_EXECUTION_IDENTITY_SCHEMA = "easyicu.expected_execution_identity/1"


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(
        dict(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _prompt_pack_sha256() -> str:
    return hashlib.sha256(
        _canonical_json(
            {
                "version": PROMPT_PACK_VERSION,
                "files": prompt_pack_files(),
            }
        ).encode("utf-8")
    ).hexdigest()


class ExecutionIdentity(BaseModel):
    """Strict coordinates that must match before reuse or paper acceptance."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal["easyicu.execution_identity/2"]
    submission_profile_ref: str | None
    runner: Literal["auto", "docker", "subprocess", "custom"]
    runner_image_digest: str | None
    network_policy: str
    provider_authorization: dict[str, Any]
    provider_authorization_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    prompt_pack_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    git_sha: str | None
    git_dirty: bool | None
    llm_seed: int | None
    data_seed: int | None
    input_authority_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    environment_identity_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    host_runner_authorized: bool
    paper_eligible: bool
    identity_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def _verify_digests_and_eligibility(self) -> "ExecutionIdentity":
        provider_sha = hashlib.sha256(
            _canonical_json(self.provider_authorization).encode("utf-8")
        ).hexdigest()
        if provider_sha != self.provider_authorization_sha256:
            raise ValueError("provider authorization digest mismatch")
        payload = self.model_dump(mode="json", exclude={"identity_sha256"})
        environment_payload = dict(payload)
        environment_payload.pop("environment_identity_sha256", None)
        environment_payload.pop("data_seed", None)
        environment_payload.pop("input_authority_sha256", None)
        environment_sha = hashlib.sha256(
            _canonical_json(environment_payload).encode("utf-8")
        ).hexdigest()
        if environment_sha != self.environment_identity_sha256:
            raise ValueError("execution environment identity digest mismatch")
        identity_sha = hashlib.sha256(
            _canonical_json(payload).encode("utf-8")
        ).hexdigest()
        if identity_sha != self.identity_sha256:
            raise ValueError("execution identity digest mismatch")
        expected_eligible = _paper_eligible(payload)
        if self.paper_eligible is not expected_eligible:
            raise ValueError("execution identity paper eligibility mismatch")
        return self

    @classmethod
    def create(
        cls,
        *,
        submission_profile_name: str | None,
        submission_profile_version: str | None,
        runner: str,
        runner_image_digest: str | None,
        network_policy: str,
        llm_seed: int | None,
        data_seed: int | None = None,
        input_authority_sha256: str | None = None,
        provider_client: Any = None,
        provider_authorization: Mapping[str, Any] | None = None,
        host_runner_authorized: bool = False,
        code_version: Mapping[str, Any] | None = None,
    ) -> "ExecutionIdentity":
        normalized_runner = str(runner or "auto").strip().lower()
        aliases = {
            "default": "auto",
            "host": "subprocess",
            "container": "docker",
            "openhands": "docker",
        }
        normalized_runner = aliases.get(normalized_runner, normalized_runner)
        if normalized_runner not in {"auto", "docker", "subprocess", "custom"}:
            raise ValueError("unsupported execution identity runner")
        profile_ref = None
        if (
            submission_profile_name is not None
            or submission_profile_version is not None
        ):
            if not submission_profile_name or not submission_profile_version:
                raise ValueError("submission profile identity is incomplete")
            profile_ref = f"{submission_profile_name}/{submission_profile_version}"
        if provider_authorization is not None and provider_client is not None:
            raise ValueError("provide either provider_client or provider_authorization")
        provider = (
            dict(provider_authorization)
            if provider_authorization is not None
            else provider_authorization_manifest(provider_client)
        )
        provider_sha = hashlib.sha256(
            _canonical_json(provider).encode("utf-8")
        ).hexdigest()
        version = dict(code_version or capture_code_version() or {})
        payload: dict[str, Any] = {
            "schema_version": EXECUTION_IDENTITY_SCHEMA,
            "submission_profile_ref": profile_ref,
            "runner": normalized_runner,
            "runner_image_digest": (
                str(runner_image_digest) if runner_image_digest is not None else None
            ),
            "network_policy": str(network_policy or "none"),
            "provider_authorization": provider,
            "provider_authorization_sha256": provider_sha,
            "prompt_pack_sha256": _prompt_pack_sha256(),
            "git_sha": version.get("git_sha"),
            "git_dirty": version.get("git_dirty"),
            "llm_seed": int(llm_seed) if llm_seed is not None else None,
            "data_seed": int(data_seed) if data_seed is not None else None,
            "input_authority_sha256": input_authority_sha256,
            "host_runner_authorized": bool(host_runner_authorized),
        }
        payload["paper_eligible"] = _paper_eligible(payload)
        environment_payload = dict(payload)
        environment_payload.pop("data_seed", None)
        environment_payload.pop("input_authority_sha256", None)
        payload["environment_identity_sha256"] = hashlib.sha256(
            _canonical_json(environment_payload).encode("utf-8")
        ).hexdigest()
        payload["identity_sha256"] = hashlib.sha256(
            _canonical_json(payload).encode("utf-8")
        ).hexdigest()
        return cls.model_validate(payload, strict=True)


class ExpectedExecutionIdentity(BaseModel):
    """An operator-frozen identity supplied independently of run outputs."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal["easyicu.expected_execution_identity/1"]
    execution_identity: ExecutionIdentity
    expected_identity_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    freeze_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def _verify_frozen_identity(self) -> "ExpectedExecutionIdentity":
        if (
            self.expected_identity_sha256
            != self.execution_identity.environment_identity_sha256
        ):
            raise ValueError("expected execution environment identity digest mismatch")
        payload = self.model_dump(mode="json", exclude={"freeze_sha256"})
        expected_freeze_sha = hashlib.sha256(
            _canonical_json(payload).encode("utf-8")
        ).hexdigest()
        if self.freeze_sha256 != expected_freeze_sha:
            raise ValueError("expected execution identity freeze digest mismatch")
        return self

    @classmethod
    def create(cls, identity: ExecutionIdentity) -> "ExpectedExecutionIdentity":
        if not isinstance(identity, ExecutionIdentity):
            raise TypeError("expected execution identity requires ExecutionIdentity")
        payload: dict[str, Any] = {
            "schema_version": EXPECTED_EXECUTION_IDENTITY_SCHEMA,
            "execution_identity": identity.model_dump(mode="json"),
            "expected_identity_sha256": identity.environment_identity_sha256,
        }
        payload["freeze_sha256"] = hashlib.sha256(
            _canonical_json(payload).encode("utf-8")
        ).hexdigest()
        return cls.model_validate(payload, strict=True)


def _paper_eligible(payload: Mapping[str, Any]) -> bool:
    profile_ref = payload.get("submission_profile_ref")
    image_digest = payload.get("runner_image_digest")
    provider = payload.get("provider_authorization")
    clients = provider.get("clients") if isinstance(provider, Mapping) else None
    provider_bound = bool(clients) and all(
        isinstance(item, Mapping)
        and item.get("authorization_mode") not in {None, "unmanaged", "mock_exempt"}
        for item in clients
    )
    image_hex = (
        image_digest.removeprefix("sha256:") if isinstance(image_digest, str) else ""
    )
    return bool(
        profile_ref
        and payload.get("runner") == "docker"
        and len(image_hex) == 64
        and all(ch in "0123456789abcdef" for ch in image_hex)
        and str(payload.get("network_policy") or "").lower() in {"none", "disabled"}
        and not payload.get("host_runner_authorized")
        and payload.get("git_sha")
        and payload.get("git_dirty") is False
        and provider_bound
    )


def execution_identity_for_pipeline(pipeline: Any) -> ExecutionIdentity:
    """Build the exact identity from one configured pipeline instance."""

    cached = getattr(pipeline, "_execution_identity", None)
    if isinstance(cached, ExecutionIdentity):
        return cached
    identity = ExecutionIdentity.create(
        submission_profile_name=pipeline._submission_profile_name,
        submission_profile_version=pipeline._submission_profile_version,
        runner=pipeline._runner_kind,
        runner_image_digest=pipeline._expected_runner_image_digest,
        network_policy=pipeline._runner_network,
        provider_client=pipeline._llm,
        llm_seed=pipeline._llm_seed,
        data_seed=pipeline._config.execution_data_seed,
        input_authority_sha256=pipeline._config.execution_input_authority_sha256,
        host_runner_authorized=pipeline._host_runner_authorized,
    )
    pipeline._execution_identity = identity
    return identity


__all__ = [
    "EXECUTION_IDENTITY_SCHEMA",
    "EXPECTED_EXECUTION_IDENTITY_SCHEMA",
    "ExpectedExecutionIdentity",
    "ExecutionIdentity",
    "execution_identity_for_pipeline",
]
