"""Reviewable capability requests for adding analytical software.

The running agent never installs packages.  An unavailable method produces an
immutable request that a maintainer may validate in a rebuilt sandbox image.
Only a digest-bound approval can become a selectable software resource.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .schema import ResourceDescriptor

CAPABILITY_REQUEST_SCHEMA = "easyicu.capability_request/1"
CAPABILITY_APPROVAL_SCHEMA = "easyicu.capability_approval/1"
CAPABILITY_ACTIVATION_SCHEMA = "easyicu.capability_activation/1"


def _canonical_bytes(payload: object) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


class CapabilityRequest(BaseModel):
    """A non-executable request to add one method package."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["easyicu.capability_request/1"] = CAPABILITY_REQUEST_SCHEMA
    request_id: str = Field(pattern=r"^cap-[0-9a-f]{16}$")
    method_name: str = Field(min_length=1, max_length=160)
    package_name: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
    import_name: str = Field(pattern=r"^[A-Za-z_][A-Za-z0-9_.]{0,127}$")
    version_spec: str = Field(min_length=1, max_length=80)
    purpose: str = Field(min_length=1, max_length=500)
    analysis_families: tuple[str, ...]
    required_input_roles: tuple[str, ...] = ()
    produced_output_roles: tuple[str, ...] = ()
    license_spdx: str = Field(min_length=1, max_length=80)
    upstream_source: str = Field(min_length=1, max_length=500)
    validation_test_refs: tuple[str, ...]
    requested_by: str = Field(min_length=1, max_length=160)
    requested_at: str = Field(min_length=1, max_length=80)
    runtime_snapshot_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    runtime_install_allowed: Literal[False] = False

    @field_validator(
        "analysis_families",
        "required_input_roles",
        "produced_output_roles",
        "validation_test_refs",
    )
    @classmethod
    def _unique_nonempty(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        cleaned = tuple(value.strip() for value in values if value.strip())
        if len(cleaned) != len(set(cleaned)):
            raise ValueError("capability coordinates must be unique")
        return cleaned

    @model_validator(mode="after")
    def _required_review_inputs(self) -> "CapabilityRequest":
        if not self.analysis_families:
            raise ValueError("capability request requires an analysis family")
        if not self.validation_test_refs:
            raise ValueError("capability request requires validation test references")
        expected = capability_request_id(self.model_dump(exclude={"request_id"}))
        if self.request_id != expected:
            raise ValueError("capability request id does not bind request contents")
        return self

    @property
    def sha256(self) -> str:
        return hashlib.sha256(
            _canonical_bytes(self.model_dump(mode="json"))
        ).hexdigest()


class CapabilityApproval(BaseModel):
    """Maintainer approval bound to tests and an immutable sandbox image."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["easyicu.capability_approval/1"] = (
        CAPABILITY_APPROVAL_SCHEMA
    )
    request_id: str
    request_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    decision: Literal["approved", "rejected"]
    reviewer: str = Field(min_length=1, max_length=200)
    reviewed_at: str = Field(min_length=1, max_length=80)
    installed_version: str | None = Field(default=None, max_length=80)
    image_reference: str | None = Field(default=None, max_length=500)
    image_digest: str | None = Field(default=None, pattern=r"^sha256:[0-9a-f]{64}$")
    validation_receipt_sha256: str | None = Field(
        default=None, pattern=r"^[0-9a-f]{64}$"
    )
    notes: str = Field(default="", max_length=1_000)

    @model_validator(mode="after")
    def _approved_is_fully_bound(self) -> "CapabilityApproval":
        required = (
            self.installed_version,
            self.image_reference,
            self.image_digest,
            self.validation_receipt_sha256,
        )
        if self.decision == "approved" and not all(required):
            raise ValueError(
                "approved capability requires version, image, and test receipt"
            )
        return self

    @property
    def sha256(self) -> str:
        return hashlib.sha256(
            _canonical_bytes(self.model_dump(mode="json"))
        ).hexdigest()


class CapabilityActivation(BaseModel):
    """A reviewed hand-off into a *new* immutable runtime/profile.

    Approval never authorises package installation inside a running container.
    A maintainer must build and validate a new image, register a new submission
    profile that pins its digest, and start a new run with this activation.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["easyicu.capability_activation/1"] = (
        CAPABILITY_ACTIVATION_SCHEMA
    )
    activation_id: str = Field(pattern=r"^capact-[0-9a-f]{16}$")
    request_id: str = Field(pattern=r"^cap-[0-9a-f]{16}$")
    request_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    approval_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_profile_ref: str = Field(min_length=3, max_length=240)
    target_profile_ref: str = Field(min_length=3, max_length=240)
    image_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    new_run_required: Literal[True] = True
    runtime_install_allowed: Literal[False] = False

    @model_validator(mode="after")
    def _bind_activation(self) -> "CapabilityActivation":
        if self.source_profile_ref == self.target_profile_ref:
            raise ValueError("capability activation requires a new profile")
        expected = capability_activation_id(self.model_dump(exclude={"activation_id"}))
        if self.activation_id != expected:
            raise ValueError("capability activation id does not bind activation bytes")
        return self

    @property
    def sha256(self) -> str:
        return hashlib.sha256(
            _canonical_bytes(self.model_dump(mode="json"))
        ).hexdigest()


def capability_request_id(payload: Mapping[str, object]) -> str:
    digest = hashlib.sha256(_canonical_bytes(dict(payload))).hexdigest()
    return f"cap-{digest[:16]}"


def capability_activation_id(payload: Mapping[str, object]) -> str:
    digest = hashlib.sha256(_canonical_bytes(dict(payload))).hexdigest()
    return f"capact-{digest[:16]}"


def runtime_snapshot_sha256(import_names: Iterable[str]) -> str:
    """Canonical digest of the runner's exact import-name allow-list."""

    return hashlib.sha256(
        _canonical_bytes(tuple(sorted(set(import_names))))
    ).hexdigest()


def build_capability_request(
    *,
    method_name: str,
    package_name: str,
    import_name: str,
    version_spec: str,
    purpose: str,
    analysis_families: Iterable[str],
    license_spdx: str,
    upstream_source: str,
    validation_test_refs: Iterable[str],
    requested_by: str,
    requested_at: str,
    runtime_import_names: Iterable[str],
    required_input_roles: Iterable[str] = (),
    produced_output_roles: Iterable[str] = (),
) -> CapabilityRequest:
    """Build a request only when the exact import is unavailable."""

    snapshot = tuple(sorted(set(runtime_import_names)))
    if import_name in snapshot:
        raise ValueError("installed software must be registered, not requested")
    snapshot_sha = runtime_snapshot_sha256(snapshot)
    payload = {
        "schema_version": CAPABILITY_REQUEST_SCHEMA,
        "method_name": method_name,
        "package_name": package_name,
        "import_name": import_name,
        "version_spec": version_spec,
        "purpose": purpose,
        "analysis_families": tuple(analysis_families),
        "required_input_roles": tuple(required_input_roles),
        "produced_output_roles": tuple(produced_output_roles),
        "license_spdx": license_spdx,
        "upstream_source": upstream_source,
        "validation_test_refs": tuple(validation_test_refs),
        "requested_by": requested_by,
        "requested_at": requested_at,
        "runtime_snapshot_sha256": snapshot_sha,
        "runtime_install_allowed": False,
    }
    return CapabilityRequest(
        request_id=capability_request_id(payload),
        **payload,
    )


def approved_capability_resource(
    request: CapabilityRequest,
    approval: CapabilityApproval,
) -> ResourceDescriptor:
    """Convert a fully reviewed package into an immutable software resource."""

    if approval.request_id != request.request_id:
        raise ValueError("capability approval belongs to a different request")
    if approval.request_sha256 != request.sha256:
        raise ValueError("capability approval does not bind request bytes")
    if approval.decision != "approved":
        raise ValueError("rejected capability cannot enter the resource catalog")
    assert approval.installed_version is not None
    assert approval.image_digest is not None
    assert approval.validation_receipt_sha256 is not None
    projection = _canonical_bytes(
        {
            "method_name": request.method_name,
            "import_name": request.import_name,
            "installed_version": approval.installed_version,
            "image_digest": approval.image_digest,
            "validation_receipt_sha256": approval.validation_receipt_sha256,
            "runtime_install_allowed": False,
        }
    ).decode("utf-8")
    return ResourceDescriptor(
        resource_id=f"software:{request.import_name.replace('_', '-')}",
        version=approval.installed_version,
        sha256=hashlib.sha256(projection.encode("utf-8")).hexdigest(),
        kind="software",
        analysis_families=request.analysis_families,
        required_input_roles=request.required_input_roles,
        produced_output_roles=request.produced_output_roles,
        permissions=("coder_context", "sandbox_import"),
        review_status="approved",
        search_terms=(request.method_name, request.package_name, request.import_name),
        prompt_projection=projection,
    )


def build_capability_activation(
    *,
    request: CapabilityRequest,
    approval: CapabilityApproval,
    source_profile_ref: str,
    target_profile_ref: str,
) -> CapabilityActivation:
    """Bind an approval to a different profile and immutable image."""

    # Reuse the catalog conversion as the single approval/request join check.
    approved_capability_resource(request, approval)
    assert approval.image_digest is not None
    payload = {
        "schema_version": CAPABILITY_ACTIVATION_SCHEMA,
        "request_id": request.request_id,
        "request_sha256": request.sha256,
        "approval_sha256": approval.sha256,
        "source_profile_ref": source_profile_ref,
        "target_profile_ref": target_profile_ref,
        "image_digest": approval.image_digest,
        "new_run_required": True,
        "runtime_install_allowed": False,
    }
    return CapabilityActivation(
        activation_id=capability_activation_id(payload),
        **payload,
    )


def verify_capability_activation(
    *,
    request: CapabilityRequest,
    approval: CapabilityApproval,
    activation: CapabilityActivation,
    current_profile_ref: str,
    expected_image_digest: str | None,
    actual_image_digest: str | None,
    runtime_import_names: Iterable[str],
    is_resume: bool,
) -> ResourceDescriptor:
    """Fail closed unless a new run uses the exact approved environment."""

    resource = approved_capability_resource(request, approval)
    if is_resume:
        raise ValueError("capability activation requires a new run, not resume")
    if activation.request_id != request.request_id:
        raise ValueError("capability activation belongs to a different request")
    if activation.request_sha256 != request.sha256:
        raise ValueError("capability activation does not bind request bytes")
    if activation.approval_sha256 != approval.sha256:
        raise ValueError("capability activation does not bind approval bytes")
    if activation.target_profile_ref != current_profile_ref:
        raise ValueError("capability activation targets a different profile")
    if not expected_image_digest:
        raise ValueError("target profile does not pin an immutable image digest")
    if activation.image_digest != expected_image_digest:
        raise ValueError("capability activation conflicts with target profile image")
    if actual_image_digest != expected_image_digest:
        raise ValueError("runtime image does not match capability activation")
    if request.import_name not in set(runtime_import_names):
        raise ValueError("approved capability import is absent from the new runtime")
    return resource


def write_capability_request(path: Path, request: CapabilityRequest) -> None:
    """Write a request once; never mutate a previously reviewed request."""

    path = Path(path)
    payload = request.model_dump_json(indent=2).encode("utf-8") + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise FileExistsError("capability request path already has other bytes")
        return
    fd, temp_name = tempfile.mkstemp(prefix=".capability-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temp_name, path)
        except FileExistsError:
            if path.read_bytes() != payload:
                raise FileExistsError(
                    "capability request path already has other bytes"
                ) from None
    finally:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass


def write_capability_activation(path: Path, activation: CapabilityActivation) -> None:
    """Write an activation receipt once, with byte-for-byte idempotence."""

    path = Path(path)
    payload = activation.model_dump_json(indent=2).encode("utf-8") + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise FileExistsError("capability activation path has other bytes")
        return
    fd, temp_name = tempfile.mkstemp(prefix=".capability-activation-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temp_name, path)
        except FileExistsError:
            if path.read_bytes() != payload:
                raise FileExistsError(
                    "capability activation path has other bytes"
                ) from None
    finally:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass


def write_capability_approval(path: Path, approval: CapabilityApproval) -> None:
    """Write the digest-bound human approval once."""

    path = Path(path)
    payload = approval.model_dump_json(indent=2).encode("utf-8") + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise FileExistsError("capability approval path has other bytes")
        return
    fd, temp_name = tempfile.mkstemp(prefix=".capability-approval-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temp_name, path)
        except FileExistsError:
            if path.read_bytes() != payload:
                raise FileExistsError(
                    "capability approval path has other bytes"
                ) from None
    finally:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass


__all__ = [
    "CapabilityActivation",
    "CapabilityApproval",
    "CapabilityRequest",
    "approved_capability_resource",
    "build_capability_activation",
    "build_capability_request",
    "verify_capability_activation",
    "runtime_snapshot_sha256",
    "write_capability_activation",
    "write_capability_approval",
    "write_capability_request",
]
