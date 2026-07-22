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


def capability_request_id(payload: Mapping[str, object]) -> str:
    digest = hashlib.sha256(_canonical_bytes(dict(payload))).hexdigest()
    return f"cap-{digest[:16]}"


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
    snapshot_sha = hashlib.sha256(_canonical_bytes(snapshot)).hexdigest()
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


__all__ = [
    "CapabilityApproval",
    "CapabilityRequest",
    "approved_capability_resource",
    "build_capability_request",
    "write_capability_request",
]
