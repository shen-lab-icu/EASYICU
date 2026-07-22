"""Runnable-input authority for paper-facing Canonical9 evaluation.

The tracked archive-v1 manifest is deliberately assessment-only and cannot
authorise a run.  This additive manifest is the benchmark owner's selector for
typed, runnable inputs.  A task remains blocked until the owner freezes the
exact scientific-identity digest and the typed source-authority references.

This module is repository-local paper infrastructure.  It may verify EasyICU
run-input authorities, but it is never imported by the research Agent.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from pathlib import Path
from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from easyicu.research_agent.authority.run_input import (
    RUN_INPUT_CAPSULE_SCHEMA_VERSION_V2,
    RUN_INPUT_CAPSULE_SCHEMA_VERSION_V3,
)

from ..typed_export_seal import (
    TypedRetrofitSealError,
    verify_retrofit_review_attestation,
    verify_retrofit_review_attestation_from_staged,
)
from .rubric_v1 import FIGURE2_TASK_IDS

CANONICAL_RUN_INPUT_BINDING_SCHEMA = "easyicu.figure2_canonical_run_input_bindings/2"
CANONICAL_RUN_INPUT_BINDING_REF = "figure2_canonical9/run_input_bindings/20260718-v2"
_MAX_MANIFEST_BYTES = 2 * 1024 * 1024

Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
CapsuleSchema = Literal[
    "easyicu.run_input_capsule/2",
    "easyicu.run_input_capsule/3",
]
# Task-level cohort identity policy (mirrors typed_export_seal.COHORT_IDENTITY_POLICIES).
CohortIdentityPolicy = Literal[
    "unique_stay_per_patient",
    "first_icu_stay",
    "repeat_admissions_clustered",
]


class CanonicalRunInputBindingError(RuntimeError):
    """The owner-frozen Canonical9 input selector is absent or invalid."""


class _StrictFrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class CanonicalSubmissionProfile(_StrictFrozenModel):
    ref: str = Field(pattern=r"^[a-z][a-z0-9_-]*/[0-9]{8}$")
    concept_dict_sha256: Sha256
    sofa2_dict_sha256: Sha256


class FrozenTypedAuthorityRef(_StrictFrozenModel):
    """Exact host-owned materialized-authority reference frozen by the owner."""

    schema_version: str = Field(min_length=1, max_length=128)
    file: str = Field(min_length=1, max_length=255)
    sha256: Sha256
    size: int = Field(ge=1)

    @field_validator("file")
    @classmethod
    def _one_component(cls, value: str) -> str:
        if Path(value).name != value or value in {".", ".."} or "\\" in value:
            raise ValueError("typed authority file must be one path component")
        return value


class RetrofitReviewAttestation(_StrictFrozenModel):
    """Frozen paper-readiness attestation for a retrofit-sealed source export.

    Minted only via ``typed_export_seal.build_retrofit_review_attestation`` (which
    fail-closes through the paper-readiness gate), then bound into the ready task
    binding so the review proof and source digests survive into the frozen
    selector and are re-verified at resolve time — never checked once and dropped.
    ``paper_ready`` is a ``Literal[True]`` so an unreviewed or identity-insufficient
    attestation cannot validate.
    """

    schema_version: Literal["easyicu.retrofit_review_attestation/1"]
    seal_kind: Literal["retrofitted_structural_typed_export"]
    value_vintage: str = Field(min_length=1, max_length=64)
    cohort_identity_policy: CohortIdentityPolicy
    review_id: str = Field(pattern=r"^review-[0-9a-f]{16}$")
    reviewer: str = Field(min_length=1, max_length=200)
    reviewed_at: str = Field(min_length=1, max_length=80)
    authority_sha256: Sha256
    request_sha256: Sha256
    decision_sha256: Sha256
    # Proof the decision flowed through a real LangGraph interrupt + checkpoint.
    checkpoint_receipt_sha256: Sha256
    source_manifest_sha256: Sha256
    source_sidecar_file: str = Field(min_length=1, max_length=255)
    source_sidecar_sha256: Sha256
    patient_identity_authority_sha256: Sha256
    # Honest patient-level facts the reviewer signed off, for task-level enforcement.
    n_subjects: int = Field(ge=0)
    n_stays_with_subject: int = Field(ge=0)
    multi_stay_patients_present: bool
    first_icu_stay_verified: bool
    paper_ready: Literal[True]

    @field_validator("source_sidecar_file")
    @classmethod
    def _one_component(cls, value: str) -> str:
        if Path(value).name != value or value in {".", ".."} or "\\" in value:
            raise ValueError("sidecar file must be one path component")
        return value

    @model_validator(mode="after")
    def _policy_matches_identity_facts(self) -> "RetrofitReviewAttestation":
        # The reviewed policy must be consistent with the frozen identity facts, so
        # a repeat-admissions source can never carry a unique-stay/first-ICU claim.
        policy = self.cohort_identity_policy
        if policy == "unique_stay_per_patient" and (
            self.multi_stay_patients_present
            or self.n_subjects != self.n_stays_with_subject
        ):
            raise ValueError(
                "unique_stay_per_patient attestation cannot carry repeat admissions"
            )
        if policy == "first_icu_stay" and not self.first_icu_stay_verified:
            raise ValueError(
                "first_icu_stay attestation requires verified first-ICU ordering"
            )
        return self


class BlockedCanonicalTaskBinding(_StrictFrozenModel):
    task_id: str = Field(min_length=3, max_length=128)
    state: Literal["blocked"]
    blockers: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_blockers(self) -> "BlockedCanonicalTaskBinding":
        if (
            self.blockers != tuple(sorted(self.blockers))
            or len(self.blockers) != len(set(self.blockers))
            or any(
                re.fullmatch(r"[A-Z][A-Z0-9_]{2,127}", item) is None
                for item in self.blockers
            )
        ):
            raise ValueError("blocked task codes must be sorted, unique constants")
        return self


class ReadyCanonicalTaskBinding(_StrictFrozenModel):
    task_id: str = Field(min_length=3, max_length=128)
    state: Literal["ready"]
    research_question_sha256: Sha256
    database: str = Field(min_length=1, max_length=128)
    operational_exposure: str | None = Field(max_length=256)
    target_outcome: str = Field(min_length=1, max_length=256)
    expected_run_input_capsule_schema_version: CapsuleSchema
    scientific_identity_sha256: Sha256
    source_materialized_cohort_authority_ref: FrozenTypedAuthorityRef
    source_materialized_trajectory_authority_ref: FrozenTypedAuthorityRef | None
    # Provenance of the materialized source authority. ``retrofit_sealed`` means it
    # derives from a structural retrofit seal of an untyped export (full6-style) and
    # MUST carry the paper-readiness attestation; ``official_typed`` is the official
    # typed-authority path and must not.
    source_kind: Literal["official_typed", "retrofit_sealed"] = "official_typed"
    source_retrofit_review_attestation: RetrofitReviewAttestation | None = None
    # The cohort identity policy THIS task requires of its patient identity. For a
    # retrofit source it is mandatory and MUST equal the reviewed attestation policy,
    # so a source reviewed for repeat-admissions can never satisfy a unique-stay task.
    required_cohort_identity_policy: CohortIdentityPolicy | None = None

    @field_validator("database", "operational_exposure", "target_outcome")
    @classmethod
    def _canonical_text(cls, value: str | None) -> str | None:
        if value is not None and (not value.strip() or value != value.strip()):
            raise ValueError("ready scientific coordinates must be canonical text")
        return value

    @model_validator(mode="after")
    def _retrofit_source_requires_attestation(self) -> "ReadyCanonicalTaskBinding":
        attestation = self.source_retrofit_review_attestation
        if self.source_kind == "retrofit_sealed":
            if attestation is None:
                raise ValueError(
                    "retrofit_sealed source requires a bound review attestation"
                )
            if self.required_cohort_identity_policy is None:
                raise ValueError(
                    "retrofit_sealed task must declare required_cohort_identity_policy"
                )
            if (
                attestation.cohort_identity_policy
                != self.required_cohort_identity_policy
            ):
                raise ValueError(
                    "retrofit attestation policy does not satisfy the task's required "
                    "cohort identity policy"
                )
        else:
            if attestation is not None:
                raise ValueError(
                    "official_typed source must not carry a retrofit review attestation"
                )
            if self.required_cohort_identity_policy is not None:
                raise ValueError(
                    "official_typed source must not declare a cohort identity policy"
                )
        return self

    @model_validator(mode="after")
    def _capsule_shape_matches_trajectory(self) -> "ReadyCanonicalTaskBinding":
        has_trajectory = self.source_materialized_trajectory_authority_ref is not None
        expected = (
            RUN_INPUT_CAPSULE_SCHEMA_VERSION_V3
            if has_trajectory
            else RUN_INPUT_CAPSULE_SCHEMA_VERSION_V2
        )
        if self.expected_run_input_capsule_schema_version != expected:
            raise ValueError(
                "typed capsule version does not match trajectory authority"
            )
        return self


CanonicalTaskBinding = Annotated[
    Union[BlockedCanonicalTaskBinding, ReadyCanonicalTaskBinding],
    Field(discriminator="state"),
]


class CanonicalRunInputBindingManifest(_StrictFrozenModel):
    schema_version: Literal["easyicu.figure2_canonical_run_input_bindings/2"]
    manifest_ref: Literal["figure2_canonical9/run_input_bindings/20260718-v2"]
    submission_profile: CanonicalSubmissionProfile
    tasks: tuple[CanonicalTaskBinding, ...]

    @model_validator(mode="after")
    def _exact_suite(self) -> "CanonicalRunInputBindingManifest":
        if tuple(item.task_id for item in self.tasks) != FIGURE2_TASK_IDS:
            raise ValueError(
                "run-input binding manifest must contain exact Canonical9 order"
            )
        return self


def _reject_constant(value: str) -> None:
    raise CanonicalRunInputBindingError(f"non-finite JSON constant: {value}")


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise CanonicalRunInputBindingError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _canonical_run_input_binding_path() -> Path:
    """Return the fixed repository-owned selector (patched only in tests)."""

    return Path(__file__).resolve().parents[1] / "canonical_run_input_bindings_v2.json"


def _read_manifest_once(
    path: Path,
) -> tuple[CanonicalRunInputBindingManifest, bytes, str]:
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NONBLOCK", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size > _MAX_MANIFEST_BYTES:
            raise CanonicalRunInputBindingError(
                "run-input binding selector must be a small regular file"
            )
        chunks: list[bytes] = []
        remaining = int(before.st_size)
        while remaining:
            chunk = os.read(descriptor, min(64 * 1024, remaining))
            if not chunk:
                raise CanonicalRunInputBindingError(
                    "run-input binding selector ended before its stat size"
                )
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise CanonicalRunInputBindingError(
                "run-input binding selector changed while being read"
            )
        after = os.fstat(descriptor)
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        ) != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
            raise CanonicalRunInputBindingError(
                "run-input binding selector changed while being read"
            )
    finally:
        os.close(descriptor)
    raw = b"".join(chunks)
    try:
        json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_constant,
        )
        manifest = CanonicalRunInputBindingManifest.model_validate_json(
            raw, strict=True
        )
    except (UnicodeDecodeError, ValueError) as exc:
        raise CanonicalRunInputBindingError(
            f"run-input binding selector is invalid: {exc}"
        ) from exc
    canonical = _canonical_json_bytes(manifest.model_dump(mode="json")) + b"\n"
    if raw != canonical:
        raise CanonicalRunInputBindingError(
            "run-input binding selector is not canonical JSON"
        )
    return manifest, raw, hashlib.sha256(raw).hexdigest()


def load_canonical_run_input_bindings() -> tuple[CanonicalRunInputBindingManifest, str]:
    """Load the fixed benchmark-owner selector and its exact byte digest."""

    manifest, _raw, digest = _read_manifest_once(_canonical_run_input_binding_path())
    return manifest, digest


def require_ready_task_binding(
    task_id: str,
    *,
    source_export_dir: str | Path | None = None,
    staged_authority_dir: str | Path | None = None,
) -> tuple[
    CanonicalRunInputBindingManifest,
    ReadyCanonicalTaskBinding,
    str,
    str,
]:
    """Resolve one exact ready binding or fail closed before run sealing.

    For a ``retrofit_sealed`` source, content-addressed re-validation is MANDATORY.
    Provide EITHER a live ``source_export_dir`` (identity re-derived from parquet
    columns; manifest/sidecar/identity digests recomputed) OR a content-addressed
    ``staged_authority_dir`` (staged, SHA-verified authority blobs re-digested — the
    production path, needing no external mutable export dir). Either way the frozen
    attestation, its bound HITL decision, checkpoint receipt, and cohort identity
    policy are re-verified. A retrofit binding with neither source resolvable fails
    closed rather than trusting frozen strings.
    """

    manifest, manifest_digest = load_canonical_run_input_bindings()
    matches = [item for item in manifest.tasks if item.task_id == task_id]
    if len(matches) != 1:
        raise CanonicalRunInputBindingError("canonical task binding is not unique")
    binding = matches[0]
    if not isinstance(binding, ReadyCanonicalTaskBinding):
        blockers = ",".join(binding.blockers)
        raise PermissionError(
            f"Canonical9 task {task_id!r} is not input-frozen: {blockers}"
        )
    if binding.source_kind == "retrofit_sealed":
        # The model_validator guarantees a non-None attestation whose policy matches
        # the task's required_cohort_identity_policy; re-verify against a content
        # source (live columns or staged content-addressed blobs).
        attestation = binding.source_retrofit_review_attestation.model_dump(mode="json")
        try:
            if source_export_dir is not None:
                verify_retrofit_review_attestation(
                    attestation, export_dir=source_export_dir
                )
            elif staged_authority_dir is not None:
                verify_retrofit_review_attestation_from_staged(
                    attestation, staged_authority_dir
                )
            else:
                raise CanonicalRunInputBindingError(
                    "retrofit_sealed acceptance requires content-addressed "
                    "re-validation: pass source_export_dir (live) or "
                    "staged_authority_dir (staged); offline-only acceptance is refused"
                )
        except TypedRetrofitSealError as exc:
            raise CanonicalRunInputBindingError(
                f"retrofit source attestation failed re-verification: {exc}"
            ) from exc
    case_bytes = _canonical_json_bytes(binding.model_dump(mode="json"))
    return manifest, binding, manifest_digest, hashlib.sha256(case_bytes).hexdigest()


__all__ = [
    "CANONICAL_RUN_INPUT_BINDING_REF",
    "CANONICAL_RUN_INPUT_BINDING_SCHEMA",
    "BlockedCanonicalTaskBinding",
    "CanonicalRunInputBindingError",
    "CanonicalRunInputBindingManifest",
    "CanonicalSubmissionProfile",
    "FrozenTypedAuthorityRef",
    "ReadyCanonicalTaskBinding",
    "load_canonical_run_input_bindings",
    "require_ready_task_binding",
]
