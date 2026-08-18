"""Capture and revalidate one clean local runtime for persisted model fits."""

from __future__ import annotations

import importlib.metadata
import platform
import re
import subprocess
from collections.abc import Mapping
from enum import Enum
from pathlib import Path
from typing import Any, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from ..canonical_json import canonical_json, canonical_sha256
from ..contracts.prediction_model_fit import (
    PredictionModelFitReceipt,
    PredictionPackageVersion,
)
from ..contracts.prediction_validation import (
    PredictionValidationArtifactBinding,
    PredictionValidationRuntimeIdentity,
)
from ..prediction_model_fit_owner import PredictionModelFitBundle
from ..schema import EvidenceRecord
from .evidence_store import EvidenceStore
from .prediction_model_fit_evidence import PredictionModelFitRuntimeAuthority
from .prediction_validation_evidence import (
    resolve_prediction_validation_runtime_authority,
)


_GIT_OBJECT_RE = re.compile(r"^[0-9a-f]{40}$")


class PredictionModelFitRuntimeCaptureReason(str, Enum):
    """Stable runtime-capture and current-runtime refusal reasons."""

    INPUT_INVALID = "prediction_model_fit_runtime_input_invalid"
    CHECKOUT_UNAVAILABLE = "prediction_model_fit_runtime_checkout_unavailable"
    CHECKOUT_DIRTY = "prediction_model_fit_runtime_checkout_dirty"
    ENVIRONMENT_MISMATCH = "prediction_model_fit_runtime_environment_mismatch"
    PERSISTED_RUNTIME_MISMATCH = (
        "prediction_model_fit_runtime_persisted_runtime_mismatch"
    )
    AUTHORITY_CEILING_VIOLATION = (
        "prediction_model_fit_runtime_authority_ceiling_violation"
    )


class PredictionModelFitRuntimeCaptureError(RuntimeError):
    """Typed refusal owned by local prediction-fit runtime capture."""

    owner = "easyicu.prediction_model_fit_runtime"
    phase = "prediction_model_fit_runtime_capture"

    def __init__(
        self,
        reason_code: PredictionModelFitRuntimeCaptureReason,
        message: str,
        **detail: Any,
    ) -> None:
        self.reason_code = reason_code
        self.detail = dict(detail)
        super().__init__(f"{reason_code.value}: {message}")


class PredictionModelFitCodeSnapshot(BaseModel):
    """Clean tracked Git tree used by one deterministic fit runtime."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["easyicu.prediction_model_fit_code_snapshot/1"] = (
        "easyicu.prediction_model_fit_code_snapshot/1"
    )
    git_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    git_tree: str = Field(pattern=r"^[0-9a-f]{40}$")
    git_dirty: Literal[False] = False
    source_scope: Literal["tracked_repository"] = "tracked_repository"


class PredictionModelFitEnvironmentLock(BaseModel):
    """Exact interpreter/platform/package coordinates used by the fit owner."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["easyicu.prediction_model_fit_environment/1"] = (
        "easyicu.prediction_model_fit_environment/1"
    )
    python_implementation: str = Field(min_length=1)
    python_version: str = Field(min_length=1)
    platform_system: str = Field(min_length=1)
    platform_machine: str = Field(min_length=1)
    package_versions: tuple[PredictionPackageVersion, ...] = Field(min_length=1)


def _raise(
    reason_code: PredictionModelFitRuntimeCaptureReason,
    message: str,
    **detail: Any,
) -> None:
    raise PredictionModelFitRuntimeCaptureError(reason_code, message, **detail)


def _canonical_text(value: object, *, name: str) -> str:
    parsed = str(value or "")
    if not parsed or parsed != parsed.strip():
        _raise(
            PredictionModelFitRuntimeCaptureReason.INPUT_INVALID,
            f"{name} must be non-empty and whitespace-canonical",
        )
    return parsed


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _run_git(*arguments: str) -> str:
    try:
        result = subprocess.run(
            ["git", *arguments],
            cwd=_repository_root(),
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise PredictionModelFitRuntimeCaptureError(
            PredictionModelFitRuntimeCaptureReason.CHECKOUT_UNAVAILABLE,
            "git runtime observation failed",
            arguments=list(arguments),
        ) from error
    if result.returncode != 0:
        _raise(
            PredictionModelFitRuntimeCaptureReason.CHECKOUT_UNAVAILABLE,
            "git runtime observation returned a non-zero status",
            arguments=list(arguments),
            returncode=result.returncode,
        )
    return (result.stdout or "").strip()


def _capture_clean_repository() -> PredictionModelFitCodeSnapshot:
    expected_root = _repository_root().resolve()
    try:
        observed_root = Path(_run_git("rev-parse", "--show-toplevel")).resolve()
    except (OSError, RuntimeError, ValueError) as error:
        raise PredictionModelFitRuntimeCaptureError(
            PredictionModelFitRuntimeCaptureReason.CHECKOUT_UNAVAILABLE,
            "prediction-fit source is not in one resolvable Git checkout",
        ) from error
    if observed_root != expected_root:
        _raise(
            PredictionModelFitRuntimeCaptureReason.CHECKOUT_UNAVAILABLE,
            "prediction-fit module resolves to a different Git checkout",
            expected_root=str(expected_root),
            observed_root=str(observed_root),
        )
    status = _run_git("status", "--porcelain", "--untracked-files=normal")
    if status:
        _raise(
            PredictionModelFitRuntimeCaptureReason.CHECKOUT_DIRTY,
            "prediction-fit runtime requires a clean tracked and untracked tree",
            changed_line_count=len(status.splitlines()),
        )
    commit = _run_git("rev-parse", "HEAD")
    tree = _run_git("rev-parse", "HEAD^{tree}")
    if (
        _GIT_OBJECT_RE.fullmatch(commit) is None
        or _GIT_OBJECT_RE.fullmatch(tree) is None
    ):
        _raise(
            PredictionModelFitRuntimeCaptureReason.CHECKOUT_UNAVAILABLE,
            "Git commit or tree identity is not canonical",
        )
    return PredictionModelFitCodeSnapshot(
        git_commit=commit,
        git_tree=tree,
        git_dirty=False,
        source_scope="tracked_repository",
    )


def _parse_fit_receipt(
    value: PredictionModelFitReceipt | Mapping[str, Any],
) -> PredictionModelFitReceipt:
    payload = (
        value.model_dump(mode="python")
        if isinstance(value, PredictionModelFitReceipt)
        else value
    )
    try:
        return PredictionModelFitReceipt.model_validate(payload)
    except ValidationError as error:
        raise PredictionModelFitRuntimeCaptureError(
            PredictionModelFitRuntimeCaptureReason.INPUT_INVALID,
            "prediction-model fit receipt is invalid",
        ) from error


def _capture_environment(
    fit_receipt: PredictionModelFitReceipt,
) -> PredictionModelFitEnvironmentLock:
    observed: list[PredictionPackageVersion] = []
    for expected in fit_receipt.package_versions:
        try:
            version = importlib.metadata.version(expected.distribution)
        except importlib.metadata.PackageNotFoundError as error:
            raise PredictionModelFitRuntimeCaptureError(
                PredictionModelFitRuntimeCaptureReason.ENVIRONMENT_MISMATCH,
                "a fit-bound distribution is unavailable in the current runtime",
                distribution=expected.distribution,
            ) from error
        observed.append(
            PredictionPackageVersion(
                distribution=expected.distribution,
                version=version,
            )
        )
    observed_versions = tuple(observed)
    if observed_versions != fit_receipt.package_versions:
        _raise(
            PredictionModelFitRuntimeCaptureReason.ENVIRONMENT_MISMATCH,
            "current package versions differ from the sealed fit receipt",
            expected=[
                item.model_dump(mode="json") for item in fit_receipt.package_versions
            ],
            observed=[item.model_dump(mode="json") for item in observed_versions],
        )
    return PredictionModelFitEnvironmentLock(
        python_implementation=platform.python_implementation(),
        python_version=platform.python_version(),
        platform_system=platform.system(),
        platform_machine=platform.machine(),
        package_versions=observed_versions,
    )


def _binding(role: str, record: EvidenceRecord) -> PredictionValidationArtifactBinding:
    return PredictionValidationArtifactBinding(
        role=role,
        evidence_id=record.evidence_id,
        sha256=record.sha256,
        kind=record.kind,
        produced_by_step=record.produced_by_step,
    )


def _register_runtime_text(
    *,
    evidence_store: EvidenceStore,
    kind: str,
    description: str,
    text: str,
    filename: str,
    evidence_id: str,
    produced_by_step: str,
    inputs: tuple[str, ...],
    metadata: dict[str, Any],
) -> EvidenceRecord:
    return evidence_store.register_text(
        kind=kind,
        description=description,
        text=text,
        filename=filename,
        evidence_id=evidence_id,
        produced_by_step=produced_by_step,
        inputs=inputs,
        producer="prediction_model_fit_runtime",
        generation_mode="deterministic_skill",
        metadata=metadata,
        publish_aliases=False,
    )


def capture_prediction_model_fit_runtime_authority(
    *,
    evidence_store: EvidenceStore,
    fit_bundle: PredictionModelFitBundle,
    producer_run_id: str,
    runtime_step_id: str,
) -> PredictionModelFitRuntimeAuthority:
    """Capture exact clean Git and package evidence without caller attestations."""

    if not isinstance(evidence_store, EvidenceStore):
        _raise(
            PredictionModelFitRuntimeCaptureReason.INPUT_INVALID,
            "runtime capture requires an EvidenceStore",
        )
    if not isinstance(fit_bundle, PredictionModelFitBundle):
        _raise(
            PredictionModelFitRuntimeCaptureReason.INPUT_INVALID,
            "runtime capture requires one owner-issued fit bundle",
        )
    run_id = _canonical_text(producer_run_id, name="producer_run_id")
    step_id = _canonical_text(runtime_step_id, name="runtime_step_id")
    fit_receipt = _parse_fit_receipt(fit_bundle.receipt)
    code = _capture_clean_repository()
    environment = _capture_environment(fit_receipt)
    code_text = canonical_json(code.model_dump(mode="json"), trailing_newline=True)
    environment_text = canonical_json(
        environment.model_dump(mode="json"),
        trailing_newline=True,
    )
    materialization_key = canonical_sha256(
        {
            "producer_run_id": run_id,
            "runtime_step_id": step_id,
            "fit_receipt_sha256": fit_receipt.receipt_sha256,
            "git_commit": code.git_commit,
            "git_tree": code.git_tree,
            "environment": environment.model_dump(mode="json"),
        }
    )[:12]
    evidence_ids = {
        "code_snapshot": f"prediction_fit_code_{materialization_key}",
        "environment_lock": f"prediction_fit_environment_{materialization_key}",
        "runtime_receipt": f"prediction_fit_runtime_{materialization_key}",
    }
    metadata = {
        "schema_version": "easyicu.prediction_model_fit_runtime_capture/1",
        "capability_id": "prediction_model_fit",
        "maturity": "experimental",
        "claim_ceiling": "analysis_only",
        "paper_authorization": False,
        "planner_selection_authorized": False,
        "fit_receipt_sha256": fit_receipt.receipt_sha256,
        "run_id": run_id,
    }
    aliases_before = dict(evidence_store.aliases())
    numeric_before = len(evidence_store.numeric_claims())
    scientific_before = len(evidence_store.scientific_claims())
    code_record = _register_runtime_text(
        evidence_store=evidence_store,
        kind="code",
        description=(
            "Clean tracked Git identity for one experimental prediction fit; "
            "analysis-only and not manuscript-authoritative."
        ),
        text=code_text,
        filename=f"{evidence_ids['code_snapshot']}.json",
        evidence_id=evidence_ids["code_snapshot"],
        produced_by_step=step_id,
        inputs=(),
        metadata={**metadata, "artifact_role": "code_snapshot"},
    )
    environment_record = _register_runtime_text(
        evidence_store=evidence_store,
        kind="code",
        description=(
            "Exact interpreter and package lock for one experimental prediction "
            "fit; analysis-only and not manuscript-authoritative."
        ),
        text=environment_text,
        filename=f"{evidence_ids['environment_lock']}.json",
        evidence_id=evidence_ids["environment_lock"],
        produced_by_step=step_id,
        inputs=(code_record.evidence_id,),
        metadata={**metadata, "artifact_role": "environment_lock"},
    )
    easyicu_version = next(
        item.version
        for item in environment.package_versions
        if item.distribution == "easyicu"
    )
    runtime = PredictionValidationRuntimeIdentity(
        git_commit=code.git_commit,
        git_dirty=False,
        source_tree_sha256=code_record.sha256,
        environment_sha256=environment_record.sha256,
        runtime_kind="local_process",
        container_image_digest=None,
        python_version=environment.python_version,
        package_version=easyicu_version,
    )
    runtime_record = _register_runtime_text(
        evidence_store=evidence_store,
        kind="log",
        description=(
            "Runtime identity for one experimental prediction fit; analysis-only "
            "and not manuscript-authoritative."
        ),
        text=canonical_json(runtime.model_dump(mode="json"), trailing_newline=True),
        filename=f"{evidence_ids['runtime_receipt']}.json",
        evidence_id=evidence_ids["runtime_receipt"],
        produced_by_step=step_id,
        inputs=(code_record.evidence_id, environment_record.evidence_id),
        metadata={**metadata, "artifact_role": "runtime_receipt"},
    )
    if (
        evidence_store.aliases() != aliases_before
        or len(evidence_store.numeric_claims()) != numeric_before
        or len(evidence_store.scientific_claims()) != scientific_before
    ):
        _raise(
            PredictionModelFitRuntimeCaptureReason.AUTHORITY_CEILING_VIOLATION,
            "runtime capture changed an alias or claim registry",
        )
    return PredictionModelFitRuntimeAuthority(
        producer_run_id=run_id,
        runtime=runtime,
        artifacts=(
            _binding("code_snapshot", code_record),
            _binding("environment_lock", environment_record),
            _binding("runtime_receipt", runtime_record),
        ),
    )


def revalidate_prediction_model_fit_runtime_authority(
    *,
    evidence_store: EvidenceStore,
    runtime_authority: PredictionModelFitRuntimeAuthority | Mapping[str, Any],
    fit_receipt: PredictionModelFitReceipt | Mapping[str, Any],
) -> PredictionModelFitRuntimeAuthority:
    """Re-observe the local checkout and environment against persisted records."""

    payload = (
        runtime_authority.model_dump(mode="python")
        if isinstance(runtime_authority, PredictionModelFitRuntimeAuthority)
        else runtime_authority
    )
    try:
        parsed_authority = PredictionModelFitRuntimeAuthority.model_validate(payload)
    except ValidationError as error:
        raise PredictionModelFitRuntimeCaptureError(
            PredictionModelFitRuntimeCaptureReason.INPUT_INVALID,
            "persisted runtime authority is invalid",
        ) from error
    parsed_receipt = _parse_fit_receipt(fit_receipt)
    resolved = resolve_prediction_validation_runtime_authority(
        evidence_store=evidence_store,
        producer_run_id=parsed_authority.producer_run_id,
        runtime=parsed_authority.runtime,
        artifacts=parsed_authority.artifacts,
    )
    try:
        persisted_code = PredictionModelFitCodeSnapshot.model_validate_json(
            resolved["code_snapshot"].read_text(encoding="utf-8")
        )
        persisted_environment = PredictionModelFitEnvironmentLock.model_validate_json(
            resolved["environment_lock"].read_text(encoding="utf-8")
        )
    except (OSError, UnicodeDecodeError, ValidationError) as error:
        raise PredictionModelFitRuntimeCaptureError(
            PredictionModelFitRuntimeCaptureReason.PERSISTED_RUNTIME_MISMATCH,
            "persisted prediction-fit runtime records are invalid",
        ) from error
    current_code = _capture_clean_repository()
    current_environment = _capture_environment(parsed_receipt)
    if current_code != persisted_code or current_environment != persisted_environment:
        _raise(
            PredictionModelFitRuntimeCaptureReason.PERSISTED_RUNTIME_MISMATCH,
            "current checkout or environment differs from persisted fit authority",
            code_changed=current_code != persisted_code,
            environment_changed=current_environment != persisted_environment,
        )
    easyicu_version = next(
        item.version
        for item in current_environment.package_versions
        if item.distribution == "easyicu"
    )
    expected_runtime = PredictionValidationRuntimeIdentity(
        git_commit=current_code.git_commit,
        git_dirty=False,
        source_tree_sha256=parsed_authority.artifacts[0].sha256,
        environment_sha256=parsed_authority.artifacts[1].sha256,
        runtime_kind="local_process",
        container_image_digest=None,
        python_version=current_environment.python_version,
        package_version=easyicu_version,
    )
    if expected_runtime != parsed_authority.runtime:
        _raise(
            PredictionModelFitRuntimeCaptureReason.PERSISTED_RUNTIME_MISMATCH,
            "persisted runtime receipt differs from current captured coordinates",
        )
    return cast(PredictionModelFitRuntimeAuthority, parsed_authority)


__all__ = [
    "PredictionModelFitCodeSnapshot",
    "PredictionModelFitEnvironmentLock",
    "PredictionModelFitRuntimeCaptureError",
    "PredictionModelFitRuntimeCaptureReason",
    "capture_prediction_model_fit_runtime_authority",
    "revalidate_prediction_model_fit_runtime_authority",
]
