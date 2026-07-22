"""Fail-closed pre-run authorization gate for a real Canonical9 ``aware`` run.

This gate is enforced INSIDE the real launcher (``tools/run_research_agent_bench.py``)
before any pipeline, subprocess, Provider, or data load.  It never launches a run,
calls a Provider, runs Docker, reads patient data, or grants paper authority.

The binding is an **operator freeze declaration**: every identity is PINNED by the
operator ahead of the run and the gate only verifies that the on-disk artifacts
still hash to those pins.  No runtime-computed SHA is ever written back and then
trusted as a binding.

Reused, never re-implemented:

* ``evaluator/input_freeze_v1.py`` — the strict typed loader + canonical digest.
  The tracked ``canonical_input_freeze_v1`` is an *assessment authority*: its
  schema forces exactly ``e2/e3/h2`` and every case ``state == "blocked"``, so it
  can NEVER be forged into a runnable full-9 authority (``cases=[]``, a missing /
  duplicate / out-of-order case, or an unknown field are rejected by the loader).
* ``ExpectedExecutionIdentity`` (engine) — clean-tree commit, docker image digest,
  non-secret provider authorization, prompt pack, network policy, ``paper_eligible``.
* ``FIGURE2_TASK_IDS`` (evaluator) — exact ordered 9-task coverage.

The authorized path additionally requires a real **typed production input
authority** covering the exact nine tasks in fixed order with per-task input and
provenance digests, whose ``authority_digest`` must equal both the operator pin
and ``ExecutionIdentity.input_authority_sha256`` (a direct comparison).  No such
production authority exists yet, so against the current repository the gate always
blocks — which is the honest state.

The one operation this gate never performs is launching the real run.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal, Optional, Sequence

from pydantic import BaseModel, ConfigDict, Field, model_validator

from easyicu.research_agent.authority.execution_identity import (
    ExpectedExecutionIdentity,
)

from .evaluator.input_freeze_v1 import (
    CanonicalInputFreezeError,
    canonical_input_freeze_manifest_sha256,
    load_canonical_input_freeze_manifest,
)
from .evaluator.rubric_v1 import FIGURE2_TASK_IDS

REALRUN_AUTHORIZATION_SCHEMA = "easyicu.figure2_realrun_authorization/1"
OPERATOR_FREEZE_DECLARATION_SCHEMA = "easyicu.figure2_operator_freeze_declaration/1"
PRODUCTION_INPUT_AUTHORITY_SCHEMA = "easyicu.figure2_production_input_authority/1"
AUTHORIZED_ARMS: tuple[str, ...] = ("aware",)
_MAX_DOC_BYTES = 8 * 1024 * 1024

Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
Sha1Hex = Annotated[str, Field(pattern=r"^[0-9a-f]{40}$")]


class _StrictFrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _read_regular_file(path: Path) -> bytes:
    fd: int | None = None
    try:
        fd = os.open(
            path,
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
        )
        meta = os.fstat(fd)
        if not stat.S_ISREG(meta.st_mode):
            raise ValueError("authorization input must be a regular file")
        if meta.st_size > _MAX_DOC_BYTES:
            raise ValueError("authorization input exceeds the size limit")
        chunks: list[bytes] = []
        while chunk := os.read(fd, 64 * 1024):
            chunks.append(chunk)
        return b"".join(chunks)
    finally:
        if fd is not None:
            os.close(fd)


def _require_safe_path(path: Path) -> Path:
    resolved = Path(path).expanduser()
    if not resolved.is_absolute() or resolved.is_symlink():
        raise ValueError("path must be an absolute, non-symlink path")
    return resolved.resolve(strict=True)


def _sha256_of(path: Path) -> str:
    return hashlib.sha256(_read_regular_file(_require_safe_path(path))).hexdigest()


# ---------------------------------------------------------------------------
# Typed production input authority (the future authorized full-9 input)
# ---------------------------------------------------------------------------


class ProductionInputTask(_StrictFrozenModel):
    task_id: str
    input_sha256: Sha256
    provenance_sha256: Sha256


class ProductionInputAuthority(_StrictFrozenModel):
    """A real, exact-nine, byte- and provenance-digested production input authority."""

    schema_version: Literal["easyicu.figure2_production_input_authority/1"]
    submission_profile_ref: str
    tasks: tuple[ProductionInputTask, ...]
    authority_digest: Sha256

    @model_validator(mode="after")
    def _exact_nine_and_self_consistent(self) -> "ProductionInputAuthority":
        if tuple(task.task_id for task in self.tasks) != tuple(FIGURE2_TASK_IDS):
            raise ValueError(
                "production input authority must cover the exact ordered nine tasks"
            )
        body = self.model_dump(mode="json", exclude={"authority_digest"})
        if (
            hashlib.sha256(_canonical_json_bytes(body)).hexdigest()
            != self.authority_digest
        ):
            raise ValueError("production input authority digest mismatch")
        return self

    @classmethod
    def build(
        cls, *, submission_profile_ref: str, tasks: Sequence[ProductionInputTask]
    ) -> "ProductionInputAuthority":
        body = {
            "schema_version": PRODUCTION_INPUT_AUTHORITY_SCHEMA,
            "submission_profile_ref": submission_profile_ref,
            "tasks": [task.model_dump(mode="json") for task in tasks],
        }
        digest = hashlib.sha256(_canonical_json_bytes(body)).hexdigest()
        return cls(
            schema_version=PRODUCTION_INPUT_AUTHORITY_SCHEMA,
            submission_profile_ref=submission_profile_ref,
            tasks=tuple(tasks),
            authority_digest=digest,
        )


def load_production_input_authority(
    path: Path,
) -> tuple[ProductionInputAuthority, str]:
    """Strict-load a production input authority, rejecting a v1 assessment manifest."""

    safe = _require_safe_path(path)
    raw = _read_regular_file(safe)
    # A v1 assessment manifest (blocked e2/e3/h2) must never masquerade as a full-9
    # production authority; detect it explicitly with the strict v1 loader.
    try:
        load_canonical_input_freeze_manifest(safe)
    except CanonicalInputFreezeError:
        pass
    else:
        raise ValueError(
            "canonical_input_freeze assessment manifest is not a full-9 "
            "production input authority"
        )
    authority = ProductionInputAuthority.model_validate_json(raw, strict=True)
    return authority, hashlib.sha256(raw).hexdigest()


# ---------------------------------------------------------------------------
# Operator freeze declaration (all pins REQUIRED)
# ---------------------------------------------------------------------------


class OperatorFreezeDeclaration(_StrictFrozenModel):
    """Operator-frozen pins fixed BEFORE the run; the gate only verifies them."""

    schema_version: Literal["easyicu.figure2_operator_freeze_declaration/1"]
    expected_execution_identity_sha256: Sha256
    input_authority_digest: Sha256
    input_freeze_manifest_sha256: Sha256
    rubric_sha256: Sha256
    code_commit_sha: Sha1Hex
    runner: Literal["docker"]
    network_policy: Literal["none", "disabled"]
    runner_image_digest: Sha256
    task_ids: tuple[str, ...]
    arms: tuple[str, ...]
    cross_run_memory: Literal[False]
    output_root: str
    run_id: str

    @model_validator(mode="after")
    def _declared_intent_is_canonical(self) -> "OperatorFreezeDeclaration":
        if self.task_ids != tuple(FIGURE2_TASK_IDS):
            raise ValueError("declaration must pin the exact ordered nine task ids")
        if self.arms != AUTHORIZED_ARMS:
            raise ValueError("declaration must pin exactly the aware arm")
        root = Path(self.output_root)
        if not root.is_absolute():
            raise ValueError("declaration output_root must be absolute")
        if not self.run_id.startswith("run_"):
            raise ValueError("declaration run_id must start with 'run_'")
        return self


# ---------------------------------------------------------------------------
# Receipt
# ---------------------------------------------------------------------------


class RealRunAuthorizationIssue(_StrictFrozenModel):
    code: str = Field(pattern=r"^[A-Z][A-Z0-9_]{2,127}$")
    detail: str = Field(min_length=1, max_length=2048)


class RealRunAuthorization(_StrictFrozenModel):
    schema_version: Literal["easyicu.figure2_realrun_authorization/1"]
    status: Literal["authorized", "blocked"]
    expected_task_ids: tuple[str, ...]
    declaration_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    input_authority_digest: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    issues: tuple[RealRunAuthorizationIssue, ...] = ()

    @model_validator(mode="after")
    def _validate_status(self) -> "RealRunAuthorization":
        if self.expected_task_ids != tuple(FIGURE2_TASK_IDS):
            raise ValueError("authorization must retain exact Canonical9 order")
        if self.status == "authorized":
            if self.issues:
                raise ValueError("authorized receipt cannot carry issues")
            if self.declaration_sha256 is None or self.input_authority_digest is None:
                raise ValueError("authorized receipt must bind declaration + authority")
        elif not self.issues:
            raise ValueError("blocked receipt must carry at least one issue")
        return self


@dataclass(frozen=True)
class RealRunAuthorizationRequest:
    """The declaration file + the actual artifacts + the concrete run intent."""

    declaration_path: Path
    expected_execution_identity_path: Path
    input_freeze_path: Path
    rubric_path: Path
    output_root: Path
    production_input_authority_path: Optional[Path] = None
    resume_run_id: Optional[str] = None
    resume_from_step_id: Optional[str] = None
    cross_run_memory: bool = False


class RealRunAuthorizationBlocked(RuntimeError):
    def __init__(self, authorization: "RealRunAuthorization") -> None:
        codes = ", ".join(issue.code for issue in authorization.issues)
        super().__init__(f"real-run authorization blocked: {codes}")
        self.authorization = authorization


def _issue(code: str, detail: str) -> RealRunAuthorizationIssue:
    return RealRunAuthorizationIssue(code=code, detail=detail)


def _load_declaration(
    path: Path,
) -> tuple[Optional[OperatorFreezeDeclaration], Optional[str], list]:
    issues: list[RealRunAuthorizationIssue] = []
    try:
        safe = _require_safe_path(path)
        raw = _read_regular_file(safe)
        declaration = OperatorFreezeDeclaration.model_validate_json(raw, strict=True)
        return declaration, hashlib.sha256(raw).hexdigest(), issues
    except Exception as exc:  # noqa: BLE001
        issues.append(
            _issue(
                "OPERATOR_DECLARATION_INVALID",
                f"cannot load operator freeze declaration: {type(exc).__name__}: {exc}",
            )
        )
        return None, None, issues


def _verify_fresh_output_root(root: Path, run_id: str) -> list:
    issues: list[RealRunAuthorizationIssue] = []
    try:
        candidate = Path(root).expanduser()
        if not candidate.is_absolute() or candidate.is_symlink():
            raise ValueError("output root must be an absolute, non-symlink path")
        if candidate.exists():
            if not candidate.is_dir():
                raise ValueError("output root exists and is not a directory")
            if any(candidate.iterdir()):
                raise ValueError(
                    "output root is not empty; a fresh run needs a clean root"
                )
        run_dir = candidate / run_id
        if run_dir.exists():
            raise ValueError(
                "run directory already exists; refusing a diagnostic reuse"
            )
    except Exception as exc:  # noqa: BLE001
        issues.append(_issue("OUTPUT_ROOT_NOT_FRESH", str(exc)))
    return issues


def verify_realrun_authorization(
    request: RealRunAuthorizationRequest,
) -> RealRunAuthorization:
    """Fail-closed pre-run authorization; ``authorized`` only if every pin holds."""

    declaration, declaration_sha, issues = _load_declaration(request.declaration_path)

    # Runtime intent (independent of the declaration bytes).
    if request.resume_run_id is not None or request.resume_from_step_id is not None:
        issues.append(
            _issue(
                "NON_FRESH_RUN",
                "a canonical run must be fresh; refusing to resume a diagnostic run",
            )
        )
    if request.cross_run_memory:
        issues.append(
            _issue(
                "CROSS_RUN_MEMORY_ENABLED",
                "cross-run memory / experience bank must be disabled",
            )
        )

    if declaration is None:
        return RealRunAuthorization(
            schema_version=REALRUN_AUTHORIZATION_SCHEMA,
            status="blocked",
            expected_task_ids=tuple(FIGURE2_TASK_IDS),
            issues=tuple(issues),
        )

    issues.extend(_verify_fresh_output_root(request.output_root, declaration.run_id))

    # 1) Execution identity: pin the file bytes, then require paper eligibility.
    identity = None
    try:
        actual_identity_sha = _sha256_of(request.expected_execution_identity_path)
        if actual_identity_sha != declaration.expected_execution_identity_sha256:
            raise ValueError("execution identity file sha differs from the pin")
        identity = ExpectedExecutionIdentity.model_validate_json(
            _read_regular_file(
                _require_safe_path(request.expected_execution_identity_path)
            ),
            strict=True,
        ).execution_identity
        if not identity.paper_eligible:
            raise ValueError("frozen execution identity is not paper eligible")
        if not identity.git_sha or identity.git_dirty is not False:
            raise ValueError("execution identity is not a clean-tree commit")
        if identity.git_sha != declaration.code_commit_sha:
            raise ValueError("execution identity commit differs from the declared pin")
        if identity.runner != declaration.runner:
            raise ValueError("execution identity runner differs from the declared pin")
        if identity.input_authority_sha256 is None:
            raise ValueError("execution identity lacks a bound input authority sha")
        if identity.input_authority_sha256 != declaration.input_authority_digest:
            raise ValueError(
                "execution identity input_authority_sha256 differs from the pin"
            )
    except Exception as exc:  # noqa: BLE001
        issues.append(
            _issue("EXECUTION_IDENTITY_MISMATCH", f"{type(exc).__name__}: {exc}")
        )

    # 2) Input freeze v1: reuse the strict loader; the acknowledged assessment must
    #    authenticate against its pin (it is always blocked, never authorizing).
    try:
        actual_freeze_sha = canonical_input_freeze_manifest_sha256(
            _require_safe_path(request.input_freeze_path)
        )
        if actual_freeze_sha != declaration.input_freeze_manifest_sha256:
            raise ValueError("input freeze manifest sha differs from the pin")
    except Exception as exc:  # noqa: BLE001
        issues.append(_issue("INPUT_FREEZE_INVALID", f"{type(exc).__name__}: {exc}"))

    # 3) Production input authority: the ONLY thing that can authorize the input.
    if request.production_input_authority_path is None:
        issues.append(
            _issue(
                "PRODUCTION_INPUT_AUTHORITY_ABSENT",
                "no typed production input authority supplied; the full-9 input is "
                "not yet frozen for a real run (the v1 assessment stays blocked)",
            )
        )
    else:
        try:
            authority, _ = load_production_input_authority(
                request.production_input_authority_path
            )
            if authority.authority_digest != declaration.input_authority_digest:
                raise ValueError("production authority digest differs from the pin")
            if (
                identity is not None
                and identity.input_authority_sha256 != authority.authority_digest
            ):
                raise ValueError(
                    "execution identity input authority differs from the production "
                    "authority digest"
                )
        except Exception as exc:  # noqa: BLE001
            issues.append(
                _issue(
                    "PRODUCTION_INPUT_AUTHORITY_INVALID",
                    f"{type(exc).__name__}: {exc}",
                )
            )

    # 4) Rubric identity.
    try:
        actual_rubric_sha = _sha256_of(request.rubric_path)
        if actual_rubric_sha != declaration.rubric_sha256:
            raise ValueError("rubric sha differs from the pin")
    except Exception as exc:  # noqa: BLE001
        issues.append(_issue("RUBRIC_IDENTITY_INVALID", f"{type(exc).__name__}: {exc}"))

    status: Literal["authorized", "blocked"] = "authorized" if not issues else "blocked"
    return RealRunAuthorization(
        schema_version=REALRUN_AUTHORIZATION_SCHEMA,
        status=status,
        expected_task_ids=tuple(FIGURE2_TASK_IDS),
        declaration_sha256=declaration_sha,
        input_authority_digest=declaration.input_authority_digest,
        issues=tuple(issues),
    )


def enforce_realrun_authorization(
    request: RealRunAuthorizationRequest,
) -> RealRunAuthorization:
    """Verify and RAISE ``RealRunAuthorizationBlocked`` unless authorized."""

    authorization = verify_realrun_authorization(request)
    if authorization.status != "authorized":
        raise RealRunAuthorizationBlocked(authorization)
    return authorization


def write_realrun_authorization_receipt(
    authorization: RealRunAuthorization, out_path: Path
) -> Path:
    path = Path(out_path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(authorization.model_dump_json(indent=2), encoding="utf-8")
    return path


def _cli(argv: Optional[Sequence[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="figure2-realrun-authority",
        description="Verify (never launch) the pre-run freeze authority. 0=authorized, 2=blocked.",
    )
    parser.add_argument("--declaration", required=True)
    parser.add_argument("--expected-identity", required=True)
    parser.add_argument("--input-freeze", required=True)
    parser.add_argument("--rubric", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--production-input-authority", default=None)
    parser.add_argument("--resume-run-id", default=None)
    parser.add_argument("--resume-from-step-id", default=None)
    parser.add_argument("--enable-cross-run-memory", action="store_true")
    parser.add_argument("--receipt-out", default=None)
    args = parser.parse_args(argv)

    request = RealRunAuthorizationRequest(
        declaration_path=Path(args.declaration),
        expected_execution_identity_path=Path(args.expected_identity),
        input_freeze_path=Path(args.input_freeze),
        rubric_path=Path(args.rubric),
        output_root=Path(args.output_root),
        production_input_authority_path=(
            Path(args.production_input_authority)
            if args.production_input_authority
            else None
        ),
        resume_run_id=args.resume_run_id,
        resume_from_step_id=args.resume_from_step_id,
        cross_run_memory=bool(args.enable_cross_run_memory),
    )
    authorization = verify_realrun_authorization(request)
    if args.receipt_out:
        write_realrun_authorization_receipt(authorization, Path(args.receipt_out))
    print(authorization.model_dump_json(indent=2))
    return 0 if authorization.status == "authorized" else 2


if __name__ == "__main__":
    import sys

    raise SystemExit(_cli(sys.argv[1:]))


__all__ = [
    "REALRUN_AUTHORIZATION_SCHEMA",
    "OPERATOR_FREEZE_DECLARATION_SCHEMA",
    "PRODUCTION_INPUT_AUTHORITY_SCHEMA",
    "AUTHORIZED_ARMS",
    "ProductionInputTask",
    "ProductionInputAuthority",
    "load_production_input_authority",
    "OperatorFreezeDeclaration",
    "RealRunAuthorizationIssue",
    "RealRunAuthorization",
    "RealRunAuthorizationRequest",
    "RealRunAuthorizationBlocked",
    "verify_realrun_authorization",
    "enforce_realrun_authorization",
    "write_realrun_authorization_receipt",
]
