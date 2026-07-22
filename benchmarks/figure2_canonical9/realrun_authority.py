"""Pre-run authorization / freeze receipt for a real Canonical9 ``aware`` run.

This is a **fail-closed pre-run gate**, produced and verified BEFORE a real
``--arms aware`` Canonical9 run is launched.  It does not launch anything, call a
Provider, read patient data, run Docker, or grant paper authority.  It composes
the already-reviewed authority surfaces into one testable receipt so a real run
can only proceed under a fully bound, frozen identity.

It **reuses** (never re-implements) the engine and evaluator authority:

* ``ExpectedExecutionIdentity`` (engine) — the operator-frozen execution
  identity: clean-tree commit (``git_sha`` + ``git_dirty is False``), docker
  runner image digest, non-secret provider authorization, prompt pack, network
  policy, host-fallback prohibition, and the bound ``full6`` input authority
  sha.  ``paper_eligible`` is the engine's own gate.
* ``FIGURE2_TASK_IDS`` (evaluator) — the exact, ordered 9-task coverage that the
  downstream ``evaluate_figure2_paper_acceptance`` gate also requires.
* the ``canonical_input_freeze_v1.json`` document — the ``full6``-derived input
  content authority (submission profile dict SHAs) with **no outstanding
  re-materialize blockers**.

and **adds** the pre-run intent bindings that no single surface covered:

* the arm is exactly ``aware`` (never the implicit/default ``naive`` ablation);
* the run is **fresh** — it refuses to resume a prior diagnostic checkpoint;
* cross-run memory / experience bank is disabled;
* the paper rubric identity is bound.

Every binding is recorded as a SHA-256, never a secret: the provider
authorization is bound through its non-secret manifest digest, so no API key is
ever read or written.

The single operation this receipt does NOT and MUST NOT perform is launching the
real run: that still requires the operator's explicit real-Provider
authorization.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Optional, Sequence

from pydantic import BaseModel, ConfigDict, Field, model_validator

from easyicu.research_agent.authority.execution_identity import (
    ExpectedExecutionIdentity,
)

from .evaluator.rubric_v1 import FIGURE2_TASK_IDS

REALRUN_AUTHORIZATION_SCHEMA = "easyicu.figure2_realrun_authorization/1"
INPUT_FREEZE_SCHEMA = "easyicu.figure2_canonical_input_freeze/1"
AUTHORIZED_ARMS: tuple[str, ...] = ("aware",)
_MAX_DOC_BYTES = 32 * 1024 * 1024


class _StrictFrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class RealRunAuthorizationIssue(_StrictFrozenModel):
    code: str = Field(pattern=r"^[A-Z][A-Z0-9_]{2,127}$")
    detail: str = Field(min_length=1, max_length=2048)
    task_id: str | None = None


class RealRunAuthorization(_StrictFrozenModel):
    """Typed pre-run authorization receipt.  ``authorized`` iff no issues."""

    schema_version: Literal["easyicu.figure2_realrun_authorization/1"]
    status: Literal["authorized", "blocked"]
    arms: tuple[str, ...]
    expected_task_ids: tuple[str, ...]
    requested_task_ids: tuple[str, ...]
    fresh_run: bool
    cross_run_memory_disabled: bool
    output_root: str
    code_git_sha: str | None = None
    expected_execution_identity_sha256: str | None = Field(
        default=None, pattern=r"^[0-9a-f]{64}$"
    )
    expected_execution_identity_freeze_sha256: str | None = Field(
        default=None, pattern=r"^[0-9a-f]{64}$"
    )
    input_authority_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    input_freeze_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    input_freeze_submission_profile_ref: str | None = None
    rubric_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    issues: tuple[RealRunAuthorizationIssue, ...] = ()

    @model_validator(mode="after")
    def _validate_status(self) -> "RealRunAuthorization":
        exact = tuple(FIGURE2_TASK_IDS)
        if self.expected_task_ids != exact:
            raise ValueError("authorization must retain exact Canonical9 order")
        if self.status == "authorized":
            if self.issues:
                raise ValueError("authorized receipt cannot carry issues")
            if self.arms != AUTHORIZED_ARMS:
                raise ValueError("authorized receipt requires exactly the aware arm")
            if not self.fresh_run or not self.cross_run_memory_disabled:
                raise ValueError("authorized receipt requires a fresh, memory-free run")
            if self.requested_task_ids != exact:
                raise ValueError("authorized receipt requires exact task coverage")
            required = (
                self.code_git_sha,
                self.expected_execution_identity_sha256,
                self.expected_execution_identity_freeze_sha256,
                self.input_authority_sha256,
                self.input_freeze_sha256,
                self.input_freeze_submission_profile_ref,
                self.rubric_sha256,
            )
            if any(value is None for value in required):
                raise ValueError("authorized receipt must bind every identity")
        elif not self.issues:
            raise ValueError("blocked receipt must carry at least one issue")
        return self


@dataclass(frozen=True)
class RealRunAuthorizationRequest:
    """Operator intent + frozen document paths for one real Canonical9 run."""

    expected_execution_identity_path: Path
    input_freeze_path: Path
    rubric_path: Path
    output_root: Path
    arms: Sequence[str] = AUTHORIZED_ARMS
    requested_task_ids: Sequence[str] = FIGURE2_TASK_IDS
    fresh_run: bool = True
    cross_run_memory: bool = False
    resume_run_id: Optional[str] = None
    expected_input_freeze_sha256: Optional[str] = None
    expected_rubric_sha256: Optional[str] = None


def _issue(
    code: str, detail: str, *, task_id: str | None = None
) -> RealRunAuthorizationIssue:
    return RealRunAuthorizationIssue(code=code, detail=detail, task_id=task_id)


def _read_regular_file(path: Path) -> bytes:
    descriptor: int | None = None
    try:
        descriptor = os.open(
            path,
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
        )
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError("authorization input must be a regular file")
        if metadata.st_size > _MAX_DOC_BYTES:
            raise ValueError("authorization input exceeds the 32 MiB limit")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 64 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        return b"".join(chunks)
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _strict_json_object(payload: bytes) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    value = json.loads(payload, object_pairs_hook=reject_duplicates)
    if not isinstance(value, dict):
        raise ValueError("document root must be a JSON object")
    return value


def _require_safe_path(path: Path) -> Path:
    resolved = Path(path).expanduser()
    if not resolved.is_absolute() or resolved.is_symlink():
        raise ValueError("path must be an absolute, non-symlink path")
    return resolved.resolve(strict=True)


def _verify_frozen_identity(
    path: Path,
) -> tuple[Optional[ExpectedExecutionIdentity], list[RealRunAuthorizationIssue], dict]:
    issues: list[RealRunAuthorizationIssue] = []
    shas: dict[str, Any] = {}
    try:
        safe_path = _require_safe_path(path)
        frozen = ExpectedExecutionIdentity.model_validate(
            _strict_json_object(_read_regular_file(safe_path)), strict=True
        )
    except Exception as exc:  # noqa: BLE001
        issues.append(
            _issue(
                "EXECUTION_IDENTITY_NOT_FROZEN",
                f"cannot load frozen execution identity: {type(exc).__name__}: {exc}",
            )
        )
        return None, issues, shas

    identity = frozen.execution_identity
    shas["environment_identity_sha256"] = frozen.expected_identity_sha256
    shas["freeze_sha256"] = frozen.freeze_sha256
    shas["git_sha"] = identity.git_sha
    shas["input_authority_sha256"] = identity.input_authority_sha256

    if not identity.git_sha or identity.git_dirty is not False:
        issues.append(
            _issue(
                "CODE_IDENTITY_UNCLEAN",
                "frozen identity must bind a clean-tree commit "
                "(git_sha set, git_dirty False)",
            )
        )
    if identity.input_authority_sha256 is None:
        issues.append(
            _issue(
                "INPUT_AUTHORITY_UNBOUND",
                "frozen identity lacks a bound full6 input authority sha",
            )
        )
    if not identity.paper_eligible:
        issues.append(
            _issue(
                "EXECUTION_IDENTITY_NOT_FROZEN",
                "frozen execution identity is not paper eligible (needs docker "
                "image digest, bound provider, network none, no host fallback)",
            )
        )
    return frozen, issues, shas


def _verify_input_freeze(
    path: Path, expected_sha: Optional[str]
) -> tuple[list[RealRunAuthorizationIssue], Optional[str], Optional[str]]:
    issues: list[RealRunAuthorizationIssue] = []
    try:
        safe_path = _require_safe_path(path)
        raw = _read_regular_file(safe_path)
        freeze_sha = hashlib.sha256(raw).hexdigest()
        doc = _strict_json_object(raw)
    except Exception as exc:  # noqa: BLE001
        issues.append(
            _issue(
                "INPUT_FREEZE_INVALID",
                f"cannot read input freeze: {type(exc).__name__}: {exc}",
            )
        )
        return issues, None, None

    if doc.get("schema_version") != INPUT_FREEZE_SCHEMA:
        issues.append(_issue("INPUT_FREEZE_INVALID", "input freeze schema mismatch"))

    profile = doc.get("submission_profile")
    profile_ref = profile.get("ref") if isinstance(profile, dict) else None
    if not isinstance(profile, dict) or not all(
        profile.get(key) for key in ("ref", "concept_dict_sha256", "sofa2_dict_sha256")
    ):
        issues.append(
            _issue(
                "INPUT_FREEZE_INVALID",
                "input freeze lacks a bound submission profile "
                "(ref + concept/sofa2 dict SHAs)",
            )
        )

    if expected_sha is not None and freeze_sha != expected_sha:
        issues.append(
            _issue(
                "INPUT_FREEZE_IDENTITY_CHANGED",
                "input freeze content sha differs from the frozen expectation",
            )
        )

    cases = doc.get("cases")
    if not isinstance(cases, list):
        issues.append(
            _issue("INPUT_FREEZE_INVALID", "input freeze cases must be a list")
        )
    else:
        for case in cases:
            if not isinstance(case, dict):
                issues.append(
                    _issue(
                        "INPUT_FREEZE_INVALID", "input freeze case must be an object"
                    )
                )
                continue
            if case.get("state") == "blocked" or bool(case.get("blockers")):
                issues.append(
                    _issue(
                        "INPUT_FREEZE_NOT_AUTHORIZED",
                        "input case is still blocked (outstanding re-materialize / "
                        "authority); full6 input is not yet frozen for a real run",
                        task_id=(
                            str(case.get("benchmark_item_id"))
                            if isinstance(case.get("benchmark_item_id"), str)
                            else None
                        ),
                    )
                )
    return issues, freeze_sha, profile_ref


def _verify_rubric(
    path: Path, expected_sha: Optional[str]
) -> tuple[list[RealRunAuthorizationIssue], Optional[str]]:
    issues: list[RealRunAuthorizationIssue] = []
    try:
        safe_path = _require_safe_path(path)
        raw = _read_regular_file(safe_path)
        rubric_sha = hashlib.sha256(raw).hexdigest()
        _strict_json_object(raw)
    except Exception as exc:  # noqa: BLE001
        issues.append(
            _issue(
                "RUBRIC_IDENTITY_INVALID",
                f"cannot read paper rubric: {type(exc).__name__}: {exc}",
            )
        )
        return issues, None
    if expected_sha is not None and rubric_sha != expected_sha:
        issues.append(
            _issue(
                "RUBRIC_IDENTITY_INVALID",
                "paper rubric sha differs from the frozen expectation",
            )
        )
    return issues, rubric_sha


def verify_realrun_authorization(
    request: RealRunAuthorizationRequest,
) -> RealRunAuthorization:
    """Fail-closed pre-run authorization.  ``authorized`` only if every bind holds."""

    issues: list[RealRunAuthorizationIssue] = []

    arms = tuple(str(arm) for arm in request.arms)
    if arms != AUTHORIZED_ARMS:
        issues.append(
            _issue(
                "ARM_AUTHORITY_INVALID",
                f"a real Canonical9 run requires exactly {AUTHORIZED_ARMS}; "
                f"the historical naive ablation is never paper authority (got {arms})",
            )
        )

    requested = tuple(str(task_id) for task_id in request.requested_task_ids)
    if requested != tuple(FIGURE2_TASK_IDS):
        issues.append(
            _issue(
                "TASK_COVERAGE_INVALID",
                "requested task ids are not the exact Canonical9 coverage/order",
            )
        )

    fresh_run = bool(request.fresh_run) and request.resume_run_id is None
    if not fresh_run:
        issues.append(
            _issue(
                "NON_FRESH_RUN",
                "a canonical run must be fresh; refusing to pass a prior diagnostic "
                "checkpoint off as a canonical run",
            )
        )

    memory_disabled = not bool(request.cross_run_memory)
    if not memory_disabled:
        issues.append(
            _issue(
                "CROSS_RUN_MEMORY_ENABLED",
                "cross-run memory / experience bank must be disabled for a canonical run",
            )
        )

    output_root_str = str(request.output_root)
    try:
        candidate = Path(request.output_root).expanduser()
        if not candidate.is_absolute() or candidate.is_symlink():
            raise ValueError("output root must be an absolute, non-symlink path")
        output_root_str = str(candidate)
    except Exception as exc:  # noqa: BLE001
        issues.append(_issue("OUTPUT_ROOT_INVALID", str(exc)))

    _identity, identity_issues, shas = _verify_frozen_identity(
        request.expected_execution_identity_path
    )
    issues.extend(identity_issues)

    freeze_issues, freeze_sha, profile_ref = _verify_input_freeze(
        request.input_freeze_path, request.expected_input_freeze_sha256
    )
    issues.extend(freeze_issues)

    rubric_issues, rubric_sha = _verify_rubric(
        request.rubric_path, request.expected_rubric_sha256
    )
    issues.extend(rubric_issues)

    status: Literal["authorized", "blocked"] = "authorized" if not issues else "blocked"
    return RealRunAuthorization(
        schema_version=REALRUN_AUTHORIZATION_SCHEMA,
        status=status,
        arms=arms,
        expected_task_ids=tuple(FIGURE2_TASK_IDS),
        requested_task_ids=requested,
        fresh_run=fresh_run,
        cross_run_memory_disabled=memory_disabled,
        output_root=output_root_str,
        code_git_sha=shas.get("git_sha"),
        expected_execution_identity_sha256=shas.get("environment_identity_sha256"),
        expected_execution_identity_freeze_sha256=shas.get("freeze_sha256"),
        input_authority_sha256=shas.get("input_authority_sha256"),
        input_freeze_sha256=freeze_sha,
        input_freeze_submission_profile_ref=profile_ref,
        rubric_sha256=rubric_sha,
        issues=tuple(issues),
    )


def write_realrun_authorization_receipt(
    authorization: RealRunAuthorization, out_path: Path
) -> Path:
    """Persist the receipt as a durable, secret-free JSON document."""

    path = Path(out_path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        authorization.model_dump_json(indent=2, by_alias=True), encoding="utf-8"
    )
    return path


def _cli(argv: Optional[Sequence[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="figure2-realrun-authority",
        description=(
            "Verify (do not launch) the pre-run freeze authority for a real "
            "Canonical9 aware run.  Exit 0 = authorized, 2 = blocked."
        ),
    )
    parser.add_argument("--expected-identity", required=True)
    parser.add_argument("--input-freeze", required=True)
    parser.add_argument("--rubric", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--arms", nargs="+", default=list(AUTHORIZED_ARMS))
    parser.add_argument("--task-ids", nargs="+", default=list(FIGURE2_TASK_IDS))
    parser.add_argument("--expected-input-freeze-sha256", default=None)
    parser.add_argument("--expected-rubric-sha256", default=None)
    parser.add_argument("--resume-run-id", default=None)
    parser.add_argument("--enable-cross-run-memory", action="store_true")
    parser.add_argument("--receipt-out", default=None)
    args = parser.parse_args(argv)

    request = RealRunAuthorizationRequest(
        expected_execution_identity_path=Path(args.expected_identity),
        input_freeze_path=Path(args.input_freeze),
        rubric_path=Path(args.rubric),
        output_root=Path(args.output_root),
        arms=args.arms,
        requested_task_ids=args.task_ids,
        fresh_run=args.resume_run_id is None,
        cross_run_memory=bool(args.enable_cross_run_memory),
        resume_run_id=args.resume_run_id,
        expected_input_freeze_sha256=args.expected_input_freeze_sha256,
        expected_rubric_sha256=args.expected_rubric_sha256,
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
    "INPUT_FREEZE_SCHEMA",
    "AUTHORIZED_ARMS",
    "RealRunAuthorizationIssue",
    "RealRunAuthorization",
    "RealRunAuthorizationRequest",
    "verify_realrun_authorization",
    "write_realrun_authorization_receipt",
]
