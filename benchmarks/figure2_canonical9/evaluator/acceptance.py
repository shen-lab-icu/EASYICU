"""Fail-closed downstream acceptance gate for a complete Canonical9 batch.

Individual research runs deliberately survive evaluator failures so the batch
can expose every capability gap.  This module is the separate paper-facing
gate: it accepts only one exact, ordered, aware-arm result for every frozen
Canonical9 task and deterministically replays each valid evaluation attempt.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from easyicu.research_agent.authority.execution_identity import (
    ExecutionIdentity,
    ExpectedExecutionIdentity,
)

from .rubric_v1 import FIGURE2_TASK_IDS
from .scoring import Figure2EvaluationAttempt, verify_figure2_evaluation_attempt

FIGURE2_PAPER_ACCEPTANCE_SCHEMA = "easyicu.figure2_paper_acceptance/2"
_MAX_RESULTS_BYTES = 32 * 1024 * 1024


class _StrictFrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class Figure2AcceptanceIssue(_StrictFrozenModel):
    code: str = Field(pattern=r"^[A-Z][A-Z0-9_]{2,127}$")
    detail: str = Field(min_length=1, max_length=2048)
    task_id: str | None = None


class VerifiedFigure2Task(_StrictFrozenModel):
    task_id: str
    run_id: str
    attempt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    tristate: str


class Figure2PaperAcceptance(_StrictFrozenModel):
    schema_version: Literal["easyicu.figure2_paper_acceptance/2"]
    status: Literal["accepted", "invalid"]
    results_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    expected_execution_identity_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    expected_execution_identity_freeze_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    expected_task_ids: tuple[str, ...]
    observed_task_ids: tuple[str, ...]
    verified_tasks: tuple[VerifiedFigure2Task, ...] = ()
    issues: tuple[Figure2AcceptanceIssue, ...] = ()

    @model_validator(mode="after")
    def _validate_status(self) -> "Figure2PaperAcceptance":
        exact = tuple(FIGURE2_TASK_IDS)
        if self.expected_task_ids != exact:
            raise ValueError("acceptance authority must retain exact Canonical9 order")
        if self.status == "accepted":
            if (
                self.issues
                or self.observed_task_ids != exact
                or self.expected_execution_identity_sha256 is None
                or self.expected_execution_identity_freeze_sha256 is None
            ):
                raise ValueError("accepted batch contradicts coverage findings")
            if tuple(row.task_id for row in self.verified_tasks) != exact:
                raise ValueError("accepted batch lacks exact replay-verified coverage")
        elif not self.issues:
            raise ValueError("invalid batch must contain at least one issue")
        return self


def _strict_json_object(payload: bytes) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def reject_nonfinite(token: str) -> None:
        raise ValueError(f"non-finite JSON number: {token}")

    value = json.loads(
        payload,
        object_pairs_hook=reject_duplicates,
        parse_constant=reject_nonfinite,
    )
    if not isinstance(value, dict):
        raise ValueError("benchmark results root must be a JSON object")
    return value


def _read_regular_file(path: Path) -> bytes:
    descriptor: int | None = None
    try:
        descriptor = os.open(
            path,
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
        )
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError("benchmark results must be a regular file")
        if metadata.st_size > _MAX_RESULTS_BYTES:
            raise ValueError("benchmark results exceed the 32 MiB acceptance limit")
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(descriptor, 64 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > _MAX_RESULTS_BYTES:
                raise ValueError("benchmark results grew beyond the acceptance limit")
            chunks.append(chunk)
        return b"".join(chunks)
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _issue(
    code: str,
    detail: str,
    *,
    task_id: str | None = None,
) -> Figure2AcceptanceIssue:
    return Figure2AcceptanceIssue(code=code, detail=detail, task_id=task_id)


def _invalid_unreadable(payload: bytes, detail: str) -> Figure2PaperAcceptance:
    return Figure2PaperAcceptance(
        schema_version=FIGURE2_PAPER_ACCEPTANCE_SCHEMA,
        status="invalid",
        results_sha256=hashlib.sha256(payload).hexdigest(),
        expected_task_ids=tuple(FIGURE2_TASK_IDS),
        observed_task_ids=(),
        issues=(_issue("RESULTS_DOCUMENT_INVALID", detail),),
    )


def evaluate_figure2_paper_acceptance(
    results_path: Path | str,
    *,
    expected_execution_identity_path: Path | str | None = None,
) -> Figure2PaperAcceptance:
    """Verify a completed EHRFlowBench result file as exact Canonical9.

    Structural or replay failures are returned as typed ``invalid`` findings;
    they never erase the per-item outputs that were already produced.
    """

    path = Path(results_path).expanduser()
    try:
        payload_bytes = _read_regular_file(path)
    except Exception as exc:
        return _invalid_unreadable(
            b"",
            f"cannot read benchmark results: {type(exc).__name__}: {exc}",
        )
    results_sha256 = hashlib.sha256(payload_bytes).hexdigest()
    try:
        payload = _strict_json_object(payload_bytes)
    except Exception as exc:
        return _invalid_unreadable(
            payload_bytes,
            f"cannot parse benchmark results: {type(exc).__name__}: {exc}",
        )

    expected = tuple(FIGURE2_TASK_IDS)
    issues: list[Figure2AcceptanceIssue] = []
    frozen_identity: ExpectedExecutionIdentity | None = None
    if expected_execution_identity_path is None:
        issues.append(
            _issue(
                "EXPECTED_EXECUTION_IDENTITY_MISSING",
                "paper acceptance requires a separately frozen execution identity",
            )
        )
    else:
        try:
            raw_expected_path = Path(expected_execution_identity_path).expanduser()
            if not raw_expected_path.is_absolute() or raw_expected_path.is_symlink():
                raise ValueError(
                    "expected identity must be an absolute non-symlink path"
                )
            expected_path = raw_expected_path.resolve(strict=True)
            results_root = path.parent.resolve()
            try:
                expected_path.relative_to(results_root)
            except ValueError:
                pass
            else:
                raise ValueError("expected identity cannot live inside result output")
            frozen_identity = ExpectedExecutionIdentity.model_validate(
                _strict_json_object(_read_regular_file(expected_path)),
                strict=True,
            )
            if not frozen_identity.execution_identity.paper_eligible:
                raise ValueError("frozen execution identity is not paper eligible")
        except Exception as exc:
            issues.append(
                _issue(
                    "EXPECTED_EXECUTION_IDENTITY_INVALID",
                    f"cannot verify frozen identity: {type(exc).__name__}: {exc}",
                )
            )
    raw_items = payload.get("items")
    observed = (
        tuple(raw_items)
        if isinstance(raw_items, list)
        and all(type(task_id) is str for task_id in raw_items)
        else ()
    )
    if observed != expected:
        issues.append(
            _issue(
                "TASK_COVERAGE_INVALID",
                "input task identifiers do not equal exact Canonical9 order",
            )
        )
    if payload.get("arms") != ["aware"]:
        issues.append(
            _issue(
                "ARM_AUTHORITY_INVALID",
                "paper acceptance requires exactly the aware arm",
            )
        )

    raw_pending = payload.get("pending")
    if not isinstance(raw_pending, list):
        issues.append(_issue("PENDING_LEDGER_INVALID", "pending must be a list"))
    else:
        for pending in raw_pending:
            if not isinstance(pending, dict):
                issues.append(
                    _issue("PENDING_LEDGER_INVALID", "pending row must be an object")
                )
                continue
            task_id = pending.get("key")
            issues.append(
                _issue(
                    (
                        "TASK_NOT_COMPLETED"
                        if task_id in expected
                        else "PENDING_LEDGER_INVALID"
                    ),
                    f"batch contains pending row: {pending.get('status', 'unknown')}",
                    task_id=(str(task_id) if type(task_id) is str else None),
                )
            )

    raw_scores = payload.get("scores")
    scores = raw_scores if isinstance(raw_scores, list) else []
    if not isinstance(raw_scores, list):
        issues.append(_issue("SCORE_LEDGER_INVALID", "scores must be a list"))
    by_task: dict[str, list[dict[str, Any]]] = {}
    score_order: list[str] = []
    for score in scores:
        if not isinstance(score, dict) or type(score.get("item_key")) is not str:
            issues.append(_issue("SCORE_LEDGER_INVALID", "score row lacks item_key"))
            continue
        task_id = str(score["item_key"])
        score_order.append(task_id)
        by_task.setdefault(task_id, []).append(score)
    if tuple(score_order) != expected:
        issues.append(
            _issue(
                "SCORE_ORDER_INVALID",
                "scored task order does not equal exact Canonical9 order",
            )
        )

    verified: list[VerifiedFigure2Task] = []
    results_root = path.parent.resolve()
    for task_id in expected:
        task_rows = by_task.get(task_id, [])
        if len(task_rows) != 1:
            issues.append(
                _issue(
                    "TASK_SCORE_CARDINALITY_INVALID",
                    f"expected one score row, found {len(task_rows)}",
                    task_id=task_id,
                )
            )
            continue
        arm = task_rows[0].get("aware")
        if not isinstance(arm, dict):
            issues.append(
                _issue(
                    "AWARE_ARM_MISSING", "aware arm score is missing", task_id=task_id
                )
            )
            continue
        if arm.get("arm") != "aware":
            issues.append(
                _issue(
                    "ARM_AUTHORITY_INVALID",
                    "aware payload discriminator is missing or incorrect",
                    task_id=task_id,
                )
            )
            continue
        if isinstance(task_rows[0].get("naive"), dict):
            issues.append(
                _issue(
                    "UNAUTHORIZED_ARM_PRESENT",
                    "naive arm cannot satisfy paper acceptance",
                    task_id=task_id,
                )
            )
            continue
        workdir = arm.get("workdir")
        run_id = arm.get("run_id")
        if type(workdir) is not str or type(run_id) is not str:
            issues.append(
                _issue(
                    "RUN_IDENTITY_INVALID",
                    "aware arm lacks string workdir/run_id",
                    task_id=task_id,
                )
            )
            continue
        try:
            raw_run_dir = Path(workdir).expanduser()
            if not raw_run_dir.is_absolute() or raw_run_dir.is_symlink():
                raise ValueError("workdir must be an absolute non-symlink path")
            run_dir = raw_run_dir.resolve(strict=True)
            run_dir.relative_to(results_root)
            if run_dir.name != run_id:
                raise ValueError("run_id does not match workdir basename")
            expected_run_dir = (results_root / task_id / "aware" / run_id).resolve()
            if run_dir != expected_run_dir:
                raise ValueError("workdir does not match task/arm/run coordinates")
        except Exception as exc:
            issues.append(
                _issue(
                    "RUN_IDENTITY_INVALID",
                    str(exc),
                    task_id=task_id,
                )
            )
            continue
        try:
            score_identity = ExecutionIdentity.model_validate(
                arm.get("execution_identity"),
                strict=True,
            )
            manifest_payload = _strict_json_object(
                _read_regular_file(run_dir / "manifest.json")
            )
            manifest_identity = ExecutionIdentity.model_validate(
                manifest_payload.get("execution_identity"),
                strict=True,
            )
            if score_identity.identity_sha256 != manifest_identity.identity_sha256:
                raise ValueError("score and run manifest identities differ")
            if not score_identity.paper_eligible:
                raise ValueError("execution identity is not paper eligible")
            if score_identity.runner != "docker":
                raise ValueError("paper acceptance requires the docker runner")
            if score_identity.host_runner_authorized:
                raise ValueError("--allow-host-runner is never paper authority")
            if frozen_identity is None:
                raise ValueError("no valid independently frozen identity")
            if (
                score_identity.environment_identity_sha256
                != frozen_identity.expected_identity_sha256
            ):
                raise ValueError(
                    "run environment identity differs from frozen submission identity"
                )
        except Exception as exc:
            issues.append(
                _issue(
                    "EXECUTION_IDENTITY_INVALID",
                    f"execution identity invalid: {type(exc).__name__}: {exc}",
                    task_id=task_id,
                )
            )
            continue
        raw_attempt = arm.get("figure2_evaluation_attempt")
        if not isinstance(raw_attempt, dict):
            issues.append(
                _issue(
                    "EVALUATION_ATTEMPT_MISSING",
                    "aware arm lacks a Figure 2 evaluation attempt",
                    task_id=task_id,
                )
            )
            continue
        try:
            attempt_bytes = json.dumps(
                raw_attempt,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            ).encode("utf-8")
            attempt = Figure2EvaluationAttempt.model_validate_json(
                attempt_bytes,
                strict=True,
            )
        except Exception as exc:
            issues.append(
                _issue(
                    "EVALUATION_ATTEMPT_INVALID",
                    f"attempt schema invalid: {type(exc).__name__}: {exc}",
                    task_id=task_id,
                )
            )
            continue
        if attempt.status != "valid":
            issues.append(
                _issue(
                    "EVALUATION_ATTEMPT_INVALID",
                    "attempt status is not valid",
                    task_id=task_id,
                )
            )
            continue
        if attempt.task_id != task_id or attempt.run_id != run_id:
            issues.append(
                _issue(
                    "EVALUATION_IDENTITY_MISMATCH",
                    "attempt task/run identity differs from the score row",
                    task_id=task_id,
                )
            )
            continue
        try:
            envelope = verify_figure2_evaluation_attempt(run_dir, attempt)
        except Exception as exc:
            issues.append(
                _issue(
                    "EVALUATION_REPLAY_FAILED",
                    f"deterministic replay failed: {type(exc).__name__}: {exc}",
                    task_id=task_id,
                )
            )
            continue
        verified.append(
            VerifiedFigure2Task(
                task_id=task_id,
                run_id=run_id,
                attempt_sha256=hashlib.sha256(attempt_bytes).hexdigest(),
                tristate=envelope.scorecard.tristate,
            )
        )

    status: Literal["accepted", "invalid"] = "accepted" if not issues else "invalid"
    return Figure2PaperAcceptance(
        schema_version=FIGURE2_PAPER_ACCEPTANCE_SCHEMA,
        status=status,
        results_sha256=results_sha256,
        expected_execution_identity_sha256=(
            frozen_identity.expected_identity_sha256
            if frozen_identity is not None
            else None
        ),
        expected_execution_identity_freeze_sha256=(
            frozen_identity.freeze_sha256 if frozen_identity is not None else None
        ),
        expected_task_ids=expected,
        observed_task_ids=observed,
        verified_tasks=tuple(verified),
        issues=tuple(issues),
    )


__all__ = [
    "FIGURE2_PAPER_ACCEPTANCE_SCHEMA",
    "Figure2AcceptanceIssue",
    "Figure2PaperAcceptance",
    "VerifiedFigure2Task",
    "evaluate_figure2_paper_acceptance",
]
