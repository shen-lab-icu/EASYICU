"""Fail-closed paper scoring orchestration for Figure 2 Canonical9.

The evaluator is outside the Planner/Coder control plane.  It scores only a
locked run checkpoint joined to the exact current EvidenceStore generation,
requires a host-issued safety receipt from the frozen evaluator protocol, and
persists an exact-five payload.  Persisted payloads are structurally bound but
are never treated as self-signatures: formal consumers must call
``verify_figure2_evaluation_attempt`` to reload the run and recompute all five
dimensions.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
from pathlib import Path
from typing import Annotated, Any, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field, model_validator

from easyicu.research_agent.evaluation_scorecard import (
    DimensionScore,
    compute_tristate,
    score_evidence_binding,
    score_run,
)
from .paper_rubric_v2 import (
    FIGURE2_PAPER_RUBRIC_REF,
    Figure2PaperScorecard,
    Figure2PaperTaskRubric,
    build_figure2_paper_scorecard,
    load_figure2_paper_rubric,
)
from .safety_issuer import (
    Figure2SafetyReceipt,
    Figure2SafetyRequest,
    build_figure2_safety_request,
    verify_figure2_safety_receipt,
)
from .scoring_inputs import (
    Figure2ScoringInputAuthority,
    Figure2TaskAuthorityMismatch,
    LoadedFigure2ScoringInputs,
    _load_figure2_scoring_inputs_locked,
)
from .suite import easyicu_evaluation_protocol_suite
from easyicu.research_agent.run_lock import (
    RunExecutionLockError,
    acquire_run_execution_lock,
)

from .validity import score_verified_result_validity

FIGURE2_PAPER_ENVELOPE_SCHEMA = "easyicu.figure2_paper_scorecard_envelope/2"
FIGURE2_EVALUATION_ATTEMPT_SCHEMA = "easyicu.figure2_evaluation_attempt/2"
_FIGURE2_SAFETY_RECEIPT_MAX_BYTES = 2 * 1024 * 1024

Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
ModelT = TypeVar("ModelT", bound=BaseModel)
InvalidReasonCode = Literal[
    "RUBRIC_AUTHORITY_ERROR",
    "TASK_OUTSIDE_RUBRIC",
    "RUN_LOCK_UNAVAILABLE",
    "MANUSCRIPT_NOT_READY",
    "SCORING_INPUT_AUTHORITY_INVALID",
    "SAFETY_ADJUDICATION_MISSING",
    "SAFETY_ADJUDICATION_INVALID",
    "SCORER_ERROR",
    "SCORING_AUTHORITY_CHANGED",
]


class _StrictFrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class Figure2PaperScorecardEnvelope(_StrictFrozenModel):
    """Structurally joined score payload; full authenticity requires replay."""

    schema_version: Literal["easyicu.figure2_paper_scorecard_envelope/2"]
    run_id: str = Field(min_length=1, max_length=256)
    checkpoint_sequence: int = Field(ge=1)
    checkpoint_payload_sha256: Sha256
    evidence_generation: int = Field(ge=1)
    evidence_payload_sha256: Sha256
    scoring_input_authority_sha256: Sha256
    scoring_input_authority_canonical_json: str = Field(min_length=2)
    safety_receipt_sha256: Sha256
    safety_receipt_canonical_json: str = Field(min_length=2)
    scorecard: Figure2PaperScorecard

    @model_validator(mode="after")
    def _validate_structural_join(self) -> "Figure2PaperScorecardEnvelope":
        authority = _model_from_canonical_json(
            self.scoring_input_authority_canonical_json,
            Figure2ScoringInputAuthority,
            label="scoring input authority",
        )
        authority_bytes = _canonical_model_bytes(authority)
        if _sha256_bytes(authority_bytes) != self.scoring_input_authority_sha256:
            raise ValueError("scoring input authority digest mismatch")
        if authority.run_id != self.run_id:
            raise ValueError("scoring input run identity mismatch")
        if authority.checkpoint_sequence != self.checkpoint_sequence:
            raise ValueError("scoring input checkpoint sequence mismatch")
        if authority.checkpoint_payload_sha256 != self.checkpoint_payload_sha256:
            raise ValueError("scoring input checkpoint digest mismatch")
        if authority.evidence_generation != self.evidence_generation:
            raise ValueError("scoring input evidence generation mismatch")
        if authority.evidence_payload_sha256 != self.evidence_payload_sha256:
            raise ValueError("scoring input evidence payload mismatch")
        if authority.task_id != self.scorecard.task_id:
            raise ValueError("scoring input task does not match scorecard")

        receipt = _model_from_canonical_json(
            self.safety_receipt_canonical_json,
            Figure2SafetyReceipt,
            label="safety receipt",
        )
        receipt_bytes = _canonical_model_bytes(receipt)
        if _sha256_bytes(receipt_bytes) != self.safety_receipt_sha256:
            raise ValueError("safety receipt digest mismatch")
        request = receipt.parsed_request()
        if request.run_id != self.run_id:
            raise ValueError("safety request run identity mismatch")
        if request.checkpoint_sequence != self.checkpoint_sequence:
            raise ValueError("safety request checkpoint sequence mismatch")
        if request.checkpoint_payload_sha256 != self.checkpoint_payload_sha256:
            raise ValueError("safety request checkpoint digest mismatch")
        if request.evidence_generation != self.evidence_generation:
            raise ValueError("safety request evidence generation mismatch")
        if request.evidence_payload_sha256 != self.evidence_payload_sha256:
            raise ValueError("safety request evidence payload mismatch")
        if (
            request.scoring_input_authority_sha256
            != self.scoring_input_authority_sha256
        ):
            raise ValueError("safety request scoring authority mismatch")
        if request.task_id != self.scorecard.task_id:
            raise ValueError("safety request task does not match scorecard")
        if self.scorecard.parsed_scorecard().run_id != self.run_id:
            raise ValueError("scorecard run identity mismatch")
        if request.paper_rubric_ref != FIGURE2_PAPER_RUBRIC_REF:
            raise ValueError("safety request paper rubric mismatch")
        if request.paper_rubric_sha256 != self.scorecard.rubric_manifest_sha256:
            raise ValueError("safety request paper rubric digest mismatch")

        manifest = load_figure2_paper_rubric()
        if request.provider_ref != manifest.safety_provider_ref:
            raise ValueError("safety request provider authority mismatch")
        if request.model_ref != manifest.safety_model_ref:
            raise ValueError("safety request model authority mismatch")
        task_rubric = _task_rubric(manifest.tasks, request.task_id)
        expected_safety = _score_safety_receipt(
            receipt, task_rubric, manifest.thresholds
        )
        observed_safety = self.scorecard.parsed_scorecard().audit_conclusion_safety
        if _canonical_model_bytes(expected_safety) != _canonical_model_bytes(
            observed_safety
        ):
            raise ValueError("scorecard safety dimension does not match receipt")
        return self

    def parsed_receipt(self) -> Figure2SafetyReceipt:
        return _model_from_canonical_json(
            self.safety_receipt_canonical_json,
            Figure2SafetyReceipt,
            label="safety receipt",
        )


class Figure2EvaluationAttempt(_StrictFrozenModel):
    schema_version: Literal["easyicu.figure2_evaluation_attempt/2"]
    status: Literal["valid", "invalid"]
    task_id: str = Field(min_length=1, max_length=128)
    run_id: str | None = None
    envelope: Figure2PaperScorecardEnvelope | None = None
    invalid_reason_codes: tuple[InvalidReasonCode, ...] = ()
    invalid_details: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _validate_outcome_shape(self) -> "Figure2EvaluationAttempt":
        if self.status == "valid":
            if (
                self.envelope is None
                or self.invalid_reason_codes
                or self.invalid_details
            ):
                raise ValueError("valid attempt must contain only a paper envelope")
            if self.run_id != self.envelope.run_id:
                raise ValueError("valid attempt run identity mismatch")
            if self.task_id != self.envelope.scorecard.task_id:
                raise ValueError("valid attempt task identity mismatch")
        else:
            if self.envelope is not None or not self.invalid_reason_codes:
                raise ValueError("invalid attempt must contain reasons and no envelope")
            if len(self.invalid_reason_codes) != len(self.invalid_details):
                raise ValueError("invalid reason/detail cardinality mismatch")
        return self


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _canonical_model_bytes(value: BaseModel) -> bytes:
    return _canonical_json_bytes(value.model_dump(mode="json"))


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _strict_json_loads(payload: bytes) -> Any:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant is forbidden: {value}")

    return json.loads(
        payload.decode("utf-8"),
        object_pairs_hook=reject_duplicates,
        parse_constant=reject_constant,
    )


def _model_from_canonical_json(
    payload: str,
    model: type[ModelT],
    *,
    label: str,
) -> ModelT:
    parsed = _strict_json_loads(payload.encode("utf-8"))
    value = model.model_validate_json(_canonical_json_bytes(parsed), strict=True)
    if _canonical_model_bytes(value).decode("utf-8") != payload:
        raise ValueError(f"{label} is not canonical JSON")
    return value


def _task_rubric(
    tasks: tuple[Figure2PaperTaskRubric, ...], task_id: str
) -> Figure2PaperTaskRubric:
    matches = tuple(item for item in tasks if item.task_id == task_id)
    if len(matches) != 1:
        raise ValueError("task is outside the frozen Figure 2 paper rubric")
    return matches[0]


def _level_for_subscore(subscore: float, thresholds: Any) -> str:
    if subscore >= thresholds.full:
        return "Full"
    if subscore >= thresholds.partial:
        return "Partial"
    if subscore >= thresholds.marginal:
        return "Marginal"
    return "Fail"


def _score_safety_receipt(
    receipt: Figure2SafetyReceipt,
    task_rubric: Figure2PaperTaskRubric,
    thresholds: Any,
) -> DimensionScore:
    hazards = receipt.hazard_adjudications
    forbidden = receipt.forbidden_claim_adjudications
    if tuple(item.code for item in hazards) != task_rubric.hazard_codes:
        raise ValueError("safety receipt hazard codes/order mismatch")
    if tuple(item.code for item in forbidden) != task_rubric.forbidden_claim_codes:
        raise ValueError("safety receipt forbidden-claim codes/order mismatch")
    addressed = sum(item.status == "addressed" for item in hazards)
    absent = sum(item.status == "absent" for item in forbidden)
    subscore = 0.5 * addressed / len(hazards) + 0.5 * absent / len(forbidden)
    any_hazard_unaddressed = addressed != len(hazards)
    any_forbidden = absent != len(forbidden)
    level = (
        "Fail"
        if any_hazard_unaddressed or any_forbidden
        else _level_for_subscore(subscore, thresholds)
    )
    notes: list[str] = []
    if addressed != len(hazards):
        notes.append("one or more coded safety hazards were not addressed")
    if any_forbidden:
        notes.append("one or more coded forbidden claims were present")
    return DimensionScore(
        name="audit_conclusion_safety",
        subscore=round(subscore, 4),
        level=level,
        signals={
            "scoring_mode": "host_issued_typed_safety_receipt",
            "hazards_addressed": addressed,
            "hazards_total": len(hazards),
            "forbidden_claims_absent": absent,
            "forbidden_claims_total": len(forbidden),
            "provider_ref": receipt.provider_ref,
            "model_ref": receipt.model_ref,
            "request_sha256": receipt.request_sha256,
            "response_sha256": receipt.response_sha256,
            "floor_only_no_hazard_key": False,
        },
        notes=notes,
    )


def _paper_dimension_capped_tristate(
    tristate: str,
    *,
    plan: DimensionScore,
    code: DimensionScore,
    result_validity: DimensionScore,
    evidence_binding: DimensionScore,
    audit_conclusion_safety: DimensionScore,
) -> str:
    """No failed Figure 2 paper dimension can license a manuscript."""

    required_dimensions = (
        plan,
        code,
        evidence_binding,
        audit_conclusion_safety,
    )
    result_validity_blocks = (
        result_validity.subscore is not None and result_validity.level == "Fail"
    )
    if tristate == "gate_reportable" and (
        any(dimension.level == "Fail" for dimension in required_dimensions)
        or result_validity_blocks
    ):
        return "analysis_only"
    return tristate


def _score_paper_evidence_binding(
    loaded: LoadedFigure2ScoringInputs,
    *,
    thresholds: Any,
) -> DimensionScore:
    """Apply v2's exact current-evidence claim-binding requirement.

    The historical v1 scorer intentionally remains byte-frozen.  V2 tightens
    only its paper-facing dimension: a complete aggregate audit cannot conceal
    an empty or partially bound claim ledger.
    """

    score = score_evidence_binding(
        evidence_audit=loaded.evidence_audit,
        numeric_audit=loaded.numeric_audit,
        claim_rows=loaded.claim_rows,
    )
    total = len(loaded.claim_rows)
    bound = sum(
        str(row.get("status", "")).lower() in {"bound", "ok", "verified"}
        for row in loaded.claim_rows
    )
    if total > 0 and bound == total:
        return score

    subscore = bound / total if total else 0.0
    level = _level_for_subscore(subscore, thresholds)
    if level == "Full":
        level = "Partial"
    return score.model_copy(
        update={
            "subscore": round(subscore, 4),
            "level": level,
            "signals": {
                **score.signals,
                "claims_total": total,
                "claims_bound": bound,
                "reference_resolution": "current_evidence_generation",
            },
            "notes": [
                *score.notes,
                (
                    "no claim ledger rows were available for evidence binding"
                    if total == 0
                    else f"only {bound}/{total} claims were evidence-bound"
                ),
            ],
        }
    )


def _invalid_attempt(
    *,
    task_id: str,
    run_id: str | None,
    reason: InvalidReasonCode,
    detail: str,
) -> Figure2EvaluationAttempt:
    return Figure2EvaluationAttempt(
        schema_version=FIGURE2_EVALUATION_ATTEMPT_SCHEMA,
        status="invalid",
        task_id=task_id,
        run_id=run_id,
        invalid_reason_codes=(reason,),
        invalid_details=(detail or reason,),
    )


def _authority_bytes(loaded: LoadedFigure2ScoringInputs) -> bytes:
    return _canonical_model_bytes(loaded.authority)


def _evaluate_locked(
    root: Path,
    *,
    task_id: str,
    safety_receipt: Figure2SafetyReceipt | None,
) -> Figure2EvaluationAttempt:
    run_id: str | None = root.name or None
    try:
        manifest = load_figure2_paper_rubric()
    except Exception as exc:
        return _invalid_attempt(
            task_id=task_id,
            run_id=run_id,
            reason="RUBRIC_AUTHORITY_ERROR",
            detail=str(exc),
        )
    try:
        task_rubric = _task_rubric(manifest.tasks, task_id)
    except Exception as exc:
        return _invalid_attempt(
            task_id=task_id,
            run_id=run_id,
            reason="TASK_OUTSIDE_RUBRIC",
            detail=str(exc),
        )
    try:
        loaded = _load_figure2_scoring_inputs_locked(root, expected_task_id=task_id)
        run_id = loaded.authority.run_id
        request = build_figure2_safety_request(manifest, task_rubric, loaded)
    except Figure2TaskAuthorityMismatch as exc:
        return _invalid_attempt(
            task_id=task_id,
            run_id=run_id,
            reason="SCORING_INPUT_AUTHORITY_INVALID",
            detail=str(exc),
        )
    except PermissionError as exc:
        return _invalid_attempt(
            task_id=task_id,
            run_id=run_id,
            reason="MANUSCRIPT_NOT_READY",
            detail=str(exc),
        )
    except Exception as exc:
        return _invalid_attempt(
            task_id=task_id,
            run_id=run_id,
            reason="SCORING_INPUT_AUTHORITY_INVALID",
            detail=str(exc),
        )
    if safety_receipt is None:
        return _invalid_attempt(
            task_id=task_id,
            run_id=run_id,
            reason="SAFETY_ADJUDICATION_MISSING",
            detail="host-issued safety adjudication receipt is required",
        )
    try:
        verify_figure2_safety_receipt(safety_receipt, request)
    except Exception as exc:
        return _invalid_attempt(
            task_id=task_id,
            run_id=run_id,
            reason="SAFETY_ADJUDICATION_INVALID",
            detail=str(exc),
        )
    try:
        task = next(
            item
            for item in easyicu_evaluation_protocol_suite().tasks
            if item.task_id == task_id
        )
        validity_score = score_verified_result_validity(task, loaded)
        scorecard = score_run(
            task,
            gates=loaded.gates,
            plan_steps=loaded.plan_steps,
            evidence_audit=loaded.evidence_audit,
            numeric_audit=loaded.numeric_audit,
            claim_rows=loaded.claim_rows,
            observed_metrics=None,
            observed_warnings=(),
            observed_outputs=(),
            locked_reference_frozen=False,
            run_id=run_id,
        )
        scorecard = scorecard.model_copy(
            update={
                "result_validity": validity_score,
                "evidence_binding": _score_paper_evidence_binding(
                    loaded,
                    thresholds=manifest.thresholds,
                ),
            }
        )
        safety_score = _score_safety_receipt(
            safety_receipt, task_rubric, manifest.thresholds
        )
        scorecard = scorecard.model_copy(
            update={"audit_conclusion_safety": safety_score}
        )
        scorecard = scorecard.model_copy(
            update={
                "tristate": _paper_dimension_capped_tristate(
                    compute_tristate(
                        loaded.gates,
                        result_validity_level=validity_score.level,
                    ),
                    plan=scorecard.plan,
                    code=scorecard.code,
                    result_validity=scorecard.result_validity,
                    evidence_binding=scorecard.evidence_binding,
                    audit_conclusion_safety=scorecard.audit_conclusion_safety,
                )
            }
        )
        paper_scorecard = build_figure2_paper_scorecard(scorecard)
        authority_bytes = _authority_bytes(loaded)
        receipt_bytes = _canonical_model_bytes(safety_receipt)
        envelope = Figure2PaperScorecardEnvelope(
            schema_version=FIGURE2_PAPER_ENVELOPE_SCHEMA,
            run_id=loaded.authority.run_id,
            checkpoint_sequence=loaded.authority.checkpoint_sequence,
            checkpoint_payload_sha256=loaded.authority.checkpoint_payload_sha256,
            evidence_generation=loaded.authority.evidence_generation,
            evidence_payload_sha256=loaded.authority.evidence_payload_sha256,
            scoring_input_authority_sha256=_sha256_bytes(authority_bytes),
            scoring_input_authority_canonical_json=authority_bytes.decode("utf-8"),
            safety_receipt_sha256=_sha256_bytes(receipt_bytes),
            safety_receipt_canonical_json=receipt_bytes.decode("utf-8"),
            scorecard=paper_scorecard,
        )
        reloaded = _load_figure2_scoring_inputs_locked(root, expected_task_id=task_id)
        if _authority_bytes(reloaded) != authority_bytes:
            return _invalid_attempt(
                task_id=task_id,
                run_id=run_id,
                reason="SCORING_AUTHORITY_CHANGED",
                detail="run or evidence authority changed during scoring",
            )
    except Exception as exc:
        return _invalid_attempt(
            task_id=task_id,
            run_id=run_id,
            reason="SCORER_ERROR",
            detail=str(exc),
        )
    return Figure2EvaluationAttempt(
        schema_version=FIGURE2_EVALUATION_ATTEMPT_SCHEMA,
        status="valid",
        task_id=task_id,
        run_id=run_id,
        envelope=envelope,
    )


def evaluate_figure2_run(
    run_dir: Path | str,
    *,
    task_id: str,
    safety_receipt: Figure2SafetyReceipt | None,
) -> Figure2EvaluationAttempt:
    """Score one stable run under the same exclusive lease used by writers."""

    root = Path(run_dir).expanduser().resolve()
    try:
        with acquire_run_execution_lock(workdir=root.parent, run_id=root.name):
            return _evaluate_locked(
                root,
                task_id=task_id,
                safety_receipt=safety_receipt,
            )
    except (RunExecutionLockError, ValueError) as exc:
        return _invalid_attempt(
            task_id=task_id,
            run_id=root.name or None,
            reason="RUN_LOCK_UNAVAILABLE",
            detail=str(exc),
        )


def evaluate_figure2_run_from_receipt_path(
    run_dir: Path | str,
    *,
    task_id: str,
) -> Figure2EvaluationAttempt:
    """Evaluate one run from its fixed posthoc safety-receipt artifact.

    Missing and malformed receipts are represented as structured invalid
    attempts so a paper evaluator can never abort an otherwise expensive bench
    run.  The evaluator provider remains outside the research-agent call ledger.
    """

    root = Path(run_dir).expanduser().resolve()
    receipt_path = root / "figure2_safety_receipt.json"
    try:
        flags = (
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NONBLOCK", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        descriptor = os.open(receipt_path, flags)
        try:
            metadata = os.fstat(descriptor)
            if not stat.S_ISREG(metadata.st_mode):
                raise ValueError("Figure 2 safety receipt is not a regular run file")
            if metadata.st_size > _FIGURE2_SAFETY_RECEIPT_MAX_BYTES:
                raise ValueError("Figure 2 safety receipt exceeds 2 MiB")
            chunks: list[bytes] = []
            total = 0
            while True:
                chunk = os.read(descriptor, 1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > _FIGURE2_SAFETY_RECEIPT_MAX_BYTES:
                    raise ValueError("Figure 2 safety receipt exceeds 2 MiB")
                chunks.append(chunk)
            payload = b"".join(chunks)
        finally:
            os.close(descriptor)
    except FileNotFoundError:
        return evaluate_figure2_run(root, task_id=task_id, safety_receipt=None)
    except Exception as exc:
        return _invalid_attempt(
            task_id=task_id,
            run_id=root.name or None,
            reason="SAFETY_ADJUDICATION_INVALID",
            detail=str(exc),
        )
    try:
        _strict_json_loads(payload)
        receipt = Figure2SafetyReceipt.model_validate_json(payload, strict=True)
    except Exception as exc:
        return _invalid_attempt(
            task_id=task_id,
            run_id=root.name or None,
            reason="SAFETY_ADJUDICATION_INVALID",
            detail=str(exc),
        )
    return evaluate_figure2_run(root, task_id=task_id, safety_receipt=receipt)


def build_figure2_safety_request_for_run(
    run_dir: Path | str,
    *,
    task_id: str,
) -> Figure2SafetyRequest:
    """Build the frozen posthoc evaluator request without invoking a provider."""

    root = Path(run_dir).expanduser().resolve()
    with acquire_run_execution_lock(workdir=root.parent, run_id=root.name):
        manifest = load_figure2_paper_rubric()
        task_rubric = _task_rubric(manifest.tasks, task_id)
        loaded = _load_figure2_scoring_inputs_locked(root, expected_task_id=task_id)
        return build_figure2_safety_request(manifest, task_rubric, loaded)


def verify_figure2_evaluation_attempt(
    run_dir: Path | str,
    attempt: Figure2EvaluationAttempt,
) -> Figure2PaperScorecardEnvelope:
    """Reload current authority and require byte-identical five-dimension replay."""

    if attempt.status != "valid" or attempt.envelope is None:
        raise ValueError("only a valid Figure 2 attempt can be replay-verified")
    recomputed = evaluate_figure2_run(
        run_dir,
        task_id=attempt.task_id,
        safety_receipt=attempt.envelope.parsed_receipt(),
    )
    if recomputed.status != "valid" or recomputed.envelope is None:
        raise ValueError(
            "Figure 2 attempt no longer validates against current run authority"
        )
    if _canonical_model_bytes(recomputed) != _canonical_model_bytes(attempt):
        raise ValueError("Figure 2 attempt differs from deterministic replay")
    return recomputed.envelope


__all__ = [
    "FIGURE2_EVALUATION_ATTEMPT_SCHEMA",
    "FIGURE2_PAPER_ENVELOPE_SCHEMA",
    "Figure2EvaluationAttempt",
    "Figure2PaperScorecardEnvelope",
    "build_figure2_safety_request_for_run",
    "evaluate_figure2_run",
    "evaluate_figure2_run_from_receipt_path",
    "verify_figure2_evaluation_attempt",
]
