"""Digest-bound selective revalidation of resumed successful steps.

This module owns the append-only replay lifecycle: it rebuilds trusted views from
sealed evidence, reruns deterministic gates, propagates invalidation through
dependency edges, and commits alias retirement only after the checkpoint write.
The execute-phase orchestrator supplies its replaceable gate/checkpoint seams
through :class:`ResumeRevalidationServices`; this module never imports the
execute-phase god module.
"""

from __future__ import annotations

import copy
import json
import re
import shutil
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Set, Tuple

from ..agents.core import CriticAgent
from ..audits.step_summary_integrity import StepSummaryIntegrityValidator
from ..audits.validators import (
    ClinicalConstraintValidator,
    CrossStepCohortLockValidator,
    CrossStepReconciliationTraceValidator,
    CrossStepRegisteredOutputValidator,
    CrossStepSourceStatusValidator,
    FigureContractQualityValidator,
    FigureSourceDataValidator,
    PrimaryModelContractValidator,
    StatisticalGuard,
    StatisticalValidator,
    StepSummaryFractionValidator,
)
from ..authority.plausibility import (
    compile_resumed_flag_only_plausibility_scope,
    restore_revalidated_resolved_inputs_sha256,
)
from ..authority.plan_scope import _serializable_plan_scientific_scope_signature
from ..authority.run_input import (
    RunInputIdentityError,
    _HOST_COHORT_MATERIALIZER_AUTHORITY_KIND,
    _HOST_COHORT_MATERIALIZER_GENERATION_MODE,
    _host_cohort_materializer_authority_error,
    _host_probe_authority_error,
)
from ..authority.runtime_artifacts import (
    current_step_records,
    verified_run_evidence_path,
)
from ..authority.typed_binding import (
    _evidence_record_field,
    _registered_source_name,
    _resume_typed_input_bindings,
    _resume_typed_input_bindings_fingerprint,
    _typed_input_product,
)
from ..contracts.runtime import ValidationFinding
from ..schema import AnalysisPlan, EvidenceRef, ResearchContext
from .final_validation import (
    _FinalDeterministicGateFindings,
    _bind_findings_to_step_attempt,
)


@dataclass(frozen=True)
class ResumeDeterministicRevalidationResult:
    """Append-only resume ledger after selective deterministic replay."""

    resume_state: Dict[str, Any]
    revalidated_step_ids: Tuple[str, ...]
    invalidated_step_ids: Tuple[str, ...]


@dataclass(frozen=True)
class ResumeRevalidationServices:
    """Replaceable execute-layer seams used by deterministic resume replay."""

    deterministic_gate_stamp: Callable[[], Mapping[str, str]]
    evaluate_final_deterministic_gates: Callable[..., _FinalDeterministicGateFindings]
    deterministic_code_gate_findings: Callable[..., Sequence[ValidationFinding]]
    actionable_validator_messages: Callable[..., List[str]]
    write_run_checkpoint: Callable[[Path, Mapping[str, Any]], None]


def _verified_explicit_step_authority(
    *,
    record: Mapping[str, Any],
    field: str,
    expected_kind: str,
    expected_source_name: Optional[str],
    evidence_by_id: Mapping[str, Any],
    run_dir: Path,
) -> Tuple[Any, Path]:
    """Resolve one exact checkpoint authority through owner/path/SHA checks."""

    step_id = str(record.get("step_id") or "").strip()
    evidence_id = str(record.get(field) or "").strip()
    listed = {
        str(value).strip()
        for value in (record.get("evidence_ids") or [])
        if str(value).strip()
    }
    if not evidence_id:
        raise ValueError(f"successful checkpoint is missing required {field}")
    if evidence_id not in listed:
        raise ValueError(f"{field} {evidence_id} is absent from evidence_ids")
    authority = evidence_by_id.get(evidence_id)
    if authority is None:
        raise ValueError(f"{field} references missing evidence {evidence_id}")
    if str(_evidence_record_field(authority, "produced_by_step") or "") != step_id:
        raise ValueError(f"{field} is not owned by step {step_id}")
    actual_kind = str(_evidence_record_field(authority, "kind") or "").lower()
    if actual_kind != expected_kind:
        raise ValueError(
            f"{field} has kind {actual_kind or '<missing>'}, expected {expected_kind}"
        )
    verified_path = verified_run_evidence_path(run_dir, authority)
    if verified_path is None:
        raise ValueError(f"{field} failed path/digest verification")
    source_name = _registered_source_name(authority, verified_path)
    if expected_source_name is not None and source_name != expected_source_name:
        raise ValueError(f"{field} does not name {expected_source_name}")
    return authority, verified_path


def _verified_resume_step_summary(
    *,
    record: Mapping[str, Any],
    evidence_by_id: Mapping[str, Any],
    run_dir: Path,
) -> Dict[str, Any]:
    """Load a summary only from the record's explicit digest-bound evidence."""

    field = (
        "probe_summary_evidence_id"
        if str(record.get("step_id") or "") == "00_probe"
        else "step_summary_evidence_id"
    )
    _, summary_path = _verified_explicit_step_authority(
        record=record,
        field=field,
        expected_kind="statistic",
        expected_source_name=(
            "probe_summary.json"
            if field == "probe_summary_evidence_id"
            else "step_summary.json"
        ),
        evidence_by_id=evidence_by_id,
        run_dir=run_dir,
    )
    try:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError) as exc:
        raise ValueError(f"{field} is not readable JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{field} payload is not an object")
    return payload


def _verify_resume_step_script_lineage(
    *,
    record: Mapping[str, Any],
    evidence_by_id: Mapping[str, Any],
) -> None:
    """Require every sealed non-code output to bind the reviewed script.

    Owner and digest checks alone are insufficient: a mutable checkpoint could
    list a second benign script from the same step and point
    ``script_evidence_id`` at it while retaining outputs produced by the real
    script.  Exact lineage closes that decoy-code path before preflight.
    """

    step_id = str(record.get("step_id") or "").strip()
    script_evidence_id = str(record.get("script_evidence_id") or "").strip()
    if not script_evidence_id:
        raise ValueError("successful checkpoint is missing script_evidence_id")
    for raw_id in record.get("evidence_ids") or []:
        evidence_id = str(raw_id).strip()
        authority = evidence_by_id.get(evidence_id)
        if authority is None:
            raise ValueError(f"listed evidence {evidence_id} is missing")
        owner = str(_evidence_record_field(authority, "produced_by_step") or "")
        if owner != step_id:
            raise ValueError(
                f"listed evidence {evidence_id} belongs to {owner or '<run-level>'}"
            )
        if evidence_id == script_evidence_id:
            if str(_evidence_record_field(authority, "kind") or "").lower() != "code":
                raise ValueError("script_evidence_id does not reference code evidence")
            continue
        bound_script_id = str(
            _evidence_record_field(authority, "script_evidence_id") or ""
        ).strip()
        if bound_script_id != script_evidence_id:
            raise ValueError(
                f"listed evidence {evidence_id} is bound to script "
                f"{bound_script_id or '<missing>'}, not {script_evidence_id}"
            )


_STALE_RESOLVED_INPUT_RECEIPT_FIELDS = (
    "resolved_inputs",
    "resolved_input_bindings",
    "resolved_inputs_path",
    "resolved_inputs_sha256",
    "revalidated_input_bindings_fingerprint",
    "flag_only_plausibility_scope",
)


def _discard_stale_resolved_input_receipts(record: Dict[str, Any]) -> None:
    """Remove mutable or superseded resolved-input receipts in place."""

    for field in _STALE_RESOLVED_INPUT_RECEIPT_FIELDS:
        record.pop(field, None)


def _trusted_resume_success_records(
    *,
    records: Sequence[Mapping[str, Any]],
    evidence_by_id: Mapping[str, Any],
    run_dir: Path,
) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    """Replace mutable checkpoint summaries with explicit evidence payloads."""

    trusted: List[Dict[str, Any]] = []
    errors: Dict[str, str] = {}
    for record in records:
        if str(record.get("status") or "").lower() != "ok":
            continue
        step_id = str(record.get("step_id") or "").strip()
        if (
            str(record.get("generation_mode") or "").strip().lower()
            == _HOST_COHORT_MATERIALIZER_GENERATION_MODE
            and record.get("step_authority_kind")
            == _HOST_COHORT_MATERIALIZER_AUTHORITY_KIND
        ):
            copy = dict(record)
            _discard_stale_resolved_input_receipts(copy)
            trusted.append(copy)
            continue
        try:
            summary = _verified_resume_step_summary(
                record=record,
                evidence_by_id=evidence_by_id,
                run_dir=run_dir,
            )
        except ValueError as exc:
            errors[step_id] = str(exc)
            continue
        copy = dict(record)
        copy["step_summary"] = summary
        # These mutable convenience receipts are never replay authority.
        _discard_stale_resolved_input_receipts(copy)
        trusted.append(copy)
    return trusted, errors


def _materialize_verified_step_output_view(
    *,
    record: Mapping[str, Any],
    evidence_by_id: Mapping[str, Any],
    run_dir: Path,
    destination: Path,
) -> Dict[str, str]:
    """Copy only listed, verified same-step evidence under source filenames."""

    step_id = str(record.get("step_id") or "").strip()
    listed = [
        str(value).strip()
        for value in (record.get("evidence_ids") or [])
        if str(value).strip()
    ]
    if not listed:
        raise ValueError("successful checkpoint has no evidence_ids")
    destination.mkdir(parents=True, exist_ok=False)
    copied: Dict[str, str] = {}
    for evidence_id in listed:
        authority = evidence_by_id.get(evidence_id)
        if authority is None:
            raise ValueError(f"listed evidence {evidence_id} is missing")
        owner = str(_evidence_record_field(authority, "produced_by_step") or "")
        if owner != step_id:
            raise ValueError(
                f"listed evidence {evidence_id} belongs to {owner or '<run-level>'}"
            )
        verified_path = verified_run_evidence_path(run_dir, authority)
        if verified_path is None:
            raise ValueError(
                f"listed evidence {evidence_id} failed digest verification"
            )
        source_name = _registered_source_name(authority, verified_path)
        if (
            not source_name
            or Path(source_name).name != source_name
            or "/" in source_name
            or "\\" in source_name
        ):
            raise ValueError(
                f"listed evidence {evidence_id} has no safe source filename"
            )
        prior_id = copied.get(source_name)
        if prior_id is not None and prior_id != evidence_id:
            raise ValueError(f"multiple listed evidence records claim {source_name}")
        target = destination / source_name
        shutil.copyfile(verified_path, target)
        target.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
        copied[source_name] = evidence_id
    return copied


_REPLAY_SUMMARY_OUTPUT_CONTAINER_KEYS = frozenset(
    {"output_files", "output_artifacts", "outputs", "figure_files"}
)
_REPLAY_SUMMARY_DESCRIPTOR_PATH_KEYS = frozenset({"path", "relative_path", "filename"})
_REPLAY_SUMMARY_DIRECT_FIGURE_KEYS = frozenset({"figure_file", "figure_path"})


def _project_verified_replay_output_paths(
    summary: Mapping[str, Any],
    *,
    materialized_evidence_by_source_name: Mapping[str, str],
) -> Dict[str, Any]:
    """Point one in-memory replay summary at its verified temporary view.

    Historical summaries may contain absolute paths into the original step
    output directory.  Resume revalidation deliberately does not trust those
    mutable files: it copies the checkpoint's digest-verified evidence into a
    temporary output view instead.  Project only path values whose basename
    is backed by exactly one materialized, same-step evidence record.  An
    unmatched absolute path is left intact so containment gates continue to
    fail closed.

    The sealed summary bytes and checkpoint record are never modified; this
    projection exists only for deterministic replay against the temporary
    evidence view.
    """

    source_names = {
        str(name)
        for name in materialized_evidence_by_source_name
        if str(name) and Path(str(name)).name == str(name)
    }

    def project_path(value: str) -> str:
        raw = str(value).strip()
        source_name = Path(raw).name
        return source_name if source_name in source_names else value

    def visit(value: Any, *, output_container: bool = False) -> Any:
        if isinstance(value, Mapping):
            projected: Dict[Any, Any] = {}
            for raw_key, child in value.items():
                key = re.sub(r"[^a-z0-9]+", "_", str(raw_key).strip().lower()).strip(
                    "_"
                )
                starts_output_container = key in _REPLAY_SUMMARY_OUTPUT_CONTAINER_KEYS
                child_is_output = output_container or starts_output_container
                if isinstance(child, str) and (
                    key in _REPLAY_SUMMARY_DIRECT_FIGURE_KEYS
                    or (
                        output_container and key in _REPLAY_SUMMARY_DESCRIPTOR_PATH_KEYS
                    )
                    or starts_output_container
                ):
                    projected[raw_key] = project_path(child)
                elif isinstance(child, str):
                    projected[raw_key] = child
                else:
                    projected[raw_key] = visit(
                        child,
                        output_container=child_is_output,
                    )
            return projected
        if isinstance(value, list):
            return [visit(item, output_container=output_container) for item in value]
        if isinstance(value, tuple):
            return tuple(
                visit(item, output_container=output_container) for item in value
            )
        if isinstance(value, str) and output_container:
            return project_path(value)
        return copy.deepcopy(value)

    return visit(summary)


def _resume_success_dependencies(
    *,
    plan: AnalysisPlan,
    current_records: Sequence[Mapping[str, Any]],
    evidence_by_id: Mapping[str, Any],
) -> Dict[str, Set[str]]:
    """Derive immutable plan/evidence producer edges for invalidation."""

    product_producers: Dict[Tuple[str, str], Set[str]] = {}
    for step in plan.steps:
        for raw_output in step.expected_outputs or []:
            product = _typed_input_product(raw_output)
            if product is not None:
                product_producers.setdefault(product, set()).add(step.step_id)
    dependencies: Dict[str, Set[str]] = {}
    steps_by_id = {step.step_id: step for step in plan.steps}
    for record in current_records:
        step_id = str(record.get("step_id") or "").strip()
        deps = dependencies.setdefault(step_id, set())
        step = steps_by_id.get(step_id)
        if step is not None:
            for raw_input in step.inputs or []:
                product = _typed_input_product(raw_input)
                producers = product_producers.get(product or ("", ""), set())
                if len(producers) == 1:
                    deps.update(producers - {step_id})
        pending = [
            str(value).strip()
            for evidence_id in (record.get("evidence_ids") or [])
            if (authority := evidence_by_id.get(str(evidence_id).strip())) is not None
            for value in (_evidence_record_field(authority, "inputs") or [])
            if str(value).strip()
        ]
        seen: Set[str] = set()
        while pending:
            evidence_id = pending.pop()
            if evidence_id in seen:
                continue
            seen.add(evidence_id)
            authority = evidence_by_id.get(evidence_id)
            if authority is None:
                continue
            owner = str(_evidence_record_field(authority, "produced_by_step") or "")
            if owner and owner != step_id:
                deps.add(owner)
                continue
            pending.extend(
                str(value).strip()
                for value in (_evidence_record_field(authority, "inputs") or [])
                if str(value).strip()
            )
    return dependencies


@dataclass(frozen=True)
class ResumeRevalidationRequest:
    """Immutable inputs for one deterministic resume-revalidation pass."""

    resume_state: Dict[str, Any]
    plan: AnalysisPlan
    context: ResearchContext
    evidence: Any
    run_dir: Path
    cohort_path: Path
    universe_path: Path
    resume_from_step_id: Optional[str]
    development_sample: Optional[Any]
    services: ResumeRevalidationServices


@dataclass
class _ResumeRevalidationLedger:
    """Mutable append-only state owned by one revalidation pass."""

    state: Dict[str, Any]
    history: List[Dict[str, Any]]
    current_records: List[Dict[str, Any]]
    current_successes: List[Dict[str, Any]]
    stale_successes: List[Dict[str, Any]]
    steps_by_id: Dict[str, Any]
    step_order: Dict[str, int]
    stamp: Mapping[str, str]
    evidence_records: List[Any]
    evidence_by_id: Dict[str, Any]
    trusted_summary_errors: Dict[str, str]
    trusted_by_step: Dict[str, Dict[str, Any]]
    current_by_step: Dict[str, Dict[str, Any]]
    dependencies: Dict[str, Set[str]]
    invalidated: Dict[str, str]
    revalidated: List[str]
    invalid_payloads: Dict[str, Dict[str, Any]]
    retirement_records: Dict[str, Mapping[str, Any]]


def _inline_history_checkpoint_payload(
    state: Mapping[str, Any],
) -> Dict[str, Any]:
    """Project hydrated resume authority back to one live checkpoint form.

    The authority loader hydrates an external JSONL history into the in-memory
    ``step_attempt_history`` field but intentionally leaves its digest-bound
    reference present.  A live partial checkpoint must store the inline form
    only; persisting both makes the next resume ambiguous and fail-closed.
    """

    payload = dict(state)
    payload.pop("step_attempt_history_ref", None)
    return payload


def _enforce_resume_cut(
    *,
    resume_from_step_id: Optional[str],
    step_order: Mapping[str, int],
    invalidated: Mapping[str, str],
    message: str,
) -> None:
    """Reject a requested cut that skips any already-invalid upstream step."""

    if not resume_from_step_id or not invalidated:
        return
    cut = step_order.get(resume_from_step_id)
    earlier_invalid = sorted(
        step_id
        for step_id in invalidated
        if cut is not None and step_order.get(step_id, cut) < cut
    )
    if earlier_invalid:
        raise RunInputIdentityError(message + ", ".join(earlier_invalid))


def _prepare_revalidation_ledger(
    request: ResumeRevalidationRequest,
) -> Optional[_ResumeRevalidationLedger]:
    """Merge monotonic history and build trusted replay indexes."""

    resume_state = request.resume_state
    state = _inline_history_checkpoint_payload(resume_state)
    authority_history = [
        dict(record)
        for record in (resume_state.get("per_step_records") or [])
        if isinstance(record, Mapping)
    ]
    saved_attempt_history = [
        dict(record)
        for record in (resume_state.get("step_attempt_history") or [])
        if isinstance(record, Mapping)
    ]
    history = saved_attempt_history or list(authority_history)
    for authority_record in authority_history:
        if authority_record not in history:
            history.append(authority_record)

    current_records = [dict(record) for record in current_step_records(history)]
    current_successes = [
        record
        for record in current_records
        if str(record.get("status") or "").strip().lower() == "ok"
    ]
    steps_by_id = {step.step_id: step for step in request.plan.steps}
    step_order = {
        "00_probe": -1,
        **{step.step_id: index for index, step in enumerate(request.plan.steps)},
    }
    seeded_invalidated = {
        str(record.get("step_id") or "").strip(): (
            "prior checkpoint already lacks current resume authority "
            f"(status={str(record.get('status') or '').strip().lower()})"
        )
        for record in current_records
        if str(record.get("status") or "").strip().lower()
        in {"resume_evidence_invalid", "resume_validator_invalid"}
    }
    _enforce_resume_cut(
        resume_from_step_id=request.resume_from_step_id,
        step_order=step_order,
        invalidated=seeded_invalidated,
        message=(
            "Cannot start resume after an already-invalid upstream authority; "
            "resume at or before: "
        ),
    )

    stamp = request.services.deterministic_gate_stamp()
    stale_successes = [
        record
        for record in current_successes
        if record.get("deterministic_gate_fingerprint")
        != stamp["deterministic_gate_fingerprint"]
    ]
    if not stale_successes and not seeded_invalidated:
        return None

    evidence_records = list(request.evidence.records())
    evidence_by_id = {
        str(_evidence_record_field(record, "evidence_id") or ""): record
        for record in evidence_records
    }
    trusted_records, trusted_summary_errors = _trusted_resume_success_records(
        records=current_successes,
        evidence_by_id=evidence_by_id,
        run_dir=request.run_dir,
    )
    dependencies = _resume_success_dependencies(
        plan=request.plan,
        current_records=current_records,
        evidence_by_id=evidence_by_id,
    )
    return _ResumeRevalidationLedger(
        state=state,
        history=history,
        current_records=current_records,
        current_successes=current_successes,
        stale_successes=stale_successes,
        steps_by_id=steps_by_id,
        step_order=step_order,
        stamp=stamp,
        evidence_records=evidence_records,
        evidence_by_id=evidence_by_id,
        trusted_summary_errors=trusted_summary_errors,
        trusted_by_step={
            str(record.get("step_id") or ""): record for record in trusted_records
        },
        current_by_step={
            str(record.get("step_id") or ""): record for record in current_successes
        },
        dependencies=dependencies,
        invalidated=dict(seeded_invalidated),
        revalidated=[],
        invalid_payloads={},
        retirement_records={},
    )


def _attempt_identity(
    ledger: _ResumeRevalidationLedger,
    step_id: str,
) -> Tuple[str, str]:
    sequence = 1 + sum(
        1
        for record in ledger.history
        if str(record.get("step_id") or "") == step_id
        and record.get("revalidated_without_execution") is True
    )
    attempt_id = f"{step_id}:resume_revalidation:{sequence}"
    return attempt_id, f"{attempt_id}:deterministic_review"


def _indexed_alias_evidence_ids(
    ledger: _ResumeRevalidationLedger,
    prior_record: Mapping[str, Any],
) -> List[str]:
    step_id = str(prior_record.get("step_id") or "").strip()
    indexed_ids: List[str] = []
    for raw_id in prior_record.get("evidence_ids") or []:
        evidence_id = str(raw_id).strip()
        authority = ledger.evidence_by_id.get(evidence_id)
        if (
            authority is not None
            and str(_evidence_record_field(authority, "produced_by_step") or "")
            == step_id
        ):
            indexed_ids.append(evidence_id)
    return list(dict.fromkeys(indexed_ids))


def _seed_recovery_coordinates(ledger: _ResumeRevalidationLedger) -> None:
    """Add monotonic capsule coordinates to legacy invalid checkpoints."""

    for invalid_step_id in tuple(ledger.invalidated):
        prior_success = next(
            (
                record
                for record in reversed(ledger.history)
                if str(record.get("step_id") or "").strip() == invalid_step_id
                and str(record.get("status") or "").strip().lower() == "ok"
            ),
            None,
        )
        if prior_success is None:
            continue
        ledger.retirement_records[invalid_step_id] = prior_success
        current_invalid = next(
            (
                record
                for record in reversed(ledger.history)
                if str(record.get("step_id") or "").strip() == invalid_step_id
                and str(record.get("status") or "").strip().lower()
                in {"resume_evidence_invalid", "resume_validator_invalid"}
            ),
            None,
        )
        raw_capsule_ref = prior_success.get("step_authority_capsule_ref")
        prior_code_sha256 = str(
            prior_success.get("executed_code_sha256")
            or prior_success.get("concept_approved_code_sha256")
            or ""
        )
        if (
            isinstance(current_invalid, Mapping)
            and "resume_revalidation_candidate_capsule_ref" not in current_invalid
            and isinstance(raw_capsule_ref, Mapping)
            and re.fullmatch(r"[0-9a-f]{64}", prior_code_sha256)
        ):
            ledger.history.append(
                {
                    **dict(current_invalid),
                    "attempt_id": (
                        f"{str(current_invalid.get('attempt_id') or invalid_step_id)}"
                        ":candidate_recovery"
                    ),
                    "resume_revalidation_candidate_capsule_ref": dict(raw_capsule_ref),
                    "resume_revalidation_candidate_code_sha256": prior_code_sha256,
                    "resume_revalidation_candidate_attempt_id": str(
                        prior_success.get("attempt_id") or ""
                    ),
                }
            )


def _append_invalid(
    ledger: _ResumeRevalidationLedger,
    *,
    prior_record: Mapping[str, Any],
    reason: str,
    code_findings: Sequence[ValidationFinding] = (),
    gate_findings: Optional[_FinalDeterministicGateFindings] = None,
) -> None:
    """Append one fail-closed invalidation without mutating prior authority."""

    step_id = str(prior_record.get("step_id") or "").strip()
    if step_id in ledger.invalidated:
        return
    attempt_id, checkpoint_id = _attempt_identity(ledger, step_id)
    if not code_findings and gate_findings is None:
        code_findings = _bind_findings_to_step_attempt(
            [
                ValidationFinding(
                    validator="resume_deterministic_revalidation",
                    severity="error",
                    message=(
                        f"Prior success for step {step_id} failed current "
                        "deterministic replay."
                    ),
                    detail={"reason": reason},
                )
            ],
            step_id=step_id,
            attempt_id=attempt_id,
            checkpoint_id=checkpoint_id,
        )
    payload: Dict[str, Any] = {
        "step_id": step_id,
        "status": "resume_validator_invalid",
        "revalidated_without_execution": True,
        "attempt_id": attempt_id,
        "review_checkpoint_id": checkpoint_id,
        "resume_invalidation_reason": reason,
        "invalidated_evidence_ids": list(prior_record.get("evidence_ids") or []),
        "evidence_ids": [],
        "deterministic_code_findings": [
            finding.model_dump(mode="json") for finding in code_findings
        ],
        "retired_current_aliases": {},
        **ledger.stamp,
    }
    raw_capsule_ref = prior_record.get("step_authority_capsule_ref")
    prior_code_sha256 = str(
        prior_record.get("executed_code_sha256")
        or prior_record.get("concept_approved_code_sha256")
        or ""
    )
    if isinstance(raw_capsule_ref, Mapping) and re.fullmatch(
        r"[0-9a-f]{64}", prior_code_sha256
    ):
        payload.update(
            {
                "resume_revalidation_candidate_capsule_ref": dict(raw_capsule_ref),
                "resume_revalidation_candidate_code_sha256": prior_code_sha256,
                "resume_revalidation_candidate_attempt_id": str(
                    prior_record.get("attempt_id") or ""
                ),
            }
        )
    for key, value in prior_record.items():
        if key.startswith("step_provider_call_") or key.startswith("step_llm_repair_"):
            payload[key] = value
    if gate_findings is not None:
        payload.update(
            {
                "stat_findings": [
                    finding.model_dump(mode="json")
                    for finding in gate_findings.stat_findings
                ],
                "clinical_findings": [
                    finding.model_dump(mode="json")
                    for finding in gate_findings.clinical_findings
                ],
                "guard_findings": [
                    finding.model_dump(mode="json")
                    for finding in gate_findings.guard_findings
                ],
                "contract_findings": [
                    finding.model_dump(mode="json")
                    for finding in gate_findings.contract_findings
                ],
                "figure_source_findings": [
                    finding.model_dump(mode="json")
                    for finding in gate_findings.figure_source_findings
                ],
            }
        )
    ledger.invalidated[step_id] = reason
    ledger.invalid_payloads[step_id] = payload
    ledger.retirement_records[step_id] = prior_record
    ledger.history.append(payload)


def _evidence_payloads(ledger: _ResumeRevalidationLedger) -> Dict[str, Dict[str, Any]]:
    return {
        evidence_id: (
            record.model_dump(mode="json")
            if hasattr(record, "model_dump")
            else dict(record)
        )
        for evidence_id, record in ledger.evidence_by_id.items()
    }


def _record_host_replay(
    request: ResumeRevalidationRequest,
    ledger: _ResumeRevalidationLedger,
    *,
    prior_record: Mapping[str, Any],
    step_id: str,
    summary: Mapping[str, Any],
    include_plan_signature: bool,
) -> None:
    attempt_id, checkpoint_id = _attempt_identity(ledger, step_id)
    replayed = {
        **prior_record,
        "status": "ok",
        "step_summary": dict(summary),
        "revalidated_without_execution": True,
        "attempt_id": attempt_id,
        "review_checkpoint_id": checkpoint_id,
        **ledger.stamp,
    }
    if include_plan_signature:
        replayed["plan_scientific_signature"] = (
            _serializable_plan_scientific_scope_signature(request.plan)
        )
    _discard_stale_resolved_input_receipts(replayed)
    ledger.history.append(replayed)
    ledger.trusted_by_step[step_id] = replayed
    ledger.revalidated.append(step_id)


def _revalidate_host_owned_success(
    request: ResumeRevalidationRequest,
    ledger: _ResumeRevalidationLedger,
    *,
    prior_record: Mapping[str, Any],
    step_id: str,
) -> bool:
    """Handle probe/cohort host authority; return whether the record was handled."""

    if step_id == "00_probe":
        summary_error = ledger.trusted_summary_errors.get(step_id)
        if summary_error is not None or step_id not in ledger.trusted_by_step:
            _append_invalid(
                ledger,
                prior_record=prior_record,
                reason=summary_error or "probe summary authority is unavailable",
            )
            return True
        error = _host_probe_authority_error(
            record=prior_record,
            evidence_ids=list(prior_record.get("evidence_ids") or []),
            step_id=step_id,
            run_dir=request.run_dir,
            records=_evidence_payloads(ledger),
        )
        if error is not None:
            _append_invalid(ledger, prior_record=prior_record, reason=error)
            return True
        _record_host_replay(
            request,
            ledger,
            prior_record=prior_record,
            step_id=step_id,
            summary=ledger.trusted_by_step[step_id]["step_summary"],
            include_plan_signature=False,
        )
        return True

    is_host_cohort_materializer = (
        str(prior_record.get("generation_mode") or "").strip().lower()
        == _HOST_COHORT_MATERIALIZER_GENERATION_MODE
        or prior_record.get("step_authority_kind")
        == _HOST_COHORT_MATERIALIZER_AUTHORITY_KIND
    )
    if not is_host_cohort_materializer:
        return False
    error = _host_cohort_materializer_authority_error(
        record=prior_record,
        evidence_ids=list(prior_record.get("evidence_ids") or []),
        step_id=step_id,
        run_dir=request.run_dir,
        records=_evidence_payloads(ledger),
    )
    if error is not None:
        _append_invalid(ledger, prior_record=prior_record, reason=error)
        return True
    _record_host_replay(
        request,
        ledger,
        prior_record=prior_record,
        step_id=step_id,
        summary=prior_record["step_summary"],
        include_plan_signature=True,
    )
    return True


def _revalidate_scientific_success(
    request: ResumeRevalidationRequest,
    ledger: _ResumeRevalidationLedger,
    *,
    prior_record: Mapping[str, Any],
    step_id: str,
) -> None:
    """Replay one agent scientific step against sealed code and output evidence."""

    step = ledger.steps_by_id.get(step_id)
    summary_error = ledger.trusted_summary_errors.get(step_id)
    if step is None or summary_error is not None:
        _append_invalid(
            ledger,
            prior_record=prior_record,
            reason=summary_error or "successful step is absent from active plan",
        )
        return
    trusted_record = ledger.trusted_by_step[step_id]
    attempt_id, checkpoint_id = _attempt_identity(ledger, step_id)
    try:
        _verify_resume_step_script_lineage(
            record=prior_record,
            evidence_by_id=ledger.evidence_by_id,
        )
        _, script_path = _verified_explicit_step_authority(
            record=prior_record,
            field="script_evidence_id",
            expected_kind="code",
            expected_source_name=None,
            evidence_by_id=ledger.evidence_by_id,
            run_dir=request.run_dir,
        )
        script_text = script_path.read_text(encoding="utf-8")
        plausibility_scope = compile_resumed_flag_only_plausibility_scope(
            prior_record=prior_record,
            run_dir=request.run_dir,
            context=request.context,
            step=step,
        )
        code_findings = _bind_findings_to_step_attempt(
            request.services.deterministic_code_gate_findings(
                context=request.context,
                step=step,
                script_text=script_text,
                plausibility_scope=plausibility_scope,
            ),
            step_id=step_id,
            attempt_id=attempt_id,
            checkpoint_id=checkpoint_id,
        )
        if any(finding.severity == "error" for finding in code_findings):
            _append_invalid(
                ledger,
                prior_record=prior_record,
                reason="current deterministic code preflight failed",
                code_findings=code_findings,
            )
            return

        trusted_current_records = [
            record
            for record in ledger.trusted_by_step.values()
            if str(record.get("status") or "").lower() == "ok"
            and str(record.get("step_id") or "") not in ledger.invalidated
        ]
        resolved_bindings, resolved_input_evidence_ids = _resume_typed_input_bindings(
            step=step,
            plan=request.plan,
            evidence_records=ledger.evidence_records,
            trusted_step_records=trusted_current_records,
            run_dir=request.run_dir,
            cohort_path=request.cohort_path,
            development_sample=request.development_sample,
        )
        with tempfile.TemporaryDirectory(
            prefix=f".resume_gate_{step_id}_",
            dir=request.run_dir,
        ) as temporary_root:
            replay_out_dir = Path(temporary_root) / "outputs"
            materialized_outputs = _materialize_verified_step_output_view(
                record=prior_record,
                evidence_by_id=ledger.evidence_by_id,
                run_dir=request.run_dir,
                destination=replay_out_dir,
            )
            replay_step_summary = _project_verified_replay_output_paths(
                trusted_record["step_summary"],
                materialized_evidence_by_source_name=materialized_outputs,
            )
            completed_records = [
                record
                for record in trusted_current_records
                if str(record.get("step_id") or "") != step_id
                and ledger.step_order.get(str(record.get("step_id") or ""), -1)
                < ledger.step_order.get(step_id, len(ledger.step_order))
            ]
            gate_findings = request.services.evaluate_final_deterministic_gates(
                context=request.context,
                plan=request.plan,
                cohort_path=request.cohort_path,
                universe_path=request.universe_path,
                run_dir=request.run_dir,
                out_dir=replay_out_dir,
                step=step,
                step_summary=replay_step_summary,
                step_record=prior_record,
                completed_step_records=completed_records,
                resolved_input_bindings=resolved_bindings,
                plausibility_scope=plausibility_scope,
                script_text=script_text,
                attempt_id=attempt_id,
                checkpoint_id=checkpoint_id,
                evidence_store=request.evidence,
                stat_validator=StatisticalValidator(),
                clinical_validator=ClinicalConstraintValidator(),
                statistical_guard=StatisticalGuard(),
                cross_step_cohort_lock_validator=CrossStepCohortLockValidator(),
                cross_step_registered_output_validator=(
                    CrossStepRegisteredOutputValidator()
                ),
                cross_step_reconciliation_trace_validator=(
                    CrossStepReconciliationTraceValidator()
                ),
                step_summary_integrity_validator=StepSummaryIntegrityValidator(),
                step_summary_fraction_validator=StepSummaryFractionValidator(),
                cross_step_source_status_validator=CrossStepSourceStatusValidator(),
                primary_model_contract_validator=PrimaryModelContractValidator(),
                figure_contract_validator=FigureContractQualityValidator(),
                figure_source_validator=FigureSourceDataValidator(),
            )
        if any(finding.severity == "error" for finding in gate_findings.all_findings()):
            _append_invalid(
                ledger,
                prior_record=prior_record,
                reason="current deterministic artifact gates failed",
                code_findings=code_findings,
                gate_findings=gate_findings,
            )
            return

        prior_critique = prior_record.get("critique_report")
        prior_critique_status = (
            str(prior_critique.get("status") or "").strip().lower()
            if isinstance(prior_critique, Mapping)
            else ""
        )
        if prior_critique_status in {"blocked", "needs_revision"}:
            _append_invalid(
                ledger,
                prior_record=prior_record,
                reason=(
                    f"prior deterministic Critic status remains {prior_critique_status}"
                ),
                code_findings=code_findings,
                gate_findings=gate_findings,
            )
            return
        evidence_refs = [
            EvidenceRef(
                evidence_id=str(_evidence_record_field(authority, "evidence_id")),
                kind=_evidence_record_field(authority, "kind"),
                description=str(_evidence_record_field(authority, "description") or ""),
                relative_path=str(
                    _evidence_record_field(authority, "relative_path") or ""
                ),
            )
            for evidence_id in (prior_record.get("evidence_ids") or [])
            if (authority := ledger.evidence_by_id.get(str(evidence_id))) is not None
            and verified_run_evidence_path(request.run_dir, authority) is not None
        ]
        critique = CriticAgent().review_step(
            step=step,
            step_summary=dict(trusted_record["step_summary"]),
            evidence_refs=evidence_refs,
            findings=request.services.actionable_validator_messages(
                code_findings,
                gate_findings.all_findings(),
            ),
        )
        if critique.status != "pass":
            _append_invalid(
                ledger,
                prior_record=prior_record,
                reason=f"current deterministic Critic status={critique.status}",
                code_findings=code_findings,
                gate_findings=gate_findings,
            )
            return
    except (OSError, TypeError, UnicodeError, ValueError) as exc:
        _append_invalid(
            ledger,
            prior_record=prior_record,
            reason=f"{type(exc).__name__}: {exc}",
        )
        return

    replayed = {
        **prior_record,
        "status": "ok",
        "step_summary": dict(trusted_record["step_summary"]),
        "resolved_input_evidence_ids": resolved_input_evidence_ids,
        "deterministic_code_findings": [
            finding.model_dump(mode="json") for finding in code_findings
        ],
        "stat_findings": [
            finding.model_dump(mode="json") for finding in gate_findings.stat_findings
        ],
        "clinical_findings": [
            finding.model_dump(mode="json")
            for finding in gate_findings.clinical_findings
        ],
        "guard_findings": [
            finding.model_dump(mode="json") for finding in gate_findings.guard_findings
        ],
        "contract_findings": [
            finding.model_dump(mode="json")
            for finding in gate_findings.contract_findings
        ],
        "figure_source_findings": [
            finding.model_dump(mode="json")
            for finding in gate_findings.figure_source_findings
        ],
        "critique_report": critique.model_dump(mode="json"),
        "revalidated_without_execution": True,
        "attempt_id": attempt_id,
        "review_checkpoint_id": checkpoint_id,
        **ledger.stamp,
    }
    _discard_stale_resolved_input_receipts(replayed)
    replayed["resolved_inputs_sha256"] = prior_record["resolved_inputs_sha256"]
    replayed["flag_only_plausibility_scope"] = plausibility_scope.to_dict()
    replayed["revalidated_input_bindings_fingerprint"] = (
        _resume_typed_input_bindings_fingerprint(resolved_bindings)
    )
    ledger.history.append(replayed)
    ledger.trusted_by_step[step_id] = replayed
    ledger.revalidated.append(step_id)


def _revalidate_stale_successes(
    request: ResumeRevalidationRequest,
    ledger: _ResumeRevalidationLedger,
) -> None:
    """Replay stale successes in immutable plan order."""

    ledger.stale_successes.sort(
        key=lambda record: ledger.step_order.get(
            str(record.get("step_id") or ""),
            len(ledger.step_order),
        )
    )
    for saved_record in ledger.stale_successes:
        prior_record = restore_revalidated_resolved_inputs_sha256(
            prior_record=saved_record,
            checkpoint_history=ledger.history,
            run_dir=request.run_dir,
        )
        step_id = str(prior_record.get("step_id") or "").strip()
        invalid_upstream = sorted(
            ledger.dependencies.get(step_id, set()).intersection(ledger.invalidated)
        )
        if invalid_upstream:
            _append_invalid(
                ledger,
                prior_record=prior_record,
                reason=(
                    "current success depends on invalidated upstream step(s): "
                    + ", ".join(invalid_upstream)
                ),
            )
            continue
        if _revalidate_host_owned_success(
            request,
            ledger,
            prior_record=prior_record,
            step_id=step_id,
        ):
            continue
        _revalidate_scientific_success(
            request,
            ledger,
            prior_record=prior_record,
            step_id=step_id,
        )


def _propagate_invalidations(ledger: _ResumeRevalidationLedger) -> None:
    """Propagate invalid authority through immutable plan/evidence edges."""

    while True:
        changed = False
        for step_id, prior_record in ledger.current_by_step.items():
            if step_id in ledger.invalidated:
                continue
            failed_dependencies = sorted(
                ledger.dependencies.get(step_id, set()).intersection(ledger.invalidated)
            )
            if not failed_dependencies:
                continue
            _append_invalid(
                ledger,
                prior_record=prior_record,
                reason=(
                    "current success depends on invalidated upstream step(s): "
                    + ", ".join(failed_dependencies)
                ),
            )
            changed = True
        if not changed:
            return


def _commit_revalidation(
    request: ResumeRevalidationRequest,
    ledger: _ResumeRevalidationLedger,
) -> ResumeDeterministicRevalidationResult:
    """Write the ledger, then atomically retire current aliases or roll back."""

    if ledger.invalid_payloads:
        state_findings = list(request.resume_state.get("findings") or [])
        for step_id, payload in ledger.invalid_payloads.items():
            reason = str(payload.get("resume_invalidation_reason") or "")
            state_findings.append(
                ValidationFinding(
                    validator="resume_deterministic_revalidation",
                    severity="warning",
                    message=(
                        f"Prior success for step {step_id} was invalidated by "
                        "current deterministic gates and requires re-execution."
                    ),
                    detail={
                        "step_id": step_id,
                        "reason": reason,
                        "requires_reexecution": True,
                    },
                ).model_dump(mode="json")
            )
        ledger.state["findings"] = state_findings
    ledger.state["step_attempt_history"] = ledger.history
    ledger.state["per_step_records"] = [
        dict(record) for record in current_step_records(ledger.history)
    ]

    retirement_batch = {
        step_id: evidence_ids
        for step_id, prior_record in ledger.retirement_records.items()
        if (evidence_ids := _indexed_alias_evidence_ids(ledger, prior_record))
    }
    current_aliases = request.evidence.aliases() if retirement_batch else {}
    for step_id, evidence_ids in retirement_batch.items():
        payload = ledger.invalid_payloads.get(step_id)
        if payload is not None:
            payload["retired_current_aliases"] = {
                alias: evidence_id
                for alias, evidence_id in current_aliases.items()
                if evidence_id in set(evidence_ids)
            }

    checkpoint_path = request.run_dir / "manifest_partial.json"
    request.services.write_run_checkpoint(checkpoint_path, ledger.state)
    if retirement_batch:
        try:
            request.evidence.retire_steps_current_aliases(retirement_batch)
        except (KeyError, OSError, TypeError, ValueError) as exc:
            try:
                request.services.write_run_checkpoint(
                    checkpoint_path,
                    _inline_history_checkpoint_payload(request.resume_state),
                )
            except (OSError, TypeError, ValueError) as rollback_exc:
                raise RuntimeError(
                    "resume revalidation alias retirement and manifest rollback "
                    "both failed"
                ) from rollback_exc
            raise RuntimeError(
                "resume revalidation alias retirement failed; manifest was rolled back"
            ) from exc
    return ResumeDeterministicRevalidationResult(
        resume_state=ledger.state,
        revalidated_step_ids=tuple(ledger.revalidated),
        invalidated_step_ids=tuple(sorted(ledger.invalidated)),
    )


def revalidate_resume_successes(
    *,
    resume_state: Dict[str, Any],
    plan: AnalysisPlan,
    context: ResearchContext,
    evidence: Any,
    run_dir: Path,
    cohort_path: Path,
    universe_path: Path,
    resume_from_step_id: Optional[str],
    development_sample: Optional[Any] = None,
    services: ResumeRevalidationServices,
) -> ResumeDeterministicRevalidationResult:
    """Replay changed deterministic gates against sealed evidence only."""

    request = ResumeRevalidationRequest(
        resume_state=resume_state,
        plan=plan,
        context=context,
        evidence=evidence,
        run_dir=run_dir,
        cohort_path=cohort_path,
        universe_path=universe_path,
        resume_from_step_id=resume_from_step_id,
        development_sample=development_sample,
        services=services,
    )
    ledger = _prepare_revalidation_ledger(request)
    if ledger is None:
        return ResumeDeterministicRevalidationResult(dict(resume_state), (), ())
    _seed_recovery_coordinates(ledger)
    _revalidate_stale_successes(request, ledger)
    _propagate_invalidations(ledger)
    _enforce_resume_cut(
        resume_from_step_id=resume_from_step_id,
        step_order=ledger.step_order,
        invalidated=ledger.invalidated,
        message=(
            "Cannot start resume after deterministic-validator-invalid upstream "
            "evidence; resume at or before: "
        ),
    )
    return _commit_revalidation(request, ledger)
