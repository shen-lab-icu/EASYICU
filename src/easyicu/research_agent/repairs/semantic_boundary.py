"""Fail closed when a source repair would change declared scientific design."""

from __future__ import annotations

from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Mapping,
    MutableMapping,
    MutableSequence,
    Optional,
    Sequence,
)

from ..repair_registry import RepairClass, repair_metadata_for
from ..schema import ValidationFinding


@dataclass(frozen=True, slots=True)
class SemanticRepairEscalation:
    """A deterministic candidate requires replanning or human review."""

    repair_id: str
    source: str
    issue_code: str = "scientific_design_change_requires_replan"
    action: str = "replan_or_human_review"


def _notify(
    repair_ids: Sequence[str],
    *,
    source: str,
    callback: Optional[Callable[[SemanticRepairEscalation], None]],
) -> None:
    if callback is None:
        return
    for repair_id in repair_ids:
        callback(SemanticRepairEscalation(repair_id=repair_id, source=source))


def mechanical_repair_or_escalate(
    candidate: Optional[tuple[str, str]],
    *,
    source: str,
    callback: Optional[Callable[[SemanticRepairEscalation], None]] = None,
) -> Optional[tuple[str, str]]:
    """Return one candidate only when the registry classifies it as mechanical."""

    if candidate is None:
        return None
    repair_id, _ = candidate
    if (
        repair_metadata_for(repair_id).repair_class
        is not RepairClass.METHOD_SUBSTITUTION
    ):
        return candidate
    _notify((repair_id,), source=source, callback=callback)
    return None


def mechanical_repair_batch_or_escalate(
    *,
    original_code: str,
    candidate_code: str,
    repair_ids: Sequence[str],
    source: str,
    callback: Optional[Callable[[SemanticRepairEscalation], None]] = None,
) -> tuple[str, list[str]]:
    """Keep an all-or-nothing repair batch only when every transform is mechanical."""

    semantic_ids = [
        repair_id
        for repair_id in repair_ids
        if repair_metadata_for(repair_id).repair_class
        is RepairClass.METHOD_SUBSTITUTION
    ]
    if not semantic_ids:
        return candidate_code, list(repair_ids)
    _notify(semantic_ids, source=source, callback=callback)
    return original_code, []


@dataclass(slots=True)
class SemanticRepairRecorder:
    """Persist one de-duplicated execution finding per blocked semantic repair."""

    step_record: MutableMapping[str, Any]
    findings: MutableSequence[ValidationFinding]
    lock: Any
    step_id: str
    attempt_id: str

    def __call__(self, escalation: SemanticRepairEscalation) -> None:
        existing = self.step_record.setdefault("semantic_repair_escalations", [])
        if any(
            item.get("repair_id") == escalation.repair_id
            for item in existing
            if isinstance(item, Mapping)
        ):
            return
        detail = {
            "issue_code": escalation.issue_code,
            "repair_id": escalation.repair_id,
            "source": escalation.source,
            "action": escalation.action,
            "step_id": self.step_id,
            "attempt_id": self.attempt_id,
        }
        existing.append(detail)
        with self.lock:
            self.findings.append(
                ValidationFinding(
                    validator="deterministic_repair_boundary",
                    severity="warning",
                    message=(
                        f"Step {self.step_id} matched retired semantic repair "
                        f"{escalation.repair_id}; EasyICU preserved the declared "
                        "estimator, coding, and predictor roster. Replan or request "
                        "human review if the faithful implementation cannot run."
                    ),
                    detail=detail,
                )
            )
