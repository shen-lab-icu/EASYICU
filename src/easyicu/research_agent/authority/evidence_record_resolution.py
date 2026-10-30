"""Small host authority for resolving and execution-binding evidence records."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from .step_runtime import (
    StepAuthorityRuntimeError,
    load_explicit_executed_success_step_capsule,
)


def _evidence_record_field(record: Any, name: str) -> Any:
    """Read one field from either a persisted mapping or typed record."""

    if isinstance(record, Mapping):
        return record.get(name)
    return getattr(record, name, None)


def current_conflicted_evidence_aliases(
    per_step_records: Sequence[Mapping[str, Any]],
    *,
    current_evidence_ids: set[str],
    aliases: Mapping[str, str],
) -> set[str]:
    """Names claimed by another currently successful, evidenced step.

    Retained aliases keep historical citations stable but do not identify which
    of two concurrent results the author intended. Ignore superseded/failed
    attempts and conflicts whose original binding is no longer current.
    """
    from .runtime_artifacts import current_successful_step_records

    conflicts: set[str] = set()
    for step in current_successful_step_records(per_step_records):
        active_ids = {
            str(item) for item in (step.get("evidence_ids") or ())
            if str(item) in current_evidence_ids
        }
        if not active_ids:
            continue
        retained = step.get("retained_cross_step_aliases")
        if not isinstance(retained, Mapping):
            continue
        for raw_alias, raw_owner_id in retained.items():
            alias, owner_id = str(raw_alias), str(raw_owner_id)
            if owner_id in current_evidence_ids and aliases.get(alias) == owner_id:
                conflicts.add(alias)
    return conflicts


def is_conflicted_evidence_reference(
    name: str, evidence_id: str, *, aliases: Mapping[str, str], conflicts: set[str],
) -> bool:
    """Cover exact aliases and the store's unique-prefix alias fallback."""
    if name == evidence_id:
        return False
    if name in aliases:
        return name in conflicts
    prefix = f"{name}_"
    return any(
        alias.startswith(prefix) and aliases.get(alias) == evidence_id
        for alias in conflicts
    )


def _current_verified_evidence_record(
    evidence_store: Any,
    name: str,
    per_step_records: Sequence[Mapping[str, Any]],
) -> Any:
    """Resolve an alias only when its producer is current and successful."""

    record = evidence_store.get(name)
    if record is None:
        return None
    current_ids = {
        item.evidence_id
        for item in evidence_store.current_verified_records(per_step_records)
    }
    if record.evidence_id not in current_ids:
        return None
    if name == record.evidence_id:
        return record
    if not any(
        isinstance(row, Mapping) and row.get("retained_cross_step_aliases")
        for row in per_step_records
    ):
        return record
    aliases_fn = getattr(evidence_store, "aliases", None)
    if not callable(aliases_fn):
        return None
    aliases = aliases_fn()
    conflicts = current_conflicted_evidence_aliases(
        per_step_records, current_evidence_ids=current_ids, aliases=aliases,
    )
    if is_conflicted_evidence_reference(
        name, record.evidence_id, aliases=aliases, conflicts=conflicts,
    ):
        return None
    return record


@dataclass(frozen=True)
class ExecutedProductEvidenceAuthority:
    """Capsule-derived digest ceiling for one producer's typed evidence."""

    failure: Optional[dict[str, Any]]
    output_sha256s: Optional[frozenset[str]]
    failure_prefix: Mapping[str, Any]

    def path_failure(
        self,
        verified_path: Path,
        *,
        evidence_record: Any,
    ) -> Optional[dict[str, Any]]:
        """Return a typed failure when evidence was not sealed by execution."""

        if self.failure is not None or self.output_sha256s is None:
            return self.failure
        observed_sha256 = hashlib.sha256(verified_path.read_bytes()).hexdigest()
        if observed_sha256 in self.output_sha256s:
            return None
        return {
            **self.failure_prefix,
            "reason": "producer_execution_output_digest_mismatch",
            "evidence_id": str(
                _evidence_record_field(evidence_record, "evidence_id") or ""
            ),
            "observed_sha256": observed_sha256,
            "sealed_output_count": len(self.output_sha256s),
        }


def executed_product_evidence_authority(
    *,
    run_dir: Path,
    producer_id: str,
    producer_record: Mapping[str, Any],
    input_name: str,
    product_fields: Mapping[str, Any],
) -> ExecutedProductEvidenceAuthority:
    """Load an explicit execution capsule without any legacy fallback."""

    prefix = {
        "input": str(input_name),
        **dict(product_fields),
        "producer_step_id": producer_id,
    }
    if producer_record.get("step_authority_capsule_ref") is None:
        return ExecutedProductEvidenceAuthority(None, None, prefix)
    try:
        executed = load_explicit_executed_success_step_capsule(
            run_dir,
            step_id=producer_id,
            record=producer_record,
        )
    except StepAuthorityRuntimeError as exc:
        return ExecutedProductEvidenceAuthority(
            {
                **prefix,
                "reason": "producer_execution_capsule_invalid",
                "detail": str(exc),
            },
            None,
            prefix,
        )
    if executed is None:  # pragma: no cover - explicit ref is checked above
        return ExecutedProductEvidenceAuthority(
            {**prefix, "reason": "producer_execution_capsule_missing"},
            None,
            prefix,
        )
    execution = executed.capsule.execution
    if execution is None:  # pragma: no cover - loader enforces this
        raise RuntimeError("executed capsule loader returned no execution seal")
    return ExecutedProductEvidenceAuthority(
        None,
        frozenset(output.content.sha256 for output in execution.outputs),
        prefix,
    )


__all__ = [
    "ExecutedProductEvidenceAuthority",
    "_current_verified_evidence_record",
    "_evidence_record_field",
    "current_conflicted_evidence_aliases",
    "is_conflicted_evidence_reference",
    "executed_product_evidence_authority",
]
