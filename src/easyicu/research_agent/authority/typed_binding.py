"""Typed-input lineage, binding, and sealed receipt authority.

This module binds Planner-declared typed products to the current verified
producer evidence. It may emit only caller-scoped resolved-input manifests or
host input-binding receipts; evidence promotion, checkpoint selection, provider
calls, repair orchestration, and scientific design remain outside this layer.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

from ..audits.step_summary_integrity import StepSummaryIntegrityValidator
from .coder_authority import HostCoderAuthority
from ..contracts.declared_product import (
    RUNTIME_BINDABLE_TYPED_INPUT_KINDS,
    RUNTIME_TYPED_INPUT_EVIDENCE_KINDS,
    merge_host_table_contract,
    typed_product_binding_contract,
    typed_product_schema_receipt,
    typed_product as _canonical_typed_product,
)
from ..contracts.artifact_consumption import (
    ArtifactConsumptionError,
    verify_artifact_consumption,
)
from ..authority.evidence_store import sha256_of_file
from ..authority.development_projection import (
    DEVELOPMENT_PRIMARY_COHORT_CONFIRMATION_ROLE,
    resolve_development_input_projection,
)
from ..authority.run_input import canonical_sha256
from ..authority.runtime_artifacts import (
    current_step_records,
    verified_run_evidence_path,
)
from ..schema import AnalysisPlan, AnalysisStep, EvidenceRef
from .plan_scope import (
    _serializable_plan_scientific_scope_signature,
    _step_scientific_signature,
)
from .typed_schema_prompt import (
    typed_parent_schema_context_block as _typed_parent_schema_context_block,
)

__all__ = [
    "TypedBindingResolver",
    "_EvidenceLineageResolutionError",
    "_assignment_model_authority_context_block",
    "_coder_authority_with_typed_parent_schema_receipts",
    "_declared_typed_artifact_paths",
    "_declared_typed_product_paths",
    "_evidence_kind_matches_typed_product",
    "_evidence_record_field",
    "_current_verified_evidence_record",
    "_lineage_failure_product_fields",
    "_normalise_typed_product_name",
    "_registered_source_name",
    "_resolve_typed_artifact_evidence",
    "_resolve_typed_input_evidence",
    "_resolved_typed_input_binding",
    "_resume_typed_input_bindings",
    "_resume_typed_input_bindings_fingerprint",
    "_step_summary_statistic_values",
    "_typed_artifact_name",
    "_typed_input_product",
    "_typed_parent_schema_context_block",
    "_write_host_input_binding_receipts",
    "_write_resolved_inputs_manifest",
]

_RESUME_TYPED_INPUT_BINDING_FINGERPRINT_SCHEMA_VERSION = (
    "easyicu.resume_typed_input_bindings/2"
)


def _attach_verified_consumption_contract(
    *,
    step: AnalysisStep,
    input_name: str,
    binding: Dict[str, Any],
) -> Dict[str, Any]:
    contracts = [
        contract
        for contract in step.input_consumption_contracts
        if contract.input_key == input_name
    ]
    if not contracts:
        return binding
    if len(contracts) != 1:  # schema validation already prevents this
        raise ArtifactConsumptionError("ambiguous input consumption contract")
    updated = dict(binding)
    updated["consumption_contract"] = verify_artifact_consumption(
        contract=contracts[0],
        binding=binding,
    )
    return updated


class _EvidenceLineageResolutionError(RuntimeError):
    """A typed plan input could not be bound to current verified evidence."""

    def __init__(self, failures: Sequence[Mapping[str, Any]]) -> None:
        self.failures = [dict(failure) for failure in failures]
        super().__init__(
            "; ".join(
                f"{failure.get('input')}: {failure.get('reason')}"
                for failure in self.failures
            )
        )


def _evidence_kind_matches_typed_product(
    record: Any,
    typed_product: Tuple[str, str],
) -> bool:
    evidence_kind = str(_evidence_record_field(record, "kind") or "").strip().lower()
    return evidence_kind in RUNTIME_TYPED_INPUT_EVIDENCE_KINDS.get(
        typed_product[0], frozenset()
    )


def _normalise_typed_product_name(value: Any) -> str:
    parsed = _canonical_typed_product(f"artifact:{value}")
    return parsed[1] if parsed is not None else ""


def _producer_output_is_already_development_scoped(
    *,
    evidence_id: str,
    producer_step_records: Sequence[Mapping[str, Any]],
    sample_sha256: str,
) -> bool:
    """Recognise an exact step-owned child of the current development sample.

    The child remains the physical input because replacing it with the base
    sample would discard the producer's deterministic cohort transformations.
    Every authority coordinate is required; ambiguous or legacy records fall
    back to the ordinary host-owned projection path.
    """

    if not evidence_id or not sample_sha256:
        return False
    for step_record in reversed(list(producer_step_records)):
        evidence_ids = step_record.get("evidence_ids")
        if not isinstance(evidence_ids, Sequence) or isinstance(
            evidence_ids, (str, bytes)
        ):
            continue
        if (
            str(step_record.get("status") or "") == "ok"
            and evidence_id in {str(value) for value in evidence_ids}
            and step_record.get("paper_authority") is False
            and str(step_record.get("execution_cohort_role") or "")
            == DEVELOPMENT_PRIMARY_COHORT_CONFIRMATION_ROLE
            and str(step_record.get("execution_cohort_sha256") or "") == sample_sha256
            and str(step_record.get("authoritative_analysis_cohort_sha256") or "")
            == sample_sha256
        ):
            return True
    return False


def _typed_input_product(value: Any) -> Optional[Tuple[str, str]]:
    """Return a canonical ``(kind, product)`` for a typed plan dependency."""

    parsed = _canonical_typed_product(value)
    if parsed is None or parsed[0] not in RUNTIME_BINDABLE_TYPED_INPUT_KINDS:
        return None
    return parsed


def _typed_artifact_name(value: Any) -> Optional[str]:
    """Backward-compatible artifact-only view of a typed plan dependency."""

    typed_product = _typed_input_product(value)
    if typed_product is None or typed_product[0] != "artifact":
        return None
    return typed_product[1]


def _evidence_record_field(record: Any, name: str) -> Any:
    if isinstance(record, Mapping):
        return record.get(name)
    return getattr(record, name, None)


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
    return record if record.evidence_id in current_ids else None


def _registered_source_name(record: Any, verified_path: Path) -> Optional[str]:
    """Recover the registered source name from ``<evidence_id>__<filename>``."""

    evidence_id = str(_evidence_record_field(record, "evidence_id") or "")
    prefix = f"{evidence_id}__"
    if not evidence_id or not verified_path.name.startswith(prefix):
        return None
    return verified_path.name[len(prefix) :] or None


def _declared_typed_product_paths(
    step_summary: Any,
    *,
    typed_product: Tuple[str, str],
) -> Tuple[bool, List[str]]:
    """Return exact typed file mappings declared by the producer summary."""

    if not isinstance(step_summary, Mapping):
        return False, []
    declared = False
    paths: List[str] = []
    for container_name in ("output_files", "outputs"):
        container = step_summary.get(container_name)
        if isinstance(container, Mapping):
            for typed_key, value in container.items():
                if _typed_input_product(typed_key) != typed_product:
                    continue
                declared = True
                if isinstance(value, str) and value.strip():
                    paths.append(value.strip())
                elif isinstance(value, (list, tuple)):
                    paths.extend(
                        str(item).strip()
                        for item in value
                        if isinstance(item, str) and item.strip()
                    )
        elif isinstance(container, (list, tuple)):
            for item in container:
                if not isinstance(item, Mapping):
                    continue
                kind = item.get("kind") or item.get("product_type")
                name = item.get("name")
                named_product = _typed_input_product(name)
                kind_product = _typed_input_product(f"{kind}:placeholder")
                if named_product is not None:
                    descriptor_product = (
                        named_product
                        if kind_product is not None
                        and named_product[0] == kind_product[0]
                        else None
                    )
                else:
                    descriptor_product = _typed_input_product(f"{kind}:{name}")
                if descriptor_product != typed_product:
                    continue
                declared = True
                value = next(
                    (
                        item.get(key)
                        for key in ("path", "relative_path", "filename")
                        if isinstance(item.get(key), str) and str(item.get(key)).strip()
                    ),
                    None,
                )
                if isinstance(value, str) and value.strip():
                    paths.append(value.strip())
                elif isinstance(value, (list, tuple)):
                    paths.extend(
                        str(path).strip()
                        for path in value
                        if isinstance(path, str) and path.strip()
                    )
    return declared, list(dict.fromkeys(paths))


def _declared_typed_artifact_paths(
    step_summary: Any,
    *,
    artifact_name: str,
) -> Tuple[bool, List[str]]:
    """Backward-compatible wrapper for artifact-specific callers."""

    return _declared_typed_product_paths(
        step_summary,
        typed_product=("artifact", artifact_name),
    )


def _lineage_failure_product_fields(
    typed_product: Tuple[str, str],
) -> Dict[str, str]:
    kind, product_name = typed_product
    fields = {"kind": kind, "product": product_name}
    if kind == "artifact":
        fields["artifact"] = product_name
    return fields


def _step_summary_statistic_values(
    step_summary: Any,
    statistic_name: str,
) -> List[float]:
    """Return finite scalar values bound to one exact statistic name."""

    values: List[float] = []

    def _append(value: Any) -> None:
        if isinstance(value, bool) or isinstance(value, (Mapping, list, tuple)):
            return
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return
        if math.isfinite(numeric):
            values.append(numeric)

    def _walk(value: Any) -> None:
        if isinstance(value, Mapping):
            declared_name = value.get("name") or value.get("statistic")
            if declared_name is not None and (
                _normalise_typed_product_name(declared_name) == statistic_name
            ):
                for result_key in ("value", "estimate", "result"):
                    if result_key in value:
                        _append(value[result_key])
            for key, nested in value.items():
                if (
                    _normalise_typed_product_name(key) == statistic_name
                    and nested is not None
                    and not isinstance(nested, (Mapping, list, tuple))
                ):
                    _append(nested)
                _walk(nested)
        elif isinstance(value, (list, tuple)):
            for item in value:
                _walk(item)

    _walk(step_summary)
    return values


def _named_structured_statistic_payload(
    payload: Any,
    statistic_name: str,
) -> bool:
    """Return true for an exact named statistic carrying finite numeric data."""

    if not isinstance(payload, Mapping):
        return False
    declared_name = payload.get("name") or payload.get("statistic")
    if _normalise_typed_product_name(declared_name) != statistic_name:
        return False

    def _has_finite_number(value: Any) -> bool:
        if isinstance(value, bool):
            return False
        if isinstance(value, (int, float)):
            return math.isfinite(float(value))
        if isinstance(value, Mapping):
            return any(_has_finite_number(item) for item in value.values())
        if isinstance(value, (list, tuple)):
            return any(_has_finite_number(item) for item in value)
        return False

    return _has_finite_number(payload)


def _resolve_typed_input_evidence(
    *,
    input_name: str,
    plan: AnalysisPlan,
    evidence_records: Sequence[Any],
    per_step_records: Sequence[Mapping[str, Any]],
    run_dir: Path,
) -> Tuple[Optional[EvidenceRef], Optional[Dict[str, Any]]]:
    """Resolve one typed input through the current execution authority.

    The plan declaration identifies the producer; the latest outer step record
    authorizes its evidence ids.  Basename aliases are deliberately excluded:
    they are first-write-wins and can still point at a superseded resume
    artifact.  Every candidate must instead be owned by the successful current
    producer and pass the registered path/SHA check.
    """

    typed_product = _typed_input_product(input_name)
    if typed_product is None:
        return None, {"input": str(input_name), "reason": "invalid_typed_input"}
    product_fields = _lineage_failure_product_fields(typed_product)

    producer_ids = {
        str(step.step_id)
        for step in plan.steps
        if any(
            _typed_input_product(output) == typed_product
            for output in (step.expected_outputs or [])
        )
    }
    if not producer_ids:
        return None, {
            "input": str(input_name),
            **product_fields,
            "reason": "producer_not_declared",
        }
    if len(producer_ids) != 1:
        return None, {
            "input": str(input_name),
            **product_fields,
            "reason": "ambiguous_producer",
            "producer_step_ids": sorted(producer_ids),
        }

    producer_id = next(iter(producer_ids))
    latest_by_step = {
        str(record.get("step_id") or ""): record
        for record in current_step_records(per_step_records)
    }
    producer_record = latest_by_step.get(producer_id)
    producer_status = str((producer_record or {}).get("status") or "").lower()
    if producer_status != "ok":
        return None, {
            "input": str(input_name),
            **product_fields,
            "reason": "producer_not_successful",
            "producer_step_id": producer_id,
            "producer_status": producer_status or "missing",
        }

    active_producer_step = next(
        step for step in plan.steps if str(step.step_id) == producer_id
    )
    analysis_request = (producer_record or {}).get("analysis_request")
    executed_step_payload = (
        analysis_request.get("step") if isinstance(analysis_request, Mapping) else None
    )
    if not isinstance(executed_step_payload, Mapping):
        return None, {
            "input": str(input_name),
            **product_fields,
            "reason": "producer_plan_snapshot_missing",
            "producer_step_id": producer_id,
        }
    try:
        executed_step = AnalysisStep.model_validate(executed_step_payload)
    except (TypeError, ValueError):
        return None, {
            "input": str(input_name),
            **product_fields,
            "reason": "producer_plan_snapshot_invalid",
            "producer_step_id": producer_id,
        }
    if _step_scientific_signature(executed_step) != _step_scientific_signature(
        active_producer_step
    ):
        return None, {
            "input": str(input_name),
            **product_fields,
            "reason": "producer_plan_snapshot_mismatch",
            "producer_step_id": producer_id,
        }

    recorded_scope_signature = (producer_record or {}).get("plan_scientific_signature")
    if not isinstance(recorded_scope_signature, (list, tuple)):
        return None, {
            "input": str(input_name),
            **product_fields,
            "reason": "producer_plan_scope_snapshot_missing",
            "producer_step_id": producer_id,
        }
    if list(recorded_scope_signature) != (
        _serializable_plan_scientific_scope_signature(plan)
    ):
        return None, {
            "input": str(input_name),
            **product_fields,
            "reason": "producer_plan_scope_snapshot_mismatch",
            "producer_step_id": producer_id,
        }

    active_ids = {
        str(evidence_id)
        for evidence_id in (producer_record or {}).get("evidence_ids", [])
        if str(evidence_id).strip()
    }
    if typed_product[0] == "statistic":
        step_summary = (producer_record or {}).get("step_summary")
        typed_mapping_declared, declared_paths = _declared_typed_product_paths(
            step_summary,
            typed_product=typed_product,
        )
        if typed_mapping_declared:
            if len(declared_paths) != 1:
                return None, {
                    "input": str(input_name),
                    **product_fields,
                    "reason": (
                        "typed_mapping_not_verified"
                        if not declared_paths
                        else "ambiguous_typed_mapping"
                    ),
                    "producer_step_id": producer_id,
                    "declared_paths": declared_paths,
                }
            declared_filename = Path(declared_paths[0]).name
            candidates: List[Tuple[Any, Path]] = []
            incompatible_evidence_kinds: Set[str] = set()
            for record in evidence_records:
                evidence_id = str(_evidence_record_field(record, "evidence_id") or "")
                if (
                    evidence_id not in active_ids
                    or str(_evidence_record_field(record, "produced_by_step") or "")
                    != producer_id
                ):
                    continue
                verified_path = verified_run_evidence_path(run_dir, record)
                if verified_path is None:
                    continue
                if _registered_source_name(record, verified_path) != declared_filename:
                    continue
                if not _evidence_kind_matches_typed_product(record, typed_product):
                    incompatible_evidence_kinds.add(
                        str(_evidence_record_field(record, "kind") or "missing")
                    )
                    continue
                candidates.append((record, verified_path))
            if len(candidates) != 1:
                return None, {
                    "input": str(input_name),
                    **product_fields,
                    "reason": (
                        "evidence_kind_mismatch"
                        if incompatible_evidence_kinds and not candidates
                        else (
                            "typed_mapping_not_verified"
                            if not candidates
                            else "ambiguous_current_artifact"
                        )
                    ),
                    "producer_step_id": producer_id,
                    "declared_path": declared_paths[0],
                    **(
                        {
                            "declared_kind": typed_product[0],
                            "observed_evidence_kinds": sorted(
                                incompatible_evidence_kinds
                            ),
                        }
                        if incompatible_evidence_kinds and not candidates
                        else {}
                    ),
                }
            record, verified_path = candidates[0]
            try:
                evidence_payload = json.loads(verified_path.read_text(encoding="utf-8"))
            except (AttributeError, OSError, TypeError, ValueError):
                evidence_payload = None
            if not _named_structured_statistic_payload(
                evidence_payload,
                typed_product[1],
            ):
                return None, {
                    "input": str(input_name),
                    **product_fields,
                    "reason": "statistic_evidence_value_missing",
                    "producer_step_id": producer_id,
                }
            evidence_values = sorted(
                set(
                    _step_summary_statistic_values(
                        evidence_payload,
                        typed_product[1],
                    )
                )
            )
            if len(evidence_values) > 1:
                return None, {
                    "input": str(input_name),
                    **product_fields,
                    "reason": "statistic_evidence_value_ambiguous",
                    "producer_step_id": producer_id,
                    "evidence_values": evidence_values,
                }
            recorded_values = sorted(
                set(
                    _step_summary_statistic_values(
                        step_summary,
                        typed_product[1],
                    )
                )
            )
            if len(recorded_values) > 1:
                return None, {
                    "input": str(input_name),
                    **product_fields,
                    "reason": "statistic_record_value_ambiguous",
                    "producer_step_id": producer_id,
                    "recorded_values": recorded_values,
                }
            if (
                recorded_values
                and evidence_values
                and not math.isclose(
                    recorded_values[0],
                    evidence_values[0],
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                )
            ):
                return None, {
                    "input": str(input_name),
                    **product_fields,
                    "reason": "statistic_evidence_payload_mismatch",
                    "producer_step_id": producer_id,
                    "recorded_value": recorded_values[0],
                    "evidence_value": evidence_values[0],
                }
            return (
                EvidenceRef(
                    evidence_id=str(
                        _evidence_record_field(record, "evidence_id") or ""
                    ),
                    kind=_evidence_record_field(record, "kind"),
                    description=_evidence_record_field(record, "description"),
                    relative_path=_evidence_record_field(record, "relative_path"),
                ),
                None,
            )
        recorded_values = _step_summary_statistic_values(
            step_summary,
            typed_product[1],
        )
        recorded_unique_values = sorted(set(recorded_values))
        if not recorded_unique_values:
            return None, {
                "input": str(input_name),
                **product_fields,
                "reason": "statistic_not_materialized",
                "producer_step_id": producer_id,
            }
        if len(recorded_unique_values) != 1:
            return None, {
                "input": str(input_name),
                **product_fields,
                "reason": "statistic_record_value_ambiguous",
                "producer_step_id": producer_id,
                "recorded_values": recorded_unique_values,
            }
        step_summary_evidence_id = str(
            (producer_record or {}).get("step_summary_evidence_id") or ""
        )
        candidates: List[Any] = []
        incompatible_evidence_kinds: Set[str] = set()
        for record in evidence_records:
            evidence_id = str(_evidence_record_field(record, "evidence_id") or "")
            if (
                evidence_id != step_summary_evidence_id
                or evidence_id not in active_ids
                or str(_evidence_record_field(record, "produced_by_step") or "")
                != producer_id
                or verified_run_evidence_path(run_dir, record) is None
            ):
                continue
            if not _evidence_kind_matches_typed_product(record, typed_product):
                incompatible_evidence_kinds.add(
                    str(_evidence_record_field(record, "kind") or "missing")
                )
                continue
            candidates.append(record)
        if len(candidates) != 1:
            return None, {
                "input": str(input_name),
                **product_fields,
                "reason": (
                    "evidence_kind_mismatch"
                    if incompatible_evidence_kinds and not candidates
                    else "no_verified_current_statistic"
                ),
                "producer_step_id": producer_id,
                "step_summary_evidence_id": step_summary_evidence_id or None,
                **(
                    {
                        "declared_kind": typed_product[0],
                        "observed_evidence_kinds": sorted(incompatible_evidence_kinds),
                    }
                    if incompatible_evidence_kinds and not candidates
                    else {}
                ),
            }
        record = candidates[0]
        verified_summary_path = verified_run_evidence_path(run_dir, record)
        try:
            evidence_summary = json.loads(
                verified_summary_path.read_text(encoding="utf-8")
            )
        except (AttributeError, OSError, TypeError, ValueError):
            return None, {
                "input": str(input_name),
                **product_fields,
                "reason": "statistic_evidence_payload_invalid",
                "producer_step_id": producer_id,
            }
        if not isinstance(evidence_summary, Mapping):
            return None, {
                "input": str(input_name),
                **product_fields,
                "reason": "statistic_evidence_payload_not_mapping",
                "producer_step_id": producer_id,
            }
        evidence_values = _step_summary_statistic_values(
            evidence_summary,
            typed_product[1],
        )
        evidence_unique_values = sorted(set(evidence_values))
        if not evidence_unique_values:
            return None, {
                "input": str(input_name),
                **product_fields,
                "reason": "statistic_evidence_value_missing",
                "producer_step_id": producer_id,
            }
        if len(evidence_unique_values) != 1:
            return None, {
                "input": str(input_name),
                **product_fields,
                "reason": "statistic_evidence_value_ambiguous",
                "producer_step_id": producer_id,
                "evidence_values": evidence_unique_values,
            }
        recorded_value = recorded_unique_values[0]
        evidence_value = evidence_unique_values[0]
        if not math.isclose(
            recorded_value,
            evidence_value,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            return None, {
                "input": str(input_name),
                **product_fields,
                "reason": "statistic_evidence_payload_mismatch",
                "producer_step_id": producer_id,
                "recorded_value": recorded_value,
                "evidence_value": evidence_value,
            }
        return (
            EvidenceRef(
                evidence_id=str(_evidence_record_field(record, "evidence_id") or ""),
                kind=_evidence_record_field(record, "kind"),
                description=_evidence_record_field(record, "description"),
                relative_path=_evidence_record_field(record, "relative_path"),
            ),
            None,
        )
    typed_mapping_declared, declared_paths = _declared_typed_product_paths(
        (producer_record or {}).get("step_summary"),
        typed_product=typed_product,
    )
    if typed_mapping_declared and not declared_paths:
        return None, {
            "input": str(input_name),
            **product_fields,
            "reason": "typed_mapping_not_verified",
            "producer_step_id": producer_id,
        }
    if len(declared_paths) > 1:
        return None, {
            "input": str(input_name),
            **product_fields,
            "reason": "ambiguous_typed_mapping",
            "producer_step_id": producer_id,
            "declared_paths": declared_paths,
        }
    declared_filename = Path(declared_paths[0]).name if declared_paths else None

    candidates: List[Tuple[Any, Path]] = []
    matching_current_ids: List[str] = []
    incompatible_evidence_kinds: Set[str] = set()
    for record in evidence_records:
        evidence_id = str(_evidence_record_field(record, "evidence_id") or "")
        if (
            evidence_id not in active_ids
            or str(_evidence_record_field(record, "produced_by_step") or "")
            != producer_id
        ):
            continue
        verified_path = verified_run_evidence_path(run_dir, record)
        if verified_path is None:
            continue
        source_name = _registered_source_name(record, verified_path)
        if source_name is None:
            continue
        if declared_filename is not None:
            matches_product = source_name == declared_filename
        else:
            matches_product = (
                _normalise_typed_product_name(source_name) == typed_product[1]
            )
        if not matches_product:
            continue
        if not _evidence_kind_matches_typed_product(record, typed_product):
            incompatible_evidence_kinds.add(
                str(_evidence_record_field(record, "kind") or "missing")
            )
            continue
        matching_current_ids.append(evidence_id)
        candidates.append((record, verified_path))

    if not candidates:
        return None, {
            "input": str(input_name),
            **product_fields,
            "reason": (
                "evidence_kind_mismatch"
                if incompatible_evidence_kinds
                else (
                    "typed_mapping_not_verified"
                    if declared_filename is not None
                    else "no_verified_current_artifact"
                )
            ),
            "producer_step_id": producer_id,
            **(
                {
                    "declared_kind": typed_product[0],
                    "observed_evidence_kinds": sorted(incompatible_evidence_kinds),
                }
                if incompatible_evidence_kinds
                else {}
            ),
            **(
                {"declared_path": declared_paths[0]}
                if declared_filename is not None
                else {}
            ),
        }
    if len(candidates) != 1:
        return None, {
            "input": str(input_name),
            **product_fields,
            "reason": "ambiguous_current_artifact",
            "producer_step_id": producer_id,
            "evidence_ids": sorted(matching_current_ids),
        }

    record, _ = candidates[0]
    return (
        EvidenceRef(
            evidence_id=str(_evidence_record_field(record, "evidence_id") or ""),
            kind=_evidence_record_field(record, "kind"),
            description=_evidence_record_field(record, "description"),
            relative_path=_evidence_record_field(record, "relative_path"),
        ),
        None,
    )


def _resolve_typed_artifact_evidence(
    *,
    input_name: str,
    plan: AnalysisPlan,
    evidence_records: Sequence[Any],
    per_step_records: Sequence[Mapping[str, Any]],
    run_dir: Path,
) -> Tuple[Optional[EvidenceRef], Optional[Dict[str, Any]]]:
    """Compatibility wrapper preserving the public artifact resolver."""

    if _typed_artifact_name(input_name) is None:
        return None, {"input": str(input_name), "reason": "invalid_artifact_input"}
    return _resolve_typed_input_evidence(
        input_name=input_name,
        plan=plan,
        evidence_records=evidence_records,
        per_step_records=per_step_records,
        run_dir=run_dir,
    )


def _resolved_typed_input_binding(
    *,
    input_name: str,
    evidence_ref: EvidenceRef,
    evidence_records: Sequence[Any],
    run_dir: Path,
    producer_step_records: Sequence[Mapping[str, Any]] = (),
    authoritative_cohort_path: Optional[Path] = None,
    development_sample: Optional[Any] = None,
) -> Optional[Dict[str, Any]]:
    """Build the exact, digest-verified runtime binding for one typed input."""

    typed_product = _typed_input_product(input_name)
    if typed_product is None:
        return None
    record = next(
        (
            candidate
            for candidate in evidence_records
            if str(_evidence_record_field(candidate, "evidence_id") or "")
            == evidence_ref.evidence_id
        ),
        None,
    )
    if record is None:
        return None
    if not _evidence_kind_matches_typed_product(record, typed_product):
        return None
    parent_verified_path = verified_run_evidence_path(run_dir, record)
    if parent_verified_path is None:
        return None
    parent_evidence_id = evidence_ref.evidence_id
    parent_sha256 = str(_evidence_record_field(record, "sha256") or "")
    parent_produced_by_step = str(
        _evidence_record_field(record, "produced_by_step") or ""
    )
    execution_projection: Optional[Dict[str, Any]] = None
    projection = None
    verified_path = parent_verified_path
    selected_record = record
    declared_kind, product_name = typed_product
    parent_already_development_scoped = (
        development_sample is not None
        and product_name == "analysis_cohort"
        and _producer_output_is_already_development_scoped(
            evidence_id=evidence_ref.evidence_id,
            producer_step_records=producer_step_records,
            sample_sha256=str(getattr(development_sample, "sample_sha256", "") or ""),
        )
    )
    if (
        development_sample is not None
        and product_name == "analysis_cohort"
        and not parent_already_development_scoped
    ):
        projection = resolve_development_input_projection(
            product_name=product_name,
            parent_evidence_id=parent_evidence_id,
            parent_sha256=parent_sha256,
            parent_produced_by_step=parent_produced_by_step,
            evidence_records=evidence_records,
            run_dir=run_dir,
            authoritative_cohort_path=authoritative_cohort_path,
            development_sample=development_sample,
        )
        if projection is None:
            return None
        selected_record = projection.evidence_record
        verified_path = projection.verified_path
        execution_projection = dict(projection.authority_payload)
    run_root = Path(run_dir).resolve()
    try:
        run_relative_path = verified_path.relative_to(run_root).as_posix()
    except ValueError:
        return None
    binding = {
        "evidence_id": str(
            _evidence_record_field(selected_record, "evidence_id") or ""
        ),
        "declared_kind": declared_kind,
        "product": product_name,
        "evidence_kind": str(_evidence_record_field(selected_record, "kind") or ""),
        "relative_path": run_relative_path,
        "absolute_path": str(verified_path),
        "sha256": str(_evidence_record_field(selected_record, "sha256") or ""),
        # The Planner product remains owned by its declared producer.  The
        # host-owned projection below records who selected the non-paper child.
        "produced_by_step": parent_produced_by_step,
    }
    if execution_projection is not None:
        binding["execution_projection"] = execution_projection
    producer_contract: Optional[Dict[str, Any]] = None
    for step_record in reversed(list(producer_step_records)):
        if str(step_record.get("status") or "") != "ok":
            continue
        evidence_ids = {str(value) for value in (step_record.get("evidence_ids") or [])}
        if evidence_ref.evidence_id not in evidence_ids:
            continue
        step_summary = step_record.get("step_summary")
        if not isinstance(step_summary, Mapping):
            break
        product_contract = typed_product_binding_contract(
            product_name=product_name,
            step_summary=step_summary,
            artifact_path=parent_verified_path,
            authoritative_cohort_path=authoritative_cohort_path,
        )
        if product_contract is not None:
            producer_contract = dict(product_contract)
        break
    contract_required = product_name in {
        "assignment_model",
        "exposure_definition",
        "primary_exposure_definition",
        "prespecified_confounder_set",
    }
    if contract_required and producer_contract is None:
        return None
    identity_row = {
        "input_key": str(input_name),
        "declared_kind": declared_kind,
        "product": product_name,
        "evidence_id": binding["evidence_id"],
        "sha256": binding["sha256"],
        "produced_by_step": binding["produced_by_step"],
    }
    if binding["evidence_kind"] == "table":
        # Table schema v2/v3 is representation-only: producer-authored role prose
        # must not become a second scientific authority. Use the verified physical
        # kind because dataset/cohort/generic artifacts may resolve to a bound table.
        schema_receipt = typed_product_schema_receipt(
            artifact_path=verified_path,
            expected_sha256=binding["sha256"],
        )
        if schema_receipt is None:
            return None
        host_contract = merge_host_table_contract(producer_contract, schema_receipt)
        if projection is not None:
            host_contract.update(projection.row_identity_contract)
    else:
        # Non-table typed products retain their pre-existing contract and
        # version. This patch adds physical table schema facts only.
        host_contract = dict(producer_contract or {})
        host_contract["schema_version"] = "easyicu.host_typed_product.v1"
    host_contract.update(
        {
            "identity_row": identity_row,
        }
    )
    try:
        host_contract_size = len(
            json.dumps(
                host_contract,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )
    except (TypeError, ValueError):
        return None
    if host_contract_size > 128 * 1024:
        return None
    binding["identity_row"] = identity_row
    binding["product_contract"] = host_contract
    return binding


def _assignment_model_authority_context_block(
    bindings: Mapping[str, Mapping[str, Any]],
) -> str:
    """Render exact typed assignment-model identities needed during repair."""

    receipts: dict[str, object] = {}
    model_fields = (
        "model_id",
        "analysis_set",
        "fit_status",
        "propensity_score_column",
        "weight_column",
        "row_identity_column",
        "analysis_set_n",
        "analysis_set_identity_sha256",
    )
    contract_fields = (
        "row_identity_column",
        "row_count",
        "row_identity_sha256",
        "authoritative_cohort_sha256",
        "diagnostic_model_id",
        "selected_model_id",
    )
    for input_key in sorted(bindings):
        if _typed_input_product(input_key) != ("artifact", "assignment_model"):
            continue
        binding = bindings[input_key]
        contract = binding.get("product_contract")
        if not isinstance(contract, Mapping):
            continue
        models = contract.get("models")
        if not isinstance(models, list) or not models:
            continue
        roster = [
            {
                field: model.get(field)
                for field in model_fields
                if model.get(field) is not None
            }
            for model in models
            if isinstance(model, Mapping)
        ]
        if not roster:
            continue
        receipts[input_key] = {
            "evidence_id": binding.get("evidence_id"),
            "sha256": binding.get("sha256"),
            "contract": {
                **{
                    field: contract.get(field)
                    for field in contract_fields
                    if contract.get(field) is not None
                },
                "models": roster,
            },
        }
    if not receipts:
        return ""
    return (
        "HOST-VERIFIED ASSIGNMENT MODEL ROSTER (binding facts only):\n"
        + json.dumps(
            receipts,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\nPreserve every declared model_id, analysis_set, propensity/weight "
        "column, row-identity digest, and denominator. Do not select the first "
        "model, refit the roster, or merge analysis sets."
    )


def _coder_authority_with_typed_parent_schema_receipts(
    *,
    authority: HostCoderAuthority,
    bindings: Mapping[str, Mapping[str, Any]],
) -> HostCoderAuthority:
    """Attach a bounded view of exact typed-parent schemas out of band."""

    authority = authority.append(_typed_parent_schema_context_block(bindings))
    return authority.append(_assignment_model_authority_context_block(bindings))


def _write_resolved_inputs_manifest(
    *,
    run_dir: Path,
    step_id: str,
    planner_declared_inputs: Sequence[str],
    bindings: Mapping[str, Mapping[str, Any]],
    context_path: Optional[Path] = None,
    raw_input_contracts: Optional[Mapping[str, Any]] = None,
) -> Path:
    """Persist the step's authority capsule outside its writable overlay."""

    safe_step_id = str(step_id or "")
    if (
        not safe_step_id
        or safe_step_id in {".", ".."}
        or Path(safe_step_id).name != safe_step_id
        or "/" in safe_step_id
        or "\\" in safe_step_id
    ):
        raise ValueError("step_id must be a single safe path component")
    declared_inputs: List[str] = []
    seen_declared_inputs: Set[str] = set()
    for item in planner_declared_inputs:
        if not isinstance(item, str) or not item.strip():
            raise ValueError(
                "planner_declared_inputs must contain only non-empty strings"
            )
        if item in seen_declared_inputs:
            raise ValueError("planner_declared_inputs must not contain duplicates")
        seen_declared_inputs.add(item)
        declared_inputs.append(item)
    declared_typed_inputs = {
        item for item in declared_inputs if _typed_input_product(item) is not None
    }
    binding_keys = set(bindings)
    if any(not isinstance(key, str) for key in binding_keys) or (
        binding_keys != declared_typed_inputs
    ):
        raise ValueError(
            "resolved input bindings must be exact Planner-declared typed inputs"
        )
    manifest_dir = Path(run_dir).resolve() / "resolved_inputs"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / f"{safe_step_id}.json"
    payload: Dict[str, Any] = {
        "schema_version": "2.1",
        "step_id": safe_step_id,
        "planner_declared_inputs": declared_inputs,
        "inputs": {key: dict(value) for key, value in bindings.items()},
    }
    if raw_input_contracts is not None:
        raw_payload = dict(raw_input_contracts)
        contracts = raw_payload.get("contracts")
        declared_raw_inputs = {item for item in declared_inputs if ":" not in item}
        if (
            raw_payload.get("schema_version")
            != "easyicu.resolved_raw_input_contracts/1"
            or not isinstance(contracts, Mapping)
            or set(contracts) != declared_raw_inputs
        ):
            raise ValueError(
                "raw input contracts must exactly match Planner-declared raw inputs"
            )
        declared_digest = str(raw_payload.pop("contracts_sha256", "") or "")
        if declared_digest != canonical_sha256(raw_payload):
            raise ValueError("raw input contract digest mismatch")
        raw_payload["contracts_sha256"] = declared_digest
        payload["raw_input_contracts"] = raw_payload
    if context_path is not None:
        resolved_context = Path(context_path).resolve()
        run_root = Path(run_dir).resolve()
        if not resolved_context.is_file():
            raise ValueError("context_path must name an existing context file")
        try:
            relative_context = resolved_context.relative_to(run_root).as_posix()
        except ValueError as exc:
            raise ValueError("context_path must be contained by run_dir") from exc
        payload["context"] = {
            "relative_path": relative_context,
            "absolute_path": str(resolved_context),
            "sha256": sha256_of_file(resolved_context),
        }
    temporary_path = manifest_path.with_suffix(".json.tmp")
    temporary_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(manifest_path)
    return manifest_path


def _write_host_input_binding_receipts(
    *,
    out_dir: Path,
    step_summary: Mapping[str, Any],
    resolved_input_bindings: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Any]:
    """Seal exact input receipts for a host-owned deterministic renderer.

    A sealed renderer consumes only the host-resolved artifacts authorized by
    its parent digest seal.  Generated code must normally report its own
    receipts, but asking a host-owned renderer to manufacture those receipts is
    both redundant and weaker than recording them here from the authority
    bindings.  An unreadable table is deliberately omitted so the downstream
    integrity validator fails closed on incomplete coverage.
    """

    receipts: List[Dict[str, Any]] = []
    for input_key, raw_binding in sorted(resolved_input_bindings.items()):
        if not isinstance(raw_binding, Mapping):
            continue
        binding = dict(raw_binding)
        path = Path(str(binding.get("absolute_path") or ""))
        receipt: Dict[str, Any] = {
            "input_key": str(input_key),
            "loaded": True,
        }
        for field in ("evidence_id", "sha256"):
            value = binding.get(field)
            if value is not None:
                receipt[field] = value
        if StepSummaryIntegrityValidator._is_tabular_binding(binding):
            try:
                receipt["row_count"] = StepSummaryIntegrityValidator._table_row_count(
                    path
                )
            except Exception:
                continue
        receipts.append(receipt)

    updated = dict(step_summary)
    updated["input_bindings"] = receipts
    summary_path = out_dir / "step_summary.json"
    temporary_path = summary_path.with_suffix(".json.tmp")
    temporary_path.write_text(
        json.dumps(updated, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(summary_path)
    return updated


def _resume_typed_input_bindings(
    *,
    step: AnalysisStep,
    plan: AnalysisPlan,
    evidence_records: Sequence[Any],
    trusted_step_records: Sequence[Mapping[str, Any]],
    run_dir: Path,
    cohort_path: Path,
    development_sample: Optional[Any] = None,
) -> Tuple[Dict[str, Dict[str, Any]], List[str]]:
    """Rebuild typed bindings without reading mutable resolved-input receipts."""

    bindings: Dict[str, Dict[str, Any]] = {}
    evidence_ids: List[str] = []
    for raw_input in step.inputs or []:
        input_name = str(raw_input)
        if _typed_input_product(input_name) is None:
            continue
        ref, failure = _resolve_typed_input_evidence(
            input_name=input_name,
            plan=plan,
            evidence_records=evidence_records,
            per_step_records=trusted_step_records,
            run_dir=run_dir,
        )
        if failure is not None or ref is None:
            reason = failure or {"reason": "verified_reference_unavailable"}
            raise ValueError(
                f"typed input {input_name} could not be resolved: "
                + json.dumps(reason, sort_keys=True, default=str)
            )
        binding = _resolved_typed_input_binding(
            input_name=input_name,
            evidence_ref=ref,
            evidence_records=evidence_records,
            run_dir=run_dir,
            producer_step_records=trusted_step_records,
            authoritative_cohort_path=cohort_path,
            development_sample=development_sample,
        )
        if binding is None:
            raise ValueError(f"typed input {input_name} has no verified host binding")
        try:
            binding = _attach_verified_consumption_contract(
                step=step,
                input_name=input_name,
                binding=binding,
            )
        except ArtifactConsumptionError as exc:
            raise ValueError(
                f"typed input {input_name} violates its consumption contract: {exc}"
            ) from exc
        bindings[input_name] = binding
        evidence_ids.append(ref.evidence_id)
        projected_evidence_id = str(binding.get("evidence_id") or "")
        if projected_evidence_id:
            evidence_ids.append(projected_evidence_id)
    return bindings, list(dict.fromkeys(evidence_ids))


def _resume_typed_input_bindings_fingerprint(
    bindings: Mapping[str, Mapping[str, Any]],
) -> str:
    """Identify bindings rebuilt from sealed evidence during revalidation.

    This is deliberately not named ``resolved_inputs_sha256``: resume does not
    recreate the original manifest file, so retaining or synthesizing that
    digest would misrepresent its authority. Paths are omitted because the
    evidence digest, host contract, and identity row carry the durable facts.
    """

    durable_bindings: Dict[str, Dict[str, Any]] = {}
    for input_key, raw_binding in sorted(bindings.items()):
        binding = dict(raw_binding)
        durable_bindings[str(input_key)] = {
            field: binding[field]
            for field in (
                "declared_kind",
                "product",
                "evidence_id",
                "sha256",
                "produced_by_step",
                "identity_row",
                "product_contract",
                "consumption_contract",
                "execution_projection",
            )
            if field in binding
        }
    return canonical_sha256(
        {
            "schema_version": (_RESUME_TYPED_INPUT_BINDING_FINGERPRINT_SCHEMA_VERSION),
            "bindings": durable_bindings,
        }
    )


@dataclass(frozen=True)
class TypedBindingResolver:
    """Resolve step inputs against current evidence and checkpoint authority.

    The caller retains the Planner-owned plan and passes its current revision to
    each resolution. This component owns no scientific choice, provider call,
    checkpoint mutation, or evidence promotion. The shared record lock preserves
    the execute loop's original per-input snapshot semantics under parallel
    auxiliary steps.
    """

    evidence_store: Any
    per_step_records: Sequence[Mapping[str, Any]]
    records_lock: Any
    run_dir: Path
    authoritative_cohort_path: Path
    development_sample: Optional[Any] = None

    def _records_snapshot(self) -> List[Mapping[str, Any]]:
        with self.records_lock:
            return list(self.per_step_records)

    def resolve_names(
        self,
        names: Sequence[str],
        *,
        plan: AnalysisPlan,
        consumer_step: Optional[AnalysisStep] = None,
        allow_unpublished_direct_ids: bool = False,
    ) -> Tuple[List[EvidenceRef], List[str], Dict[str, Dict[str, Any]]]:
        """Return exact evidence refs, typed ids, and host-owned bindings."""

        refs: List[EvidenceRef] = []
        typed_evidence_ids: List[str] = []
        typed_bindings: Dict[str, Dict[str, Any]] = {}
        seen: Set[str] = set()
        failures: List[Dict[str, Any]] = []
        for name in names:
            value = str(name)
            if _typed_input_product(value) is not None:
                records_snapshot = self._records_snapshot()
                evidence_snapshot = self.evidence_store.records()
                ref, failure = _resolve_typed_input_evidence(
                    input_name=value,
                    plan=plan,
                    evidence_records=evidence_snapshot,
                    per_step_records=records_snapshot,
                    run_dir=self.run_dir,
                )
                if failure is not None:
                    failures.append(failure)
                    continue
                if ref is not None and ref.evidence_id not in seen:
                    refs.append(ref)
                    seen.add(ref.evidence_id)
                    typed_evidence_ids.append(ref.evidence_id)
                if ref is not None:
                    binding = _resolved_typed_input_binding(
                        input_name=value,
                        evidence_ref=ref,
                        evidence_records=evidence_snapshot,
                        run_dir=self.run_dir,
                        producer_step_records=records_snapshot,
                        authoritative_cohort_path=self.authoritative_cohort_path,
                        development_sample=self.development_sample,
                    )
                    if binding is None:
                        failures.append(
                            {
                                "input": value,
                                "reason": "verified_binding_unavailable",
                            }
                        )
                    else:
                        if consumer_step is not None:
                            try:
                                binding = _attach_verified_consumption_contract(
                                    step=consumer_step,
                                    input_name=value,
                                    binding=binding,
                                )
                            except ArtifactConsumptionError as exc:
                                failures.append(
                                    {
                                        "input": value,
                                        "reason": (
                                            "artifact_consumption_contract_invalid"
                                        ),
                                        "message": str(exc),
                                    }
                                )
                                continue
                        typed_bindings[value] = binding
                        projected_evidence_id = str(binding.get("evidence_id") or "")
                        if projected_evidence_id and projected_evidence_id not in seen:
                            refs.append(EvidenceRef(evidence_id=projected_evidence_id))
                            seen.add(projected_evidence_id)
                            typed_evidence_ids.append(projected_evidence_id)
                continue

            direct_record = self.evidence_store.get(value)
            if (
                allow_unpublished_direct_ids
                and direct_record is not None
                and direct_record.evidence_id == value
            ):
                # The Critic reviews evidence registered by the in-flight
                # attempt before that attempt can be promoted to ``ok``.
                # Permit only its exact evidence IDs here; aliases remain
                # subject to current-success authority below.
                record = direct_record
            else:
                record = _current_verified_evidence_record(
                    self.evidence_store,
                    value,
                    self._records_snapshot(),
                )
            if record is not None and record.evidence_id not in seen:
                refs.append(
                    EvidenceRef(
                        evidence_id=record.evidence_id,
                        kind=record.kind,
                        description=record.description,
                        relative_path=record.relative_path,
                    )
                )
                seen.add(record.evidence_id)
        if failures:
            raise _EvidenceLineageResolutionError(failures)
        return refs, typed_evidence_ids, typed_bindings
