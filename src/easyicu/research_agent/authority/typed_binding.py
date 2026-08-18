"""Typed-input lineage, binding, and sealed receipt authority.

This module binds Planner-declared typed products to the current verified
producer evidence. It may emit only caller-scoped resolved-input manifests or
host input-binding receipts; evidence promotion, checkpoint selection, provider
calls, repair orchestration, and scientific design remain outside this layer.
"""

from __future__ import annotations

import json
import math
import re
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
from ..contracts.primary_cohort import locked_primary_cohort_product
from ..contracts.artifact_consumption import (
    ArtifactConsumptionError,
    verify_artifact_consumption,
)
from ..contracts.typed_schema import (
    merge_host_json_contract,
    typed_json_structure_receipt,
)
from ..contracts.cohort_receipt import cohort_receipt_authorized_columns
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
from .evidence_record_resolution import (
    _current_verified_evidence_record,
    _evidence_record_field,
    executed_product_evidence_authority,
)
from ..schema import (
    AnalysisPlan,
    AnalysisStep,
    ArtifactConsumptionContract,
    EvidenceRef,
)
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
    "host_authorized_ambient_trajectory_entry",
    "host_owns_input_binding_receipts",
    "rank_scale_columns_entry",
    "study_endpoint_declaration_entry",
]

HOST_AUTHORIZED_AMBIENT_INPUTS_SCHEMA_VERSION = (
    "easyicu.host_authorized_ambient_inputs/1"
)

STUDY_ENDPOINT_DECLARATION_SCHEMA_VERSION = "easyicu.study_endpoint_declaration/1"

RANK_SCALE_COLUMNS_SCHEMA_VERSION = "easyicu.rank_scale_columns/1"

_RESUME_TYPED_INPUT_BINDING_FINGERPRINT_SCHEMA_VERSION = (
    "easyicu.resume_typed_input_bindings/2"
)


_TYPED_INPUT_KEY_PATTERN = re.compile(r"[a-z][a-z0-9_]*:[a-z][a-z0-9_]*")


def _neutral_consumption_contract(
    *,
    input_name: str,
    binding: Mapping[str, Any],
) -> Optional[ArtifactConsumptionContract]:
    """Return the no-claim ``all_rows`` contract when one can be verified.

    ``all_rows`` is the absence of a selection, not a selection: this module's
    own rule is that a consumer with no explicit role selection must preserve
    every row.  ``single_row`` and ``one_per_role`` are the modes that assert
    something, and the host never compiles those -- a consumer that needs one
    still fails closed against this receipt, because the mode will not match.

    Returns ``None`` when the binding does not already carry what the receipt
    is made of.  That is the pre-existing state, so nothing that works today
    can start failing here.
    """

    if not _TYPED_INPUT_KEY_PATTERN.fullmatch(input_name or ""):
        return None
    product_contract = binding.get("product_contract")
    if not isinstance(product_contract, Mapping):
        return None
    row_count = product_contract.get("row_count")
    if isinstance(row_count, bool) or not isinstance(row_count, int) or row_count < 0:
        return None
    artifact_sha256 = str(binding.get("sha256") or "")
    if len(artifact_sha256) != 64:
        return None
    if not Path(str(binding.get("absolute_path") or "")).is_file():
        return None
    return ArtifactConsumptionContract(input_key=input_name, mode="all_rows")


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
    if len(contracts) > 1:  # schema validation already prevents this
        raise ArtifactConsumptionError("ambiguous input consumption contract")
    if contracts:
        contract = contracts[0]
    else:
        # The Planner was being asked to transcribe a constant: every one of
        # the 235 contracts declared across the recorded corpus is
        # ``all_rows`` with no role column and no roles. When it omitted the
        # line instead, the consumer died inside the container for want of a
        # receipt the host could have compiled from bytes it had already
        # verified. The declaration still wins wherever it is made.
        contract = _neutral_consumption_contract(
            input_name=input_name,
            binding=binding,
        )
        if contract is None:
            return binding
    updated = dict(binding)
    updated["consumption_contract"] = verify_artifact_consumption(
        contract=contract,
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

    executed_authority = executed_product_evidence_authority(
        run_dir=run_dir,
        producer_id=producer_id,
        producer_record=producer_record or {},
        input_name=str(input_name),
        product_fields=product_fields,
    )
    if executed_authority.failure is not None:
        return None, executed_authority.failure

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

    record, verified_path = candidates[0]
    if failure := executed_authority.path_failure(
        verified_path, evidence_record=record
    ):
        return None, failure
    return (
        EvidenceRef(
            evidence_id=str(_evidence_record_field(record, "evidence_id") or ""),
            kind=_evidence_record_field(record, "kind"),
            description=_evidence_record_field(record, "description"),
            relative_path=_evidence_record_field(record, "relative_path"),
        ),
        None,
    )


#: Serializations a consumer can parse without being told any coordinates.
#: Read off the recorded corpus rather than enumerated by intent: every typed
#: input that ever bound without a compiled schema receipt was one of these.
_SELF_DESCRIBING_TYPED_INPUT_SUFFIXES = frozenset({".json", ".jsonl"})


def _binding_is_readable_without_a_schema_receipt(
    verified_path: Path,
    producer_contract: Optional[Mapping[str, Any]],
) -> bool:
    """Say whether a consumer can locate values without a host schema receipt.

    Two ways, and only two.  The bytes describe themselves, or the producer
    declared coordinates for them.  A binding that has neither reaches the
    consumer as a path and a digest.
    """

    if Path(verified_path).suffix.lower() in _SELF_DESCRIBING_TYPED_INPUT_SUFFIXES:
        return True
    return bool(
        producer_contract and any(key != "schema_version" for key in producer_contract)
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
    locked_cohort_name: object = None,
    refusals: Optional[List[Dict[str, Any]]] = None,
) -> Optional[Dict[str, Any]]:
    """Build the exact, digest-verified runtime binding for one typed input.

    ``refusals`` collects a typed reason for the refusals this function can
    explain. A caller that passes one gets a diagnosis it can hand back to the
    Planner instead of the generic "no verified host binding".
    """

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
    # The primary-cohort identity is decided by its owner, not by one spelling
    # of it: ``cohort:analysis_set`` and the plan's own ``cohort:<cohort.name>``
    # are the same locked population as ``analysis_cohort``.  Recognising only
    # a subset here let typed consumers execute on the full cohort while the
    # run still reported the development sample -- twice now, most recently on
    # canary20's primary model (94,425 rows against a contract expecting the
    # 1,000-row sample), so the reader must be the one that knows all three.
    binds_primary_cohort = (
        locked_primary_cohort_product(input_name, locked_cohort_name=locked_cohort_name)
        is not None
    )
    parent_already_development_scoped = (
        development_sample is not None
        and binds_primary_cohort
        and _producer_output_is_already_development_scoped(
            evidence_id=evidence_ref.evidence_id,
            producer_step_records=producer_step_records,
            sample_sha256=str(getattr(development_sample, "sample_sha256", "") or ""),
        )
    )
    if (
        development_sample is not None
        and binds_primary_cohort
        and not parent_already_development_scoped
    ):
        projection = resolve_development_input_projection(
            declared_input=input_name,
            parent_evidence_id=parent_evidence_id,
            parent_sha256=parent_sha256,
            parent_produced_by_step=parent_produced_by_step,
            parent_verified_path=parent_verified_path,
            evidence_records=evidence_records,
            run_dir=run_dir,
            authoritative_cohort_path=authoritative_cohort_path,
            development_sample=development_sample,
            locked_cohort_name=locked_cohort_name,
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
    elif verified_path.suffix.lower() == ".json":
        structure_receipt = typed_json_structure_receipt(
            artifact_path=verified_path,
            expected_sha256=binding["sha256"],
        )
        if structure_receipt is None:
            # Unusually large or structurally unsafe JSON remains directly
            # parseable by suffix, but no structural coordinates are promoted.
            host_contract = dict(producer_contract or {})
            host_contract.pop("json_structure", None)
            host_contract["schema_version"] = "easyicu.host_typed_product.v1"
        else:
            host_contract = merge_host_json_contract(
                producer_contract,
                structure_receipt,
            )
    elif not _binding_is_readable_without_a_schema_receipt(
        verified_path, producer_contract
    ):
        # A BINDING MUST TELL ITS CONSUMER WHERE TO LOOK, OR NOT BE PUBLISHED.
        #
        # The table branch above compiles a schema receipt; a self-describing
        # serialization needs none because the reader parses the file itself.
        # Anything else reaches the consumer as a path and a digest, and the
        # generated code has no choice but to guess what is inside.
        #
        # MEASURED over every recorded resolved input on 2026-08-03 (1,071
        # bindings): 992 resolved to physical tables and carried a full column /
        # dtype / row-count receipt; 76 were self-describing JSON values; THREE
        # were neither. Structured JSON now receives a host-sealed, value-free
        # path/key receipt in the branch above. All three opaque files were the
        # same pickle bound as
        # ``artifact:trained_prediction_model``, and all three killed the step
        # that consumed them -- each in a different way, because each generated
        # script invented its own guess about the coordinates:
        #
        #   06_held_out_discrimination  ValueError: Prediction artifact contract
        #                               must declare id_column and
        #                               prediction_column
        #   08_held_out_calibration     RuntimeError: does not contain a
        #                               supported held-out prediction table or
        #                               aligned prediction vectors
        #   10_clinical_utility         RuntimeError: lacks consumption_contract
        #
        # Two further steps died as their collateral, so one unreadable binding
        # cost five of thirteen steps.  The refusal is keyed on the
        # SERIALIZATION rather than on the evidence kind or the product name:
        # ``RUNTIME_TYPED_INPUT_EVIDENCE_KINDS`` maps four typed kinds onto
        # ``log`` evidence, and what makes a binding readable is whether its
        # bytes can be parsed without coordinates -- not what the registry
        # labelled it.
        if refusals is not None:
            refusals.append(
                {
                    "input": str(input_name),
                    "reason": "typed_input_serialization_is_unreadable",
                    "produced_by_step": binding["produced_by_step"],
                    "serialization": Path(verified_path).suffix.lower() or "(none)",
                    "message": (
                        f"{input_name} resolves to a "
                        f"{Path(verified_path).suffix.lower() or 'suffixless'} file "
                        "whose contents the host cannot describe, so a consumer "
                        "cannot locate any value inside it. Declare this product "
                        "as a table whose columns carry the values downstream "
                        "steps read."
                    ),
                }
            )
        return None
    else:
        # Readable: either the consumer parses the serialization directly or the
        # producer declared its own coordinates. An empty contract here is
        # honest, not missing; structured JSON was handled above.
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


def _validated_primary_cohort_execution_receipt(
    receipt: Mapping[str, Any],
) -> Dict[str, Any]:
    """Return a JSON-safe, row-accounted host cohort execution receipt."""

    try:
        payload = json.loads(
            json.dumps(
                receipt,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("host cohort execution receipt must be finite JSON") from exc
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != "easyicu.primary_cohort_execution_prompt/1"
    ):
        raise ValueError("host cohort execution receipt schema is invalid")

    def _is_sha256(value: Any) -> bool:
        return (
            isinstance(value, str)
            and len(value) == 64
            and all(character in "0123456789abcdef" for character in value)
        )

    def _row_count(value: Any, *, field: str) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(
                f"host cohort execution receipt {field} must be a non-negative integer"
            )
        return value

    if not _is_sha256(payload.get("cohort_definition_sha256")):
        raise ValueError("host cohort execution receipt cohort digest is invalid")
    raw_universe = payload.get("raw_universe")
    analysis_cohort = payload.get("authoritative_analysis_cohort")
    flow = payload.get("ordered_predicate_flow")
    if (
        not isinstance(raw_universe, dict)
        or not isinstance(analysis_cohort, dict)
        or not isinstance(flow, list)
        or not flow
        or any(not isinstance(row, dict) for row in flow)
    ):
        raise ValueError("host cohort execution receipt structure is invalid")
    if not _is_sha256(raw_universe.get("sha256")) or not _is_sha256(
        analysis_cohort.get("sha256")
    ):
        raise ValueError("host cohort execution receipt artifact digest is invalid")
    for optional_digest in ("row_identity_sha256", "authority_sha256"):
        value = analysis_cohort.get(optional_digest)
        if value is not None and not _is_sha256(value):
            raise ValueError(
                f"host cohort execution receipt {optional_digest} is invalid"
            )
    identity_column = analysis_cohort.get("identity_column")
    if identity_column is not None and (
        not isinstance(identity_column, str) or not identity_column.strip()
    ):
        raise ValueError("host cohort execution receipt identity_column is invalid")

    raw_rows = _row_count(raw_universe.get("rows"), field="raw rows")
    analysis_rows = _row_count(analysis_cohort.get("rows"), field="analysis rows")
    previous_remaining: Optional[int] = None
    for index, row in enumerate(flow):
        before = _row_count(row.get("n_before"), field="n_before")
        excluded = _row_count(row.get("n_excluded"), field="n_excluded")
        remaining = _row_count(row.get("n_remaining"), field="n_remaining")
        resolved_column = row.get("resolved_column")
        if resolved_column is not None and not (
            isinstance(resolved_column, str)
            and resolved_column.strip()
            and ":" not in resolved_column
        ):
            raise ValueError("host cohort execution receipt resolved_column is invalid")
        if row.get("step_order") != index:
            raise ValueError("host cohort execution receipt step order is invalid")
        if before != excluded + remaining:
            raise ValueError("host cohort execution receipt partition is invalid")
        if previous_remaining is not None and before != previous_remaining:
            raise ValueError("host cohort execution receipt flow is discontinuous")
        previous_remaining = remaining
    if (
        flow[0].get("predicate_kind") != "universe"
        or flow[0]["n_before"] != raw_rows
        or flow[0]["n_remaining"] != raw_rows
        or previous_remaining != analysis_rows
    ):
        raise ValueError("host cohort execution receipt row accounting is invalid")
    return payload


def host_authorized_ambient_trajectory_entry(
    trajectory: Any,
) -> Optional[Dict[str, Any]]:
    """Describe the ambient long trajectory in the step's own typed record.

    The host stages this table, verifies it to a SHA-256, and hands its path
    to every step through ``TRAJECTORY_PARQUET``.  It is deliberately NOT a
    Planner-declared input: it has no name in the executable roster, and the
    plan contract refuses a plan that lists one.  Until now that decision was
    published to the agent only as prompt prose, while the one machine-readable
    record the generated script opens and verifies -- ``resolved_inputs`` --
    named the cohort and nothing else.

    MEASURED: a step that declared ``manifest:trajectory_window_manifest``, and
    whose prompt therefore carried the MANDATORY paragraph saying its windows
    come from this table, wrote a correct loader for it and then discarded the
    result::

        # This step has only the typed analysis-cohort input.  Use the
        # explicitly registered fixed-window columns and do not process the
        # undeclared, potentially very large trajectory table.
        trajectory = pd.DataFrame()

    It then died on the empty frame.  The premise in that comment is what has
    to go: the code trusted its typed record over the prose, and the record
    agreed with it.  Naming the table here -- with the same relative path,
    digest and role columns the cohort entry carries -- makes "undeclared"
    false rather than arguing with it.

    Returns ``None`` when no trajectory is bound, so a wide-column run and a
    non-trajectory run produce a byte-identical manifest.
    """

    if trajectory is None:
        return None
    relative_path = str(getattr(trajectory, "trajectory_file", "") or "").strip()
    digest = str(getattr(trajectory, "trajectory_sha256", "") or "").strip()
    if not relative_path or not digest:
        return None
    columns = [
        str(item)
        for item in (getattr(trajectory, "trajectory_columns", None) or ())
        if str(item).strip()
    ]
    roles = {
        "identity_column": getattr(trajectory, "identity_column", None),
        "time_column": getattr(trajectory, "time_column", None),
        "concept_column": getattr(trajectory, "concept_column", None),
        "numeric_value_column": getattr(trajectory, "numeric_value_column", None),
        "text_value_column": getattr(trajectory, "text_value_column", None),
    }
    resolved_roles = {
        key: str(value) for key, value in roles.items() if str(value or "").strip()
    }
    if not columns or len(resolved_roles) != len(roles):
        return None
    if any(value not in columns for value in resolved_roles.values()):
        raise ValueError("trajectory role columns must exist in the bound table")
    # `concepts` is published as "the whole vocabulary present in the table",
    # which is a property of the TABLE, not of one step.  The per-step scoped
    # projection narrows this list to the concepts the step's declared
    # variables select -- and a LONG-bound run declares no trajectory
    # variables at all, so that intersection is empty by construction.
    #
    # MEASURED: handed the scoped projection, this builder published
    # `"concepts": []` under a sentence promising completeness. A record that
    # asserts it is complete and lists nothing is worse than no record: it
    # tells the agent, with the host's authority, that the table is empty.
    # Refuse the scoped projection instead of narrowing the claim.
    scope = str(getattr(trajectory, "projection_scope", "full") or "full")
    if scope != "full":
        raise ValueError(
            "the ambient trajectory entry publishes the table's complete "
            "vocabulary and must be built from the unscoped context, not "
            f"from a {scope!r} projection"
        )
    concepts = [
        str(item)
        for item in (getattr(trajectory, "materialized_concepts", None) or ())
        if str(item).strip()
    ]
    if not concepts:
        return None
    window = getattr(trajectory, "window", None)
    entry: Dict[str, Any] = {
        "access": "TRAJECTORY_PARQUET",
        "relative_path": relative_path,
        "sha256": digest,
        "columns": columns,
        "concepts": concepts,
        "authorization": (
            "Host-staged and host-verified. Reading this table in this step is "
            "authorized: it is not an undeclared file. It is deliberately not a "
            "Planner-declared input and must never be added to `inputs`. Select "
            "concepts by exact string from `concepts`; that list is the whole "
            "vocabulary present in the table."
        ),
    }
    entry.update(resolved_roles)
    for field, key in (
        ("trajectory_rows", "row_count"),
        ("time_unit", "time_unit"),
        ("time_origin", "time_origin"),
    ):
        value = getattr(trajectory, field, None)
        if value is not None and str(value).strip():
            entry[key] = value
    if isinstance(window, Mapping) and window:
        entry["window"] = dict(window)
    return entry


#: Roles whose values are ranks, not measurements on an interval scale.
#: Read from the context's own ``VariableRole`` vocabulary rather than from a
#: list of score names: the dictionary already assigns the role, and a name list
#: here would be a second, divergent opinion about what GCS is.
_RANK_SCALE_VARIABLE_ROLES = frozenset({"ordinal_score"})


def rank_scale_columns_entry(context: Any) -> Optional[Dict[str, Any]]:
    """Publish which bound columns are ranks, and what their domain is.

    MEASURED on the five never-passing tasks: 6 of 29 scientific blocking
    findings are an ordinal score used as an interval measurement -- GCS as a
    continuous propensity-score covariate and in standardized mean differences,
    SOFA components passed to a summary that reports arithmetic means, per-hour
    medians of ordinal levels emitted as fractional SOFA, and availability
    counted from non-missingness with no level-domain check at all.

    The concept layer already knows all of it. ``gcs_max`` arrives with
    ``role="ordinal_score"``, ``valid_range=[3.0, 15.0]`` and the pitfall "GCS is
    ordinal; do not take its mean. Report worst (min) or representative
    (last/first) GCS."

    The record the generated script opens knows none of it. ``product_contract``
    is built from the artifact file alone, so it publishes closed value sets for
    string categoricals (``adm``, ``sex``) and lists every ordinal in
    ``numeric_columns`` beside lactate -- ``if name in numeric_set: continue`` is
    the line an ordinal falls through. A script reading that record sees a
    float32 and averages it.

    Published as its own context-derived entry rather than merged into
    ``product_contract``: that profile is a digest-verified receipt for the bytes
    of one file, and a fact that came from the context does not belong inside it.
    """

    variables = getattr(context, "variables", None) or ()
    columns: Dict[str, Any] = {}
    for variable in variables:
        role = getattr(getattr(variable, "role", None), "value", None) or getattr(
            variable, "role", None
        )
        if str(role) not in _RANK_SCALE_VARIABLE_ROLES:
            continue
        name = str(getattr(variable, "name", "") or "").strip()
        if not name:
            continue
        entry: Dict[str, Any] = {"role": str(role)}
        # The declared plausible domain and the domain actually present are
        # different facts and both matter: the first says which values are legal
        # levels, the second says which of them this cohort contains. A check
        # written against only the second passes a cohort that happens to be
        # clean and misses the invalid level the audit asked about.
        valid_range = getattr(variable, "valid_range", None)
        if valid_range:
            entry["valid_range"] = [float(value) for value in valid_range]
        observed = getattr(variable, "observed_domain", None)
        if isinstance(observed, Mapping):
            levels = observed.get("levels")
            if isinstance(levels, list) and levels:
                entry["observed_levels"] = list(levels)
            for key in ("min", "max", "n_unique"):
                if observed.get(key) is not None:
                    entry[f"observed_{key}"] = observed[key]
        aggregation = getattr(variable, "aggregation_default", None)
        # ``.value`` before ``str``: an enum stringifies to "AggregationRule.
        # MAX_LAST", which is a Python identifier and not the vocabulary the rest
        # of the record uses. A reader matching it against the aggregation names
        # it knows would find no match and fall back to choosing one.
        aggregation = getattr(aggregation, "value", aggregation)
        if aggregation is not None and str(aggregation).strip():
            entry["aggregation_default"] = str(aggregation)
        columns[name] = entry
    if not columns:
        return None
    return {
        "columns": columns,
        "authorization": (
            "These bound columns carry RANKS, not interval measurements, as "
            "declared by the concept layer -- they appear among the numeric "
            "columns of the artifact profile because their storage dtype is "
            "numeric, which is a fact about storage and not about the scale. "
            "Summarise them rank-preservingly (median, quantile, worst, "
            "first/last, or a declared level distribution); an arithmetic mean "
            "or any statistic that can land between two levels is not a value "
            "of this scale. A value outside the declared domain is an invalid "
            "level and must stop the step, not be counted as available. Using "
            "one as a numeric model covariate is permitted only if the script's "
            "own output states that coding."
        ),
    }


def study_endpoint_declaration_entry(endpoint: Any) -> Optional[Dict[str, Any]]:
    """Publish the plan's typed endpoint in the step's own machine-readable record.

    The endpoint is the other half of what the cohort declaration says: the
    cohort names who is counted, the endpoint names what happened to them and
    when follow-up ended.  ``EndpointSpec`` has carried ``time_column``,
    ``time_origin`` and ``censoring_rule`` for some time, with a validator that
    refuses to infer any of them, and the typed context already verifies the
    columns it names against the sealed cohort.

    MEASURED over 291 recorded runs: nothing ever declared one.  What the
    generated code got instead was the follow-up rule as prose in one step's
    ``icu_rule_refs`` -- written in 3 of 13 survival plans, absent from the
    other 10 -- so it reached for whatever time column it could find.  Across
    the 11 runs with recovered source that produced SEVEN distinct combinations
    of ``{los_icu, los_hosp, death_time, discharge_time, END_HOURS}``.  The
    concept auditor then blocked steps for contradicting a "planner-required ICU
    discharge time ``los_icu``" that appears in no plan for that task -- the
    plans that stated anything stated hospital discharge.  Neither side was
    reading a declaration, because there was none to read.

    Published here rather than in the Coder prompt because the prompt is where
    the losing copy already lived: the generated script opens
    ``resolved_inputs``, hash-verifies it, and trusts it over prose.  The Coder
    prompt is also 152 bytes from its hard budget on the widest step, so a
    paragraph there would evict typed context to restate what a record can hold.
    """

    if endpoint is None:
        return None
    kind = str(getattr(endpoint, "kind", "") or "").strip()
    name = str(getattr(endpoint, "name", "") or "").strip()
    if not kind or not name:
        return None
    entry: Dict[str, Any] = {
        "name": name,
        "kind": kind,
        "authorization": (
            "The study's endpoint as DECLARED by the locked plan, not as "
            "inferred. Implement exactly these fields: they are the study "
            "definition, and a step that substitutes a different time column, "
            "origin, censoring rule or level set is analysing a different study "
            "than the one under review. If a field this step needs is absent "
            "here, stop and report that rather than choosing one. Before "
            "building a risk set or a landmark outcome from event_column and "
            "time_column, reconcile the pair: an event whose time is missing or "
            "non-finite cannot be placed on the follow-up axis, and comparing it "
            "against a horizon silently recodes it to 'no event'. A censored row "
            "with no event time is the expected shape and must not be excluded "
            "on that basis. `event_time_reconciliation_receipt` in "
            "easyicu.research_agent.methods.survival_inputs checks exactly this "
            "against these declared levels and returns counts only."
        ),
    }
    for field in (
        "absence_semantics",
        "event_column",
        "time_column",
        "time_origin",
        "censoring_rule",
    ):
        value = getattr(endpoint, field, None)
        if value is not None and str(value).strip():
            entry[field] = str(value)
    levels = getattr(endpoint, "levels", None)
    if levels is not None:
        entry["levels"] = list(levels)
    return entry


def _write_resolved_inputs_manifest(
    *,
    run_dir: Path,
    step_id: str,
    planner_declared_inputs: Sequence[str],
    bindings: Mapping[str, Mapping[str, Any]],
    context_path: Optional[Path] = None,
    raw_input_contracts: Optional[Mapping[str, Any]] = None,
    host_verified_cohort_execution_receipt: Optional[Mapping[str, Any]] = None,
    host_authorized_ambient_trajectory: Optional[Mapping[str, Any]] = None,
    study_endpoint: Optional[Mapping[str, Any]] = None,
    rank_scale_columns: Optional[Mapping[str, Any]] = None,
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
        # A name repeated in this list is not ambiguous, so refusing it buys
        # nothing and costs the run.  Both copies index the same entry of
        # ``bindings`` / ``contracts`` and resolve to the same column, and both
        # readers below (``declared_typed_inputs`` and ``authorized_raw_inputs``)
        # are sets, so a repeat already collapses before anything compares it.
        #
        # The host manufactures the repeat itself.  ``close_measurement_
        # companion_inputs`` appends registered ``_measured``/``_n`` companions
        # to a step's public inputs; a later replan rewrites that step's inputs,
        # keeps the appended tail verbatim and re-declares one of the same names
        # in its own body.  A real run died exactly there -- ``lact_n`` absent in
        # revision 1, appended by the closure in revision 2, declared a second
        # time by revision 3 -- and the refusal killed the run mid-plan.
        #
        # Deduplicating here also makes the uniqueness precondition that
        # ``typed_input_receipt`` re-checks on the written manifest true by
        # construction rather than by hope; that reader stays as it is.
        if item in seen_declared_inputs:
            continue
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
    validated_cohort_receipt: Optional[Dict[str, Any]] = None
    receipt_raw_inputs: Set[str] = set()
    if host_verified_cohort_execution_receipt is not None:
        validated_cohort_receipt = _validated_primary_cohort_execution_receipt(
            host_verified_cohort_execution_receipt
        )
        # Both fields, from the one declaration the producer also reads.  This
        # side used to take ``resolved_column`` alone while
        # ``raw_contract_inputs_for_step`` already authorized the event-time
        # column too, so a plan whose cohort predicate carried a time window
        # produced contracts this check called unauthorized and killed the run
        # at its first step.
        receipt_raw_inputs = cohort_receipt_authorized_columns(
            validated_cohort_receipt["ordered_predicate_flow"]
        )
    if raw_input_contracts is not None:
        raw_payload = dict(raw_input_contracts)
        contracts = raw_payload.get("contracts")
        authorized_raw_inputs = {
            item for item in declared_inputs if ":" not in item
        } | receipt_raw_inputs
        if (
            raw_payload.get("schema_version")
            != "easyicu.resolved_raw_input_contracts/1"
            or not isinstance(contracts, Mapping)
            or set(contracts) != authorized_raw_inputs
        ):
            raise ValueError(
                "raw input contracts must exactly match Planner-declared or "
                "host-receipt raw inputs"
            )
        declared_digest = str(raw_payload.pop("contracts_sha256", "") or "")
        if declared_digest != canonical_sha256(raw_payload):
            raise ValueError("raw input contract digest mismatch")
        raw_payload["contracts_sha256"] = declared_digest
        payload["raw_input_contracts"] = raw_payload
    if validated_cohort_receipt is not None:
        payload["host_verified_cohort_execution_receipt"] = validated_cohort_receipt
    if host_authorized_ambient_trajectory is not None:
        ambient = dict(host_authorized_ambient_trajectory)
        ambient_relative = str(ambient.get("relative_path", "") or "")
        ambient_path = (Path(run_dir).resolve() / ambient_relative).resolve()
        try:
            ambient_path.relative_to(Path(run_dir).resolve())
        except ValueError as exc:
            raise ValueError(
                "ambient trajectory path must be contained by run_dir"
            ) from exc
        if not ambient_path.is_file():
            raise ValueError("ambient trajectory path must name an existing file")
        ambient_digest = str(ambient.get("sha256", "") or "")
        if len(ambient_digest) != 64 or any(
            character not in "0123456789abcdef" for character in ambient_digest
        ):
            raise ValueError("ambient trajectory sha256 is invalid")
        payload["host_authorized_ambient_inputs"] = {
            "schema_version": HOST_AUTHORIZED_AMBIENT_INPUTS_SCHEMA_VERSION,
            "trajectory": ambient,
        }
    if study_endpoint is not None:
        declaration = dict(study_endpoint)
        # The two fields that make this a declaration rather than a label. A
        # record naming an endpoint whose kind is unknown would put the reader
        # straight back into inferring one from the column name.
        for required in ("name", "kind"):
            if not str(declaration.get(required, "") or "").strip():
                raise ValueError(f"study endpoint declaration must carry {required}")
        # A time axis without its origin is the defect this record exists to
        # close: a duration and a timestamp are indistinguishable by dtype, and
        # a step that guesses wrong reports follow-up from the wrong zero.
        if declaration.get("time_column") and not str(
            declaration.get("time_origin", "") or ""
        ).strip():
            raise ValueError(
                "a study endpoint declaring time_column must declare time_origin"
            )
        payload["study_endpoint"] = {
            "schema_version": STUDY_ENDPOINT_DECLARATION_SCHEMA_VERSION,
            **declaration,
        }
    if rank_scale_columns is not None:
        ranks = dict(rank_scale_columns)
        declared_columns = ranks.get("columns")
        # An entry naming no column would publish "nothing here is a rank" with
        # the host's authority -- the same shape as the ambient-trajectory entry
        # that shipped `"concepts": []` under a promise of completeness. Refuse
        # it; the builder returns None when there is nothing to declare.
        if not isinstance(declared_columns, Mapping) or not declared_columns:
            raise ValueError(
                "a rank-scale declaration must name at least one column"
            )
        payload["rank_scale_columns"] = {
            "schema_version": RANK_SCALE_COLUMNS_SCHEMA_VERSION,
            **ranks,
        }
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


def host_owns_input_binding_receipts(
    *,
    deterministic_standard_executor_used: bool,
    deterministic_fallback_used: bool,
    sealed_renderer_repair: bool,
) -> bool:
    """Whether the HOST, not the generated script, must write the receipts.

    Exactly one rule, stated once, for every producer whose code the host
    rendered itself: a registered standard executor, one of the deterministic
    fallback runners (robustness/sensitivity, absolute-risk context,
    missingness audit), or a sealed renderer repair.  None of them can be
    asked to manufacture a receipt for its own input -- it would be attesting
    to itself -- so ``_write_host_input_binding_receipts`` records it from the
    authority bindings instead.

    The execute layer used to spell this rule out at each call site, and the
    two copies disagreed: the site that runs BEFORE the contract gate omitted
    ``deterministic_fallback_used``, and the site that includes it runs AFTER
    the gate.  A fallback runner therefore had its receipts written only at a
    point it could never reach, because the gate had already refused the step
    for the absence of exactly those receipts.  Measured over every recorded
    run: 12 of 18 deterministic-fallback steps were refused that way and 11
    of the 12 died; the single survivor was a Coder rewrite that hand-built
    the receipt block.
    """

    return bool(
        deterministic_standard_executor_used
        or deterministic_fallback_used
        or sealed_renderer_repair
    )


def _write_host_input_binding_receipts(
    *,
    out_dir: Path,
    step_summary: Mapping[str, Any],
    resolved_input_bindings: Mapping[str, Mapping[str, Any]],
    consumed_input_keys: Sequence[str],
) -> Dict[str, Any]:
    """Seal exact input receipts for a host-owned deterministic renderer.

    A sealed renderer consumes only the host-resolved artifacts authorized by
    its parent digest seal.  Generated code must normally report its own
    receipts, but asking a host-owned renderer to manufacture those receipts is
    both redundant and weaker than recording them here from the authority
    bindings.  An unreadable table is deliberately omitted so the downstream
    integrity validator fails closed on incomplete coverage.
    """

    consumed = tuple(dict.fromkeys(str(key) for key in consumed_input_keys))
    unknown = sorted(set(consumed) - set(resolved_input_bindings))
    if unknown:
        raise ValueError(
            "host consumed-input proof names unresolved bindings: " + ", ".join(unknown)
        )
    receipts: List[Dict[str, Any]] = []
    for input_key in sorted(consumed):
        raw_binding = resolved_input_bindings[input_key]
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
        binding_refusals: List[Dict[str, Any]] = []
        binding = _resolved_typed_input_binding(
            input_name=input_name,
            evidence_ref=ref,
            evidence_records=evidence_records,
            run_dir=run_dir,
            producer_step_records=trusted_step_records,
            authoritative_cohort_path=cohort_path,
            development_sample=development_sample,
            locked_cohort_name=getattr(getattr(plan, "cohort", None), "name", None),
            refusals=binding_refusals,
        )
        if binding is None:
            if binding_refusals:
                raise ValueError(
                    f"typed input {input_name} has no verified host binding: "
                    + json.dumps(binding_refusals[0], sort_keys=True, default=str)
                )
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
                    binding_refusals: List[Dict[str, Any]] = []
                    binding = _resolved_typed_input_binding(
                        input_name=value,
                        evidence_ref=ref,
                        evidence_records=evidence_snapshot,
                        run_dir=self.run_dir,
                        producer_step_records=records_snapshot,
                        authoritative_cohort_path=self.authoritative_cohort_path,
                        development_sample=self.development_sample,
                        locked_cohort_name=getattr(
                            getattr(plan, "cohort", None), "name", None
                        ),
                        refusals=binding_refusals,
                    )
                    if binding is None:
                        failures.append(
                            binding_refusals[0]
                            if binding_refusals
                            else {
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
