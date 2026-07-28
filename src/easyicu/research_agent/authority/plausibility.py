"""Step-scoped authority for ``retain_and_flag`` plausibility obligations.

The ResearchContext describes the whole study.  It is intentionally broader
than any one analysis step, so its ranged variables are not an execution
contract for that step.  The exact executable authority already lives in the
step's sealed ``raw_input_contracts`` payload.

This module compiles that payload once into a small immutable scope shared by
the Coder prompt, deterministic code gate, post-execution receipt gate, cache
identity, and resume revalidation.  Generated code can provide evidence that
it implemented an obligation; it cannot create or erase the obligation by
mentioning (or omitting) a string in its own source.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Dict, Mapping, Optional, Sequence

from ..schema import AnalysisStep, ResearchContext
from .step_capsule import (
    StepAuthorityCapsuleError,
    StepAuthorityCapsuleRef,
    load_verified_step_authority_capsule,
    read_verified_content,
)


RAW_INPUT_CONTRACT_SCHEMA = "easyicu.resolved_raw_input_contracts/1"
PLAUSIBILITY_SCOPE_SCHEMA = "easyicu.flag_only_plausibility_scope/1"
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_EXECUTED_CAPSULE_STAGES = frozenset({"executed", "executed_concept_audited"})


class PlausibilityScopeError(ValueError):
    """The host-owned raw contract cannot produce a trustworthy scope."""


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        dict(payload),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class FlagOnlyPlausibilityScope:
    """Exact columns for which this step owes a ``retain_and_flag`` receipt."""

    step_id: str
    expected_columns: tuple[str, ...]
    source_contracts_sha256: str
    authority_kind: str

    def __post_init__(self) -> None:
        if not isinstance(self.step_id, str) or not self.step_id.strip():
            raise PlausibilityScopeError("plausibility scope requires a step_id")
        if any(
            not isinstance(column, str)
            or not column.strip()
            or column != column.strip()
            or ":" in column
            for column in self.expected_columns
        ):
            raise PlausibilityScopeError(
                "plausibility scope columns must be exact raw column strings"
            )
        if self.expected_columns != tuple(sorted(set(self.expected_columns))):
            raise PlausibilityScopeError(
                "plausibility scope columns must be sorted and unique"
            )
        if (
            not isinstance(self.source_contracts_sha256, str)
            or _SHA256_PATTERN.fullmatch(self.source_contracts_sha256) is None
        ):
            raise PlausibilityScopeError(
                "plausibility scope requires a source contract SHA-256"
            )
        if not isinstance(self.authority_kind, str) or not self.authority_kind.strip():
            raise PlausibilityScopeError(
                "plausibility scope requires an authority kind"
            )

    def identity_payload(self) -> dict[str, Any]:
        return {
            "schema_version": PLAUSIBILITY_SCOPE_SCHEMA,
            "step_id": self.step_id,
            "source_contracts_sha256": self.source_contracts_sha256,
            "authority_kind": self.authority_kind,
            "expected_columns": list(self.expected_columns),
        }

    @property
    def scope_sha256(self) -> str:
        return _canonical_sha256(self.identity_payload())

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.identity_payload(),
            "scope_sha256": self.scope_sha256,
        }

    def require_step(self, step_id: str) -> "FlagOnlyPlausibilityScope":
        """Fail closed when a consumer receives another step's authority."""

        if self.step_id != str(step_id):
            raise PlausibilityScopeError(
                "plausibility scope belongs to a different analysis step"
            )
        return self


@dataclass(frozen=True, slots=True)
class StepPlausibilityAuthority:
    """Immutable raw-contract payload plus its compiled step scope."""

    scope: FlagOnlyPlausibilityScope
    raw_input_contracts_canonical_json: str

    def __post_init__(self) -> None:
        try:
            payload = json.loads(self.raw_input_contracts_canonical_json)
        except (TypeError, json.JSONDecodeError) as exc:
            raise PlausibilityScopeError(
                "step plausibility authority contains invalid canonical JSON"
            ) from exc
        if not isinstance(payload, Mapping):
            raise PlausibilityScopeError(
                "step plausibility authority contracts must be an object"
            )
        compiled = _scope_from_resolved_raw_contracts(
            step_id=self.scope.step_id,
            raw_input_contracts=payload,
        )
        if compiled != self.scope:
            raise PlausibilityScopeError(
                "step plausibility authority scope does not match its contracts"
            )

    def raw_input_contracts(self) -> dict[str, Any]:
        """Return a fresh mutable projection for the manifest writer."""

        payload = json.loads(self.raw_input_contracts_canonical_json)
        if not isinstance(payload, dict):  # guarded by ``__post_init__``
            raise PlausibilityScopeError(
                "step plausibility authority contracts must be an object"
            )
        return payload


def _validated_range(value: Any, *, column: str) -> Optional[Mapping[str, Any]]:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise PlausibilityScopeError(
            f"analysis_plausibility_range for {column!r} must be an object"
        )
    if not set(value).issubset({"minimum", "maximum"}):
        raise PlausibilityScopeError(
            f"analysis_plausibility_range for {column!r} has unknown fields"
        )
    minimum = value.get("minimum")
    maximum = value.get("maximum")
    if minimum is None and maximum is None:
        raise PlausibilityScopeError(
            f"analysis_plausibility_range for {column!r} has no finite bound"
        )
    for name, bound in (("minimum", minimum), ("maximum", maximum)):
        if bound is None:
            continue
        if (
            isinstance(bound, bool)
            or not isinstance(bound, (int, float))
            or not math.isfinite(float(bound))
        ):
            raise PlausibilityScopeError(
                f"{name} for {column!r} must be a finite number or null"
            )
    if minimum is not None and maximum is not None and float(minimum) > float(maximum):
        raise PlausibilityScopeError(
            f"analysis_plausibility_range for {column!r} is reversed"
        )
    return value


def _scope_from_resolved_raw_contracts(
    *,
    step_id: str,
    raw_input_contracts: Mapping[str, Any],
) -> FlagOnlyPlausibilityScope:
    payload = dict(raw_input_contracts)
    declared_digest = str(payload.pop("contracts_sha256", "") or "").strip().lower()
    if (
        payload.get("schema_version") != RAW_INPUT_CONTRACT_SCHEMA
        or _SHA256_PATTERN.fullmatch(declared_digest) is None
        or _canonical_sha256(payload) != declared_digest
    ):
        raise PlausibilityScopeError(
            "resolved raw-input contract schema or digest is invalid"
        )
    contracts = payload.get("contracts")
    if not isinstance(contracts, Mapping):
        raise PlausibilityScopeError(
            "resolved raw-input contracts must contain a contracts object"
        )

    expected: list[str] = []
    for raw_column, raw_contract in contracts.items():
        if not isinstance(raw_column, str) or not raw_column.strip():
            raise PlausibilityScopeError(
                "resolved raw-input contract keys must be non-empty strings"
            )
        column = raw_column.strip()
        if ":" in column:
            raise PlausibilityScopeError(
                "resolved raw-input contracts cannot contain typed product keys"
            )
        if not isinstance(raw_contract, Mapping):
            raise PlausibilityScopeError(
                f"resolved raw-input contract for {column!r} must be an object"
            )
        declared_column = raw_contract.get("column")
        if declared_column is not None and str(declared_column) != column:
            raise PlausibilityScopeError(
                f"resolved raw-input contract key disagrees for {column!r}"
            )

        plausibility_range = _validated_range(
            raw_contract.get("analysis_plausibility_range"),
            column=column,
        )
        raw_policy = raw_contract.get("plausibility_policy")
        if raw_policy is not None and not isinstance(raw_policy, Mapping):
            raise PlausibilityScopeError(
                f"plausibility_policy for {column!r} must be an object"
            )
        action = (
            str(raw_policy.get("out_of_range_action") or "").strip()
            if isinstance(raw_policy, Mapping)
            else ""
        )
        range_policy = (
            str(raw_policy.get("range_policy") or "").strip()
            if isinstance(raw_policy, Mapping)
            else ""
        )
        if plausibility_range is None:
            if action == "retain_and_flag":
                raise PlausibilityScopeError(
                    f"{column!r} declares retain_and_flag without a range"
                )
            continue
        if not action:
            raise PlausibilityScopeError(
                f"{column!r} has a plausibility range without an action"
            )
        if action != "retain_and_flag":
            continue
        if range_policy != "flag_only":
            raise PlausibilityScopeError(
                f"{column!r} declares retain_and_flag without flag_only policy"
            )
        expected.append(column)

    return FlagOnlyPlausibilityScope(
        step_id=str(step_id),
        expected_columns=tuple(sorted(expected)),
        source_contracts_sha256=declared_digest,
        authority_kind="resolved_raw_input_contracts",
    )


def _legacy_step_scope(
    *,
    context: ResearchContext,
    step: AnalysisStep,
) -> FlagOnlyPlausibilityScope:
    """Narrow legacy contexts to exact raw Planner inputs, never the full study."""

    raw_inputs = tuple(
        str(value).strip()
        for value in step.inputs or ()
        if isinstance(value, str) and str(value).strip() and ":" not in str(value)
    )
    descriptors = {
        str(descriptor.name): descriptor
        for descriptor in getattr(context, "variables", ())
    }
    expected = tuple(
        sorted(
            name
            for name in raw_inputs
            if name in descriptors and descriptors[name].valid_range is not None
        )
    )
    source_payload = {
        "schema_version": "easyicu.legacy_step_plausibility_inputs/1",
        "step_id": str(step.step_id),
        "raw_inputs": list(raw_inputs),
        "ranged_inputs": list(expected),
    }
    return FlagOnlyPlausibilityScope(
        step_id=str(step.step_id),
        expected_columns=expected,
        source_contracts_sha256=_canonical_sha256(source_payload),
        authority_kind="legacy_step_raw_inputs",
    )


def compile_flag_only_plausibility_scope(
    *,
    context: ResearchContext,
    step: AnalysisStep,
    raw_input_contracts: Optional[Mapping[str, Any]],
) -> FlagOnlyPlausibilityScope:
    """Compile the one immutable scope shared by every obligation consumer."""

    if raw_input_contracts is not None:
        if not isinstance(raw_input_contracts, Mapping):
            raise PlausibilityScopeError(
                "resolved raw-input contracts must be an object"
            )
        return _scope_from_resolved_raw_contracts(
            step_id=step.step_id,
            raw_input_contracts=raw_input_contracts,
        )
    return _legacy_step_scope(context=context, step=step)


def compile_step_plausibility_authority(
    *,
    context: ResearchContext,
    step: AnalysisStep,
    raw_input_contracts: Mapping[str, Any],
) -> StepPlausibilityAuthority:
    """Compile and freeze the one raw-contract value used by every consumer."""

    scope = compile_flag_only_plausibility_scope(
        context=context,
        step=step,
        raw_input_contracts=raw_input_contracts,
    )
    canonical = json.dumps(
        dict(raw_input_contracts),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return StepPlausibilityAuthority(
        scope=scope,
        raw_input_contracts_canonical_json=canonical,
    )


def compile_resumed_flag_only_plausibility_scope(
    *,
    prior_record: Mapping[str, Any],
    run_dir: Path,
    context: ResearchContext,
    step: AnalysisStep,
) -> FlagOnlyPlausibilityScope:
    """Recover scope from immutable capsule bytes, never checkpoint projection.

    A successful modern checkpoint must name the exact completed execution
    capsule and independently bind both its code and resolved-input bytes.
    Missing legacy coordinates invalidate revalidation instead of allowing an
    empty inferred scope to make a prior success current under a newer gate.
    """

    step_id = str(step.step_id)
    if str(prior_record.get("status") or "").strip().lower() != "ok":
        raise PlausibilityScopeError(
            "plausibility scope resume authority requires a successful checkpoint"
        )
    if str(prior_record.get("step_id") or "") != step_id:
        raise PlausibilityScopeError(
            "plausibility scope checkpoint belongs to a different analysis step"
        )

    recorded_code_sha256 = str(
        prior_record.get("executed_code_sha256") or ""
    ).strip()
    recorded_resolved_inputs_sha256 = str(
        prior_record.get("resolved_inputs_sha256") or ""
    ).strip()
    if _SHA256_PATTERN.fullmatch(recorded_code_sha256) is None:
        raise PlausibilityScopeError(
            "successful checkpoint lacks an executed-code SHA-256"
        )
    if _SHA256_PATTERN.fullmatch(recorded_resolved_inputs_sha256) is None:
        raise PlausibilityScopeError(
            "successful checkpoint lacks a resolved-inputs SHA-256"
        )

    raw_capsule_ref = prior_record.get("step_authority_capsule_ref")
    if raw_capsule_ref is None:
        raise PlausibilityScopeError(
            "successful checkpoint lacks sealed step authority"
        )
    if not isinstance(raw_capsule_ref, Mapping):
        raise PlausibilityScopeError(
            "step authority capsule reference is not an object"
        )
    try:
        capsule_ref = StepAuthorityCapsuleRef.model_validate(dict(raw_capsule_ref))
        verified = load_verified_step_authority_capsule(
            run_dir,
            ref=capsule_ref,
            expected_step_id=step_id,
        )
        capsule = verified.capsule
        execution = capsule.execution
        if capsule.stage not in _EXECUTED_CAPSULE_STAGES or execution is None:
            raise PlausibilityScopeError(
                "successful checkpoint does not select an executed capsule"
            )
        if (
            capsule.candidate_code.sha256 != recorded_code_sha256
            or execution.code_sha256 != recorded_code_sha256
        ):
            raise PlausibilityScopeError(
                "executed capsule code does not match the successful checkpoint"
            )
        if (
            capsule.resolved_inputs.sha256 != recorded_resolved_inputs_sha256
            or execution.resolved_inputs_sha256
            != recorded_resolved_inputs_sha256
        ):
            raise PlausibilityScopeError(
                "executed capsule inputs do not match the successful checkpoint"
            )
        if (
            execution.returncode != 0
            or execution.timed_out
            or not execution.outputs_safe_to_collect
        ):
            raise PlausibilityScopeError(
                "successful checkpoint selects an unsuccessful execution capsule"
            )
        resolved_inputs_payload = json.loads(
            read_verified_content(
                run_dir,
                capsule.resolved_inputs,
            ).decode("utf-8")
        )
    except PlausibilityScopeError:
        raise
    except (
        AttributeError,
        json.JSONDecodeError,
        StepAuthorityCapsuleError,
        UnicodeError,
        ValueError,
    ) as exc:
        raise PlausibilityScopeError(
            "sealed resolved-input authority cannot be verified"
        ) from exc
    if not isinstance(resolved_inputs_payload, Mapping):
        raise PlausibilityScopeError(
            "sealed resolved-input authority must be an object"
        )
    if str(resolved_inputs_payload.get("step_id") or "") != str(
        step.step_id
    ) or resolved_inputs_payload.get("schema_version") not in {
        "2.1",
        "easyicu.resolved_inputs/2",
    }:
        raise PlausibilityScopeError(
            "sealed resolved-input authority has the wrong schema or step"
        )
    if "raw_input_contracts" not in resolved_inputs_payload:
        raise PlausibilityScopeError(
            "sealed resolved-input authority lacks raw-input contracts"
        )
    raw_input_contracts = resolved_inputs_payload["raw_input_contracts"]
    if not isinstance(raw_input_contracts, Mapping):
        raise PlausibilityScopeError("sealed raw-input contracts must be an object")
    return compile_flag_only_plausibility_scope(
        context=context,
        step=step,
        raw_input_contracts=raw_input_contracts,
    )


def restore_revalidated_resolved_inputs_sha256(
    *,
    prior_record: Mapping[str, Any],
    checkpoint_history: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Recover only the immutable input digest lost by an old replay projection.

    A short-lived resume implementation removed ``resolved_inputs_sha256`` from
    a successful revalidation checkpoint along with mutable convenience
    receipts.  Recovery is allowed only for that explicit checkpoint shape and
    only from an earlier success naming the same executed capsule and code.  The
    caller must still pass the result through capsule verification.
    """

    restored = dict(prior_record)
    current_digest = str(restored.get("resolved_inputs_sha256") or "").strip()
    if current_digest or restored.get("revalidated_without_execution") is not True:
        return restored
    capsule_ref = restored.get("step_authority_capsule_ref")
    step_id = str(restored.get("step_id") or "")
    code_sha256 = str(restored.get("executed_code_sha256") or "")
    if not isinstance(capsule_ref, Mapping) or not step_id or not code_sha256:
        return restored
    for candidate in reversed(checkpoint_history):
        candidate_digest = str(
            candidate.get("resolved_inputs_sha256") or ""
        ).strip()
        if (
            _SHA256_PATTERN.fullmatch(candidate_digest) is not None
            and str(candidate.get("status") or "").strip().lower() == "ok"
            and str(candidate.get("step_id") or "") == step_id
            and candidate.get("step_authority_capsule_ref") == capsule_ref
            and str(candidate.get("executed_code_sha256") or "") == code_sha256
        ):
            restored["resolved_inputs_sha256"] = candidate_digest
            break
    return restored


__all__ = [
    "FlagOnlyPlausibilityScope",
    "PLAUSIBILITY_SCOPE_SCHEMA",
    "PlausibilityScopeError",
    "RAW_INPUT_CONTRACT_SCHEMA",
    "StepPlausibilityAuthority",
    "compile_flag_only_plausibility_scope",
    "compile_resumed_flag_only_plausibility_scope",
    "compile_step_plausibility_authority",
    "restore_revalidated_resolved_inputs_sha256",
]
