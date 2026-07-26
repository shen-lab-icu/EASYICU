"""Versioned scientific identity for proving that a dropped step returned.

This contract is narrower than the immutable-plan tamper signature: a
replanner may reword ``intent`` prose, but it may not silently change any
structured execution input, method, scientific role, product, ICU rule, model
requirement, input-consumption rule, Table 1 design, or trajectory-stability
design and still claim that the original step was recovered.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from ..canonical_json import canonical_json, canonical_sha256
from ..schema import AnalysisStep


def _canonical_strings(values: list[str]) -> tuple[str, ...]:
    return tuple(sorted({str(value).strip() for value in values if str(value).strip()}))


def _canonical_model_payloads(values: list[Any]) -> tuple[dict[str, Any], ...]:
    payloads = [value.model_dump(mode="json") for value in values]
    return tuple(sorted(payloads, key=canonical_json))


class StepRecoverySignature(BaseModel):
    """Structured, inspectable identity used only for truncation recovery."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["easyicu.step_recovery_signature/1"] = (
        "easyicu.step_recovery_signature/1"
    )
    step_id: str
    planned_analysis_role: str
    method: str
    inputs: tuple[str, ...]
    expected_outputs: tuple[str, ...]
    icu_rule_refs: tuple[str, ...]
    model_requirements: tuple[dict[str, Any], ...]
    input_consumption_contracts: tuple[dict[str, Any], ...]
    table_one_spec: dict[str, Any] | None
    trajectory_stability_spec: dict[str, Any] | None

    @classmethod
    def from_step(cls, step: AnalysisStep) -> "StepRecoverySignature":
        """Project every structured scientific field except free-text intent."""

        return cls(
            step_id=str(step.step_id or "").strip(),
            planned_analysis_role=str(step.planned_analysis_role or "").strip(),
            method=str(step.method or "").strip(),
            inputs=_canonical_strings(step.inputs),
            expected_outputs=_canonical_strings(step.expected_outputs),
            icu_rule_refs=_canonical_strings(step.icu_rule_refs),
            model_requirements=_canonical_model_payloads(step.model_requirements),
            input_consumption_contracts=_canonical_model_payloads(
                step.input_consumption_contracts
            ),
            table_one_spec=(
                step.table_one_spec.model_dump(mode="json")
                if step.table_one_spec is not None
                else None
            ),
            trajectory_stability_spec=(
                step.trajectory_stability_spec.model_dump(mode="json")
                if step.trajectory_stability_spec is not None
                else None
            ),
        )

    def canonical_digest(self) -> str:
        """Hash this protocol through the shared canonical JSON owner."""

        return canonical_sha256(self.model_dump(mode="json"))


__all__ = ["StepRecoverySignature"]
