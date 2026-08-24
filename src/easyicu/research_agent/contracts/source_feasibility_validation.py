"""Typed validation for a deterministic, fail-closed source-feasibility result."""

from __future__ import annotations

import csv
from pathlib import Path
import re
from typing import Any, Literal, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .capability_ids import SOURCE_FEASIBILITY_ANALYSIS_KIND

_CONTRACT_REF = re.compile(r"^scientific_runtime_contract:([0-9a-f]{64})$")


class SourceFeasibilityRuntimeReceipt(BaseModel):
    """Evidence that a signed source audit prohibited an unidentified contrast."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["easyicu.source_feasibility_runtime_receipt/1"]
    protocol_content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    execution_contract_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    runtime_projection_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source: str = Field(min_length=1)
    window_start_hours: int
    window_end_hours: int
    verified_non_use_available: Literal[False]
    binary_control_arm_authorized: Literal[False]
    causal_contrast_authorized: Literal[False]
    decision: Literal["fail_closed"]
    reason_code: str = Field(min_length=1)
    effect_estimate: None
    forbidden_actions_enforced: list[str] = Field(min_length=1)
    future_design_authorized: Literal[False]

    @model_validator(mode="after")
    def _coherent_refusal(self) -> "SourceFeasibilityRuntimeReceipt":
        if self.window_end_hours <= self.window_start_hours:
            raise ValueError("source-feasibility window must be ordered")
        if len(self.forbidden_actions_enforced) != len(
            set(self.forbidden_actions_enforced)
        ):
            raise ValueError("forbidden source-feasibility actions must be unique")
        if any(not value.strip() for value in self.forbidden_actions_enforced):
            raise ValueError("forbidden source-feasibility action is empty")
        return self


def source_feasibility_plan_claimed(plan: object) -> bool:
    steps = tuple(getattr(plan, "steps", ()) or ())
    return len(steps) == 1 and str(getattr(steps[0], "method", "") or "") == (
        SOURCE_FEASIBILITY_ANALYSIS_KIND
    )


def source_feasibility_plan_contract_errors(plan: object) -> list[str]:
    """Validate the generic no-effect plan shape without reading case prose."""

    if not source_feasibility_plan_claimed(plan):
        return ["source-feasibility plan requires exactly one signed owner step"]
    step = tuple(getattr(plan, "steps", ()) or ())[0]
    errors: list[str] = []
    if getattr(step, "planned_analysis_role", None) != "auxiliary":
        errors.append("source-feasibility owner must be auxiliary, not a primary effect")
    if tuple(getattr(step, "inputs", ()) or ()):
        errors.append("source-feasibility owner must not consume an effect-analysis input")
    outputs = tuple(getattr(step, "expected_outputs", ()) or ())
    if (
        len(outputs) != 2
        or sum(str(value).startswith("table:") for value in outputs) != 1
        or sum(str(value).startswith("log:") for value in outputs) != 1
    ):
        errors.append("source-feasibility owner has an invalid output contract")
    if tuple(getattr(step, "model_requirements", ()) or ()) or getattr(
        step, "family_primary_result_requirement", None
    ) is not None:
        errors.append("source-feasibility owner must not declare an effect model")
    refs = tuple(getattr(step, "icu_rule_refs", ()) or ())
    if len(refs) != 1 or _CONTRACT_REF.fullmatch(str(refs[0])) is None:
        errors.append("source-feasibility owner lacks one signed runtime contract")
    return errors


def _summary_records(
    records: Sequence[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    return [
        record
        for record in records
        if record.get("deterministic_standard_analysis")
        == SOURCE_FEASIBILITY_ANALYSIS_KIND
        and isinstance(record.get("step_summary"), Mapping)
    ]


def _parse_false(value: Any) -> bool:
    return str(value).strip().lower() in {"false", "0"}


def source_feasibility_runtime_bundle_errors(
    *,
    plan: object,
    records: Sequence[Mapping[str, Any]],
    run_dir: Path,
) -> list[str]:
    """Validate the signed refusal receipt, summary and materialized table."""

    errors = source_feasibility_plan_contract_errors(plan)
    if errors:
        return errors
    matches = _summary_records(records)
    if len(matches) != 1:
        return [
            "source-feasibility validator requires exactly one deterministic receipt"
        ]
    record = matches[0]
    summary = record["step_summary"]
    try:
        receipt = SourceFeasibilityRuntimeReceipt.model_validate(
            summary.get("scientific_runtime_receipt")
        )
    except Exception as exc:
        return [f"source-feasibility runtime receipt is invalid: {exc}"]

    step = tuple(getattr(plan, "steps", ()) or ())[0]
    contract_match = _CONTRACT_REF.fullmatch(str(step.icu_rule_refs[0]))
    assert contract_match is not None
    if receipt.execution_contract_sha256 != contract_match.group(1):
        errors.append("source-feasibility receipt disagrees with the signed plan")
    if not (
        record.get("status") == "ok"
        and summary.get("status") == "ok"
        and summary.get("analysis_family") == "causal_feasibility"
        and summary.get("scientific_decision") == "blocked_by_source_authority"
        and summary.get("reason_code") == receipt.reason_code
        and summary.get("causal_contrast_authorized") is False
        and summary.get("effect_estimate") is None
    ):
        errors.append("source-feasibility summary does not encode the sealed refusal")

    output_files = summary.get("output_files")
    table_products = [
        (str(product), str(filename))
        for product, filename in (
            output_files.items() if isinstance(output_files, Mapping) else ()
        )
        if str(product).startswith("table:")
    ]
    if not isinstance(output_files, Mapping) or set(output_files) != set(
        getattr(step, "expected_outputs", ()) or ()
    ):
        errors.append("source-feasibility output files disagree with the signed plan")
    table_filename = table_products[0][1] if len(table_products) == 1 else ""
    if not table_filename or Path(table_filename).name != table_filename:
        errors.append("source-feasibility table path is not a safe output filename")
        table_filename = ""
    table_path = (
        Path(run_dir)
        / "steps"
        / str(getattr(step, "step_id", ""))
        / "outputs"
        / table_filename
    )
    try:
        with table_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        row = rows[0]
        table_matches = (
            len(rows) == 1
            and row.get("source") == receipt.source
            and int(row.get("window_start_hours", "")) == receipt.window_start_hours
            and int(row.get("window_end_hours", "")) == receipt.window_end_hours
            and _parse_false(row.get("verified_non_use_available"))
            and _parse_false(row.get("binary_control_arm_authorized"))
            and _parse_false(row.get("causal_contrast_authorized"))
            and row.get("decision") == receipt.decision
            and row.get("reason_code") == receipt.reason_code
            and not str(row.get("effect_estimate") or "").strip()
        )
    except (IndexError, OSError, TypeError, ValueError):
        table_matches = False
    if not table_matches:
        errors.append("source-feasibility table disagrees with its refusal receipt")
    return errors


__all__ = [
    "SourceFeasibilityRuntimeReceipt",
    "source_feasibility_plan_claimed",
    "source_feasibility_plan_contract_errors",
    "source_feasibility_runtime_bundle_errors",
]
