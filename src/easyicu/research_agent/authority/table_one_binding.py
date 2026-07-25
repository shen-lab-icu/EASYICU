"""Host-only binding for private Table 1 categorical levels.

The public :class:`AnalysisPlan` retains the opaque tokens emitted by the
Planner.  This module binds those tokens to locally observed values for trusted
execution and output validation without adding the values to any serialised
plan or provider prompt.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..methods.table_one import table_one_spec_sha256
from ..research_context.prompt_variables import opaque_level_tokens
from ..schema import AnalysisPlan, AnalysisStep, ResearchContext, TableOneSpec

TABLE_ONE_EXECUTION_BINDING_SCHEMA = "easyicu.table_one_execution_binding/1"
TABLE_ONE_PRIVATE_CHECKPOINT_SCHEMA = "easyicu.table_one_private_checkpoint/1"
TABLE_ONE_PRIVATE_CHECKPOINT_RELATIVE_PATH = Path(
    ".runtime/table_one_private_checkpoint.json"
)


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


class TableOneExecutionBinding(BaseModel):
    """Digest-bound local execution view; never serialize into agent context."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal["easyicu.table_one_execution_binding/1"]
    step_id: str
    planner_spec_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    observed_domain_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    token_secret_hex: str = Field(pattern=r"^[0-9a-f]{64}$")
    execution_spec: TableOneSpec
    execution_spec_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    binding_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def _verify_digests(self) -> "TableOneExecutionBinding":
        if table_one_spec_sha256(self.execution_spec) != self.execution_spec_sha256:
            raise ValueError("Table 1 execution-spec digest mismatch")
        payload = self.model_dump(mode="json", exclude={"binding_sha256"})
        expected = hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()
        if self.binding_sha256 != expected:
            raise ValueError("Table 1 execution-binding digest mismatch")
        return self


def _observed_levels(
    *,
    name: str,
    variables: dict[str, Any],
) -> list[Any]:
    variable = variables.get(name)
    if variable is None or not variable.observed_domain:
        return []
    domain = variable.observed_domain
    levels = domain.get("levels")
    if isinstance(levels, list):
        return list(levels)
    if not domain.get("is_binary"):
        return []
    dtype = str(variable.dtype or "").lower()
    if dtype.startswith(("int", "uint")):
        return [0, 1]
    if dtype.startswith("bool"):
        return [False, True]
    if dtype.startswith(("float", "double")):
        return [0.0, 1.0]
    return []


def _typed_token(value: Any) -> tuple[str, str]:
    return type(value).__name__, repr(value)


def _resolve_levels(
    *,
    name: str,
    declared: list[Any],
    variables: dict[str, Any],
) -> tuple[list[Any], list[Any]]:
    observed = _observed_levels(name=name, variables=variables)
    if not observed:
        return list(declared), []
    opaque = list(opaque_level_tokens(len(observed)))
    if opaque and list(declared) == opaque:
        return observed, observed
    if {_typed_token(value) for value in declared} == {
        _typed_token(value) for value in observed
    }:
        return list(declared), observed
    # JSON has one ``number`` type even though Python/pandas distinguish
    # integral and floating scalar representations.  A Planner declaration
    # of ``[0, 1]`` therefore denotes the same closed binary domain as a
    # float-backed cohort column whose verified levels are ``[0.0, 1.0]``.
    # Canonicalise execution back to the host-observed scalar types; never
    # apply this equivalence to booleans or categorical strings.
    declared_numeric = all(
        isinstance(value, (int, float)) and not isinstance(value, bool)
        for value in declared
    )
    observed_numeric = all(
        isinstance(value, (int, float)) and not isinstance(value, bool)
        for value in observed
    )
    if (
        declared_numeric
        and observed_numeric
        and len(declared) == len(observed)
        and {float(value) for value in declared} == {float(value) for value in observed}
    ):
        return observed, observed
    safe_expected = opaque or ["<host-observed numeric scalar types>"]
    raise ValueError(
        "Planner Table 1 levels for "
        f"{name!r} must preserve the exact observed scalar types or use the "
        f"exact host-safe tokens {safe_expected!r}; expected_count="
        f"{len(observed)}, declared_count={len(declared)}, declared_types="
        f"{[type(value).__name__ for value in declared]!r}. No observed "
        "category literal is available to the Provider."
    )


def bind_table_one_execution_spec(
    step: AnalysisStep,
    context: ResearchContext,
    *,
    token_secret_hex: str | None = None,
) -> TableOneExecutionBinding | None:
    """Attach and return the local binding without mutating the public spec."""

    planner_spec = step.table_one_spec
    if planner_spec is None:
        step._table_one_execution_binding = None
        return None
    variables = {variable.name: variable for variable in context.variables}
    payload = planner_spec.model_dump(mode="python")
    group_levels, observed_groups = _resolve_levels(
        name=planner_spec.group_by,
        declared=list(planner_spec.group_levels),
        variables=variables,
    )
    payload["group_levels"] = group_levels
    observed_payload: dict[str, Any] = {planner_spec.group_by: observed_groups}
    for index, variable_spec in enumerate(planner_spec.variables):
        if variable_spec.summary != "count_percent":
            continue
        levels, observed = _resolve_levels(
            name=variable_spec.name,
            declared=list(variable_spec.levels),
            variables=variables,
        )
        payload["variables"][index]["levels"] = levels
        observed_payload[variable_spec.name] = observed
    execution_spec = TableOneSpec.model_validate(payload)
    observed_domain_sha256 = hashlib.sha256(
        _canonical_json(observed_payload).encode("utf-8")
    ).hexdigest()
    planner_spec_sha256 = table_one_spec_sha256(planner_spec)
    secret_coordinate = hashlib.sha256(
        _canonical_json(
            {
                "step_id": step.step_id,
                "planner_spec_sha256": planner_spec_sha256,
                "observed_domain_sha256": observed_domain_sha256,
            }
        ).encode("utf-8")
    ).hexdigest()
    private_secrets = context._table_one_token_secrets
    if token_secret_hex is None:
        token_secret_hex = private_secrets.get(secret_coordinate)
    if token_secret_hex is None:
        token_secret_hex = secrets.token_hex(32)
    if (
        not isinstance(token_secret_hex, str)
        or not all(char in "0123456789abcdef" for char in token_secret_hex)
        or len(token_secret_hex) != 64
    ):
        raise ValueError("Table 1 token secret must be 32-byte lowercase hex")
    prior_secret = private_secrets.get(secret_coordinate)
    if prior_secret is not None and prior_secret != token_secret_hex:
        raise ValueError("Table 1 private checkpoint secret mismatch")
    private_secrets[secret_coordinate] = token_secret_hex
    binding_payload: dict[str, Any] = {
        "schema_version": TABLE_ONE_EXECUTION_BINDING_SCHEMA,
        "step_id": step.step_id,
        "planner_spec_sha256": planner_spec_sha256,
        "observed_domain_sha256": observed_domain_sha256,
        "token_secret_hex": token_secret_hex,
        "execution_spec": execution_spec.model_dump(mode="json"),
        "execution_spec_sha256": table_one_spec_sha256(execution_spec),
    }
    binding_payload["binding_sha256"] = hashlib.sha256(
        _canonical_json(binding_payload).encode("utf-8")
    ).hexdigest()
    binding = TableOneExecutionBinding.model_validate(binding_payload, strict=True)
    step._table_one_execution_binding = binding
    return binding


def table_one_execution_spec(step: AnalysisStep) -> TableOneSpec | None:
    """Return a verified local spec, falling back to the public declaration."""

    planner_spec = step.table_one_spec
    binding = step._table_one_execution_binding
    if not isinstance(binding, TableOneExecutionBinding):
        return planner_spec
    if planner_spec is None or binding.step_id != step.step_id:
        raise ValueError("stale Table 1 execution binding")
    if binding.planner_spec_sha256 != table_one_spec_sha256(planner_spec):
        raise ValueError("Table 1 execution binding does not match the public plan")
    return binding.execution_spec


def table_one_private_label_map(step: AnalysisStep) -> dict[tuple[str, str], str]:
    """Map local execution labels to public opaque tokens for outbound copies."""

    binding = step._table_one_execution_binding
    planner = step.table_one_spec
    if not isinstance(binding, TableOneExecutionBinding) or planner is None:
        return {}
    if binding.planner_spec_sha256 != table_one_spec_sha256(planner):
        raise ValueError("stale Table 1 execution binding")
    pairs: list[tuple[Any, Any]] = list(
        zip(binding.execution_spec.group_levels, planner.group_levels)
    )
    for execution_variable, planner_variable in zip(
        binding.execution_spec.variables, planner.variables
    ):
        pairs.extend(zip(execution_variable.levels, planner_variable.levels))
    return {
        (type(private).__name__, repr(private)): str(public)
        for private, public in pairs
        if private != public
    }


def table_one_private_code_label_map(
    step: AnalysisStep,
) -> dict[tuple[str, str], str]:
    """Map private labels to unique reversible tokens for outbound code."""

    private = table_one_private_label_map(step)
    binding = step._table_one_execution_binding
    if not isinstance(binding, TableOneExecutionBinding):
        return {}
    token_key = bytes.fromhex(binding.token_secret_hex)
    return {
        typed_value: "__easyicu_table1_label_"
        + hmac.new(
            token_key,
            _canonical_json(
                {
                    "step_id": step.step_id,
                    "type": typed_value[0],
                    "repr": typed_value[1],
                    "public": public,
                }
            ).encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()[:16]
        + "__"
        for typed_value, public in private.items()
    }


def table_one_code_token_value_map(step: AnalysisStep) -> dict[str, Any]:
    """Return the inverse unique-token mapping for host-only restoration."""

    binding = step._table_one_execution_binding
    planner = step.table_one_spec
    if not isinstance(binding, TableOneExecutionBinding) or planner is None:
        return {}
    code_tokens = table_one_private_code_label_map(step)
    values: list[Any] = list(binding.execution_spec.group_levels)
    for variable in binding.execution_spec.variables:
        values.extend(variable.levels)
    restored: dict[str, Any] = {}
    for value in values:
        token = code_tokens.get((type(value).__name__, repr(value)))
        if token is not None:
            restored[token] = value
    return restored


def _private_checkpoint_payload(plan: AnalysisPlan) -> dict[str, Any]:
    steps: list[dict[str, str]] = []
    for step in plan.steps:
        binding = step._table_one_execution_binding
        if not isinstance(binding, TableOneExecutionBinding):
            if step.table_one_spec is not None:
                raise ValueError(
                    f"Table 1 step {step.step_id!r} lacks a private execution binding"
                )
            continue
        if not table_one_private_label_map(step):
            continue
        steps.append(
            {
                "step_id": step.step_id,
                "planner_spec_sha256": binding.planner_spec_sha256,
                "observed_domain_sha256": binding.observed_domain_sha256,
                "token_secret_hex": binding.token_secret_hex,
                "binding_sha256": binding.binding_sha256,
            }
        )
    payload: dict[str, Any] = {
        "schema_version": TABLE_ONE_PRIVATE_CHECKPOINT_SCHEMA,
        "steps": steps,
    }
    payload["checkpoint_sha256"] = hashlib.sha256(
        _canonical_json(payload).encode("utf-8")
    ).hexdigest()
    return payload


def write_table_one_private_checkpoint(*, run_dir: Path, plan: AnalysisPlan) -> Path:
    """Persist random token secrets outside the public plan with mode 0600."""

    path = Path(run_dir) / TABLE_ONE_PRIVATE_CHECKPOINT_RELATIVE_PATH
    payload = _private_checkpoint_payload(plan)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(_canonical_json(payload), encoding="utf-8")
    os.chmod(temporary, 0o600)
    temporary.replace(path)
    os.chmod(path, 0o600)
    return path


def restore_table_one_private_checkpoint(
    *,
    run_dir: Path,
    plan: AnalysisPlan,
    context: ResearchContext,
) -> None:
    """Restore and rebind Table 1 tokens from the private runtime checkpoint."""

    def _uses_opaque_levels(step: AnalysisStep) -> bool:
        spec = step.table_one_spec
        if spec is None:
            return False
        levels = [*spec.group_levels]
        for variable in spec.variables:
            levels.extend(variable.levels)
        return any(
            isinstance(value, str) and value.startswith("__easyicu_level_")
            for value in levels
        )

    table_steps = [step for step in plan.steps if _uses_opaque_levels(step)]
    if not table_steps:
        return
    path = Path(run_dir) / TABLE_ONE_PRIVATE_CHECKPOINT_RELATIVE_PATH
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError) as exc:
        raise ValueError("Table 1 private checkpoint is missing or invalid") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != (
        TABLE_ONE_PRIVATE_CHECKPOINT_SCHEMA
    ):
        raise ValueError("Table 1 private checkpoint schema mismatch")
    checkpoint_sha256 = payload.pop("checkpoint_sha256", None)
    expected_checkpoint_sha256 = hashlib.sha256(
        _canonical_json(payload).encode("utf-8")
    ).hexdigest()
    if checkpoint_sha256 != expected_checkpoint_sha256:
        raise ValueError("Table 1 private checkpoint digest mismatch")
    rows = payload.get("steps")
    if not isinstance(rows, list):
        raise ValueError("Table 1 private checkpoint steps are invalid")
    by_step: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("step_id"), str):
            raise ValueError("Table 1 private checkpoint row is invalid")
        step_id = row["step_id"]
        if step_id in by_step:
            raise ValueError("Table 1 private checkpoint repeats a step")
        by_step[step_id] = row
    if set(by_step) != {step.step_id for step in table_steps}:
        raise ValueError("Table 1 private checkpoint does not match the plan")
    for step in table_steps:
        row = by_step[step.step_id]
        binding = bind_table_one_execution_spec(
            step,
            context,
            token_secret_hex=row.get("token_secret_hex"),
        )
        if binding is None or any(
            row.get(field) != getattr(binding, field)
            for field in (
                "planner_spec_sha256",
                "observed_domain_sha256",
                "binding_sha256",
            )
        ):
            raise ValueError("Table 1 private checkpoint authority mismatch")


__all__ = [
    "TABLE_ONE_EXECUTION_BINDING_SCHEMA",
    "TABLE_ONE_PRIVATE_CHECKPOINT_RELATIVE_PATH",
    "TABLE_ONE_PRIVATE_CHECKPOINT_SCHEMA",
    "TableOneExecutionBinding",
    "bind_table_one_execution_spec",
    "table_one_private_label_map",
    "table_one_private_code_label_map",
    "table_one_code_token_value_map",
    "table_one_execution_spec",
    "restore_table_one_private_checkpoint",
    "write_table_one_private_checkpoint",
]
