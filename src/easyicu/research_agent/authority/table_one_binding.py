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
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..methods.table_one import table_one_spec_sha256
from ..research_context.prompt_variables import opaque_level_tokens
from ..schema import AnalysisStep, ResearchContext, TableOneSpec

TABLE_ONE_EXECUTION_BINDING_SCHEMA = "easyicu.table_one_execution_binding/1"


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
    token_key_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
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
    raise ValueError(
        "Planner Table 1 levels must preserve the exact observed scalar types "
        "or the exact opaque level tokens supplied by the host"
    )


def bind_table_one_execution_spec(
    step: AnalysisStep,
    context: ResearchContext,
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
    token_key_sha256 = hashlib.sha256(
        _canonical_json(
            {
                "schema": "easyicu.table_one_host_token_key/1",
                "context_created_at": context.created_at.isoformat(),
                "step_id": step.step_id,
                "planner_spec_sha256": planner_spec_sha256,
                "observed_domain_sha256": observed_domain_sha256,
            }
        ).encode("utf-8")
    ).hexdigest()
    binding_payload: dict[str, Any] = {
        "schema_version": TABLE_ONE_EXECUTION_BINDING_SCHEMA,
        "step_id": step.step_id,
        "planner_spec_sha256": planner_spec_sha256,
        "observed_domain_sha256": observed_domain_sha256,
        "token_key_sha256": token_key_sha256,
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
    token_key = bytes.fromhex(binding.token_key_sha256)
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


__all__ = [
    "TABLE_ONE_EXECUTION_BINDING_SCHEMA",
    "TableOneExecutionBinding",
    "bind_table_one_execution_spec",
    "table_one_private_label_map",
    "table_one_private_code_label_map",
    "table_one_code_token_value_map",
    "table_one_execution_spec",
]
