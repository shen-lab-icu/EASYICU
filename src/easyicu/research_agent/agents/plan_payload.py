"""Schema-driven projection of raw Planner payloads.

This module owns the boundary between untrusted model JSON and the strict
``AnalysisPlan`` schema.  It removes unsupported keys, records each removal,
and normalizes only representation-level aliases.  It never makes a clinical
choice, fills a scientific field, calls an LLM, or validates the final plan;
those responsibilities remain with the planner and schema owners.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Dict, List, Tuple

from ..contracts.declared_product import typed_product as _canonical_typed_product
from ..planning.robustness_contract import RobustnessSpec
from ..schema import (
    AnalysisPlan,
    AnalysisStep,
    ArtifactConsumptionContract,
    PlannedModelRequirement,
    TableOneSpec,
    TableOneVariableSpec,
)


def _canonicalise_figure_output_alias(token: object) -> object:
    """Canonicalize only a colon-typed alias for a declared figure product."""

    if not isinstance(token, str):
        return token
    parsed = _canonical_typed_product(token)
    if parsed is None or parsed[0] != "figure":
        return token
    _kind, _separator, name = token.partition(":")
    return f"figure:{name.strip()}"


def _is_untyped_figure_alias_output(token: object) -> bool:
    """Reject an underscore figure alias before it silently loses its role."""

    if not isinstance(token, str):
        return False
    text = token.strip()
    if not text or ":" in text:
        return False
    if text.lower().endswith((".png", ".svg", ".pdf", ".tif", ".tiff")):
        return False
    head, separator, _rest = text.partition("_")
    if not separator:
        return False
    probe = _canonical_typed_product(f"{head}:probe")
    return probe is not None and probe[0] == "figure"


def _canonicalise_planned_analysis_role(
    value: object,
    *,
    method: object,
) -> object:
    """Normalize a closed set of representation-only planner role variants."""

    if not isinstance(value, str):
        return value
    token = value.strip().casefold()
    if token in {"primary", "secondary", "sensitivity", "auxiliary"}:
        return token
    method_token = str(method or "").strip().casefold()
    if token == "robustness" and method_token == "robustness_sensitivity":
        return "sensitivity"
    return value


def _declared_field_names(model: type) -> set:
    """Read accepted fields from the declaring Pydantic model or dataclass."""

    fields = getattr(model, "model_fields", None)
    if fields is not None:
        return set(fields)
    if dataclasses.is_dataclass(model):
        return {field.name for field in dataclasses.fields(model)}
    raise TypeError(
        f"{model.__name__} declares neither pydantic model_fields nor "
        "dataclass fields, so its accepted Planner keys cannot be read "
        "from the schema; do not transcribe them by hand."
    )


def _normalise_plan_payload(
    data: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, List[str]]]:
    """Drop invented keys while preserving every field declared by the schema."""

    allowed_plan = _declared_field_names(AnalysisPlan)
    allowed_step = _declared_field_names(AnalysisStep)
    allowed_model_requirement = _declared_field_names(PlannedModelRequirement)
    allowed_consumption_contract = _declared_field_names(ArtifactConsumptionContract)
    allowed_table_one_spec = _declared_field_names(TableOneSpec)
    allowed_table_one_variable = _declared_field_names(TableOneVariableSpec)
    allowed_robustness_spec = _declared_field_names(RobustnessSpec)
    dropped: Dict[str, List[str]] = {
        "top_level": [],
        "steps": [],
        "model_requirements": [],
        "input_consumption_contracts": [],
        "table_one_spec": [],
        "robustness_specs": [],
    }
    out = {key: value for key, value in data.items() if key in allowed_plan}
    dropped["top_level"] = [
        str(key) for key in data if key not in allowed_plan
    ]
    steps = []
    for idx, raw_step in enumerate(out.get("steps", []) or []):
        if not isinstance(raw_step, dict):
            continue
        step_payload = {
            key: value for key, value in raw_step.items() if key in allowed_step
        }
        step_id = raw_step.get("step_id") or f"step[{idx}]"
        dropped["steps"].extend(
            f"{step_id}:{key}" for key in raw_step if key not in allowed_step
        )
        if "planned_analysis_role" in step_payload:
            step_payload["planned_analysis_role"] = _canonicalise_planned_analysis_role(
                step_payload["planned_analysis_role"],
                method=step_payload.get("method"),
            )
        requirements = []
        for req_idx, raw_requirement in enumerate(
            step_payload.get("model_requirements", []) or []
        ):
            if not isinstance(raw_requirement, dict):
                requirements.append(raw_requirement)
                continue
            requirement_payload = {
                key: value
                for key, value in raw_requirement.items()
                if key in allowed_model_requirement
            }
            requirement_id = (
                raw_requirement.get("requirement_id")
                or f"step[{idx}].model_requirements[{req_idx}]"
            )
            dropped["model_requirements"].extend(
                f"{requirement_id}:{key}"
                for key in raw_requirement
                if key not in allowed_model_requirement
            )
            if "analysis_role" in requirement_payload:
                requirement_payload["analysis_role"] = _canonicalise_planned_analysis_role(
                    requirement_payload["analysis_role"],
                    method=step_payload.get("method"),
                )
            requirements.append(requirement_payload)
        if "model_requirements" in step_payload:
            step_payload["model_requirements"] = requirements
        consumption_contracts = []
        for contract_idx, raw_contract in enumerate(
            step_payload.get("input_consumption_contracts", []) or []
        ):
            if not isinstance(raw_contract, dict):
                consumption_contracts.append(raw_contract)
                continue
            contract_payload = {
                key: value
                for key, value in raw_contract.items()
                if key in allowed_consumption_contract
            }
            contract_id = (
                raw_contract.get("input_key")
                or f"step[{idx}].input_consumption_contracts[{contract_idx}]"
            )
            dropped["input_consumption_contracts"].extend(
                f"{contract_id}:{key}"
                for key in raw_contract
                if key not in allowed_consumption_contract
            )
            if contract_payload:
                consumption_contracts.append(contract_payload)
            else:
                dropped["input_consumption_contracts"].append(
                    f"{contract_id}:empty_after_normalization"
                )
        if "input_consumption_contracts" in step_payload:
            step_payload["input_consumption_contracts"] = consumption_contracts
        raw_table_one = step_payload.get("table_one_spec")
        if isinstance(raw_table_one, dict):
            table_one_payload = {
                key: value
                for key, value in raw_table_one.items()
                if key in allowed_table_one_spec
            }
            dropped["table_one_spec"].extend(
                f"step[{idx}]:{key}"
                for key in raw_table_one
                if key not in allowed_table_one_spec
            )
            variables = []
            for variable_index, raw_variable in enumerate(
                table_one_payload.get("variables", []) or []
            ):
                if not isinstance(raw_variable, dict):
                    variables.append(raw_variable)
                    continue
                variable_payload = {
                    key: value
                    for key, value in raw_variable.items()
                    if key in allowed_table_one_variable
                }
                dropped["table_one_spec"].extend(
                    f"step[{idx}].variables[{variable_index}]:{key}"
                    for key in raw_variable
                    if key not in allowed_table_one_variable
                )
                variables.append(variable_payload)
            table_one_payload["variables"] = variables
            step_payload["table_one_spec"] = table_one_payload
        raw_outputs = step_payload.get("expected_outputs")
        if isinstance(raw_outputs, list):
            normalised_outputs: List[Any] = []
            for item in raw_outputs:
                if _is_untyped_figure_alias_output(item):
                    suggested = str(item).strip().partition("_")[2]
                    raise ValueError(
                        f"Planner step {step_id!r} declares figure output "
                        f"{item!r} with an underscore instead of the typed "
                        "'figure:' separator; re-emit it as "
                        f"'figure:{suggested}' so the declared figure binds "
                        "to an exact output file."
                    )
                normalised_outputs.append(_canonicalise_figure_output_alias(item))
            figure_identity_aliases: Dict[Tuple[str, str], List[str]] = {}
            for candidate in normalised_outputs:
                if not isinstance(candidate, str):
                    continue
                identity = _canonical_typed_product(candidate)
                if identity is None or identity[0] != "figure":
                    continue
                figure_identity_aliases.setdefault(identity, []).append(candidate)
            collisions = {
                identity: aliases
                for identity, aliases in figure_identity_aliases.items()
                if len(aliases) > 1
            }
            if collisions:
                detail = "; ".join(
                    f"figure:{product} declared as {sorted(set(aliases))}"
                    for (_kind, product), aliases in sorted(collisions.items())
                )
                raise ValueError(
                    f"Planner step {step_id!r} declares the same figure "
                    f"product under more than one output alias ({detail}); "
                    "declare each figure exactly once as 'figure:<name>'."
                )
            step_payload["expected_outputs"] = normalised_outputs
        steps.append(step_payload)
    out["steps"] = steps
    specs = []
    for idx, raw_spec in enumerate(out.get("robustness_specs", []) or []):
        if not isinstance(raw_spec, dict):
            specs.append(raw_spec)
            continue
        spec_payload = {
            key: value
            for key, value in raw_spec.items()
            if key in allowed_robustness_spec
        }
        spec_id = raw_spec.get("spec_id") or f"robustness_specs[{idx}]"
        dropped["robustness_specs"].extend(
            f"{spec_id}:{key}"
            for key in raw_spec
            if key not in allowed_robustness_spec
        )
        specs.append(spec_payload)
    if "robustness_specs" in out:
        out["robustness_specs"] = specs
    return out, dropped


__all__ = [
    "_canonicalise_figure_output_alias",
    "_canonicalise_planned_analysis_role",
    "_declared_field_names",
    "_is_untyped_figure_alias_output",
    "_normalise_plan_payload",
]
