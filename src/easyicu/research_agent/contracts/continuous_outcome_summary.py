"""Exact secondary continuous-outcome summary contract, without inference."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from ..authority.declared_levels import closed_planning_levels_for
from ..research_context.typed import declares_factor_values

from .cohort_product_keys import sole_typed_cohort_input
from .ownership_verdict import OwnershipVerdict
from .table_one_semantics import table_one_variable_kind

CONTINUOUS_OUTCOME_SUMMARY = "continuous_outcome_summary"
CONTINUOUS_OUTCOME_OUTPUT = "table:continuous_outcome_summary"


def continuous_outcome_columns(step: Any) -> tuple[str, ...]:
    """The first raw input is the grouping; all subsequent inputs are outcomes."""
    return tuple(str(value) for value in step.inputs if ":" not in str(value))


def continuous_outcome_summary_verdict(step: Any) -> OwnershipVerdict:
    """A malformed declaration must not fall through to generated analysis."""
    method = str(getattr(step, "method", "") or "").strip().casefold()
    outputs = list(getattr(step, "expected_outputs", ()) or ())
    if method != CONTINUOUS_OUTCOME_SUMMARY and CONTINUOUS_OUTCOME_OUTPUT not in outputs:
        return OwnershipVerdict.wrong_shape(CONTINUOUS_OUTCOME_SUMMARY, reason="a different product")
    columns = continuous_outcome_columns(step)
    cohort = sole_typed_cohort_input(step)
    foreign = (
        "model_requirements", "family_primary_result_requirement", "scientific_capability",
        "table_one_spec", "exposure_outcome_distribution_spec", "trajectory_stability_spec",
        "cohort_definition_spec", "measurement_audit_spec", "robustness_replay_spec",
        "functional_form_spec", "scientific_action_id",
    )
    valid = (
        method == CONTINUOUS_OUTCOME_SUMMARY
        and outputs == [CONTINUOUS_OUTCOME_OUTPUT]
        and getattr(step, "planned_analysis_role", None) == "secondary"
        and cohort is not None
        and 2 <= len(columns) <= 65 and len(set(columns)) == len(columns)
        and all(name.strip() == name and name for name in columns)
        and len(step.inputs) == len(columns) + 1
        and not any(getattr(step, key, None) for key in foreign)
    )
    if not valid:
        return OwnershipVerdict.incomplete_declaration(
            CONTINUOUS_OUTCOME_SUMMARY, missing=("continuous_outcome_summary_contract",),
            reason="Declare a secondary summary, one cohort, one grouping then continuous outcomes, and its sole table; no model or effect contract.",
        )
    return OwnershipVerdict.claim(CONTINUOUS_OUTCOME_SUMMARY, reason="exact secondary grouped summaries without inference")


def continuous_summary_columns(
    *, grouping: str, raw_inputs: Sequence[str], variables: Mapping[str, Any],
    planning_context: Any = None,
) -> tuple[str, ...]:
    """Order a declared roster; never add, infer, or drop an outcome."""
    if (not grouping or grouping not in raw_inputs or len(raw_inputs) < 2
            or len(raw_inputs) != len(set(raw_inputs))
            or not set(raw_inputs).issubset(variables)):
        raise ValueError("continuous summary needs an explicit grouping and unique available outcomes")
    levels = closed_planning_levels_for(name=grouping, variables=dict(variables))
    cohort = getattr(planning_context, "cohort", None)
    provenance = getattr(cohort, "provenance", None) or {}
    deferred_factor = (
        not levels and declares_factor_values(variables[grouping])
        and getattr(cohort, "n_stays", None) == 0
        and provenance.get("evidence_stage") == "metadata_only_planning"
        and provenance.get("patient_rows_read") is False
    )
    if (str(variables[grouping].role.value) in {"id", "index", "meta"}
            or (len(levels) < 2 and not deferred_factor)):
        raise ValueError("continuous summary grouping needs a closed domain with at least two levels")
    outcomes = tuple(name for name in raw_inputs if name != grouping)
    for name in outcomes:
        variable = variables[name]
        levels = closed_planning_levels_for(name=name, variables=dict(variables))
        dtype = str(variable.dtype or "").casefold()
        if (str(variable.role.value) != "outcome"
                or not dtype.startswith(("float", "double", "int", "uint", "decimal"))
                or table_one_variable_kind(variable, levels) != "continuous"):
            raise ValueError(f"continuous summary requires a typed numeric continuous outcome: {name}")
    return (grouping, *outcomes)
