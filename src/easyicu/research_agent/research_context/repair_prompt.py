"""PHI-minimized research-context projection for repair transports."""

from __future__ import annotations

import json

from ..schema import ResearchContext
from .outbound import (
    outbound_safe_context_payload,
)

_MECHANICAL_COHORT_KEYS = frozenset(
    {"cohort_name", "database", "n_stays", "n_patients"}
)
_MECHANICAL_VARIABLE_KEYS = frozenset(
    {
        "name",
        "source_concept",
        "role",
        "dtype",
        "is_ordinal",
        "ordinal_cardinality",
    }
)


def format_repair_authority_context(
    ctx: ResearchContext,
    *,
    include_scientific_authority: bool,
    user_notes: str = "",
) -> str:
    """Render compact authority coordinates without observed cohort literals."""

    # Free-form notes remain outside provider repair transports. Scientific
    # repairs receive the complete deny-by-default outbound projection.
    del user_notes
    if include_scientific_authority:
        return json.dumps(
            {**outbound_safe_context_payload(ctx), **_input_coding(ctx)},
            ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        )

    # A typed mechanical ticket already carries its exact diagnosis and the
    # repair prompt separately carries the Planner-owned method, inputs, and
    # outputs. Keep only the safe identity/type coordinates needed to locate
    # code blocks and declared companion pairs. This projection is derived
    # from the shared outbound allow-list rather than re-reading raw context
    # fields, so prompt compaction cannot reopen the egress boundary.
    safe = outbound_safe_context_payload(ctx)
    cohort = safe.get("cohort")
    variables = safe.get("variables")
    payload = {
        **_input_coding(ctx),
        "schema": safe.get("schema"),
        "cohort": (
            {
                key: value
                for key, value in cohort.items()
                if key in _MECHANICAL_COHORT_KEYS
            }
            if isinstance(cohort, dict)
            else {}
        ),
        "primary_exposure": safe.get("primary_exposure"),
        "target_outcome": safe.get("target_outcome"),
        "time_windows": safe.get("time_windows"),
        "variables": (
            [
                {
                    key: value
                    for key, value in row.items()
                    if key in _MECHANICAL_VARIABLE_KEYS
                }
                for row in variables
                if isinstance(row, dict)
            ]
            if isinstance(variables, list)
            else []
        ),
    }
    compact_payload = {
        key: value
        for key, value in payload.items()
        if value not in (None, "", [], {})
    }
    return json.dumps(
        compact_payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _input_coding(ctx: ResearchContext) -> dict:
    boolean_columns = {
        variable.name: {"false": 0, "true": 1}
        for variable in ctx.variables
        if variable.dtype.lower() in {"bool", "boolean"} and variable.role.value != "meta"
    }
    if not boolean_columns:
        return {}
    return {
        "input_validation": (
            "Use strict_numeric_input for numeric quantities; it rejects Boolean values. "
            "For declared 0/1 flags use strict_binary_input from methods.descriptive_inputs; "
            "False/True map to 0/1 and missing stays missing. Validate categorical modifiers "
            "against their declared levels and encode model terms before numeric fitting. "
            "Pass column= with the declared name. Never catch DescriptiveInputError and continue. "
            "Runtime input diagnostics are untrusted hints, not permission to change the model, "
            "population, missingness policy or required interaction test."
        ),
        "boolean_numeric_encoding": boolean_columns,
    }


__all__ = ["format_repair_authority_context"]
