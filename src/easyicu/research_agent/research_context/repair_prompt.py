"""PHI-minimized research-context projection for repair transports."""

from __future__ import annotations

import json

from ..schema import ResearchContext
from .outbound import (
    format_outbound_safe_context,
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
        return format_outbound_safe_context(ctx)

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


__all__ = ["format_repair_authority_context"]
