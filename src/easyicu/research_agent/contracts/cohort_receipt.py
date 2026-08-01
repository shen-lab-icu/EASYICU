"""Which host cohort-execution receipt fields name a column a step may read.

One declaration, imported by both halves that read it, because they drifted.

``research_context.typed.raw_contract_inputs_for_step`` authorizes a raw column
for every predicate coordinate the host's mask actually read, and its docstring
argues the case: "For a predicate the host narrowed to an event-time window
that is two columns, not one, so the event-time column is authorized on the
same footing as ``resolved_column``: the Coder is asked to reproduce that
predicate's counts, and it cannot do so from a column it has no contract for."

``authority.typed_binding._write_resolved_inputs_manifest`` then re-derives the
authorized set to check the contracts it is handed -- and read only
``resolved_column``.  The producer was widened; the checker was not.

The cost, measured on canary33's real run: the Planner wrote a cohort with an
early-death exclusion narrowed to hours 0-24, so the receipt named
``resolved_column="death"`` and ``event_time_column="death_time"``.  The
producer authorized ``{age, death, death_time}``, the checker authorized
``{age, death}``, and step 01 raised

    ValueError: raw input contracts must exactly match Planner-declared or
    host-receipt raw inputs

before any analysis ran -- 0 of 12 steps.  canary32 survived only because its
plan declared no cohort predicates at all, so both sides were empty.

This module holds no logic and imports nothing from the package, so either
half may depend on it without creating an edge between them.
"""

from __future__ import annotations

from typing import Final, Mapping, Tuple

__all__ = [
    "COHORT_RECEIPT_COLUMN_FIELDS",
    "cohort_receipt_authorized_columns",
]


#: ``(field, human reason)`` for every receipt field naming an authorized raw
#: column.  The reason is the phrase each reader puts in its own error, kept
#: here so adding a field cannot add it to one side only.
COHORT_RECEIPT_COLUMN_FIELDS: Final[Tuple[Tuple[str, str], ...]] = (
    ("resolved_column", "resolved column"),
    ("event_time_column", "predicate event-time column"),
)


def cohort_receipt_authorized_columns(ordered_predicate_flow) -> set:
    """Every raw column an already-validated predicate flow authorizes.

    Takes the flow rather than the whole receipt so the caller keeps ownership
    of validating it; this only answers which names it names.  Non-string and
    empty values are skipped rather than raised on, because each caller has
    already applied its own validation with its own exception type and a second
    opinion here would report the same defect under a third name.
    """

    authorized: set = set()
    for row in ordered_predicate_flow or ():
        if not isinstance(row, Mapping):
            continue
        for field, _reason in COHORT_RECEIPT_COLUMN_FIELDS:
            column = row.get(field)
            if isinstance(column, str) and column.strip():
                authorized.add(column)
    return authorized
