"""What each event time the source export issues is, as the launch recorded it.

A Web launch reads the producer's label for every event-time companion the
bound export issues, once, from the manifest it binds
(``bound_export_event_time_semantics``), and records it in
``data_constraints.event_time_semantics``: ``death_time`` labelled by the
death time its source records (``easyicu.utils.death_time_semantics``).  Planning
reads the record here and never opens the export.

A context without the record (one written before it, the CLI, a benchmark)
gives ``None``: nothing is known, which differs from a record that labels
nothing (an export written before the label, or a prepared package), an empty
mapping.
"""

from __future__ import annotations

from typing import Mapping, Optional

from ..schema import ResearchContext
from .concept_population import context_data_constraints

#: ``data_constraints`` key the launch records the labels under.
EVENT_TIME_SEMANTICS_CONSTRAINT = "event_time_semantics"


def recorded_event_time_semantics(
    context: ResearchContext,
) -> Optional[Mapping[str, str]]:
    """Each issued event-time column's label; ``None`` for a context without the record.

    A record that is not a mapping, or an entry that is not a non-empty label,
    labels nothing.
    """

    constraints = context_data_constraints(context)
    if EVENT_TIME_SEMANTICS_CONSTRAINT not in constraints:
        return None
    record = constraints[EVENT_TIME_SEMANTICS_CONSTRAINT]
    if not isinstance(record, Mapping):
        return {}
    return {
        column: label.strip()
        for column, label in record.items()
        if isinstance(column, str) and isinstance(label, str) and label.strip()
    }


__all__ = ["EVENT_TIME_SEMANTICS_CONSTRAINT", "recorded_event_time_semantics"]
