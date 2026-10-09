"""Which source database a typed cohort binds a run to.

Owner
-----
A typed (materialized) cohort's authority records the source database it was
drawn from and the class prefixes its column metadata wrote.  Before a run
binds the cohort, that database must be the one the run declares, and that
class policy the one the host source registry states.  The run then carries
the authority's canonical database name, so a public alias (for example
``mimiciv``) reaches neither scientific identity, the research context, the
cache nor resume authority.  :mod:`easyicu.research_agent.pipeline` asks it.
"""

from __future__ import annotations

from typing import Sequence

from ..concept_availability import normalize_database_name
from .materialized_metadata import (
    MaterializedMetadataError,
    VerifiedMaterializedCohortAuthority,
)


def typed_cohort_source_resolution_chain(
    database: str, class_prefixes: Sequence[str]
) -> tuple[str, ...]:
    """The resolution order a source class policy denotes.

    Column metadata records the database followed by its class prefixes with
    repeats removed, and a typed cohort recovers its prefixes as that chain
    minus its head.  A source listed among its own prefixes (``eicu_demo`` ->
    ``eicu_demo, eicu``) therefore comes back as ``eicu`` alone, so a class
    policy is compared as the chain it denotes, not as the list that wrote it.
    """

    return tuple(dict.fromkeys((database, *class_prefixes)))


def typed_cohort_source_database(
    authority: VerifiedMaterializedCohortAuthority, database: str
) -> str:
    """The canonical database ``authority`` binds a run declaring ``database`` to.

    Raises :class:`MaterializedMetadataError` when the declared database or
    the host registry's class policy differs from the authority's.
    """

    normalized_database = normalize_database_name(database)
    if authority.sidecar.source_database != normalized_database:
        raise MaterializedMetadataError(
            "declared database does not match typed cohort authority"
        )
    from easyicu.config import load_src_cfg

    expected_prefixes = tuple(
        str(value).strip().lower()
        for value in load_src_cfg(normalized_database).class_prefix
        if str(value).strip()
    )
    if typed_cohort_source_resolution_chain(
        normalized_database,
        authority.sidecar.source_database_class_prefixes,
    ) != typed_cohort_source_resolution_chain(normalized_database, expected_prefixes):
        raise MaterializedMetadataError(
            "typed cohort source class policy does not match host registry"
        )
    return normalized_database


__all__ = ["typed_cohort_source_database", "typed_cohort_source_resolution_chain"]
