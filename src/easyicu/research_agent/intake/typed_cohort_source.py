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

from pathlib import Path
from typing import Mapping
from easyicu.databases.research_capabilities import research_capability_profile
from ..contracts.prediction_validation import ExternalValidationError, FrozenLogisticModel
from ..methods.prediction_validation import ValidationSource
from ..replication.metrics import require_transport_capabilities, transport_concept_meaning
from .materialized_metadata import load_verified_materialized_cohort_authority

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


def validation_source_from_binding(
    *, database: str, model: FrozenLogisticModel, input_key: str,
    run_dir: Path, resolved_inputs: Mapping, step_id: str,
) -> ValidationSource:
    """Called only after model freeze; sidecar verification can read file bytes."""
    from ..execution.runners.typed_input_binding import contained_regular_file, load_typed_input

    binding = resolved_inputs.get("inputs", {}).get(input_key, {})
    path = contained_regular_file(run_dir / str(binding.get("relative_path", "")), run_dir)
    if path is None:
        raise ExternalValidationError("validation_source_binding_invalid", database=database)
    verified = load_verified_materialized_cohort_authority(path)
    if verified is None:
        raise ExternalValidationError("validation_source_authority_missing", database=database)
    typed_cohort_source_database(verified, database)
    if verified.authority.cohort_sha256 != binding.get("sha256"):
        raise ExternalValidationError("validation_source_binding_invalid", database=database)
    profile = research_capability_profile(database)
    require_transport_capabilities(profile, model)
    file_metadata = next((f for f in verified.sidecar.files if f.relative_path == path.name), None)
    if file_metadata is None:
        raise ExternalValidationError("validation_source_authority_missing", database=database)
    concepts = (*[p.concept for p in model.predictors], model.outcome_concept)
    frozen_meanings = {p.concept: p.meaning for p in model.predictors}
    frozen_meanings[model.outcome_concept] = model.outcome_meaning
    column_map, meanings = {}, {}
    for concept in concepts:
        candidates = [(column, b) for column, b in file_metadata.columns.items() if b.metadata.source_concept == concept]
        exact = [(column, b) for column, b in candidates
                 if transport_concept_meaning(b, profile) == frozen_meanings[concept]]
        candidates = exact or candidates
        if len(candidates) != 1:
            raise ExternalValidationError(
                "outcome_unavailable" if concept == model.outcome_concept else "predictor_unavailable",
                database=database, concepts=[concept],
            )
        column, metadata = candidates[0]
        column_map[column] = concept
        meanings[concept] = transport_concept_meaning(metadata, profile)

    def read():
        loaded = load_typed_input(
            input_key=input_key, run_dir=run_dir, resolved_inputs=resolved_inputs,
            step_id=step_id, expected_evidence_kind="table", require_consumption_contract=True,
            minimum_row_count=4,
        )
        if loaded.sha256 != verified.authority.cohort_sha256:
            raise ExternalValidationError("validation_source_binding_invalid", database=database)
        return loaded.frame.rename(columns=column_map), loaded.sha256

    columns = tuple(column_map.get(c, c) for c in verified.authority.cohort_columns)
    if len(set(columns)) != len(columns):
        raise ExternalValidationError("ambiguous_harmonized_columns", database=database)
    return ValidationSource(database=database, columns=columns, meanings=meanings, read=read)
