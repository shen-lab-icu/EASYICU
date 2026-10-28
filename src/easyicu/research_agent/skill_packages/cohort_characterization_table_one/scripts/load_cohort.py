"""Validate prepared row identity and columns; the host owns missing-data rules."""

from ...reference_io import LoadedCohort, load_prepared_cohort
from ..spec import TableOneReferenceSpec


def load_cohort(source, spec: TableOneReferenceSpec, *, provenance=None) -> LoadedCohort:
    return load_prepared_cohort(
        source, required_columns=spec.required_columns(),
        identity_column=spec.identity_column, provenance=provenance,
    )
