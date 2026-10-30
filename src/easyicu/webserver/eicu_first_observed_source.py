"""Bind the conservative eICU candidate to an exact registered source."""

from pathlib import Path

from easyicu.research_agent.acquisition.eicu_first_observed_candidate import (
    REQUIRED_COLUMNS,
    FirstObservedCandidate,
    FirstObservedCandidateError,
    derive_first_observed_candidate,
)
from easyicu.research_agent.acquisition.patient_grouping import read_digest_bound_parquet

from .raw_source_authority import resolve_raw_source_identity_binding


def derive_from_registered_eicu_export(export_path: str | Path) -> FirstObservedCandidate:
    """Read only identity and age columns from an exact digest-bound table.

    A registered export alone does not prove that its raw source is a complete
    official Demo release. The receipt therefore claims only the bound source;
    callers must independently verify release identity and source scope.
    """
    binding = resolve_raw_source_identity_binding(export_path=export_path, database="eicu_demo")
    if binding is None or binding.family != "eicu" or "patient" not in binding.table_paths:
        raise FirstObservedCandidateError("registered eICU Demo raw patient source is unavailable")
    table, _ = read_digest_bound_parquet(
        binding.table_paths["patient"], expected_sha256=binding.table_sha256["patient"],
        columns=REQUIRED_COLUMNS, label="eICU identity and age table", error=FirstObservedCandidateError,
    )
    result = derive_first_observed_candidate(table)
    return FirstObservedCandidate(result.frame, {
        **result.receipt,
        "source_binding": {
            "database": binding.database,
            "export_manifest_sha256": binding.export_manifest_sha256,
            "patient_table_sha256": binding.table_sha256["patient"],
            "official_release_independently_verified": False,
            "raw_path_returned": False,
        },
    })
