"""Bind one execution phase to an exact universe/analysis cohort authority."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from ..cohort_schema import analysis_cohort_authority_coordinates
from ..intake.materialized_metadata import (
    MaterializedMetadataError,
    VerifiedMaterializedCohortAuthority,
    canonical_parameters_sha256,
    implementation_bundle_sha256,
    load_verified_materialized_cohort_authority,
    read_verified_materialized_cohort_table,
)


@dataclass(frozen=True, slots=True)
class ExecutionCohortAuthority:
    """Verified cohort coordinates consumed by one execute-phase attempt."""

    selected_path: Path
    universe_path: Path
    universe_authority: Optional[VerifiedMaterializedCohortAuthority]
    analysis_authority: Optional[VerifiedMaterializedCohortAuthority]

    @property
    def universe_is_typed(self) -> bool:
        return self.universe_authority is not None

    @property
    def universe_columns(self) -> Optional[tuple[str, ...]]:
        if self.universe_authority is None:
            return None
        return self.universe_authority.authority.cohort_columns


def bind_execution_cohort_authority(
    *,
    universe_path: Path,
    analysis_path: Path,
    plan: Any,
    context: Any,
) -> ExecutionCohortAuthority:
    """Verify and select the exact cohort bytes available to step execution."""

    universe_path = Path(universe_path)
    analysis_path = Path(analysis_path)
    universe = load_verified_materialized_cohort_authority(universe_path)
    analysis = None
    selected_path = universe_path
    if analysis_path.exists():
        analysis = load_verified_materialized_cohort_authority(analysis_path)
        if universe is not None:
            if analysis is None:
                raise MaterializedMetadataError(
                    "typed analysis cohort is missing its child authority"
                )
            if analysis.authority.parent_authority_sha256 != universe.reference.sha256:
                raise MaterializedMetadataError(
                    "typed analysis cohort does not descend from the selected universe"
                )
            expected = analysis_cohort_authority_coordinates(
                plan=plan,
                context=context,
                columns=universe.authority.cohort_columns,
                data=read_verified_materialized_cohort_table(
                    universe_path,
                    verified=universe,
                ).to_pandas(),
            )
            actual = {
                "cohort_definition_sha256": (
                    analysis.authority.producer_parameters.get(
                        "cohort_definition_sha256"
                    )
                ),
                "predicate_column_bindings": (
                    analysis.authority.producer_parameters.get(
                        "predicate_column_bindings"
                    )
                ),
                "selected_row_count": (
                    analysis.authority.producer_parameters.get("selected_row_count")
                ),
                "selected_row_positions_sha256": (
                    analysis.authority.producer_parameters.get(
                        "selected_row_positions_sha256"
                    )
                ),
            }
            implementation_paths = (
                Path(__file__).resolve().parents[1] / "cohort_schema.py",
                Path(__file__).resolve().parents[1]
                / "intake"
                / "materialized_metadata.py",
            )
            if (
                analysis.authority.producer != "analysis_cohort_ordered_subset"
                or canonical_parameters_sha256(actual)
                != canonical_parameters_sha256(expected)
                or analysis.authority.semantic_provenance.get("cohort_sha256")
                != expected["cohort_definition_sha256"]
                or analysis.authority.producer_implementation_sha256
                != implementation_bundle_sha256(implementation_paths)
            ):
                raise MaterializedMetadataError(
                    "typed analysis cohort does not match the locked cohort authority"
                )
        selected_path = analysis_path
    return ExecutionCohortAuthority(
        selected_path=selected_path,
        universe_path=universe_path,
        universe_authority=universe,
        analysis_authority=analysis,
    )


__all__ = [
    "ExecutionCohortAuthority",
    "bind_execution_cohort_authority",
]
