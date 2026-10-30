"""Bind one execution phase to an exact universe/analysis cohort authority."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from ..cohort.schema import analysis_cohort_authority_coordinates
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
    approved_case_duration_binding: Any = None,
) -> ExecutionCohortAuthority:
    """Verify and select the exact cohort bytes available to step execution."""

    universe_path = Path(universe_path)
    analysis_path = Path(analysis_path)
    universe = load_verified_materialized_cohort_authority(universe_path)
    if approved_case_duration_binding is not None and universe is None:
        raise MaterializedMetadataError("approved landmark universe is missing its sealed authority")
    analysis = None
    selected_path = universe_path
    if approved_case_duration_binding is not None and not analysis_path.exists():
        raise MaterializedMetadataError("approved landmark cohort is missing")
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
            if approved_case_duration_binding is not None:
                _verify_case_duration_landmark_cohort(
                    universe_path=universe_path, universe=universe, analysis=analysis,
                    plan=plan, context=context, approved=approved_case_duration_binding,
                )
                return ExecutionCohortAuthority(
                    selected_path=analysis_path, universe_path=universe_path,
                    universe_authority=universe, analysis_authority=analysis,
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
                Path(__file__).resolve().parents[1] / "cohort" / "schema.py",
                Path(__file__).resolve().parents[1]
                / "planning"
                / "cohort_contract.py",
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


def _verify_case_duration_landmark_cohort(*, universe_path, universe, analysis, plan, context, approved):
    from easyicu.databases.research_approval import ApprovedCaseDurationBinding
    from easyicu.databases.research_substitutions import case_duration_selection, time_origin_selection
    from easyicu.research_agent.authority.case_duration_landmark import apply_case_duration_landmark
    from easyicu.research_agent.canonical_json import canonical_sha256
    from easyicu.research_agent.intake.materialized_metadata import _ordered_positions_sha256

    if not isinstance(approved, ApprovedCaseDurationBinding):
        raise MaterializedMetadataError("landmark cohort requires the approved binding")
    authority, axis = approved.authority, approved.source_axis
    authority.validate_plan(plan)
    from easyicu.research_agent.cohort.case_duration_exposure import require_case_duration_exposure_materialization
    require_case_duration_exposure_materialization(universe, authority)
    authority.validate_source_selections(time_origin_selection(context), case_duration_selection(context))
    axis_record = {"source_identity_sha256": axis.source_identity_sha256,
                   "export_authority_sha256": axis.export_authority_sha256,
                   "manifest_sha256": axis.manifest_sha256, "selected_origin": axis.selected_origin}
    provenance = analysis.authority.semantic_provenance
    implementation_paths = (
        Path(__file__).resolve().parents[1] / "cohort" / "schema.py",
        Path(__file__).resolve().parents[1] / "cohort" / "case_duration_exposure.py",
        Path(__file__).resolve().parents[1] / "authority" / "case_duration_landmark.py",
        Path(__file__).resolve().parents[1] / "authority" / "care_unit_landmark.py",
    )
    if (universe.authority.semantic_provenance.get("source_time_axis") != axis_record
            or provenance.get("source_time_axis") != axis_record
            or provenance.get("plan_sha256") != canonical_sha256(plan.model_dump(mode="json"))
            or provenance.get("landmark_contract_sha256") != authority.execution_contract_sha256
            or analysis.authority.producer != "analysis_cohort_ordered_subset"
            or analysis.authority.producer_implementation_sha256 != implementation_bundle_sha256(implementation_paths)):
        raise MaterializedMetadataError("landmark cohort binding differs from the signed plan")
    primary = next(step for step in plan.steps if step.planned_analysis_role == "primary")
    frame = read_verified_materialized_cohort_table(universe_path, verified=universe).to_pandas().reset_index(drop=True)
    selected, counts = apply_case_duration_landmark(frame, authority=authority, source_axis=axis, primary_step=primary)
    parameters = analysis.authority.producer_parameters
    if (parameters.get("selected_row_positions_sha256") != _ordered_positions_sha256(tuple(int(i) for i in selected.index))
            or dict(provenance.get("counts") or {}) != counts):
        raise MaterializedMetadataError("landmark cohort rows differ from the approved risk rule")


__all__ = [
    "ExecutionCohortAuthority",
    "bind_execution_cohort_authority",
]
