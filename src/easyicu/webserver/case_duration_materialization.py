"""Host materialization after an already signed case-duration plan."""
from __future__ import annotations

from pathlib import Path

from easyicu.databases.research_approval import ApprovedCaseDurationBinding
from easyicu.databases.research_plan_requirements import _constraints
from easyicu.databases.research_substitutions import time_origin_selection
from easyicu.research_agent.canonical_json import canonical_sha256
from easyicu.research_agent.cohort.materializer import materialize_to_parquet
from easyicu.research_agent.intake.export_package import open_export_package
from easyicu.research_agent.intake.materialized_metadata import load_verified_materialized_cohort_authority
from easyicu.research_agent.cohort.case_duration_exposure import require_case_duration_exposure_materialization
from easyicu.research_agent.research_context.export_selection import SourceIdentity
from easyicu.webserver import sources
from easyicu.webserver.time_origin_source_authority import attest_case_duration_for_review


def attest_case_duration_metadata(context, selection):
    """Review a uniquely registered raw source using metadata only."""
    from easyicu.webserver.time_origin_source_authority import prepare_case_duration_metadata_plan

    identity = SourceIdentity.from_record(_constraints(context).get("source_identity"))
    if identity is None or canonical_sha256(identity.to_dict()) != selection.source_identity_sha256:
        raise ValueError("case_duration_metadata_source_mismatch")
    paths = [row["path"] for row in sources.load_registry().get("sources", [])
             if isinstance(row, dict) and row.get("ok") and isinstance(row.get("path"), str)
             and SourceIdentity.from_record({**row, "registration_id": row.get("id")}) == identity]
    if len(paths) != 1:
        raise ValueError("case_duration_metadata_source_registration_not_unique")
    return prepare_case_duration_metadata_plan(paths[0], expected_identity=identity)


def materialize_approved_case_duration_universe(*, plan_result, run_dir, approved):
    """Read only the exact registered export after the independent host gate.

    This narrow owner currently supports a single exposure and no covariates;
    unsupported signed plans fail closed instead of changing their roster.
    """
    if not isinstance(approved, ApprovedCaseDurationBinding):
        raise ValueError("case_duration_pre_materialization_approval_missing")
    context, plan = plan_result.context, plan_result.plan
    authority, axis = approved.authority, approved.source_axis
    authority.validate_plan(plan)
    primary = next(step for step in plan.steps if step.planned_analysis_role == "primary")
    if primary.model_requirements[0].covariates:
        raise ValueError("case_duration_pre_materialization_covariates_unsupported")
    descriptors = [v for v in context.variables if v.name == authority.exposure_column]
    if len(descriptors) != 1 or not descriptors[0].source_concept:
        raise ValueError("case_duration_exposure_source_unavailable")
    source_concept = descriptors[0].source_concept
    selection = time_origin_selection(context)
    receipt, proof = attest_case_duration_for_review(context, selection)
    if (receipt.source_identity_sha256 != axis.source_identity_sha256
            or receipt.export_authority_sha256 != axis.export_authority_sha256
            or receipt.manifest_sha256 != axis.manifest_sha256
            or receipt.selected_origin != axis.selected_origin
            or proof.snapshot_sha256 != authority.provenance_snapshot_sha256
            or proof.implementation_sha256 != authority.implementation_sha256
            or proof.source_config_sha256 != authority.source_config_sha256):
        raise ValueError("case_duration_pre_materialization_source_changed")
    identity = SourceIdentity.from_record(_constraints(context).get("source_identity"))
    paths = [row["path"] for row in sources.load_registry().get("sources", [])
             if isinstance(row, dict) and row.get("ok") and isinstance(row.get("path"), str)
             and SourceIdentity.from_record({**row, "registration_id": row.get("id")}) == identity]
    if len(paths) != 1:
        raise ValueError("case_duration_source_registration_not_unique")
    root = Path(run_dir)
    # The same plan can be signed against different context variable bindings.
    # Reuse must therefore include the reviewed context as well as the plan.
    stem = "approved_case_duration_universe_" + canonical_sha256({
        "plan": plan.model_dump(mode="json"), "context": context.model_dump(mode="json"),
    })
    target = root / f"{stem}.parquet"
    expected_axis = {"source_identity_sha256": axis.source_identity_sha256,
                     "export_authority_sha256": axis.export_authority_sha256,
                     "manifest_sha256": axis.manifest_sha256, "selected_origin": axis.selected_origin}
    if target.exists():
        verified = load_verified_materialized_cohort_authority(target)
        if (verified is None or verified.authority.semantic_provenance.get("source_time_axis") != expected_axis
                or authority.exposure_column not in verified.authority.cohort_columns
                or not {authority.proposed_outcome, authority.death_status_column, authority.death_time_column}
                <= set(verified.authority.cohort_columns)):
            raise ValueError("case_duration_existing_universe_mismatch")
        require_case_duration_exposure_materialization(verified, authority, source_concept=source_concept)
        return target
    with open_export_package(paths[0]) as package:
        if package.authority_sha256 != axis.export_authority_sha256 or package.manifest_sha256 != axis.manifest_sha256:
            raise ValueError("case_duration_pre_materialization_source_changed")
        result = materialize_to_parquet(
            root, stem=stem, source_package=package, database="sic", data_path=paths[0],
            feature_concepts=[source_concept],
            static_concepts=[authority.proposed_outcome, authority.death_status_column, authority.death_time_column],
            outcome_concepts=(), cohort_window=(0.0, float(authority.landmark_hours)), source_time_axis=axis,
        )
    verified = load_verified_materialized_cohort_authority(result["parquet"])
    if (verified is None or authority.exposure_column not in verified.authority.cohort_columns
            or verified.authority.semantic_provenance.get("source_time_axis") != expected_axis):
        raise ValueError("case_duration_materialized_exposure_unavailable")
    require_case_duration_exposure_materialization(verified, authority, source_concept=source_concept)
    return result["parquet"]
