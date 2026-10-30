"""Bind a reviewed first-observed ICU plan to its actual, sealed input.

This validates an already prepared run.  It creates no approval, reads no
new source data and cannot turn a metadata-only candidate into an execution.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from easyicu.databases.research_plan_requirements import _constraints
from easyicu.research_agent.authority.filesystem import AnchoredDirectory
from easyicu.research_agent.authority.plan_review import PlanReviewAuthority
from easyicu.research_agent.authority.raw_preparation import RawPreparationCheckpoint
from easyicu.research_agent.authority.run_input import (
    RUN_INPUT_CAPSULE_SCHEMA_VERSION_V2, RUN_INPUT_CAPSULE_SCHEMA_VERSION_V3,
    _MAX_RUN_INPUT_CAPSULE_BYTES,
    load_verified_run_input_capsule,
)
from easyicu.research_agent.canonical_json import canonical_sha256
from easyicu.research_agent.intake.materialized_metadata import (
    _local_authority_reference, load_verified_materialized_cohort_authority,
    read_verified_materialized_cohort_table,
)
from easyicu.research_agent.acquisition.first_observed_icu_coordinate import restrict_to_first_observed_icu
from easyicu.research_agent.orchestration.human_review_checkpoint import load_checkpoint
from easyicu.research_agent.planning.population_spec import PopulationSpec
from easyicu.research_agent.planning.progressive_contract import ProgressivePlanSkeleton
from easyicu.research_agent.research_context.export_selection import SourceIdentity
from easyicu.research_agent.research_context.typed import parse_research_context_json
from easyicu.webserver import agent_pipeline_runs, state_paths
from easyicu.webserver.first_observed_preparation import _SCOPE_KEY, resolve_prepared_first_observed_coordinate
from easyicu.webserver.pi_copilot.population_spec_review import reviewed_population_spec
from easyicu.webserver.raw_preparation_decisions import RawPreparationHostBinding


class FirstObservedRunBindingError(ValueError):
    """The signed coordinate and reviewed run input cannot be proven equal."""


def _restriction_from_cohort(*, run_dir: Path, cohort_path: Path) -> tuple[Mapping[str, Any] | None, Any]:
    verified = load_verified_materialized_cohort_authority(cohort_path)
    if verified is None:
        raise FirstObservedRunBindingError("first_observed_materialization_authority_missing")
    authority = verified.authority
    seen: set[str] = set()
    while authority is not None:
        restriction = authority.producer_parameters.get("first_observed_icu_restriction")
        if restriction is not None:
            if authority.producer != "cohort_materializer" or not isinstance(restriction, Mapping):
                raise FirstObservedRunBindingError("first_observed_materialization_authority_invalid")
            return restriction, verified
        parent = authority.parent_authority_sha256
        if not parent or parent in seen or len(seen) >= 32:
            break
        seen.add(parent)
        _, authority = _local_authority_reference(run_dir, authority_sha256=parent)
    return None, verified


def _reject_unrequested_restriction(run_dir: Path) -> None:
    """A narrower prior cohort cannot silently survive removing this criterion."""
    capsule_path = run_dir / "run_input_capsule.json"
    if capsule_path.exists() or capsule_path.is_symlink():
        with AnchoredDirectory.open(run_dir) as directory:
            raw = directory.read_bytes("run_input_capsule.json", max_bytes=_MAX_RUN_INPUT_CAPSULE_BYTES)
            directory.assert_still_selected()
        payload = json.loads(raw)
        identity = payload.get("scientific_identity") if isinstance(payload, Mapping) else None
        if not isinstance(identity, dict):
            raise FirstObservedRunBindingError("first_observed_run_capsule_invalid")
        capsule = load_verified_run_input_capsule(run_dir=run_dir, scientific_identity=identity).capsule
        relative = Path(capsule.cohort_relative_path)
        if relative.is_absolute() or ".." in relative.parts:
            raise FirstObservedRunBindingError("first_observed_run_cohort_path_invalid")
        cohort_path = run_dir / relative
        if not cohort_path.resolve().is_relative_to(run_dir.resolve()):
            raise FirstObservedRunBindingError("first_observed_run_cohort_path_invalid")
    else:
        cohort_path = run_dir / "cohort.parquet"
    if not cohort_path.exists() and not cohort_path.is_symlink():
        return  # The existing run-input verifier owns missing legacy inputs.
    verified = load_verified_materialized_cohort_authority(cohort_path)
    if verified is None:
        return  # The existing run-input verifier owns legacy inputs.
    restriction, _ = _restriction_from_cohort(run_dir=cohort_path.parent, cohort_path=cohort_path)
    if restriction is not None:
        raise FirstObservedRunBindingError("first_observed_run_population_not_requested")


def verify_first_observed_execution_inputs(*, pending: Any, study_context_id: str) -> bool:
    """Verify the full chain when the reviewed question selects this cohort.

    Returns false for other populations. A true result does not approve the
    plan; the ordinary host decision and scientific review remain mandatory.
    """
    run_dir = Path(pending.run_dir)
    authorities = [PlanReviewAuthority.model_validate(request.payload["plan_review_authority"])
                   for request in pending.requests
                   if isinstance(request.payload.get("plan_review_authority"), Mapping)]
    if not authorities:
        _reject_unrequested_restriction(run_dir)
        return False  # Existing non-progressive plans have no such criterion.
    if any(authority != authorities[0] for authority in authorities):
        raise FirstObservedRunBindingError("first_observed_run_plan_unavailable")
    plan = authorities[0]
    skeleton_sha = plan.evidence_sha256.get("progressive_plan_skeleton")
    if skeleton_sha is None:
        _reject_unrequested_restriction(run_dir)
        return False
    with AnchoredDirectory.open(run_dir) as directory:
        skeleton_raw = directory.read_bytes("progressive_plan_skeleton.json", max_bytes=16 * 1024 * 1024)
        directory.assert_still_selected()
    if hashlib.sha256(skeleton_raw).hexdigest() != skeleton_sha:
        raise FirstObservedRunBindingError("first_observed_run_population_unverified")
    skeleton = ProgressivePlanSkeleton.model_validate_json(skeleton_raw)
    if skeleton.cohort.population_spec is None:
        _reject_unrequested_restriction(run_dir)
        return False
    spec = PopulationSpec.model_validate(skeleton.cohort.population_spec)
    matches = [criterion.model_dump(mode="json") for criterion in spec.criteria
               if criterion.kind == "first_observed_icu_single_hospital_adult"]
    if not matches:
        _reject_unrequested_restriction(run_dir)
        return False
    if len(matches) != 1 or matches[0].get("role") != "include":
        raise FirstObservedRunBindingError("first_observed_run_population_invalid")
    population = reviewed_population_spec(run_dir=run_dir, pending=pending)
    if not isinstance(population, Mapping) or population.get("plan_sha256") != plan.plan_sha256:
        raise FirstObservedRunBindingError("first_observed_run_population_unverified")
    if plan.execution is None:
        raise FirstObservedRunBindingError("first_observed_run_execution_authority_missing")
    with AnchoredDirectory.open(run_dir) as directory:
        raw = directory.read_bytes("run_input_capsule.json", max_bytes=_MAX_RUN_INPUT_CAPSULE_BYTES)
        directory.assert_still_selected()
    digest = hashlib.sha256(raw).hexdigest()
    review = load_checkpoint(run_dir / "human_review_checkpoint.json", require_pending=False)
    if (review.run_id != str(pending.run_id) or review.run_input_capsule_sha256 != digest
            or plan.execution.run_input_capsule_sha256 != digest
            or tuple(review.requests) != tuple(pending.requests)):
        raise FirstObservedRunBindingError("first_observed_run_capsule_changed")
    payload = json.loads(raw)
    identity = payload.get("scientific_identity") if isinstance(payload, Mapping) else None
    if not isinstance(identity, dict):
        raise FirstObservedRunBindingError("first_observed_run_capsule_invalid")
    verified = load_verified_run_input_capsule(run_dir=run_dir, scientific_identity=identity)
    if (verified.capsule.schema_version not in {RUN_INPUT_CAPSULE_SCHEMA_VERSION_V2, RUN_INPUT_CAPSULE_SCHEMA_VERSION_V3}
            or (verified.capsule.trajectory_relative_path is not None
                and verified.capsule.schema_version != RUN_INPUT_CAPSULE_SCHEMA_VERSION_V3)):
        # Legacy capsules can hash a trajectory without proving its universe.
        # Only the V3 contract verifies trajectory → staged-cohort authority.
        raise FirstObservedRunBindingError("first_observed_run_trajectory_authority_missing")
    relative = Path(verified.capsule.cohort_relative_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise FirstObservedRunBindingError("first_observed_run_cohort_path_invalid")
    cohort_path = run_dir / relative
    if not cohort_path.resolve().is_relative_to(run_dir.resolve()):
        raise FirstObservedRunBindingError("first_observed_run_cohort_path_invalid")
    restriction, verified_cohort = _restriction_from_cohort(run_dir=cohort_path.parent, cohort_path=cohort_path)
    if restriction is None:
        raise FirstObservedRunBindingError("first_observed_materialization_authority_missing")
    source_run = restriction.get("preparation_run_id")
    binding_sha = restriction.get("preparation_binding_sha256")
    if not isinstance(source_run, str) or not isinstance(binding_sha, str):
        raise FirstObservedRunBindingError("first_observed_host_origin_missing")
    source_pending, _ = agent_pipeline_runs.raw_preparation_review_owner(source_run, study_context_id)
    source_dir = Path(source_pending.run_dir)
    with AnchoredDirectory.open(source_dir) as directory:
        preparation = RawPreparationCheckpoint.model_validate_json(
            directory.read_bytes("raw_preparation_checkpoint.json", max_bytes=1024 * 1024))
        directory.assert_still_selected()
    scope = preparation.request.payload.get(_SCOPE_KEY)
    if preparation.state != "approved" or not isinstance(scope, Mapping):
        raise FirstObservedRunBindingError("first_observed_host_preparation_unavailable")
    raw_pending = preparation.model_dump(mode="json", exclude={"checkpoint_sha256"})
    raw_pending.update(state="pending", decision=None)
    nonce = canonical_sha256({"run_id": source_run, "study_context_id": study_context_id,
                              "checkpoint_sha256": canonical_sha256(raw_pending)})
    with AnchoredDirectory.open(state_paths.state_root() / "private-preparation-decisions") as directory:
        receipt = json.loads(directory.read_bytes(nonce + ".json", max_bytes=64 * 1024))
        directory.assert_still_selected()
    binding = RawPreparationHostBinding(project_id=receipt["project_id"], session_id=receipt["session_id"],
        study_context_id=study_context_id, run_id=source_run, nonce=nonce)
    coordinate = resolve_prepared_first_observed_coordinate(
        run_dir=source_dir, checkpoint=preparation, binding=binding, scope=scope)
    expected = {"schema_version": "easyicu.first-observed-icu-restriction/1",
        "coordinate_sha256": coordinate.sha256, "archive_sha256": coordinate.archive_sha256,
        "patient_table_sha256": coordinate.patient_table_sha256,
        "export_manifest_sha256": coordinate.export_manifest_sha256, "plan_sha256": coordinate.plan_sha256,
        "preparation_checkpoint_sha256": coordinate.preparation_checkpoint_sha256,
        "population_definition": "adult_first_observed_icu_in_single_visible_hospitalization",
        "not_lifetime_first_icu": True, "preparation_run_id": binding.run_id,
        "preparation_binding_sha256": canonical_sha256(binding.model_dump(mode="json"))}
    if dict(restriction) != expected or binding_sha != expected["preparation_binding_sha256"]:
        raise FirstObservedRunBindingError("first_observed_materialization_binding_changed")
    # Recheck the actual staged identities; provenance alone is not proof
    # that a constructible coordinate was applied to these rows.
    stays = read_verified_materialized_cohort_table(cohort_path, verified=verified_cohort).select(["stay_id"]).to_pandas()
    if len(restrict_to_first_observed_icu(stays, coordinate=coordinate)) != len(stays):
        raise FirstObservedRunBindingError("first_observed_materialization_rows_changed")
    if canonical_sha256(matches[0]) != canonical_sha256(scope.get("criterion")):
        raise FirstObservedRunBindingError("first_observed_run_population_changed")
    context = parse_research_context_json(verified.context_evidence_path.read_text(encoding="utf-8"))
    source = SourceIdentity.from_record(_constraints(context).get("source_identity"))
    if source is None or canonical_sha256(source.to_dict()) != scope.get("source_identity_sha256"):
        raise FirstObservedRunBindingError("first_observed_run_source_changed")
    return True
