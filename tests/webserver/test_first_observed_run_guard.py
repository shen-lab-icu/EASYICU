"""Bind actual typed cohort rows to the host's prepared coordinate.

The capsule, host resolver and plan projection are stubbed here; their own
cryptographic and process contracts have separate tests.
"""
from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from easyicu.research_agent.acquisition.eicu_first_observed_candidate import FLAG_COLUMN, REASON_COLUMN
from easyicu.research_agent.acquisition.first_observed_icu_coordinate import FirstObservedIcuCoordinate
from easyicu.research_agent.authority.plan_review import PlanReviewAuthority, ReviewExecutionAuthority
from easyicu.research_agent.authority.raw_preparation import RawPreparationCheckpoint
from easyicu.research_agent.canonical_json import canonical_sha256
from easyicu.research_agent.cohort.materializer import materialize_to_parquet
from easyicu.research_agent.intake.export_package import open_export_package
from easyicu.research_agent.orchestration.workflow import HumanReviewDecision, HumanReviewRequest
from easyicu.research_agent.planning.population_spec import PopulationSpec
from easyicu.research_agent.schema import AnalysisPlan, CohortDescriptor, ResearchContext, UserPreferences
from easyicu.research_agent.research_context.export_selection import SourceIdentity
from easyicu.webserver import first_observed_run_guard as guard
from easyicu.webserver.raw_preparation_decisions import RawPreparationHostBinding
from tests.research_agent.planning.progressive_planner_fixtures import _payload
from tests.support.typed_export import typed_export


def _case(tmp_path, monkeypatch, *, apply_coordinate=True, cohort_stem="cohort"):
    monkeypatch.setenv("EASYICU_HOME", str(tmp_path / "home"))
    source = typed_export(tmp_path / "export", database="eicu")
    with open_export_package(source) as package:
        source_manifest = package.manifest_sha256
    source_identity = SourceIdentity(database="eicu_demo", registration_id="synthetic", archive_sha256="a" * 64)
    spec = PopulationSpec.model_validate({"criteria": [{"id": "c1", "kind": "first_observed_icu_single_hospital_adult",
        "role": "include", "source": "question", "quote": "first observed ICU"}]})
    criterion = spec.criteria[0].model_dump(mode="json")
    source_run = tmp_path / "source_run"
    source_run.mkdir()
    scope = {"criterion": criterion, "source_identity_sha256": canonical_sha256(source_identity.to_dict())}
    prep_payload = {"first_observed_icu_preparation_scope": scope}
    prep_request = HumanReviewRequest.create(kind="scientific_stop", summary="Synthetic preparation", payload=prep_payload,
                                             authority_sha256=canonical_sha256(prep_payload))
    prep_pending = RawPreparationCheckpoint.pending(plan_sha256="b" * 64, request=prep_request)
    prep = prep_pending.decide(HumanReviewDecision(review_id=prep_request.review_id,
        authority_sha256=prep_request.authority_sha256, decision="approved", reviewer="synthetic",
        decided_at=datetime.now(timezone.utc).isoformat()))
    (source_run / "raw_preparation_checkpoint.json").write_text(prep.model_dump_json())
    nonce = canonical_sha256({"run_id": "source_run", "study_context_id": "study",
                              "checkpoint_sha256": prep_pending.checkpoint_sha256})
    binding = RawPreparationHostBinding(project_id="project", session_id="session", study_context_id="study",
                                        run_id="source_run", nonce=nonce)
    private = tmp_path / "home" / ".easyicu" / "private-preparation-decisions"
    private.mkdir(parents=True, mode=0o700)
    (private / (nonce + ".json")).write_text(json.dumps({"project_id": "project", "session_id": "session"}))
    coord_path = tmp_path / "coordinate.parquet"
    pd.DataFrame({"stay_id": [1, 2], FLAG_COLUMN: [False, True],
                  REASON_COLUMN: ["multiple_observed_hospitalizations", "kept"]}).to_parquet(coord_path, index=False)
    coordinate = FirstObservedIcuCoordinate(path=coord_path.absolute(), sha256=hashlib.sha256(coord_path.read_bytes()).hexdigest(),
        archive_sha256="a" * 64, patient_table_sha256="c" * 64, export_manifest_sha256=source_manifest,
        plan_sha256="b" * 64, preparation_checkpoint_sha256=prep.checkpoint_sha256,
        preparation_run_id="source_run", preparation_binding_sha256=canonical_sha256(binding.model_dump(mode="json")))
    run = tmp_path / "target_run"
    run.mkdir()
    from easyicu.research_agent.acquisition import first_observed_icu_coordinate
    original_restrict = first_observed_icu_coordinate.restrict_to_first_observed_icu
    if not apply_coordinate:
        monkeypatch.setattr(first_observed_icu_coordinate, "restrict_to_first_observed_icu", lambda frame, **_: frame)
    with open_export_package(source) as package:
        materialize_to_parquet(run, stem=cohort_stem, database="eicu", data_path=source, source_package=package,
            feature_concepts=("lact",), static_concepts=("age",), outcome_concepts=("death",),
            first_observed_icu_coordinate=coordinate)
    if not apply_coordinate:
        monkeypatch.setattr(first_observed_icu_coordinate, "restrict_to_first_observed_icu", original_restrict)
    context = ResearchContext(research_question="first observed ICU", cohort=CohortDescriptor(
        cohort_name="synthetic", database="eicu_demo", n_stays=1, id_columns=["stay_id"], outcome_columns=["death"]),
        variables=[], target_outcome="death", user_preferences=UserPreferences(
            data_constraints=json.dumps({"source_identity": source_identity.to_dict()})))
    context_path = run / "context-evidence.json"
    context_path.write_text(context.model_dump_json())
    capsule_raw = json.dumps({"scientific_identity": {"question": "synthetic"}}).encode()
    (run / "run_input_capsule.json").write_bytes(capsule_raw)
    capsule_sha = hashlib.sha256(capsule_raw).hexdigest()
    skeleton = _payload()
    skeleton["cohort"]["population_spec"] = spec.model_dump(mode="json")
    raw_skeleton = json.dumps(skeleton).encode()
    (run / "progressive_plan_skeleton.json").write_bytes(raw_skeleton)
    plan = AnalysisPlan(research_question="first observed ICU", steps=[])
    authority = PlanReviewAuthority.create(plan=plan, evidence_sha256={"progressive_plan_skeleton": hashlib.sha256(raw_skeleton).hexdigest()},
        execution=ReviewExecutionAuthority(pipeline_config_sha256="1"*64, capability_activation_sha256="2"*64,
                                           run_input_capsule_sha256=capsule_sha))
    plan_payload = {"plan_review_authority": authority.model_dump(mode="json")}
    request = HumanReviewRequest.create(kind="scientific_stop", summary="Synthetic plan", payload=plan_payload,
                                        authority_sha256=canonical_sha256(plan_payload))
    pending = SimpleNamespace(run_id="target_run", run_dir=str(run), requests=[request])
    monkeypatch.setattr(guard, "reviewed_population_spec", lambda **_: {"plan_sha256": authority.plan_sha256})
    monkeypatch.setattr(guard, "load_checkpoint", lambda *a, **k: SimpleNamespace(
        run_id="target_run", run_input_capsule_sha256=capsule_sha, requests=[request]))
    monkeypatch.setattr(guard, "load_verified_run_input_capsule", lambda **_: SimpleNamespace(
        capsule=SimpleNamespace(schema_version=guard.RUN_INPUT_CAPSULE_SCHEMA_VERSION_V2,
                                cohort_relative_path=f"{cohort_stem}.parquet", trajectory_relative_path=None),
        context_evidence_path=context_path))
    monkeypatch.setattr(guard.agent_pipeline_runs, "raw_preparation_review_owner", lambda *_: (SimpleNamespace(run_dir=str(source_run)), {}))
    monkeypatch.setattr(guard, "resolve_prepared_first_observed_coordinate", lambda **_: coordinate)
    return pending, coordinate, authority


def test_verified_coordinate_and_typed_cohort_are_bound(tmp_path, monkeypatch):
    pending, _, _ = _case(tmp_path, monkeypatch)
    assert guard.verify_first_observed_execution_inputs(pending=pending, study_context_id="study")


def test_rows_not_filtered_by_coordinate_cannot_run(tmp_path, monkeypatch):
    pending, _, _ = _case(tmp_path, monkeypatch, apply_coordinate=False)
    with pytest.raises(ValueError, match="rows_changed"):
        guard.verify_first_observed_execution_inputs(pending=pending, study_context_id="study")


def test_changed_or_forged_host_coordinate_cannot_run(tmp_path, monkeypatch):
    pending, coordinate, _ = _case(tmp_path, monkeypatch)
    monkeypatch.setattr(guard, "resolve_prepared_first_observed_coordinate", lambda **_: replace(coordinate, sha256="0"*64))
    with pytest.raises(ValueError, match="binding_changed"):
        guard.verify_first_observed_execution_inputs(pending=pending, study_context_id="study")


def test_plan_without_typed_population_is_unaffected(tmp_path):
    pending = SimpleNamespace(run_dir=str(tmp_path), requests=[])
    assert guard.verify_first_observed_execution_inputs(pending=pending, study_context_id="study") is False


def test_removing_first_icu_requirement_cannot_reuse_a_narrowed_cohort(tmp_path, monkeypatch):
    pending, _, _ = _case(tmp_path, monkeypatch)
    pending.requests = []
    with pytest.raises(ValueError, match="population_not_requested"):
        guard.verify_first_observed_execution_inputs(pending=pending, study_context_id="study")


def test_removed_requirement_checks_the_verified_capsule_cohort_path(tmp_path, monkeypatch):
    pending, _, _ = _case(tmp_path, monkeypatch, cohort_stem="alternative")
    pending.requests = []
    assert not (Path(pending.run_dir) / "cohort.parquet").exists()
    with pytest.raises(ValueError, match="population_not_requested"):
        guard.verify_first_observed_execution_inputs(pending=pending, study_context_id="study")


def test_removed_requirement_does_not_trust_an_unverified_capsule(tmp_path, monkeypatch):
    pending, _, _ = _case(tmp_path, monkeypatch, cohort_stem="alternative")
    pending.requests = []
    def reject_capsule(**_):
        raise ValueError("capsule not verified")
    monkeypatch.setattr(guard, "load_verified_run_input_capsule", reject_capsule)
    with pytest.raises(ValueError, match="capsule not verified"):
        guard.verify_first_observed_execution_inputs(pending=pending, study_context_id="study")


def test_capsule_larger_than_two_megabytes_reaches_the_owner_verifier(tmp_path, monkeypatch):
    pending, _, _ = _case(tmp_path, monkeypatch)
    pending.requests = []
    capsule_path = Path(pending.run_dir) / "run_input_capsule.json"
    capsule_path.write_text(json.dumps({"scientific_identity": {"padding": "x" * (2 * 1024 * 1024)}}))
    with pytest.raises(ValueError, match="population_not_requested"):
        guard.verify_first_observed_execution_inputs(pending=pending, study_context_id="study")


def test_unrestricted_typed_cohort_is_not_blocked_without_first_icu_requirement(tmp_path):
    source = typed_export(tmp_path / "export", database="eicu")
    run = tmp_path / "run"
    materialize_to_parquet(run, database="eicu", data_path=source, feature_concepts=("lact",),
                           static_concepts=("age",), outcome_concepts=("death",))
    pending = SimpleNamespace(run_dir=str(run), requests=[])
    assert guard.verify_first_observed_execution_inputs(pending=pending, study_context_id="study") is False


def test_real_v3_capsule_binds_coordinate_cohort_and_trajectory(tmp_path, monkeypatch):
    from easyicu.research_agent.authority.evidence_store import EvidenceStore
    from easyicu.research_agent.authority.run_input import build_scientific_identity, seal_run_input_capsule
    from easyicu.research_agent.intake.materialized_metadata import (
        load_verified_materialized_cohort_authority, stage_materialized_cohort_authority,
    )
    from easyicu.research_agent.intake.materialized_trajectory import (
        StagedTrajectoryBinding, load_verified_materialized_trajectory_authority,
        stage_materialized_trajectory_authority,
    )
    from easyicu.research_agent.orchestration.human_review_checkpoint import HumanReviewCheckpoint, write_checkpoint
    from easyicu.research_agent.research_context.builder import build_research_context

    original_load_checkpoint = guard.load_checkpoint
    original_load_capsule = guard.load_verified_run_input_capsule
    old_pending, coordinate, old_plan = _case(tmp_path, monkeypatch)
    export = tmp_path / "export"
    with open_export_package(export) as package:
        outputs = materialize_to_parquet(tmp_path / "materialized", database="eicu", data_path=export,
            source_package=package, feature_concepts=("lact",), static_concepts=("age",), outcome_concepts=("death",),
            emit_trajectory=True, trajectory_concepts=("lact",), first_observed_icu_coordinate=coordinate)
    original_cohort = load_verified_materialized_cohort_authority(outputs["parquet"])
    original_traj = load_verified_materialized_trajectory_authority(outputs["trajectory"], expected_universe_authority=original_cohort.reference)
    run = tmp_path / "run_v3"
    run.mkdir()
    cohort = run / "cohort.parquet"
    traj = run / "cohort_trajectory.parquet"
    staged = stage_materialized_cohort_authority(outputs["parquet"], cohort,
        expected_source_authority=original_cohort.reference, producer_implementation_sha256="f" * 64)
    staged_traj = stage_materialized_trajectory_authority(outputs["trajectory"], traj,
        source_universe_path=outputs["parquet"], target_universe_path=cohort,
        expected_source_authority=original_traj.reference, expected_target_universe_authority=staged.reference,
        producer_implementation_sha256="f" * 64)
    identity_source = SourceIdentity(database="eicu_demo", registration_id="synthetic", archive_sha256="a" * 64)
    preferences = {"data_constraints": json.dumps({"source_identity": identity_source.to_dict()})}
    context = build_research_context(research_question="first observed ICU", cohort=cohort, cohort_name="synthetic",
        database="eicu", target_outcome="death", id_columns=("stay_id",), outcome_columns=("death",),
        user_preferences=preferences, trajectory_binding=StagedTrajectoryBinding(path=traj,
            sha256=staged_traj.authority.trajectory_sha256, size=staged_traj.authority.trajectory_size,
            authority_ref=staged_traj.reference))
    context_path = run / "research_context.json"
    context_path.write_text(context.model_dump_json())
    evidence = EvidenceStore(run)
    evidence.register_file(kind="log", description="Synthetic context", source_path=context_path,
                           evidence_id="research_context", producer="pipeline", generation_mode="system")
    identity = build_scientific_identity(cohort=outputs["parquet"], question="first observed ICU", cohort_name="synthetic",
        database="eicu", target_outcome="death", primary_exposure=None, cross_database_validation=None,
        inclusion_criteria=None, exclusion_criteria=None, id_columns=("stay_id",), time_columns=None,
        outcome_columns=("death",), time_windows=None, concept_descriptions=None, user_preferences=preferences,
        notes=None, skill_key=None, experiment_spec=None, source_files=None, disable_icu_context=False,
        materialized_cohort_authority_ref=original_cohort.reference.to_dict(), trajectory_path=outputs["trajectory"],
        materialized_trajectory_authority_ref=original_traj.reference.to_dict())
    identity["synthetic_large_input"] = "x" * (2 * 1024 * 1024)
    capsule = seal_run_input_capsule(run_dir=run, evidence=evidence, scientific_identity=identity,
        initial_environment={"llm_signature": "synthetic"}, context_path=context_path, cohort_path=cohort,
        experiment_spec_path=None)
    assert capsule.schema_version == guard.RUN_INPUT_CAPSULE_SCHEMA_VERSION_V3
    capsule_sha = hashlib.sha256((run / "run_input_capsule.json").read_bytes()).hexdigest()
    skeleton = (Path(old_pending.run_dir) / "progressive_plan_skeleton.json").read_bytes()
    (run / "progressive_plan_skeleton.json").write_bytes(skeleton)
    new_plan = PlanReviewAuthority.create(plan=old_plan.model_dump(mode="json")["plan_payload"],
        evidence_sha256={"progressive_plan_skeleton": hashlib.sha256(skeleton).hexdigest()},
        execution=ReviewExecutionAuthority(pipeline_config_sha256="1" * 64, capability_activation_sha256="2" * 64,
                                           run_input_capsule_sha256=capsule_sha))
    request_payload = {"plan_review_authority": new_plan.model_dump(mode="json")}
    request = HumanReviewRequest.create(kind="scientific_stop", summary="Synthetic plan", payload=request_payload,
                                        authority_sha256=canonical_sha256(request_payload))
    checkpoint = HumanReviewCheckpoint.create(run_id="run_v3", pipeline_config_sha256="1" * 64,
        environment_identity={}, llm_signature_sha256="3" * 64, run_input_capsule_sha256=capsule_sha,
        capability_activation_sha256="2" * 64, runtime_capabilities=(), runtime_bundle=None,
        requests=(request,), plan_handoff={}, execution_coordinates={})
    write_checkpoint(run / "human_review_checkpoint.json", checkpoint)
    pending = SimpleNamespace(run_id="run_v3", run_dir=str(run), requests=[request])
    monkeypatch.setattr(guard, "load_checkpoint", original_load_checkpoint)
    monkeypatch.setattr(guard, "load_verified_run_input_capsule", original_load_capsule)
    monkeypatch.setattr(guard, "reviewed_population_spec", lambda **_: {"plan_sha256": new_plan.plan_sha256})
    assert guard.verify_first_observed_execution_inputs(pending=pending, study_context_id="study")
    pending.requests = []
    with pytest.raises(ValueError, match="population_not_requested"):
        guard.verify_first_observed_execution_inputs(pending=pending, study_context_id="study")
    pending.requests = [request]
    traj.write_bytes(traj.read_bytes() + b"tampered")
    with pytest.raises((ValueError, RuntimeError), match="trajectory|Trajectory"):
        guard.verify_first_observed_execution_inputs(pending=pending, study_context_id="study")


@pytest.mark.parametrize("schema", ["easyicu.run_input_capsule/1", "easyicu.run_input_capsule/2"])
def test_legacy_capsule_cannot_pair_a_filtered_cohort_with_unbound_trajectory(tmp_path, monkeypatch, schema):
    pending, _, _ = _case(tmp_path, monkeypatch)
    monkeypatch.setattr(guard, "load_verified_run_input_capsule", lambda **_: SimpleNamespace(
        capsule=SimpleNamespace(schema_version=schema, cohort_relative_path="cohort.parquet",
                                trajectory_relative_path="cohort_trajectory.parquet"),
        context_evidence_path=Path(pending.run_dir) / "context-evidence.json"))
    with pytest.raises(ValueError, match="trajectory_authority_missing"):
        guard.verify_first_observed_execution_inputs(pending=pending, study_context_id="study")
