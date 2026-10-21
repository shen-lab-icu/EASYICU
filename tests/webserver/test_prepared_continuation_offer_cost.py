"""Continuation projections reuse verified work; launches remain authoritative."""

import hashlib
import json
import os
import shutil
import time
from unittest.mock import Mock

import pytest

from easyicu.research_agent.authority.evidence_store import EvidenceStore
from easyicu.research_agent.intake import export_package
from easyicu.research_agent.providers.mocks import ScriptedMockLLMClient
from easyicu.webserver import agent_review_recovery, dataio, research_launch_resume as owner
from easyicu.webserver import research_prepared_resume as prepared_owner, study_contexts
from easyicu.webserver.pi_copilot import run_authority, workflow
from tests.support.grouped_resume import ANSWER, phase, prepared, seal_prepared_source, source as _source
from tests.webserver.copilot.research_workflow_fixtures import complete_study

source = _source


@pytest.fixture(autouse=True)
def empty_offer_memos():
    from easyicu.webserver import research_resume_package

    prepared_owner._OFFER_VERDICTS.clear()
    research_resume_package._package_offer_verdict.cache_clear()
    yield
    prepared_owner._OFFER_VERDICTS.clear()
    research_resume_package._package_offer_verdict.cache_clear()


@pytest.fixture
def prepared_source(source, tmp_path, monkeypatch):
    cohort = prepared(tmp_path / "input", patient_grouped=True)
    shutil.copytree(cohort.parent, source.run, dirs_exist_ok=True)
    source.cohort = source.run / "cohort.parquet"
    source.original = cohort
    result = phase(source.cohort, ScriptedMockLLMClient([ANSWER]))
    source.context = result.context
    (source.run / "research_context.json").write_text(result.context.model_dump_json())
    evidence = EvidenceStore(root=source.run)
    for name in ("research_context", "exposure_groupings"):
        evidence.register_file(kind="log", description=name, source_path=source.run / f"{name}.json",
                               evidence_id=name, producer="pipeline", generation_mode="system")
    seal_prepared_source(source, evidence)
    study = complete_study()
    study.update(id=source.study["id"], question=source.context.research_question,
                 primary_exposure="", covariates=[], covariate_rationales={}, covariate_temporal_roles={})
    source.export = cohort.parent.parent / "export"
    study["data_source"] = {"path": str(source.export), "database": "miiv"}
    study["execution_concepts"] = {"outcome": "death"}
    source.study.clear()
    source.study.update(study)
    source.binding = dataio.validate_research_pipeline_source(str(source.export), database="miiv")["binding"]
    update_seed(source, study=study, budget_mode="full_reviewed", prepared_package_binding=source.binding,
                scientific_configuration_sha256=study_contexts.scientific_configuration_sha256(study))
    # The checkpoint-chain owner has separate integrity tests. Bind these
    # synthetic checkpoint bytes, including any later mutation, to the offer.
    import hashlib

    def checkpoint(**kwargs):
        run = source.run.parents[2] / f"run_{kwargs['source_job_id']}" / "pipeline/run_pipeline"
        path = run / "progressive_planner_checkpoint_002.json"
        return path, hashlib.sha256(path.read_bytes()).hexdigest()

    monkeypatch.setattr(owner, "_development_progressive_resume_binding", checkpoint)
    return source


def update_seed(source, **changes):
    seed = agent_review_recovery.load_recovery_seed(source.run.parents[1])
    payload = seed.model_dump(exclude={"seed_sha256", "schema_version"})
    payload.update(changes)
    agent_review_recovery.put_recovery_seed(agent_review_recovery.WebReviewRecoverySeed.create(**payload))


def offer(source):
    return owner.development_resume_available(
        wrapper_dir=source.run.parents[1], study_id=source.study["id"],
        scientific_configuration_sha256=study_contexts.scientific_configuration_sha256(source.study),
    )


def projection(source, monkeypatch):
    digest = study_contexts.scientific_configuration_sha256(source.study)
    failed = {
        "run_id": "run_prior", "study_id": source.study["id"], "run_type": "full",
        "engine": "easyicu.research_agent.pipeline", "gate_status": "blocked",
        "gate_reason": "research_pipeline_planner_provider_unavailable", "run_status": "failed",
        "scientific_configuration_sha256": digest, "artifact_names": ["source_run_manifest.json"],
        "project_dir": str(source.run.parents[1]), "updated_at_epoch": 3,
    }
    second_wrapper = source.run.parents[2] / "run_second"
    shutil.copytree(source.run.parents[1], second_wrapper)
    seed = agent_review_recovery.load_recovery_seed(source.run.parents[1])
    payload = seed.model_dump(exclude={"seed_sha256", "schema_version"})
    payload["wrapper_dir"] = str(second_wrapper.resolve())
    agent_review_recovery.put_recovery_seed(agent_review_recovery.WebReviewRecoverySeed.create(**payload))
    second = dict(failed, run_id="run_second", updated_at_epoch=2, project_dir=str(second_wrapper))
    candidate = dict(failed, run_id="run_candidate", updated_at_epoch=1,
                     gate_reason="human_plan_review_required", run_status="human_review_pending",
                     artifact_names=["agent_plan.json", "source_run_manifest.json"],
                     pending_review_reason_codes=["operator_plan_approval_required"])
    # Both roots see two distinct wrappers with the same content identities.
    history = Mock(return_value={"runs": [failed, second, candidate]})
    monkeypatch.setattr(run_authority.agent_runs, "list_run_history", history)
    monkeypatch.setattr(workflow.sources, "load_registry", lambda: {})
    monkeypatch.setattr(workflow.agent_pipeline_runs, "pending_review", lambda _: {
        "run_id": candidate["run_id"], "resumable_here": True, "budget_mode": "planner_canary",
        "scientific_configuration_sha256": digest, "research_input_state": "metadata_only",
        "plan_approval_allowed": True,
    })
    monkeypatch.setattr(workflow.agent_runs, "read_run_review", lambda _: {})

    def project():
        return workflow.build_project_workflow_projection(
            study_context_id=source.study["id"], study_override=source.study,
        )

    return project, history


def test_two_workflow_reads_and_both_roots_rehearse_and_snapshot_once(prepared_source, monkeypatch):
    rehearsal = Mock(wraps=prepared_owner.restore_exposure_grouping_phase)
    package = Mock(wraps=export_package.open_export_package)
    monkeypatch.setattr(prepared_owner, "restore_exposure_grouping_phase", rehearsal)
    monkeypatch.setattr(export_package, "open_export_package", package)
    project, history = projection(prepared_source, monkeypatch)
    start = time.perf_counter()
    first = project()
    first_seconds = time.perf_counter() - start
    start = time.perf_counter()
    second = project()
    second_seconds = time.perf_counter() - start
    print(json.dumps({"first_seconds": first_seconds, "second_seconds": second_seconds,
                      "rehearsals": rehearsal.call_count, "package_snapshots": package.call_count}))
    assert first.workflow.next_action_code == second.workflow.next_action_code == "planner_checkpoint_resume_available"
    assert history.call_count == 4
    assert rehearsal.call_count == 1
    assert package.call_count == 1


def test_unexpected_rehearsal_error_only_hides_offer(prepared_source, monkeypatch, caplog):
    rehearsal = Mock(side_effect=RuntimeError("synthetic rehearsal defect"))
    monkeypatch.setattr(prepared_owner, "restore_exposure_grouping_phase", rehearsal)
    project, _ = projection(prepared_source, monkeypatch)
    result = project()
    assert result.workflow.latest_attempt_failure.checkpoint_resume_available is False
    assert result.workflow.next_action_code == "plan_execution_upgrade_required"
    assert "RuntimeError" in caplog.text
    assert rehearsal.call_count == 1
    # Programming errors in the authoritative launch must still surface.
    with pytest.raises(RuntimeError, match="synthetic rehearsal defect"):
        prepared_source.scope()


def test_launch_rehearses_after_an_offer_memo_hit(prepared_source, monkeypatch):
    rehearsal = Mock(wraps=prepared_owner.restore_exposure_grouping_phase)
    monkeypatch.setattr(prepared_owner, "restore_exposure_grouping_phase", rehearsal)
    assert offer(prepared_source)
    assert offer(prepared_source)
    rehearsal.reset_mock()
    assert prepared_source.scope().prepared_grouping is not None
    assert rehearsal.call_count == 1


def test_failed_rehearsal_is_memoized(prepared_source, monkeypatch):
    rehearsal = Mock(side_effect=ValueError("synthetic context mismatch"))
    monkeypatch.setattr(prepared_owner, "restore_exposure_grouping_phase", rehearsal)
    assert offer(prepared_source) is False
    assert offer(prepared_source) is False
    assert rehearsal.call_count == 1


@pytest.mark.parametrize("changed", ["checkpoint", "requirements", "config", "study"])
def test_verified_scope_changes_rehearse_again(prepared_source, monkeypatch, changed):
    source = prepared_source
    rehearsal = Mock(wraps=prepared_owner.restore_exposure_grouping_phase)
    monkeypatch.setattr(prepared_owner, "restore_exposure_grouping_phase", rehearsal)
    assert offer(source)
    if changed == "checkpoint":
        path = source.run / "progressive_planner_checkpoint_002.json"
        path.write_bytes(path.read_bytes() + b"\n")
    elif changed == "requirements":
        seed = agent_review_recovery.load_recovery_seed(source.run.parents[1])
        update_seed(source, pipeline_config={**seed.pipeline_config, "required_primary_cohort_selection_mode": "all_input_rows"})
    else:
        source.study["question" if changed == "config" else "id"] += "-revised"
        update_seed(source, study=source.study,
                    scientific_configuration_sha256=study_contexts.scientific_configuration_sha256(source.study))
    assert offer(source)
    assert rehearsal.call_count == 2


@pytest.mark.parametrize("changed", ["record", "capsule", "cohort", "package_binding"])
def test_changed_sealed_inputs_cannot_hit_a_true_offer(prepared_source, changed):
    source = prepared_source
    assert offer(source)
    if changed == "record":
        path = source.run / "exposure_groupings.json"
    elif changed == "capsule":
        path = source.run / "run_input_capsule.json"
    elif changed == "cohort":
        continuation = source.scope().prepared_grouping
        path = source.universe.parent / continuation.authority_ref.file
    else:
        update_seed(source, prepared_package_binding={**source.binding, "export_authority_sha256": "b" * 64})
        path = None
    if path is not None:
        path.write_bytes(path.read_bytes() + b"\n")
    assert offer(source) is False


@pytest.mark.parametrize("changed", [
    "scope", "capsule", "grouping_record", "grouped_authority",
    "source_cohort_authority", "reviewed_configuration", "prepared_package",
])
def test_each_verified_content_identity_separates_memo_entries(prepared_source, monkeypatch, changed):
    """Isolate each verified identity, including seals redundant in real inputs.

    Integrity tests above reject unsealed changes. Here the cheap verifier has
    already accepted a new receipt; changing exactly one identity must miss.
    """
    original = prepared_owner._rehearsal_identity
    captured = []

    def identity(inputs):
        captured.append(dict(inputs))
        return original(inputs)

    monkeypatch.setattr(prepared_owner, "_rehearsal_identity", identity)
    assert offer(prepared_source)
    capsule_path = prepared_source.run / "run_input_capsule.json"
    capsule = json.loads(capsule_path.read_bytes())
    context = prepared_source.context.model_dump(mode="json")
    expected = {
        "scope": (
            hashlib.sha256((prepared_source.run / "progressive_planner_checkpoint_002.json").read_bytes()).hexdigest(),
            study_contexts.scientific_configuration_sha256(prepared_source.study), prepared_source.study["id"],
        ),
        "capsule": hashlib.sha256(capsule_path.read_bytes()).hexdigest(),
        "grouping_record": hashlib.sha256((prepared_source.run / "exposure_groupings.json").read_bytes()).hexdigest(),
        "grouped_authority": context["materialized_inputs"]["cohort"]["authority_ref"]["sha256"],
        "source_cohort_authority": capsule["scientific_identity"]["materialized_cohort_authority_ref"]["sha256"],
        "reviewed_configuration": {"bound_plan_revision_contract": None},
        "prepared_package": prepared_source.binding["export_authority_sha256"],
    }
    assert captured[0][changed] == expected[changed]
    old_key = original(captured[0])
    new_inputs = dict(captured[0], **{changed: {"synthetic_revised_receipt": True}})
    new_key = original(new_inputs)
    assert new_key != old_key
    assert old_key in prepared_owner._OFFER_VERDICTS
    assert new_key not in prepared_owner._OFFER_VERDICTS


def test_requirements_key_is_canonical_and_ignores_execution_options(prepared_source, monkeypatch):
    source = prepared_source
    seed = agent_review_recovery.load_recovery_seed(source.run.parents[1])
    config = dict(seed.pipeline_config, required_primary_cohort_selection_mode="all_input_rows")
    update_seed(source, pipeline_config=config)
    rehearsal = Mock(wraps=prepared_owner.restore_exposure_grouping_phase)
    monkeypatch.setattr(prepared_owner, "restore_exposure_grouping_phase", rehearsal)
    assert offer(source)
    config = dict(reversed(list(config.items())))
    config["execute_until_phase"] = "planning"
    update_seed(source, pipeline_config=config)
    assert offer(source)
    assert rehearsal.call_count == 1


def test_offer_rehearsal_memo_is_bounded(prepared_source, monkeypatch):
    source = prepared_source
    monkeypatch.setattr(prepared_owner, "_OFFER_MEMO_LIMIT", 2)
    rehearsal = Mock(wraps=prepared_owner.restore_exposure_grouping_phase)
    monkeypatch.setattr(prepared_owner, "restore_exposure_grouping_phase", rehearsal)
    seed = agent_review_recovery.load_recovery_seed(source.run.parents[1])
    for mode in (None, "all_input_rows", "predicate_filtered", None):
        update_seed(source, pipeline_config={**seed.pipeline_config, "required_primary_cohort_selection_mode": mode})
        assert offer(source)
        assert len(prepared_owner._OFFER_VERDICTS) <= 2
    assert rehearsal.call_count == 4


@pytest.mark.parametrize("change", ["inode", "mtime", "size", "sidecar", "manifest"])
def test_package_memo_invalidates_on_members_and_manifest(prepared_source, monkeypatch, change):
    source = prepared_source
    package = Mock(wraps=export_package.open_export_package)
    monkeypatch.setattr(export_package, "open_export_package", package)
    assert offer(source)
    assert package.call_count == 1
    path = source.export / "labs.parquet"
    original_stat = path.stat()
    if change == "inode":
        replacement = path.with_suffix(".replacement")
        replacement.write_bytes(path.read_bytes())
        os.utime(replacement, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
        replacement.replace(path)
    elif change == "mtime":
        stat = path.stat()
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
    else:
        if change == "sidecar":
            manifest = json.loads((source.export / "_manifest.json").read_text())
            path = source.export / manifest["column_metadata"]["file"]
        elif change == "manifest":
            path = source.export / "_manifest.json"
        raw = path.read_bytes()
        path.write_bytes(raw.replace(b'"explicit"', b'"changed!"') if change == "manifest" else raw + b"\n")
        if change == "size":
            os.utime(path, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
    assert offer(source) is (change in {"inode", "mtime"})
    # A manifest mismatch is refused before opening any member snapshot.
    assert package.call_count == (1 if change == "manifest" else 2)
    assert offer(source) is (change in {"inode", "mtime"})
    assert package.call_count == (1 if change == "manifest" else 2)


def test_launch_refuses_in_place_package_mutation_even_with_unchanged_stat(prepared_source):
    source = prepared_source
    assert offer(source)
    path = source.export / "labs.parquet"
    stat = path.stat()
    raw = path.read_bytes()
    path.write_bytes(b"FAIL" + raw[4:])
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    assert offer(source)  # documented metadata-memo residual case
    with pytest.raises(owner.ResearchPipelineRunError) as refused:
        source.scope()
    assert refused.value.code == owner.DEVELOPMENT_RESUME_PHASE_NOT_REPLAYED
    assert str(refused.value) == "The saved preparation could not be verified for continuation; prepare the same reviewed candidate again."


def test_package_false_verdict_is_memoized(prepared_source, monkeypatch):
    validate = Mock(side_effect=dataio.ExportCohortError("synthetic_package_refusal"))
    monkeypatch.setattr(dataio, "validate_research_pipeline_source", validate)
    assert offer(prepared_source) is False
    assert offer(prepared_source) is False
    assert validate.call_count == 1


def test_launch_compares_its_existing_package_receipt(prepared_source, monkeypatch):
    source = prepared_source
    validate = Mock(side_effect=AssertionError("launch must reuse its existing package receipt"))
    monkeypatch.setattr(dataio, "validate_research_pipeline_source", validate)
    kwargs = dict(project_root=str(source.run.parents[3]), study=source.study, source_job_id="prior")
    assert owner._development_resume_launch_scope(**kwargs, validated_package_binding=source.binding).prepared_grouping
    with pytest.raises(owner.ResearchPipelineRunError) as refused:
        owner._development_resume_launch_scope(
            **kwargs, validated_package_binding={**source.binding, "export_authority_sha256": "c" * 64},
        )
    assert refused.value.code == owner.DEVELOPMENT_RESUME_PHASE_NOT_REPLAYED
    assert validate.call_count == 0


def test_package_identity_includes_device(prepared_source, monkeypatch):
    from pathlib import Path
    from easyicu.webserver.research_resume_package import prepared_package_identity

    source = prepared_source
    old = prepared_package_identity(str(source.export), source.binding)
    original = Path.stat

    class AnotherDevice:
        def __init__(self, value):
            self.value = value
            self.st_dev = value.st_dev + 1

        def __getattr__(self, name):
            return getattr(self.value, name)

    def stat(path, *args, **kwargs):
        value = original(path, *args, **kwargs)
        return AnotherDevice(value) if path == source.export / "labs.parquet" else value

    monkeypatch.setattr(Path, "stat", stat)
    assert prepared_package_identity(str(source.export), source.binding) != old


def test_package_memo_is_bounded():
    from easyicu.webserver import research_resume_package

    assert research_resume_package._package_offer_verdict.cache_info().maxsize == 128


def test_memo_hit_does_not_copy_build_or_snapshot(prepared_source, monkeypatch):
    assert offer(prepared_source)
    refuse = Mock(side_effect=AssertionError("a memo hit must not redo expensive work"))
    monkeypatch.setattr(prepared_owner, "stage_materialized_cohort_authority", refuse)
    monkeypatch.setattr(prepared_owner, "build_research_context", refuse)
    monkeypatch.setattr(export_package, "open_export_package", refuse)
    assert offer(prepared_source)
    assert refuse.call_count == 0


@pytest.mark.parametrize("name", [
    "bound_target_trial", "bound_baseline_requirements", "bound_population_requirements",
    "bound_analysis_inputs", "required_primary_cohort_selection_mode", "bound_plan_revision_contract",
])
def test_every_reviewed_config_coordinate_invalidates_rehearsal(prepared_source, monkeypatch, name):
    source = prepared_source
    rehearsal = Mock(wraps=prepared_owner.restore_exposure_grouping_phase)
    monkeypatch.setattr(prepared_owner, "restore_exposure_grouping_phase", rehearsal)
    assert offer(source)
    seed = agent_review_recovery.load_recovery_seed(source.run.parents[1])
    update_seed(source, pipeline_config={**seed.pipeline_config, name: "" if name == "bound_plan_revision_contract" else None})
    assert offer(source)
    assert rehearsal.call_count == 2


def test_rehearsal_identity_canonicalizes_nested_reviewed_requirements():
    assert prepared_owner._rehearsal_identity({"requirements": {"a": 1, "b": 2}, "id": "synthetic"}) == (
        prepared_owner._rehearsal_identity({"id": "synthetic", "requirements": {"b": 2, "a": 1}})
    )


def test_package_mutation_during_validation_cannot_populate_old_identity(prepared_source, monkeypatch):
    source = prepared_source
    original = dataio.validate_research_pipeline_source

    def mutate(*args, **kwargs):
        receipt = original(*args, **kwargs)
        path = source.export / "labs.parquet"
        stat = path.stat()
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
        return receipt

    monkeypatch.setattr(dataio, "validate_research_pipeline_source", mutate)
    assert offer(source) is False


@pytest.mark.parametrize("kind", ["column_metadata", "feature_definitions"])
def test_queued_runner_byte_fence_covers_bound_sidecars(prepared_source, kind):
    from easyicu.webserver.research_resume_package import prepared_package_identity

    source = prepared_source
    binding = dict(source.binding)
    manifest_path = source.export / "_manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    if kind == "column_metadata":
        path = source.export / manifest[kind]["file"]
    else:
        # Isolate the receipt fence, not the intake schema owner. Its validated
        # feature authority is a raw-byte digest of the one JSON sidecar.
        path = source.export / "features.json"
        path.write_bytes(b"{}")
        manifest[kind] = {"included": True, "files": [{"file": path.name, "kind": kind}]}
        raw = json.dumps(manifest).encode()
        manifest_path.write_bytes(raw)
        binding["feature_definitions_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        binding["manifest"] = {**binding["manifest"], "sha256": hashlib.sha256(raw).hexdigest(), "size_bytes": len(raw)}
    old = prepared_package_identity(str(source.export), binding, verify_content=True)
    stat = path.stat()
    path.write_bytes(b"!" + path.read_bytes()[1:])
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    assert prepared_package_identity(str(source.export), binding) == old
    with pytest.raises(ValueError, match="bytes changed"):
        prepared_package_identity(str(source.export), binding, verify_content=True)


def test_rehearsal_keeps_the_plan_review_flag_real_seeds_carry(prepared_source, monkeypatch):
    """A reviewed requirement is valid config only under plan review (attempt 10's seed)."""
    from easyicu.research_agent.planning.baseline_requirements import AcceptedBaselineRequirements

    source = prepared_source
    seed = agent_review_recovery.load_recovery_seed(source.run.parents[1])
    requirements = AcceptedBaselineRequirements.model_validate({
        "source_plan_sha256": "b" * 64, "tables": [{
            "source_step_id": "baseline", "group_by": {"name": "lact_group_x1"},
            "variables": [{"name": "age"}],
        }],
    }).model_dump(mode="json")
    update_seed(source, pipeline_config={
        **seed.pipeline_config, "bound_baseline_requirements": requirements,
        "require_human_plan_review": True,
    })
    # The source context here predates the added requirement, so the restore
    # comparison is not the subject; the rehearsal's config construction is.
    rehearsal = Mock(return_value=None)
    monkeypatch.setattr(prepared_owner, "restore_exposure_grouping_phase", rehearsal)
    assert offer(source)
    assert rehearsal.call_count == 1
