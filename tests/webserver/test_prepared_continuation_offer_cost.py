"""Continuation projections reuse verified work; launches remain authoritative."""

import hashlib
import json
import os
import shutil
import time
import threading
from unittest.mock import Mock

import pytest

from easyicu.research_agent.authority.evidence_store import EvidenceStore
from easyicu.research_agent.intake import export_package
from easyicu.research_agent.providers.mocks import ScriptedMockLLMClient
from easyicu.webserver import agent_review_recovery, dataio, research_launch_resume as owner
from easyicu.webserver import research_prepared_resume as prepared_owner, study_contexts
from easyicu.webserver import research_resume_offer as offers
from easyicu.webserver.pi_copilot import run_authority, workflow
from tests.support.grouped_resume import ANSWER, phase, prepared, seal_prepared_source, source as _source
from tests.webserver.copilot.research_workflow_fixtures import complete_study

source = _source
_SUBPROCESS_VERIFY = offers._verify_background
_REAL_CHECKPOINT_BINDING = owner._development_progressive_resume_binding


@pytest.fixture(autouse=True)
def finish_offer_work(monkeypatch):
    offers._WORK.join()
    # Run the exact verifier in an instrumentable thread for unit assertions.
    # The real subprocess boundary has its own integration test below.
    monkeypatch.setattr(offers, '_verify_background', offers._verify)
    yield
    offers._WORK.join()


def page_offer(source):
    return owner.development_resume_available(
        wrapper_dir=source.run.parents[1], study_id=source.study["id"],
        scientific_configuration_sha256=study_contexts.scientific_configuration_sha256(source.study),
    )




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
        verify_now=True,
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




def test_unexpected_rehearsal_error_only_hides_offer(prepared_source, monkeypatch, caplog):
    rehearsal = Mock(side_effect=RuntimeError("synthetic rehearsal defect"))
    monkeypatch.setattr(prepared_owner, "restore_exposure_grouping_phase", rehearsal)
    project, _ = projection(prepared_source, monkeypatch)
    project()
    offers._WORK.join()
    result = project()
    assert result.workflow.latest_attempt_failure.checkpoint_resume_available is False
    assert result.workflow.next_action_code == "plan_execution_upgrade_required"
    assert "RuntimeError" in caplog.text
    assert rehearsal.call_count == 2
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




def test_configuration_changes_invalidate_the_persisted_verdict(prepared_source, monkeypatch):
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
    assert rehearsal.call_count == 2




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




def test_memo_hit_does_not_copy_build_or_snapshot(prepared_source, monkeypatch):
    from easyicu.research_agent.intake import materialized_metadata

    assert offer(prepared_source)
    refuse = Mock(side_effect=AssertionError("a memo hit must not redo expensive work"))
    monkeypatch.setattr(prepared_owner, "stage_materialized_cohort_authority", refuse)
    monkeypatch.setattr(prepared_owner, "build_research_context", refuse)
    monkeypatch.setattr(export_package, "open_export_package", refuse)
    monkeypatch.setattr(materialized_metadata, "_row_identity_sha256", refuse)
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


def test_workflow_reads_only_metadata_and_stats_with_persisted_verdicts(prepared_source, monkeypatch):
    from easyicu.research_agent.intake import materialized_metadata

    source = prepared_source
    project, history = projection(source, monkeypatch)
    for name in ('run_prior', 'run_second'):
        assert owner.development_resume_available(
            wrapper_dir=source.run.parents[2] / name, study_id=source.study['id'],
            scientific_configuration_sha256=study_contexts.scientific_configuration_sha256(source.study),
            verify_now=True,
        )
    refuse = Mock(side_effect=AssertionError('a page must not verify cohort bytes or rehearse'))
    for module, name in (
        (materialized_metadata, '_row_identity_sha256'),
        (prepared_owner, 'build_research_context'),
        (prepared_owner, 'stage_materialized_cohort_authority'),
        (export_package, 'open_export_package'),
        (prepared_owner, 'load_verified_run_input_capsule'),
    ):
        monkeypatch.setattr(module, name, refuse)
    first, second = project(), project()
    assert first.workflow.next_action_code == second.workflow.next_action_code == 'planner_checkpoint_resume_available'
    assert history.call_count == 4
    assert refuse.call_count == 0


@pytest.mark.parametrize('stale', [False, True])
def test_missing_or_stale_verdict_is_prompt_and_deduplicates_until_finished(prepared_source, monkeypatch, stale):
    source = prepared_source
    if stale:
        assert offer(source)
        path = source.universe
        before = path.stat()
        os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns + 1_000_000))
    entered, release = threading.Event(), threading.Event()
    real = owner.verify_development_resume_offer
    calls = []

    def slow(**kwargs):
        calls.append(threading.current_thread().name)
        entered.set()
        assert release.wait(10)
        return real(**kwargs)

    monkeypatch.setattr(owner, 'verify_development_resume_offer', slow)
    try:
        start = time.perf_counter()
        assert page_offer(source) is False
        assert time.perf_counter() - start < 0.1
        assert entered.wait(5)
        for _ in range(5):
            assert page_offer(source) is False
        assert calls == ['continuation-offers']
        assert len(offers._PENDING) == 1
        assert offers._WORK.qsize() == 0
    finally:
        release.set()
        offers._WORK.join()
    assert page_offer(source) is True
    assert calls == ['continuation-offers']


def test_restart_reads_only_the_persisted_key_and_boolean(prepared_source, monkeypatch):
    from easyicu.webserver import research_resume_package

    source = prepared_source
    assert offer(source)
    prepared_owner._OFFER_VERDICTS.clear()
    research_resume_package._package_offer_verdict.cache_clear()
    wrapper = source.run.parents[1]
    record = json.loads((wrapper / offers.RECORD_NAME).read_bytes())
    assert set(record) == {'schema_version', 'key', 'verdict'}
    assert len(record['key']) == 64 and record['verdict'] is True
    # No in-memory verdict is needed. Reload in a fresh interpreter as well.
    import subprocess
    import sys

    code = """
import json, sys
from pathlib import Path
from easyicu.webserver import research_launch_resume as owner
owner.verify_development_resume_offer = lambda **kw: (_ for _ in ()).throw(AssertionError('restart rehearsed'))
print(json.dumps(owner.development_resume_available(wrapper_dir=Path(sys.argv[1]), study_id=sys.argv[2], scientific_configuration_sha256=sys.argv[3])))
"""
    result = subprocess.run([sys.executable, '-c', code, str(wrapper), source.study['id'],
                             study_contexts.scientific_configuration_sha256(source.study)],
                            check=True, capture_output=True, text=True)
    assert json.loads(result.stdout) is True


@pytest.mark.parametrize('changed', ['device', 'inode', 'size', 'mtime', 'provenance', 'source_authority', 'grouped_authority', 'record', 'config', 'package'])
def test_every_input_change_hides_the_old_verdict_before_reverification(prepared_source, monkeypatch, changed):
    from pathlib import Path

    source = prepared_source
    assert offer(source)
    record = source.run.parents[1] / offers.RECORD_NAME
    before = json.loads(record.read_bytes())
    path = source.universe
    if changed == 'device':
        original = Path.lstat
        class AnotherDevice:
            def __init__(self, value):
                self.value, self.st_dev = value, value.st_dev + 1
            def __getattr__(self, name):
                return getattr(self.value, name)
        monkeypatch.setattr(Path, 'lstat', lambda p, *a, **kw: AnotherDevice(original(p, *a, **kw)) if p == path else original(p, *a, **kw))
    elif changed == 'inode':
        replacement = path.with_suffix('.replacement')
        replacement.write_bytes(path.read_bytes())
        info = path.stat()
        replacement.chmod(info.st_mode)
        os.utime(replacement, ns=(info.st_atime_ns, info.st_mtime_ns))
        replacement.replace(path)
    elif changed in {'mtime', 'package'}:
        path = source.export / 'labs.parquet' if changed == 'package' else path
        info = path.stat()
        os.utime(path, ns=(info.st_atime_ns, info.st_mtime_ns + 1_000_000))
    elif changed == 'config':
        seed = agent_review_recovery.load_recovery_seed(source.run.parents[1])
        update_seed(source, pipeline_config={**seed.pipeline_config, 'require_human_plan_review': True})
    else:
        if changed == 'record':
            path = source.run / 'exposure_groupings.json'
        elif changed == 'provenance':
            path = prepared_owner.materialized_provenance_path(path)
        elif changed == 'source_authority':
            capsule = json.loads((source.run / 'run_input_capsule.json').read_bytes())
            path = path.parent / capsule['scientific_identity']['materialized_cohort_authority_ref']['file']
        elif changed == 'grouped_authority':
            path = source.run / source.context.materialized_inputs.cohort.authority_ref['file']
        info = path.stat()
        path.write_bytes(path.read_bytes() + b'\n')
        if changed == 'size':
            os.utime(path, ns=(info.st_atime_ns, info.st_mtime_ns))
    assert page_offer(source) is False
    offers._WORK.join()
    after = json.loads(record.read_bytes())
    assert after['key'] != before['key']
    assert page_offer(source) is after['verdict']


def test_background_worker_is_bounded_and_queue_full_never_verifies_inline(prepared_source, monkeypatch):
    from queue import Queue

    source = prepared_source
    assert offers._WORK.maxsize == 128
    work = Queue(maxsize=2)
    monkeypatch.setattr(offers, '_WORK', work)
    monkeypatch.setattr(offers, '_WORKER', Mock(is_alive=lambda: True))
    monkeypatch.setattr(owner, 'verify_development_resume_offer', lambda **kw: pytest.fail('inline verification'))
    try:
        for index in range(3):
            seed = agent_review_recovery.load_recovery_seed(source.run.parents[1])
            update_seed(source, pipeline_config={**seed.pipeline_config, 'synthetic_generation': index})
            assert page_offer(source) is False
        assert work.qsize() == len(offers._PENDING) == 2
        assert offers.Queue is Queue
    finally:
        while not work.empty():
            wrapper, _, _, _, key = work.get_nowait()
            offers._finish((wrapper, key))
            work.task_done()


def test_failed_job_records_the_verdict_in_its_thread_before_terminal(prepared_source, monkeypatch):
    from easyicu.webserver import agent_pipeline_runs, research_job_recovery
    from easyicu.webserver import provider_job_recovery as policy
    from easyicu.webserver.jobs import Job
    from types import SimpleNamespace

    source = prepared_source
    job = Job('synthetic', 'agent-run')
    job.status = 'running'
    state = policy.ProviderJobRecovery(job)
    state.bind(wrapper_dir=source.run.parents[1], client=object(), task=SimpleNamespace(
        ledger=SimpleNamespace(snapshot=lambda: {'tasks': []})))
    monkeypatch.setattr(study_contexts, 'get_context', lambda _: source.study)
    # Stop at the selection boundary: no provider probe, job or approval.
    def no_launch(**kw):
        assert (source.run.parents[1] / offers.RECORD_NAME).is_file()
        assert job.status == 'running'
        raise owner.ResearchPipelineRunError('research_pipeline_development_resume_checkpoint_invalid', 'synthetic stop')
    monkeypatch.setattr(agent_pipeline_runs, 'make_research_pipeline_run_runner', no_launch)
    real = owner.verify_development_resume_offer
    verified = []
    def verify(**kw):
        verified.append((threading.current_thread().name, job.status))
        return real(**kw)
    monkeypatch.setattr(owner, 'verify_development_resume_offer', verify)
    failures = []
    def run():
        try:
            state.run(lambda _: (_ for _ in ()).throw(ConnectionError('synthetic')),
                      lambda exc: research_job_recovery._continuation(state, {'study_context': source.study}))
        except ConnectionError:
            failures.append(json.loads((source.run.parents[1] / offers.RECORD_NAME).read_bytes())['verdict'])
            job.status = 'failed'
    thread = threading.Thread(target=run, name='synthetic-job-thread')
    thread.start()
    thread.join(20)
    assert not thread.is_alive()
    assert verified == [('synthetic-job-thread', 'running')]
    assert failures == [True] and job.status == 'failed'


def test_verification_never_calls_a_provider_or_changes_approval(prepared_source, monkeypatch):
    from easyicu.webserver import agent_pipeline_runs, provider_adapter
    from easyicu.research_agent.providers.hard_stop import HardStopClient

    refuse = Mock(side_effect=AssertionError('verification cannot spend or launch'))
    monkeypatch.setattr(HardStopClient, 'complete', refuse)
    monkeypatch.setattr(provider_adapter, 'build_research_agent_provider_client', refuse)
    monkeypatch.setattr(agent_pipeline_runs, 'make_research_pipeline_run_runner', refuse)
    monkeypatch.setattr(agent_pipeline_runs, 'resume_research_pipeline', refuse)
    assert page_offer(prepared_source) is False
    offers._WORK.join()
    assert page_offer(prepared_source) is True
    assert refuse.call_count == 0
    assert not list(prepared_source.run.parents[1].rglob('*approval*'))


def test_grouping_content_digest_invalidates_even_with_unchanged_stat(prepared_source):
    source = prepared_source
    assert offer(source)
    path = source.run / 'exposure_groupings.json'
    before = path.stat()
    raw = path.read_bytes()
    changed = raw.replace(b'lactate below', b'Lactate below')
    assert changed != raw and len(changed) == len(raw)
    path.write_bytes(changed)
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    assert page_offer(source) is False
    offers._WORK.join()
    assert page_offer(source) is False


def test_launch_rechecks_cohort_bytes_even_when_stat_and_offer_are_unchanged(prepared_source):
    source = prepared_source
    assert offer(source)
    path = source.universe
    before = path.stat()
    path.write_bytes(b'FAIL' + path.read_bytes()[4:])
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    assert page_offer(source) is True
    with pytest.raises(owner.ResearchPipelineRunError) as refused:
        source.scope()
    assert refused.value.code == owner.DEVELOPMENT_RESUME_PHASE_NOT_REPLAYED


def real_checkpoint(source):
    from easyicu.research_agent.canonical_json import canonical_sha256
    from easyicu.research_agent.planning.progressive_artifacts import (
        ProgressivePlannerCheckpointEmitter, persist_progressive_planner_checkpoint,
    )
    from easyicu.research_agent.planning.progressive_contract import ProgressivePlanOutline
    from tests.research_agent.planning.progressive_planner_fixtures import _outline_payload

    outline = ProgressivePlanOutline.model_validate(_outline_payload())
    checkpoints = []
    ProgressivePlannerCheckpointEmitter(callback=checkpoints.append, request_authority_sha256='a' * 64).emit(
        stage='outline', outline=outline, foundation=None, materializations=[],
        prompt_metrics={'outline_sha256': canonical_sha256(outline.model_dump(mode='json'))},
    )
    (source.run / 'progressive_planner_checkpoint_002.json').unlink()
    persist_progressive_planner_checkpoint(
        run_dir=source.run, evidence=EvidenceStore(root=source.run), checkpoint=checkpoints[0],
        prompt_pack_version='synthetic-v1',
    )


def test_real_background_subprocess_publishes_a_reusable_verdict(prepared_source, monkeypatch):
    source = prepared_source
    real_checkpoint(source)
    monkeypatch.setattr(owner, '_development_progressive_resume_binding', _REAL_CHECKPOINT_BINDING)
    monkeypatch.setattr(offers, '_verify_background', _SUBPROCESS_VERIFY)
    monkeypatch.setattr(owner, 'verify_development_resume_offer', Mock(side_effect=AssertionError('background verification must leave the web interpreter')))
    assert page_offer(source) is False
    offers._WORK.join()
    assert page_offer(source) is True
    record = json.loads((source.run.parents[1] / offers.RECORD_NAME).read_bytes())
    assert record['verdict'] is True
    assert not list(source.run.parents[1].rglob('*approval*'))


def test_distinct_keys_share_one_background_verifier(prepared_source, monkeypatch):
    source = prepared_source
    entered, second, release = threading.Event(), threading.Event(), threading.Event()
    calls = []
    real = offers._verify
    def blocked(*args):
        calls.append(args[-1])
        (entered if len(calls) == 1 else second).set()
        assert release.wait(10)
        return real(*args)
    monkeypatch.setattr(offers, '_verify_background', blocked)
    try:
        assert page_offer(source) is False
        assert entered.wait(5)
        seed = agent_review_recovery.load_recovery_seed(source.run.parents[1])
        update_seed(source, pipeline_config={**seed.pipeline_config, 'require_human_plan_review': True})
        assert page_offer(source) is False
        assert not second.wait(0.1)
        assert offers._WORK.qsize() == 1
    finally:
        release.set()
        offers._WORK.join()
    assert len(calls) == 2 and calls[0] != calls[1]
    assert page_offer(source) is True


def test_verdict_replace_is_atomic_and_failed_replace_preserves_old(tmp_path, monkeypatch):
    offers._write_verdict(tmp_path, 'a' * 64, True)
    path = tmp_path / offers.RECORD_NAME
    before = path.read_bytes()
    def refuse_replace(temporary, target):
        from pathlib import Path
        assert target == path and path.read_bytes() == before
        assert json.loads(Path(temporary).read_bytes())['key'] == 'b' * 64
        raise OSError('synthetic interrupted replace')
    monkeypatch.setattr(offers.os, 'replace', refuse_replace)
    with pytest.raises(OSError, match='synthetic interrupted replace'):
        offers._write_verdict(tmp_path, 'b' * 64, False)
    assert path.read_bytes() == before
    assert not list(tmp_path.glob('.resume-offer-*'))


def test_offer_metadata_never_follows_a_symlink(tmp_path):
    target = tmp_path / 'synthetic.json'
    target.write_text('{}')
    link = tmp_path / 'linked.json'
    link.symlink_to(target)
    with pytest.raises(OSError):
        offers._small_bytes(link)


def test_a_hung_background_verifier_is_bounded(monkeypatch, caplog):
    import subprocess
    from pathlib import Path

    seen = {}

    def hung(command, **kwargs):
        seen.update(kwargs)
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])

    monkeypatch.setattr(offers.subprocess, "run", hung)
    _SUBPROCESS_VERIFY(Path("run_synthetic"), None, "study", "digest", "key")
    assert seen["timeout"] == offers._VERIFIER_TIMEOUT_SECONDS
    assert "development_resume_offer_worker_timeout" in caplog.text
