"""Grouped Planner continuations retain sealed input authority; synthetic data only."""

import hashlib
import json
import shutil

import pytest

from easyicu.research_agent.authority.evidence_store import EvidenceStore
from easyicu.research_agent.orchestration import exposure_grouping_phase as grouping
from easyicu.research_agent.providers.mocks import ScriptedMockLLMClient
from easyicu.webserver import research_launch_resume as owner
from tests.support.grouped_resume import (
    ANSWER as _ANSWER,
    catalog as _catalog,
    context as _context,
    phase as _phase,
    prepared as _prepared,
    seal_prepared_source,
    source as _source_fixture,
)

source = _source_fixture


@pytest.fixture(params=["metadata", "prepared", "patient_grouped"])
def grouped_source(source, tmp_path, request):
    cohort = (
        _catalog(tmp_path / "input", "lact", "age")
        if request.param == "metadata" else _prepared(tmp_path / "input", patient_grouped=request.param == "patient_grouped")
    )
    # Both copies begin with exactly the same input, including its typed authority.
    shutil.copytree(cohort.parent, source.run, dirs_exist_ok=True)
    source.cohort = source.run / "cohort.parquet"
    source.original = cohort
    phase = _phase(source.cohort, ScriptedMockLLMClient([_ANSWER]))
    source.context = phase.context
    (source.run / "research_context.json").write_text(phase.context.model_dump_json())
    evidence = EvidenceStore(root=source.run)
    for name in ("research_context", "exposure_groupings"):
        evidence.register_file(
            kind="log", description=name, source_path=source.run / f"{name}.json",
            evidence_id=name, producer="pipeline", generation_mode="system",
        )
    if request.param != "metadata":
        seal_prepared_source(source, evidence)
    return source


def test_sealed_grouped_source_is_available_without_a_provider(grouped_source):
    assert grouped_source.scope().budget_mode == "planner_canary"


def test_offer_refuses_a_missing_prepared_materialization(grouped_source):
    if grouping.load_resume_exposure_groupings(grouped_source.run).materialized_authority_raw is None:
        return
    grouped_source.cohort.unlink()
    with pytest.raises(owner.ResearchPipelineRunError) as refused:
        grouped_source.scope()
    assert refused.value.code == owner.DEVELOPMENT_RESUME_PHASE_NOT_REPLAYED


@pytest.mark.parametrize("damage", ["input", "authority", "capsule", "trajectory", "restore"])
def test_prepared_offer_and_launch_share_the_actual_restore_verdict(grouped_source, monkeypatch, damage):
    from easyicu.webserver import research_prepared_resume, study_contexts

    source = grouped_source
    if not hasattr(source, "universe"):
        return
    if damage == "input":
        source.universe.write_bytes(b"changed source materialization")
    elif damage == "authority":
        authority = grouping.load_verified_materialized_cohort_authority(source.universe)
        (source.universe.parent / authority.reference.file).write_text("{}")
    elif damage == "trajectory":
        source.universe.with_name("universe_trajectory.parquet").write_bytes(b"unexpected trajectory")
    elif damage == "capsule":
        capsule = source.run / "run_input_capsule.json"
        capsule.write_bytes(capsule.read_bytes() + b"\n")
    else:
        def refuse(**kwargs):
            raise ValueError("synthetic rebuilt context drift")
        monkeypatch.setattr(research_prepared_resume, "restore_exposure_grouping_phase", refuse)
    with pytest.raises(owner.ResearchPipelineRunError) as refused:
        source.scope()
    assert refused.value.code == owner.DEVELOPMENT_RESUME_PHASE_NOT_REPLAYED
    assert owner.development_resume_available(
        wrapper_dir=source.run.parents[1], study_id=source.study["id"],
        scientific_configuration_sha256=study_contexts.scientific_configuration_sha256(source.study),
        verify_now=True,
    ) is False


def _declared_source(source, tmp_path, *, earlier_rule, monkeypatch):
    """A source whose study declared the lactate maximum its groups are formed from."""
    from functools import partial

    shutil.copytree(_catalog(tmp_path / "input", "lact", "age").parent, source.run, dirs_exist_ok=True)
    cohort = source.run / "cohort.parquet"
    if earlier_rule:
        # Before 10-22 a declared exposure was never bound to its groups.
        with monkeypatch.context() as patch:
            patch.setattr(grouping, "_primary_exposure_binding", lambda declared, applied: None)
            phase = grouping.run_exposure_grouping_phase(
                context=_context(cohort, primary_exposure="lact_max"), cohort_path=cohort,
                run_dir=source.run, planner=ScriptedMockLLMClient([_ANSWER]),
                rebuild_context=partial(_context, primary_exposure="lact_max"),
                capability_review_pending=False, trajectory_staged=False,
                emit_progress=lambda *_args, **_kwargs: None,
            )
    else:
        phase = grouping.run_exposure_grouping_phase(
            context=_context(cohort, primary_exposure="lact_max"), cohort_path=cohort,
            run_dir=source.run, planner=ScriptedMockLLMClient([_ANSWER]),
            rebuild_context=partial(_context, primary_exposure="lact_max"),
            capability_review_pending=False, trajectory_staged=False,
            emit_progress=lambda *_args, **_kwargs: None,
        )
    (source.run / "research_context.json").write_text(phase.context.model_dump_json())
    evidence = EvidenceStore(root=source.run)
    for name in ("research_context", "exposure_groupings"):
        evidence.register_file(
            kind="log", description=name, source_path=source.run / f"{name}.json",
            evidence_id=name, producer="pipeline", generation_mode="system",
        )
    return phase


def test_a_source_that_bound_its_exposure_by_an_earlier_rule_is_planned_anew(source, tmp_path, monkeypatch):
    # On the hosted pilot "generate a fresh plan" continued a failed planning
    # run whose record kept the declared lactate maximum beside its groups,
    # and failed again in ten seconds on the same exposure mismatch.
    phase = _declared_source(source, tmp_path, earlier_rule=True, monkeypatch=monkeypatch)
    assert phase.context.primary_exposure == "lact_max"

    with pytest.raises(owner.ResearchPipelineRunError) as refused:
        source.scope()
    assert refused.value.code == owner.DEVELOPMENT_RESUME_PHASE_NOT_REPLAYED


def test_a_source_that_bound_its_declared_exposure_to_its_groups_continues(source, tmp_path, monkeypatch):
    phase = _declared_source(source, tmp_path, earlier_rule=False, monkeypatch=monkeypatch)
    assert phase.context.primary_exposure == "lact_group_x1"

    assert source.scope().budget_mode == "planner_canary"
    restored = grouping.load_resume_exposure_groupings(source.run)
    assert restored.context.primary_exposure == "lact_group_x1"


def test_prepared_input_belongs_to_the_same_package(grouped_source):
    from easyicu.webserver.research_prepared_resume import verify_prepared_grouped_continuation

    source = grouped_source
    if not hasattr(source, "universe"):
        return
    with pytest.raises(ValueError, match="another prepared package"):
        verify_prepared_grouped_continuation(
            run_dir=source.run, source=grouping.load_resume_exposure_groupings(source.run),
            pipeline_config={}, package_binding={"export_authority_sha256": "e" * 64},
        )


def test_second_prepared_continuation_reuses_the_original_sibling_input(grouped_source):
    from easyicu.webserver.research_prepared_resume import verify_prepared_grouped_continuation

    source = grouped_source
    if not hasattr(source, "universe"):
        return
    second = source.run.parents[2] / "run_second" / "pipeline" / "run_second_pipeline"
    shutil.copytree(source.run, second)
    continuation = verify_prepared_grouped_continuation(
        run_dir=second, source=grouping.load_resume_exposure_groupings(second),
        pipeline_config={}, package_binding=None,
    )
    assert continuation.cohort_path == source.universe
    assert continuation == source.scope().prepared_grouping


def test_prepared_continuation_restores_science_without_execution_options(grouped_source):
    from dataclasses import replace

    if not hasattr(grouped_source, "universe"):
        return
    continuation = grouped_source.scope().prepared_grouping
    execution_options = {
        "require_human_plan_review": False,
        "execution_resume_approval_path": "prior-approval.json",
        "execute_until_phase": "analysis",
    }
    with_execution_options = replace(
        continuation, scientific_identity={**continuation.scientific_identity, **execution_options},
    )
    restored = with_execution_options.run_arguments()
    assert not execution_options.keys() & restored.keys()
    assert restored == continuation.run_arguments()


def test_prepared_continuation_reaches_planner_on_the_original_context(grouped_source, tmp_path, monkeypatch):
    from easyicu.research_agent import pipeline as pipeline_module
    from easyicu.research_agent.orchestration import progressive_planning
    from easyicu.research_agent.planning.progressive_resume import progressive_resume_context_sha256
    from easyicu.research_agent.providers.mocks import MockLLMClient

    source = grouped_source
    if not hasattr(source, "universe"):
        return
    continuation = source.scope().prepared_grouping
    acquisition = continuation.acquisition()
    assert acquisition.universe_path == source.universe
    assert acquisition.trajectory_path is None
    client = MockLLMClient()

    class PlannerReached(BaseException):
        pass

    def planner(**kwargs):
        from easyicu.research_agent.research_context.typed import parse_research_context_json

        restored = parse_research_context_json((kwargs["run_dir"] / "research_context.json").read_text())
        assert progressive_resume_context_sha256(restored) == progressive_resume_context_sha256(source.context)
        assert (kwargs["run_dir"] / "exposure_groupings.json").read_bytes() == (source.run / "exposure_groupings.json").read_bytes()
        assert kwargs["resume_checkpoint_path"].parent == source.run
        raise PlannerReached

    monkeypatch.setattr(progressive_planning, "run_progressive_planner", planner)
    pipeline = pipeline_module.ResearchAgentPipeline(
        workdir=tmp_path / "continued", llm=client, planner_strategy="progressive_v2",
        development_diagnostic=True,
        require_human_plan_review=True, enable_visual_qa=False,
        enable_publication_figure_skill=False, enable_nature_writing_skill=False,
        development_progressive_resume_checkpoint_path=source.run / "progressive_planner_checkpoint_002.json",
        development_progressive_resume_checkpoint_sha256="a" * 64,
    )
    with pytest.raises(PlannerReached):
        pipeline.run(
            **continuation.run_arguments(), cohort=acquisition.universe_path,
            cohort_authority_path=acquisition.cohort_authority_path,
            cohort_authority_ref=acquisition.cohort_authority_ref,
        )
    assert client.calls == []
    assert not list((tmp_path / "continued").rglob("plan_approval.json"))


@pytest.mark.parametrize("changed_after_offer", [None, "input", "scope", "package", "package_same_stat", "package_touch"])
def test_web_preparation_continuation_never_materializes_again(grouped_source, tmp_path, monkeypatch, changed_after_offer):
    from easyicu.research_agent import pipeline as pipeline_module
    from easyicu.research_agent.orchestration import progressive_planning
    from easyicu.research_agent.acquisition import foundation
    from easyicu.research_agent.execution import runner as runner_module
    from easyicu.research_agent.planning.progressive_resume import progressive_resume_context_sha256
    from easyicu.research_agent.providers.mocks import MockLLMClient
    from easyicu.webserver import agent_pipeline_runs, agent_review_recovery, dataio
    from easyicu.webserver import provider_adapter, research_pipeline_run_preparation, study_contexts
    from easyicu.webserver.jobs import Job
    from tests.webserver.copilot.research_workflow_fixtures import complete_study

    source = grouped_source
    if not hasattr(source, "universe"):
        return
    study = complete_study()
    study.update(id=source.study["id"], question=source.context.research_question,
                 primary_exposure="", covariates=[], covariate_rationales={}, covariate_temporal_roles={})
    export = source.original.parent.parent / "export"
    study["data_source"] = {"path": str(export), "database": "miiv"}
    study["execution_concepts"] = {"outcome": "death"}
    source.study.clear()
    source.study.update(study)
    wrapper = source.run.parents[1]
    seed = agent_review_recovery.load_recovery_seed(wrapper)
    payload = seed.model_dump(exclude={"seed_sha256", "schema_version"})
    payload.update(study=study, scientific_configuration_sha256=study_contexts.scientific_configuration_sha256(study),
                   budget_mode="full_reviewed", prepared_package_binding=dataio.validate_research_pipeline_source(str(export), database="miiv")["binding"])
    agent_review_recovery.put_recovery_seed(agent_review_recovery.WebReviewRecoverySeed.create(**payload))
    monkeypatch.setattr(runner_module, "probe_runner_availability", lambda kind, **kw:
                        runner_module.RunnerAvailability(kind=kind, available=True, image="synthetic"))
    monkeypatch.setattr(pipeline_module.ResearchAgentPipeline, "_preflight_execution_runtime", lambda *a, **kw: ())
    from easyicu.research_agent.literature import LiteratureBundle

    monkeypatch.setattr(research_pipeline_run_preparation, "_development_resume_literature_bundle",
                        lambda **kw: LiteratureBundle(research_question=study["question"], citations=[]).model_dump(mode="json"))
    monkeypatch.setattr(research_pipeline_run_preparation, "_data_foundation_profile", lambda **kw: {
        "allowed_modules": ("labs", "outcomes"), "static_concepts": ("age",),
        "outcome_concepts": ("death",), "required_feature_concepts": ("lact",),
        "require_outcome": True, "primary_exposure_source_concept": None,
    })
    monkeypatch.setattr(research_pipeline_run_preparation, "_development_progressive_resume_binding", lambda **kw: (source.run / "progressive_planner_checkpoint_002.json", "a" * 64))
    monkeypatch.setattr(foundation, "acquire_universe_for_question", lambda **kw: pytest.fail("a continuation must not materialize again"))
    client = MockLLMClient()
    monkeypatch.setattr(provider_adapter, "build_research_agent_provider_client", lambda *a, **kw: (client, {"provider": "openai", "model": "synthetic"}))

    class PlannerReached(BaseException):
        pass

    def planner(**kwargs):
        from easyicu.research_agent.research_context.typed import parse_research_context_json

        restored = parse_research_context_json((kwargs["run_dir"] / "research_context.json").read_text())
        assert progressive_resume_context_sha256(restored) == progressive_resume_context_sha256(source.context)
        assert kwargs["resume_checkpoint_path"].parent == source.run
        raise PlannerReached

    monkeypatch.setattr(progressive_planning, "run_progressive_planner", planner)
    from unittest.mock import Mock
    from easyicu.webserver import research_prepared_resume

    validate_package = Mock(wraps=dataio.validate_research_pipeline_source)
    rehearse = Mock(wraps=research_prepared_resume._rehearse_grouped_continuation)
    monkeypatch.setattr(dataio, "validate_research_pipeline_source", validate_package)
    monkeypatch.setattr(research_prepared_resume, "_rehearse_grouped_continuation", rehearse)
    runner = agent_pipeline_runs.make_research_pipeline_run_runner(
        export_path=str(export), study_context=study, project_root=str(wrapper.parent.parent),
        provider={"provider": "openai", "external": True},
        provider_environment={"EASYICU_DISABLE_PROVIDER_ENV_FILE": "1"}, budget_mode="full_reviewed",
        development_resume_source_job_id="prior",
    )
    if changed_after_offer == "input":
        source.universe.write_bytes(b"changed after the offer")
    elif changed_after_offer == "package_same_stat":
        import os

        path = export / "labs.parquet"
        stat = path.stat()
        path.write_bytes(b"FAIL" + path.read_bytes()[4:])
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    elif changed_after_offer == "package_touch":
        import os

        path = export / "labs.parquet"
        stat = path.stat()
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
    elif changed_after_offer == "package":
        manifest = export / "_manifest.json"
        manifest.write_bytes(manifest.read_bytes() + b"\n")
        assert owner.development_resume_available(
            wrapper_dir=wrapper, study_id=study["id"],
            scientific_configuration_sha256=study_contexts.scientific_configuration_sha256(study),
        ) is False
    elif changed_after_offer == "scope":
        from dataclasses import replace

        original = source.scope()
        monkeypatch.setattr(owner, "_development_resume_launch_scope", lambda **kw: replace(original, prepared_grouping=None))
    if changed_after_offer in {None, "package_touch"}:
        with pytest.raises(PlannerReached):
            runner(Job("continued", "agent-run"))
        assert validate_package.call_count == 1
        assert rehearse.call_count == 1
    else:
        with pytest.raises(owner.ResearchPipelineRunError) as refused:
            runner(Job("continued", "agent-run"))
        assert refused.value.code in {owner.DEVELOPMENT_RESUME_PHASE_NOT_REPLAYED, "research_pipeline_development_resume_scope_mismatch"}
    assert client.calls == []


def test_reacquisition_adds_an_incompatible_trajectory_but_continuation_reuses_the_source(grouped_source, tmp_path):
    from easyicu.research_agent.cohort import materializer
    from easyicu.research_agent.intake.materialized_metadata import stage_materialized_cohort_authority
    from easyicu.research_agent.intake.materialized_trajectory import (
        MaterializedTrajectoryError, StagedTrajectoryBinding, stage_materialized_trajectory_authority,
    )

    source = grouped_source
    if not hasattr(source, "universe"):
        return
    paths = materializer.materialize_to_parquet(
        tmp_path / "reacquired", stem="universe", data_path=source.original.parent.parent / "export",
        database="miiv", static_concepts=("age",), feature_concepts=("lact",), outcome_concepts=("death",),
        emit_trajectory=True, trajectory_concepts=("lact",), trajectory_window=(0.0, 24.0),
    )
    target = tmp_path / "wrong-input/cohort.parquet"
    staged = stage_materialized_cohort_authority(paths["parquet"], target, producer_implementation_sha256="a" * 64)
    trajectory_path = target.with_name("cohort_trajectory.parquet")
    trajectory = stage_materialized_trajectory_authority(
        paths["trajectory"], trajectory_path, source_universe_path=paths["parquet"], target_universe_path=target,
        expected_source_authority=None,
        expected_target_universe_authority=staged.reference, producer_implementation_sha256="b" * 64,
    )
    binding = StagedTrajectoryBinding(
        path=trajectory_path, sha256=trajectory.authority.trajectory_sha256,
        size=trajectory.authority.trajectory_size, authority_ref=trajectory.reference,
    )

    def rebuild(path, **kwargs):
        return _context(path, trajectory_binding=binding, **kwargs)

    with pytest.raises(MaterializedTrajectoryError, match="different universe authority"):
        grouping.restore_exposure_grouping_phase(
            source_run_dir=source.run, context=rebuild(target), cohort_path=target,
            run_dir=target.parent, rebuild_context=rebuild,
        )
    continuation = source.scope().prepared_grouping
    assert continuation.acquisition().universe_path == source.universe
    assert continuation.acquisition().trajectory_path is None


def test_restore_keeps_record_bytes_columns_labels_and_context(grouped_source, tmp_path):
    source = grouped_source
    target = tmp_path / "continued"
    shutil.copytree(source.original.parent, target)
    cohort = target / "cohort.parquet"
    restored = grouping.restore_exposure_grouping_phase(
        source_run_dir=source.run, context=_context(cohort), cohort_path=cohort,
        run_dir=target, rebuild_context=_context,
    )
    assert (target / "exposure_groupings.json").read_bytes() == (
        source.run / "exposure_groupings.json"
    ).read_bytes()
    assert hashlib.sha256(cohort.read_bytes()).hexdigest() == hashlib.sha256(
        source.cohort.read_bytes()
    ).hexdigest()
    from easyicu.research_agent.planning.progressive_resume import progressive_resume_context_sha256

    assert progressive_resume_context_sha256(restored.context) == (
        progressive_resume_context_sha256(source.context)
    )


@pytest.mark.parametrize("target", ["record", "sealed_record", "sealed_context", "missing", "not_asked", "symlink", "whitespace"])
def test_tampered_grouping_authority_is_refused(grouped_source, target):
    source = grouped_source
    path = {
        "record": source.run / "exposure_groupings.json",
        "sealed_record": source.run / "evidence/exposure_groupings__exposure_groupings.json",
        "sealed_context": source.run / "evidence/research_context__research_context.json",
        "missing": source.run / "exposure_groupings.json",
        "not_asked": source.run / "exposure_groupings.json",
        "symlink": source.run / "exposure_groupings.json",
        "whitespace": source.run / "exposure_groupings.json",
    }[target]
    if target == "whitespace":
        path.write_bytes(path.read_bytes() + b"\n")
    elif target in {"missing", "symlink"}:
        path.unlink()
        if target == "symlink":
            path.symlink_to(source.run / "evidence/exposure_groupings__exposure_groupings.json")
    elif target == "not_asked":
        path.write_text(json.dumps({
            "schema_version": grouping.EXPOSURE_GROUPINGS_RECORD_SCHEMA,
            "not_asked": "no_value_to_group",
        }))
    else:
        path.write_text(json.dumps({"tampered": True}))
    with pytest.raises(owner.ResearchPipelineRunError) as refused:
        source.scope()
    assert refused.value.code == owner.DEVELOPMENT_RESUME_PHASE_NOT_REPLAYED


def test_rebuilt_context_drift_is_refused_before_planning(grouped_source, tmp_path):
    source = grouped_source
    target = tmp_path / "continued"
    shutil.copytree(source.original.parent, target)
    cohort = target / "cohort.parquet"

    def changed_context(*args, **kwargs):
        return _context(*args, **kwargs).model_copy(update={"notes": "changed scope"})

    with pytest.raises(ValueError, match="context"):
        grouping.restore_exposure_grouping_phase(
            source_run_dir=source.run, context=_context(cohort), cohort_path=cohort,
            run_dir=target, rebuild_context=changed_context,
        )


def test_changed_grouping_input_is_refused_before_staging(grouped_source, tmp_path):
    source = grouped_source
    target = tmp_path / "continued"
    shutil.copytree(source.original.parent, target)
    cohort = target / "cohort.parquet"
    context = _context(cohort)
    context = context.model_copy(update={
        "variables": [item for item in context.variables if item.source_concept != "lact" and item.name != "lact"],
    })
    with pytest.raises(ValueError, match="does not compile"):
        grouping.restore_exposure_grouping_phase(
            source_run_dir=source.run, context=context, cohort_path=cohort,
            run_dir=target, rebuild_context=_context,
        )
    assert not (target / "exposure_groupings.json").exists()


@pytest.mark.parametrize("tampered", [False, True])
def test_offer_uses_launch_verdict_for_the_same_grouped_source(grouped_source, tampered):
    from easyicu.webserver import study_contexts
    from easyicu.webserver.pi_copilot.run_authority import _development_planner_checkpoint_available

    source = grouped_source
    if tampered:
        (source.run / "exposure_groupings.json").write_text("{}")
    row = {
        "project_dir": str(source.run.parents[1]), "study_id": source.study["id"],
        "scientific_configuration_sha256": study_contexts.scientific_configuration_sha256(source.study),
        "gate_reason": "research_pipeline_planner_provider_unavailable",
    }
    try:
        source.scope()
    except owner.ResearchPipelineRunError:
        accepted = False
    else:
        accepted = True
    assert accepted is not tampered
    owner.development_resume_available(
        wrapper_dir=source.run.parents[1], study_id=source.study["id"],
        scientific_configuration_sha256=row["scientific_configuration_sha256"], verify_now=True,
    )
    assert _development_planner_checkpoint_available(row) is accepted


def test_offer_calls_the_launch_owner_instead_of_reconstructing_it(grouped_source, monkeypatch):
    from easyicu.webserver import study_contexts

    calls = []

    def refuse(**kwargs):
        calls.append(kwargs)
        raise owner.ResearchPipelineRunError("research_pipeline_development_resume_scope_mismatch", "changed")

    monkeypatch.setattr(owner, "_development_resume_launch_scope", refuse)
    source = grouped_source
    assert owner.development_resume_available(
        wrapper_dir=source.run.parents[1], study_id=source.study["id"],
        scientific_configuration_sha256=study_contexts.scientific_configuration_sha256(source.study),
        verify_now=True,
    ) is False
    assert calls == [{
        "project_root": str(source.run.parents[3]), "study": source.study, "source_job_id": "prior",
    }]


def test_planner_continues_from_the_grouped_context_checkpoint(grouped_source, tmp_path):
    from easyicu.research_agent.agents.progressive_planner import ProgressivePlannerAgent
    from easyicu.research_agent.orchestration.progressive_planning import run_progressive_planner
    from easyicu.research_agent.planning.preplan_know_how import PlannerKnowHowBinding
    from easyicu.research_agent.planning.progressive_artifacts import persist_progressive_planner_checkpoint
    from tests.research_agent.planning import progressive_planner_fixtures as fixtures

    # The existing synthetic plan has a primary binary exposure; this study also
    # formed a secondary measured-value grouping. Both remain in its context.
    def plan_context(grouped):
        base = fixtures._context()
        return base.model_copy(update={
            "variables": [*base.variables, *grouped.variables],
            "cohort": base.cohort.model_copy(update={"provenance": grouped.cohort.provenance}),
        })

    source = grouped_source
    materializations = fixtures._materialization_payloads()
    client = ScriptedMockLLMClient([
        json.dumps(fixtures._outline_payload()), json.dumps(fixtures._foundation_payload()),
        *[json.dumps(item) for item in materializations],
    ])
    client.supports_strict_json_schema = True
    checkpoints = []
    ProgressivePlannerAgent(client).run(
        plan_context(source.context), checkpoint_callback=checkpoints.append,
        resume_dependency_context={
            "cohort_file_sha256": hashlib.sha256(source.cohort.read_bytes()).hexdigest(),
            "llm_signature": "mock:test", "prompt_version": "test-v1",
        },
    )
    # Replace only the fixture's placeholder checkpoint with a real chain.
    (source.run / "progressive_planner_checkpoint_002.json").unlink()
    evidence = EvidenceStore(root=source.run)
    paths = [persist_progressive_planner_checkpoint(
        run_dir=source.run, evidence=evidence, checkpoint=item, prompt_pack_version="test-v1",
    ) for item in checkpoints[:5]]
    target = tmp_path / "continued"
    shutil.copytree(source.original.parent, target)
    cohort = target / "cohort.parquet"
    restored = grouping.restore_exposure_grouping_phase(
        source_run_dir=source.run, context=_context(cohort), cohort_path=cohort,
        run_dir=target, rebuild_context=_context,
    )
    resumed_client = ScriptedMockLLMClient([json.dumps(item) for item in materializations[3:]])
    resumed_client.supports_strict_json_schema = True
    result = run_progressive_planner(
        planner=ProgressivePlannerAgent(resumed_client), context=plan_context(restored.context),
        run_dir=target, evidence=EvidenceStore(root=target), prompt_pack_version="test-v1",
        resume_checkpoint_path=paths[-1],
        resume_checkpoint_sha256=hashlib.sha256(paths[-1].read_bytes()).hexdigest(),
        cohort_path=cohort, llm_signature="mock:test", planner_kwargs={},
        know_how_binding=PlannerKnowHowBinding(), planning_contract_context="",
        finding_sink=lambda _: None,
    )
    assert result.generation_mode == "llm_progressive_v2_dev_resume"
    assert result.facts.resume_validated is True
    assert len(resumed_client.calls) == 4
    assert len(result.plan.steps) == 7
    assert not (target / "plan_approval.json").exists()


def test_pipeline_restores_grouping_before_any_planner_or_analysis_call(tmp_path, monkeypatch):
    from easyicu.research_agent import pipeline as pipeline_module
    from easyicu.research_agent.providers.mocks import MockLLMClient
    from easyicu.webserver.agent_pipeline_runs import _pipeline_failure_code

    calls = []

    def refuse(**kwargs):
        calls.append(kwargs)
        raise ValueError("restored context mismatch")

    monkeypatch.setattr(pipeline_module, "restore_exposure_grouping_phase", refuse)
    client = MockLLMClient()
    checkpoint = tmp_path / "source/progressive_planner_checkpoint_004.json"
    pipeline = pipeline_module.ResearchAgentPipeline(
        workdir=tmp_path / "pipeline", llm=client, planner_strategy="progressive_v2",
        development_diagnostic=True, require_human_plan_review=True,
        development_progressive_resume_checkpoint_path=checkpoint,
        development_progressive_resume_checkpoint_sha256="a" * 64,
        enable_exposure_grouping=None, enable_visual_qa=False,
        enable_publication_figure_skill=False, enable_nature_writing_skill=False,
    )
    _prepared(tmp_path / "input")
    with pytest.raises(Exception) as refused:
        pipeline.run(
            question="Describe lactate categories", cohort=tmp_path / "input/materialized/universe.parquet",
            target_outcome="death",
        )
    assert calls and calls[0]["source_run_dir"] == checkpoint.parent, repr(refused.value)
    assert client.calls == []
    assert _pipeline_failure_code(refused.value) == owner.DEVELOPMENT_RESUME_PHASE_NOT_REPLAYED


def test_continuation_retains_reviewed_requirements_without_approval(source):
    from easyicu.webserver import agent_review_recovery as recovery
    from easyicu.research_agent.planning.baseline_requirements import AcceptedBaselineRequirements

    path = source.run.parents[1] / ".runtime/web_review_recovery_seed.json"
    seed = recovery.load_recovery_seed(source.run.parents[1])
    raw = seed.model_dump(mode="json")
    raw["pipeline_config"].update({
        "bound_baseline_requirements": AcceptedBaselineRequirements.model_validate({
            "source_plan_sha256": "b" * 64, "tables": [{
                "source_step_id": "baseline", "group_by": {"name": "lact_group_x1"},
                "variables": [{"name": "age"}],
            }],
        }).model_dump(mode="json"),
        "required_primary_cohort_selection_mode": "all_input_rows",
        "require_human_plan_review": True,
    })
    raw["seed_sha256"] = recovery.canonical_sha256({k: v for k, v in raw.items() if k != "seed_sha256"})
    path.write_text(json.dumps(raw))
    scope = source.scope()
    assert scope.planning_requirements == {
        "bound_baseline_requirements": raw["pipeline_config"]["bound_baseline_requirements"],
        "required_primary_cohort_selection_mode": "all_input_rows",
    }
    assert owner.development_resume_requirement(scope, "bound_baseline_requirements", None) == scope.planning_requirements["bound_baseline_requirements"]
    with pytest.raises(owner.ResearchPipelineRunError) as refused:
        owner.development_resume_requirement(scope, "bound_baseline_requirements", {"new": "plan"})
    assert refused.value.code == "research_pipeline_development_resume_plan_contract_mismatch"


@pytest.mark.parametrize("field", [
    "bound_baseline_requirements", "bound_population_requirements",
    "bound_analysis_inputs", "required_primary_cohort_selection_mode",
])
def test_untyped_reviewed_requirements_cannot_offer_resume(source, field):
    from easyicu.webserver import agent_review_recovery as recovery
    from easyicu.webserver import study_contexts

    wrapper = source.run.parents[1]
    raw = recovery.load_recovery_seed(wrapper).model_dump(mode="json")
    raw["pipeline_config"][field] = "invalid requirement"
    raw["seed_sha256"] = recovery.canonical_sha256({k: v for k, v in raw.items() if k != "seed_sha256"})
    (wrapper / ".runtime/web_review_recovery_seed.json").write_text(json.dumps(raw))
    with pytest.raises(owner.ResearchPipelineRunError) as refused:
        source.scope()
    assert refused.value.code == "research_pipeline_development_resume_scope_invalid"
    assert owner.development_resume_available(
        wrapper_dir=wrapper, study_id=source.study["id"],
        scientific_configuration_sha256=study_contexts.scientific_configuration_sha256(source.study),
        verify_now=True,
    ) is False


def test_changed_implementation_restores_original_authority_and_records_verifier(grouped_source, tmp_path, monkeypatch):
    source = grouped_source
    target = tmp_path / "continued"
    cohort = target / "cohort.parquet"
    materialized = source.context.model_dump(mode="json").get("materialized_inputs")
    if materialized:
        # The pipeline's initial copy also binds its implementation. Recreate
        # that copy with a different identity, as happens after a code repair.
        from easyicu.research_agent.intake.materialized_metadata import stage_materialized_cohort_authority

        stage_materialized_cohort_authority(
            source.original.parent.parent / "materialized/universe.parquet", cohort,
            producer_implementation_sha256="f" * 64,
        )
    else:
        shutil.copytree(source.original.parent, target)
    monkeypatch.setattr(grouping, "_grouping_implementation_sha256", lambda: "0" * 64)
    assert source.scope().budget_mode == "planner_canary"
    restored = grouping.restore_exposure_grouping_phase(
        source_run_dir=source.run, context=_context(cohort), cohort_path=cohort,
        run_dir=target, rebuild_context=_context,
    )
    from easyicu.research_agent.planning.progressive_resume import progressive_resume_context_sha256

    assert progressive_resume_context_sha256(restored.context) == progressive_resume_context_sha256(source.context)
    if materialized:
        reference = materialized["cohort"]["authority_ref"]
        assert (target / reference["file"]).read_bytes() == (source.run / reference["file"]).read_bytes()
        receipt = json.loads((target / "exposure_grouping_resume.json").read_text())
        assert receipt == {
            "source_authority_sha256": reference["sha256"],
            "groupings_record_sha256": hashlib.sha256((source.run / "exposure_groupings.json").read_bytes()).hexdigest(),
            "verified_by_implementation_sha256": "0" * 64,
        }
        evidence = EvidenceStore(root=target)
        grouping.register_exposure_groupings_record(evidence, target)
        assert evidence.get("exposure_grouping_resume") is not None


def test_source_input_seals_the_exact_grouping_record(grouped_source):
    source = grouped_source
    materialized = source.context.model_dump(mode="json").get("materialized_inputs")
    if not materialized:
        return
    path = source.run / materialized["cohort"]["authority_ref"]["file"]
    # Valid JSON with identical semantics still changes the sealed file bytes.
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="does not seal"):
        grouping.load_resume_exposure_groupings(source.run)


def test_materialization_drift_cannot_restore_original_authority(grouped_source, tmp_path, monkeypatch):
    from types import SimpleNamespace

    source = grouped_source
    if not source.context.model_dump(mode="json").get("materialized_inputs"):
        return
    target = tmp_path / "continued"
    shutil.copytree(source.original.parent, target)
    cohort = target / "cohort.parquet"
    verified = grouping.load_verified_materialized_cohort_authority

    def changed_authority(*args, **kwargs):
        result = verified(*args, **kwargs)
        observed = result.authority.to_dict()
        observed["cohort_rows"] += 1
        return SimpleNamespace(authority=SimpleNamespace(to_dict=lambda: observed))

    monkeypatch.setattr(grouping, "load_verified_materialized_cohort_authority", changed_authority)
    with pytest.raises(ValueError, match="materialization differs"):
        grouping.restore_exposure_grouping_phase(
            source_run_dir=source.run, context=_context(cohort), cohort_path=cohort,
            run_dir=target, rebuild_context=_context,
        )
    assert not (target / "exposure_grouping_resume.json").exists()
