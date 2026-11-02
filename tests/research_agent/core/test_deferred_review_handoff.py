"""Live and reconstructed review handoffs use the same verified deferred input."""
from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from easyicu.research_agent.orchestration.services import DeferredFirstObservedInput
from easyicu.research_agent.orchestration.human_review_checkpoint import load_checkpoint
from easyicu.research_agent.orchestration.workflow import HumanReviewDecision
from easyicu.research_agent.pipeline import ResearchAgentPipeline
from easyicu.research_agent.research_context.typed import parse_research_context
from tests.support.durable_human_review import _cohort, _force_approvable_plan_review, _pipeline


def _hash(data):
    return hashlib.sha256(data).hexdigest()


class _SyntheticReviewedTask:
    def resume(self):
        pass
    def assert_active(self):
        return 1800.0
    def cap_timeout(self, seconds):
        return min(seconds, 1800.0)
    def release_deferred_review_transport(self, *, decision_sha256, receipt_sha256):
        assert len(decision_sha256) == len(receipt_sha256) == 64


@pytest.mark.parametrize('restore', [False, True])
@pytest.mark.parametrize('tamper', ['none', 'deleted', 'changed', 'replay'])
def test_real_review_checkpoint_binds_deferred_input_after_decision(monkeypatch, tmp_path, restore, tamper):
    _force_approvable_plan_review(monkeypatch)
    from easyicu.research_agent.orchestration import workflow as workflow_module
    original_requests = workflow_module.human_review_requests_for_plan
    registered = [False]
    def register_recipe(**kwargs):
        evidence = kwargs['evidence']
        recipe = Path(evidence.root) / 'first_observed_deferred_recipe.json'
        if not registered[0]:
            recipe.write_text('{}')
            evidence.register_file(kind='log', description='Synthetic deferred recipe',
                                   source_path=recipe, evidence_id='first_observed_deferred_recipe')
            registered[0] = True
        return original_requests(**kwargs)
    monkeypatch.setattr(workflow_module, 'human_review_requests_for_plan', register_recipe)
    workdir = tmp_path / 'runs'
    original = _pipeline(workdir)
    consumer_holder = {}
    original._services = replace(original._services, deferred_first_observed_consumer=lambda **kwargs: consumer_holder['consumer'](**kwargs))
    pending = original.run(question='Does age describe hospital mortality?', cohort=_cohort(), target_outcome='death')
    run_dir = Path(pending.run_dir)
    checkpoint_file = run_dir / 'human_review_checkpoint.json'
    original_checkpoint = load_checkpoint(checkpoint_file)
    metadata_context = dict(original_checkpoint.plan_handoff['context'])
    calls = []
    def consumer(*, run_dir, approved_plan_sha256, decision_set_sha256, evidence_store):
        checkpoint = load_checkpoint(checkpoint_file, require_pending=False)
        assert checkpoint.state == 'approved_pending_execution'
        assert checkpoint.consumed_decision_sha256 == decision_set_sha256
        calls.append(1)
        dest = run_dir / 'deferred_input'
        if dest.exists():
            receipt_bytes = (run_dir / 'deferred_first_observed_receipt.json').read_bytes()
            capsule_bytes = (dest / 'run_input_capsule.json').read_bytes()
            return DeferredFirstObservedInput(dest / 'cohort.parquet', parse_research_context(metadata_context),
                                              _hash(capsule_bytes), _hash(receipt_bytes))
        dest.mkdir()
        cohort = dest / 'cohort.parquet'
        cohort.write_bytes(b'synthetic cohort')
        capsule = b'{"synthetic":true}'
        (dest / 'run_input_capsule.json').write_bytes(capsule)
        context_bytes = json.dumps(metadata_context, sort_keys=True).encode()
        (dest / 'research_context.json').write_bytes(context_bytes)
        body = {'schema_version': 'easyicu.first-observed-deferred-input-receipt/1',
                'approved_plan_sha256': approved_plan_sha256, 'decision_set_sha256': decision_set_sha256,
                'recipe_sha256': _hash((run_dir / 'first_observed_deferred_recipe.json').read_bytes()), 'capsule_sha256': _hash(capsule),
                'context_sha256': _hash(context_bytes)}
        receipt = json.dumps({'body': body, 'signature': 'synthetic'}).encode()
        (run_dir / 'deferred_first_observed_receipt.json').write_bytes(receipt)
        return DeferredFirstObservedInput(cohort, parse_research_context(metadata_context), _hash(capsule), _hash(receipt))
    seen = []
    def execute(self, **kwargs):
        seen.append(('execute', kwargs['cohort_path'], kwargs['plan_result'].context))
        return SimpleNamespace(plan=kwargs['plan_result'].plan)
    def write(self, **kwargs):
        return SimpleNamespace()
    def finalise(self, **kwargs):
        seen.append(('finalise', kwargs['cohort_path'], kwargs['plan_result'].context))
        return {'done': True}
    monkeypatch.setattr(ResearchAgentPipeline, '_run_execute_phase', execute)
    monkeypatch.setattr(ResearchAgentPipeline, '_run_write_phase', write)
    monkeypatch.setattr(ResearchAgentPipeline, '_finalise_success', finalise)
    consumer_holder['consumer'] = consumer
    target = _pipeline(workdir) if restore else original
    target._services = replace(target._services, deferred_first_observed_consumer=consumer)
    target._provider_hard_stop = _SyntheticReviewedTask()
    decisions = [HumanReviewDecision(review_id=r.review_id, authority_sha256=r.authority_sha256,
                 decision='approved', reviewer='synthetic', decided_at='2026-08-14T03:12:00Z') for r in pending.requests]
    assert calls == []
    recipe = run_dir / 'first_observed_deferred_recipe.json'
    if tamper == 'deleted':
        recipe.unlink()
    elif tamper == 'changed':
        recipe.write_text('{"changed":true}')
    if tamper in {'deleted', 'changed'}:
        with pytest.raises((ValueError, RuntimeError)):
            target.resume_human_review(decisions, run_id=pending.run_id)
        assert calls == [] and seen == []
        return
    if tamper == 'replay':
        from easyicu.research_agent import pipeline as pipeline_module
        from easyicu.research_agent.orchestration import human_review_restore as restore_module
        original_mark = pipeline_module.mark_human_review_execution_started
        def interrupted(_path):
            raise SystemExit('synthetic crash after durable receipt')
        monkeypatch.setattr(pipeline_module, 'mark_human_review_execution_started', interrupted)
        monkeypatch.setattr(restore_module, 'mark_human_review_execution_started', interrupted)
        with pytest.raises(SystemExit):
            target.resume_human_review(decisions, run_id=pending.run_id)
        assert calls == [1] and seen == []
        monkeypatch.setattr(pipeline_module, 'mark_human_review_execution_started', original_mark)
        monkeypatch.setattr(restore_module, 'mark_human_review_execution_started', original_mark)
        target = _pipeline(workdir)
        target._services = replace(target._services, deferred_first_observed_consumer=consumer)
        target._provider_hard_stop = _SyntheticReviewedTask()
    assert target.resume_human_review(decisions, run_id=pending.run_id) == {'done': True}
    assert calls == ([1, 1] if tamper == 'replay' else [1])
    assert [entry[0] for entry in seen] == ['execute', 'finalise']
    assert seen[0][1] == seen[1][1] == run_dir / 'deferred_input' / 'cohort.parquet'
    assert seen[0][2] is seen[1][2]
    assert load_checkpoint(checkpoint_file, require_pending=False).state == 'completed'


def test_rejected_review_never_invokes_deferred_consumer(monkeypatch, tmp_path):
    from easyicu.research_agent.orchestration.workflow import HumanReviewRejected
    _force_approvable_plan_review(monkeypatch)
    pipeline = _pipeline(tmp_path / 'runs')
    pending = pipeline.run(question='Does age describe hospital mortality?', cohort=_cohort(), target_outcome='death')
    (Path(pending.run_dir) / 'first_observed_deferred_recipe.json').write_text('{}')
    pipeline._services = replace(pipeline._services, deferred_first_observed_consumer=lambda **_: pytest.fail('read before approval'))
    decisions = [HumanReviewDecision(review_id=r.review_id, authority_sha256=r.authority_sha256,
                 decision='rejected', reviewer='synthetic', decided_at='2026-08-14T03:12:00Z') for r in pending.requests]
    with pytest.raises(HumanReviewRejected):
        pipeline.resume_human_review(decisions, run_id=pending.run_id)


@pytest.mark.parametrize('tamper', ['recipe', 'receipt', 'capsule', 'context', 'decision', 'plan', 'cohort_path'])
def test_core_refuses_deferred_handoff_tampering_before_execution(monkeypatch, tmp_path, tamper):
    from easyicu.research_agent.orchestration import services as owner
    recipe = b'approved recipe'
    (tmp_path / owner.RECIPE_NAME).write_bytes(recipe)
    root = tmp_path / 'deferred_input'
    root.mkdir()
    (root / 'cohort.parquet').write_bytes(b'synthetic')
    capsule = b'{"cohort":"synthetic"}'
    context_bytes = b'{"synthetic":true}'
    (root / 'run_input_capsule.json').write_bytes(capsule)
    (root / 'research_context.json').write_bytes(context_bytes)
    plan = 'a' * 64
    decision = 'b' * 64
    recipe_hash = _hash(recipe)
    context = SimpleNamespace(model_dump=lambda **_: {'synthetic': True})
    monkeypatch.setattr(owner, 'parse_research_context_json', lambda _: context)
    body = {'schema_version': 'easyicu.first-observed-deferred-input-receipt/1',
            'approved_plan_sha256': plan, 'decision_set_sha256': decision,
            'recipe_sha256': recipe_hash, 'capsule_sha256': _hash(capsule), 'context_sha256': _hash(context_bytes)}
    receipt = json.dumps({'body': body, 'signature': 'synthetic'}).encode()
    (tmp_path / owner.RECEIPT_NAME).write_bytes(receipt)
    calls = []
    def consumer(**kwargs):
        calls.append(1)
        if tamper in {'receipt', 'capsule', 'context'}:
            path = {'receipt': tmp_path / owner.RECEIPT_NAME, 'capsule': root / 'run_input_capsule.json',
                    'context': root / 'research_context.json'}[tamper]
            path.write_bytes(b'tampered')
        return owner.DeferredFirstObservedInput(root / ('other.parquet' if tamper == 'cohort_path' else 'cohort.parquet'),
                                                context, _hash(capsule), _hash(receipt))
    if tamper == 'recipe':
        (tmp_path / owner.RECIPE_NAME).write_bytes(b'tampered')
    with pytest.raises((ValueError, RuntimeError)):
        owner.bind_deferred_first_observed(run_dir=tmp_path, plan_result=SimpleNamespace(evidence=object()),
            approved_plan_sha256=('c' * 64 if tamper == 'plan' else plan),
            decision_set_sha256=('c' * 64 if tamper == 'decision' else decision),
            recipe_sha256=recipe_hash, consumer=consumer)
    assert len(calls) == (0 if tamper == 'recipe' else 1)


def test_recipe_without_review_authority_cannot_fall_back(tmp_path):
    from easyicu.research_agent.orchestration.services import bind_deferred_first_observed
    (tmp_path / 'first_observed_deferred_recipe.json').write_text('{}')
    with pytest.raises(ValueError, match='recipe_authority_missing'):
        bind_deferred_first_observed(run_dir=tmp_path, plan_result=object(),
            approved_plan_sha256='', decision_set_sha256='', recipe_sha256='',
            consumer=lambda **_: pytest.fail('unapproved read'))


def test_real_execute_step_capsule_and_finalize_bind_actual_input(monkeypatch, tmp_path):
    from easyicu.research_agent.orchestration import workflow as workflow_module
    from easyicu.research_agent.providers import mocks as mock_provider
    from easyicu.research_agent.reporting import write_phase
    # This test verifies execute/finalise input binding, not the separately
    # covered signed late source-flow publication boundary.
    monkeypatch.setattr(write_phase, '_append_deferred_source_flow', lambda pipeline, **kwargs: kwargs['binding'])
    _force_approvable_plan_review(monkeypatch)
    original_coder = mock_provider._mock_code_for_step
    def coder(ctx, prompt):
        code = original_coder(ctx, prompt)
        if mock_provider._extract_step_id(prompt) == '03_cohort_denominator':
            code += "\n" + "\n".join((
                "pd.DataFrame([{'cohort': 'analysis', 'n': len(df)}]).to_csv(out_dir / 'cohort_denominator.csv', index=False)",
                "summary = json.loads((out_dir / 'step_summary.json').read_text())",
                "summary['output_files']['table:cohort_denominator'] = 'cohort_denominator.csv'",
                "(out_dir / 'step_summary.json').write_text(json.dumps(summary))",
            )) + "\n"
        return code
    monkeypatch.setattr(mock_provider, '_mock_code_for_step', coder)
    original_requests = workflow_module.human_review_requests_for_plan
    registered = [False]
    def with_recipe(**kwargs):
        evidence = kwargs['evidence']
        recipe = Path(evidence.root) / 'first_observed_deferred_recipe.json'
        if not registered[0]:
            recipe.write_bytes(b'{}')
            evidence.register_file(kind='log', description='synthetic recipe', source_path=recipe,
                                   evidence_id='first_observed_deferred_recipe')
            registered[0] = True
        return original_requests(**kwargs)
    monkeypatch.setattr(workflow_module, 'human_review_requests_for_plan', with_recipe)
    pipeline = _pipeline(tmp_path / 'runs')
    consumer_holder = {}
    pipeline._services = replace(pipeline._services, deferred_first_observed_consumer=lambda **kwargs: consumer_holder['consumer'](**kwargs))
    pending = pipeline.run(question='Does age describe hospital mortality?', cohort=_cohort(), target_outcome='death')
    root = Path(pending.run_dir)
    checkpoint = load_checkpoint(root / 'human_review_checkpoint.json')
    original_capsule = (root / 'run_input_capsule.json').read_bytes()
    context = checkpoint.plan_handoff['context']
    calls = []
    def consumer(*, run_dir, approved_plan_sha256, decision_set_sha256, evidence_store):
        import shutil
        calls.append(1)
        dest = run_dir / 'deferred_input'
        dest.mkdir()
        # Synthetic data already staged by this test's ordinary pipeline.
        original_cohort = run_dir / json.loads(original_capsule)['cohort_relative_path']
        shutil.copyfile(original_cohort, dest / 'cohort.parquet')
        capsule = b'{"synthetic_actual_capsule":true}'
        (dest / 'run_input_capsule.json').write_bytes(capsule)
        context_bytes = json.dumps(context, sort_keys=True).encode()
        (dest / 'research_context.json').write_bytes(context_bytes)
        body = {'schema_version': 'easyicu.first-observed-deferred-input-receipt/1',
            'approved_plan_sha256': approved_plan_sha256, 'decision_set_sha256': decision_set_sha256,
            'recipe_sha256': _hash((run_dir / 'first_observed_deferred_recipe.json').read_bytes()),
            'capsule_sha256': _hash(capsule), 'context_sha256': _hash(context_bytes)}
        receipt = json.dumps({'body': body, 'signature': 'synthetic'}).encode()
        (run_dir / 'deferred_first_observed_receipt.json').write_bytes(receipt)
        return DeferredFirstObservedInput(dest / 'cohort.parquet', parse_research_context(context), _hash(capsule), _hash(receipt))
    consumer_holder['consumer'] = consumer
    pipeline._services = replace(pipeline._services, deferred_first_observed_consumer=consumer)
    pipeline._provider_hard_stop = _SyntheticReviewedTask()
    decisions = [HumanReviewDecision(review_id=r.review_id, authority_sha256=r.authority_sha256,
                 decision='approved', reviewer='synthetic', decided_at='2026-08-14T03:12:00Z') for r in pending.requests]
    result = pipeline.resume_human_review(decisions, run_id=pending.run_id)
    assert result.run_id == pending.run_id and calls == [1]
    assert (root / 'run_input_capsule.json').read_bytes() == original_capsule
    actual_hash = _hash((root / 'deferred_input' / 'run_input_capsule.json').read_bytes())
    manifest = json.loads((root / 'manifest.json').read_text())
    assert manifest['context_path'] == 'deferred_input/research_context.json'
    records = manifest['per_step_records']
    assert records and all(r['status'] == 'ok' for r in records)
    assert all(r['run_input_capsule_sha256'] == actual_hash for r in records if r.get('run_input_capsule_sha256'))
    assert any(r.get('run_input_capsule_sha256') == actual_hash for r in records)
    assert pipeline._pending_human_review is None
    from easyicu.research_agent.authority.evidence_store import EvidenceStore
    from easyicu.research_agent.authority.run_input import RunInputIdentityError
    from easyicu.research_agent.execution.phase_support import _verified_run_input_capsule_digest
    evidence = EvidenceStore(root)
    assert _verified_run_input_capsule_digest(run_dir=root, evidence_store=evidence) == actual_hash
    actual = root / 'deferred_input' / 'run_input_capsule.json'
    saved = actual.read_bytes()
    actual.write_bytes(b'tampered')
    with pytest.raises(RunInputIdentityError):
        _verified_run_input_capsule_digest(run_dir=root, evidence_store=evidence)
    actual.write_bytes(saved)
    (root / 'first_observed_deferred_recipe.json').unlink()
    with pytest.raises(RunInputIdentityError):
        _verified_run_input_capsule_digest(run_dir=root, evidence_store=evidence)


def test_host_stage_compiles_without_planner_and_reaches_normalized_lineage(monkeypatch, tmp_path):
    from contextlib import nullcontext
    from easyicu.research_agent.agents.progressive_planner import ProgressivePlannerAgent
    from easyicu.research_agent.agents.planner import PlannerAgent
    _force_approvable_plan_review(monkeypatch)
    original = _pipeline(tmp_path / 'original')
    original.run(question='Does age describe hospital mortality?', cohort=_cohort(), target_outcome='death')
    saved_plan = original._pending_human_review['workflow']._plan_result.plan.model_copy(deep=True)
    seen = []
    class StopAfterLineage(Exception):
        pass
    class SyntheticHostStage:
        def normalize_review_interval_text(self, *, context, plan):
            return plan
        def reuse_preplan_literature(self, *, context, run_dir, evidence):
            return None
        def compile_plan(self, *, context, allowed_literature_citation_keys):
            seen.append('compile')
            return saved_plan.model_copy(deep=True)
        def persist_replayed_progressive_evidence(self, *, context, run_dir, evidence, prompt_pack_version):
            return None
        def prepare_review(self, *, context, agent_context, normalized_lineage, run_dir, evidence):
            seen.append('lineage')
            assert normalized_lineage.proposed.plan_sha256
            assert normalized_lineage.plan_sha256
            assert evidence.get(f'plan_lifecycle_revision_{normalized_lineage.analysis_plan().revision}') is not None
            raise StopAfterLineage
        def review_scope(self, *, context):
            return nullcontext()
    stage = _pipeline(tmp_path / 'stage')
    stage._services = replace(stage._services, deferred_first_observed_stage=SyntheticHostStage())
    monkeypatch.setattr(ProgressivePlannerAgent, 'run', lambda *a, **k: pytest.fail('planner call'))
    monkeypatch.setattr(PlannerAgent, 'run', lambda *a, **k: pytest.fail('planner call'))
    with pytest.raises(StopAfterLineage):
        stage.run(question='Does age describe hospital mortality?', cohort=_cohort(), target_outcome='death')
    assert seen == ['compile', 'lineage']


def test_transport_gate_never_releases_without_durable_execution_checkpoint(monkeypatch, tmp_path):
    from easyicu.research_agent.orchestration import human_review_checkpoint, services
    called = []
    monkeypatch.setattr(human_review_checkpoint, 'load_checkpoint', lambda *a, **k:
                        SimpleNamespace(state='approved_pending_execution', consumed_decision_sha256='a' * 64))
    task = SimpleNamespace(release_deferred_review_transport=lambda **kwargs: called.append(kwargs))
    with pytest.raises(ValueError, match='execution_checkpoint_invalid'):
        services.release_reviewed_provider_transport(run_dir=tmp_path,
            binding={'deferred_decision_sha256': 'a' * 64, 'deferred_receipt_sha256': 'b' * 64},
            provider_task=task, evidence=object())
    assert called == []

@pytest.mark.parametrize('relative', [
    'first_observed_source_flow.json',
    'evidence/outcome_missing_local__outcome_missing_local.json',
    'evidence/first_observed_source_flow__first_observed_source_flow.json',
    'evidence/first_observed_source_flow_v2__first_observed_source_flow.json',
])
def test_published_or_interrupted_source_flow_blocks_model_reentry(tmp_path, relative):
    from easyicu.research_agent.orchestration.services import refuse_model_access_to_local_source_flow
    refuse_model_access_to_local_source_flow(tmp_path)
    path = tmp_path / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('{}', encoding='utf-8')
    with pytest.raises(RuntimeError, match='source_flow_model_reentry_forbidden'):
        refuse_model_access_to_local_source_flow(tmp_path)


def test_execution_reentry_refuses_local_source_flow_before_any_step(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from easyicu.research_agent.execution import phase
    from easyicu.research_agent.orchestration.services import PipelineServices

    evidence_dir = tmp_path / 'evidence'
    evidence_dir.mkdir()
    (evidence_dir / 'first_observed_source_flow__first_observed_source_flow.json').write_text('{}')
    monkeypatch.setattr(phase, '_materialize_approved_case_duration_if_needed',
                        lambda *args, **kwargs: pytest.fail('execution started'))
    with pytest.raises(RuntimeError, match='source_flow_model_reentry_forbidden'):
        phase.run_execute_phase(SimpleNamespace(_services=PipelineServices()), plan_result=None,
            cohort_path=tmp_path / 'cohort.parquet', trajectory_binding=None, run_dir=tmp_path,
            run_id='example', skill_obj=None, notes=None, emit_progress=lambda *args, **kwargs: None)


def test_deferred_preplan_reuses_only_host_verified_citation_keys(tmp_path):
    from easyicu.research_agent.bibliographic_metadata import LiteratureScreeningDecision
    from easyicu.research_agent.literature import LiteratureBundle
    from easyicu.research_agent.orchestration.services import PipelineServices
    from tests.support.plan_literature import record

    included, excluded = record(0), record(1)
    bundle = LiteratureBundle(research_question='synthetic', citations=[included, excluded], screening_decisions=[
        LiteratureScreeningDecision(citation_key=included.key, source='synthetic', disposition='include',
            evidence_role='direct_comparator', rationale='synthetic'),
        LiteratureScreeningDecision(citation_key=excluded.key, source='synthetic', disposition='exclude',
            evidence_role='related_context', rationale='synthetic'),
    ])
    calls = []
    class HostStage:
        def reuse_preplan_literature(self, **kwargs):
            calls.append(kwargs)
            return bundle
    services = PipelineServices(deferred_first_observed_stage=HostStage())
    result = services.deferred_preplan_literature(context='context', run_dir=tmp_path, evidence='evidence')
    assert result == (None, 'context', [included.key], [included.key], bundle)
    assert calls == [{'context': 'context', 'run_dir': tmp_path, 'evidence': 'evidence'}]


def test_deferred_stage_preplan_never_enters_search_for_any_profile(tmp_path, monkeypatch):
    from easyicu.research_agent import pipeline as owner
    from easyicu.research_agent.orchestration.services import PipelineServices
    class HostStage:
        def reuse_preplan_literature(self, **kwargs):
            return None
    monkeypatch.setattr(owner._scientific_plan_gate, 'defer_preplan_literature',
                        lambda *args, **kwargs: pytest.fail('normal literature routing must not run'))
    host = SimpleNamespace(_services=PipelineServices(deferred_first_observed_stage=HostStage()), _config=object())
    result = owner._run_preplan_literature_and_hypothesis(host, skill_obj=None,
        emit_progress=lambda *args, **kwargs: None, run_id='synthetic', agent_context='context',
        run_dir=tmp_path, evidence='evidence', findings=[], context='context',
        context_path=tmp_path / 'context.json', llm=object(), resume_state=None)
    assert result == (None, 'context', [], [], None)


def test_deferred_plan_dependence_scope_preserves_reviewed_interval_and_expires():
    from contextlib import contextmanager
    from easyicu.research_agent.orchestration.services import PipelineServices
    from easyicu.research_agent.planning.dependence_authority import host_independent_units_scope
    from easyicu.research_agent.schema import AnalysisPlan, AnalysisStep
    from tests.support.plan_literature import context as synthetic_context

    context = synthetic_context()
    context.cohort.provenance = {'evidence_stage': 'metadata_only_planning', 'analysis_unit': 'icu_stay'}
    spec = {'exposure':'marker', 'exposure_levels':[0, 1], 'outcome':'endpoint', 'outcome_levels':[0, 1],
        'outcome_positive_value':1, 'level_match_policy':'exact_typed',
        'denominator_policy':'observed_outcome_rows', 'missing_outcome_policy':'exclude_from_denominator',
        'confidence_level':0.95}
    step = AnalysisStep(step_id='distribution', intent='Describe outcome', planned_analysis_role='primary',
        method='descriptive', inputs=['marker', 'endpoint'], expected_outputs=['table:exposure_outcome_distribution'],
        exposure_outcome_distribution_spec=spec)
    plan = AnalysisPlan(research_question=context.research_question, analysis_type='descriptive_study', steps=[step])
    checked = []
    class Stage:
        @contextmanager
        def review_scope(self, *, context):
            checked.append(context)
            with host_independent_units_scope(lambda observed: observed is context):
                yield
    services = PipelineServices(deferred_first_observed_stage=Stage())
    for _ in range(2):
        bound = services.bind_plan_dependence(plan=plan, context=context)
        assert bound.steps[0].exposure_outcome_distribution_spec.confidence_level == 0.95
        assert bound.steps[0].exposure_outcome_distribution_spec.interval_method == 'wilson'
    assert checked == [context, context]
    ordinary = PipelineServices().bind_plan_dependence(plan=plan, context=context)
    assert ordinary.steps[0].exposure_outcome_distribution_spec.confidence_level is None
    assert ordinary.steps[0].exposure_outcome_distribution_spec.interval_method == 'none_counts_only'


def test_deferred_plan_dependence_refuses_invalid_host_proof():
    from contextlib import contextmanager
    from easyicu.research_agent.orchestration.services import PipelineServices
    class InvalidStage:
        @contextmanager
        def review_scope(self, *, context):
            raise ValueError('synthetic signed preparation changed')
            yield
    with pytest.raises(ValueError, match='preparation changed'):
        PipelineServices(deferred_first_observed_stage=InvalidStage()).bind_plan_dependence(plan=object(), context=object())
