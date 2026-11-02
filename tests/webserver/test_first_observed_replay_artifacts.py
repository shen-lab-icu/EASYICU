"""Synthetic replay preserves actual Planner artifact evidence; no Provider is used."""
from __future__ import annotations

import json
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from easyicu.research_agent.agents.progressive_planner import ProgressivePlannerAgent
from easyicu.research_agent.authority.evidence_store import EvidenceStore
from easyicu.research_agent.authority.filesystem import AuthorityFilesystemError
from easyicu.research_agent.authority.plan_review import PlanReviewAuthority
from easyicu.research_agent.authority.raw_preparation import RawPreparationCheckpoint
from easyicu.research_agent.canonical_json import canonical_sha256
from easyicu.research_agent.orchestration.workflow import HumanReviewRequest
from easyicu.research_agent.planning.progressive_artifacts import (
    persist_progressive_planning_artifacts, persist_progressive_planning_authority,
)
from easyicu.research_agent.planning.progressive_contract import ProgressivePlanOutline
from easyicu.research_agent.authority.plan_lifecycle import (
    build_normalized_plan_lineage, persist_normalized_plan,
)
from easyicu.research_agent.providers.mocks import ScriptedMockLLMClient
from easyicu.webserver import agent_pipeline_runs
from easyicu.webserver.first_observed_deferred_recipe import FirstObservedDeferredReviewStage
from tests.research_agent.planning.progressive_planner_fixtures import (
    _foundation_payload, _materialization_payloads, _outline_payload, _payload,
)
from tests.support.first_observed_synthetic_review import positive_case
from tests.support.first_observed_synthetic_literature import KEY


def test_replay_uses_digest_bound_source_artifacts_and_new_compile_receipt(tmp_path, monkeypatch):
    context, skeleton, _, _, _, literature, _ = positive_case(tmp_path)
    context = context.model_copy(update={'cohort': context.cohort.model_copy(update={
        'n_stays': 0, 'provenance': {**context.cohort.provenance, 'patient_rows_read': False},
    })})
    payload = skeleton.model_dump(mode='json')
    payload['steps'].append(_payload()['steps'][3])  # Real measurement-audit owner.
    outline = _outline_payload(payload)
    for design in outline['design_selection']['candidates']:
        design['figure_role'] = 'Show counts and observed proportions.'
    typed_outline = ProgressivePlanOutline.model_validate(outline)
    foundation = _foundation_payload(payload)
    foundation['outline_sha256'] = canonical_sha256(typed_outline.model_dump(mode='json'))
    materializations = _materialization_payloads(payload)
    for item, step in zip(materializations, typed_outline.steps, strict=True):
        item['outline_step_sha256'] = canonical_sha256(step.model_dump(mode='json'))
    responses = [json.dumps(outline), json.dumps(foundation),
                 *(json.dumps(item) for item in materializations)]
    agent = ProgressivePlannerAgent(ScriptedMockLLMClient(responses))
    plan = agent.run(context, allowed_literature_citation_keys=[KEY])
    facts = agent.last_result.facts
    source = tmp_path / 'source'
    source.mkdir()
    evidence = EvidenceStore(source)
    context_path = source / 'research_context.json'
    context_path.write_text(context.model_dump_json())
    evidence.register_file(kind='log', description='synthetic context', source_path=context_path,
                           evidence_id='research_context', producer='test', generation_mode='system')
    literature_path = source / 'preplan_literature_bundle.json'
    literature_path.write_text(literature.model_dump_json())
    evidence.register_file(kind='log', description='synthetic literature', source_path=literature_path,
                           evidence_id='preplan_literature_bundle', producer='test', generation_mode='system')
    paths = persist_progressive_planning_artifacts(run_dir=source, evidence=evidence,
        outline=facts.outline, foundation=facts.foundation, materializations=facts.materializations,
        skeleton=facts.skeleton, compile_receipt=facts.compile_receipt,
        prompt_metrics=facts.prompt_metrics, prompt_pack_version='test')
    metrics_path = source / 'planner_prompt_metrics.json'
    metrics_path.write_text(json.dumps(facts.prompt_metrics))
    evidence.register_file(kind='log', description='synthetic metrics', source_path=metrics_path,
                           evidence_id='planner_prompt_metrics', producer='test', generation_mode='system')
    plan_path = source / 'analysis_plan.json'
    plan_path.write_text(plan.model_dump_json())
    evidence.register_file(kind='log', description='synthetic plan', source_path=plan_path,
                           evidence_id='analysis_plan', producer='test', generation_mode='system')
    lineage = build_normalized_plan_lineage(proposed_plan=plan, proposed_source='fresh',
        pre_normalization_plan=plan, normalized_plan=plan, resume_scientific_semantics_changed=False,
        host_scientific_semantics_changed=False)
    lineage_path = persist_normalized_plan(run_dir=source, evidence=evidence, normalized=lineage)
    progressive = persist_progressive_planning_authority(run_dir=source, evidence=evidence,
        proposed_plan_sha256=lineage.proposed.plan_sha256, normalized_plan_sha256=lineage.plan_sha256,
        normalized_plan_authority_sha256=lineage.authority_sha256,
        normalized_plan_evidence_id=f'plan_lifecycle_revision_{plan.revision}',
        normalized_plan_filename=lineage_path.name, prompt_pack_version='test')
    assert progressive.normalized_plan_sha256 == lineage.plan_sha256
    authority = PlanReviewAuthority.create(plan=plan,
        evidence_sha256={record.evidence_id: record.sha256 for record in evidence.verified_records()})
    payload = {'plan_review_authority': authority.model_dump(mode='json')}
    request = HumanReviewRequest.create(kind='scientific_stop', summary='synthetic', payload=payload,
                                        authority_sha256=canonical_sha256(payload))
    checkpoint = RawPreparationCheckpoint.pending(plan_sha256=authority.plan_sha256, request=request)
    (source / 'raw_preparation_checkpoint.json').write_text(checkpoint.model_dump_json())
    monkeypatch.setattr(agent_pipeline_runs, 'raw_preparation_review_owner',
                        lambda *_: (SimpleNamespace(run_dir=str(source)), {}))
    monkeypatch.setattr(FirstObservedDeferredReviewStage, '_skeleton_and_source',
                        lambda *_: (facts.skeleton, None, None, None))
    monkeypatch.setattr(FirstObservedDeferredReviewStage, 'review_scope', lambda *_args, **_kwargs: nullcontext())
    stage = FirstObservedDeferredReviewStage('run', 'study', 'project', 'session', 'a'*64)
    target = tmp_path / 'target'
    target.mkdir()
    target_evidence = EvidenceStore(target)
    target_context = target / 'research_context.json'
    target_context.write_text(context.model_dump_json())
    target_evidence.register_file(kind='log', description='synthetic context', source_path=target_context,
                                  evidence_id='research_context', producer='test', generation_mode='system')
    restored = stage.reuse_preplan_literature(context=context, run_dir=target, evidence=target_evidence)
    assert restored == literature
    assert (target / 'preplan_literature_bundle.json').read_bytes() == literature_path.read_bytes()
    original_literature = literature_path.read_bytes()
    literature_path.write_bytes(original_literature + b' ')
    with pytest.raises(AuthorityFilesystemError, match='digest mismatch'):
        stage.reuse_preplan_literature(context=context, run_dir=target, evidence=target_evidence)
    literature_path.write_bytes(original_literature)
    metrics = stage.persist_replayed_progressive_evidence(context=context, run_dir=target,
        evidence=target_evidence, prompt_pack_version='test')
    assert metrics == facts.prompt_metrics
    assert target_evidence.get('progressive_plan_compile_receipt') is not None
    paths.outline.write_text('{}')
    with pytest.raises(AuthorityFilesystemError, match='digest mismatch'):
        stage.persist_replayed_progressive_evidence(context=context, run_dir=target,
            evidence=target_evidence, prompt_pack_version='test')
