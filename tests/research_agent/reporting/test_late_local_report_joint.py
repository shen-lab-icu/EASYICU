"""Joint late-report contract; only signed-host verification is a unit stub.

The plan, EvidenceStore, input capsule, result envelope, table, claims, reader
and disclosure owner are real. No Provider or patient data is used.
"""
import hashlib
import hmac
import json
from types import SimpleNamespace

import pytest

from easyicu.research_agent.authority import run_input
from easyicu.research_agent.canonical_json import canonical_sha256
from easyicu.research_agent.reporting import write_phase
from easyicu.research_agent.reporting.manuscript_reader import build_manuscript_reader
from easyicu.research_agent.schema import AnalysisPlan, AnalysisStep
from tests.support.local_outcome_missingness import _setup


def _plan():
    spec = {'exposure':'exposure', 'exposure_levels':[0, 1], 'outcome':'death',
        'outcome_levels':[0, 1], 'outcome_positive_value':1, 'level_match_policy':'exact_typed',
        'denominator_policy':'observed_outcome_rows', 'missing_outcome_policy':'exclude_from_denominator',
        'confidence_level':0.95}
    step = AnalysisStep(step_id='distribution', intent='Describe exposure and outcome',
        planned_analysis_role='primary', method='descriptive', inputs=['exposure','death'], expected_outputs=['table:exposure_outcome_distribution'], exposure_outcome_distribution_spec=spec)
    return AnalysisPlan(research_question='Describe mortality', steps=[step])


def _sealed_input(tmp_path, evidence, plan_sha, decision_sha, required):
    recipe = json.dumps({'fixture':'signed-host-boundary-stub',
                         'source_flow_report':{'unknown_outcome_count_required':required}}).encode()
    (tmp_path / 'first_observed_deferred_recipe.json').write_bytes(recipe)
    evidence.register_file(kind='log', description='recipe', source_path=tmp_path / 'first_observed_deferred_recipe.json',
        evidence_id='first_observed_deferred_recipe', producer='host', generation_mode='system')
    output = tmp_path / 'deferred_input'
    output.mkdir()
    capsule = json.dumps({'cohort_sha256':'c' * 64, 'cohort_rows':100}).encode()
    (output / 'run_input_capsule.json').write_bytes(capsule)
    capsule_record = evidence.register_file(kind='log', description='actual capsule',
        source_path=output / 'run_input_capsule.json', evidence_id='deferred_first_observed_actual_capsule')
    body = {'schema_version':'easyicu.first-observed-deferred-input-receipt/1',
        'approved_plan_sha256':plan_sha, 'decision_set_sha256':decision_sha,
        'recipe_sha256':hashlib.sha256(recipe).hexdigest(), 'capsule_sha256':capsule_record.sha256,
        'cohort_sha256':'c' * 64}
    signature = hmac.new(b'synthetic-test-secret', json.dumps(body, sort_keys=True).encode(), hashlib.sha256).hexdigest()
    receipt_path = tmp_path / 'deferred_first_observed_receipt.json'
    receipt_path.write_text(json.dumps({'body':body, 'signature':signature}))
    receipt_record = evidence.register_file(kind='log', description='synthetic signed receipt',
        source_path=receipt_path, evidence_id='deferred_first_observed_receipt')
    return body, signature, receipt_record


@pytest.mark.parametrize('missing,required', [(0, True), (2, True), (2, False)])
def test_late_combined_report_keeps_original_and_both_local_sources(tmp_path, monkeypatch, missing, required):
    real_input_verifier = run_input.verified_deferred_input_capsule_digest
    _, records, evidence = _setup(tmp_path, monkeypatch, missing=missing, real_envelope=True)
    monkeypatch.setattr(run_input, 'verified_deferred_input_capsule_digest', real_input_verifier)
    plan = _plan()
    plan_sha = canonical_sha256(plan.model_dump(mode='json'))
    decision_sha = 'b' * 64
    body, signature, receipt = _sealed_input(tmp_path, evidence, plan_sha, decision_sha, required)
    records[0]['run_input_capsule_sha256'] = body['capsule_sha256']
    events = []

    def late_host_boundary(*, run_dir, evidence, approved_plan_sha256, decision_set_sha256):
        # Explicit unit stub: independent host receipt signature validation and
        # source-flow acquisition are exercised by Web consumer tests, not here.
        events.append('publish')
        assert approved_plan_sha256 == plan_sha and decision_set_sha256 == decision_sha
        assert hmac.compare_digest(signature, hmac.new(b'synthetic-test-secret',
            json.dumps(body, sort_keys=True).encode(), hashlib.sha256).hexdigest())
        payload = {'schema_version':'easyicu.first-observed-source-flow-evidence/1',
            'not_lifetime_first_icu':True, 'disclosure':'local_report_with_publication_review',
            'binding':{'approved_plan_sha256':plan_sha, 'decision_set_sha256':decision_sha,
                'recipe_sha256':body['recipe_sha256'], 'coordinate_sha256':'e'*64,
                'source_attestation_sha256':'f'*64, 'actual_capsule_sha256':body['capsule_sha256'],
                'actual_cohort_sha256':body['cohort_sha256'], 'deferred_receipt_sha256':receipt.sha256},
            'counts':{'source_patients_n':101, 'source_stays_n':101, 'eligible_patients_n':100,
                'first_icu_order_or_type_unverifiable_n':1,
                'excluded_patients_by_reason':{'first_icu_order_or_type_unverifiable_n':1},
                'excluded_stays_by_reason':{'first_icu_order_or_type_unverifiable_n':1}}}
        assert payload['counts']['source_patients_n'] == 100 + sum(payload['counts']['excluded_patients_by_reason'].values())
        assert payload['counts']['source_stays_n'] == 100 + sum(payload['counts']['excluded_stays_by_reason'].values())
        capsule = json.loads((run_dir / 'deferred_input' / 'run_input_capsule.json').read_text())
        assert body['cohort_sha256'] == capsule['cohort_sha256']
        assert payload['counts']['eligible_patients_n'] == capsule['cohort_rows']
        rec = evidence.register_json(kind='table', description='local flow', payload=payload,
            filename='first_observed_source_flow.json', evidence_id='first_observed_source_flow',
            inputs=['deferred_first_observed_actual_capsule', 'deferred_first_observed_receipt'],
            producer='host', generation_mode='system')
        for key, value in payload['counts'].items():
            for sub, number in value.items() if isinstance(value, dict) else ((None, value),):
                field = 'counts.' + key + ('.' + sub if sub else '')
                evidence.register_numeric_claim(value=str(number), canonical=float(number), evidence_id=rec.evidence_id,
                    step_id=rec.evidence_id, source_field=field, tolerance=0)

    monkeypatch.setattr('easyicu.research_agent.orchestration.human_review_checkpoint.load_checkpoint',
                        lambda *args, **kwargs: SimpleNamespace(consumed_decision_sha256=decision_sha))
    # Only unrelated quality/report-fact hooks are stubbed; reader and disclosure
    # persistence remain the production implementations.
    monkeypatch.setattr(write_phase, '_persist_manuscript_quality_artifacts', lambda **kwargs: None)
    monkeypatch.setattr(write_phase, 'record_result_facts', lambda *args, **kwargs: None)
    monkeypatch.setattr(write_phase, 'expected_manuscript_display_labels', lambda *args, **kwargs: ())
    monkeypatch.setattr(write_phase, 'baseline_reporting_mentions', lambda *args, **kwargs: ())
    monkeypatch.setattr(write_phase, '_draft_bind_and_repair_manuscript',
                        lambda *args, **kwargs: pytest.fail('model work after publication'))
    original = evidence.register_json(kind='statistic', description='prior result', payload={'n':7},
        filename='original.json', evidence_id='original')
    evidence.register_numeric_claim(value='7', canonical=7, evidence_id='original', step_id='original', source_field='n')
    claim = next(c for c in evidence.numeric_claims() if c.evidence_id == 'original')
    original_text = ('## Results\n\nOriginal result: 7[^original].\n\n'
        f'[^original]: value=7; step=original; field=n; evidence=original; sha256={original.sha256}\n')
    path = tmp_path / 'manuscript_scaffold_bound.md'
    path.write_text(original_text)
    binding = write_phase._BindingStageResult(bound=original_text, bound_path=path, manuscript_critique=None,
        numeric_binding_map={'original':claim})
    draft = write_phase._DraftStageResult(current_verified_evidence_records=[], current_evidence_names=[],
        manuscript_packet=None, writer_error_message=None, scaffold=original_text)
    if not required:
        monkeypatch.setattr('easyicu.research_agent.reporting.descriptive_report_facts.render_local_outcome_missingness',
                            lambda **kwargs: pytest.fail('missingness not requested by approved recipe'))
    context = SimpleNamespace(cohort=SimpleNamespace(provenance={}, database='eicu_demo'), cross_database_validation=[])
    pipeline = SimpleNamespace(_services=SimpleNamespace(publish_deferred_source_flow=late_host_boundary))
    result = write_phase._append_deferred_source_flow(pipeline, binding=binding, draft=draft, context=context,
        plan=plan, evidence=evidence, run_dir=tmp_path, findings=[], literature=None,
        per_step_records=records, language='en')
    assert events == ['publish']
    assert original_text.rstrip() in result.bound and path.read_text() == result.bound
    if required:
        assert f'unknown for {missing}[^outcome_missing_missing]' in result.bound
        assert {'outcome_missing_missing', 'outcome_missing_known'} <= set(result.numeric_binding_map)
    else:
        assert '[^outcome_missing_' not in result.bound
    assert 'original' in result.numeric_binding_map
    assert any(key.startswith('source_flow_') for key in result.numeric_binding_map)
    reader = build_manuscript_reader(manuscript=result.bound, evidence=evidence, binding_map=result.numeric_binding_map)
    ids = {claim['claim_id'] for claim in reader['claims']}
    assert 'original' in ids
    if required:
        assert {'outcome_missing_missing', 'outcome_missing_known'} <= ids
    assert any(key.startswith('source_flow_') for key in ids)
    assert evidence.get('manuscript_provenance') is not None
    assert evidence.get('publication_disclosure_review') is not None
    from easyicu.research_agent.orchestration.services import refuse_model_access_to_local_source_flow
    with pytest.raises(RuntimeError, match='model_reentry_forbidden'):
        refuse_model_access_to_local_source_flow(tmp_path)
