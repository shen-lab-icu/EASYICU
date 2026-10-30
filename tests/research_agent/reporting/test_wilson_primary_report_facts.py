"""Reviewed primary intervals survive into source-bound report facts."""

from copy import deepcopy
from types import SimpleNamespace

import pytest

from easyicu.research_agent.execution.runners.exposure_outcome_distribution_executor import wilson_interval
from easyicu.research_agent.reporting.descriptive_report_facts import (
    compile_counts_only_report_facts,
    writer_question_coverage,
)
from tests.research_agent.reporting.test_descriptive_report_facts import _inputs


def _wilson():
    records, evidence = _inputs()
    evidence.get('summary').relative_path = 'steps/distribution/summary.json'
    summary = records[0]['step_summary']
    summary.update(interval_method='wilson', effective_interval_method='wilson', confidence_level=0.95)
    for kind, numerator in [('exposure_prevalence', 'n'), ('outcome_absolute_risks', 'events')]:
        for row in summary['descriptive_estimates'][kind]:
            low, high = wilson_interval(row[numerator], row['denominator'], confidence_level=0.95)
            row.update(interval_method='wilson', covariance='binomial_independent',
                       ci_low_pct=low, ci_high_pct=high, confidence_level=0.95)
    spec = SimpleNamespace(interval_method='wilson', confidence_level=0.95,
                           exposure='exposure', outcome='outcome')
    step = SimpleNamespace(step_id='distribution', planned_analysis_role='primary',
                           exposure_outcome_distribution_spec=spec, expected_outputs=[])
    return records, evidence, SimpleNamespace(steps=[step])


def test_reviewed_intervals_and_known_outcome_denominators_reach_result_coverage():
    records, evidence, plan = _wilson()
    risk = records[0]['step_summary']['descriptive_estimates']['outcome_absolute_risks'][0]
    low, high = wilson_interval(6, 50, confidence_level=0.95)
    risk.update(denominator=50, missing_n=10, estimate_pct=12.0, ci_low_pct=low, ci_high_pct=high)
    before = deepcopy(records)
    facts = compile_counts_only_report_facts(records, evidence=evidence, reader_display_labels={}, plan=plan)
    assert len(facts) == 4
    assert f'95% Wilson CI, {low:.2f}% to {high:.2f}%' in facts[2].text
    assert '6 of 50 records' in facts[2].text and '10 records had missing outcomes' in facts[2].text
    assert any(field.endswith('.ci_low_pct') for field in facts[2].source_fields)
    manuscript = '## Results\n\n' + '\n'.join(f.scaffold for f in facts) + '\n'
    assert not writer_question_coverage(manuscript, plan=plan, facts=facts, evidence=evidence)
    assert records == before


def test_explicit_95_percent_request_is_reviewed_and_its_report_cannot_omit_interval():
    from tests.research_agent.planning.test_descriptive_question_uncertainty import _review
    from tests.support.descriptive_plan import _plan

    question = "报告各组院内死亡人数、比例及 95% 置信区间"
    plan = _plan()
    steps = [step.model_copy(update={"exposure_outcome_distribution_spec":
        step.exposure_outcome_distribution_spec.model_copy(update={
            "interval_method": "wilson", "confidence_level": 0.95,
        })}) if step.exposure_outcome_distribution_spec is not None else step
        for step in plan.steps]
    review = _review(question, plan=plan.model_copy(update={"steps": steps}))
    assert review.facts["requested_descriptive_uncertainty"]["requested_confidence_levels"] == [0.95]
    assert not any(f.code == "QUESTION_DESCRIPTIVE_UNCERTAINTY_NOT_MET" for f in review.findings)
    missing_plan = _review(question)
    assert any(f.code == "QUESTION_DESCRIPTIVE_UNCERTAINTY_NOT_MET" for f in missing_plan.findings)

    records, evidence, executed_plan = _wilson()
    facts = compile_counts_only_report_facts(records, evidence=evidence, reader_display_labels={}, plan=executed_plan)
    manuscript = '## Results\n\n' + '\n'.join(f.scaffold for f in facts) + '\n'
    assert not writer_question_coverage(manuscript, plan=executed_plan, facts=facts, evidence=evidence)
    missing_interval = manuscript.replace('95% Wilson CI', 'interval')
    findings = writer_question_coverage(missing_interval, plan=executed_plan, facts=facts, evidence=evidence)
    assert {finding.detail['outcome'] for finding in findings} == {'exposure_distribution', 'outcome'}


@pytest.mark.parametrize('change', ['wrong_level', 'wrong_endpoint', 'counts_plan', 'wrong_row_level',
                                    'wrong_method', 'interval_missing', 'outside_interval', 'wrong_interval',
                                    'missing_plan'])
def test_intervals_without_matching_plan_and_source_are_rejected(change):
    records, evidence, plan = _wilson()
    risk = records[0]['step_summary']['descriptive_estimates']['outcome_absolute_risks'][0]
    spec = plan.steps[0].exposure_outcome_distribution_spec
    if change == 'wrong_level':
        spec.confidence_level = 0.90
    elif change == 'wrong_endpoint':
        spec.outcome = 'other'
    elif change == 'counts_plan':
        spec.interval_method = 'none_counts_only'
    elif change == 'wrong_row_level':
        risk['confidence_level'] = 0.90
    elif change == 'wrong_method':
        risk['covariance'] = 'cluster_robust'
    elif change == 'interval_missing':
        risk['ci_low_pct'] = None
    elif change == 'outside_interval':
        risk['ci_low_pct'] = risk['estimate_pct'] + 1
    elif change == 'wrong_interval':
        risk['ci_low_pct'] = 1.0  # Contains the point estimate, but is not its Wilson bound.
    elif change == 'missing_plan':
        plan = None
    with pytest.raises(ValueError):
        compile_counts_only_report_facts(records, evidence=evidence, reader_display_labels={}, plan=plan)


@pytest.mark.parametrize('change', ['clustered', 'contrast'])
def test_other_inference_designs_are_not_misrepresented_as_independent_wilson(change):
    records, evidence, plan = _wilson()
    summary = records[0]['step_summary']
    if change == 'clustered':
        summary['effective_interval_method'] = 'patient_cluster_robust_wald'
        summary['descriptive_estimates']['dependence'] = {'unit': 'patient'}
    else:
        summary['descriptive_estimates']['risk_difference'] = {
            'direction': 'comparison_minus_reference',
            'interpretation_ceiling': 'descriptive_unadjusted_not_causal',
        }
    assert compile_counts_only_report_facts(records, evidence=evidence, reader_display_labels={}, plan=plan) == ()


def test_reviewed_interval_facts_survive_strict_numeric_binding(tmp_path):
    import json
    from easyicu.research_agent.authority.evidence_store import EvidenceStore, EvidenceEnforcementMode
    from easyicu.research_agent.reporting.descriptive_report_facts import (
        missing_primary_result_facts, render_descriptive_report_claims,
        verified_descriptive_source_records,
    )
    from easyicu.research_agent.reporting.manuscript_post import bind_numeric_values

    records, _, plan = _wilson()
    store = EvidenceStore(tmp_path / 'run')
    source = tmp_path / 'summary.json'
    source.write_text(json.dumps(records[0]['step_summary']))
    store.register_file(kind='statistic', source_path=source, evidence_id='summary',
                        description='Recorded Wilson intervals', produced_by_step='distribution')
    store.register_step_summary_numerics(step_id='distribution', evidence_id='summary', summary=records[0]['step_summary'])
    records[0]['evidence_ids'] = ['summary']
    verified = verified_descriptive_source_records(records, store)
    assert len(verified) == 1
    facts = compile_counts_only_report_facts(verified, evidence=store, reader_display_labels={}, plan=plan)
    text = ('## Abstract\n\n**Results:**\n\n**Conclusions:**\nCaution.\n\n'
            '## Results\n\n### Cohort characteristics\n\n### Primary outcome\n\n'
            '## Discussion\n\nBoundary.\n\n## Conclusion\n\nCaution.')
    projected = render_descriptive_report_claims(text, facts)
    bound, bindings, untraced = bind_numeric_values(projected, evidence=store,
        enforcement_mode=EvidenceEnforcementMode.STRICT, per_step_records=records)
    assert bindings and not untraced
    assert missing_primary_result_facts(bound, facts) == {}


def test_shared_report_compiler_admits_sealed_wilson_source_and_binds_all_parts(tmp_path, monkeypatch):
    import json
    from easyicu.research_agent.audits.envelope_consumers import RegisteredOutputEnvelopeConsumer
    from easyicu.research_agent.authority.evidence_store import EvidenceStore
    from easyicu.research_agent.reporting.descriptive_report_facts import (
        compile_primary_counts_only_report_facts, render_descriptive_report_claims,
    )
    from easyicu.research_agent.reporting.manuscript_post import bind_numeric_values

    records, _, plan = _wilson()
    store = EvidenceStore(tmp_path / 'run')
    source = tmp_path / 'summary.json'
    source.write_text(json.dumps(records[0]['step_summary']))
    store.register_file(kind='statistic', source_path=source, evidence_id='summary',
                        description='Recorded Wilson intervals', produced_by_step='distribution')
    store.register_step_summary_numerics(step_id='distribution', evidence_id='summary', summary=records[0]['step_summary'])
    records[0]['evidence_ids'] = ['summary']
    monkeypatch.setattr(RegisteredOutputEnvelopeConsumer, 'authoritative_writer_records',
                        lambda self, completed, **kwargs: completed)
    facts = compile_primary_counts_only_report_facts(records, evidence=store, reader_display_labels={}, plan=plan)
    manuscript = '## Abstract\n\n**Results:**\n\n**Conclusions:**\nCaution.\n\n## Results\n\n### Descriptive results\n'
    manuscript = render_descriptive_report_claims(manuscript, facts)
    bound, bindings, untraced = bind_numeric_values(store.bind_manuscript(manuscript, per_step_records=records),
        evidence=store, per_step_records=records)
    assert bindings and not untraced and '95% Wilson CI' in manuscript
    assert not writer_question_coverage(bound, plan=plan, facts=facts, evidence=store)
