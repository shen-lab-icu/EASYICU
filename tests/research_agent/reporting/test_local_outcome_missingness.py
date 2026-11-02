"""Local whole-cohort outcome availability follows sealed distribution inputs."""
import pytest

from easyicu.research_agent.reporting.descriptive_report_facts import render_local_outcome_missingness
from tests.support.local_outcome_missingness import _setup


@pytest.mark.parametrize('missing', [0, 2])
def test_local_report_uses_overall_sealed_table_and_bound_claims(tmp_path, monkeypatch, missing):
    plan, records, evidence = _setup(tmp_path, monkeypatch, missing=missing)
    text, bindings = render_local_outcome_missingness(plan=plan, records=records, evidence=evidence, language='en')
    assert f'unknown for {missing}[^outcome_missing_missing]' in text
    assert f'{100-missing}[^outcome_missing_known]' in text
    assert bindings['outcome_missing_missing'].canonical == missing
    assert bindings['outcome_missing_known'].canonical == 100-missing
    assert evidence.get('outcome_missing_local').inputs == ['summary', 'distribution_table', 'envelope']
    from easyicu.research_agent.reporting.manuscript_reader import build_manuscript_reader
    reader = build_manuscript_reader(manuscript=text, evidence=evidence, binding_map=bindings)
    assert len(reader['claims']) == 2


@pytest.mark.parametrize('case', ['excluded', 'no_overall', 'bad_table', 'wrong_claim', 'wrong_capsule'])
def test_local_report_rejects_partial_or_unverified_full_cohort(tmp_path, monkeypatch, case):
    plan, records, evidence = _setup(tmp_path, monkeypatch, excluded=1 if case == 'excluded' else 0,
                                     overall=case != 'no_overall')
    if case == 'bad_table':
        (tmp_path / evidence.get('distribution_table').relative_path).write_text('tampered')
    if case == 'wrong_claim':
        evidence._numeric_claims = []
    if case == 'wrong_capsule':
        records[0]['run_input_capsule_sha256'] = 'b' * 64
        from easyicu.research_agent.audits.envelope_consumers import RegisteredOutputEnvelopeConsumer
        old = RegisteredOutputEnvelopeConsumer.authoritative_writer_records
        monkeypatch.setattr(RegisteredOutputEnvelopeConsumer, 'authoritative_writer_records',
            lambda self, rows, evidence_store: [{**old(self, rows, evidence_store)[0], 'run_input_capsule_sha256': 'b' * 64}])
    with pytest.raises(ValueError):
        render_local_outcome_missingness(plan=plan, records=records, evidence=evidence, language='zh')
    assert evidence.get('outcome_missing_local') is None


def test_local_report_with_real_result_envelope(tmp_path, monkeypatch):
    plan, records, evidence = _setup(tmp_path, monkeypatch, real_envelope=True)
    text, claims = render_local_outcome_missingness(plan=plan, records=records, evidence=evidence, language='en')
    assert 'unknown for 2[^outcome_missing_missing]' in text
    assert claims['outcome_missing_known'].canonical == 98
