"""Shared synthetic distribution and evidence fixtures for local-report tests."""
from copy import deepcopy
from types import SimpleNamespace
import csv

from easyicu.research_agent.audits.envelope_consumers import RegisteredOutputEnvelopeConsumer
from easyicu.research_agent.authority.evidence_store import EvidenceStore
from tests.support.descriptive_report import _inputs


def _setup(tmp_path, monkeypatch, *, missing=2, excluded=0, overall=True, real_envelope=False):
    records, _ = _inputs()
    from easyicu.research_agent.authority import run_input
    monkeypatch.setattr(run_input, 'verified_deferred_input_capsule_digest', lambda **kwargs: 'a' * 64)
    records[0]['run_input_capsule_sha256'] = 'a' * 64
    summary = records[0]['step_summary']
    risks = summary['descriptive_estimates']['outcome_absolute_risks']
    risks[0]['missing_n'] = missing
    risks[0]['denominator'] = 60 - missing
    risks[1]['missing_n'] = 0
    summary.update(outcome='death', denominator_policy='observed_outcome_rows',
        source_row_count_reconciliation={'source_rows': 100 + excluded, 'analyzed_rows': 100,
            'excluded_missing_exposure_rows': excluded, 'filtering_performed': bool(excluded)})
    if real_envelope:
        summary['output_files'] = {'table:exposure_outcome_distribution': 'distribution.csv'}
    evidence = EvidenceStore(tmp_path)
    source = evidence.register_json(kind='statistic', description='summary', payload=summary,
        filename='summary.json', evidence_id='summary', produced_by_step='distribution')
    records[0]['evidence_ids'] = [source.evidence_id]
    for index, risk in enumerate(risks):
        for key in ('missing_n', 'denominator'):
            evidence.register_numeric_claim(value=str(risk[key]), canonical=float(risk[key]),
                evidence_id='summary', step_id='distribution',
                source_field=f'descriptive_estimates.outcome_absolute_risks[{index}].{key}')
    path = tmp_path / 'distribution.csv'
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['row_role', 'exposure_level_index', 'n_rows', 'outcome_missing_n',
                                                     'outcome_observed_n', 'outcome_denominator'])
        writer.writeheader()
        writer.writerow(dict(row_role='overall' if overall else 'unrelated', n_rows=100,
            outcome_missing_n=missing, outcome_observed_n=100-missing, outcome_denominator=100-missing))
        writer.writerow(dict(row_role='exposure_level', exposure_level_index='0.0', n_rows=60,
            outcome_missing_n=missing, outcome_observed_n=60-missing, outcome_denominator=60-missing))
        writer.writerow(dict(row_role='exposure_level', exposure_level_index='1.0', n_rows=40,
            outcome_missing_n=0, outcome_observed_n=40, outcome_denominator=40))
    table = evidence.register_file(kind='table', description='distribution', source_path=path,
        evidence_id='distribution_table', produced_by_step='distribution', script_evidence_id='code')
    evidence.register_json(kind='log', description='synthetic envelope', payload={}, filename='envelope.json', evidence_id='envelope')
    record = {**records[0], 'writer_result_envelope_evidence_id':'envelope',
        'writer_artifact_bindings': {'table:exposure_outcome_distribution':
            {'evidence_id':table.evidence_id, 'sha256':table.sha256}}}
    if real_envelope:
        from easyicu.research_agent.authority.registration import StepEvidenceCommit
        from easyicu.research_agent.authority.result_envelope_sidecar import publish_terminal_step_result_envelope_sidecar
        from easyicu.research_agent.contracts.result_envelope import normalize_step_result_shadow
        envelope = normalize_step_result_shadow(step_id='distribution', step_summary=summary,
                                                output_dir=tmp_path, status='ok')
        sidecar = publish_terminal_step_result_envelope_sidecar(snapshot_envelope=envelope,
            step_id='distribution', attempt_id='attempt-1', checkpoint_id='checkpoint-1',
            script_evidence_id='code', terminal_status='ok', evidence_store=evidence)
        assert sidecar is not None
        StepEvidenceCommit(evidence).commit_validated_step(step_id='distribution',
            pending_aliases={sidecar.evidence_id: [sidecar.alias]}, allowed_evidence_ids=[sidecar.evidence_id],
            register_numeric_claims=lambda: None)
        records[0].update(evidence_ids=['summary', 'distribution_table', sidecar.evidence_id],
                          attempt_id='attempt-1', review_checkpoint_id='checkpoint-1', script_evidence_id='code')
    else:
        monkeypatch.setattr(RegisteredOutputEnvelopeConsumer, 'authoritative_writer_records',
                            lambda self, rows, evidence_store: [deepcopy(record)])
    spec = SimpleNamespace(exposure='exposure', outcome='death', denominator_policy='observed_outcome_rows')
    plan = SimpleNamespace(steps=[SimpleNamespace(step_id='distribution', planned_analysis_role='primary',
                                                  exposure_outcome_distribution_spec=spec)])
    return plan, records, evidence

