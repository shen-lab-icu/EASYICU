"""Report revisions have their own UI lifecycle without changing source gates."""
import copy
import json
from pathlib import Path
import subprocess

import pytest
from easyicu.webserver.pi_copilot.projections import project_job


def snapshot():
    return {'id': 'revision-job', 'kind': 'agent-run', 'status': 'done',
            'events': [{'step': 'report_repair', 'type': 'progress'}],
            'result': {'run_id': 'original-analysis',
                       'gate': {'status': 'blocked', 'reportable': False, 'checks': []},
                       'report_revision': {
                           'schema_version': 'easyicu.web-report-revision/1',
                           'revision_id': 'revision-job', 'source_run_id': 'original-analysis',
                           'status': 'pass', 'analysis_steps_executed': 0,
                           'claim_ceiling': 'analysis_only', 'publication_authorized': False,
                           'output_sha256': 'a' * 64,
                           'pdf_artifact': {'name': 'manuscript_revision.pdf', 'sha256': 'b' * 64,
                                            'revision_id': 'revision-job', 'manuscript_sha256': 'a' * 64}},
                       'artifacts': [{'name': 'manuscript_revision.pdf', 'sha256': 'b' * 64}]}}


def test_current_job_report_receipt_and_pdf_keep_source_gate_blocked():
    original = snapshot()
    before = copy.deepcopy(original)
    p = project_job(original)
    assert p['report_only'] and p['report_revision_ready'] and p['report_revision_pdf_ready']
    assert p['report_revision_id'] == 'revision-job'
    assert p['gate_status'] == 'blocked' and not p['reportable']
    assert not p['manuscript_ready']
    assert p['artifact_refs'][0]['artifact'] == 'manuscript_revision.pdf'
    assert original == before


@pytest.mark.parametrize('mutation', [
    lambda s: s.update(status='running'),
    lambda s: s.update(status='failed'),
    lambda s: s['result']['report_revision'].update(revision_id='older-revision'),
    lambda s: s['result']['report_revision'].update(source_run_id='foreign-analysis'),
    lambda s: s['result']['report_revision'].update(status='failed'),
    lambda s: s['result']['report_revision'].update(analysis_steps_executed=1),
    lambda s: s['result']['report_revision'].update(publication_authorized=True),
    lambda s: s['result']['report_revision'].update(output_sha256='invalid'),
    lambda s: s['result']['report_revision']['pdf_artifact'].update(sha256='c' * 64),
    lambda s: s['result']['report_revision']['pdf_artifact'].update(revision_id='older-revision'),
    lambda s: s['result']['report_revision']['pdf_artifact'].update(manuscript_sha256='c' * 64),
])
def test_wrong_or_unfinished_revision_never_claims_current_pdf_ready(mutation):
    s = snapshot(); mutation(s)
    assert project_job(s)['report_revision_pdf_ready'] is False


def test_live_and_archived_report_labels_do_not_claim_a_new_analysis():
    p = project_job(snapshot())
    cases = [p, {**p, 'status': 'failed'}, {**p, 'status': 'running'},
             {**p, 'report_revision_ready': False, 'report_revision_pdf_ready': False},
             {'status': 'done', 'progress': [{'step': 'report_repair'}], 'gate_status': 'blocked'},
             {'status': 'done', 'analysis_results_available': True, 'analysis_validated': True}]
    source = Path(__file__).resolve().parents[3] / 'src/easyicu/webserver/static/js/screens-guided-pi-replay.js'
    script = '''
global.window = {EasyICU: {guidedPi: {declare: (name, api) => {global.api = api;}}}};
require(process.argv[1]);
process.stdout.write(JSON.stringify(JSON.parse(process.argv[2]).map(j=>api.childJobPresentation(j,(en,zh)=>zh))));
'''
    result = subprocess.run(['node', '-e', script, str(source), json.dumps(cases)], capture_output=True, text=True, check=True)
    rows = json.loads(result.stdout)
    assert rows[0]['title'] == '新版报告与 PDF 已生成，待审阅'
    assert not rows[0]['blocked']
    assert rows[1]['title'] == '报告修订失败；原报告保留'
    assert rows[2]['title'] == '正在复用分析结果修订报告'
    assert all(r['blocked'] for r in rows[3:5])
    assert rows[5]['title'] == '分析已完成；完整质量审阅尚未通过'
