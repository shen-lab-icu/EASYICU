'use strict';
const assert = require('node:assert/strict');
const path = require('node:path');
require('./guided_pi_module_harness.cjs');
global.window = { EU_LANG: 'zh' };
const js = path.resolve(__dirname, '../../src/easyicu/webserver/static/js');
for (const name of ['html-escape', 'screens-guided-pi-error-text', 'screens-guided-pi-result-summary',
  'screens-guided-pi-run-answer', 'screens-guided-pi-run-stop', 'screens-guided-pi-plan-actions']) require(path.join(js, `${name}.js`));
const tr = (en, zh) => zh;
const errors = window.EasyICU.guidedPi.require('errorText').create({ tr, staticPreview: false });
const plan = { analysis_type: 'association_study', steps: [
  { step_id: 'original', planned_analysis_role: 'primary', method: 'adjusted_association_models' },
] };
const original = { evidence_id: 'copied_original', sha256: 'a'.repeat(64), source_status: 'current',
  produced_by_step: 'original', origin: { run_id: 'source', evidence_id: 'original', sha256: 'a'.repeat(64) },
  headers: ['fit_status', 'estimate', 'ci_low', 'ci_high', 'effect_scale', 'is_primary_contrast', 'model_id'],
  rows: [['fitted', '1.20', '1.10', '1.30', 'odds_ratio', 'true', 'main']] };
const run = { run_id: 'append', artifact_refs: ['agent_plan.json', 'result_tables.json'].map(artifact => ({
  run_id: 'append', artifact, sha256: 'b'.repeat(64),
})) };
const answer = window.EasyICU.guidedPi.require('runAnswer').create({
  tr, esc: window.EU_HTML.esc, projectId: () => 'project', resourceButton: ref => ref.evidence_id || '',
  api: () => ({ loadPiCopilotResearchArtifact: async (_project, _run, name) => ({ payload:
    name === 'agent_plan.json' ? { analysis_type: 'association_study', steps: [
      { step_id: 'added', planned_analysis_role: 'secondary' },
    ] } : { tables: [original, { ...original, evidence_id: 'other_definition', produced_by_step: 'old_append',
      origin: { run_id: 'previous_append' }, rows: [['fitted', '2.20', '2.10', '2.30', 'odds_ratio', 'true', 'other']] }],
      append: { source_run_id: 'source', kind: 'subgroup', label_en: 'Add a subgroup', label_zh: '加亚组',
      source_plan: plan, strata: [{ subgroup: 'stratum', stratum: 'A', estimate: 1.4, ci_low: 1.1,
        ci_high: 1.8, interaction_p_value: 0.03 }] } },
  }) }),
});
(async () => {
  await answer.load(run);
  const html = answer.render(run);
  assert.match(html, /OR 1\.20/, 'the earlier primary estimate stays visible beside the subgroup answer');
  assert.match(html, /OR 1\.40/);
  assert.match(html, /之前的主要结果/);
  assert.match(html, /copied_original/, 'the earlier estimate links to its digest-verified copy');
  for (const code of ['result_append_source_changed', 'result_append_execution_failed', 'result_append_unavailable', 'plan_append_scope_changed', 'plan_append_source_changed', 'plan_append_source_not_approvable']) {
    const text = errors.runFailureText(code);
    assert.match(text, /追加|封存/);
    assert.doesNotMatch(text, /(?:result|plan)_append_|\bundefined\b/);
  }
  const stop = window.EasyICU.guidedPi.require('runStop').create({ tr, runFailureText: errors.runFailureText });
  const stopped = stop.render({ savedRuns: [{ run_ref: 'stopped', run_status: 'failed', gate_reason: 'result_append_source_changed' }] });
  assert.match(stopped, /改动/);
  assert.match(stopped, /核对|检查/);
  assert.doesNotMatch(stopped, /result_append_source_changed/);
  let approvals = 0;
  const actions = window.EasyICU.guidedPi.require('planActions').create({
    tr, busy: () => false, sessionIsStale: () => false,
    session: () => ({ session_id: 'conversation', binding: { study_context_id: 'study', run_id: 'append-synthetic' } }),
    workflow: () => ({ next_action_code: 'operator_plan_approval_required', plan_review_summary: { run_id: 'append-synthetic' } }),
    api: () => ({ submitAgentRunReview: async () => { approvals += 1; } }),
  });
  for (const message of ['continue', 'go ahead', 'approved', '继续', '同意', '批准']) {
    assert.equal(await actions.continueUserRequestedSystemProgression(message), false);
  }
  assert.equal(await actions.continueSystemOwnedPlanProgression({ passive: true }), false);
  assert.equal(approvals, 0, 'messages and passive reopening never approve an append');
  console.log('append evidence: earlier estimate, new strata, source link and readable stop passed');
})().catch(error => { console.error(error); process.exitCode = 1; });
