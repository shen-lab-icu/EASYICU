'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const js = path.resolve('src/easyicu/webserver/static/js');
const at = seconds => new Date(Date.UTC(2026, 0, 1) + seconds * 1000).toISOString();
const plain = value => JSON.parse(JSON.stringify(value));
const supplied = process.argv.find(arg => arg.startsWith('{'));

function fixture() {
  const cells = [];
  for (let index = 0; index < 30; index++) {
    cells.push(['step', `analysis_${index}`, null, null, at(index * 10), at(index * 10 + 1)]);
    cells.push(['coder', `analysis_${index}`, 'repair', 1, at(index * 10 + 2), at(index * 10 + 3)]);
  }
  return { present: true, job_id: 'job-synthetic', kind: 'agent-run', status: 'done', run_id: 'run-synthetic',
    created_at_epoch: Date.parse(at(0)) / 1000, finished_at_epoch: Date.parse(at(800)) / 1000,
    progress: [], artifact_refs: [], progress_compacted: true,
    progress_table: { schema_version: 'easyicu.pi-job-progress/1', defaults: {
      type: 'progress', status: 'complete', run_id: 'run-synthetic',
    }, columns: ['step', 'step_id', 'action', 'attempt', 'started_at', 'ended_at'], batches: [cells] } };
}

for (const language of ['en', 'zh']) {
  const context = vm.createContext({ window: { EU_LANG: language }, document: { getElementById: () => null } });
  vm.runInContext('Date.now = () => 1767225600000', context);
  for (const name of ['modules', 'activity', 'childjob', 'job-narration', 'replay', 'transcript']) {
    vm.runInContext(fs.readFileSync(path.join(js, `screens-guided-pi-${name}.js`), 'utf8'), context);
  }
  const modules = context.window.EasyICU.guidedPi;
  const job = supplied ? JSON.parse(supplied) : fixture();
  const rows = modules.jobProgress(job);
  const legacy = { ...job, progress: rows };
  delete legacy.progress_table;
  assert.equal(modules.jobProgress(legacy), rows, 'legacy receipts retain their event shape');
  assert.equal(modules.jobProgress({ progress: [] }).length, 0);
  assert.ok(rows.every(row => row.timestamp && row.started_at && row.ended_at));
  const tr = (en, zh) => language === 'zh' ? zh : en;
  const activity = modules.require('activity').create({ tr, esc: String, iconHtml: () => '',
    resourceName: () => '', resourceKey: () => '', resourceButton: () => '', runFailureText: () => '' });
  const upsertActivityStep = (target, row) => {
    const found = target.steps.find(item => item.id === row.id);
    if (found) Object.assign(found, row); else target.steps.push(row);
  };
  function hydrate(receipt) {
    const messages = [];
    const child = modules.require('childJob').create({ tr, activity, upsertActivityStep,
      render() {}, messages: () => messages, session: () => ({}), childJobId: () => '' });
    child.hydrateProjectedJob(receipt);
    child.hydrateProjectedJob(receipt);
    return messages[0];
  }
  const hydrated = hydrate(job);
  assert.deepEqual(plain(hydrated.steps), plain(hydrate(legacy).steps));
  const steps = hydrated.steps.filter(row => row.step === 'plan_step');
  assert.equal(steps.length, 30, 'each early and late step survives a reload');
  assert.ok(steps.every(row => row.endedAt - row.startedAt === 1000));
  assert.ok(steps.every(row => row.runId === job.run_id), 'detail links keep the run coordinate');
  const repairs = hydrated.steps.filter(row => row.action === 'repair');
  assert.equal(repairs.length, 30);
  assert.ok(repairs.every(row => row.durationKnown && row.endedAt - row.startedAt === 1000));
  assert.ok(activity.render(hydrated).includes(tr('1.0 s', '1.0 秒')), 'measured durations are rendered');
  const narration = modules.require('jobNarration');
  assert.equal(narration.processNotes(job, null, tr), narration.processNotes(legacy, null, tr));
  assert.match(narration.processNotes(job, null, tr), /30/);

  const transcript = modules.require('transcript').create({ tr, activity, upsertActivityStep,
    timeMs: value => Date.parse(value) || 0, resourceKey: () => '', modelErrorText: () => '',
    workflowActionCode: () => '', runFailureText: () => '' });
  function history(receipt) {
    return transcript.transcriptMessages({ transcript: [], archived_child_jobs: [receipt],
      conversation_replay: { turns: [{ job_id: 'host-synthetic', kind: 'host_action',
        action_code: 'execute_plan', action_key: 'synthetic', child_job_id: job.job_id,
        status: 'done', started_at: at(0), ended_at: at(800) }] } });
  }
  assert.deepEqual(plain(history(job)), plain(history(legacy)));
  const historical = history(job).find(row => row.role === 'activity');
  assert.equal(historical.steps.filter(row => row.step === 'plan_step').length, 30);

  const revision = { ...fixture(), progress_table: { ...fixture().progress_table,
    defaults: { type: 'progress', step: 'report_repair' }, columns: [], batches: [[[]]] } };
  const presentation = modules.require('replay').childJobPresentation(revision, tr);
  assert.match(presentation.title, language === 'zh' ? /报告/ : /[Rr]eport/);
}
console.log('job projection contracts passed');
