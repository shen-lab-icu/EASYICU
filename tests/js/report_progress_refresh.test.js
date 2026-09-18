'use strict';
const assert = require('node:assert/strict');
require('./guided_pi_module_harness.cjs');
global.window = {};
require(require('node:path').resolve(process.argv[2]));
const messages = [];
let count = 0;
const owner = window.EU_GUIDED_PI_CHILDJOB.create({
  tr: en => en,
  activity: { pipelineEventLabel: event => event.step },
  upsertActivityStep: (activity, step) => activity.steps.push(step),
  render: () => {}, api: {}, loadWorkflow: async () => { count += 1; },
  messages: () => messages,
  childJobId: () => 'job-1',
});
for (let i = 0; i < 10; i += 1) {
  owner.handleChildJobEvent('job-1', 'easyicu_full_run_report_resume_submitted', { type: 'progress', step: 'report_repair' });
}
assert.equal(count, 1, 'repeated section messages must not flood the workflow endpoint');
owner.handleChildJobEvent('job-1', 'easyicu_full_run_report_resume_submitted', { type: 'progress', step: 'report_validation' });
assert.equal(count, 2, 'the next reported stage must refresh the workflow');
owner.handleChildJobEvent('another-job', '', { type: 'progress', step: 'planning' });
assert.equal(count, 2, 'foreign jobs must not change the selected workflow');
process.stdout.write(JSON.stringify({ ok: true, cases: 3 }));
