/* Executable copy contract for actionable versus locked workflow stages. */
'use strict';

const assert = require('node:assert/strict');
const path = require('node:path');

global.window = global;
const head = { innerHTML: '' };
const body = { innerHTML: '' };
const aside = { querySelector: () => head };
global.document = {
  getElementById(id) {
    if (id === 'gdStudyAside') return aside;
    if (id === 'gdAsideBody') return body;
    return null;
  },
};

for (const modulePath of process.argv.slice(2)) require(path.resolve(modulePath));

let workflow = null;
const owner = global.EU_GUIDED_PI_ASIDE.create({
  tr: en => en,
  esc: value => String(value == null ? '' : value),
  iconHtml: () => '',
  projectId: () => 'project-copy',
  displayProjectTitle: value => String(value || ''),
  demoMode: () => false,
  shell: () => 'pi',
  workflow: () => workflow,
  project: () => ({ title: 'Copy contract project' }),
});

workflow = {
  current_stage: 'plan',
  completed_required_stages: 4,
  required_stage_count: 7,
  stages: [
    { id: 'plan', status: 'ready', reason_code: 'plan_ready' },
    { id: 'analysis', status: 'blocked', reason_code: 'validated_analysis_required' },
  ],
};
owner.syncProjectWorkflowAside();
assert.match(body.innerHTML, /Later stage/);
assert.doesNotMatch(body.innerHTML, /Next step/);

workflow.stages[1].status = 'ready';
owner.syncProjectWorkflowAside();
assert.match(body.innerHTML, /Next step/);
assert.doesNotMatch(body.innerHTML, /Later stage/);

process.stdout.write(JSON.stringify({ locked: true, actionable: true }));
