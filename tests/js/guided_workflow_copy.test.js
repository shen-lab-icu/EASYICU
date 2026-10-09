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
const owner = global.EasyICU.guidedPi.require('aside').create({
  tr: en => en,
  esc: value => String(value == null ? '' : value),
  iconHtml: name => `[${name}]`,
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

workflow.stages.unshift({ id: 'idea', status: 'optional', required_for_completion: false });
owner.syncProjectWorkflowAside();
assert.match(body.innerHTML, /4\/7<\/strong> required stages complete/);
assert.match(body.innerHTML, /Idea mining · Optional/);
assert.match(body.innerHTML, /study-item optional.*?\[dot\]/);
assert.doesNotMatch(body.innerHTML, /study-item optional.*?\[lock\]/);
assert.equal(workflow.completed_required_stages, 4);
workflow.stages[0] = { id: 'idea', status: 'complete', required_for_completion: true };
owner.syncProjectWorkflowAside();
assert.doesNotMatch(body.innerHTML, /· Optional/);

// Once a plan exists, the open setup row says the plan fills it in, so it is
// not read as a step the researcher skipped.
workflow = {
  current_stage: 'plan', completed_required_stages: 2, required_stage_count: 7,
  missing_setup_fields: ['cohort_eligibility', 'primary_exposure', 'analysis_goal'],
  stages: [
    { id: 'setup', status: 'ready', reason_code: 'cohort_eligibility_confirmation_required' },
    { id: 'plan', status: 'review_required', reason_code: 'plan_execution_upgrade_required' },
  ],
};
owner.syncProjectWorkflowAside();
assert.match(body.innerHTML, /Study setup<\/div><div class="si-s">Filled in from the plan; inclusion criteria are confirmed before approval<\/div>/);
workflow.stages[0].reason_code = 'study_setup_incomplete';
owner.syncProjectWorkflowAside();
assert.match(body.innerHTML, /Study setup<\/div><div class="si-s">Filled in from the plan; nothing to enter by hand<\/div>/);
// Before a plan exists nothing is promised, and a missing data source is the
// researcher's choice, not the plan's.
workflow.current_stage = 'setup';
workflow.missing_setup_fields = ['data_source', 'outcome'];
workflow.stages[1] = { id: 'plan', status: 'blocked', reason_code: 'active_export_or_setup_required' };
owner.syncProjectWorkflowAside();
assert.doesNotMatch(body.innerHTML, /Filled in from the plan/);
assert.match(body.innerHTML, /Choose the data source for this study/);
workflow.missing_setup_fields = ['outcome'];
workflow.stages[0].reason_code = 'cohort_eligibility_confirmation_required';
owner.syncProjectWorkflowAside();
assert.match(body.innerHTML, /Confirm the inclusion criteria before the plan is approved/);
assert.doesNotMatch(body.innerHTML, /Waiting for the preceding governed stage/);

process.stdout.write(JSON.stringify({ locked: true, actionable: true }));
