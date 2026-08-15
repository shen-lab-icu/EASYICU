/* Executable contract for Agent/StudyContext projects that are not Guided folders. */
'use strict';

const assert = require('node:assert/strict');
const path = require('node:path');

global.window = global;
const host = { innerHTML: '' };
global.document = {
  getElementById(id) { return id === 'gdSessions' ? host : null; },
};

require(path.resolve(process.argv[2]));

function context(selectedGuidedDraft) {
  return {
    t: en => en,
    icon: () => '',
    esc: value => String(value == null ? '' : value),
    attr: value => String(value == null ? '' : value),
    compactPath: value => String(value || ''),
    slugifyDraftFolder: value => String(value || ''),
    fmtRunTime: () => '',
    localDraftRows: () => [],
    selectedGuidedDraft,
    guidedDrafts: { loading: false, error: null, data: { drafts: [] } },
  };
}

global.EU_GUIDED_PROJECTS.renderProjectRail(context({
  id: 'study-e1',
  title: 'Existing E1 project',
  study_context_id: 'study-e1',
  study_context_revision: 9,
}));
assert.match(host.innerHTML, /Existing E1 project/);
assert.match(host.innerHTML, /Bound StudyContext · r9/);
assert.doesNotMatch(host.innerHTML, /No study folders yet/);

global.EU_GUIDED_PROJECTS.renderProjectRail(context({
  id: 'idea-unbound',
  title: 'Unbound idea project',
}));
assert.match(host.innerHTML, /Project selected · setup continues here/);
assert.doesNotMatch(host.innerHTML, /No study folders yet/);

global.EU_GUIDED_PROJECTS.renderProjectRail(context(null));
assert.match(host.innerHTML, /No study folders yet/);

process.stdout.write(JSON.stringify({ bound: true, unbound: true, empty: true }));
