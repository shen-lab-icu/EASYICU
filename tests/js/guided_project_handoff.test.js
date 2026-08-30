/* Executable contract for Agent/StudyContext projects that are not Guided folders. */
'use strict';

const assert = require('node:assert/strict');
const path = require('node:path');

global.window = global;
const localStorageValues = new Map();
global.localStorage = {
  getItem(key) { return localStorageValues.has(key) ? localStorageValues.get(key) : null; },
  setItem(key, value) { localStorageValues.set(key, String(value)); },
};
const host = { innerHTML: '' };
const removalHost = { innerHTML: '' };
global.document = {
  getElementById(id) {
    if (id === 'gdSessions') return host;
    if (id === 'gdRemoveDraftDialogHost') return removalHost;
    return null;
  },
};

for (const modulePath of process.argv.slice(2)) require(path.resolve(modulePath));

function context(selectedGuidedDraft, rows, guidedDraftRemoval) {
  return {
    t: en => en,
    icon: () => '',
    esc: value => String(value == null ? '' : value),
    attr: value => String(value == null ? '' : value),
    compactPath: value => String(value || ''),
    slugifyDraftFolder: value => String(value || ''),
    fmtRunTime: () => '',
    localDraftRows: () => rows || [],
    selectedGuidedDraft,
    guidedDraftRemoval: guidedDraftRemoval || null,
    guidedDrafts: { loading: false, error: null, data: { drafts: [] } },
  };
}

const shellRail = global.EU_GUIDED_PROJECTS.renderShellRail(context(null));
assert.match(shellRail, /data-project-rail-toggle/g);
assert.match(shellRail, /gd-rail-restore/);
assert.match(shellRail, /gd-rail-collapse/);
assert.equal(global.EU_GUIDED_PROJECTS.isProjectRailCollapsed(), false);
global.EU_GUIDED_PROJECTS.setProjectRailCollapsed(true);
assert.equal(global.EU_GUIDED_PROJECTS.isProjectRailCollapsed(), true);
assert.equal(localStorageValues.get('easyicu.guided.projectRailCollapsed.v1'), '1');
global.EU_GUIDED_PROJECTS.setProjectRailCollapsed(false);

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

const allRows = Array.from({ length: 12 }, (_, index) => ({
  id: `project-${index}`,
  title: `Project ${index}`,
  status: 'ready',
  data_mode: 'real',
}));
global.EU_GUIDED_PROJECTS.renderProjectRail(context(null, allRows));
assert.equal((host.innerHTML.match(/class="gd-sessline\b/g) || []).length, 12);
assert.match(host.innerHTML, /Project 11/);

global.EU_GUIDED_PROJECTS.setProjectManagement(true);
global.EU_GUIDED_PROJECTS.renderProjectRail(context(allRows[0], allRows));
assert.match(host.innerHTML, /data-project-manage/);
assert.match(host.innerHTML, /data-select-all-projects/);
assert.match(host.innerHTML, /data-select-localdraft="0"[^>]*disabled/);
global.EU_GUIDED_PROJECTS.toggleProjectSelection(allRows[1], true);
global.EU_GUIDED_PROJECTS.toggleProjectSelection(allRows[2], true);
global.EU_GUIDED_PROJECTS.renderProjectRail(context(allRows[0], allRows));
assert.match(host.innerHTML, /<strong>2<\/strong> selected/);
assert.doesNotMatch(host.innerHTML, /data-remove-selected-projects disabled/);
global.EU_GUIDED_PROJECTS.selectAllProjects(allRows, allRows[0].id, true);
assert.equal(global.EU_GUIDED_PROJECTS.selectedProjects(allRows, allRows[0].id).length, 11);

global.EU_GUIDED_PROJECTS.renderDraftRemovalDialog(context(null, allRows, {
  row: allRows[1],
  rows: [allRows[1], allRows[2]],
  trashProjectFolder: false,
  busy: false,
  error: null,
}));
assert.match(removalHost.innerHTML, /2 projects selected/);
assert.match(removalHost.innerHTML, /Project folders remain unchanged on disk/);
global.EU_GUIDED_PROJECTS.setProjectManagement(false);

process.stdout.write(JSON.stringify({ bound: true, unbound: true, empty: true, allProjects: true, multiSelect: true, collapsibleRail: true }));
