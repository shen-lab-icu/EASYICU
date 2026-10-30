/* Synthetic contracts for new/open project and its actual shell wiring. */
'use strict';
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const { test } = require('node:test');
const js = path.join(__dirname, '../../src/easyicu/webserver/static/js');
const shell = fs.readFileSync(path.join(js, 'screens-guided.js'), 'utf8');
const owner = fs.readFileSync(path.join(js, 'screens-guided-projects.js'), 'utf8');
const serverRoot = '/synthetic/isolated-home/Documents/EasyICU';
const escape = value => String(value ?? '').replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('"', '&quot;');

function fakeHost() {
  return { innerHTML: '', querySelector: () => null, querySelectorAll: () => [] };
}

function harness({ language = 'en', hosted = false, create, rows = [] } = {}) {
  const hosts = { gdSessions: fakeHost(), gdFolderDialogHost: fakeHost(), gdFolderControls: fakeHost() };
  const picker = { open: false };
  const nameInput = { focused: false, selected: false, focus() { this.focused = true; }, select() { this.selected = true; } };
  const attempts = [];
  const payloads = [];
  // The shell's closure state lives on the sandbox; each context reads it
  // afresh, as guidedProjectContext() does.
  const base = {
    t: (en, zh) => language === 'zh' ? zh : en,
    icon: () => '', esc: escape, attr: escape, compactPath: value => value || '', fmtRunTime: () => '',
    guidedDrafts: { loading: false, error: null, data: { default_parent_dir: serverRoot } },
    selectedGuidedDraft: null, pendingGuidedGoal: null, localDraftRows: () => rows,
    createProject: title => sandbox.createLocalGuidedDraft(title),
  };
  const context = () => ({
    ...base,
    guidedFolderDialogMode: sandbox.guidedFolderDialogMode,
    guidedFolderSeedTitle: sandbox.guidedFolderSeedTitle,
    guidedFolderBrowser: sandbox.guidedFolderBrowser,
  });
  const sandbox = {
    guidedFolderDialogMode: null, guidedFolderSeedTitle: '', guidedFolderBrowser: {},
    esc: escape, compactPath: value => value || '',
    console, setTimeout, clearTimeout,
    document: {
      getElementById: id => hosts[id] || null,
      querySelector: selector => selector === '#gdResearchProjectRail .gd-project-picker' ? picker
        : (selector === '[data-project-new-title]' && hosts.gdSessions.innerHTML.includes('data-project-new-title') ? nameInput : null),
    },
    EasyICU: { guidedPi: { optional: name => name === 'hosted' ? { active: () => hosted } : null } },
    selectedGuidedDraft: null, pendingGuidedGoal: null,
    pushUser: text => attempts.push(text), pushBot: () => {},
    renderThread: () => {}, loadGuidedDrafts: () => Promise.resolve(),
    bindGuidedDraftMemory: draft => Promise.resolve({ ok: true, session: {
      draft_id: draft.id, project_title: draft.title, project_dir: draft.project_dir,
    } }),
    closeGuidedFolderDialog: () => { sandbox.guidedFolderDialogMode = null; },
    renderGuidedFolderControls: () => {}, renderGuidedFolderDialog: () => {},
    piProjectShellActive: () => false,
    startFreshGuidedProjectThread: () => {}, continuePendingGuidedGoal: () => {},
    guidedProjectContext: context,
    EU_API: { createGuidedDraft: payload => {
      payloads.push(payload);
      return create ? create(payload) : Promise.resolve({ ok: true, draft: {
        id: 'synthetic-project', title: payload.title, project_dir: `${serverRoot}/${payload.title}`,
      } });
    } },
  };
  sandbox.window = sandbox;
  vm.createContext(sandbox);
  vm.runInContext(owner, sandbox);
  const projects = sandbox.EU_GUIDED_PROJECTS;
  sandbox.guidedProjectRenderer = name => projects[name](context());
  sandbox.defaultStudyTitle = () => projects.defaultStudyTitle(context());
  sandbox.blankGuidedDraftPayload = title => ({ title });
  for (const [from, to] of [
    ['  function hostedInstance() {', '  function closeGuidedFolderDialog() {'],
    ['  function createLocalGuidedDraft(', '  /* ============== conversation script'],
  ]) {
    const start = shell.indexOf(from);
    const end = shell.indexOf(to, start);
    assert.ok(start >= 0 && end > start, from);
    vm.runInContext(shell.slice(start, end), sandbox);
  }
  const rail = () => hosts.gdSessions.innerHTML;
  const submit = value => hosts.gdSessions.onsubmit({
    target: { matches: selector => selector === '[data-project-new-form]', querySelector: () => ({ value }) },
    preventDefault() {},
  });
  return { hosts, sandbox, projects, picker, nameInput, payloads, attempts, rail, submit,
    get ctx() { return context(); },
    renderRail: () => projects.renderProjectRail(context()) };
}

test('the project list ends with New project and Open existing project; no separate menu', () => {
  for (const language of ['en', 'zh']) {
    const h = harness({ language });
    h.renderRail();
    h.projects.renderFolderControls(h.ctx);
    assert.ok(h.rail().includes('data-project-new'));
    assert.ok(h.rail().includes(language === 'zh' ? '新建项目' : 'New project'));
    assert.ok(h.rail().includes('data-folder-choice="open"'));
    assert.ok(h.rail().includes(language === 'zh' ? '打开已有项目…' : 'Open existing project…'));
    assert.equal(h.hosts.gdFolderControls.innerHTML, '');
    assert.doesNotMatch(h.rail(), /gd-project-summary|本地研究工作区|Local research workspace/);
  }
  assert.doesNotMatch(owner, /新建 \/ 打开研究项目|data-folder-menu-toggle|data-newstudy|新建空白项目/);
});

test('an empty project picker remains dismissible and stays closed across a repaint', () => {
  const h = harness();
  h.renderRail();
  assert.match(h.rail(), /<details class="gd-project-picker" data-project-id="" open data-popover-menu>/);
  const oldPicker = { dataset: { projectId: '' }, open: false, querySelector: () => null };
  h.hosts.gdSessions.querySelector = selector => selector === '.gd-project-picker' ? oldPicker : null;
  h.renderRail();
  assert.match(h.rail(), /<details class="gd-project-picker" data-project-id="" data-popover-menu>/);
  oldPicker.open = true;
  h.renderRail();
  assert.match(h.rail(), /<details class="gd-project-picker" data-project-id="" open data-popover-menu>/);
});

test('a hosted instance keeps its single New research project control and no list foot', () => {
  const h = harness({ hosted: true });
  h.renderRail();
  h.projects.renderFolderControls(h.ctx);
  assert.doesNotMatch(h.rail(), /data-project-new|data-folder-choice="open"/);
  assert.match(h.hosts.gdFolderControls.innerHTML, /data-folder-choice="new"/);
});

test('New project opens the list with a selected name field in the interface language', () => {
  for (const [language, title] of [['zh', '新研究'], ['en', 'New study']]) {
    const h = harness({ language });
    h.sandbox.openGuidedFolderDialog('new');
    assert.equal(h.sandbox.guidedFolderDialogMode, null);
    assert.equal(h.picker.open, true);
    assert.ok(h.rail().includes(`data-project-new-title value="${title}"`), language);
    assert.ok(h.nameInput.focused && h.nameInput.selected);
    assert.doesNotMatch(h.rail(), /data-draft-slug|data-draft-parent-dir|guided-/);
  }
});

test('submitting the name creates one project named by the server, with no folder or parent fields', async () => {
  const h = harness({ language: 'zh' });
  h.projects.startProjectNaming(h.ctx);
  await h.submit('  合成 队列 甲  ');
  assert.equal(h.payloads.length, 1);
  assert.deepEqual(Object.keys(h.payloads[0]), ['title']);
  assert.equal(h.payloads[0].title, '合成 队列 甲');
  assert.equal(h.attempts.length, 1);
  assert.doesNotMatch(h.rail(), /data-project-new-title/);
  // An empty name falls back to the default title.
  await h.submit('   ');
  assert.equal(h.payloads[1].title, '新研究');
});

test('a create in flight disables the field; a repeated submit sends one request', async () => {
  let complete;
  const pending = new Promise(resolve => { complete = resolve; });
  const h = harness({ create: () => pending });
  h.projects.startProjectNaming(h.ctx);
  const first = h.submit('Synthetic study');
  assert.match(h.rail(), /data-project-new-title[^>]*disabled/);
  assert.match(h.rail(), /Creating…/);
  const second = h.submit('Synthetic study');
  assert.equal(h.payloads.length, 1);
  complete({ ok: true, draft: { id: 'synthetic', title: 'Synthetic study', project_dir: `${serverRoot}/Synthetic study` } });
  await Promise.all([first, second]);
  assert.equal(h.attempts.length, 1);
  assert.doesNotMatch(h.rail(), /data-project-new-title/);
});

test('a refused create keeps the name field with a localized inline reason; retry clears it', async () => {
  for (const language of ['en', 'zh']) {
    for (const [code, en, zh] of [
      ['guided_project_folder_create_failed', 'Check that this folder can be written to', '请检查该目录能否写入'],
      ['unfamiliar_code', 'Could not create the study folder', '无法创建研究文件夹'],
    ]) {
      let fail = true;
      const h = harness({ language, create: () => Promise.resolve(fail
        ? { ok: false, blocked: true, error: code, reason: '<server reason>' }
        : { ok: true, draft: { id: 'retry', title: 'Retry', project_dir: `${serverRoot}/Retry` } }) });
      h.projects.startProjectNaming(h.ctx);
      await h.submit('Synthetic study');
      assert.match(h.rail(), /data-project-new-title value="Synthetic study"/);
      assert.match(h.rail(), /gd-project-new-error" role="alert"/);
      assert.ok(h.rail().includes(language === 'zh' ? zh : en));
      assert.match(h.rail(), /&lt;server reason>/);
      assert.equal(h.attempts.length, 0);
      fail = false;
      await h.submit('Retry');
      assert.doesNotMatch(h.rail(), /gd-project-new-error|data-project-new-title/);
      assert.equal(h.attempts.length, 1);
    }
  }
});

test('Escape leaves the name field without closing the project list', () => {
  const h = harness();
  h.projects.startProjectNaming(h.ctx);
  let stopped = false;
  h.hosts.gdSessions.onkeydown({
    key: 'Escape', target: { matches: selector => selector === '[data-project-new-title]' },
    preventDefault() {}, stopPropagation() { stopped = true; },
  });
  assert.ok(stopped);
  assert.doesNotMatch(h.rail(), /data-project-new-title/);
  assert.ok(h.rail().includes('data-project-new'));
});

test('Open existing project is a path field, Browse and Open, with no new-project tabs', () => {
  const h = harness({ language: 'zh' });
  h.sandbox.openGuidedFolderDialog('open', '');
  assert.equal(h.sandbox.guidedFolderDialogMode, 'open');
  h.projects.renderFolderDialog(h.ctx);
  const dialog = h.hosts.gdFolderDialogHost.innerHTML;
  assert.match(dialog, /data-existing-project-dir/);
  assert.match(dialog, /data-browseprojectfolder/);
  assert.match(dialog, /data-openprojectfolder/);
  assert.ok(dialog.includes('打开已有项目'));
  assert.doesNotMatch(dialog, /data-folder-choice|data-draft-title|data-createdraft|gds-known|guided-\*/);
});

test('a hosted New research project still asks for its name in a dialog', () => {
  const h = harness({ hosted: true });
  h.sandbox.openGuidedFolderDialog('new', 'Synthetic hosted study');
  assert.equal(h.sandbox.guidedFolderDialogMode, 'new');
  h.projects.renderFolderDialog(h.ctx);
  assert.match(h.hosts.gdFolderDialogHost.innerHTML, /data-hosted-project-dialog/);
  assert.match(h.hosts.gdFolderDialogHost.innerHTML, /data-draft-title value="Synthetic hosted study"/);
});

test('long server reasons wrap within the hosted dialog', () => {
  const css = fs.readFileSync(path.join(js, '../css/guided-projects.css'), 'utf8');
  assert.match(css, /\[data-folder-create-error\]\{[^}]*min-width:0;[^}]*overflow-wrap:anywhere/);
});
