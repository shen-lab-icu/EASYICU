/* Synthetic contracts for the project dialog and its actual shell wiring. */
'use strict';
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const { test } = require('node:test');
const js = path.join(__dirname, '../../src/easyicu/webserver/static/js');
const shell = fs.readFileSync(path.join(js, 'screens-guided.js'), 'utf8');
const owner = fs.readFileSync(path.join(js, 'screens-guided-projects.js'), 'utf8');
const serverRoot = '/synthetic/isolated-home/easyicu/projects';
const escape = value => String(value ?? '').replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('"', '&quot;');

function harness({ root = serverRoot, language = 'en', create, seed = 'Synthetic study' } = {}) {
  const host = { innerHTML: '' };
  const attempts = [];
  const bots = [];
  const payloads = [];
  const ctx = {
    t: (en, zh) => language === 'zh' ? zh : en,
    icon: () => '', esc: escape, attr: escape, compactPath: value => value || '',
    slugifyDraftFolder: () => 'synthetic-study',
    guidedFolderDialogMode: 'new', guidedFolderSeedTitle: seed,
    guidedDraftParentDir: '', guidedDrafts: { data: { default_parent_dir: root } },
    guidedFolderBrowser: {},
  };
  ctx.openFolderDialog = (mode, title) => {
    ctx.guidedFolderDialogMode = mode;
    ctx.guidedFolderSeedTitle = title;
  };
  const sandbox = {
    console, setTimeout, clearTimeout,
    document: { getElementById: id => id === 'gdFolderDialogHost' ? host : null },
    ...ctx, selectedGuidedDraft: null, pendingGuidedGoal: null,
    blankGuidedDraftPayload: title => ({ title }),
    pushUser: text => attempts.push(text), pushBot: (...text) => bots.push(text),
    renderThread: () => {}, loadGuidedDrafts: () => Promise.resolve(),
    bindGuidedDraftMemory: draft => Promise.resolve({ ok: true, session: {
      draft_id: draft.id, project_title: draft.title, project_dir: draft.project_dir,
    } }),
    closeGuidedFolderDialog: () => { ctx.guidedFolderDialogMode = null; },
    piProjectShellActive: () => false,
    startFreshGuidedProjectThread: () => {}, continuePendingGuidedGoal: () => {},
    guidedProjectContext: () => ctx,
    EU_API: { createGuidedDraft: payload => {
      payloads.push(payload);
      return create ? create(payload) : Promise.resolve({ ok: true, draft: {
        id: 'synthetic-project', title: payload.title,
        project_dir: `${serverRoot}/guided-synthetic-study`,
      } });
    } },
  };
  sandbox.window = sandbox;
  vm.createContext(sandbox);
  vm.runInContext(owner, sandbox);
  sandbox.guidedProjectRenderer = name => sandbox.EU_GUIDED_PROJECTS[name](ctx);
  sandbox.defaultStudyTitle = () => sandbox.EU_GUIDED_PROJECTS.defaultStudyTitle(ctx);
  const start = shell.indexOf('  function createLocalGuidedDraft(');
  const end = shell.indexOf('  /* ============== conversation script', start);
  assert.ok(start >= 0 && end > start);
  vm.runInContext(shell.slice(start, end), sandbox);
  return { host, ctx, sandbox, payloads, attempts, bots,
    render: () => sandbox.EU_GUIDED_PROJECTS.renderFolderDialog(ctx) };
}

function previewHarness(options) {
  const h = harness(options);
  const slugStart = shell.indexOf('  function slugifyDraftFolder(');
  const slugEnd = shell.indexOf('  function guidedProjectContext()', slugStart);
  const rendererStart = shell.indexOf('  function guidedProjectRenderer(');
  const rendererEnd = shell.indexOf('  function guidedProjectRailClass()', rendererStart);
  assert.ok(slugStart >= 0 && slugEnd > slugStart);
  assert.ok(rendererStart >= 0 && rendererEnd > rendererStart);
  vm.runInContext(shell.slice(slugStart, slugEnd) + shell.slice(rendererStart, rendererEnd), h.sandbox);
  h.ctx.slugifyDraftFolder = h.sandbox.slugifyDraftFolder;
  h.render();
  const decode = value => value.replaceAll('&quot;', '"').replaceAll('&lt;', '<').replaceAll('&amp;', '&');
  const code = h.host.innerHTML.match(/<code\b([^>]*)>([^<]*)<\/code>/);
  assert.ok(code, 'the new-study dialog renders a path preview');
  const preview = {
    textContent: decode(code[2]),
    dataset: { parentDisplay: decode(code[1].match(/data-parent-display="([^"]*)"/)?.[1] || '') },
    set innerHTML(_) { assert.fail('the preview must be written through textContent'); },
  };
  const field = attribute => ({
    value: decode(h.host.innerHTML.match(new RegExp(`${attribute} value="([^"]*)"`))[1]),
    dataset: {},
    closest(selector) {
      if (selector === `[${attribute}]`) return this;
      return selector === '[data-draft-setup]' ? box : null;
    },
  });
  const title = field('data-draft-title');
  const slug = field('data-draft-slug');
  const parent = field('data-draft-parent-dir');
  const fields = {
    '[data-draft-title]': title, '[data-draft-slug]': slug, '[data-draft-parent-dir]': parent,
    '[data-draft-folder-preview]': code[1].includes('data-draft-folder-preview') ? preview : null,
  };
  const box = { querySelector: selector => fields[selector] || null };
  h.host.querySelector = selector => selector === '[data-draft-setup]' ? box : null;
  const listeners = {};
  Object.assign(h.sandbox, {
    shell: { addEventListener: (name, handler) => { listeners[name] = handler; } },
    EXTRACT: { handleInput: () => false }, IDEA: { state: () => null },
  });
  const start = shell.indexOf("      shell.addEventListener('input', (e) => {");
  const end = shell.indexOf('\n\n      // composer', start);
  assert.ok(start >= 0 && end > start);
  vm.runInContext(shell.slice(start, end), h.sandbox);
  return { ...h, title, slug, parent, preview,
    dispatch(name, target, value) { target.value = value; listeners[name]({ target }); } };
}

test('typing a study title refreshes the preview and keeps Chinese folder names', () => {
  const h = previewHarness({ language: 'zh', seed: '' });
  h.dispatch('input', h.title, '合成队列 甲号');
  assert.equal(h.slug.value, '合成队列-甲号');
  assert.equal(h.preview.textContent, `${serverRoot}/guided-合成队列-甲号-...`);
});

test('editing a folder name refreshes the preview and survives later title typing', () => {
  const h = previewHarness();
  h.dispatch('change', h.slug, '手动 文件夹');
  assert.equal(h.slug.value, '手动-文件夹');
  assert.equal(h.slug.dataset.edited, 'true');
  assert.equal(h.preview.textContent, `${serverRoot}/guided-手动-文件夹-...`);
  h.dispatch('input', h.title, '另一个合成标题');
  assert.equal(h.slug.value, '手动-文件夹');
  assert.equal(h.preview.textContent, `${serverRoot}/guided-手动-文件夹-...`);
});

test('parent input refreshes the preview and clearing it restores the rendered parent', () => {
  const h = previewHarness();
  assert.equal(h.preview.dataset.parentDisplay, serverRoot);
  h.dispatch('input', h.parent, ' /synthetic/另一个父目录/ ');
  assert.equal(h.preview.textContent, '/synthetic/另一个父目录/guided-synthetic-study-...');
  h.dispatch('input', h.parent, '');
  assert.equal(h.preview.textContent, `${serverRoot}/guided-synthetic-study-...`);
});

test('an empty folder-name preview uses the study fallback', () => {
  const h = previewHarness();
  h.slug.value = '';
  h.sandbox.EU_GUIDED_PROJECTS.updateFolderPreview(h.ctx);
  assert.equal(h.preview.textContent, `${serverRoot}/guided-study-...`);
});

test('the preview owner applies the same Unicode slug policy as rendering', () => {
  const h = previewHarness();
  h.slug.value = ' 自定义 ＡＢ ';
  h.sandbox.EU_GUIDED_PROJECTS.updateFolderPreview(h.ctx);
  assert.equal(h.preview.textContent, `${serverRoot}/guided-自定义-ab-...`);
});

test('title and parent markup are handled as preview text', () => {
  const h = previewHarness();
  h.dispatch('input', h.title, '<b>合成 标题</b>');
  assert.equal(h.preview.textContent, `${serverRoot}/guided-b-合成-标题-b-...`);
  h.dispatch('input', h.parent, '/synthetic/<b>父目录</b>');
  assert.equal(h.preview.textContent, '/synthetic/<b>父目录</b>/guided-b-合成-标题-b-...');
});

test('server root supplies dialog, starter, Browse and path labels', async () => {
  const h = harness();
  h.render();
  assert.match(h.host.innerHTML, new RegExp(`data-draft-parent-dir value="${serverRoot}"`));
  assert.match(h.host.innerHTML, new RegExp(`${serverRoot}/guided-`));
  assert.equal(h.sandbox.EU_GUIDED_PROJECTS.defaultParentDir(h.ctx), serverRoot);
  assert.equal(h.sandbox.EU_GUIDED_PROJECTS.draftParentDir(h.ctx), serverRoot);
  assert.equal(h.sandbox.EU_GUIDED_PROJECTS.parentPathLabel(h.ctx), serverRoot);
  await h.sandbox.quickCreateGuidedStarterFolder();
  assert.equal(h.payloads[0].parent_dir, serverRoot);
  // Execute the real open/new Browse click branches without a browser/server.
  for (const [selector, inputSelector] of [
    ['data-browseprojectfolder', 'data-existing-project-dir'],
    ['data-browsedraftparent', 'data-draft-parent-dir'],
  ]) {
    const start = shell.indexOf(`        if (e.target.closest('[${selector}]'))`);
    const end = shell.indexOf('\n        if (', start + 1);
    let requested;
    const box = { querySelector: query => query === `[${inputSelector}]` ? { value: '' } : null };
    h.sandbox.e = { target: { closest: query => query === `[${selector}]` ? {} : box } };
    h.sandbox.captureGuidedDraftDialogState = () => {};
    h.sandbox.loadGuidedFolderBrowser = value => { requested = value; };
    vm.runInContext(`(function () { ${shell.slice(start, end)} })()`, h.sandbox);
    assert.equal(requested, serverRoot);
  }
});

test('missing server value sends blank parent_dir and never invents a path', async () => {
  const h = harness({ root: '' });
  h.render();
  assert.match(h.host.innerHTML, /data-draft-parent-dir value=""/);
  assert.doesNotMatch(h.host.innerHTML, /~\/easyicu/);
  await h.sandbox.quickCreateGuidedStarterFolder();
  assert.equal(h.payloads[0].parent_dir, '');
});

test('starter replaces a previously selected parent with the server default', async () => {
  const h = harness();
  h.sandbox.guidedDraftParentDir = '/synthetic/previous-parent';
  await h.sandbox.quickCreateGuidedStarterFolder();
  assert.equal(h.sandbox.guidedDraftParentDir, serverRoot);
  assert.equal(h.payloads[0].parent_dir, serverRoot);
});

test('busy dialog disables create; repeated click sends one request and one success line', async () => {
  let complete;
  const pending = new Promise(resolve => { complete = resolve; });
  const h = harness({ create: () => pending });
  const first = h.sandbox.createLocalGuidedDraft('Synthetic study', 'study', serverRoot);
  assert.match(h.host.innerHTML, /data-createdraft[^>]*disabled/);
  assert.match(h.host.innerHTML, /Creating study folder/);
  const second = h.sandbox.createLocalGuidedDraft('Synthetic study', 'study', serverRoot);
  assert.equal(h.payloads.length, 1);
  assert.equal(h.attempts.length, 0);
  complete({ ok: true, draft: { id: 'synthetic', title: 'Synthetic study', project_dir: serverRoot + '/guided-study' } });
  await Promise.all([first, second]);
  assert.equal(h.attempts.length, 1);
  assert.equal(h.ctx.guidedFolderDialogMode, null);
  assert.equal(h.host.innerHTML, '');
});

test('blocked create stays open with localized inline reason; retry clears error', async () => {
  for (const language of ['en', 'zh']) {
    for (const [code, en, zh] of [
      ['invalid_guided_parent_dir', 'The parent must be an existing folder', '父目录必须是已存在的文件夹'],
      ['guided_project_folder_create_failed', 'Check that this folder can be written to', '请检查该目录能否写入'],
      ['unfamiliar_code', 'Could not create the study folder', '无法创建研究文件夹'],
    ]) {
      let fail = true;
      const h = harness({ language, create: () => Promise.resolve(fail
        ? { ok: false, blocked: true, error: code, reason: '<server reason>' }
        : { ok: true, draft: { id: 'retry', title: 'Retry study', project_dir: serverRoot + '/guided-retry' } }) });
      await h.sandbox.createLocalGuidedDraft('Synthetic study', 'study', serverRoot);
      assert.equal(h.ctx.guidedFolderDialogMode, 'new');
      assert.match(h.host.innerHTML, /data-folder-create-error[^>]*role="alert"/);
      assert.ok(h.host.innerHTML.includes(language === 'zh' ? zh : en));
      // A known parent refusal reads in the interface language alone; other
      // failures keep the server's reason as an escaped detail line.
      if (code === 'invalid_guided_parent_dir') assert.doesNotMatch(h.host.innerHTML, /server reason/);
      else assert.match(h.host.innerHTML, /gds-status-detail">&lt;server reason>/);
      assert.doesNotMatch(h.host.innerHTML, /data-createdraft[^>]*disabled/);
      assert.equal(h.attempts.length, 0);
      assert.equal(h.bots.length, 0);
      fail = false;
      await h.sandbox.createLocalGuidedDraft('Retry study', 'retry', serverRoot);
      assert.equal(h.ctx.guidedFolderDialogMode, null);
      assert.equal(h.attempts.length, 1);
    }
  }
});

test('missing default Browse remains selectable and explains deferred creation', () => {
  for (const language of ['en', 'zh']) {
    const h = harness({ language });
    h.ctx.guidedFolderBrowser = { open: true, error: 'not_a_directory', data: {
      ok: false, error: 'not_a_directory', path: serverRoot, entries: [], shortcuts: [],
    } };
    h.render();
    assert.doesNotMatch(h.host.innerHTML, /not_a_directory/);
    assert.ok(h.host.innerHTML.includes(language === 'zh' ? '创建研究时会自动创建此默认目录' : 'This default folder will be created with your study'));
    assert.doesNotMatch(h.host.innerHTML, /data-folder-browser-use[^>]*disabled/);
    h.ctx.guidedFolderBrowser.data.path = '/synthetic/arbitrary-absent-parent';
    h.render();
    assert.match(h.host.innerHTML, /not_a_directory/);
    assert.doesNotMatch(h.host.innerHTML, /This default folder will be created with your study/);
  }
});

test('transport failure is inline and keeps the dialog open', async () => {
  const h = harness({ create: () => Promise.reject(new Error('Synthetic network failure')) });
  await h.sandbox.createLocalGuidedDraft('Synthetic study', 'study', serverRoot);
  assert.equal(h.ctx.guidedFolderDialogMode, 'new');
  assert.match(h.host.innerHTML, /data-folder-create-error/);
  assert.match(h.host.innerHTML, /Synthetic network failure/);
  assert.equal(h.attempts.length, 0);
});

test('one-click starter also opens creation feedback on failure', async () => {
  const h = harness({ create: () => Promise.resolve({ ok: false,
    error: 'guided_project_folder_create_failed', reason: 'Synthetic permission failure' }) });
  h.ctx.guidedFolderDialogMode = null;
  await h.sandbox.quickCreateGuidedStarterFolder();
  assert.equal(h.ctx.guidedFolderDialogMode, 'new');
  assert.match(h.host.innerHTML, /data-folder-create-error/);
  assert.match(h.host.innerHTML, /Synthetic permission failure/);
});

test('long server reasons wrap within the folder dialog', () => {
  const css = fs.readFileSync(path.join(js, '../css/guided-projects.css'), 'utf8');
  assert.match(css, /\[data-folder-create-error\]\{[^}]*min-width:0;[^}]*overflow-wrap:anywhere/);
});

test('a study with no title starts with one in the interface language', async () => {
  for (const [language, title] of [['zh', '新研究'], ['en', 'New local study']]) {
    const h = harness({ language, seed: '' });
    h.render();
    assert.ok(h.host.innerHTML.includes(`value="${title}"`), language);
    await h.sandbox.createLocalGuidedDraft('', 'study', serverRoot);
    assert.equal(h.payloads[0].title, title);
  }
  // The shell keeps no English seed of its own.
  assert.doesNotMatch(shell, /guidedFolderSeedTitle = 'New local study'/);
  assert.equal((shell.match(/'New local study'/g) || []).length, 1);
});
