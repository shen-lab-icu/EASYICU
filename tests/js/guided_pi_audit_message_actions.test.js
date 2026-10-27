/* Conversation filenames and suggested questions never acquire authority
   from a later run or from the act of clicking a model-written question. */
'use strict';
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const { test } = require('node:test');

// The optional root lets the same synthetic regressions exercise a read-only base.
const root = path.resolve(process.argv[2] || path.join(__dirname, '../..'));
const js = path.join(root, 'src/easyicu/webserver/static/js');
const shell = fs.readFileSync(path.join(js, 'screens-guided-pi.js'), 'utf8');
function section(start, end) {
  const begin = shell.indexOf(start);
  assert.ok(begin >= 0, start);
  const finish = shell.indexOf(end, begin);
  assert.ok(finish > begin, end);
  return shell.slice(begin, finish);
}

function harness(language = 'zh') {
  const callbacks = {}, opened = [], sent = [], queued = [];
  let progression = 0, collectionLookups = 0, created = 0, creationPause;
  const input = { value: '', focus() {}, scrollIntoView() {}, setSelectionRange() {} };
  const state = {
    session: { session_id: 'session-a', language }, sessionSelectionRevision: 0,
    project: { id: 'project-a' }, messages: [], draft: '', pendingEntryIntent: '',
    busy: false, childJobId: '', currentTurnResources: [], accessMode: 'full',
    latestRun: { run_id: 'run-new' }, workflow: {},
    host: {
      addEventListener: (name, handler) => { callbacks[name] = handler; },
      querySelector: selector => selector === '[data-gpi-input]' ? input : null,
    },
  };
  const context = vm.createContext({
    state, URL, console, window: { EU_LANG: language, requestAnimationFrame: fn => fn() },
    requestAnimationFrame: fn => fn(),
    tr: (en, zh) => language === 'zh' ? zh : en,
    projectId: () => state.project.id,
    sessionIsStale: () => Boolean(state.session?.stale?.stale),
    sessionMatchesUiLanguage: () => true,
    CONNECT: { ask: () => true }, IDEA_SOURCE: { prepareForMessage: async () => null },
    PLAN_ACTIONS: { continueUserRequestedSystemProgression: async () => { progression++; return true; } },
    MESSAGE_QUEUE: { send: async text => queued.push(text) },
    turnGrants: () => ['extract', 'run', 'provider_run'],
    ensureActivity: () => ({}), upsertActivityStep() {}, watchJob() {},
    errorText: String,
    render: () => { input.value = state.draft; },
    api: () => ({ sendPiCopilotMessage: async (id, payload) => { sent.push({ id, ...payload }); return { job_id: 'synthetic-job' }; } }),
    STUDY_WORKSPACE: {
      hasReference: () => false, decorateMessage: text => text, decorateSkillMessage: text => text,
      consume() {}, consumeSkill() {}, removeReference() {}, removeSkill() {}, removeMethod() {},
    },
  });
  for (const name of ['screens-guided-pi-modules.js', 'html-escape.js',
    'screens-guided-pi-markdown.js', 'screens-guided-pi-resources.js', 'screens-guided-pi-access.js', 'screens-guided-pi-events.js']) {
    vm.runInContext(fs.readFileSync(path.join(js, name), 'utf8'), context);
  }
  const modules = context.window.EasyICU.guidedPi;
  context.MODULES = modules;
  context.ACCESS = modules.require('access');
  context.esc = context.window.EU_HTML.esc;
  const resources = modules.require('resources').create({ esc: context.esc });
  // Use the unchanged production descriptor validator; never load file bytes.
  const preview = fs.readFileSync(path.join(js, 'screens-guided-pi-preview.js'), 'utf8');
  vm.runInContext(preview.slice(preview.indexOf('  function safeResource('), preview.indexOf('  function resourceKey(')), context);
  modules.declare('preview', { close() {}, open(descriptor, project) {
    const safe = context.safeResource(descriptor);
    if (!safe) return;
    opened.push({ ...safe, project });
    return true;
  } });
  vm.runInContext(section(shell.includes('  function enhanceWithInteractiveFilePills(')
    ? '  function enhanceWithInteractiveFilePills(' : '  function assistantTextHtml(', '  function api()'), context);
  vm.runInContext(section('  function accessModeLabel(', '  function turnGrants('), context);
  vm.runInContext(section('  async function sendText(', '  async function regenerateMessage('), context);
  vm.runInContext(section('  async function sendMessage(', '  async function confirmCohortEligibility('), context);
  const outcome = {
    collection() { collectionLookups++; return [{ kind: 'research_report', run_id: 'run-new', artifact: 'technical_report.json', sha256: 'b'.repeat(64) }]; },
    followUps: () => [language === 'zh' ? '查看合成分组结果' : 'Review the synthetic group results'],
  };
  modules.require('events').create({
    state, RESOURCE_OWNER: resources, RUN_OUTCOME: outcome,
    RUN_FILES: { handleClick: () => false }, MESSAGE_ACTIONS: { handleClick: () => false },
    STARTERS: { actionFromEvent: () => null }, COHORT_ELIGIBILITY: { actionFromEvent: () => null },
    STUDY_WORKSPACE: context.STUDY_WORKSPACE,
    projectId: context.projectId, previewWorkflowContext: () => ({}),
    render: context.render, tr: context.tr, sendText: context.sendText,
    sendMessage: context.sendMessage, recordHostAction() {}, rememberSession() {},
    createSession: async () => {
      created++; state.session = { session_id: 'session-new', language };
      if (creationPause) await creationPause;
    },
  }).wire();
  const click = (selector, dataset) => callbacks.click({
    preventDefault() {}, stopPropagation() {},
    target: { closest: query => query === selector ? { dataset, closest: () => null } : null },
  });
  const type = text => {
    input.value = text;
    callbacks.input({ target: { value: text, matches: selector => selector === '[data-gpi-input]' } });
  };
  const resourceClick = descriptor => {
    const markup = resources.button(descriptor);
    const dataset = Object.fromEntries([...markup.matchAll(/data-([a-z-]+)="([^"]*)"/g)]
      .map(([, name, value]) => [name.replace(/-([a-z])/g, (_, letter) => letter.toUpperCase()), value]));
    click('[data-gpi-resource-kind]', dataset);
  };
  return { context, state, input, sent, opened, queued, click, type, resourceClick,
    delayCreation: () => { let release; creationPause = new Promise(resolve => { release = resolve; }); return release; },
    progression: () => progression, collectionLookups: () => collectionLookups, created: () => created };
}

for (const language of ['zh', 'en']) {
  test(`${language}: bare filename mentions stay text without a message resource`, () => {
    const h = harness(language);
    const html = h.context.assistantTextHtml('`notes.md` / `technical_report.json` / `summary.csv`');
    assert.match(html, /<code>notes\.md<\/code>/);
    assert.doesNotMatch(html, /<button|data-gpi-file/);
    assert.equal(h.collectionLookups(), 0);
  });

  test(`${language}: C27 labels describe routine actions without promising research approval`, () => {
    const h = harness(language);
    assert.equal(h.context.accessModeLabel('assist'), language === 'zh' ? '常规操作自动确认' : 'Confirm routine actions automatically');
    assert.equal(h.context.accessModeLabel('ask'), language === 'zh' ? '每步先问我' : 'Ask me first');
    // The retired mode reads as the routine default.
    assert.equal(h.context.accessModeLabel('full'), language === 'zh' ? '常规操作自动确认' : 'Confirm routine actions automatically');
  });

  for (const origin of ['model_option', 'host_prompt']) {
    test(`${language}: ${origin} follows draft, edit, explicit send with zero grants`, async () => {
      const h = harness(language);
      h.click(origin === 'model_option' ? '[data-gpi-model-followup]' : '[data-gpi-followup]',
        origin === 'model_option' ? { gpiModelFollowup: 'Explain the synthetic comparison' } : { gpiFollowup: '0' });
      assert.equal(h.sent.length, 0);
      assert.ok(h.state.draft);
      assert.equal(h.input.value, h.state.draft);
      h.type('请开始分析这个合成示例');
      await h.context.sendMessage();
      assert.equal(h.sent.length, 1);
      assert.equal(h.sent[0].message, '请开始分析这个合成示例');
      assert.equal(h.sent[0].message_origin, origin);
      assert.deepEqual(Array.from(h.sent[0].allowed_actions), []);
      assert.equal(h.sent[0].turn_intent, undefined);
      assert.equal(h.progression(), 0, 'a suggestion cannot trigger a privileged host progression');
      assert.equal(h.state.openingQuestion, undefined, 'a suggestion is not a typed opening question');
    });
  }
}

test('old attached resource keeps its run and digest when filenames recur in newer runs', () => {
  const h = harness();
  for (const [run, digest] of [['run-old', 'a'], ['run-new', 'b']]) {
    h.resourceClick({ kind: 'research_report', run_id: run, artifact: 'technical_report.json', sha256: digest.repeat(64) });
    assert.equal(h.opened.at(-1).run_id, run);
    assert.equal(h.opened.at(-1).sha256, digest.repeat(64));
  }
  assert.equal(h.collectionLookups(), 0);
});

test('legacy filename pills and missing descriptors never look up a later run or invent a preview', () => {
  const h = harness();
  for (const filename of ['notes.md', 'technical_report.json']) {
    h.click('[data-gpi-file]', { gpiFile: filename });
  }
  h.resourceClick({ kind: 'research_report', artifact: 'technical_report.json' });
  h.resourceClick({ kind: 'research_report', run_id: 'run-old', artifact: 'technical_report.json', sha256: 'invalid' });
  assert.equal(h.opened.length, 0);
  assert.equal(h.collectionLookups(), 0);
});

test('both follow-up origins leave stale, busy, child-job and absent sessions untouched', () => {
  for (const block of [{ busy: true }, { childJobId: 'synthetic-child' }, { session: null },
    { session: { session_id: 'session-a', stale: { stale: true } } }]) {
    for (const [selector, dataset] of [['[data-gpi-model-followup]', { gpiModelFollowup: 'Explain this example' }],
      ['[data-gpi-followup]', { gpiFollowup: '0' }], ['[data-gpi-followup-new]', { gpiFollowupNew: '0' }]]) {
      const h = harness();
      Object.assign(h.state, block);
      h.click(selector, dataset);
      assert.equal(h.state.draft, '');
      assert.equal(h.sent.length, 0);
      assert.equal(h.created(), 0);
    }
  }
});

test('a drafted model suggestion waits while busy instead of entering a typed-message queue', async () => {
  const h = harness();
  h.click('[data-gpi-model-followup]', { gpiModelFollowup: 'Explain this example' });
  h.state.busy = true;
  await h.context.sendMessage();
  assert.equal(h.queued.length, 0);
  assert.equal(h.sent.length, 0);
  assert.equal(h.state.draft, 'Explain this example');
  h.state.busy = false;
  await h.context.sendMessage();
  assert.equal(h.sent[0].message_origin, 'model_option');
});

test('new-conversation follow-up drafts first and sends with host origin and no grants', async () => {
  const h = harness();
  h.click('[data-gpi-followup-new]', { gpiFollowupNew: '0' });
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(h.created(), 1);
  assert.equal(h.sent.length, 0);
  assert.ok(h.state.draft);
  await h.context.sendMessage();
  assert.equal(h.sent.length, 1);
  assert.equal(h.sent[0].id, 'session-new');
  assert.equal(h.sent[0].message_origin, 'host_prompt');
  assert.deepEqual(Array.from(h.sent[0].allowed_actions), []);
});

test('clearing a suggestion restores ordinary typed-message handling', async () => {
  const h = harness();
  h.click('[data-gpi-model-followup]', { gpiModelFollowup: 'Explain this example' });
  h.type('');
  h.type('A new independently typed question');
  h.state.busy = true;
  await h.context.sendMessage();
  assert.deepEqual(h.queued, ['A new independently typed question']);
});

test('a new-task draft cannot land in another selection, project or a newly blocked session', async () => {
  for (const change of [
    h => { h.state.session = { session_id: 'other' }; h.state.sessionSelectionRevision++; },
    h => { h.state.project.id = 'other-project'; },
    h => { h.state.busy = true; },
    h => { h.state.childJobId = 'child'; },
    h => { h.state.session.stale = { stale: true }; },
  ]) {
    const h = harness(), release = h.delayCreation();
    h.click('[data-gpi-followup-new]', { gpiFollowupNew: '0' });
    change(h);
    release();
    await new Promise(resolve => setImmediate(resolve));
    assert.equal(h.state.draft, '');
    assert.equal(h.sent.length, 0);
  }
});

test('suggested drafts cannot send after their conversation coordinates change', async () => {
  for (const change of [
    h => { h.state.session.session_id = 'other'; },
    h => { h.state.project.id = 'other-project'; },
    h => { h.state.sessionSelectionRevision++; },
  ]) {
    const h = harness();
    h.click('[data-gpi-model-followup]', { gpiModelFollowup: 'Explain this example' });
    change(h);
    await h.context.sendMessage();
    assert.equal(h.sent.length, 0);
    assert.equal(h.progression(), 0);
  }
});

test('suggested drafts include selected materials and skills without acquiring grants', async () => {
  const h = harness();
  h.context.STUDY_WORKSPACE.decorateSkillMessage = text => `${text}\nSelected synthetic method`;
  h.context.STUDY_WORKSPACE.decorateMessage = text => `${text}\nSelected synthetic resource`;
  h.click('[data-gpi-model-followup]', { gpiModelFollowup: 'Explain this example' });
  await h.context.sendMessage();
  assert.equal(h.sent[0].message, 'Explain this example\nSelected synthetic method\nSelected synthetic resource');
  assert.equal(h.sent[0].message_origin, 'model_option');
  assert.deepEqual(Array.from(h.sent[0].allowed_actions), []);
});
