'use strict';

// A draft is a read-only projection of completed assistant messages. It never
// confirms scientific settings or replaces the governed plan/review owners.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');
const JS = path.resolve(__dirname, '../../src/easyicu/webserver/static/js');
const esc = value => String(value).replace(/[&<>"']/g, char => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[char]));
const tr = (en, zh) => zh;
function draft(overrides = {}) {
  return { schema: 'easyicu.research-draft/1', title: '研究讨论草案', question: '比较入院后指标变化与结局的关系',
    population: '首次入住 ICU 的成人（拟议）', exposure: '待核对的指标', outcome: '院内死亡（候选）',
    candidate_data: '尚未选定，需比较数据库可用性', approach: '讨论设计与偏倚，尚未核对真实数据',
    data_requirements: ['ICU 入出时间', '指标及测量时间'], open_questions: ['时间零点的定义'], changes: ['首次整理'], ...overrides };
}
const block = value => '```easyicu-research-draft\n' + JSON.stringify(value) + '\n```';

test('source limits display only for the current study, revision and registered source', () => {
  const { owner } = harness();
  const snapshot = owner.current([{ role: 'assistant', complete: true, text: block(draft()) }]);
  const fact = { schema: 'easyicu.source-boundary-discussion/1', status: 'discussion_only_not_approved_not_executable',
    database: 'sic', study_context_id: 'study-a', study_revision: 3, registration_id: 'src-a',
    start_column: 'ICUOffset', end_column: 'TimeOfStay' };
  const receipt = { study_context_id: 'study-a', revision: 3,
    configuration: { data_source: { database: 'sic', source_id: 'src-a' } } };
  const render = workflow => owner.render(snapshot, { tr, esc, workflow });
  assert.match(render({ source_boundary_review: fact, study_setup_receipt: receipt }), /当前数据来源的限制/);
  assert.match(render({ source_boundary_review: fact, study_setup_receipt: receipt }), /讨论草案未记录成稿时的数据来源身份/);
  for (const wrong of [
    { ...receipt, study_context_id: 'study-other' }, { ...receipt, revision: 4 },
    { ...receipt, configuration: { data_source: { database: 'sic', source_id: 'src-other' } } },
    { ...receipt, configuration: { data_source: { database: 'miiv', source_id: 'src-a' } } },
  ]) assert.doesNotMatch(render({ source_boundary_review: fact, study_setup_receipt: wrong }), /当前数据来源的限制/);
  assert.doesNotMatch(render({ study_setup_receipt: receipt }), /当前数据来源的限制/);
});
function harness() {
  const body = { innerHTML: '', querySelector: () => null };
  const head = { innerHTML: '' };
  const aside = { querySelector: () => head };
  const window = { EU_LANG: 'zh', EU_HTML: { esc }, localStorage: { getItem: () => null }, EU_API: {} };
  const document = { getElementById: id => id === 'gdAsideBody' ? body : id === 'gdStudyAside' ? aside : null, querySelector: () => null };
  const context = vm.createContext({ window, document, setTimeout, clearTimeout });
  for (const file of ['screens-guided-pi-modules.js', 'screens-guided-pi-research-draft.js', 'screens-guided-pi-data-consent.js', 'screens-guided-pi-aside.js', 'screens-guided-pi-message-actions.js', 'screens-guided-pi-session-view.js', 'screens-guided-pi-replay.js', 'screens-guided-pi-transcript.js']) {
    vm.runInContext(fs.readFileSync(path.join(JS, file), 'utf8'), context, { filename: file });
  }
  return { registry: window.EasyICU.guidedPi, owner: window.EasyICU.guidedPi.require('researchDraft'), body, window };
}

test('copy removes only metadata actually lifted for the same message and preserves the rest', async () => {
  const { registry, owner, window } = harness();
  let copied = '';
  window.navigator = { clipboard: { writeText: value => { copied = value; return Promise.resolve(); } } };
  registry.declare('nextActions', { project: () => null });
  let row = { id: 'a', role: 'assistant', complete: true, text: '' };
  const actions = registry.require('messageActions').create({ tr, iconHtml: () => '', rows: () => [row] });
  const view = registry.require('sessionView').create({
    state: {}, modules: registry, tr, esc, projectId: () => 'p1', iconHtml: () => '',
    studyWorkspace: { messageView: value => ({ text: value.text }) },
    resourceOwner: { renderForMessage: () => '' }, messageActions: actions,
    runFiles: { render: () => '' }, publicAssistantText: text => text,
    assistantTextHtml: (text, resources, options) => esc(options.liftDraft ? owner.strip(text) : text),
  });
  const click = () => actions.handleClick({ target: { closest: query => query === '[data-gpi-message-copy]'
    ? { dataset: {}, setAttribute: () => {}, closest: () => ({ dataset: { gpiMessageId: 'a' } }) } : null } });
  const valid = block(draft());
  const ordinary = '```json\n{"example":true}\n```';
  const invalid = '```easyicu-research-draft\n{bad}\n```';
  row.text = '  正文\n' + valid + '\n下一步\n' + ordinary + '\n' + invalid + '\n追问  \n';
  const original = row.text;
  view.messageHtml(row, {}); click(); await Promise.resolve();
  assert.equal(copied, original.replace(valid, ''));
  assert.equal(row.text, original);
  // Failed host attempts do not lift even a syntactically valid draft.
  view.messageHtml(row, { hostFailedAttempt: true }); click(); await Promise.resolve();
  assert.equal(copied, original);
  for (const change of [{ role: 'user' }, { complete: false }, { errorCode: 'failed' }, { stopReason: 'interrupted' }]) {
    row = { ...row, role: 'assistant', complete: true, errorCode: '', stopReason: '', ...change };
    view.messageHtml(row, {}); click(); await Promise.resolve();
    assert.equal(copied, original);
  }
  row = { id: 'a', role: 'assistant', complete: true, text: original };
  view.messageHtml(row, {});
  row.text += ' changed after rendering'; click(); await Promise.resolve();
  assert.equal(copied, row.text);
  assert.equal(owner.stripCopied(original, original.replace(valid, invalid)), original);
  assert.equal(owner.stripCopied(valid + '\n' + valid, valid), '\n' + valid);
  const realReply = fs.readFileSync(path.resolve(__dirname, '../fixtures/pi_copilot/research_draft_json_fence_20261029.txt'), 'utf8');
  assert.ok(owner.read(realReply));
  assert.doesNotMatch(owner.stripCopied(realReply, realReply), /easyicu\.research-draft\/1/);
  const incomplete = '正文\n```easyicu-research-draft\n{';
  row.text = incomplete; view.messageHtml(row, {}); click(); await Promise.resolve();
  assert.equal(copied, incomplete);
});

test('valid blocks are lifted; normal prose and malformed/foreign blocks remain visible', () => {
  const { owner } = harness();
  const source = '设计说明\n' + block(draft()) + '\n\n## 下一步\n继续讨论';
  assert.equal(owner.read(source).title, '研究讨论草案');
  assert.equal(owner.strip(source), '设计说明\n\n\n## 下一步\n继续讨论');
  for (const value of [draft({ schema: 'other/1' }), draft({ authority: 'approved' }), draft({ title: '' }),
    draft({ outcome: 1 }), draft({ changes: 'changed' }), draft({ changes: ['x'.repeat(301)] }),
    draft({ question: 'x'.repeat(1201) }), draft({ open_questions: Array(11).fill('x') }), draft({ exposure: 'a\u0001b' })]) {
    assert.equal(owner.read(block(value)), null);
    assert.equal(owner.strip(block(value)), block(value));
  }
  const missing = draft(); delete missing.open_questions;
  assert.equal(owner.read(block(missing)), null);
  const oversized = draft({ data_requirements: Array(12).fill('x'.repeat(300)), open_questions: Array(10).fill('y'.repeat(300)), changes: Array(10).fill('z'.repeat(300)) });
  assert.equal(owner.read(block(oversized)), null);
  assert.equal(owner.strip(block(oversized)), block(oversized));
  const malformed = '```easyicu-research-draft\n{bad}\n```';
  assert.equal(owner.strip(malformed), malformed);
  const partial = block(draft()).slice(0, -2);
  assert.equal(owner.read(partial), null);
  assert.equal(owner.strip(partial), partial);
});

test('only completed assistant messages can project a draft; later user turns mark it pending', () => {
  const { owner } = harness();
  assert.equal(owner.current([{ role: 'user', text: block(draft()) }]), null);
  assert.equal(owner.current([{ role: 'assistant', complete: false, text: block(draft()) }]), null);
  assert.equal(owner.current([{ role: 'assistant', errorCode: 'failed', text: block(draft()) }]), null);
  assert.equal(owner.current([{ role: 'assistant', hostFailedAttempt: true, text: block(draft()) }]), null);
  for (const stopReason of ['error', 'aborted', 'cancelled', 'interrupted']) {
    assert.equal(owner.current([{ role: 'assistant', complete: true, stopReason, text: block(draft()) }]), null);
    assert.equal(owner.current([{ role: 'assistant', complete: true, stop_reason: stopReason, text: block(draft()) }]), null);
  }
  const message = { role: 'assistant', complete: true, entryId: 'stable-a', timelineAt: 1793242800000, text: block(draft()) };
  const latest = { role: 'assistant', complete: true, entryId: 'stable-b', timelineAt: 1793242900000,
    text: block(draft({ outcome: 'ICU 住院时长（候选）', changes: ['主要结局由院内死亡改为 ICU 住院时长；数据库仍未绑定'] })) };
  const rows = [message, { role: 'user', text: '将结局改为 ICU 住院时长' }, latest];
  const projected = owner.current(rows);
  assert.equal(projected.draft.outcome, 'ICU 住院时长（候选）');
  assert.equal(projected.history.length, 1);
  assert.equal(projected.pendingUpdate, false);
  assert.equal(owner.current([...rows, { role: 'user', text: '再修改人群' }]).pendingUpdate, true);
  const refreshed = rows.map((row, i) => ({ ...row, id: 'history-' + i }));
  assert.equal(JSON.stringify(owner.current(refreshed)), JSON.stringify(projected));
  const paged = owner.current([latest]);
  assert.equal(paged.draft.outcome, projected.draft.outcome);
  assert.equal(paged.at, projected.at);
  assert.doesNotMatch(owner.render(paged, { tr, esc }), /版本\s*\d|Version\s*\d/);
  assert.equal(owner.current([...rows, latest]).history.length, 1);
});

test('all model text is escaped and the card names its unapproved boundary', () => {
  const { owner } = harness();
  const attack = '<img src=x onerror=alert(1)>';
  const value = draft(Object.fromEntries(['title', 'question', 'population', 'exposure', 'outcome', 'candidate_data', 'approach'].map(key => [key, attack])));
  for (const key of ['data_requirements', 'open_questions', 'changes']) value[key] = [attack];
  const snapshot = owner.current([{ role: 'assistant', text: block(value), complete: true, entryId: '1', timelineAt: 1793242800000 }]);
  const html = owner.render({ ...snapshot, history: [{ draft: value }] }, { tr, esc });
  assert.doesNotMatch(html, /<img/);
  assert.match(html, /&lt;img src=x onerror=alert\(1\)&gt;/);
  assert.match(html, /讨论草案 · 未批准/);
  assert.match(html, /不代表批准准备数据或执行分析/);
  assert.match(html, /待定事项/);
  assert.match(html, /本次草案变更/);
  assert.match(html, /草案更新：/);
  assert.doesNotMatch(html, /data-gpi-.*(?:approve|execute|configure|plan-action)/);
  assert.equal(owner.render(null, { tr, esc }), '');
});

test('pending source offers draft discussion without adding any grant; actual source controls remain', () => {
  const { registry } = harness();
  const owner = registry.require('dataConsent');
  const pending = { data_source_authorization: { status: 'pending', reason: 'local_data_selection_required' } };
  const before = JSON.stringify(pending);
  const html = owner.render(pending, { tr, esc, icon: () => '' });
  assert.match(html, /继续讨论草案/);
  assert.match(html, /data-gpi-next-choice=/);
  assert.match(html, /可稍后/);
  assert.match(html, /data-gpi-data-source-action="begin_local_selection"/);
  assert.doesNotMatch(html, /data-gpi-(?:grants|run|configure|approve)/);
  assert.equal(JSON.stringify(pending), before);
  const named = { key: 'miiv', label: 'MIMIC-IV v3.1' };
  assert.doesNotMatch(owner.render(pending, { tr, esc, icon: () => '', namedDatabase: named }), /继续讨论草案/);
  const drafted = owner.render(pending, { tr, esc, icon: () => '', namedDatabase: named, discussionDraft: {} });
  assert.doesNotMatch(drafted, /继续讨论草案|data-gpi-next-choice=/);
  assert.match(drafted, /选择用于核验的数据 · 可稍后/);
  assert.match(drafted, /data-gpi-data-source-action="begin_local_selection"/);
  assert.doesNotMatch(drafted, /data-gpi-(?:grants|run|configure|approve)/);
  const projectSource = { data_source_authorization: { status: 'pending', reason: 'project_source_confirmation_required',
    source: { label: 'Example ICU' } } };
  const projectHtml = owner.render(projectSource, { tr, esc, icon: () => '', discussionDraft: {} });
  assert.match(projectHtml, /选择用于核验的数据 · 可稍后/);
  assert.match(projectHtml, /data-gpi-data-source-action="use_study_required_data"/);
  assert.match(projectHtml, /选择数据源不等于批准分析/);
  assert.equal(projectSource.data_source_authorization.status, 'pending');
  assert.equal(owner.requiresConfirmation(pending), true);
  assert.equal(owner.requiresConfirmation({ data_source_authorization: { status: 'confirmed' } }), false);
});

test('aside keeps formal plan review and active tasks visible beside the advisory draft', () => {
  const { registry, body } = harness();
  const rows = [{ role: 'assistant', complete: true, text: block(draft()), entryId: 'draft' }];
  let pending = false, activeTask = '', planStatus = 'locked';
  const workflow = () => ({ current_stage: 'setup', missing_setup_fields: ['data_source'], stages: [
    { id: 'question', status: 'complete' }, { id: 'setup', status: 'ready', reason_code: 'study_setup_incomplete' },
    { id: 'plan', status: planStatus },
  ] });
  const host = { tr, esc, iconHtml: () => '', projectId: () => 'p1', displayProjectTitle: () => 'Project',
    demoMode: () => false, shell: () => 'pi', workflow, messages: () => rows,
    hasPendingReview: () => pending, activeTaskTitle: () => activeTask };
  const instance = registry.require('aside').create(host);
  instance.syncProjectWorkflowAside();
  assert.match(body.innerHTML, /研究讨论草案/);
  assert.match(body.innerHTML, /class="gpi-formal-progress"><summary>/);
  assert.match(body.innerHTML, /data-gpi-project-workflow-list/);
  pending = true; instance.syncProjectWorkflowAside();
  assert.match(body.innerHTML, /class="gpi-formal-progress" open/);
  assert.match(body.innerHTML, /data-gpi-aside-pending/);
  pending = false; planStatus = 'review_required'; instance.syncProjectWorkflowAside();
  assert.match(body.innerHTML, /class="gpi-formal-progress" open/);
  planStatus = 'locked'; activeTask = '正在执行已批准任务'; instance.syncProjectWorkflowAside();
  assert.match(body.innerHTML, /class="gpi-formal-progress" open/);
  rows.length = 0; instance.syncProjectWorkflowAside();
  assert.doesNotMatch(body.innerHTML, /data-gpi-research-draft/);
  assert.doesNotMatch(body.innerHTML, /gpi-formal-progress/);
});


test('message rendering does not claim or swallow draft metadata in incomplete or failed replies', () => {
  const { registry, owner } = harness();
  registry.declare('nextActions', { project: () => null });
  const view = registry.require('sessionView').create({
    state: {}, modules: registry, tr, esc, projectId: () => 'p1', iconHtml: () => '',
    studyWorkspace: { messageView: row => ({ text: row.text }) },
    resourceOwner: { renderForMessage: () => '', button: () => '' },
    messageActions: { render: () => ({ actionsHtml: '', editorHtml: '' }) },
    runFiles: { render: () => '' }, publicAssistantText: text => text,
    assistantTextHtml: (text, resources, options) => {
      if (options.streaming) {
        const projected = owner.streaming(text);
        return esc(projected.text) + (projected.preparing ? '<span role="status">正在整理讨论草案</span>' : '');
      }
      return esc(options.liftDraft ? owner.strip(text) : text);
    },
  });
  const row = { role: 'assistant', complete: true, text: block(draft()), entryId: 'm1' };
  assert.match(view.messageHtml(row, {}), /本条回复整理了一份研究讨论草案/);
  assert.doesNotMatch(view.messageHtml(row, {}), /```easyicu-research-draft/);
  for (const [reply, options] of [
    [{ ...row, errorCode: 'cancelled' }, {}],
    [{ ...row, stopReason: 'aborted' }, {}], [{ ...row, stopReason: 'error' }, {}],
    [row, { hostFailedAttempt: true }],
  ]) {
    const html = view.messageHtml(reply, options);
    assert.match(html, /```easyicu-research-draft/);
    assert.doesNotMatch(html, /本条回复整理了一份研究讨论草案/);
  }
  const streaming = view.messageHtml({ ...row, complete: false }, {});
  assert.match(streaming, /正在整理讨论草案/);
  assert.doesNotMatch(streaming, /```easyicu-research-draft|本条回复整理了一份研究讨论草案/);
  assert.match(view.messageHtml({ ...row, role: 'user' }, {}), /```easyicu-research-draft/);
});


test('history refresh preserves failed terminal reasons instead of adopting an interrupted draft', () => {
  const { registry, owner } = harness();
  const transcript = registry.require('transcript').create({
    tr, activity: { focusLatest: rows => rows }, timeMs: Date.parse, resourceKey: () => '',
    upsertActivityStep: () => {}, modelErrorText: () => '', workflowActionCode: () => '',
  });
  for (const reason of ['aborted', 'error']) {
    const history = transcript.transcriptMessages({ transcript: [
      { role: 'assistant', entry_id: 'good', timestamp: '2026-10-29T03:00:00Z', stop_reason: 'stop', content: [{ type: 'text', text: block(draft()) }] },
      { role: 'assistant', entry_id: 'stopped', timestamp: '2026-10-29T03:01:00Z', stop_reason: reason, content: [{ type: 'text', text: block(draft({ title: '中止回复中的草案' })) }] },
    ] });
    assert.equal(history.find(row => row.entryId === 'stopped').stopReason, reason);
    assert.equal(owner.current(history).draft.title, '研究讨论草案');
  }
});


test('real first UAT reply with json fence lifts only after the exact schema validates', () => {
  const { owner } = harness();
  const source = fs.readFileSync(path.resolve(__dirname, '../fixtures/pi_copilot/research_draft_json_fence_20261029.txt'), 'utf8');
  const parsed = owner.read(source);
  assert.equal(parsed.schema, 'easyicu.research-draft/1');
  assert.equal(parsed.title, '入ICU后24小时内乳酸变化与院内死亡的关联');
  assert.match(parsed.population, /首次ICU住院.*尚未决定/);
  assert.equal(owner.current([{ role: 'assistant', complete: true, text: source, entryId: 'uat-1' }]).draft.title, parsed.title);
  const visible = owner.strip(source);
  assert.doesNotMatch(visible, /```json|"schema"|"candidate_data"/);
  assert.match(visible, /需要你决定的关键点/);
  assert.match(visible, /下一步/);
  const asJson = value => '```json\n' + JSON.stringify(value) + '\n```';
  for (const value of [{ outcome: 'ordinary JSON' }, draft({ schema: 'other/1' }), draft({ extra: 'not accepted' }), draft({ outcome: 10 }), draft({ changes: ['x'.repeat(301)] })]) {
    const text = asJson(value);
    assert.equal(owner.read(text), null);
    assert.equal(owner.strip(text), text);
  }
  assert.equal(owner.read(source.replace('"schema": "easyicu.research-draft/1"', '"schema": "other/1"')), null);
  assert.equal(owner.read(source.replace('"changes": []', '"changes": "invalid"')), null);
});

test('streaming schema metadata becomes a temporary status, never a completed or authoritative draft', () => {
  const { owner } = harness();
  const source = fs.readFileSync(path.resolve(__dirname, '../fixtures/pi_copilot/research_draft_json_fence_20261029.txt'), 'utf8');
  const schemaEnd = source.indexOf('"title"');
  const partial = source.slice(0, schemaEnd + 14);
  const projected = owner.streaming(partial);
  assert.equal(projected.preparing, true);
  assert.doesNotMatch(projected.text, /```json|"schema"|"title"/);
  assert.match(projected.text, /需要你决定的关键点/);
  assert.equal(owner.read(partial), null);
  assert.equal(owner.strip(partial), partial.trim(), 'a final malformed reply retains its raw contents');
  assert.equal(owner.current([{ role: 'assistant', complete: false, text: source }]), null);
  assert.equal(owner.isStreaming({ role: 'assistant', complete: false, text: source }), true);
  for (const reply of [
    { role: 'assistant', complete: true }, { role: 'user', complete: false },
    { role: 'assistant', complete: false, errorCode: 'failed' },
    { role: 'assistant', complete: false, stopReason: 'aborted' },
  ]) assert.equal(owner.isStreaming(reply), false);
  const ordinary = '```json\n{"count":1}\n```';
  assert.equal(owner.streaming(ordinary).text, ordinary);
  assert.equal(owner.streaming(ordinary).preparing, false);
  assert.equal(owner.streaming('```json\n{"schema":"other/1",').preparing, false);
  assert.equal(owner.streaming('```easyicu-research-draft\n{').preparing, true);
});
