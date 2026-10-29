'use strict';
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const test = require('node:test');
const js = path.resolve(__dirname, '../../src/easyicu/webserver/static/js');
const source = name => fs.readFileSync(path.join(js, `screens-guided-pi-${name}.js`), 'utf8');
const shell = fs.readFileSync(path.join(js, 'screens-guided-pi.js'), 'utf8');
const at = seconds => new Date(Date.UTC(2026, 9, 29) + seconds * 1000).toISOString();
const ms = seconds => Date.parse(at(seconds));
const esc = value => String(value == null ? '' : value).replaceAll('&', '&amp;').replaceAll('<', '&lt;')
  .replaceAll('>', '&gt;').replaceAll('"', '&quot;').replaceAll("'", '&#39;');
function context(names = []) {
  const ctx = vm.createContext({ URL, Date, console, window: { EU_LANG: 'zh', EU_HTML: { esc } } });
  vm.runInContext(source('modules'), ctx);
  names.forEach(name => vm.runInContext(source(name), ctx));
  return ctx;
}
function section(begin, end) {
  const start = shell.indexOf(begin);
  assert.ok(start >= 0, begin);
  const stop = shell.indexOf(end, start);
  assert.ok(stop > start, end);
  return shell.slice(start, stop);
}

test('Host recovery kind selects an explicit bounded action and rejects stale generic replay', () => {
  for (const language of ['zh', 'en']) {
    const ctx = context(['message-actions']);
    const calls = [];
    const rows = [{ id: 'u', role: 'user', entryId: 'u', text: 'Create a synthetic brief' },
      { id: 'a', role: 'assistant', complete: true, errorCode: 'pi_model_provider_unavailable', text: 'Interrupted' }];
    const actions = ctx.window.EasyICU.guidedPi.require('messageActions').create({
      tr: (en, zh) => language === 'zh' ? zh : en, iconHtml: () => '', rows: () => rows,
      regenerate: () => calls.push('unsafe'), continueReply: id => calls.push('explain:' + id),
      retryUnstarted: id => calls.push('retry:' + id),
    });
    const options = { workspaceMode: true, canRetry: true, retryUserEntryId: 'u', canContinueReply: true,
      sourceErrorEntryId: 'failed', recoveryKind: 'retry_unstarted', mutationState: 'not_started', budgetState: 'available' };
    const click = selector => actions.handleClick({ target: { closest: query => query === selector
      ? { closest: () => ({ dataset: { gpiMessageId: 'a' } }) } : null } });
    let html = actions.render(rows[1], options).actionsHtml;
    assert.match(html, language === 'zh' ? /安全重试任务/ : /Retry task safely/);
    assert.match(html, language === 'zh' ? /原模型和剩余预算/ : /original model and remaining budget/);
    assert.doesNotMatch(html, /data-gpi-message-retry(?:\s|>)/);
    click('[data-gpi-message-retry]'); click('[data-gpi-message-continue-reply]');
    assert.deepEqual(calls, []);
    click('[data-gpi-message-retry-unstarted]'); assert.deepEqual(calls, ['retry:failed']);
    html = actions.render(rows[1], { ...options, recoveryKind: 'continue_explanation', mutationState: 'completed' }).actionsHtml;
    assert.match(html, language === 'zh' ? /继续完成说明/ : /Continue explanation/);
    assert.doesNotMatch(html, /data-gpi-message-retry-unstarted/);
    click('[data-gpi-message-retry-unstarted]'); assert.equal(calls.length, 1);
    click('[data-gpi-message-continue-reply]'); assert.deepEqual(calls, ['retry:failed', 'explain:failed']);
    const blocked = actions.render(rows[1], { ...options, canContinueReply: false,
      recoveryKind: 'blocked', mutationState: 'unconfirmed', continuationReasonCode: 'pi_reply_continuation_tool_outcome_unknown' }).actionsHtml;
    assert.match(blocked, language === 'zh' ? /核对文件状态/ : /file state/);
    assert.doesNotMatch(blocked, /data-gpi-message-retry|data-gpi-message-continue/);
    // The default unconfirmed state also appears when workspace recovery is
    // inapplicable; it is not evidence that a file operation was attempted.
    for (const [reason, expected] of [
      ['pi_reply_continuation_workspace_only', language === 'zh' ? /模型回复已中断/ : /model response was interrupted/],
      ['pi_reply_continuation_model_changed', language === 'zh' ? /当前模型与失败时不同/ : /current model differs/],
      ['pi_reply_continuation_budget_unknown', language === 'zh' ? /剩余预算尚无法核对/ : /budget cannot be verified/],
      ['pi_session_connection_changed', language === 'zh' ? /执行状态尚未核实/ : /operation state is unconfirmed/],
    ]) {
      const fallback = actions.render(rows[1], { ...options, canContinueReply: false, recoveryKind: 'blocked',
        mutationState: 'unconfirmed', budgetState: 'unverified', continuationReasonCode: reason }).actionsHtml;
      assert.match(fallback, expected);
      assert.doesNotMatch(fallback, language === 'zh' ? /文件操作可能已经开始/ : /file operation may have started/);
      assert.doesNotMatch(fallback, /data-gpi-message-retry|data-gpi-message-continue/);
    }
    const budget = actions.render(rows[1], { ...options, canContinueReply: false, recoveryKind: 'blocked',
      budgetState: 'exhausted', continuationReasonCode: 'pi_shell_message_provider_call_budget_exhausted' }).actionsHtml;
    assert.match(budget, language === 'zh' ? /预算已用尽/ : /no remaining budget/);
    assert.doesNotMatch(budget, /新的任务预算|新对话|new conversation|reset|重置/i);
    const missing = actions.render(rows[1], { workspaceMode: true, canRetry: true, retryUserEntryId: 'u' }).actionsHtml;
    assert.doesNotMatch(missing, /data-gpi-message-retry/);
    const lateStop = actions.render({ ...rows[1], errorCode: '' }, { workspaceMode: true,
      hostFailedAttempt: true, canRetry: true, retryUserEntryId: 'u', recoveryKind: 'blocked',
      continuationReasonCode: 'pi_reply_continuation_terminal_state_unconfirmed', mutationState: 'unconfirmed' }).actionsHtml;
    assert.doesNotMatch(lateStop, /data-gpi-message-retry/);
    assert.match(lateStop, language === 'zh' ? /晚到回复.*主机任务终态/ : /late model reply.*Host task outcome/);
  }
});

test('terminal timing belongs to the current job and cannot reuse a prior failed attempt', () => {
  const ctx = context();
  const old = { id: 'old', messageJobId: 'A', role: 'activity', status: 'error', startedAt: ms(0), endedAt: ms(23.7), steps: [] };
  const current = { id: 'replacement', role: 'activity', messageJobId: 'B', status: 'running', startedAt: ms(120),
    steps: [{ id: 'thinking-1', kind: 'thinking', status: 'running', startedAt: ms(122) }] };
  ctx.state = { jobId: 'B', messages: [old], regenerating: true, regeneration: { activity: current } };
  ctx.timeMs = value => value ? Date.parse(value) : ms(220);
  vm.runInContext(section('  function activeActivity(', '  function statusBanner('), ctx);
  ctx.finishActivity('error', at(210), 'failed');
  assert.equal(old.endedAt, ms(23.7)); assert.equal(current.endedAt - current.startedAt, 90000);
  assert.equal(current.steps[0].endedAt - current.steps[0].startedAt, 88000);
  ctx.finishActivity('error', at(215), 'failed');
  assert.equal(current.endedAt - current.startedAt, 95000, 'the Host terminal envelope can extend the same completed job');
});

test('SSE callbacks are fenced by source, job and session, with sequence deduplication', async () => {
  const ctx = context(); const streams = [], received = [];
  ctx.state = { session: { session_id: 's' }, jobId: '', sessionSelectionRevision: 1, messages: [],
    busy: true, pendingAuthorityRebind: false };
  ctx.EventSource = class { constructor(url) { this.url = url; streams.push(this); } close() { this.closed = true; } };
  ctx.closeSource = () => { if (ctx.state.source) ctx.state.source.close(); ctx.state.source = null; };
  ctx.projectId = () => 'project'; ctx.handlePiEvent = (event, job) => received.push([event.type, job]);
  ctx.activityHasCompletedAction = () => false; ctx.activityHasSideEffectAttempt = () => false;
  ctx.finishActivity = () => {}; ctx.activeActivity = () => null;
  ctx.assistantRow = () => ({ complete: false, text: '' });
  ctx.tr = (en, zh) => zh; ctx.modelErrorText = String; ctx.ERROR_TEXT = { jobErrorText: String };
  ctx.refreshSession = async () => {}; ctx.MESSAGE_QUEUE = { reconcile: async () => {} };
  ctx.adoptPersistedEntryIds = () => {}; ctx.loadWorkflow = async () => {};
  ctx.PLAN_ACTIONS = { continueSystemOwnedPlanProgression: async () => false };
  ctx.render = () => {}; ctx.api = () => ({ loadJobSnapshot: async () => ({}) });
  vm.runInContext(section('  function watchJob(', '  async function sendText('), ctx);
  ctx.watchJob('A'); const first = streams[0]; ctx.watchJob('B'); const second = streams[1];
  const event = (stream, row) => stream.onmessage({ data: JSON.stringify(row) });
  await event(first, { seq: 0, type: 'pi_event', event: { type: 'run_start' } });
  await event(second, { seq: 0, type: 'pi_event', event: { type: 'run_start' } });
  await event(second, { seq: 0, type: 'pi_event', event: { type: 'run_start' } });
  await event(second, { seq: 1, type: 'pi_event', event: { type: 'thinking_start' } });
  assert.deepEqual(received, [['run_start', 'B'], ['thinking_start', 'B']]);
  ctx.state.sessionSelectionRevision++;
  await event(second, { seq: 2, type: 'end', status: 'failed', error: 'pi_gateway_timeout' });
  assert.equal(ctx.state.busy, true, 'a late terminal from an abandoned selection cannot settle the new selection');
  vm.runInContext(source('regeneration'), ctx);
  ctx.REGENERATION = ctx.window.EasyICU.guidedPi.require('regeneration');
  ctx.state.messages = [{ id: 'u', role: 'user', entryId: 'u' },
    { id: 'activity', role: 'activity', status: 'error', steps: [] },
    { id: 'reply', role: 'assistant', complete: true, text: 'earlier failure' }];
  ctx.state.regenerating = true;
  ctx.state.regeneration = ctx.REGENERATION.create(ctx.state.messages, { userEntryId: 'u', targetMessageId: 'reply', startedAt: ms(120) });
  let release;
  ctx.refreshSession = () => new Promise(resolve => { release = resolve; });
  ctx.watchJob('C');
  const terminal = event(streams[2], { seq: 0, type: 'end', status: 'failed', error: 'pi_gateway_timeout', timestamp: at(210) });
  assert.equal(ctx.state.regenerating, false, 'the terminal replacement is frozen before an async refresh');
  assert.equal(ctx.state.messages.at(-1).errorCode, 'pi_gateway_timeout');
  const newer = { activity: { steps: [] }, message: {} };
  ctx.state.regenerating = true; ctx.state.regeneration = newer; ctx.state.jobId = 'D';
  release(); await terminal;
  assert.equal(ctx.state.regeneration, newer, 'late cleanup cannot clear a newer replacement attempt');
});

test('replayed resend steps and a long terminal wait are isolated from the original 23.7-second failure', () => {
  const ctx = context(['activity', 'transcript']);
  ctx.window.EasyICU.guidedPi.declare('replay', { lifecycleTurns: session => session.conversation_replay.turns });
  const activity = ctx.window.EasyICU.guidedPi.require('activity').create({ tr: (_en, zh) => zh,
    esc, iconHtml: () => '', resourceName: () => '', resourceKey: () => '', resourceButton: () => '' });
  const upsert = (row, step) => { const found = row.steps.find(item => item.id === step.id);
    if (found) Object.assign(found, step); else row.steps.push(step); };
  const transcript = ctx.window.EasyICU.guidedPi.require('transcript').create({ tr: (_en, zh) => zh,
    activity, timeMs: Date.parse, upsertActivityStep: upsert, resourceKey: () => '',
    modelErrorText: code => code, activityHasCompletedAction: () => false });
  const session = { agent_mode: 'workspace', transcript: [
    { role: 'user', timestamp: at(0), entry_id: 'u', content: [{ type: 'text', text: 'Same original request' }] },
    { role: 'assistant', timestamp: at(121), content: [{ type: 'thinking' },
      { type: 'tool_call', tool_call_id: 'same-tool-id', tool_name: 'easyicu_read_project_file' }] },
    { role: 'tool', timestamp: at(122), content: [{ type: 'tool_result', tool_call_id: 'same-tool-id', tool_name: 'easyicu_read_project_file' }] },
    { role: 'assistant', timestamp: at(125), entry_id: 'new-failed', error_code: 'pi_gateway_timeout', content: [] },
  ], prior_failed_attempts: [{ job_id: 'A', status: 'failed', started_at: at(0), ended_at: at(23.7), events: [
    { type: 'run_start', at: at(0) }, { type: 'assistant_start', at: at(0.1) },
    { type: 'thinking_start', at: at(0.2) }, { type: 'thinking_end', at: at(23.5) },
    { type: 'message_end', at: at(23.7), error_code: 'pi_model_provider_unavailable' },
  ] }], conversation_replay: { turns: [{ job_id: 'B', status: 'failed', started_at: at(120), ended_at: at(210),
    assistant_entry_id: 'new-failed', events: [
      { type: 'run_start', at: at(120) }, { type: 'assistant_start', at: at(121) },
      { type: 'tool_start', at: at(121.1), tool_call_id: 'same-tool-id', tool_name: 'easyicu_read_project_file' },
      { type: 'tool_end', at: at(122), tool_call_id: 'same-tool-id', tool_name: 'easyicu_read_project_file' },
      { type: 'assistant_start', at: at(122) }, { type: 'thinking_start', at: at(122) },
    ] }] } };
  const result = transcript.transcriptMessages(session);
  const old = result.find(row => row.messageJobId === 'A');
  const current = result.find(row => row.role === 'activity' && row.messageJobId === 'B');
  assert.ok(old.supersededAttempt); assert.equal(old.endedAt - old.startedAt, 23700);
  assert.equal(current.endedAt - current.startedAt, 90000);
  assert.equal(current.steps.filter(step => step.kind === 'tool').length, 1);
  assert.equal(current.steps.filter(step => step.kind === 'assistant').length, 2, 'SDK marker phases are not added twice');
  assert.equal(current.steps.find(step => step.kind === 'thinking').endedAt - ms(122), 88000);
  assert.match(activity.render(current), /总耗时 1 分 30 秒/);
  assert.doesNotMatch(activity.render(current), /总耗时 23.7 秒/);
  assert.match(activity.render(old), /已被重发替代/);
  assert.deepEqual(Array.from(old.allowedActions), []);
});

test('a completed workspace retry remains protected from generic regeneration after reload', () => {
  const ctx = context(['activity', 'transcript', 'message-actions']);
  ctx.window.EasyICU.guidedPi.declare('replay', { lifecycleTurns: session => session.conversation_replay.turns });
  const activity = ctx.window.EasyICU.guidedPi.require('activity').create({ tr: en => en,
    esc, iconHtml: () => '', resourceName: () => '', resourceKey: () => '', resourceButton: () => '' });
  const owner = ctx.window.EasyICU.guidedPi.require('transcript').create({ activity, timeMs: Date.parse,
    resourceKey: () => '', upsertActivityStep: (row, step) => row.steps.push(step) });
  const rows = owner.transcriptMessages({ agent_mode: 'workspace', transcript: [
    { role: 'user', timestamp: at(0), entry_id: 'u', content: [{ type: 'text', text: 'Original task' }] },
    { role: 'assistant', timestamp: at(8), entry_id: 'recovered', content: [{ type: 'text', text: 'Saved once' }] },
  ], conversation_replay: { turns: [{ kind: 'workspace_retry', job_id: 'retry', status: 'done',
    assistant_entry_id: 'recovered', started_at: at(5), ended_at: at(10), events: [
      { type: 'run_start', at: at(5) }, { type: 'message_end', stop_reason: 'stop', at: at(10) },
    ] }] } });
  const reply = rows.find(row => row.entryId === 'recovered');
  assert.equal(reply.replyContinuation, true);
  const actions = ctx.window.EasyICU.guidedPi.require('messageActions').create({
    tr: en => en, iconHtml: () => '', rows: () => rows,
  });
  assert.doesNotMatch(actions.render(reply, { canRetry: true, retryUserEntryId: 'u' }).actionsHtml, /data-gpi-message-retry/);
  assert.match(actions.render(rows.find(row => row.entryId === 'u'), { allowEdit: true, canEdit: true }).actionsHtml,
    /data-gpi-message-edit/);
});

test('the explicit artifact entry preserves the draft and selects workspace without sending', () => {
  const ctx = context(['events']); const calls = []; let click;
  const state = { draft: 'Create a synthetic 24-row brief', messages: [],
    host: { addEventListener: (name, fn) => { if (name === 'click') click = fn; }, querySelector: () => null } };
  ctx.window.EasyICU.guidedPi.require('events').create({ state,
    tr: en => en, RUN_FILES: { handleClick: () => false }, MESSAGE_ACTIONS: { handleClick: () => false },
    STARTERS: { actionFromEvent: () => null }, COHORT_ELIGIBILITY: { actionFromEvent: () => null },
    STUDY_WORKSPACE: { removeReference() {}, removeSkill() {}, removeMethod() {} },
    switchMode: (...args) => calls.push(args), sendText: () => calls.push('sent'),
  }).wire();
  click({ target: { closest: selector => selector === '[data-gpi-new-workspace]' ? {} : null },
    preventDefault() {}, stopPropagation() {} });
  assert.equal(calls.length, 1); assert.equal(calls[0][0], 'workspace');
  assert.equal(calls[0][1].newSession, true); assert.equal(calls[0][1].draft, state.draft);
});

test('failed workspace tool arguments never become artifacts while confirmed earlier files remain available', () => {
  const ctx = context(['activity', 'transcript', 'resources', 'markdown']);
  ctx.window.EasyICU.guidedPi.declare('replay', { lifecycleTurns: () => [] });
  const resources = ctx.window.EasyICU.guidedPi.require('resources').create({ esc });
  const owner = ctx.window.EasyICU.guidedPi.require('transcript').create({
    activity: { focusLatest: rows => rows }, timeMs: Date.parse, resourceKey: resources.key, mergeResources: resources.merge,
    upsertActivityStep: (row, step) => { const old = row.steps.find(item => item.id === step.id);
      if (old) Object.assign(old, step); else row.steps.push(step); },
  });
  const actual = { kind: 'file', file: 'saved-before.txt', sha256: 'a'.repeat(64) };
  const predicted = { kind: 'file', file: 'not-created.html', media_type: 'text/html' };
  const rows = owner.transcriptMessages({ agent_mode: 'workspace', transcript: [
    { role: 'user', timestamp: at(0), content: [{ type: 'text', text: 'Create brief' }] },
    { role: 'tool', timestamp: at(1), content: [{ type: 'tool_result', tool_call_id: 'read',
      tool_name: 'easyicu_read_project_file', resource: actual }] },
    { role: 'assistant', timestamp: at(2), content: [{ type: 'tool_call', tool_call_id: 'write',
      tool_name: 'easyicu_write_project_file', resource: predicted }] },
    { role: 'tool', timestamp: at(3), content: [{ type: 'tool_result', tool_call_id: 'write',
      tool_name: 'easyicu_write_project_file', is_error: true, code: 'pi_workspace_write_authorization_required' }] },
    { role: 'assistant', timestamp: at(4), content: [{ type: 'text', text: '[Not created](not-created.html)' }] },
  ] });
  const answer = rows.find(row => row.role === 'assistant' && row.text);
  assert.equal(answer.resources.length, 1); assert.equal(answer.resources[0].file, actual.file);
  assert.doesNotMatch(resources.renderForMessage(answer), /not-created\.html/);
  const text = ctx.window.EasyICU.guidedPi.require('markdown').render(answer.text, { resources: answer.resources });
  assert.doesNotMatch(text, /data-gpi-resource/);
  assert.match(text, /not-created\.html/);
});
