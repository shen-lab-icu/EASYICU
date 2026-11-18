'use strict';
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const { test } = require('node:test');

const root = path.resolve(__dirname, '../../src/easyicu/webserver');
const finalizerPath = path.join(root, 'pi_copilot/node_app/src/post-tool-finalization.mjs');
// Only the SDK event queue is replaced; every routing/finalization function
// comes from the owner, so these offline tests also run without node_modules.
const source = fs.readFileSync(finalizerPath, 'utf8')
  .replace(/^import \{ createAssistantMessageEventStream \} from "@earendil-works\/pi-ai";\n/, '')
  .replace(/^export /gm, '');
const sandbox = { createAssistantMessageEventStream: () => {
  const events = [];
  return { push: event => events.push(event), async *[Symbol.asyncIterator]() { yield* events; } };
} };
vm.runInNewContext(source + '\nthis.owner = {settleQuickAnswerOffer, hostPostToolFinalization, quickAnswerReadingNeedsRepair, quickAnswerOfferRetryAvailable: '
  + 'typeof quickAnswerOfferRetryAvailable === "function" ? quickAnswerOfferRetryAvailable : () => false};', sandbox);
const { settleQuickAnswerOffer, hostPostToolFinalization, quickAnswerReadingNeedsRepair, quickAnswerOfferRetryAvailable } = sandbox.owner;
const model = { api: 'test', provider: 'synthetic', id: 'scripted' };
const ack = '已确认使用模拟数据。';
const message = (stopReason, content = []) => ({ role: 'assistant', stopReason, content, usage: { input: 3 } });
const stream = events => ({ async *[Symbol.asyncIterator]() { yield* events; } });
const collect = async value => { const events = []; for await (const event of value) events.push(event); return events; };
const host = { type: 'custom_message', customType: 'easyicu_host_transition',
  content: '[EASYICU_DATA_SOURCE_CONFIRMED_V1]\n[EASYICU_QUICK_ANSWER_OFFER_V1]',
  details: { intent: 'advance_after_data_source_confirmation' } };

test('an exhausted quick-answer decision remains failed, including its usage', async () => {
  const failure = { ...message('error'), errorMessage: '503: synthetic unavailable' };
  const events = await collect(await settleQuickAnswerOffer(model, () => stream([
    { type: 'error', reason: 'error', error: failure },
  ]), ack));
  assert.equal(events.at(-1).type, 'error');
  assert.equal(events.at(-1).error, failure);
  assert.equal(events.at(-1).error.stopReason, 'error');
  assert.ok(!JSON.stringify(events).includes(ack));
});

test('transport exceptions and missing terminal events cannot become a successful decline', async () => {
  const thrown = new Error('503: synthetic unavailable');
  await assert.rejects(settleQuickAnswerOffer(model, () => { throw thrown; }, ack), error => error === thrown);
  const incomplete = { ...stream([{ type: 'done', message: undefined }]), result: async () => undefined };
  await assert.rejects(settleQuickAnswerOffer(model, () => incomplete, 'The synthetic export is selected.'), {
    name: 'Error', message: 'Quick-answer decision ended without a terminal message',
  });
});

test('a failed partial tool call is never replayed', async () => {
  const failure = message('error', [{ type: 'toolCall', name: 'easyicu_quick_answer', arguments: {} }]);
  const events = await collect(await settleQuickAnswerOffer(model, () => stream([
    { type: 'error', error: failure },
  ]), ack));
  assert.deepEqual(events.map(event => event.type), ['error']);
  assert.equal(events[0].error.stopReason, 'error');
});

test('a successful model decline still permits the full-study path', async () => {
  const events = await collect(await settleQuickAnswerOffer(model, () => stream([
    { type: 'done', message: message('stop', [{ type: 'text', text: 'decline' }]) },
  ]), ack));
  assert.equal(events.at(-1).message.stopReason, 'stop');
  assert.equal(events.at(-1).message.content[0].text, ack);
});

test('a completed decision preserves only its first quick-answer tool call', async () => {
  const first = { type: 'toolCall', id: 'first', name: 'easyicu_quick_answer',
    arguments: { reading: { question: 'Synthetic question' } } };
  const result = message('toolUse', [first, { ...first, id: 'duplicate' },
    { type: 'toolCall', id: 'other', name: 'easyicu_run' }]);
  const events = await collect(await settleQuickAnswerOffer(model, () => stream([
    { type: 'done', message: result },
  ]), ack));
  const terminal = events.at(-1).message;
  assert.equal(terminal.stopReason, 'toolUse');
  assert.equal(terminal.content.length, 2);
  assert.equal(terminal.content[0].text, ack);
  assert.equal(terminal.content[1], first);
  assert.equal(terminal.usage, result.usage);
});

test('a typed reading issue opens one re-read on source confirmation and on direct questions', async () => {
  const feedback = { role: 'toolResult', toolName: 'easyicu_quick_answer', details: {
    code: 'quick_reading_concept_unregistered',
    details: { reading_issues: [{ field: 'measure.concept', received: 'quartz_metric_zz',
      message: 'not a registered concept id', candidates: [] }] },
  } };
  const repair = { type: 'toolCall', id: 'repair', name: feedback.toolName,
    arguments: { reading: { measure: { concept: 'lact' } } } };
  const unexpectedText = { type: 'text', text: 'Only the host card should appear' };
  const response = message('toolUse', [unexpectedText, repair, { ...repair, id: 'duplicate' }]);
  const terminalFailure = { ...feedback, isError: true, details: { code: 'quick_reading_invalid' } };
  const terminalHandoff = { ...feedback, details: {
    code: feedback.details.code, details: { quick_answer: { proposal: { status: 'full_study' } } },
  } };
  for (const content of [host.content, 'Describe a synthetic measure.']) {
    const context = { messages: [{ role: 'user', content }, feedback] };
    assert.equal(quickAnswerReadingNeedsRepair(feedback), true);
    assert.equal(hostPostToolFinalization(model, context, 'en'), null);
    let opens = 0;
    const events = await collect(await settleQuickAnswerOffer(model, () => {
      opens += 1;
      return stream([{ type: 'done', message: response }]);
    }, ''));
    assert.equal(opens, 1);
    const reread = events.at(-1).message;
    assert.equal(reread.content.filter(part => part.type === 'toolCall').length, 1);
    assert.equal(reread.content.find(part => part.type === 'toolCall'), repair);
    assert.ok(!JSON.stringify(events).includes(unexpectedText.text));
    for (const terminal of [terminalFailure, terminalHandoff]) {
      assert.equal(quickAnswerReadingNeedsRepair(terminal), false);
      const settled = await collect(hostPostToolFinalization(model, {
        messages: [...context.messages, reread, terminal],
      }, 'en'));
      assert.equal(settled.at(-1).message.stopReason, 'stop');
    }
  }
});

test('an aborted decision remains aborted', async () => {
  const aborted = message('aborted');
  const events = await collect(await settleQuickAnswerOffer(model, () => stream([
    { type: 'error', reason: 'aborted', error: aborted },
  ]), ack));
  assert.deepEqual(events.map(event => event.type), ['error']);
  assert.equal(events[0].error, aborted);
});

test('retry eligibility belongs to a failed host offer with no attempted tool', () => {
  const failed = { type: 'message', message: message('error') };
  assert.equal(quickAnswerOfferRetryAvailable([host, failed]), true);
  assert.equal(quickAnswerOfferRetryAvailable([host, { type: 'custom', customType: 'usage' }, failed]), true);
  assert.equal(quickAnswerOfferRetryAvailable([{ type: 'message', message: { role: 'user', content: host.content } }, failed]), false);
  assert.equal(quickAnswerOfferRetryAvailable([host, { type: 'message', message: message('stop') }]), false);
  assert.equal(quickAnswerOfferRetryAvailable([host, { type: 'message', message: message('aborted') }]), false);
  assert.equal(quickAnswerOfferRetryAvailable([host,
    { type: 'message', message: message('toolUse', [{ type: 'toolCall' }]) }, failed]), false);
  assert.equal(quickAnswerOfferRetryAvailable([host,
    { type: 'message', message: { role: 'user', content: 'another turn' } }, failed]), false);
});

test('session state publishes the pending offer only for an idle research conversation', () => {
  const main = fs.readFileSync(path.join(root, 'pi_copilot/node_app/src/main.mjs'), 'utf8');
  const start = main.indexOf('function sessionState(');
  const end = main.indexOf('\nasync function createSession(', start);
  const sessionState = vm.runInNewContext('(' + main.slice(start, end) + ')', {
    transcriptPage: () => ({ items: [] }), replyContinuationState: () => ({}),
    ALL_TOOL_NAMES: [], quickAnswerOfferRetryAvailable,
  });
  const record = { agentMode: 'research', session: { messages: [], isStreaming: false,
    getSessionStats: () => ({ tokens: { total: 0 } }), getActiveToolNames: () => [] },
    sessionManager: { getBranch: () => [host, { type: 'message', message: message('error') }] },
    budgetGuard: { state: () => ({ provider_calls: 4 }) }, usageLedger: { summary: () => ({}) } };
  assert.equal(sessionState(record).quick_answer_offer_retry_available, true);
  assert.equal(sessionState({ ...record, agentMode: 'workspace' }).quick_answer_offer_retry_available, false);
  record.session.isStreaming = true;
  assert.equal(sessionState(record).quick_answer_offer_retry_available, false);
});

test('the conversation holds automatic planning and offers the existing source-transition retry', () => {
  let owner;
  const window = { EasyICU: { guidedPi: { declare: (_name, value) => { owner = value; } } } };
  vm.runInNewContext(fs.readFileSync(path.join(root, 'static/js/screens-guided-pi-quick-answers.js'), 'utf8'), { window });
  const session = { quick_answer_offer_retry_available: true, last_message_job_id: 'failed-offer' };
  assert.equal(owner.holdsAutomaticPlan(session, {}), true);
  const ui = owner.create({ tr: (_en, zh) => zh, esc: String, disabled: () => false });
  const rows = ui.rows(session);
  assert.equal(rows.length, 1);
  const html = ui.render(rows[0]);
  assert.match(html, /data-gpi-data-source-continue/);
  assert.match(html, /重试/);
  assert.doesNotMatch(html, /data-gpi-quick-approve|data-gpi-quick-expand/);
  assert.equal(owner.holdsAutomaticPlan({ ...session, quick_answer_offer_retry_available: false }, {}), false);
});
