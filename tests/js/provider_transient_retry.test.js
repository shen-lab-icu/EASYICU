'use strict';
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const { pathToFileURL } = require('node:url');
const { test } = require('node:test');
const { setTimeout: sleep } = require('node:timers/promises');

const files = process.argv.slice(2);
assert.equal(files.length, 4, 'Run through tools/run_js_contracts.py with all four owners');
const [mainFile, budgetFile, projectionFile, retryFile] = files;
const source = fs.readFileSync(mainFile, 'utf8');
const start = source.indexOf('    session.agent.streamFunction =');
const end = source.indexOf('    sessions.set(externalId, record);', start);
assert.ok(start >= 0 && end > start, 'Exercise the installed session wrapper');
const wrapper = source.slice(start, end);
const eventStart = source.includes('    const emitSessionEvent =')
  ? source.indexOf('    const emitSessionEvent =')
  : source.indexOf('    unsubscribe = session.subscribe(');
const eventEnd = source.indexOf('    const record = {', eventStart);
assert.ok(eventStart >= 0 && eventEnd > eventStart, 'Exercise the installed event publisher');
const eventPublisher = source.slice(eventStart, eventEnd);
const model = { api: 'test', provider: 'test', id: 'synthetic-model' };
const context = { messages: [{ role: 'user', content: 'Summarize the available methods.' }] };
const message = (content = [], stopReason = 'stop', errorMessage = '') => ({
  role: 'assistant', api: model.api, provider: model.provider, model: model.id,
  content, stopReason, errorMessage,
});
const failed = (detail = 'HTTP 500: internal_server_error', content = []) => ({
  type: 'error', reason: 'error', error: message(content, 'error', detail),
});
const opened = attempt => ({ type: 'start', partial: message(), attempt });
const relayFailure = 'HTTP 500 ' + JSON.stringify({ error: {
  message: 'Post "https://synthetic-model.invalid/responses": proxy CONNECT returned status 503 Service Unavailable',
  type: 'server_error', code: 'internal_server_error',
} });
const success = attempt => [
  opened(attempt),
  { type: 'text_delta', contentIndex: 0, delta: 'Available.', partial: message([{ type: 'text', text: 'Available.' }]), attempt },
  { type: 'done', reason: 'stop', message: message([{ type: 'text', text: 'Available.' }]), attempt },
];

// Only the SDK's lazy error adapter is stubbed. The real assignment from main,
// real budget guard, retry owner, and event projection run together below.
function lazyStream(_model, factory) {
  let result;
  return {
    async *[Symbol.asyncIterator]() {
      try {
        const stream = await factory();
        for await (const event of stream) {
          if (event.type === 'error') result = event.error;
          if (event.type === 'done') result = event.message;
          yield event;
        }
      } catch (error) {
        result = message([], 'error', error.message);
        yield { type: 'error', reason: 'error', error: result };
      }
    },
    result: async () => result,
  };
}

const sdkInstalled = fs.existsSync(
  path.join(path.dirname(budgetFile), '../node_modules/@earendil-works/pi-ai/package.json'));
const needsSdk = { skip: sdkInstalled ? false : 'Pinned Pi dependencies are not installed in this checkout' };

const imports = Promise.all([
  import(pathToFileURL(budgetFile)),
  import(pathToFileURL(projectionFile)),
  import(pathToFileURL(retryFile)),
  // The usage ledger sits beside the budget guard and wraps each attempt.
  import(pathToFileURL(path.join(path.dirname(budgetFile), 'shell-usage.mjs'))),
  // The data-source transition's quick-answer offer settles the decision call
  // through the pinned SDK, which only the Pi workspace security lane installs.
  sdkInstalled
    ? import(pathToFileURL(path.join(path.dirname(budgetFile), 'post-tool-finalization.mjs')))
    : null,
]);

async function harness(attempts, config = {}) {
  const [{ ShellBudgetGuard }, projection, retry, { ShellUsageLedger }, finalization] = await imports;
  const guard = new ShellBudgetGuard({
    tokenBudget: 100000, maxOutputTokens: 1000,
    maxProviderCallsPerMessage: 8, maxProviderCallsPerSession: 128,
    consumedTokens: () => 0, ...config.guard,
  });
  if (config.previousCall) {
    guard.beginMessage();
    guard.authorize(context, { maxTokens: 1000 });
    guard.endMessage();
  }
  guard.beginMessage();
  const receipts = [];
  const egressReceipts = [];
  const calls = [];
  const waits = [];
  const order = [];
  const notices = [];
  const retryStarts = [];
  const closed = [];
  const authorize = guard.authorize.bind(guard);
  guard.authorize = (...args) => { order.push('authorize'); return authorize(...args); };
  const usage = [];
  const record = { budgetGuard: guard, language: 'zh', usageLedger: new ShellUsageLedger({ persist: row => usage.push(row) }) };
  let subscribed;
  const session = { agent: {}, subscribe: publisher => { subscribed = publisher; return () => {}; } };
  const manager = {
    appendCustomEntry(schema, receipt) {
      order.push('receipt');
      assert.equal(schema, receipt.schema_version);
      receipts.push(receipt);
      config.onReceipt?.(receipts.length);
    },
  };
  const originalStreamFunction = async (selected, suppliedContext, options) => {
    order.push('provider');
    assert.equal(selected, model);
    assert.equal(suppliedContext, context);
    assert.equal(options.maxRetries, 0, 'SDK retries stay disabled for each attempt');
    calls.push(options);
    const call = calls.length;
    const attempt = attempts[Math.min(calls.length - 1, attempts.length - 1)];
    if (attempt instanceof Error) throw attempt;
    return {
      async *[Symbol.asyncIterator]() {
        try {
          for (const event of attempt) {
            await config.beforeEvent?.(event);
            if (event instanceof Error) throw event;
            yield event;
          }
        } finally { closed.push(call); }
      },
      result: async () => attempt.at(-1)?.error || attempt.at(-1)?.message,
    };
  };
  const retryWithFastSleep = (openAttempt, options) => retry.retryTransientProviderStream(openAttempt, {
      ...options,
      onRetry: event => {
        retryStarts.push(event);
        options.onRetry?.(event);
      },
      sleep: async (ms, value, sleepOptions) => {
        waits.push(ms);
        await config.onWait?.(sleepOptions?.signal, waits.length);
        sleepOptions?.signal?.throwIfAborted();
      },
    });
  vm.runInNewContext(eventPublisher + wrapper, {
    session, record, manager, originalStreamFunction, lazyStream,
    externalId: 'synthetic-session',
    SESSION_DIR: 'synthetic-session-root',
    appendModelRequestReceipt(root, sessionId, selected, suppliedContext) {
      assert.equal(root, 'synthetic-session-root');
      assert.equal(sessionId, 'synthetic-session');
      assert.equal(selected, model);
      assert.equal(suppliedContext, context);
      egressReceipts.push({ selected, suppliedContext });
      assert.equal(calls.length + 1, egressReceipts.length, 'receipt precedes transport');
    },
    activeRequestBySession: new Map(config.activeRequest === false ? [] : [['synthetic-session', 'synthetic-request']]),
    normalizePiEvent: projection.normalizePiEvent,
    emit: envelope => { order.push('retry'); notices.push(envelope); },
    hostPostToolFinalization: () => config.finalization || null,
    quickAnswerOfferReply: () => config.quickOfferReply || '',
    settleQuickAnswerOffer: finalization?.settleQuickAnswerOffer,
    ...retry, retryTransientProviderStream: retryWithFastSleep,
  }, { filename: mainFile });
  if (config.sessionEvent) subscribed(config.sessionEvent);
  const events = [];
  const stream = session.agent.streamFunction(model, context, { maxTokens: 1000, ...config.options });
  for await (const event of stream) {
    events.push(event);
    config.onEvent?.(event, calls);
  }
  const result = await stream.result();
  const projected = projection.normalizePiEvent({ type: 'message_end', message: result });
  return { guard, receipts, egressReceipts, calls, waits, order, events, result, projected, notices, retryStarts, closed, usage,
    openCall: () => session.agent.streamFunction(model, context, { maxTokens: 1000 }),
  };
}

for (const attempt of [2, 3, 4]) {
  test(`empty relay failure succeeds on attempt ${attempt} through the guard with private failed prefixes`, async () => {
    const failures = Array.from({ length: attempt - 1 }, (_, i) => [opened(i + 1), failed(relayFailure)]);
    const result = await harness([...failures, success(attempt)]);
    assert.equal(result.calls.length, attempt);
    assert.equal(result.egressReceipts.length, attempt, 'every retry retains a request receipt');
    assert.deepEqual(result.events, success(attempt));
    // Each retried attempt is its own accounted call, never folded into the success.
    assert.deepEqual(result.usage.filter(row => row.state !== 'in_flight').map(row => [row.call_id, row.state]),
      Array.from({ length: attempt }, (_, i) => [i + 1, i + 1 < attempt ? 'failed' : 'completed']));
    assert.deepEqual(result.receipts.map(row => row.provider_calls), Array.from({ length: attempt }, (_, i) => i + 1));
    assert.deepEqual(result.order, [
      ...Array.from({ length: attempt - 1 }, () => ['authorize', 'receipt', 'provider', 'retry']).flat(),
      'authorize', 'receipt', 'provider',
    ]);
    assert.deepEqual(result.waits, [1500, 4000, 10000].slice(0, attempt - 1));
    assert.deepEqual(result.closed, Array.from({ length: attempt }, (_, i) => i + 1));
    assert.equal(result.guard.state().message_provider_calls, attempt);
    assert.equal(result.projected.error_code, undefined);
    assert.match(source, /retry: \{ enabled: false, maxRetries: 0 \}/);
    assert.deepEqual(result.retryStarts, Array.from({ length: attempt - 1 }, (_, i) => ({
      type: 'auto_retry_start', attempt: i + 1, maxAttempts: 3,
    })));
    assert.equal(result.notices.length, attempt - 1);
    for (const [i, notice] of result.notices.entries()) {
      assert.deepEqual(JSON.parse(JSON.stringify(notice)), {
        kind: 'event', request_id: 'synthetic-request', session_id: 'synthetic-session',
        event: { type: 'retry', at: notice.event.at, attempt: i + 1, max_attempts: 3 },
      });
      assert.ok(Number.isFinite(Date.parse(notice.event.at)));
    }
  });
}

test('retry progress is not published without an active request', async () => {
  const result = await harness([[failed()], success(2)], { activeRequest: false });
  assert.equal(result.calls.length, 2);
  assert.deepEqual(result.notices, []);
});

test('the retry event publisher still forwards ordinary session lifecycle events', async () => {
  const result = await harness([success(1)], { sessionEvent: { type: 'agent_start' } });
  assert.equal(result.notices.length, 1);
  assert.equal(result.notices[0].event.type, 'run_start');
  assert.equal(result.notices[0].session_id, 'synthetic-session');
  assert.equal(result.notices[0].request_id, 'synthetic-request');
});

for (const detail of ['HTTP 502: bad gateway', 'HTTP 503: unavailable', 'HTTP 504: gateway timeout',
  'connection reset by peer', 'request timeout', 'request timed out', 'read ECONNRESET', 'connect ETIMEDOUT',
  'dial tcp 203.0.113.1:443: connect: operation timed out', relayFailure,
  'proxy CONNECT failed', 'upstream connection failure', 'upstream connect error or disconnect/reset before headers']) {
  test(`${detail} before content retries`, async () => {
    const result = await harness([[opened(1), failed(detail)], success(2)]);
    assert.equal(result.calls.length, 2);
    assert.deepEqual(result.events, success(2));
  });
}

for (const event of [
  { type: 'text_start' }, { type: 'text_delta', delta: 'Some text' },
  { type: 'thinking_start' }, { type: 'thinking_delta', delta: 'Some reasoning' },
  { type: 'toolcall_start' }, { type: 'toolcall_delta', delta: '{' },
  { type: 'toolcall_end', toolCall: { type: 'toolCall', name: 'synthetic_tool', arguments: {} } },
]) {
  test(`HTTP 500 after ${event.type} is never retried`, async () => {
    const first = [opened(1), event, failed()];
    const result = await harness([first, success(2)]);
    assert.equal(result.calls.length, 1);
    assert.deepEqual(result.events, first);
    assert.deepEqual(result.waits, []);
    assert.equal(result.projected.error_code, 'pi_model_provider_unavailable');
  });
}

for (const content of [
  [{ type: 'text', text: 'Partial answer' }],
  [{ type: 'thinking', thinking: 'Partial reasoning' }],
  [{ type: 'toolCall', id: 'synthetic-call', name: 'synthetic_tool', arguments: {} }],
]) {
  test(`content in a terminal ${content[0].type} snapshot prevents retry`, async () => {
    const first = [opened(1), failed('HTTP 500', content)];
    const result = await harness([first, success(2)]);
    assert.equal(result.calls.length, 1);
    assert.deepEqual(result.events, first);
  });
  test(`content in an opening ${content[0].type} snapshot prevents retry`, async () => {
    const first = [{ type: 'start', partial: message(content) }, failed()];
    const result = await harness([first, success(2)]);
    assert.equal(result.calls.length, 1);
    assert.deepEqual(result.events, first);
  });
}

test('a mutated SDK prefix snapshot with content prevents retry', async () => {
  const partial = message();
  const first = [{ type: 'start', partial }, failed()];
  const result = await harness([first, success(2)], {
    beforeEvent: event => {
      if (event.type === 'error') partial.content.push({ type: 'text', text: 'Partial answer' });
    },
  });
  assert.equal(result.calls.length, 1);
  assert.deepEqual(result.events, first);
});

test('a transient exception after a mutated content snapshot is never retried', async () => {
  const partial = message();
  const failure = new Error('HTTP 500');
  const result = await harness([[{ type: 'start', partial }, failure], success(2)], {
    beforeEvent: event => {
      if (event === failure) partial.content.push({ type: 'thinking', thinking: 'Partial reasoning' });
    },
  });
  assert.equal(result.calls.length, 1);
  assert.equal(result.projected.error_code, 'pi_model_provider_unavailable');
});

test('an SDK aborted terminal message is never retried', async () => {
  const first = [opened(1), { type: 'error', reason: 'aborted', error: message([], 'aborted', 'request timed out') }];
  const result = await harness([first, success(2)]);
  assert.equal(result.calls.length, 1);
  assert.deepEqual(result.events, first);
  assert.equal(result.projected.error_code, undefined);
});

for (const [detail, code] of [
  ['HTTP 429', 'pi_model_rate_limited'],
  ['HTTP 429: rate limit', 'pi_model_rate_limited'],
  ['HTTP 500: rate_limit_exceeded', 'pi_model_rate_limited'],
  ['HTTP 500: insufficient_quota', 'pi_model_rate_limited'],
  ['HTTP 500: budget_exhausted', 'pi_model_provider_error'],
  ['HTTP 500: context_length_exceeded', 'pi_model_context_limit'],
  ['HTTP 400: maximum length exceeded', 'pi_model_context_limit'],
  ['HTTP 401: authentication failed', 'pi_model_provider_error'],
  ['HTTP 403: permission denied', 'pi_model_provider_error'],
  ['HTTP 404: model missing', 'pi_model_provider_error'],
  ['HTTP 408: request timeout', 'pi_model_provider_error'],
  ['HTTP 422: server_error', 'pi_model_provider_error'],
  ['HTTP 400: prior request returned 500', 'pi_model_provider_error'],
  ['unsupported model configuration', 'pi_model_provider_error'],
  ['HTTP 500: server_error: authentication failed', 'pi_model_provider_error'],
  ['HTTP 500: server_error: authorization denied', 'pi_model_provider_error'],
  ['HTTP 500: server_error: invalid_api_key', 'pi_model_provider_error'],
  ['HTTP 500: server_error: missing credentials', 'pi_model_provider_error'],
  ['HTTP 500: server_error: permission denied', 'pi_model_provider_error'],
  ['HTTP 500: server_error: unauthorized', 'pi_model_provider_error'],
  ['HTTP 500: server_error: forbidden', 'pi_model_provider_error'],
  ['HTTP 500: server_error: invalid bearer token', 'pi_model_provider_error'],
  ['HTTP 500: server_error: content_policy_violation', 'pi_model_provider_error'],
  ['HTTP 500: server_error: content_filter', 'pi_model_provider_error'],
  ['HTTP 500: server_error: safety refusal', 'pi_model_provider_error'],
  ['HTTP 500: server_error: request refused', 'pi_model_provider_error'],
  ['HTTP 500: server_error: invalid_request_error', 'pi_model_provider_error'],
  ['HTTP 500: server_error: malformed request', 'pi_model_provider_error'],
  ['HTTP 500: server_error: malformed JSON', 'pi_model_provider_error'],
  ['HTTP 500: server_error: bad_request', 'pi_model_provider_error'],
  ['HTTP 500: server_error: invalid parameter', 'pi_model_provider_error'],
  ['HTTP 500: server_error: unsupported model configuration', 'pi_model_provider_error'],
  ['HTTP 500: server_error: context_length_exceeded', 'pi_model_context_limit'],
  ['HTTP 403: proxy CONNECT returned status 503 Service Unavailable', 'pi_model_provider_error'],
  ['HTTP 400: upstream connection failure', 'pi_model_provider_error'],
]) {
  test(`${detail} is never retried`, async () => {
    const first = [opened(1), failed(detail)];
    const result = await harness([first, success(2)]);
    assert.equal(result.calls.length, 1);
    assert.deepEqual(result.events, first);
    assert.deepEqual(result.waits, []);
    assert.deepEqual(result.notices, []);
    assert.equal(result.projected.error_code, code);
  });
}

test('four transient failures exhaust retries and expose only the final stream and result', async () => {
  const attempts = Array.from({ length: 4 }, (_, i) => [opened(i + 1), failed(`HTTP 500: request ${i + 1} failed`)]);
  const result = await harness([...attempts, success(5)]);
  assert.equal(result.calls.length, 4);
  assert.deepEqual(result.events, attempts[3]);
  assert.equal(result.projected.error_code, 'pi_model_provider_unavailable');
  assert.equal(result.result, attempts[3].at(-1).error);
  assert.deepEqual(result.waits, [1500, 4000, 10000]);
  assert.equal(result.notices.length, 3);
});

test('a permanent failure on a later attempt stops immediately', async () => {
  const second = [opened(2), failed('HTTP 500: invalid_request_error')];
  const result = await harness([[failed()], second, success(3)]);
  assert.equal(result.calls.length, 2);
  assert.deepEqual(result.events, second);
  assert.deepEqual(result.waits, [1500]);
  assert.equal(result.projected.error_code, 'pi_model_provider_error');
});

for (const event of [
  { type: 'text_start' }, { type: 'thinking_start' }, { type: 'toolcall_start' },
  { type: 'toolcall_end', toolCall: { type: 'toolCall', name: 'synthetic_tool', arguments: {} } },
  failed('HTTP 500', [{ type: 'text', text: 'Partial answer' }]),
]) {
  test(`content on a later attempt (${event.type}) prevents further retries or replay`, async () => {
    const second = [opened(2), event, failed()];
    const result = await harness([[failed()], second, success(3)]);
    assert.equal(result.calls.length, 2);
    assert.deepEqual(result.events, second);
    assert.deepEqual(result.waits, [1500]);
  });
}

for (const stage of ['open', 'iterate']) {
  test(`transient exceptions during ${stage} can succeed on the fourth attempt`, async () => {
    const error = new Error('HTTP 500: transient request failure');
    const failures = Array.from({ length: 3 }, (_, i) => stage === 'open' ? error : [opened(i + 1), error]);
    const result = await harness([...failures, success(4)]);
    assert.equal(result.calls.length, 4);
    assert.deepEqual(result.events, success(4));
    assert.equal(result.receipts.length, 4);
    assert.deepEqual(result.waits, [1500, 4000, 10000]);
  });
  test(`transient exceptions during ${stage} give up after the fourth attempt`, async () => {
    const error = new Error('HTTP 500: transient request failure');
    const result = await harness([stage === 'open' ? error : [opened(1), error]]);
    assert.equal(result.calls.length, 4);
    assert.equal(result.projected.error_code, 'pi_model_provider_unavailable');
    assert.deepEqual(result.waits, [1500, 4000, 10000]);
  });
  test(`a nontransient exception during ${stage} is never retried`, async () => {
    const error = new Error('HTTP 429: too many requests');
    const result = await harness([stage === 'open' ? error : [opened(1), error], success(2)]);
    assert.equal(result.calls.length, 1);
    assert.equal(result.projected.error_code, 'pi_model_rate_limited');
    assert.deepEqual(result.waits, []);
  });
}

test('a transient exception after text is never retried', async () => {
  const first = [opened(1), success(1)[1], new Error('HTTP 500')];
  const result = await harness([first, success(2)]);
  assert.equal(result.calls.length, 1);
  assert.equal(result.events[1], first[1]);
  assert.equal(result.projected.error_code, 'pi_model_provider_unavailable');
});

for (const [guard, code, previousCall = false] of [
  [{ maxProviderCallsPerMessage: 1 }, 'pi_shell_message_provider_call_budget_exhausted'],
  [{ maxProviderCallsPerSession: 1 }, 'pi_shell_session_provider_call_budget_exhausted'],
  [{ pricing: { inputPriceUsdPerMillionTokens: 10, outputPriceUsdPerMillionTokens: 30,
    maxCostUsdPerMessage: 0.1, maxCostUsdPerSession: 1 } }, 'pi_shell_message_cost_budget_exhausted'],
  [{ pricing: { inputPriceUsdPerMillionTokens: 10, outputPriceUsdPerMillionTokens: 30,
    maxCostUsdPerMessage: 0.15, maxCostUsdPerSession: 0.15 } }, 'pi_shell_session_cost_budget_exhausted', true],
]) {
  test(`retry authorization preserves ${code}`, async () => {
    const result = await harness([[opened(1), failed()], success(2)], { guard, previousCall });
    assert.equal(result.calls.length, 1);
    assert.equal(result.receipts.length, 1);
    assert.equal(result.projected.error_code, code);
    assert.match(result.result.errorMessage, new RegExp(code));
    assert.deepEqual(result.order, ['authorize', 'receipt', 'provider', 'retry', 'authorize']);
  });
}

test('retry authorization preserves token exhaustion after the first failure', async () => {
  let consumed = 0;
  const result = await harness([[opened(1), failed()], success(2)], {
    guard: { consumedTokens: () => consumed }, onWait: () => { consumed = 100000; },
  });
  assert.equal(result.calls.length, 1);
  assert.equal(result.receipts.length, 1);
  assert.equal(result.projected.error_code, 'pi_shell_token_budget_exhausted');
});

test('an initial guard refusal makes no provider call and does not retry', async () => {
  const result = await harness([[failed()]], { guard: { tokenBudget: 4000 } });
  assert.equal(result.calls.length, 0);
  assert.equal(result.receipts.length, 0);
  assert.deepEqual(result.waits, []);
  assert.equal(result.projected.error_code, 'pi_shell_token_budget_exhausted');
});

test('a successful retry consumes the message ceiling for the following call', async () => {
  const result = await harness([[opened(1), failed()], success(2)], { guard: { maxProviderCallsPerMessage: 2 } });
  assert.equal(result.guard.state().message_provider_calls, 2);
  assert.throws(() => result.guard.authorize(context, {}), { code: 'pi_shell_message_provider_call_budget_exhausted' });
});

test('a later retry still cannot exceed the host provider-call budget', async () => {
  const result = await harness([[failed()]], { guard: { maxProviderCallsPerMessage: 3 } });
  assert.equal(result.calls.length, 3);
  assert.deepEqual(result.receipts.map(row => row.provider_calls), [1, 2, 3]);
  assert.equal(result.projected.error_code, 'pi_shell_message_provider_call_budget_exhausted');
  assert.deepEqual(result.waits, [1500, 4000, 10000]);
});

test('each call within a message has its own retry allowance', async () => {
  const result = await harness([[failed()], [failed()], [failed()], success(4), [failed()], success(6)]);
  const events = [];
  for await (const event of result.openCall()) events.push(event);
  assert.equal(result.calls.length, 6);
  assert.deepEqual(events, success(6));
  assert.deepEqual(result.receipts.map(row => row.provider_calls), [1, 2, 3, 4, 5, 6]);
  assert.deepEqual(result.waits, [1500, 4000, 10000, 1500]);
  assert.deepEqual(result.retryStarts.map(event => event.attempt), [1, 2, 3, 1]);
});

test('the retry reserves provider cost in its own durable receipt', async () => {
  const result = await harness([[failed()], success(2)], { guard: { pricing: {
    inputPriceUsdPerMillionTokens: 10, outputPriceUsdPerMillionTokens: 30,
    maxCostUsdPerMessage: 0.3, maxCostUsdPerSession: 1,
  } } });
  assert.equal(result.receipts.length, 2);
  assert.equal(result.receipts[0].schema_version, 'easyicu.shell-budget/2');
  assert.equal(result.receipts[1].reserved_cost_micro_usd, 2 * result.receipts[0].reserved_cost_micro_usd);
  assert.equal(result.guard.state().cost, result.receipts[1].reserved_cost_micro_usd / 1000000);
});

test('successful streams are forwarded while content is still arriving', async () => {
  const first = success(1);
  let release;
  const gate = new Promise(resolve => { release = resolve; });
  let timer;
  const deadline = new Promise((_, reject) => {
    timer = setTimeout(() => reject(new Error('Content was buffered until completion')), 500);
  });
  const result = await harness([first], {
    beforeEvent: event => event.type === 'done' ? Promise.race([gate, deadline]) : undefined,
    onEvent: event => { if (event.type === 'text_delta') { clearTimeout(timer); release(); } },
  });
  assert.equal(result.calls.length, 1);
  assert.deepEqual(result.events, first);
  assert.equal(result.result, first.at(-1).message);
  assert.deepEqual(result.waits, []);
});

test('host finalization avoids provider authorization and retries', async () => {
  const final = success(0);
  const finalization = { async *[Symbol.asyncIterator]() { yield* final; } };
  const result = await harness([[failed()]], { finalization });
  assert.equal(result.calls.length, 0);
  assert.equal(result.receipts.length, 0);
  assert.deepEqual(result.events, final);
});

test('an aborted call does not retry', async () => {
  const controller = new AbortController();
  controller.abort();
  const result = await harness([[opened(1), failed()], success(2)], { options: { signal: controller.signal } });
  assert.equal(result.calls.length, 1);
  assert.equal(result.receipts.length, 1);
  assert.deepEqual(result.waits, []);
});

for (const stopAt of [1, 2, 3]) {
  test(`abort during wait ${stopAt} prevents another authorization`, async () => {
    const controller = new AbortController();
    const result = await harness([[opened(1), failed()]], {
      options: { signal: controller.signal },
      onWait: (signal, count) => {
        assert.equal(signal, controller.signal);
        if (count === stopAt) controller.abort();
      },
    });
    assert.equal(result.calls.length, stopAt);
    assert.equal(result.receipts.length, stopAt);
    assert.equal(result.notices.length, stopAt);
    assert.deepEqual(result.waits, [1500, 4000, 10000].slice(0, stopAt));
    assert.ok(result.events.every(event => event.type === 'error'), 'failed prefixes stay private on abort');
  });
}

test('the default delay is bounded and abortable', async () => {
  const [, , retry] = await imports;
  assert.equal(typeof retry.retryTransientProviderStream, 'function');
  const controller = new AbortController();
  let calls = 0;
  const started = Date.now();
  const promise = retry.retryTransientProviderStream(async () => {
    calls += 1;
    return { async *[Symbol.asyncIterator]() { yield failed(); } };
  }, { signal: controller.signal });
  await sleep(10);
  controller.abort();
  await assert.rejects(promise, { name: 'AbortError' });
  assert.equal(calls, 1);
  assert.ok(Date.now() - started < 1000);
});

for (const stage of ['cleanup', 'wait completion']) {
  test(`abort at ${stage} prevents opening or announcing another attempt`, async () => {
    const [, , retry] = await imports;
    const controller = new AbortController();
    const notices = [];
    const waits = [];
    let calls = 0;
    await assert.rejects(retry.retryTransientProviderStream(async () => {
      calls += 1;
      assert.equal(calls, 1, 'no provider call after abort');
      return {
        async *[Symbol.asyncIterator]() {
          try { yield failed(); }
          finally { if (stage === 'cleanup') controller.abort(); }
        },
      };
    }, {
      signal: controller.signal,
      onRetry: event => notices.push(event),
      sleep: async ms => { waits.push(ms); controller.abort(); },
    }), { name: 'AbortError' });
    assert.equal(calls, 1);
    assert.equal(notices.length, stage === 'cleanup' ? 0 : 1);
    assert.deepEqual(waits, stage === 'cleanup' ? [] : [1500]);
  });
}

test('prefix replay preserves the SDK result and closes the underlying iterator', async () => {
  const [, , retry] = await imports;
  const events = success(1);
  let closed = false;
  const original = {
    async *[Symbol.asyncIterator]() {
      try { yield* events; } finally { closed = true; }
    },
    result: async () => events.at(-1).message,
  };
  const stream = await retry.retryTransientProviderStream(async () => original);
  const observed = [];
  for await (const event of stream) observed.push(event);
  assert.deepEqual(observed, events);
  assert.equal(await stream.result(), events.at(-1).message);
  assert.equal(closed, true);
});

// The data-source transition may offer the saved question to the quick-answer
// owner. That provider call is settled, never streamed: the host writes the
// reply, and only the first easyicu_quick_answer call goes on to the host.
const ACK = '已确认使用 eICU Demo 数据。';
const reading = { question: '入 ICU 第一天最高乳酸的中位数是多少？' };
const decided = (content, stopReason, usage) => [
  opened(1),
  { type: 'done', reason: stopReason, message: { ...message(content, stopReason), usage } },
];

test('a quick-answer offer the model declines shows only the host acknowledgement', needsSdk, async () => {
  const usage = { input: 40, output: 9, cacheRead: 0, cacheWrite: 0, totalTokens: 49, cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 } };
  const result = await harness([decided([{ type: 'text', text: 'Model prose with a number 2.1' }], 'stop', usage)], { quickOfferReply: ACK });
  assert.equal(result.calls.length, 1);
  assert.equal(result.result.stopReason, 'stop');
  assert.deepEqual(JSON.parse(JSON.stringify(result.result.content)), [{ type: 'text', text: ACK }]);
  assert.ok(!JSON.stringify(result.events).includes('2.1'), 'the model text is never streamed');
  assert.deepEqual(result.result.usage, usage, 'the decision call stays accounted');
  assert.deepEqual(result.usage.filter(row => row.state !== 'in_flight').map(row => row.state), ['completed']);
});

test('a quick-answer offer keeps the first reading call behind the acknowledgement', needsSdk, async () => {
  const first = { type: 'toolCall', id: 'call-quick', name: 'easyicu_quick_answer', arguments: { reading } };
  const result = await harness([decided([
    { type: 'text', text: 'I will submit it.' }, first,
    { type: 'toolCall', id: 'call-again', name: 'easyicu_quick_answer', arguments: { reading } },
    { type: 'toolCall', id: 'call-other', name: 'easyicu_run', arguments: {} },
  ], 'toolUse')], { quickOfferReply: ACK });
  assert.equal(result.result.stopReason, 'toolUse');
  assert.deepEqual(JSON.parse(JSON.stringify(result.result.content)), [{ type: 'text', text: ACK }, first]);
  assert.deepEqual(result.events.map(event => event.type), [
    'start', 'text_start', 'text_delta', 'text_end', 'toolcall_start', 'toolcall_delta', 'toolcall_end', 'done',
  ]);
});

test('a failed or aborted quick-answer decision keeps the transition safe', needsSdk, async () => {
  const failedCall = await harness([[opened(1), failed('HTTP 401: authentication failed')]], { quickOfferReply: ACK });
  assert.equal(failedCall.result.stopReason, 'stop');
  assert.deepEqual(JSON.parse(JSON.stringify(failedCall.result.content)), [{ type: 'text', text: ACK }]);
  const aborted = await harness([[opened(1), { type: 'error', reason: 'aborted', error: message([], 'aborted') }]], { quickOfferReply: ACK });
  assert.equal(aborted.result.stopReason, 'aborted');
  assert.deepEqual(aborted.events.map(event => event.type), ['error']);
});
