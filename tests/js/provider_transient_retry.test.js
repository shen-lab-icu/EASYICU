'use strict';
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const { pathToFileURL } = require('node:url');
const { test } = require('node:test');
const { setTimeout: sleep } = require('node:timers/promises');

const DELAYS = [2000, 4000, 8000, 16000, 30000, 30000, 29000];
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
        if (stream.result) result = await stream.result();
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
const finalizerFile = path.join(path.dirname(budgetFile), 'post-tool-finalization.mjs');

function offlineFinalizer() {
  const source = fs.readFileSync(finalizerFile, 'utf8');
  const names = [...source.matchAll(/^export (?:async )?(?:function\*?|const) (\w+)/gm)].map(match => match[1]);
  return vm.runInNewContext(source.replace(/^import .*?;\n/m, '').replace(/^export /gm, '')
    + '\n({' + names.join(', ') + '});', {
    createAssistantMessageEventStream() {
      const events = [];
      return { push: event => events.push(event), async *[Symbol.asyncIterator]() { yield* events; } };
    },
  }, { filename: finalizerFile });
}

const imports = Promise.all([
  import(pathToFileURL(budgetFile)),
  import(pathToFileURL(projectionFile)),
  import(pathToFileURL(retryFile)),
  // The usage ledger sits beside the budget guard and wraps each attempt.
  import(pathToFileURL(path.join(path.dirname(budgetFile), 'shell-usage.mjs'))),
  // On offline checkouts, replace only the SDK event queue.
  sdkInstalled ? import(pathToFileURL(finalizerFile)) : offlineFinalizer(),
]);

async function harness(attempts, config = {}) {
  const [{ ShellBudgetGuard }, projection, retry, { ShellUsageLedger }, finalization] = await imports;
  const turnContext = config.context || context;
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
  let elapsed = 0;
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
    assert.equal(suppliedContext, egressReceipts.at(-1).suppliedContext);
    if (config.checkContext) config.checkContext(suppliedContext);
    else assert.equal(suppliedContext, turnContext);
    assert.equal(options.maxRetries, 0, 'SDK retries stay disabled for each attempt');
    calls.push(options);
    elapsed += config.callDuration || 0;
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
      now: () => elapsed,
      onRetry: event => {
        retryStarts.push(event);
        options.onRetry?.(event);
      },
      sleep: async (ms, value, sleepOptions) => {
        waits.push(ms);
        elapsed += ms;
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
      egressReceipts.push({ selected, suppliedContext });
      assert.equal(calls.length + 1, egressReceipts.length, 'receipt precedes transport');
    },
    activeRequestBySession: new Map(config.activeRequest === false ? [] : [['synthetic-session', 'synthetic-request']]),
    normalizePiEvent: projection.normalizePiEvent,
    emit: envelope => { order.push('retry'); notices.push(envelope); },
    ...finalization,
    hostPostToolFinalization: config.finalization ? () => config.finalization : finalization.hostPostToolFinalization,
    quickAnswerOfferReply: config.quickOfferReply ? () => config.quickOfferReply : finalization.quickAnswerOfferReply,
    ...retry, retryTransientProviderStream: retryWithFastSleep,
  }, { filename: mainFile });
  if (config.sessionEvent) subscribed(config.sessionEvent);
  const events = [];
  const stream = session.agent.streamFunction(model, turnContext, { maxTokens: 1000, ...config.options });
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

for (const attempt of [2, 3, 4, 7, 8]) {
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
    assert.deepEqual(result.waits, DELAYS.slice(0, attempt - 1));
    assert.deepEqual(result.closed, Array.from({ length: attempt }, (_, i) => i + 1));
    assert.equal(result.guard.state().message_provider_calls, attempt);
    assert.equal(result.projected.error_code, undefined);
    assert.match(source, /retry: \{ enabled: false, maxRetries: 0 \}/);
    assert.deepEqual(result.retryStarts, Array.from({ length: attempt - 1 }, (_, i) => ({
      type: 'auto_retry_start', attempt: i + 1, maxAttempts: 7,
    })));
    assert.equal(result.notices.length, attempt - 1);
    for (const [i, notice] of result.notices.entries()) {
      assert.deepEqual(JSON.parse(JSON.stringify(notice)), {
        kind: 'event', request_id: 'synthetic-request', session_id: 'synthetic-session',
        event: { type: 'retry', at: notice.event.at, attempt: i + 1, max_attempts: 7 },
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
  'HTTP 408: request timeout', 'HTTP 429: rate limit', 'connect ECONNREFUSED', 'fetch failed', 'connection reset by peer', 'request timeout', 'request timed out', 'read ECONNRESET', 'connect ETIMEDOUT',
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
  ['HTTP 500: rate_limit_exceeded', 'pi_model_rate_limited'],
  ['HTTP 500: insufficient_quota', 'pi_model_rate_limited'],
  ['HTTP 500: budget_exhausted', 'pi_model_provider_error'],
  ['HTTP 500: context_length_exceeded', 'pi_model_context_limit'],
  ['HTTP 400: maximum length exceeded', 'pi_model_context_limit'],
  ['HTTP 401: authentication failed', 'pi_model_provider_error'],
  ['HTTP 403: permission denied', 'pi_model_provider_error'],
  ['HTTP 404: model missing', 'pi_model_provider_error'],
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

test('eight transient failures exhaust retries and expose only the final stream and result', async () => {
  const attempts = Array.from({ length: 8 }, (_, i) => [opened(i + 1), failed(`HTTP 500: request ${i + 1} failed`)]);
  const result = await harness([...attempts, success(9)]);
  assert.equal(result.calls.length, 8);
  assert.deepEqual(result.events, attempts[7]);
  assert.equal(result.projected.error_code, 'pi_model_provider_unavailable');
  assert.equal(result.result, attempts[7].at(-1).error);
  assert.deepEqual(result.waits, DELAYS);
  assert.equal(result.notices.length, 7);
});

test('a permanent failure on a later attempt stops immediately', async () => {
  const second = [opened(2), failed('HTTP 500: invalid_request_error')];
  const result = await harness([[failed()], second, success(3)]);
  assert.equal(result.calls.length, 2);
  assert.deepEqual(result.events, second);
  assert.deepEqual(result.waits, [2000]);
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
    assert.deepEqual(result.waits, [2000]);
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
    assert.deepEqual(result.waits, DELAYS.slice(0, 3));
  });
  test(`transient exceptions during ${stage} give up after the eighth attempt`, async () => {
    const error = new Error('HTTP 500: transient request failure');
    const result = await harness([stage === 'open' ? error : [opened(1), error]]);
    assert.equal(result.calls.length, 8);
    assert.equal(result.projected.error_code, 'pi_model_provider_unavailable');
    assert.deepEqual(result.waits, DELAYS);
  });
  test(`a nontransient exception during ${stage} is never retried`, async () => {
    const error = new Error('HTTP 401: authentication failed');
    const result = await harness([stage === 'open' ? error : [opened(1), error], success(2)]);
    assert.equal(result.calls.length, 1);
    assert.equal(result.projected.error_code, 'pi_model_provider_error');
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
  assert.deepEqual(result.waits, DELAYS.slice(0, 3));
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
    context: { ...context, tools: quickTools },
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
    assert.deepEqual(result.waits, DELAYS.slice(0, stopAt));
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
    assert.deepEqual(waits, stage === 'cleanup' ? [] : [2000]);
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

test('a quick-answer offer the model declines shows only the host acknowledgement', async () => {
  const usage = { input: 40, output: 9, cacheRead: 0, cacheWrite: 0, totalTokens: 49, cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 } };
  const result = await harness([decided([{ type: 'text', text: 'Model prose with a number 2.1' }], 'stop', usage)], { quickOfferReply: ACK });
  assert.equal(result.calls.length, 1);
  assert.equal(result.result.stopReason, 'stop');
  assert.deepEqual(JSON.parse(JSON.stringify(result.result.content)), [{ type: 'text', text: ACK }]);
  assert.ok(!JSON.stringify(result.events).includes('2.1'), 'the model text is never streamed');
  assert.deepEqual(result.result.usage, usage, 'the decision call stays accounted');
  assert.deepEqual(result.usage.filter(row => row.state !== 'in_flight').map(row => row.state), ['completed']);
});

test('a quick-answer offer keeps the first reading call behind the acknowledgement', async () => {
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

test('a failed or aborted quick-answer decision keeps the transition safe', async () => {
  const failedCall = await harness([[opened(1), failed('HTTP 401: authentication failed')]], { quickOfferReply: ACK });
  assert.equal(failedCall.result.stopReason, 'error');
  assert.equal(failedCall.projected.error_code, 'pi_model_provider_error');
  assert.deepEqual(failedCall.events.map(event => event.type), ['error']);
  const aborted = await harness([[opened(1), { type: 'error', reason: 'aborted', error: message([], 'aborted') }]], { quickOfferReply: ACK });
  assert.equal(aborted.result.stopReason, 'aborted');
  assert.deepEqual(aborted.events.map(event => event.type), ['error']);
});

const QUICK = 'easyicu_quick_answer';
const CATALOG = 'easyicu_list_data_sources';
const quickTools = [{ name: QUICK }, { name: CATALOG }];
const reviewQuestion = 'Could you summarize typical peak lactate in the opening six hours?';
const initial = { type: 'toolCall', id: 'initial', name: QUICK,
  arguments: { reading: { question: reviewQuestion, measure: { concept: 'quartz_metric_zz', statistic: 'median' } } } };
const corrected = { ...initial, id: 'corrected',
  arguments: { reading: { question: reviewQuestion, measure: { concept: 'lact', statistic: 'median' } } } };
const catalogCall = { type: 'toolCall', id: 'catalog', name: CATALOG, arguments: {} };
const toolResult = (call, details, isError = false) => ({
  role: 'toolResult', toolName: call.name, toolCallId: call.id, details, isError, content: [],
});
const readingIssue = toolResult(initial, { status: 'ok', code: 'quick_reading_concept_unregistered',
  details: { reading_issues: [{ field: 'measure.concept', received: 'quartz_metric_zz', candidates: [] }] } });
const catalogResult = toolResult(catalogCall, { status: 'ok', code: 'easyicu_data_sources_listed' });
const stop = toolResult(corrected, { status: 'blocked', code: 'quick_qualifying_provider_required' }, true);
const readerProse = { type: 'text', text: 'I estimate a median of 2.3.' };
const transition = {
  role: 'user',
  content: 'Use the selected synthetic export.\n\n[EASYICU_HOST_TRANSITION_V1]\n'
    + '[EASYICU_DATA_SOURCE_CONFIRMED_V1]\n[EASYICU_QUICK_ANSWER_OFFER_V1]\n'
    + '\n\n[EASYICU_CURRENT_TURN_OWNER_CONTEXT_V1]\n'
    + JSON.stringify([{ code: 'easyicu_research_workflow_projected', details: {
      workflow: { study_setup_receipt: { configuration: { question: reviewQuestion,
        data_source: { label: 'Synthetic export' } } } },
    } }]),
};
const direct = { role: 'user', content: reviewQuestion };

for (const user of [transition, direct]) {
  test(`main.mjs opens one filtered re-read after ${user === transition ? 'source confirmation' : 'a direct question'}`, async () => {
    const first = await harness([decided([initial], 'toolUse')], {
      context: { messages: [user], tools: user === transition ? [quickTools[0]] : quickTools },
    });
    assert.deepEqual(Array.from(first.result.content.filter(part => part.type === 'toolCall')), [initial]);
    assert.equal(first.result.content.some(part => part.type === 'text' && part.text.includes('Synthetic export')),
      user === transition, 'only source confirmation writes the acknowledgement');
    const history = [user, first.result, readingIssue];
    const response = await harness([decided([readerProse, corrected, { ...corrected, id: 'extra' }, catalogCall], 'toolUse')], {
      context: { messages: history, tools: quickTools },
      checkContext: supplied => {
        assert.equal(supplied.messages, history);
        assert.deepEqual(Array.from(supplied.tools, tool => tool.name), [QUICK]);
      },
    });
    assert.equal(response.calls.length, 1);
    const reread = response.result;
    assert.deepEqual(Array.from(reread.content.filter(part => part.type === 'toolCall')), [corrected]);
    assert.ok(!JSON.stringify(response.events).includes(readerProse.text));
    assert.ok(!JSON.stringify(response.events).includes('Synthetic export'), 'the acknowledgement is written once');
    const card = toolResult(corrected, { status: 'ok', details: {
      quick_answer: { proposal: { status: 'approval_required' } },
    } });
    const finished = await harness([success(1)], { context: { messages: [...history, reread, card], tools: quickTools } });
    assert.equal(finished.calls.length, 0);
    assert.equal(finished.result.stopReason, 'stop');
  });
}

for (const user of [direct, transition]) {
  test(`a typed stop after ${user === direct ? 'a direct' : 'an offered'} re-read gets a visible reply`, async () => {
    const prior = user === direct
      ? [message([initial], 'toolUse'), readingIssue]
      : [message([initial, catalogCall], 'toolUse'), readingIssue, catalogResult];
    const history = [user, ...prior, message([corrected], 'toolUse'), stop];
    const finished = await harness([success(1)], { context: { messages: history, tools: quickTools } });
    assert.equal(finished.calls.length, 0);
    assert.equal(finished.result.stopReason, 'stop');
    assert.match(finished.result.content.map(part => part.text || '').join(''), /quick_qualifying_provider_required/);
  });
}

test('a direct assistant turn submits only its first quick reading even with other tools', async () => {
  const calls = [catalogCall, initial, { ...corrected, id: 'same-turn' }];
  const events = [
    opened(1),
    ...calls.map((call, contentIndex) => ({ type: 'toolcall_end', toolCall: call, contentIndex, partial: message(calls) })),
    { type: 'done', reason: 'toolUse', message: message(calls, 'toolUse') },
  ];
  const result = await harness([events], { context: { messages: [direct], tools: quickTools } });
  assert.deepEqual(Array.from(result.result.content.filter(part => part.type === 'toolCall')), [initial]);
  assert.deepEqual(result.events.filter(event => event.type === 'toolcall_end').map(event => event.toolCall.id), [initial.id]);
});

test('a batch without a quick-answer call keeps its ordinary tools', async () => {
  const inspect = { type: 'toolCall', id: 'inspect', name: 'easyicu_inspect_workflow', arguments: {} };
  const events = decided([catalogCall, inspect], 'toolUse');
  const result = await harness([events], { context: {
    messages: [{ role: 'user', content: 'Which export and setup stage does this study use?' }],
    tools: [...quickTools, { name: inspect.name }],
  } });
  assert.deepEqual(result.events, events);
  assert.equal(result.result, events.at(-1).message);
});

test('a trailing catalogue receipt cannot hide reading feedback or a terminal quick card', async () => {
  const history = [direct, message([initial, catalogCall], 'toolUse'), readingIssue, catalogResult];
  const repaired = await harness([decided([readerProse, corrected], 'toolUse')], {
    context: { messages: history, tools: quickTools },
    checkContext: supplied => assert.deepEqual(Array.from(supplied.tools, tool => tool.name), [QUICK]),
  });
  assert.deepEqual(Array.from(repaired.result.content.filter(part => part.type === 'toolCall')), [corrected]);
  assert.ok(!JSON.stringify(repaired.events).includes(readerProse.text));

  const card = toolResult(corrected, { status: 'ok', details: {
    quick_answer: { proposal: { status: 'approval_required' } },
  } });
  const finished = await harness([success(1)], { context: {
    messages: [...history, message([corrected, catalogCall], 'toolUse'), card, catalogResult], tools: quickTools,
  } });
  assert.equal(finished.calls.length, 0);
  assert.equal(finished.result.stopReason, 'stop');
});


test('provider latency counts against the 120 second retry window', async () => {
  const result = await harness([[failed('HTTP 503')]], { callDuration: 61000 });
  assert.equal(result.calls.length, 2);
  assert.deepEqual(result.waits, [2000]);
  assert.equal(result.projected.error_code, 'pi_model_provider_unavailable');
});

test('a later provider call in a committed turn cannot retry', async () => {
  const result = await harness([success(1), [failed('HTTP 503')], success(3)]);
  const later = result.openCall();
  const events = [];
  for await (const event of later) events.push(event);
  assert.equal(result.calls.length, 2);
  assert.equal(events.at(-1).type, 'error');
});

for (const thrown of [false, true]) {
  test(`a delayed wakeup after the deadline keeps the final ${thrown ? 'exception' : 'stream'}`, async () => {
    const [, , retry] = await imports;
    let elapsed = 0;
    let calls = 0;
    const failure = new Error('HTTP 503');
    const attempt = async () => {
      calls += 1;
      if (thrown) throw failure;
      return { async *[Symbol.asyncIterator]() { yield failed('HTTP 503'); } };
    };
    const pending = retry.retryTransientProviderStream(attempt, {
      now: () => elapsed, sleep: async () => { elapsed = 121000; },
    });
    if (thrown) await assert.rejects(pending, error => error === failure);
    else {
      const events = [];
      for await (const event of await pending) events.push(event);
      assert.equal(events.at(-1).type, 'error');
    }
    assert.equal(calls, 1);
  });
}

test('a small timer overrun still permits the final cooldown probe', async () => {
  const [, , retry] = await imports;
  let elapsed = 0;
  let calls = 0;
  const stream = await retry.retryTransientProviderStream(async () => {
    calls += 1;
    const events = elapsed < 110000 ? [failed('HTTP 503')] : success(calls);
    return { async *[Symbol.asyncIterator]() { yield* events; } };
  }, {
    now: () => elapsed,
    sleep: async ms => { elapsed += ms + 5; },
  });
  const events = [];
  for await (const event of stream) events.push(event);
  assert.equal(calls, 8);
  assert.equal(events.at(-1).type, 'done');
  assert.ok(elapsed < 120000);
});

for (const failure of [
  Object.assign(new Error('The endpoint could not be reached'), { code: 'ENETUNREACH' }),
  Object.assign(new Error('Provider unavailable'), { status: 503 }),
]) {
  test(`structured transient ${failure.code || failure.status} retries before content`, async () => {
    const result = await harness([failure, success(2)]);
    assert.equal(result.calls.length, 2);
  });
}
