'use strict';
const assert = require('node:assert/strict');
const fs = require('node:fs');
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

const imports = Promise.all([
  import(pathToFileURL(budgetFile)),
  import(pathToFileURL(projectionFile)),
  import(pathToFileURL(retryFile)),
]);

async function harness(attempts, config = {}) {
  const [{ ShellBudgetGuard }, projection, retry] = await imports;
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
  const calls = [];
  const waits = [];
  const order = [];
  const authorize = guard.authorize.bind(guard);
  guard.authorize = (...args) => { order.push('authorize'); return authorize(...args); };
  const record = { budgetGuard: guard, language: 'zh' };
  const session = { agent: {} };
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
    const attempt = attempts[Math.min(calls.length - 1, attempts.length - 1)];
    if (attempt instanceof Error) throw attempt;
    return {
      async *[Symbol.asyncIterator]() {
        for (const event of attempt) {
          await config.beforeEvent?.(event);
          if (event instanceof Error) throw event;
          yield event;
        }
      },
      result: async () => attempt.at(-1)?.error || attempt.at(-1)?.message,
    };
  };
  const retryWithFastSleep = (openAttempt, options) => retry.retryTransientProviderStream(openAttempt, {
      ...options,
      sleep: async (ms, value, sleepOptions) => {
        waits.push(ms);
        await config.onWait?.(sleepOptions?.signal);
        sleepOptions?.signal?.throwIfAborted();
      },
    });
  vm.runInNewContext(wrapper, {
    session, record, manager, originalStreamFunction, lazyStream,
    hostPostToolFinalization: () => config.finalization || null,
    ...retry, retryTransientProviderStream: retryWithFastSleep,
  }, { filename: mainFile });
  const events = [];
  const stream = session.agent.streamFunction(model, context, { maxTokens: 1000, ...config.options });
  for await (const event of stream) {
    events.push(event);
    config.onEvent?.(event, calls);
  }
  const result = await stream.result();
  const projected = projection.normalizePiEvent({ type: 'message_end', message: result });
  return { guard, receipts, calls, waits, order, events, result, projected,
    openCall: () => session.agent.streamFunction(model, context, { maxTokens: 1000 }),
  };
}

test('empty HTTP 500 retries once through the guard and exposes only attempt two', async () => {
  const second = success(2);
  const result = await harness([[opened(1), failed()], second]);
  assert.equal(result.calls.length, 2);
  assert.deepEqual(result.events, second);
  assert.deepEqual(result.receipts.map(row => row.provider_calls), [1, 2]);
  assert.deepEqual(result.order, ['authorize', 'receipt', 'provider', 'authorize', 'receipt', 'provider']);
  assert.deepEqual(result.waits, [1500]);
  assert.equal(result.guard.state().message_provider_calls, 2);
  assert.equal(result.projected.error_code, undefined);
  assert.match(source, /retry: \{ enabled: false, maxRetries: 0 \}/);
});

for (const detail of ['HTTP 502: bad gateway', 'HTTP 503: unavailable', 'HTTP 504: gateway timeout',
  'connection reset by peer', 'request timeout', 'request timed out', 'read ECONNRESET', 'connect ETIMEDOUT',
  'dial tcp 203.0.113.1:443: connect: operation timed out']) {
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
]) {
  test(`${detail} is never retried`, async () => {
    const first = [opened(1), failed(detail)];
    const result = await harness([first, success(2)]);
    assert.equal(result.calls.length, 1);
    assert.deepEqual(result.events, first);
    assert.deepEqual(result.waits, []);
    assert.equal(result.projected.error_code, code);
  });
}

test('a second HTTP 500 ends the call with the existing projected error', async () => {
  const second = [opened(2), failed('HTTP 500: second request failed')];
  const result = await harness([[opened(1), failed()], second, success(3)]);
  assert.equal(result.calls.length, 2);
  assert.deepEqual(result.events, second);
  assert.equal(result.projected.error_code, 'pi_model_provider_unavailable');
  assert.equal(result.result, second.at(-1).error);
});

for (const stage of ['open', 'iterate']) {
  test(`a transient exception during ${stage} retries once`, async () => {
    const error = new Error('HTTP 500: transient request failure');
    const result = await harness([stage === 'open' ? error : [opened(1), error], success(2)]);
    assert.equal(result.calls.length, 2);
    assert.deepEqual(result.events, success(2));
    assert.equal(result.receipts.length, 2);
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
    assert.deepEqual(result.order, ['authorize', 'receipt', 'provider', 'authorize']);
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

test('each call within a message has its own single retry', async () => {
  const result = await harness([[failed()], success(2), [failed()], success(4)]);
  const events = [];
  for await (const event of result.openCall()) events.push(event);
  assert.equal(result.calls.length, 4);
  assert.deepEqual(events, success(4));
  assert.deepEqual(result.receipts.map(row => row.provider_calls), [1, 2, 3, 4]);
  assert.deepEqual(result.waits, [1500, 1500]);
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

test('abort during the delay prevents a second authorization', async () => {
  const controller = new AbortController();
  const result = await harness([[opened(1), failed()], success(2)], {
    options: { signal: controller.signal }, onWait: () => controller.abort(),
  });
  assert.equal(result.calls.length, 1);
  assert.equal(result.receipts.length, 1);
});

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
