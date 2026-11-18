'use strict';

const assert = require('node:assert/strict');
const path = require('node:path');
const { test } = require('node:test');
require('./guided_pi_module_harness.cjs');

const AUTH_REQUIRED = 'research_pipeline_codex_user_auth_required';
const FINISHED_AT = '2026-01-05T18:00:00Z';
const VERIFIED_AT = '2026-01-05T19:00:00Z';
const job = {
  job_id: 'handover_codex',
  kind: 'agent-run',
  status: 'failed',
  error_code: AUTH_REQUIRED,
  finished_at_epoch: Date.parse(FINISHED_AT) / 1000,
  next_actions: [{
    kind: 'check_connection',
    action_id: 'check_connection_0123456789abcdef',
    label_key: 'stop_next_actions.check_connection',
    params: { route: 'model_connection' },
  }],
};

function status(accountStatus, verifiedAt) {
  const verified = accountStatus === 'codex_auth_ready';
  return { ok: true, auth: {
    schema_version: 'easyicu.codex-user-auth-status/1',
    provider: 'codex',
    provider_identity: 'codex-app-server',
    authentication_mode: 'chatgpt_account',
    session_present: true,
    authentication_verified: verified,
    account_session_present: verified,
    account_session_status: accountStatus,
    plan_type: verified ? 'plus' : null,
    account_label: verified ? 'r***@example.org' : null,
    secrets_returned: false,
    ...(verifiedAt === undefined ? {} : { verified_at: verifiedAt }),
  } };
}

const pending = status('codex_auth_login_pending');
const ready = status('codex_auth_ready');

function harness(nextWorkflow = { next_action_code: 'provider_ready_to_generate_plan' }) {
  const state = {
    codexAuth: null, codexLogin: null, codexPoll: null,
    workflow: { next_action_code: AUTH_REQUIRED },
    session: { research_provider: { provider: 'codex' }, archived_child_jobs: [job] },
  };
  const timers = new Map();
  const events = {};
  const renders = [];
  let timerId = 0;
  let loads = 0;
  let payload = pending;
  window = {
    setTimeout(callback) { timers.set(++timerId, callback); return timerId; },
    clearTimeout(id) { timers.delete(id); },
  };
  for (const file of process.argv.slice(2)) {
    const resolved = path.resolve(file);
    delete require.cache[resolved];
    require(resolved);
  }
  const modules = window.EasyICU.guidedPi;
  const currentStop = () => modules.require('runStop').latestJobStop(state.session, state.workflow, {
    codexAuth: state.codexAuth,
  });
  const provider = modules.require('providerControl').create({
    state,
    api: () => ({
      loadPiCopilotStatus: async () => ({ runtime: {} }),
      startPiCopilotCodexLogin: async () => ({ ...pending, login_started: true, flow: 'browser' }),
      loadPiCopilotCodexStatus: async () => {
        if (payload instanceof Error) throw payload;
        return payload;
      },
    }),
    async loadWorkflow() { loads++; state.workflow = nextWorkflow; },
    render: () => renders.push(currentStop()),
    errorText: error => error.message,
  });
  return {
    state, timers, renders, currentStop, provider, workflowLoads: () => loads,
    async refresh(result) {
      payload = result;
      await provider.loadCodexResearchStatus(true);
    },
    async poll(result) {
      payload = result;
      const [id, callback] = timers.entries().next().value;
      timers.delete(id);
      await callback();
    },
    clickWith(callbacks) {
      state.host = { addEventListener: (name, handler) => { events[name] = handler; } };
      modules.require('events').create({
        state, RUN_FILES: { handleClick: () => false }, ...callbacks,
      }).wire();
      const button = { dataset: {
        gpiStopAction: job.next_actions[0].action_id,
        gpiStopJob: job.job_id,
        gpiStopWorkflow: AUTH_REQUIRED,
      } };
      events.click({ target: {
        closest: selector => selector === '[data-gpi-stop-action]' ? button : null,
      } });
    },
  };
}

test('pending sign-in refreshes the workflow before withdrawing the stop', async () => {
  const h = harness();
  await h.provider.startCodexLogin('browser');
  assert.ok(h.state.codexLogin);
  assert.equal(h.timers.size, 1);
  assert.equal(h.currentStop(), job);

  await h.poll(ready);
  assert.equal(h.workflowLoads(), 1, 'verification refreshes the workflow through its owner');
  assert.ok(Date.parse(h.state.codexAuth.verified_at) > Date.parse(FINISHED_AT));
  assert.equal(h.state.codexLogin, null);
  assert.equal(h.timers.size, 0);
  assert.equal(h.currentStop(), null);
  assert.equal(h.renders.at(-1), null, 'the next render already sees the refreshed workflow');
});

test('a fresh tab uses server evidence and never dates a routine ready response', async () => {
  const h = harness();
  await h.refresh(ready);
  assert.equal(h.state.codexAuth.verified_at, undefined);
  assert.equal(h.workflowLoads(), 0);

  await h.refresh(status('codex_auth_ready', VERIFIED_AT));
  assert.equal(h.state.codexAuth.verified_at, VERIFIED_AT);
  assert.equal(h.currentStop(), job, 'the workflow still owns a current auth stop');
  h.state.workflow = { next_action_code: 'provider_ready_to_generate_plan' };
  for (const response of [ready, { ...ready, auth: null }, null, new Error('status unavailable'), ready]) {
    await h.refresh(response);
    assert.equal(h.state.codexAuth.verified_at, VERIFIED_AT);
    assert.equal(h.currentStop(), null);
  }
  assert.equal(h.workflowLoads(), 0, 'routine and transient polls do not refresh the workflow');
});

test('a verified in-tab sign-in uses the server time even when it predates the stop', async () => {
  const h = harness();
  const serverTime = '2026-01-05T17:00:00Z';
  await h.provider.startCodexLogin('browser');
  await h.refresh(status('codex_auth_ready', serverTime));
  assert.equal(h.state.codexAuth.verified_at, serverTime);
  assert.equal(h.currentStop(), job, 'an older server timestamp cannot supersede the stop');
  assert.equal(h.workflowLoads(), 1);
});

test('a reload during a pending sign-in polls and refreshes without inventing a timestamp', async () => {
  const h = harness();
  await h.refresh(pending);
  await h.poll({ ok: true, auth: null });
  assert.equal(h.timers.size, 1);
  await h.poll(ready);
  assert.equal(h.state.codexAuth.verified_at, undefined);
  assert.equal(h.workflowLoads(), 1);
  assert.equal(h.timers.size, 0);
  assert.equal(h.currentStop(), job);
});

test('a locally dated verification survives routine and transient polls', async () => {
  const h = harness();
  await h.provider.startCodexLogin('browser');
  await h.refresh(ready);
  const verifiedAt = h.state.codexAuth.verified_at;
  assert.ok(verifiedAt);
  await h.refresh(null);
  await h.refresh(ready);
  assert.equal(h.state.codexAuth.verified_at, verifiedAt);
  assert.equal(h.workflowLoads(), 1);
  assert.equal(h.currentStop(), null);
});

test('a newer verification withdraws a stop when the workflow code is missing or empty', async () => {
  for (const workflow of [null, {}, { next_action_code: null }, { next_action_code: '' }]) {
    const h = harness(workflow);
    await h.provider.startCodexLogin('browser');
    await h.poll(status('codex_auth_ready', VERIFIED_AT));
    assert.equal(h.currentStop(), null);
  }
});

test('the re-login action requires both callbacks and an idle provider', () => {
  const h = harness();
  const calls = [];
  const popup = {};
  const openAuthorizationPopup = () => { calls.push('popup'); return popup; };
  const startCodexLogin = (flow, target) => calls.push([flow, target]);
  h.clickWith({ openAuthorizationPopup });
  h.clickWith({ startCodexLogin });
  h.clickWith({ startCodexLogin: true, openAuthorizationPopup });
  h.clickWith({ startCodexLogin, openAuthorizationPopup: true });
  h.state.codexBusy = true;
  h.clickWith({ startCodexLogin, openAuthorizationPopup });
  assert.deepEqual(calls, []);
  h.state.codexBusy = false;
  h.clickWith({ startCodexLogin, openAuthorizationPopup });
  assert.deepEqual(calls, ['popup', ['browser', popup]]);
});
