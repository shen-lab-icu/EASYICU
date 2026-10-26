'use strict';

// Request records read as plain words and never as "nothing sent" when they
// are missing; a run's records load once when opened; the problem report
// downloads through the host and says when it failed.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');

const JS = path.resolve(__dirname, '../../src/easyicu/webserver/static/js');
const settle = async () => { for (let i = 0; i < 6; i += 1) await new Promise(resolve => setImmediate(resolve)); };
const VIEW = {
  schema_version: 'easyicu.egress-view/1', status: 'available', reason_code: 'request_metadata_available',
  payload_status: 'withheld',
  requests: [
    { stage: 'pipeline', sequence: 1, purpose: 'planner', provider: 'openrouter', data_classes: ['text', 'schema'], byte_size: 42, state: 'completed', failed: false },
    { stage: 'execution_retry', sequence: 2, purpose: 'coder', provider: 'unknown', data_classes: [], byte_size: 20480, state: 'failed', failed: true },
    { sequence: 3, purpose: 'something_new', provider: 'ollama', data_classes: ['aggregate', 'mystery'], byte_size: null, state: 'in_progress', failed: false },
  ],
};

function harness(overrides = {}) {
  const calls = [];
  const listeners = {};
  const declared = {};
  const nodes = [];
  const window = {
    EU_LANG: 'zh',
    EasyICU: { guidedPi: { declare: (name, api) => { declared[name] = api; } } },
    EU_API: {
      loadRunEgress: async ref => { calls.push(['run', ref]); if (overrides.runFailure) throw overrides.runFailure; return overrides.view || VIEW; },
      loadConversationEgress: async id => { calls.push(['conversation', id]); return { ...VIEW, requests: [] }; },
      downloadProblemBundle: async ref => { calls.push(['bundle', ref]); if (overrides.bundleFailure) throw overrides.bundleFailure; },
      downloadStudyReplay: async ref => { calls.push(['replay', ref]); if (overrides.replayFailure) throw overrides.replayFailure; },
    },
  };
  const document = {
    addEventListener(name, handler) { listeners[name] = handler; },
    querySelectorAll: selector => selector === '[data-gpi-trust-run]' ? nodes : [],
  };
  const context = vm.createContext({ window, document });
  for (const file of ['html-escape.js', 'screens-guided-pi-run-trust.js']) {
    vm.runInContext(fs.readFileSync(path.join(JS, file), 'utf8'), context, { filename: file });
  }
  const section = ref => {
    const body = { innerHTML: '' };
    const node = { dataset: { gpiTrustRun: ref }, open: true, matches: selector => selector === '[data-gpi-trust-run]', querySelector: () => body, body };
    nodes.push(node);
    return node;
  };
  return { trust: declared.runTrust, calls, listeners, section };
}

test('records are plain words: purpose, provider, contents, size and state, never the text', () => {
  const { trust } = harness();
  const html = trust.recordsHtml(VIEW);
  assert.match(html, /这里不显示发出的原文/);
  assert.match(html, /<td>1<\/td><td>制定计划<\/td><td>OpenRouter<\/td><td>文字、数据结构<\/td><td class="gpi-trust-size">42 B<\/td><td>已完成<\/td>/);
  assert.match(html, /<tr class="is-failed"><td>2<\/td><td>编写分析代码 · 重试<\/td><td>未记录<\/td><td>未记录<\/td><td class="gpi-trust-size">20 KB<\/td><td>失败<\/td>/);
  // Unknown codes stay unnamed rather than guessed; the size is not invented.
  assert.match(html, /<td>3<\/td><td>未记录<\/td><td>Ollama（本机）<\/td><td>汇总统计<\/td><td class="gpi-trust-size">—<\/td><td>已发起<\/td>/);
});

test('a missing or unverifiable record never reads as nothing sent', () => {
  const { trust } = harness();
  assert.match(trust.recordsHtml({ status: 'not_recorded', requests: [] }), /没有发送记录。较早的版本不记录这些信息，所以这不代表没有发送过。/);
  assert.match(trust.recordsHtml({ status: 'unavailable', requests: [] }), /发送记录无法核对。/);
  assert.match(trust.recordsHtml({ status: 'available', requests: [] }), /还没有向模型发送过请求。/);
});

test('a run’s records load once when its section opens, and stay for the next render', async () => {
  const { trust, calls, listeners, section } = harness();
  assert.match(trust.sectionHtml('rr_1', false), /<details class="gpi-run-trust" data-gpi-trust-run="rr_1"><summary>发送给模型的记录<\/summary><div data-gpi-trust-body><\/div><\/details>/);
  const node = section('rr_1');
  listeners.toggle({ target: node });
  await settle();
  listeners.toggle({ target: node });
  await settle();
  assert.deepEqual(calls, [['run', 'rr_1']]);
  assert.match(node.body.innerHTML, /制定计划/);
  assert.match(trust.sectionHtml('rr_1', true), /<details class="gpi-run-trust" data-gpi-trust-run="rr_1" open>.*制定计划/s);
});

test('a lapsed run list asks for a refresh and is read again next time', async () => {
  const { trust, calls, listeners, section } = harness({ runFailure: Object.assign(new Error('x'), { code: 'run_ref_unknown' }) });
  const node = section('rr_2');
  listeners.toggle({ target: node });
  await settle();
  assert.match(node.body.innerHTML, /运行列表已过期，请刷新状态后再试。/);
  listeners.toggle({ target: node });
  await settle();
  assert.equal(calls.filter(row => row[0] === 'run').length, 2);
  assert.doesNotMatch(trust.sectionHtml('rr_2', true), /过期/);
});

test('the problem report downloads by run_ref and says when it failed', async () => {
  const ok = harness();
  assert.match(ok.trust.problemButtonHtml('rr_1'), /<button type="button" class="btn sm" data-gpi-problem-bundle="rr_1">下载问题报告<\/button>/);
  assert.match(ok.trust.problemButtonHtml('rr_1', ''), /<button type="button" data-gpi-problem-bundle="rr_1">/);
  assert.equal(ok.trust.problemButtonHtml(''), '');
  const button = { dataset: { gpiProblemBundle: 'rr_1' }, disabled: false, textContent: '下载问题报告' };
  ok.listeners.click({ target: { closest: () => button }, preventDefault() {} });
  await settle();
  assert.deepEqual(ok.calls, [['bundle', 'rr_1']]);
  assert.equal(button.textContent, '下载问题报告');

  const failing = harness({ bundleFailure: new Error('x') });
  const other = { dataset: { gpiProblemBundle: 'rr_1' }, disabled: false, textContent: '下载问题报告' };
  failing.listeners.click({ target: { closest: () => other }, preventDefault() {} });
  await settle();
  assert.equal(other.textContent, '下载失败，点此重试');
  assert.equal(other.disabled, false);
});

test('a replay downloads by run_ref; a refusal is said in plain words beside the actions', async () => {
  const refusal = Object.assign(new Error('x'), { code: 'replay_small_cell_forbidden' });
  const { trust, calls, listeners } = harness({ replayFailure: refusal });
  assert.match(trust.replayButtonHtml('rr_1'), /<button type="button" data-gpi-study-replay="rr_1">下载研究回放<\/button>/);
  const inserted = [];
  const actions = { insertAdjacentHTML: (where, html) => inserted.push([where, html]), parentElement: { querySelector: () => null } };
  const button = { dataset: { gpiStudyReplay: 'rr_1' }, disabled: false, parentElement: actions };
  listeners.click({ target: { closest: selector => selector === '[data-gpi-study-replay]' ? button : null }, preventDefault() {} });
  await settle();
  assert.deepEqual(calls, [['replay', 'rr_1']]);
  assert.equal(inserted[0][0], 'afterend');
  assert.match(inserted[0][1], /role="alert" data-gpi-study-replay-refusal>有分组少于 20 人，按数据使用协议不能分享这份回放。<\/p>/);
  assert.equal(button.disabled, false);
  // Every exporter code reads as a sentence; an unknown one still says no replay was made.
  assert.equal(trust.replayRefusal('replay_figure_uncleared'), '有图没有通过外传检查，不能生成回放。');
  assert.equal(trust.replayRefusal('replay_digest_mismatch'), '运行记录不完整或已改动，不能生成回放。');
  assert.equal(trust.replayRefusal('something_else'), '无法生成回放。');
});
