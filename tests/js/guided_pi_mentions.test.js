'use strict';

// "@" mentions are the host's draft: the composer searches the host, resolves
// the chosen item to its exact reference and sends references only. A
// variable's role is chosen on its chip; a Skill mention stays in the draft.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');

const JS = path.resolve(__dirname, '../../src/easyicu/webserver/static/js');
// Values built inside the vm context carry its prototypes; compare their JSON.
const plain = value => JSON.parse(JSON.stringify(value));
// Searches wait on a timer (debounce); let it fire, then the promise chain.
const settle = async () => {
  await new Promise(resolve => setTimeout(resolve, 5));
  for (let i = 0; i < 6; i += 1) await new Promise(resolve => setImmediate(resolve));
};
const SKILL = { kind: 'skill', attachment: { source: 'builtin', id: 'survival-time-to-event', version: 'v', digest: 'e'.repeat(64) } };

function owner(handlers = {}) {
  const declared = {};
  const calls = [];
  const window = {
    EU_LANG: 'zh',
    EasyICU: { guidedPi: { declare: (name, api) => { declared[name] = api; } } },
    EU_CATALOG: { dict: { lactate: ['Lactate', '乳酸', 'mmol/L'] } },
    EU_DATABASE_NAMES: { name: key => ({ miiv: 'MIMIC-IV', eicu: 'eICU-CRD' }[key] || '') },
    EU_API: {
      searchPiMentions: async (sessionId, params) => {
        calls.push(['search', sessionId, plain(params)]);
        return handlers.search ? handlers.search(params) : { items: [], has_more: false, max_mentions: 16 };
      },
      resolvePiMention: async (sessionId, body) => {
        calls.push(['resolve', sessionId, plain(body)]);
        if (handlers.resolve) return handlers.resolve(body);
        const reference = { kind: body.kind, id: body.id, digest: 'a'.repeat(64),
          ...(body.kind === 'concept' ? { role: body.role || 'candidate' } : {}), ...(body.source ? { source: body.source } : {}) };
        const object = body.kind === 'file' ? { kind: 'file', file: 'notes/protocol.md' }
          : body.kind === 'database' ? { kind: 'database', database: 'eicu', label: 'eICU-CRD' } : { kind: body.kind };
        return { reference, object };
      },
      setPiDraftMentions: async (sessionId, body) => {
        calls.push(['draft', sessionId, plain(body)]);
        if (handlers.draft) return handlers.draft(body);
        return { mentions: body.mentions, mentioned_objects: [] };
      },
    },
  };
  const context = vm.createContext({ window, setTimeout, clearTimeout });
  for (const file of ['html-escape.js', 'screens-guided-pi-mentions.js']) {
    vm.runInContext(fs.readFileSync(path.join(JS, file), 'utf8'), context, { filename: file });
  }
  const mentions = declared.mentions;
  let changes = 0;
  mentions.onChange(() => { changes += 1; });
  return { mentions, calls, changes: () => changes };
}

test('a chosen variable is resolved by the host and sent as its reference', async () => {
  const { mentions, calls, changes } = owner({
    search: () => ({ items: [{ kind: 'concept', id: 'lactate', label: 'Lactate' }, { kind: 'concept', id: 'lac_max', label: 'Max lactate' }], has_more: true, max_mentions: 16 }),
  });
  const session = { session_id: 's1', draft_mentions: [SKILL] };
  mentions.search('p1', session, 'concept', '');
  assert.match(mentions.listHtml('p1', session, 'concept'), /正在查找/);
  await settle();
  assert.equal(changes(), 1);
  assert.deepEqual(calls[0], ['search', 's1', { project_id: 'p1', kind: 'concept', query: '', limit: 30 }]);
  const list = mentions.listHtml('p1', session, 'concept');
  assert.match(list, /data-gpi-mention-add="0"><strong>乳酸<\/strong><span>lactate<\/span>/);
  assert.match(list, /继续输入以缩小范围/);

  assert.equal(await mentions.add('p1', session, 0), true);
  assert.deepEqual(calls[1], ['resolve', 's1', { project_id: 'p1', kind: 'concept', id: 'lactate' }]);
  // The draft is replaced whole, and the Skill mention it already held is kept.
  assert.deepEqual(calls[2][2], { project_id: 'p1', mentions: [SKILL, { kind: 'concept', id: 'lactate', digest: 'a'.repeat(64), role: 'candidate' }] });
  const chips = mentions.chips('p1', session);
  // The chip names the variable; the study reading assigns its role.
  assert.match(chips, /<span>变量<\/span><strong>乳酸<\/strong><button type="button" data-gpi-mention-remove="concept:lactate"/);
  assert.doesNotMatch(chips, /<select|data-gpi-mention-role|未指定/);
  assert.doesNotMatch(chips, /survival-time-to-event/);
  // An item already in the draft cannot be chosen twice.
  assert.match(mentions.listHtml('p1', session, 'concept'), /data-gpi-mention-add="0" aria-pressed="true" disabled/);
});

test('a local dataset and a file are named after one lookup; removing keeps the rest', async () => {
  const { mentions, calls, changes } = owner();
  const session = { session_id: 's1', draft_mentions: [
    { kind: 'database', id: 'src_1', source: 'registered_source', digest: 'b'.repeat(64) },
    { kind: 'file', id: 'f'.repeat(32), digest: 'c'.repeat(64) },
    { kind: 'database', id: 'miiv', source: 'database', digest: 'd'.repeat(64) },
  ] };
  let chips = mentions.chips('p1', session);
  assert.match(chips, /<span>本机数据引用<\/span><strong>…<\/strong>/);
  assert.match(chips, /<span>数据库引用<\/span><strong>MIMIC-IV<\/strong>/);
  await settle();
  assert.equal(calls.filter(row => row[0] === 'resolve').length, 2);
  assert.equal(changes(), 2);
  chips = mentions.chips('p1', session);
  assert.match(chips, /<span>本机数据引用<\/span><strong>eICU-CRD<\/strong>/);
  assert.match(chips, /<span>文件<\/span><strong>notes\/protocol\.md<\/strong>/);
  mentions.chips('p1', session);
  await settle();
  assert.equal(calls.filter(row => row[0] === 'resolve').length, 2);

  assert.equal(await mentions.remove('p1', session, 'database:database:miiv'), true);
  assert.deepEqual(calls.at(-1)[2].mentions.map(ref => ref.id), ['src_1', 'f'.repeat(32)]);
});

test('a host refusal reads as words and the draft stays as it was', async () => {
  const refusal = Object.assign(new Error('x'), { status: 409, code: 'mention_database_conflict' });
  const { mentions } = owner({
    search: () => ({ items: [{ kind: 'database', id: 'eicu', source: 'database', label: 'eICU-CRD' }], has_more: false }),
    draft: () => { throw refusal; },
  });
  const before = [{ kind: 'database', id: 'miiv', source: 'database', digest: 'd'.repeat(64) }];
  const session = { session_id: 's1', draft_mentions: before };
  mentions.search('p1', session, 'database', '');
  assert.match(mentions.listHtml('p1', session, 'database'), /引用仅用于描述问题，不会选择或授权研究数据来源/);
  await settle();
  assert.match(mentions.listHtml('p1', session, 'database'), /引用仅用于描述问题，不会选择或授权研究数据来源/);
  assert.equal(await mentions.add('p1', session, 0), false);
  assert.equal(session.draft_mentions, before);
  assert.match(mentions.chips('p1', session), /一条消息只能引用一个数据库/);

  const egress = owner({ search: () => ({ items: [{ kind: 'database', id: 'src_2', source: 'registered_source', label: 'miiv' }] }),
    resolve: () => { throw Object.assign(new Error('x'), { status: 403, code: 'provider_data_handling_unqualified' }); } });
  const fresh = { session_id: 's2', draft_mentions: [] };
  egress.mentions.search('p1', fresh, 'database', '');
  await settle();
  assert.match(egress.mentions.listHtml('p1', fresh, 'database'), /<strong>MIMIC-IV<\/strong><span>本机已登记<\/span>/);
  assert.equal(await egress.mentions.add('p1', fresh, 0), false);
  assert.match(egress.mentions.chips('p1', fresh), /当前模型连接不能接收这份数据/);
});
