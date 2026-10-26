'use strict';

// Skills attached to the next message are the host's draft: the composer sends
// exact identities from the conversation's catalogue, never Skill text, and a
// project's defaults change only when the researcher saves them.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');

const JS = path.resolve(__dirname, '../../src/easyicu/webserver/static/js');
// Values built inside the vm context carry its prototypes; compare their JSON.
const plain = value => JSON.parse(JSON.stringify(value));
const CATALOG = [
  { source: 'builtin', id: 'survival-time-to-event', title: 'Survival analysis', title_zh: '生存分析（时间到事件）',
    category: 'Survival', enabled: true, version: 'easyicu.method-skills/3' },
  { source: 'builtin', id: 'clinical-risk-prediction', title: 'Risk prediction', title_zh: '临床风险预测模型',
    category: 'Prediction', enabled: false, version: 'easyicu.method-skills/3' },
  { source: 'builtin', id: 'cohort-characterization-table-one', title: 'Table 1', title_zh: '队列描述与 Table 1',
    category: 'Descriptive', enabled: true, version: 'easyicu.method-skills/3' },
  { source: 'user', id: 'my-notes', title: 'my-notes', title_zh: '', category: 'user', enabled: true, digest: 'd'.repeat(64) },
];

function owner(handlers = {}) {
  const declared = {};
  const calls = [];
  const window = {
    EU_LANG: 'zh',
    EasyICU: { guidedPi: { declare: (name, api) => { declared[name] = api; } } },
    EU_API: {
      loadPiSessionSkills: async (...args) => { calls.push(['catalog', ...args]); return { items: CATALOG, max_attached_skills: 2 }; },
      setPiDraftSkills: async (sessionId, body) => {
        calls.push(['draft', sessionId, body]);
        if (handlers.draft) return handlers.draft(body);
        return { attached_skills: body.attached_skills };
      },
      loadProjectDefaultSkills: async projectId => { calls.push(['defaults', projectId]); return { attached_skills: handlers.defaults || [] }; },
      setProjectDefaultSkills: async (projectId, body) => { calls.push(['save-defaults', projectId, body]); return { attached_skills: body.attached_skills }; },
    },
  };
  const context = vm.createContext({ window });
  for (const file of ['html-escape.js', 'screens-guided-pi-skill-attach.js']) {
    vm.runInContext(fs.readFileSync(path.join(JS, file), 'utf8'), context, { filename: file });
  }
  return { attach: declared.skillAttach, calls };
}

test('attaching sends the catalogue identity, and the chip names the Skill', async () => {
  const { attach, calls } = owner();
  const session = { session_id: 's1', draft_attached_skills: [] };
  assert.equal(await attach.toggle('p1', session, 'builtin', 'survival-time-to-event'), true);
  assert.deepEqual(plain(calls.find(row => row[0] === 'draft')[2]), {
    project_id: 'p1',
    attached_skills: [{ source: 'builtin', id: 'survival-time-to-event', version: 'easyicu.method-skills/3' }],
  });
  assert.equal(attach.attached(session, 'builtin', 'survival-time-to-event'), true);
  assert.equal(attach.hasDraft(session), true);
  const html = attach.chips(session);
  assert.match(html, /<span>技能<\/span><strong>生存分析（时间到事件）<\/strong>/);
  assert.match(html, /data-gpi-attached-remove="builtin:survival-time-to-event"/);
  await attach.toggle('p1', session, 'user', 'my-notes');
  assert.deepEqual(plain(session.draft_attached_skills[1]), { source: 'user', id: 'my-notes', digest: 'd'.repeat(64) });
  assert.match(attach.chips(session), /<span>我的技能<\/span><strong>my-notes<\/strong>/);
});

test('picking an attached Skill again or its × removes it from the draft', async () => {
  const { attach } = owner();
  const session = { session_id: 's1', draft_attached_skills: [] };
  await attach.toggle('p1', session, 'builtin', 'survival-time-to-event');
  await attach.toggle('p1', session, 'builtin', 'survival-time-to-event');
  assert.deepEqual(plain(session.draft_attached_skills), []);
  assert.equal(attach.chips(session), '');
  await attach.toggle('p1', session, 'user', 'my-notes');
  await attach.remove('p1', session, 'user:my-notes');
  assert.deepEqual(plain(session.draft_attached_skills), []);
});

test('a disabled, unknown or over-limit Skill is refused in words, without a host write', async () => {
  const { attach, calls } = owner();
  const session = { session_id: 's1', draft_attached_skills: [] };
  assert.equal(await attach.toggle('p1', session, 'builtin', 'clinical-risk-prediction'), false);
  assert.match(attach.chips(session), /这个技能未启用，请先在「技能」里启用。/);
  assert.equal(await attach.toggle('p1', session, 'builtin', 'not-a-skill'), false);
  assert.match(attach.chips(session), /这个技能在本对话中不可用。/);
  await attach.toggle('p1', session, 'builtin', 'survival-time-to-event');
  await attach.toggle('p1', session, 'user', 'my-notes');
  assert.equal(await attach.toggle('p1', session, 'builtin', 'cohort-characterization-table-one'), false);
  assert.match(attach.chips(session), /一条消息最多挂 2 个技能。/);
  assert.equal(calls.filter(row => row[0] === 'draft').length, 2);
});

test('a host refusal keeps the draft and says why', async () => {
  const { attach } = owner({ draft: () => { throw Object.assign(new Error('x'), { code: 'skill_attachment_family_conflict' }); } });
  const session = { session_id: 's1', draft_attached_skills: [] };
  assert.equal(await attach.toggle('p1', session, 'builtin', 'survival-time-to-event'), false);
  assert.deepEqual(plain(session.draft_attached_skills), []);
  assert.match(attach.chips(session), /role="alert">这些技能属于不同的研究类型，一条消息只能选一种。/);
});

test('project defaults are offered from the current picks and saved only on request', async () => {
  const { attach, calls } = owner({ defaults: [] });
  const session = { session_id: 's1', draft_attached_skills: [] };
  await attach.loadDefaults('p1');
  assert.equal(attach.defaultsControl('p1', session), '');
  await attach.toggle('p1', session, 'builtin', 'survival-time-to-event');
  assert.match(attach.defaultsControl('p1', session), /data-gpi-attached-defaults[^>]*>设为本项目新对话的默认技能/);
  assert.equal(calls.some(row => row[0] === 'save-defaults'), false);
  await attach.saveDefaults('p1', session);
  assert.deepEqual(plain(calls.find(row => row[0] === 'save-defaults').slice(1)), ['p1', {
    attached_skills: [{ source: 'builtin', id: 'survival-time-to-event', version: 'easyicu.method-skills/3' }],
  }]);
  assert.match(attach.defaultsControl('p1', session), /本项目新对话默认带上/);
  // A new conversation that starts with the defaults can drop them for the project.
  const fresh = { session_id: 's2', draft_attached_skills: [] };
  assert.match(attach.defaultsControl('p1', fresh), /本项目新对话不再默认带技能/);
});
