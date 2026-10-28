'use strict';
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const test = require('node:test');
const JS = path.resolve(__dirname, '../../src/easyicu/webserver/static/js');
function setup(payload = { items: [] }) {
  const storage = new Map();
  const window = { EU_LANG: 'zh', t: (_en, zh) => zh,
    sessionStorage: { setItem: (key, value) => storage.set(key, value), getItem: key => storage.get(key), removeItem: key => storage.delete(key) },
    EU_API: { getJSON: async () => payload },
    EU_HTML: { esc: value => String(value ?? '').replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/"/g, '&quot;') },
    EasyICU: { guidedPi: { optional: name => name === 'starters' ? { exampleQuestion: id => id === 'example' ? { prompt: '合成问题' } : null } : null } },
  };
  const location = { hash: '#skills' };
  const context = vm.createContext({ window, location });
  vm.runInContext(fs.readFileSync(path.join(JS, 'screens-skills-workflows.js'), 'utf8'), context);
  return { owner: window.EU_SKILL_WORKFLOWS, storage, location, window, context };
}
const fixture = { id: 'synthetic', title: 'Synthetic study', title_zh: '合成流程',
  description: 'Purpose', description_zh: '流程用途', enabled: true, version: 'v1', example: 'example',
  inputs: [{ role: 'exposure', kind: 'categorical' }], outputs: [{ id: 'table:synthetic' }],
  steps: [{ id: 'cohort', label: 'Cohort', kind: 'cohort' }, { id: 'model', label: 'Model', kind: 'model', optional: true }],
  edges: [{ source: 'cohort', target: 'model', products: ['artifact:cohort'] }],
  databases: [{ id: 'synthetic_db', label: 'Synthetic DB', status: 'degraded',
    decision: { substitutions: [{ reason_code: 'mortality_meaning_differs' }], missing: [], unassessed_concepts: [] } }],
};
test('workflow list and detail render the fixture graph and metadata', () => {
  const { owner } = setup();
  const html = owner.render({ items: [fixture] }, 'synthetic');
  assert.match(html, /合成流程/);
  assert.match(html, /流程用途/);
  assert.match(html, /v1/);
  assert.match(html, /所需输入/);
  assert.match(html, /生成的输出/);
  assert.match(html, /用此流程开始/);
  assert.equal((html.match(/data-wf-node=/g) || []).length, 2);
  assert.equal((html.match(/data-wf-edge=/g) || []).length, 1);
  assert.match(html, /data-wf-select="synthetic"/);
  assert.match(html, /data-wf-start="synthetic"/);
  assert.match(html, /tabindex="0"[^>]*title="[^"]*mortality_meaning_differs/);
  assert.match(html, /可选/);
  assert.match(html, /role="tooltip"/);
});
test('start writes the workflow request and queues the existing typed attachment intent', () => {
  const { owner, storage, location } = setup();
  owner.start({ name: 'synthetic', title: '合成流程', kind: 'method', skillLayer: 'research_workflow' }, fixture);
  assert.equal(location.hash, '#guided');
  assert.equal(storage.get('easyicu.skillHub.question'), '请用「合成流程」流程帮我开展一项研究：按流程指南先问我研究问题、数据来源和分析目标，确认配置后再生成研究计划。');
  assert.deepEqual(JSON.parse(storage.get('easyicu.skillHub.method')), {
    id: 'synthetic', title: '合成流程', kind: 'method', layer: 'research_workflow',
    capability_id: '', action_ids: [], included_module_ids: [], method_family: '', method_key: '',
  });
});
test('empty data and unknown selections remain usable; graph text is escaped', () => {
  const { owner } = setup();
  assert.doesNotMatch(owner.render({ items: [] }), /data-wf-start/);
  const html = owner.render({ items: [{ ...fixture, title_zh: '<script>',
    steps: [{ id: 'one', label: '<unsafe>', kind: 'model' }], edges: [] }] }, 'missing');
  assert.match(html, /&lt;script>/);
  assert.doesNotMatch(html, /<unsafe>|<script>/);
  assert.equal((html.match(/data-wf-node=/g) || []).length, 1);
});
test('workflow overview replaces repeated template sections and retains other instructions', () => {
  const { owner } = setup();
  const result = owner.overviewMarkdown('# Skill\n\n## 用途\nPurpose.\n\n## 输出\nGeneric output.\n\n## 步骤与检查\nGeneric steps.\n\n## 各数据库说明\nGeneric databases.\n\n## 参考依据\nReference.\n');
  assert.match(result, /Purpose\./);
  assert.match(result, /Reference\./);
  assert.doesNotMatch(result, /Generic|## 输出|## 步骤与检查|## 各数据库说明/);
  assert.equal(owner.overviewMarkdown('## Outputs\nGeneric trailing section.'), '');
});
test('a workflow start asks for the study inputs instead of copying an example or long instructions', () => {
  const { owner, storage } = setup();
  owner.start({ name: 'synthetic', skillLayer: 'research_workflow', prompt: 'Long instructions.' }, { example: 'example' });
  const request = storage.get('easyicu.skillHub.question');
  assert.match(request, /^请用「synthetic」流程帮我开展一项研究：按流程指南先问我研究问题、数据来源和分析目标/);
  assert.doesNotMatch(request, /合成问题|Long instructions/);
});
test('list selection and the start button use the selected host row', async () => {
  const { owner, storage } = setup({ items: [fixture] });
  await owner.load();
  assert.equal(owner.handle({ dataset: { wfSelect: 'synthetic' } }, []), true);
  assert.match(owner.render({ items: [fixture] }), /aria-pressed="true"/);
  assert.equal(owner.handle({ dataset: {} }, []), false);
  assert.equal(storage.size, 0);
  assert.equal(owner.handle({ dataset: { wfStart: 'synthetic' } }, [{ name: 'synthetic', kind: 'method' }]), true);
  assert.equal(JSON.parse(storage.get('easyicu.skillHub.method')).id, 'synthetic');
  assert.equal(storage.get('easyicu.skillHub.question'), '请用「合成流程」流程帮我开展一项研究：按流程指南先问我研究问题、数据来源和分析目标，确认配置后再生成研究计划。');
});
test('workflow start reaches the removable typed Skill chip without consuming the question', async () => {
  const { owner, window, context, storage } = setup();
  const modules = {};
  window.EasyICU.guidedPi = { declare: (name, api) => { modules[name] = api; }, optional: name => modules[name] || null };
  window.EU_API = {
    loadPiSessionSkills: async () => ({ items: [{ source: 'builtin', id: 'synthetic', version: 'v1', enabled: true, title_zh: '合成流程' }] }),
    setPiDraftSkills: async (_id, body) => ({ attached_skills: body.attached_skills }),
  };
  for (const file of ['screens-guided-pi-skill-attach.js', 'screens-guided-pi-study-workspace.js']) {
    vm.runInContext(fs.readFileSync(path.join(JS, file), 'utf8'), context, { filename: file });
  }
  owner.start({ name: 'synthetic', title: '合成流程', kind: 'method', skillLayer: 'research_workflow' }, {});
  const workspace = modules.studyWorkspace.create({ tr: window.t, esc: window.EU_HTML.esc, iconHtml: () => '' });
  const session = { session_id: 's1', draft_attached_skills: [] };
  workspace.applyHubIntent('p1', session);
  assert.match(storage.get('easyicu.skillHub.question'), /^请用「合成流程」流程帮我开展一项研究/);
  const intent = workspace.takeHubAttachment('p1', session);
  assert.equal(intent.source, 'builtin');
  await modules.skillAttach.toggle('p1', session, intent.source, intent.id);
  assert.equal(session.draft_attached_skills[0].id, 'synthetic');
  assert.match(workspace.renderSkillReference('p1', session), /data-gpi-attached-remove="builtin:synthetic"/);
  await modules.skillAttach.remove('p1', session, 'builtin:synthetic');
  assert.equal(session.draft_attached_skills.length, 0);
});
test('six database chips expose reasons; inputs list roles, not one example\'s variables', () => {
  const { owner } = setup();
  const html = owner.sections({ ...fixture,
    inputs: [{ role: 'outcome', concept: 'synthetic_outcome', label: 'Example outcome', label_zh: '示例结局', kind: 'binary' },
      { role: 'concept', concept: 'a', label_zh: '示例甲' }, { role: 'concept', concept: 'b', label_zh: '示例乙' }],
    databases: Array.from({ length: 6 }, (_, i) => ({ ...fixture.databases[0], id: `synthetic_${i}` })),
  });
  assert.match(html, /<li>结局 · 二分类<\/li>/);
  assert.equal((html.match(/调整变量 · 由研究问题决定/g) || []).length, 1);
  assert.doesNotMatch(html, /示例结局|示例甲|示例乙/);
  assert.equal((html.match(/class="eusk-workflow-database /g) || []).length, 6);
  assert.equal((html.match(/role="tooltip"/g) || []).length, 6);
  assert.equal((html.match(/aria-describedby=/g) || []).length, 6);
});
test('a disabled workflow cannot queue a start intent', async () => {
  const disabled = { ...fixture, enabled: false };
  const { owner, storage } = setup({ items: [disabled] });
  await owner.load();
  assert.match(owner.render({ items: [disabled] }), /data-wf-start="synthetic" disabled/);
  owner.handle({ dataset: { wfStart: 'synthetic' } }, [{ name: 'synthetic' }]);
  assert.equal(storage.size, 0);
});
test('starter examples are read through the existing starters owner', () => {
  const { window, context } = setup();
  const modules = {};
  window.EasyICU.guidedPi.declare = (name, api) => { modules[name] = api; };
  vm.runInContext(fs.readFileSync(path.join(JS, 'screens-guided-pi-starters.js'), 'utf8'), context);
  const row = modules.starters.exampleQuestion('example-lactate-death', window.t);
  assert.equal(row.id, 'example-lactate-death');
  assert.equal(typeof row.prompt, 'string');
  assert.equal(modules.starters.exampleQuestion('unknown', window.t), null);
});
test('a hub start keeps a workflow already attached as a project default', async () => {
  const { window, context } = setup();
  const modules = {};
  window.EasyICU.guidedPi.declare = (name, api) => { modules[name] = api; };
  const writes = [];
  window.EU_API = {
    loadPiSessionSkills: async () => ({ items: [{ source: 'builtin', id: 'synthetic', version: 'v1', enabled: true }] }),
    setPiDraftSkills: async (_id, body) => { writes.push(body); return { attached_skills: body.attached_skills }; },
  };
  vm.runInContext(fs.readFileSync(path.join(JS, 'screens-guided-pi-skill-attach.js'), 'utf8'), context);
  const session = { session_id: 's1', draft_attached_skills: [{ source: 'builtin', id: 'synthetic', version: 'v1' }] };
  assert.equal(await modules.skillAttach.ensure('p1', session, 'builtin', 'synthetic'), true);
  assert.equal(session.draft_attached_skills.length, 1);
  assert.equal(writes.length, 0);
  const fresh = { session_id: 's2', draft_attached_skills: [] };
  assert.equal(await modules.skillAttach.ensure('p1', fresh, 'builtin', 'synthetic'), true);
  assert.equal(fresh.draft_attached_skills[0].id, 'synthetic');
});
test('the graph drops implied edges and routes column-skipping edges between step boxes', () => {
  const { owner } = setup();
  const step = (id, kind) => ({ id, kind, label: `Long instruction for ${id} that is not a name and keeps going past the box`, module_id: 'custom_analysis' });
  const html = owner.graph({ id: 'g', title: 'G', steps: [step('cohort', 'cohort'), step('summary', 'diagnostic'),
    step('audit', 'preflight'), step('fit', 'model'), step('check', 'sensitivity'), step('report', 'report')],
  edges: [['cohort', 'summary'], ['cohort', 'audit'], ['audit', 'fit'], ['cohort', 'fit'], ['fit', 'check'],
    ['check', 'report'], ['summary', 'report'], ['cohort', 'report'], ['fit', 'report']].map(([source, target]) => ({ source, target })) });
  const drawn = [...html.matchAll(/data-wf-edge="([^"]+)" d="([^"]+)"/g)].map(match => [match[1], match[2]]);
  assert.deepEqual(drawn.map(([id]) => id).sort(), ['audit:fit', 'check:report', 'cohort:audit', 'cohort:summary', 'fit:check', 'summary:report']);
  const boxes = [...html.matchAll(/data-wf-node="([^"]+)" transform="translate\((\d+),(\d+)\)"/g)]
    .map(match => ({ id: match[1], x: Number(match[2]), y: Number(match[3]) }));
  const [, skip] = drawn.find(([id]) => id === 'summary:report');
  // The long horizontal run of a skipping edge lies along one channel.
  const points = [...skip.matchAll(/([\d.]+),([\d.]+)/g)].map(match => [Number(match[1]), Number(match[2])]);
  const run = points.slice(1).map((point, i) => [points[i], point]).find(([a, b]) => a[1] === b[1] && b[0] - a[0] > 176);
  assert.ok(run, 'skipping edge has a channel run');
  const [[x1, y], [x2]] = run;
  for (const box of boxes) {
    const overlapsX = box.x < x2 && box.x + 176 > x1;
    assert.ok(!overlapsX || y < box.y || y > box.y + 56, `channel crosses ${box.id}`);
  }
  // Long step instructions never become box names.
  assert.doesNotMatch(html, /Long instruction/);
  assert.match(html, /<title>主分析<\/title>/);
});
