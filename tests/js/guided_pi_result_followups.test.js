'use strict';

// A finished result lists the host's follow-ups first, with why, and only
// those its own additions can run; other actions are not offered as buttons.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');

const JS = path.resolve(__dirname, '../../src/easyicu/webserver/static/js');

function owner(payload) {
  const declared = {};
  const window = { EasyICU: { guidedPi: { declare: (name, api) => { declared[name] = api; } } } };
  const context = vm.createContext({ window });
  vm.runInContext(fs.readFileSync(path.join(JS, 'screens-guided-pi-result-append.js'), 'utf8'), context, { filename: 'append' });
  const esc = value => String(value ?? '').replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/"/g, '&quot;');
  const proposals = [];
  return { append: declared.resultAppend.create({
    tr: (en, zh) => zh, esc, projectId: () => 'p1', host: () => null, session: () => ({ session_id: 's1' }),
    canPreview: () => true, onError: () => {}, errorText: String,
    api: () => ({
      loadPiCopilotAppendOptions: async () => payload,
      proposePiCopilotAppend: async (sessionId, body) => { proposals.push([sessionId, JSON.parse(JSON.stringify(body))]); },
    }),
  }), proposals };
}

const OPTIONS = {
  run_id: 'r1', source_run_sha256: 'a'.repeat(64),
  options: [
    { kind: 'sensitivity', available: true, label_en: 'Sensitivity', label_zh: '敏感性分析' },
    { kind: 'subgroup', available: true, label_en: 'Subgroup', label_zh: '亚组分析' },
    { kind: 'article', available: true, label_en: 'Article', label_zh: '完整论文' },
  ],
  followups: { suggestions: [
    { kind: 'prespecified_sensitivity', target_action: 'append', reason_codes: ['prespecified_not_run', 'append_owner_compiled'], delta: { action: 'append', kind: 'sensitivity' } },
    { kind: 'functional_form_check', target_action: 'append', reason_codes: ['sealed_linear_covariate'], delta: { action: 'append', kind: 'sensitivity' } },
    { kind: 'same_study_other_database', target_action: 'other_database', reason_codes: [], delta: { action: 'other_database' } },
    { kind: 'complementary_survival', target_action: 'new_study', reason_codes: ['paired_event_time_available'], delta: { action: 'new_study' } },
    { kind: 'named_subgroup', target_action: 'append', reason_codes: ['design_named_subgroup_not_run'], delta: { action: 'append', kind: 'subgroup' } },
  ], unavailable_actions: ['multi_database_unavailable'] },
};

test('host follow-ups lead with their reason; other actions are not offered', async () => {
  const { append } = owner(OPTIONS);
  await append.load({ run_id: 'r1' });
  const html = append.render({ run_id: 'r1' });
  assert.match(html, /<ul class="gpi-result-followups"><li><button class="btn sm" type="button" data-gpi-result-append="sensitivity">补做方案预设的敏感性分析<\/button><span>方案里预设了，但还没做。<\/span><\/li><li><button[^>]*data-gpi-result-append="subgroup">分析设计中写明的亚组<\/button>/);
  assert.equal((html.match(/data-gpi-result-append="sensitivity"/g) || []).length, 1, 'one button per addition kind');
  assert.doesNotMatch(html, /生存分析|other_database/);
  assert.match(html, /data-gpi-result-append="article">完整论文<\/button>/, 'the remaining offered additions stay');
});

test('a suggested follow-up is proposed through the ordinary addition with the source digest', async () => {
  const { append, proposals } = owner(OPTIONS);
  await append.load({ run_id: 'r1' });
  assert.equal(await append.propose('subgroup'), true);
  assert.deepEqual(proposals, [['s1', { project_id: 'p1', run_id: 'r1', kind: 'subgroup', expected_source_sha256: 'a'.repeat(64) }]]);
});

test('a follow-up whose addition is not offered is not shown', async () => {
  const { append } = owner({ ...OPTIONS, options: [{ kind: 'article', available: true, label_en: 'Article', label_zh: '完整论文' }] });
  await append.load({ run_id: 'r1' });
  assert.doesNotMatch(append.render({ run_id: 'r1' }), /gpi-result-followups/);
});
