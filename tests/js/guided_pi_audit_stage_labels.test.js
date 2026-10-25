/* Synthetic stage receipts; no research execution or artifact access. */
'use strict';
const assert = require('node:assert/strict');
const path = require('node:path');
const { test } = require('node:test');

global.window = global;
const js = path.resolve('src/easyicu/webserver/static/js');
for (const name of ['screens-guided-pi-modules.js', 'html-escape.js',
  'screens-guided-pi-study-workspace.js', 'screens-guided-pi-aside.js']) require(path.join(js, name));
const modules = EasyICU.guidedPi;

function render(runs, language, extra = {}) {
  const head = { innerHTML: '' };
  const body = { innerHTML: '', querySelector: () => null };
  global.document = { getElementById: id => id === 'gdAsideBody' ? body
    : id === 'gdStudyAside' ? { querySelector: () => head } : null };
  const workflow = { current_stage: 'plan', stages: [{ id: 'plan', status: 'blocked' }], runs, ...extra };
  const before = JSON.stringify(workflow);
  modules.require('aside').create({
    tr: (en, zh) => language === 'zh' ? zh : en, esc: EU_HTML.esc, iconHtml: () => '',
    projectId: () => 'synthetic-project', demoMode: () => false, shell: () => 'pi',
    workflow: () => workflow, manuscriptNotPassedText: () => 'Draft needs revision',
  }).syncProjectWorkflowAside();
  assert.equal(JSON.stringify(workflow), before, 'Rendering must not mutate authority');
  assert.doesNotMatch(body.innerHTML, /data-gpi-resource-kind/, 'History must not invent file actions');
  return body.innerHTML;
}

for (const language of ['zh', 'en']) {
  const pick = (zh, en) => language === 'zh' ? zh : en;
  test(`${language}: replies, legacy completion and research stops have distinct labels`, () => {
    const owner = modules.require('studyWorkspace').create({ tr: (en, zh) => pick(zh, en) });
    for (const row of [{ conversation_status: 'replied' }, { last_turn_status: 'done' }]) {
      assert.equal(owner.sessionStatusLabel(row), pick('已回复', 'Replied'));
    }
    assert.equal(owner.sessionStatusLabel({ conversation_status: 'done' }), pick('任务已结束', 'Task ended'));
    for (const [state, label] of [['needs_user', pick('待确认', 'Needs you')], ['stopped', pick('已停止', 'Stopped')]]) {
      assert.equal(owner.sessionStatusLabel({ conversation_status: state, last_turn_status: 'done' }), label);
      assert.equal(owner.sessionStatusLabel({ conversation_status: state, active_message_job_id: 'reply' }), pick('进行中', 'Running'));
    }
  });
  const cases = [
    ['planning failed', { run_status: 'failed', gate_detail_code: 'research_pipeline_plan_contract_exhausted' }, '计划生成 · 失败', 'Plan generation · Failed'],
    ['planner unavailable', { run_status: 'failed', gate_reason_code: 'research_pipeline_planner_provider_unavailable' }, '计划生成 · 失败', 'Plan generation · Failed'],
    ['data preparation failed', { run_status: 'failed', gate_reason_code: 'data_foundation_blocked' }, '数据准备 · 失败', 'Data preparation · Failed'],
    ['pending plan', { run_status: 'human_review_pending', plan_available: true }, '计划生成 · 待审阅', 'Plan generation · Awaiting review'],
    ['pending gate', { gate_status: 'blocked', gate_reason_code: 'human_plan_review_required' }, '计划生成 · 待审阅', 'Plan generation · Awaiting review'],
    ['completed analysis', { analysis_outcome: { status: 'completed' }, manuscript_outcome: { status: 'not_passed' } }, '分析完成', 'Analysis complete'],
    ['historical completion receipt', { gate_status: 'analysis_only', gate_reason_code: 'research_agent_pipeline_complete_human_interpretation_required' }, '分析完成', 'Analysis complete'],
    ['inconsistent completion receipt', { gate_status: 'blocked', gate_reason_code: 'research_agent_pipeline_complete_human_interpretation_required' }, '研究任务 · 阻断', 'Research task · Blocked'],
    ['execution failed', { run_status: 'failed', gate_detail_code: 'research_pipeline_execution_failed' }, '分析 · 失败', 'Analysis · Failed'],
    ['full request is not execution', { run_status: 'failed', plan_available: true, artifact_names: ['result_tables.json', 'figure_gallery.json'] }, '研究任务 · 失败', 'Research task · Failed'],
    ['timeout has no phase', { run_status: 'failed', gate_reason_code: 'research_pipeline_provider_timeout' }, '研究任务 · 失败', 'Research task · Failed'],
    ['explicit pending outcome is not completion', { run_status: 'pass', gate_status: 'analysis_only', analysis_outcome: { status: 'pending' }, gate_reason_code: 'research_agent_pipeline_complete_human_interpretation_required' }, '研究任务 · 已通过', 'Research task · Passed'],
    ['missing receipt', { run_status: 'pass' }, '研究任务 · 已通过', 'Research task · Passed'],
    ['pending beats completion', { run_status: 'human_review_pending', analysis_outcome: { status: 'completed' } }, '计划生成 · 待审阅', 'Plan generation · Awaiting review'],
  ];
  for (const [name, facts, zh, en] of cases) test(`${language}: ${name}`, () => {
    const html = render([{ run_id: 'run_synthetic', run_type: 'full', ...facts }], language);
    const title = html.match(/<div class="gpi-run-copy"><strong>(.*?)<\/strong>/s)[1];
    assert.equal(title, pick(zh, en));
    assert.doesNotMatch(title, /run_|完整分析|Full analysis/);
  });
  test(`${language}: run identifiers stay escaped in closed support details`, () => {
    const id = 'run_<img src=x onerror=alert(1)>';
    const html = render([{ run_id: id, run_type: 'full', run_status: 'failed' }], language);
    const details = html.match(/<details data-gpi-run-support>([\s\S]*?)<\/details>/);
    assert.ok(details, 'Support details are collapsed by default');
    assert.ok(details[1].includes(pick('技术详情', 'Technical details')));
    assert.ok(details[1].includes(EU_HTML.esc(id)));
    assert.doesNotMatch(html.replace(details[0], ''), /run_&lt;img|<img/);
  });
  test(`${language}: stale reviewer variant uses ordinary stage labels`, () => {
    const html = render([], language, { kind: 'reviewer_validation_demo', current_stage: 'question',
      stages: [{ id: 'question', status: 'blocked' }, { id: 'setup', status: 'blocked' }] });
    assert.ok(html.includes(pick('科学问题', 'Scientific question')));
    assert.ok(html.includes(pick('研究配置', 'Study setup')));
    assert.doesNotMatch(html, /审稿协议|Reviewer protocol|Data contract/);
  });
  test(`${language}: waiting copy names the preceding step`, () => {
    assert.ok(render([], language).includes(pick('等待上一步完成', 'Waiting for the previous step')));
  });
}
