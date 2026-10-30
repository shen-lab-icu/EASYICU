'use strict';
const assert = require('node:assert/strict');
const path = require('node:path');
const fs = require('node:fs');
const { pathToFileURL } = require('node:url');
require(path.resolve('tests/js/guided_pi_module_harness.cjs'));
const js = path.resolve('src/easyicu/webserver/static/js');
const at = seconds => new Date(Date.UTC(2026, 0, 1) + seconds * 1000).toISOString();
const privateText = 'Plan it as this family; do not ask the researcher. 比例';
const upsert = (target, row) => {
  const found = target.steps.find(item => item.id === row.id);
  if (found) Object.assign(found, row); else target.steps.push(row);
};

(async () => {
  const projection = await import(pathToFileURL(path.resolve('src/easyicu/webserver/pi_copilot/node_app/src/event-projection.mjs')));
  const delta = projection.normalizePiEvent({ type: 'message_update', timestamp: Date.parse(at(1)),
    assistantMessageEvent: { type: 'thinking_delta', delta: 'Preparing exact internal entry wording' } });
  assert.ok(!JSON.stringify(delta || {}).includes('Preparing exact'));
  const tool = projection.normalizePiEvent({ type: 'tool_execution_end', toolName: 'easyicu_update_study_context',
    result: { details: { status: 'ok', code: 'study_context_updated', summary: privateText,
      details: { study_family_reading: { source: 'question_wording', family: 'descriptive_epidemiology',
        elements: [{ evidence: '发生频率' }] } } } } });
  assert.ok(!JSON.stringify(tool).includes('Plan it as'));
  const end = projection.normalizePiEvent({ type: 'message_update', timestamp: Date.parse(at(6.2)),
    assistantMessageEvent: { type: 'thinking_end' } });
  assert.equal(end.at, at(6.2));

  for (const language of ['zh', 'en']) {
    global.window = { EU_LANG: language, EU_GUIDED_PI_REPLAY: {
      childJobPresentation: job => ({ durationKnown: true, startedAt: job.created_at_epoch * 1000, endedAt: job.finished_at_epoch * 1000 }),
    } };
    for (const name of ['screens-guided-pi-activity.js', 'screens-guided-pi-childjob.js', 'screens-guided-pi-live-stream.js']) {
      eval(fs.readFileSync(path.join(js, name), 'utf8'));
    }
    const tr = (en, zh) => language === 'zh' ? zh : en;
    const activity = window.EU_GUIDED_PI_ACTIVITY.create({ tr, esc: String, iconHtml: () => '',
      resourceName: () => '', resourceKey: () => '', resourceButton: () => '', runFailureText: () => tr('Retry stopped', '重试已停止') });
    const thought = activity.render({ status: 'complete', startedAt: Date.parse(at(1)), endedAt: Date.parse(at(6.2)), steps: [
      { id: 'thinking-1', kind: 'thinking', status: 'complete', startedAt: Date.parse(at(1)), endedAt: Date.parse(at(6.2)), text: '**Preparing exact internal entry wording**' },
      { id: 'tool-1', kind: 'tool', toolName: 'easyicu_update_study_context', status: 'complete', text: privateText, receipt: tool.receipt },
      { id: 'tool-2', kind: 'tool', toolName: 'internal_opaque_tool', status: 'complete', text: privateText },
    ] });
    assert.doesNotMatch(decodeURIComponent(thought), /Preparing exact|Plan it as|do not ask|internal_opaque_tool/);
    assert.ok(thought.includes(tr('Thought', '思考')));
    assert.ok(thought.includes(tr('5.2 s', '5.2 秒')));
    assert.ok(thought.includes(tr('Saved the study configuration: a descriptive study', '保存了研究配置：按题目定为描述性研究')));
    for (const step of ['runner_repair', 'contract', 'pause', 'cache', 'human_review', 'source', 'snapshot', 'scientific_action_gap']) {
      assert.doesNotMatch(activity.pipelineEventLabel({ step, label: privateText }),
        /Plan it as|Research task is progressing|Research-task status updated|研究任务正在推进|研究任务状态已更新/);
    }

    const turn = { status: 'running', steps: [] };
    const stream = window.EasyICU.guidedPi.require('liveStream').create({
      state: { messages: [] }, timeMs: Date.parse, ensureActivity: () => turn, upsertActivityStep: upsert,
      activity, render() {}, addAssistantResources() {}, modules: { optional: () => null },
      hostJobs: { noteToolResult() {} },
    });
    stream.handlePiEvent({ type: 'thinking_start', at: at(1) });
    stream.handlePiEvent({ type: 'thinking_delta', at: at(2), delta: privateText });
    stream.handlePiEvent({ type: 'thinking_end', at: at(6.2) });
    stream.handlePiEvent({ ...tool, at: at(7), summary: privateText });
    assert.equal(turn.steps.find(row => row.kind === 'thinking').endedAt
      - turn.steps.find(row => row.kind === 'thinking').startedAt, 5200);
    assert.doesNotMatch(JSON.stringify(turn), /Plan it as|do not ask/);
    assert.ok(activity.render(turn).includes(tr('Saved the study configuration: a descriptive study', '保存了研究配置：按题目定为描述性研究')));

    const events = [{ type: 'progress', step: 'step', step_id: 'baseline_context', status: 'complete', timestamp: at(0),
      label: 'Skipped step already completed by pre-execution: baseline_context.' }];
    for (let i = 2; i <= 8; i++) {
      events.push({ type: 'progress', step: 'step', step_id: `analysis_${i}`, current_step: i, total_steps: 8,
        status: 'running', timestamp: at(i * 10), label: `Step ${i}/8 started: analysis_${i}.` });
      events.push({ type: 'progress', step: 'step', step_id: `analysis_${i}`, current_step: i, total_steps: 8,
        status: 'complete', timestamp: at(i * 10 + 3), label: `Step ${i}/8 complete: analysis_${i}.` });
    }
    events.push({ type: 'progress', step: 'provider_auto_resume', current: 1, actor: 'system', timestamp: at(84) });
    events.push({ type: 'progress', step: 'writer', status: 'complete', timestamp: at(85), label: privateText });
    const messages = [];
    const child = window.EU_GUIDED_PI_CHILDJOB.create({ tr, activity, upsertActivityStep: upsert, render() {},
      messages: () => messages, session: () => ({}), childJobId: () => 'execution', loadWorkflow: async () => {},
    });
    for (const event of events) child.handleChildJobEvent('execution', '', event);
    const live = messages.find(row => row.childJobId === 'execution');
    assert.equal(live.steps.filter(row => row.step === 'plan_step').length, 8);
    // The code that started a watched job keys its opening sentence; its stop
    // code gives the failed row a cause.
    const coded = [];
    const codedChild = window.EU_GUIDED_PI_CHILDJOB.create({ tr, activity, upsertActivityStep: upsert, render() {},
      messages: () => coded, session: () => null, childJobId: () => 'coded', loadWorkflow: async () => {},
      childSource: () => null, setChildSource() {}, setChildJobId() {}, archiveChildJob: async () => null,
      refreshSession: async () => null, sessionIsStale: () => false, rebind: async () => {},
    });
    codedChild.handleChildJobEvent('coded', 'easyicu_review_submitted', events[1]);
    const codedRow = coded.find(row => row.childJobId === 'coded');
    assert.equal(codedRow.jobCode, 'easyicu_review_submitted');
    codedChild.handleChildJobEvent('coded', 'easyicu_full_run_submitted', { type: 'end', status: 'failed',
      error: 'research_pipeline_execution_failed: ' + privateText, timestamp: at(90) });
    assert.equal(codedRow.jobCode, 'easyicu_review_submitted', 'a later guess never replaces the job’s code');
    assert.equal(codedRow.failureCode, 'research_pipeline_execution_failed');
    const job = { present: true, job_id: 'execution', kind: 'agent-run', status: 'done',
      created_at_epoch: Date.parse(at(0)) / 1000, finished_at_epoch: Date.parse(at(85)) / 1000, progress: events };
    child.hydrateProjectedJob(job);
    const rows = live.steps.filter(row => row.step === 'plan_step');
    assert.equal(rows.length, 8);
    assert.ok(rows.slice(1).every(row => row.endedAt - row.startedAt === 3000));
    const rendered = activity.render(live);
    assert.ok(rendered.includes(tr('Already completed during preparation; skipped', '已在准备阶段完成，跳过')));
    assert.ok(rendered.includes(tr('System action:', '系统操作：')));
    assert.doesNotMatch(decodeURIComponent(rendered), /Plan it as|do not ask|analysis_[2-8]/);
    child.hydrateProjectedJob({ ...job, job_id: 'planning', human_review_pending: true, progress: [
      { type: 'progress', step: 'planning', timestamp: at(1), status: 'running' },
      { type: 'progress', step: 'planning', timestamp: at(6), status: 'complete' },
    ] });
    assert.ok(messages.some(row => row.childJobId === 'planning'));
    assert.ok(messages.some(row => row.childJobId === 'execution'));
    assert.equal(messages.find(row => row.childJobId === 'planning').steps.find(row => row.step === 'planning').endedAt
      - messages.find(row => row.childJobId === 'planning').steps.find(row => row.step === 'planning').startedAt, 5000);
    // A fresh page must derive durations from receipts, independent of live state.
    const freshMessages = [];
    const fresh = window.EU_GUIDED_PI_CHILDJOB.create({ tr, activity, upsertActivityStep: upsert, render() {},
      messages: () => freshMessages, session: () => ({}), childJobId: () => '',
    });
    fresh.hydrateProjectedJob(job);
    const freshRows = freshMessages[0].steps.filter(row => row.step === 'plan_step');
    assert.equal(freshRows.length, 8);
    assert.ok(freshRows.slice(1).every(row => row.endedAt - row.startedAt === 3000));
    fresh.hydrateProjectedJob({ ...job, job_id: 'old-without-timestamps', progress: events.map(event => ({ ...event, timestamp: undefined })) });
    const legacy = freshMessages.find(row => row.childJobId === 'old-without-timestamps');
    assert.ok(legacy.steps.filter(row => row.step === 'plan_step').every(row => row.durationKnown === false));
    const recovery = { steps: [] };
    activity.notePipelineEvent(recovery, { type: 'progress', step: 'provider_recovery_wait', timestamp: at(1) });
    activity.notePipelineEvent(recovery, { type: 'progress', step: 'provider_auto_resume', timestamp: at(8) });
    assert.equal(recovery.steps[0].endedAt - recovery.steps[0].startedAt, 7000);
    const transport = { status: 'running', runningTitle: 'Preparing', steps: [] };
    const retryId = 'a'.repeat(32);
    activity.notePipelineEvent(transport, { type: 'progress', step: 'data_foundation', timestamp: at(1) });
    for (const [phase, current, time] of [['failed', 1, 2], ['waiting', 1, 3], ['started', 2, 4]]) {
      activity.notePipelineEvent(transport, { type: 'progress', step: 'provider_transport_retry',
        retry_id: retryId, retry_phase: phase, current, status: 'running', timestamp: at(time) });
    }
    assert.equal(transport.steps.filter(row => row.step === 'provider_transport_retry').length, 1);
    assert.equal(transport.steps.find(row => row.step === 'data_foundation').status, 'running');
    assert.match(activity.render(transport), language === 'zh' ? /正在发起第 2 次模型请求/ : /Retrying model request 2/);
    activity.notePipelineEvent(transport, { type: 'progress', step: 'provider_transport_retry',
      retry_id: retryId, retry_phase: 'continued', current: 2, status: 'complete', timestamp: at(5) });
    assert.match(activity.render(transport), language === 'zh' ? /模型重试已返回/ : /continued after a retry/);
    const endOnly = { steps: [] };
    activity.notePipelineEvent(endOnly, { type: 'progress', step: 'step', step_id: 'baseline_context',
      current_step: 1, total_steps: 1, status: 'complete', timestamp: at(1) });
    assert.equal(endOnly.steps[0].durationKnown, false, 'A terminal event alone cannot measure a step duration');
  }
  process.stdout.write('trace contracts passed\n');
})().catch(error => { console.error(error); process.exitCode = 1; });
