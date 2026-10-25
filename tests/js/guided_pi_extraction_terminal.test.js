'use strict';

// Replays the observed immediate extraction failure and a later model
// "still running" reply through the production live/reload owners.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const js = path.resolve('src/easyicu/webserver/static/js');
const at = n => new Date(1767225600000 + n * 1000).toISOString();
const upsert = (target, row) => {
  const old = target.steps.find(item => item.id === row.id);
  if (old) Object.assign(old, row); else target.steps.push(row);
};

(async () => {
  for (const language of ['zh', 'en']) {
    const context = vm.createContext({ window: { EU_LANG: language, EU_HTML: { esc: String } },
      document: { getElementById: () => null } });
    for (const owner of ['modules', 'error-text', 'activity', 'replay', 'transcript', 'childjob', 'live-stream']) {
      vm.runInContext(fs.readFileSync(path.join(js, `screens-guided-pi-${owner}.js`), 'utf8'), context);
    }
    const modules = context.window.EasyICU.guidedPi;
    const tr = (en, zh) => language === 'zh' ? zh : en;
    const errors = modules.require('errorText').create({ tr });
    const activity = modules.require('activity').create({ tr, esc: String, iconHtml: () => '',
      resourceName: () => '', resourceKey: () => '', resourceButton: () => '', runFailureText: errors.runFailureText });
    const messages = [];
    let watched = 'failed-extract';
    let rendered = 0;
    const child = modules.require('childJob').create({ tr, activity, upsertActivityStep: upsert,
      render: () => rendered++, messages: () => messages, session: () => ({ session_id: 's' }),
      childJobId: () => watched, setChildJobId: value => { watched = value; },
      childSource: () => null, archiveChildJob: () => new Promise(() => {}),
      loadWorkflow: async () => {},
    });
    const row = child.childActivity(watched, 'easyicu_extraction_submitted');
    row.runningNote = 'still running';
    child.handleChildJobEvent(watched, 'easyicu_extraction_submitted', {
      type: 'progress', step: 'demographics_filter', status: 'running', timestamp: at(1),
    });
    const before = rendered;
    child.handleChildJobEvent(watched, 'easyicu_extraction_submitted', {
      type: 'end', status: 'failed', error: '不支持的数据库: eicu_demo', timestamp: at(2),
    });
    assert.equal(row.status, 'error');
    assert.equal(row.runningNote, '');
    assert.equal(row.failureCode, 'extraction_database_filter_unsupported');
    assert.ok(rendered > before, 'terminal rendering cannot wait for archival or session refresh');
    const html = activity.render(row);
    assert.doesNotMatch(html, /Research task is progressing|研究任务正在推进|still running/);
    assert.match(html, language === 'zh' ? /官方 Demo/ : /official Demo/);
    row.runningNote = 'outdated planning progress';
    child.hydrateProjectedJob({ present: true, job_id: 'failed-extract', kind: 'extract',
      status: 'failed', error_code: 'extraction_database_filter_unsupported', progress: [] });
    assert.equal(row.runningNote, '', 'reload clears the old running annotation');

    const turn = { status: 'running', steps: [] };
    const state = { messages: [] };
    const stream = modules.require('liveStream').create({ state, timeMs: Date.parse,
      ensureActivity: () => turn, upsertActivityStep: upsert, activity, render() {},
      assistantRow: () => { const reply = { role: 'assistant', text: '' }; state.messages.push(reply); return reply; },
      completeLatestAssistant() {}, modelErrorText: code => code, activityHasCompletedAction: () => true,
      addAssistantResources() {}, modules: { optional: () => null }, loadWorkflow: async () => {},
      watchChildJob() {}, hostJobs: { noteToolResult() {} },
    });
    stream.handlePiEvent({ type: 'run_start', at: at(0) });
    stream.handlePiEvent({ type: 'tool_end', at: at(1), job_id: 'failed-extract', code: 'easyicu_extraction_submitted' });
    stream.handlePiEvent({ type: 'text_delta', at: at(8), delta: '已提交数据提取任务，当前正在运行。' });
    assert.equal(state.messages.length, 0, 'post-handoff model prose cannot replace the host result');
    stream.handlePiEvent({ type: 'message_end', at: at(9), error_code: 'late_model_failure' });
    assert.equal(state.messages.length, 0, 'a late model error cannot replace the host result');
    stream.handlePiEvent({ type: 'run_start', at: at(10) });
    stream.handlePiEvent({ type: 'tool_end', at: at(10), is_error: true,
      job_id: 'rejected-extract', code: 'easyicu_extraction_submitted' });
    stream.handlePiEvent({ type: 'text_delta', at: at(11), delta: 'new answer' });
    assert.equal(state.messages[0].text, 'new answer', 'a new turn or failed handoff keeps ordinary model replies');

    const transcript = modules.require('transcript').create({ tr, activity, upsertActivityStep: upsert,
      timeMs: Date.parse, resourceKey: () => '', modelErrorText: () => '',
      workflowActionCode: () => '', runFailureText: errors.runFailureText });
    const session = {
      transcript: [
        { role: 'user', timestamp: at(0), content: [{ type: 'text', text: '开始提取' }] },
        { role: 'tool', timestamp: at(1), content: [{ type: 'tool_result', tool_call_id: 'extract',
          code: 'easyicu_extraction_submitted', job_id: 'failed-extract' }] },
        { role: 'assistant', timestamp: at(8), content: [{ type: 'text', text: '当前正在运行。' }] },
      ],
      archived_child_jobs: [{ present: true, job_id: 'failed-extract', kind: 'extract', status: 'failed',
        error_code: 'extraction_database_filter_unsupported', created_at_epoch: Date.parse(at(1)) / 1000,
        finished_at_epoch: Date.parse(at(2)) / 1000 }],
    };
    const history = transcript.transcriptMessages(session);
    const replies = history.filter(item => item.role === 'assistant').map(item => item.text).join('\n');
    assert.doesNotMatch(replies, /当前正在运行/);
    assert.match(replies, language === 'zh' ? /数据提取没有完成.*官方 Demo/ : /Data extraction did not complete.*official Demo/);
    for (const status of ['done', 'cancelled', 'interrupted']) {
      session.archived_child_jobs[0].status = status;
      const terminalText = transcript.transcriptMessages(session).filter(item => item.role === 'assistant').map(item => item.text).join('\n');
      assert.doesNotMatch(terminalText, /当前正在运行/);
      assert.match(terminalText, status === 'done'
        ? (language === 'zh' ? /数据提取已完成.*再批准分析/ : /Data extraction completed.*before approving analysis/)
        : status === 'cancelled'
          ? (language === 'zh' ? /数据提取已取消/ : /Data extraction was cancelled/)
          : (language === 'zh' ? /数据提取没有完成/ : /Data extraction did not complete/));
    }
  }
  console.log('extraction terminal contracts passed');
})().catch(error => { console.error(error); process.exitCode = 1; });
