/* Owner: lightweight task routing in the conversation; never grants tools. */
(function () {
  'use strict';
  function kind(text) {
    const value = String(text || '').toLowerCase();
    const realData = /真实.{0,12}(?:数据|队列|患者|病人)|\breal(?:[ -]world)?\s+(?:\w+\s+){0,3}(?:data|cohort|patients|records)\b/;
    const execution = /分析|执行|运行|提取|\b(?:analy[sz]e?|analysis|run|execute|extract)\b/;
    const laterSteps = value.split(/然后|随后|接着|再|\b(?:then|subsequently|afterwards)\b/).slice(1);
    if (laterSteps.some(step => realData.test(step) && execution.test(step))) return '';
    const discussionOnly = /(?:只|仅)(?:想)?(?:讨论|聊聊|探讨|设计).{0,12}(?:方案|计划|方法|研究)|\b(?:only|just)\s+(?:discuss|design)\b/.test(value);
    const discussionFirst = /先(?:想)?(?:讨论|聊聊|探讨|设计).{0,12}(?:方案|计划|方法|研究)|\b(?:first|start by)\s+(?:discuss|design|plan)\b/.test(value);
    const noExecution = /(?:不要|不用|无需|不)(?:执行|运行|分析|提取)|\b(?:do not|don't|no|without)\s+(?:execute|run|perform|analy[sz]e?|analysis)\b/.test(value);
    if (discussionOnly || (discussionFirst && noExecution)) return 'discussion';
    if (/(?:不是|不要|不用|非)\s*合成数据|\b(?:not|no)\s+synthetic\s+data/.test(value)) return '';
    if (/\b(?:mimic[ -]?(?:iii|iv|3|4)|eicu|sicdb|hirid|amsterdamumcdb)\b/.test(value)) return '';
    return /(?:生成|模拟|创建|演示|\b(?:generate|create|simulate|demo)\b)/.test(value)
      && /合成.{0,12}(?:数据|记录)|模拟.{0,6}(?:数据|记录)|\b(?:synthetic|simulated)\s+(?:\w+\s+){0,3}(?:data|records)\b/.test(value) ? 'synthetic' : '';
  }
  function lastQuestion(rows) {
    return String((rows || []).slice().reverse().find(row => row && row.role === 'user')?.text || '');
  }
  function current(state) {
    const text = state.taskDraft || lastQuestion(state.messages);
    return { text, kind: kind(text) };
  }
  function render(state, { tr, esc, disabled }) {
    const task = current(state);
    if (!task.kind) return '';
    const title = task.kind === 'synthetic'
      ? tr('Create a synthetic-data demonstration', '创建合成数据演示')
      : tr('Discuss the study design first', '先讨论研究方案');
    return `<section class="gpi-data-consent" aria-label="${esc(tr('Task entry', '任务入口'))}"><div class="gpi-data-consent-body">
      <strong>${esc(title)}</strong><p>${esc(tr('This task does not need a clinical database. Open a separate project-workspace conversation with this question; file access stays under its existing controls.', '这项任务无需绑定临床数据库。可带着原问题新建产物对话；文件访问仍由原有权限控制。'))}</p>
      <button class="btn primary" type="button" data-gpi-task-workspace ${disabled ? 'disabled' : ''}>${esc(tr('Open project workspace', '打开产物工作区'))}</button>
    </div></section>`;
  }
  window.EasyICU.guidedPi.declare('taskScope', { kind, current, lastQuestion, render });
})();
