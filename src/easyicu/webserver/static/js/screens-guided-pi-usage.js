/* Owner: model usage shown in the conversation's model menu.
   The host owns the accounting (Pi session journal for this conversation's
   Copilot calls; the run's provider ledger for its analysis). This file reads
   both and says what they record: completed calls with the tokens and cost the
   provider reported, and failed, in-progress or unknown calls counted apart,
   because those sit at their reservation, not at what was spent. */
(function () {
  'use strict';
  const { esc } = window.EU_HTML;

  function tr(en, zh) { return window.EU_LANG === 'zh' ? zh : en; }
  function api() { return window.EU_API || {}; }

  let context = null;
  let request = 0;

  const ERRORS = {
    conversation_usage_history_unavailable: ['This conversation predates usage records.', '这段对话创建于旧版本，没有用量记录。'],
    provider_usage_ledger_invalid: ['The analysis usage record could not be verified.', '分析的用量记录无法核对。'],
  };

  function number(value) {
    return Number(value || 0).toLocaleString(window.EU_LANG === 'zh' ? 'zh-CN' : 'en-US');
  }
  function cost(value) {
    if (value === null || value === undefined || !Number.isFinite(Number(value))) return '';
    const usd = Number(value);
    return usd > 0 && usd < 0.01 ? '<$0.01' : `$${usd.toFixed(2)}`;
  }
  function bucket(usage, name) {
    const row = usage && usage.by_state && usage.by_state[name];
    return row && typeof row === 'object' ? row : { calls: 0 };
  }

  // One line per scope; a null cost is unknown, never free.
  function summaryText(usage) {
    if (!usage) return '';
    if (usage.accounting_complete === false) {
      const known = ERRORS[usage.reason_code];
      return known ? tr(...known) : tr('Usage records are incomplete.', '用量记录不完整。');
    }
    const done = bucket(usage, 'completed');
    const parts = [tr(`${number(done.calls)} call${done.calls === 1 ? '' : 's'}`, `${number(done.calls)} 次调用`)];
    if (done.provider_reported_calls > 0) parts.push(`${number(done.provider_reported_tokens)} tokens`);
    const spent = done.calls > 0 ? cost(done.estimated_cost_usd) : '';
    if (spent) parts.push(tr(`about ${spent}`, `约 ${spent}`));
    const apart = [
      done.usage_unknown_calls > 0 ? tr(`${done.usage_unknown_calls} without reported usage`, `${done.usage_unknown_calls} 次未报告用量`) : '',
      bucket(usage, 'failed').calls > 0 ? tr(`${bucket(usage, 'failed').calls} failed`, `${bucket(usage, 'failed').calls} 次失败`) : '',
      bucket(usage, 'in_flight').calls > 0 ? tr(`${bucket(usage, 'in_flight').calls} in progress`, `${bucket(usage, 'in_flight').calls} 次进行中`) : '',
      bucket(usage, 'unknown').calls > 0 ? tr(`${bucket(usage, 'unknown').calls} with unknown state`, `${bucket(usage, 'unknown').calls} 次状态未知`) : '',
    ].filter(Boolean);
    return parts.join(' · ') + (apart.length ? tr(`; also ${apart.join(', ')}`, `；另有 ${apart.join('、')}`) : '');
  }
  function failureText(failure) {
    const known = ERRORS[failure && failure.code];
    return known ? tr(...known) : tr('Usage is unavailable right now.', '暂时读不到用量。');
  }

  function lineHtml(label, text) {
    return text ? `<p><span>${esc(label)}</span>${esc(text)}</p>` : '';
  }
  function paint(control, rows) {
    const popover = control.querySelector('[data-gpi-model-popover]');
    if (!popover) return;
    popover.querySelectorAll('[data-gpi-model-usage]').forEach(node => node.remove());
    const html = rows.map(([label, text]) => lineHtml(label, text)).join('');
    if (!html) return;
    const box = document.createElement('div');
    box.className = 'gpi-model-usage';
    box.setAttribute('data-gpi-model-usage', '');
    box.innerHTML = html;
    popover.appendChild(box);
  }

  async function read(caller) {
    try { return [await caller(), '']; } catch (failure) { return [null, failureText(failure)]; }
  }
  async function load(control) {
    if (!context || typeof api().loadPiSessionUsage !== 'function') return;
    const session = context.session();
    const projectId = context.projectId();
    if (!session || !session.session_id || !projectId || session.project_id !== projectId) return;
    const mine = ++request;
    const runId = session.binding && session.binding.run_id;
    const [conversation, conversationError] = await read(() => api().loadPiSessionUsage(session.session_id, projectId));
    const [run, runError] = runId && typeof api().loadPiRunUsage === 'function'
      ? await read(() => api().loadPiRunUsage(projectId, runId)) : [null, ''];
    if (mine !== request || !context || (context.session() || {}).session_id !== session.session_id || control.isConnected === false) return;
    paint(control, [
      [tr('This conversation', '本对话'), conversationError || summaryText(conversation && conversation.usage)],
      [tr('Latest analysis', '最近一次分析'), runError || summaryText(run && run.usage)],
    ]);
  }

  function bind(options) { context = options; ++request; }
  function onToggle(event) {
    const control = event.target.closest ? event.target.closest('.gpi-model-control') : null;
    if (control && control.open) void load(control);
  }
  if (typeof document !== 'undefined' && typeof document.addEventListener === 'function') {
    document.addEventListener('toggle', onToggle, true);
  }
  window.EasyICU.guidedPi.declare('usage', { bind, load, summaryText });
})();
