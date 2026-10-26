/* Owner: what leaves the machine for a run: what it sent to the model, its
   problem report and its shareable replay.
   The host's request records (webserver/model_request_records.py) own what
   is recorded for each model request: its purpose, provider, kind of content,
   size and state, never its text. The host's problem bundle
   (webserver/problem_bundle.py) owns the diagnostic zip, and its replay
   exporter (webserver/study_replay.py) owns what a replay may contain and
   refuses anything else with a code. This file lists the records in plain
   words and starts the local downloads; it sends nothing and reads nothing
   else. The run-files and run-stop owners place its controls. */
(function () {
  'use strict';
  const { esc } = window.EU_HTML;

  function tr(en, zh) { return window.EU_LANG === 'zh' ? zh : en; }
  function api() { return window.EU_API || {}; }
  function icon(name, size) { return typeof window.icon === 'function' ? window.icon(name, size, 1.6) : ''; }

  const PURPOSES = {
    planner: ['Plan the study', '制定计划'],
    replanner: ['Revise the plan', '修订计划'],
    coder: ['Write analysis code', '编写分析代码'],
    repair: ['Repair analysis code', '修复分析代码'],
    analyzer: ['Read the results', '解读结果'],
    writer: ['Write the report', '撰写报告'],
    reviewer: ['Review', '审阅'],
    auditor: ['Check the run', '核查运行'],
    concept_auditor: ['Check variable definitions', '核查变量定义'],
    vlm: ['Read a figure', '读图'],
    visual_qa: ['Check a figure', '检查图表'],
    figure_review: ['Review figures', '审阅图表'],
    conversation: ['Conversation', '对话'],
    literature: ['Search literature', '检索文献'],
    idea_mining: ['Find study ideas', '挖掘选题'],
    provider_recovery_probe: ['Check the connection', '检测连接'],
    legacy_model_roster_migration: ['Update model settings', '更新模型设置'],
  };
  const STAGES = { execution_retry: ['retry', '重试'], report_revision: ['report revision', '报告修订'] };
  const PROVIDERS = {
    openai: 'OpenAI', openrouter: 'OpenRouter', anthropic: 'Anthropic', claude: 'Claude', gemini: 'Gemini',
    deepseek: 'DeepSeek', qwen: 'Qwen', 'openai-codex': 'Codex', codex: 'Codex',
    ollama: ['Ollama (this computer)', 'Ollama（本机）'], local: ['This computer', '本机'],
    custom: ['Custom service', '自定义服务'], mock: ['Simulated', '模拟'], replay: ['Recorded replay', '录制回放'],
  };
  const CLASSES = { aggregate: ['summary statistics', '汇总统计'], schema: ['data structure', '数据结构'], text: ['text', '文字'], code: ['code', '代码'] };
  const STATES = {
    completed: ['Done', '已完成'], completed_usage_unreported: ['Done', '已完成'], completed_usage_overflow: ['Done', '已完成'],
    failed: ['Failed', '失败'], failed_usage_unknown: ['Failed', '失败'],
    in_progress: ['Started', '已发起'], pending: ['Started', '已发起'], running: ['Started', '已发起'],
    cancelled: ['Cancelled', '已取消'], budget_exhausted: ['Over budget', '超出预算'],
  };
  const FAILURES = {
    run_ref_unknown: ['The run list is out of date. Refresh the status and try again.', '运行列表已过期，请刷新状态后再试。'],
    conversation_unavailable: ['This conversation’s records could not be read.', '读不到这段对话的发送记录。'],
  };

  function label(pair) { return Array.isArray(pair) ? tr(...pair) : String(pair || ''); }
  function size(bytes) {
    if (!Number.isInteger(bytes) || bytes < 0) return '—';
    if (bytes < 1024) return `${bytes} B`;
    const kb = bytes / 1024;
    return kb < 1024 ? `${kb >= 10 ? Math.round(kb) : Math.round(kb * 10) / 10} KB` : `${Math.round(kb / 102.4) / 10} MB`;
  }
  function purpose(row) {
    const known = PURPOSES[row.purpose];
    const text = known ? tr(...known) : tr('Not recorded', '未记录');
    const stage = STAGES[row.stage];
    return stage ? `${text} · ${tr(...stage)}` : text;
  }
  function contents(row) {
    const classes = Array.isArray(row.data_classes) ? row.data_classes.filter(name => CLASSES[name]) : [];
    return classes.length ? classes.map(name => tr(...CLASSES[name])).join(tr(', ', '、')) : tr('Not recorded', '未记录');
  }

  // The host's view, as a table. A missing record is never shown as "nothing sent".
  function recordsHtml(view) {
    if (!view || view.status === 'unavailable') {
      return `<p class="gpi-trust-note" role="status">${esc(tr('The records could not be verified.', '发送记录无法核对。'))}</p>`;
    }
    if (view.status === 'not_recorded') {
      return `<p class="gpi-trust-note" role="status">${esc(tr('No records. Earlier versions did not keep them, so this does not mean nothing was sent.', '没有发送记录。较早的版本不记录这些信息，所以这不代表没有发送过。'))}</p>`;
    }
    const requests = Array.isArray(view.requests) ? view.requests : [];
    if (!requests.length) return `<p class="gpi-trust-note" role="status">${esc(tr('No model requests yet.', '还没有向模型发送过请求。'))}</p>`;
    const rows = requests.map(row => {
      const state = STATES[row.state] || ['Unknown', '未知'];
      return `<tr${row.failed ? ' class="is-failed"' : ''}><td>${esc(String(row.sequence || ''))}</td><td>${esc(purpose(row))}</td><td>${esc(label(PROVIDERS[row.provider]) || tr('Not recorded', '未记录'))}</td><td>${esc(contents(row))}</td><td class="gpi-trust-size">${esc(size(row.byte_size))}</td><td>${esc(tr(...state))}</td></tr>`;
    }).join('');
    return `<p class="gpi-trust-note">${esc(tr('Purpose, provider, kind and size of each request; the text sent is not shown here.', '每次请求的用途、服务商、内容类型和大小；这里不显示发出的原文。'))}</p>
      <table class="gpi-trust-table"><thead><tr><th scope="col">#</th><th scope="col">${esc(tr('Purpose', '用途'))}</th><th scope="col">${esc(tr('Provider', '服务商'))}</th><th scope="col">${esc(tr('Contents', '内容'))}</th><th scope="col">${esc(tr('Size', '大小'))}</th><th scope="col">${esc(tr('State', '状态'))}</th></tr></thead><tbody>${rows}</tbody></table>`;
  }
  function failureHtml(failure) {
    const known = FAILURES[failure && failure.code];
    return `<p class="gpi-trust-note" role="alert">${esc(known ? tr(...known) : tr('The records are unavailable right now.', '暂时读不到发送记录。'))}</p>`;
  }

  // One cached view per run_ref; the run-files owner re-renders from it.
  const runs = new Map();
  function sectionHtml(runRef, open) {
    if (!runRef) return '';
    const cached = runs.get(runRef);
    const body = cached ? cached.html : open ? `<p class="gpi-trust-note" role="status">${esc(tr('Reading…', '正在读取…'))}</p>` : '';
    return `<details class="gpi-run-trust" data-gpi-trust-run="${esc(runRef)}"${open ? ' open' : ''}><summary>${esc(tr('What was sent to the model', '发送给模型的记录'))}</summary><div data-gpi-trust-body>${body}</div></details>`;
  }
  function paintRun(runRef) {
    const cached = runs.get(runRef);
    if (!cached || typeof document === 'undefined') return;
    document.querySelectorAll('[data-gpi-trust-run]').forEach(node => {
      if (node.dataset.gpiTrustRun !== runRef) return;
      const body = node.querySelector('[data-gpi-trust-body]');
      if (body) body.innerHTML = cached.html;
    });
  }
  async function loadRun(runRef) {
    if (!runRef || runs.has(runRef) || typeof api().loadRunEgress !== 'function') return;
    let html;
    try { html = recordsHtml(await api().loadRunEgress(runRef)); } catch (failure) {
      paintFailure(runRef, failureHtml(failure));
      return;
    }
    runs.set(runRef, { html });
    paintRun(runRef);
  }
  // A failed read is shown but not cached, so reopening tries again.
  function paintFailure(runRef, html) {
    runs.set(runRef, { html });
    paintRun(runRef);
    runs.delete(runRef);
  }

  // className follows the placing owner's buttons.
  function problemButtonHtml(runRef, className = 'btn sm') {
    return runRef ? `<button type="button"${className ? ` class="${esc(className)}"` : ''} data-gpi-problem-bundle="${esc(runRef)}">${esc(tr('Download problem report', '下载问题报告'))}</button>` : '';
  }
  async function downloadProblem(button) {
    if (button.disabled || typeof api().downloadProblemBundle !== 'function') return;
    const text = button.textContent;
    button.disabled = true;
    try {
      await api().downloadProblemBundle(button.dataset.gpiProblemBundle);
      button.textContent = text;
    } catch (failure) {
      button.textContent = tr('Download failed, try again', '下载失败，点此重试');
    } finally {
      button.disabled = false;
    }
  }

  // The exporter's refusals, in the researcher's words.
  const REPLAY_REFUSALS = [
    [/^replay_small_cell_forbidden$/, ['A group has fewer than 20 patients, so the data use agreement does not allow sharing this replay.', '有分组少于 20 人，按数据使用协议不能分享这份回放。']],
    [/^replay_patient_level_forbidden$/, ['The results contain patient-level data, so no replay is made.', '结果里含有患者级数据，不能生成回放。']],
    [/^replay_aggregate_unproven$|^replay_evidence_class_forbidden$/, ['A result file could not be confirmed as aggregate, so no replay is made.', '有结果文件无法确认是汇总数据，不能生成回放。']],
    [/^replay_figure_/, ['A figure did not pass the sharing check, so no replay is made.', '有图没有通过外传检查，不能生成回放。']],
    [/^replay_path_forbidden$|^replay_credential_in_identity$/, ['The records name a local path or a credential, so no replay is made.', '记录里有本机路径或凭据，不能生成回放。']],
    [/^replay_source_unsupported$/, ['Replays are not available for this data source yet.', '这个数据来源暂不支持生成回放。']],
    [/^replay_size_limit$/, ['The replay would exceed 32 MB.', '回放会超过 32 MB 上限。']],
    [/^run_ref_unknown$/, ['The run list is out of date. Refresh the status and try again.', '运行列表已过期，请刷新状态后再试。']],
    [/^replay_/, ['The run’s records are incomplete or changed, so no replay is made.', '运行记录不完整或已改动，不能生成回放。']],
  ];
  function replayRefusal(code) {
    const known = REPLAY_REFUSALS.find(([pattern]) => pattern.test(String(code || '')));
    return known ? tr(...known[1]) : tr('The replay could not be made.', '无法生成回放。');
  }
  function replayButtonHtml(runRef) {
    return runRef ? `<button type="button" data-gpi-study-replay="${esc(runRef)}">${esc(tr('Download replay', '下载研究回放'))}</button>` : '';
  }
  async function downloadReplay(button) {
    if (button.disabled || typeof api().downloadStudyReplay !== 'function') return;
    const actions = button.parentElement;
    const old = actions && actions.parentElement ? actions.parentElement.querySelector('[data-gpi-study-replay-refusal]') : null;
    if (old) old.remove();
    button.disabled = true;
    try {
      await api().downloadStudyReplay(button.dataset.gpiStudyReplay);
    } catch (failure) {
      if (actions) actions.insertAdjacentHTML('afterend', `<p class="gpi-run-files-error" role="alert" data-gpi-study-replay-refusal>${esc(replayRefusal(failure && failure.code))}</p>`);
    } finally {
      button.disabled = false;
    }
  }

  // A conversation's records open in a dialog from the model menu.
  let dialog = null;
  async function openConversation(sessionId) {
    if (!sessionId || typeof document === 'undefined' || typeof api().loadConversationEgress !== 'function') return;
    if (!dialog) {
      dialog = document.createElement('dialog');
      dialog.className = 'gpi-trust-dialog';
      dialog.addEventListener('click', event => {
        if (event.target === dialog || event.target.closest('[data-gpi-trust-close]')) dialog.close();
      });
      document.body.appendChild(dialog);
    }
    const head = `<header><h2>${esc(tr('What this conversation sent to the model', '本对话发送给模型的记录'))}</h2><button type="button" data-gpi-trust-close aria-label="${esc(tr('Close', '关闭'))}">${icon('x', 15)}</button></header>`;
    dialog.innerHTML = `${head}<div data-gpi-trust-body><p class="gpi-trust-note" role="status">${esc(tr('Reading…', '正在读取…'))}</p></div>`;
    if (!dialog.open) dialog.showModal();
    let html;
    try { html = recordsHtml(await api().loadConversationEgress(sessionId)); } catch (failure) { html = failureHtml(failure); }
    const body = dialog.querySelector('[data-gpi-trust-body]');
    if (body) body.innerHTML = html;
  }

  if (typeof document !== 'undefined' && typeof document.addEventListener === 'function') {
    document.addEventListener('toggle', event => {
      const node = event.target;
      if (node && node.matches && node.matches('[data-gpi-trust-run]') && node.open) void loadRun(node.dataset.gpiTrustRun);
    }, true);
    document.addEventListener('click', event => {
      const closest = selector => (event.target && event.target.closest ? event.target.closest(selector) : null);
      const button = closest('[data-gpi-problem-bundle]');
      if (button) { event.preventDefault(); void downloadProblem(button); return; }
      const replay = closest('[data-gpi-study-replay]');
      if (replay) { event.preventDefault(); void downloadReplay(replay); }
    });
  }

  window.EasyICU.guidedPi.declare('runTrust', { sectionHtml, problemButtonHtml, replayButtonHtml, replayRefusal, recordsHtml, openConversation, loadRun });
})();
