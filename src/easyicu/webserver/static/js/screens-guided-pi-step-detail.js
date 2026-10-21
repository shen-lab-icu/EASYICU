/* Owner: one executed plan step's Input/Output in the workbench.
   The step is read through the read-only step route by its run and step
   ids -- server-issued coordinates, never a client path. Input is the code
   that ran (displayed, never executed) and how it ran; Output is the files
   the step produced. Runner output stays on the host: no owner redacts it
   for disclosure, so the panel never receives log text. */
(function () {
  'use strict';

  const { esc } = window.EU_HTML;
  function icon(name, size) { return typeof window.icon === 'function' ? window.icon(name, size) : ''; }
  function tr(en, zh) { return window.EU_LANG === 'zh' ? zh : en; }

  const RUN_ID = /^[A-Za-z][A-Za-z0-9_.-]{0,159}$/;
  const STEP_ID = /^[A-Za-z0-9_][A-Za-z0-9_.-]{0,159}$/;
  const EVIDENCE_ID = /^[A-Za-z0-9_.-]{1,160}$/;
  const SHA256 = /^[a-f0-9]{64}$/;
  const PNG_DATA = /^data:image\/png;base64,[A-Za-z0-9+/]+=*$/;
  const FILE_NAME = /^[A-Za-z0-9_][A-Za-z0-9_.-]{0,159}$/;
  const STATES = new Set(['running', 'complete', 'failed', 'skipped', 'precompleted', 'not_started', 'unknown']);
  const MODES = new Set(['standard', 'generated', 'fallback', 'host', 'unknown']);
  const OUTPUT_KINDS = new Set(['figure', 'table', 'summary', 'record', 'document']);
  const LANGUAGES = new Set(['python', 'r', 'sql', 'shell', 'text']);
  // The server caps code at 64 KiB; anything longer was not produced by it.
  const MAX_CODE_CHARS = 70 * 1024;

  function coordinates(runId, stepId) {
    const run = String(runId || '');
    const step = String(stepId || '');
    return RUN_ID.test(run) && STEP_ID.test(step) ? { runId: run, stepId: step } : null;
  }
  function count(value) {
    const number = Number(value);
    return Number.isInteger(number) && number >= 0 ? number : 0;
  }
  function bounded(value, upper) {
    const number = Number(value);
    return value != null && Number.isInteger(number) && number >= 0 && number <= upper ? number : null;
  }
  function instant(value) {
    const at = Date.parse(String(value || ''));
    return Number.isFinite(at) ? at : null;
  }

  /* Accept only the step route's fixed shape; anything else is dropped. */
  function validDetail(raw) {
    if (!raw || typeof raw !== 'object' || !STEP_ID.test(String(raw.step_id || ''))) return null;
    const execution = raw.execution && typeof raw.execution === 'object' ? raw.execution : {};
    const code = raw.code && typeof raw.code === 'object' && typeof raw.code.text === 'string'
      && raw.code.text.length <= MAX_CODE_CHARS ? raw.code : null;
    const outputs = (Array.isArray(raw.outputs) ? raw.outputs : []).slice(0, 24).map(row => {
      if (!row || !FILE_NAME.test(String(row.name || '')) || !OUTPUT_KINDS.has(row.kind)) return null;
      const evidence = row.evidence && EVIDENCE_ID.test(String(row.evidence.id || ''))
        && SHA256.test(String(row.evidence.sha256 || '')) ? { id: String(row.evidence.id), sha256: String(row.evidence.sha256) } : null;
      return {
        name: String(row.name), kind: row.kind, bytes: count(row.bytes), evidence,
        thumbnail: PNG_DATA.test(String(row.thumbnail || '')) ? String(row.thumbnail) : '',
      };
    }).filter(Boolean);
    const headline = raw.headline && typeof raw.headline === 'object' ? raw.headline : {};
    const inputs = raw.inputs && typeof raw.inputs === 'object' ? raw.inputs : {};
    const timing = raw.timing && typeof raw.timing === 'object' ? raw.timing : {};
    return {
      stepId: String(raw.step_id),
      state: STATES.has(raw.state) ? raw.state : 'unknown',
      execution: {
        mode: MODES.has(execution.mode) ? execution.mode : 'unknown',
        executor: /^[a-z][a-z0-9_]{0,79}$/.test(String(execution.executor || '')) ? String(execution.executor) : '',
      },
      code: code ? {
        text: code.text,
        language: LANGUAGES.has(code.language) ? code.language : 'text',
        displayName: FILE_NAME.test(String(code.display_name || '')) ? String(code.display_name) : 'analysis.py',
        lineCount: count(code.line_count), truncated: code.truncated === true,
      } : null,
      outputs,
      withheld: count(raw.withheld_output_count),
      more: count(raw.more_output_count),
      headline: { en: String(headline.en || '').slice(0, 240), zh: String(headline.zh || '').slice(0, 240) },
      // What the step read, from receipts only (run_step_trace._step_inputs).
      inputs: {
        sourceLabel: String(inputs.source_label || '').slice(0, 160),
        sourceDatabase: /^[A-Za-z0-9_.-]{1,64}$/.test(String(inputs.source_database || '')) ? String(inputs.source_database) : '',
        stayCount: bounded(inputs.stay_count, 10000000),
        variableCount: bounded(inputs.variable_count, 100000),
        stepIndex: bounded(inputs.step_index, 10000),
        stepTotal: bounded(inputs.step_total, 10000),
      },
      timing: { startedAt: instant(timing.started_at), endedAt: instant(timing.ended_at) },
    };
  }

  async function load(projectId, runId, stepId) {
    const api = window.EU_API || {};
    if (typeof api.loadPiCopilotRunStep !== 'function') {
      throw new Error(tr('The step detail API is unavailable.', '步骤详情接口不可用。'));
    }
    const response = await api.loadPiCopilotRunStep(projectId, runId, stepId);
    const detail = validDetail(response && response.step);
    if (!detail || detail.stepId !== stepId) throw new Error(tr('This step cannot be displayed safely.', '这一步无法安全展示。'));
    return detail;
  }

  function executionBadge(mode, name) {
    if (mode === 'standard') {
      return `<span class="gpi-step-badge is-standard">${icon('layers', 12)}${esc(tr('Standard method', '标准方法'))}${name ? `: ${esc(name)}` : ''}</span>`;
    }
    if (mode === 'generated') return `<span class="gpi-step-badge is-generated">${icon('edit', 12)}${esc(tr('Code written for this run', '本次编写的代码'))}</span>`;
    if (mode === 'fallback') return `<span class="gpi-step-badge is-standard">${icon('shield', 12)}${esc(tr('Fallback script', '备用脚本'))}</span>`;
    if (mode === 'host') return `<span class="gpi-step-badge">${icon('db', 12)}${esc(tr('Prepared by EasyICU', '由 EasyICU 准备'))}</span>`;
    return '';
  }
  function bytesText(value) {
    const bytes = count(value);
    return bytes >= 1024 * 1024 ? `${(bytes / 1024 / 1024).toFixed(1)} MB`
      : bytes >= 1024 ? `${(bytes / 1024).toFixed(1)} KB` : `${bytes} B`;
  }
  function note(text, kind) {
    return `<p class="gpi-step-note${kind ? ` is-${kind}` : ''}">${esc(text)}</p>`;
  }
  // The step's own time from its audit envelopes, when the row had none.
  function elapsed(timing) {
    if (!timing || timing.startedAt == null || timing.endedAt == null || timing.endedAt < timing.startedAt) return '';
    const seconds = (timing.endedAt - timing.startedAt) / 1000;
    if (seconds < 60) return tr(`${seconds.toFixed(1)} s`, `${seconds.toFixed(1)} 秒`);
    const minutes = Math.floor(Math.round(seconds) / 60);
    const rest = Math.round(seconds) % 60;
    return rest ? tr(`${minutes} min ${rest} s`, `${minutes} 分 ${rest} 秒`) : tr(`${minutes} min`, `${minutes} 分`);
  }
  /* Receipt facts above the code: the data source, the research cohort's
     ICU stays, the variables the plan step names and its place in the plan.
     A fact no receipt records is left out. The variable count is the
     plan step's named inputs, not the columns the code read. */
  function inputFacts(detail, operation) {
    const facts = [];
    const add = (label, value) => { if (value != null && value !== '') facts.push([label, String(value)]); };
    const inputs = detail ? detail.inputs : {};
    add(tr('Data source', '数据源'), [inputs.sourceLabel, inputs.sourceDatabase ? `(${inputs.sourceDatabase})` : ''].filter(Boolean).join(' '));
    add(tr('ICU stays in the research cohort', '研究队列 ICU 入住数'), inputs.stayCount);
    add(tr('Input variables listed in the plan', '计划列出的输入变量数'), inputs.variableCount);
    const index = (inputs && inputs.stepIndex) || operation.stepIndex;
    const total = (inputs && inputs.stepTotal) || operation.stepTotal;
    add(tr('Plan step', '计划步骤'), index && total ? tr(`Step ${index} of ${total}`, `第 ${index}/${total} 步`) : '');
    return facts.length
      ? `<dl class="gpi-operation-facts gpi-step-facts">${facts.map(([label, value]) => `<div><dt>${esc(label)}</dt><dd>${esc(value)}</dd></div>`).join('')}</dl>` : '';
  }

  function inputSection(tab, codeRows) {
    const detail = tab.stepDetail;
    const operation = tab.operation;
    const mode = detail ? detail.execution.mode : operation.execution;
    const name = operation.executorName || (detail && detail.execution.executor) || '';
    const head = `<div class="gpi-source-block-head"><strong>${esc(tr('Input', '输入'))}</strong>${executionBadge(mode, name)}</div>`;
    if (tab.stepLoading && !detail) return `<section class="gpi-source-block">${head}${note(tr('Reading the step…', '正在读取这一步…'))}</section>`;
    if (tab.stepError && !detail) return `<section class="gpi-source-block">${head}${note(tab.stepError, 'error')}</section>`;
    const code = detail && detail.code;
    const facts = inputFacts(detail, operation);
    if (!code) {
      const text = detail && detail.state === 'precompleted'
        ? tr('EasyICU completed this step while preparing the data; it has no analysis code.', '这一步由 EasyICU 在准备数据时完成，没有分析代码。')
        : detail && detail.state === 'running'
          ? tr('The code appears here once it is written.', '代码写好后会显示在这里。')
          : tr('This step has no code to show.', '这一步没有可显示的代码。');
      return `<section class="gpi-source-block">${head}${facts}${note(text)}</section>`;
    }
    const language = code.language.toUpperCase();
    const truncated = code.truncated
      ? note(tr(`Only the first ${code.lineCount} lines are shown.`, `仅显示前 ${code.lineCount} 行`)) : '';
    return `<section class="gpi-source-block">${head}${facts}
      <div class="gpi-source-cell-meta"><span>${esc(code.displayName)}</span><div><b>${esc(language)}</b><button type="button" data-gpi-step-copy>${icon('copy', 13)} ${esc(tr('Copy code', '复制代码'))}</button></div></div>
      <pre class="gpi-source-code" tabindex="0">${codeRows(code.text, code.language)}</pre>${truncated}
    </section>`;
  }

  function outputFile(row, index) {
    const label = `${row.name} · ${bytesText(row.bytes)}`;
    if (row.thumbnail) {
      return `<figure class="gpi-step-figure"><img src="${esc(row.thumbnail)}" alt="${esc(row.name)}" loading="lazy"><figcaption>${esc(label)}</figcaption></figure>`;
    }
    const iconName = row.kind === 'figure' ? 'viz' : row.kind === 'table' ? 'grid' : 'file';
    if (row.evidence) {
      return `<button type="button" class="gpi-step-file" data-gpi-step-evidence="${index}">${icon(iconName, 14)}<span>${esc(label)}</span>${icon('arrow', 12)}</button>`;
    }
    return `<span class="gpi-step-file">${icon(iconName, 14)}<span>${esc(label)}</span></span>`;
  }
  function outputSection(tab) {
    const detail = tab.stepDetail;
    const head = `<div class="gpi-source-block-head"><strong>${esc(tr('Output', '输出'))}</strong></div>`;
    if (!detail) return `<section class="gpi-source-block is-output">${head}</section>`;
    if (detail.state === 'running' || detail.state === 'not_started') {
      return `<section class="gpi-source-block is-output">${head}${note(tr('The output appears when this step finishes.', '这一步完成后显示输出。'))}</section>`;
    }
    const parts = [];
    // A failed step shows its error in a red box, with the stop's cause
    // from the error-text owner when the trace row carried one.
    if (detail.state === 'failed') {
      parts.push(`<div class="gpi-step-error" role="alert">${note(tr('This step stopped with an error.', '这一步出错停止。'), 'error')}${tab.operation.cause ? note(tab.operation.cause, 'error') : ''}</div>`);
    }
    if (detail.state === 'skipped') parts.push(note(tr('Skipped because a step it depends on failed.', '所依赖的步骤失败，这一步已跳过。')));
    const headline = tr(detail.headline.en || '', detail.headline.zh || '');
    if (headline) parts.push(`<p class="gpi-operation-detail">${esc(headline)}</p>`);
    const files = detail.outputs.map(outputFile).join('');
    if (files) parts.push(`<div class="gpi-step-files">${files}</div>`);
    const hidden = detail.withheld + detail.more;
    if (!files && !hidden && detail.state === 'complete') {
      parts.push(note(tr('This step updated the run record and produced no file.', '这一步没有生成文件，只更新了运行记录。')));
    }
    if (hidden) {
      parts.push(note(tr(`${hidden} more file(s) are not listed: patient-level data, unsupported formats or over the list limit.`,
        `另有 ${hidden} 个文件未列出：患者级数据、不支持的格式或超出列表上限。`)));
    }
    parts.push(note(tr('The run log stays in the local run record and is not shown here.', '运行日志保存在本机运行记录中，不在这里显示。')));
    return `<section class="gpi-source-block is-output">${head}${parts.join('')}</section>`;
  }

  function render(tab, helpers) {
    const operation = tab.operation;
    // The header's status chip follows the step's own record once read.
    const status = tab.stepDetail ? ({ complete: 'complete', precompleted: 'complete',
      failed: 'error', skipped: 'skipped', running: 'running' }[tab.stepDetail.state] || operation.status) : operation.status;
    const statusText = helpers.statusLabel(status);
    const duration = operation.duration || (tab.stepDetail ? elapsed(tab.stepDetail.timing) : '');
    const position = operation.stepIndex && operation.stepTotal
      ? `<span>${esc(tr(`Step ${operation.stepIndex}/${operation.stepTotal}`, `第 ${operation.stepIndex}/${operation.stepTotal} 步`))}</span>` : '';
    return `<div class="gpi-source-document gpi-operation-document gpi-step-document">
      <header class="gpi-source-title gpi-operation-title"><span class="gpi-operation-title-icon" aria-hidden="true">${operation.icon ? icon(operation.icon, 18) : ''}</span><div><h2>${esc(operation.title)}</h2><p>${position}<b class="gpi-status-chip status-${esc(status)}">${esc(statusText)}</b>${duration ? `<span>${icon('clock', 12)}${esc(duration)}</span>` : ''}</p></div></header>
      ${inputSection(tab, helpers.codeRows)}
      ${outputSection(tab)}
    </div>`;
  }

  function evidenceAt(tab, index) {
    const row = tab.stepDetail && tab.stepDetail.outputs[Number(index)];
    return row && row.evidence ? { id: row.evidence.id, sha256: row.evidence.sha256, title: row.name } : null;
  }

  window.EasyICU.guidedPi.declare('stepDetail', { coordinates, evidenceAt, load, render, validDetail });
})();
