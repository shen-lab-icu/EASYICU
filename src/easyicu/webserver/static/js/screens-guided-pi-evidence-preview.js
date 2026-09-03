/* Owner: read-only renderer dispatch for digest-pinned Research Agent evidence. */
(function () {
  'use strict';
  const { esc } = window.EU_HTML;

  function tr(en, zh) { return window.EU_LANG === 'zh' ? zh : en; }
  function text(value, limit) { return String(value == null ? '' : value).slice(0, limit || 500); }
  function title(payload) {
    const p = payload && typeof payload === 'object' ? payload : {};
    return text(p.display_name || p.evidence_id || tr('Evidence', '证据'), 160);
  }
  function kindLabel(payload) {
    const renderer = String(payload && payload.renderer || 'metadata');
    const labels = {
      code: tr('Code', '代码'), json: 'JSON', table: tr('Table', '表格'),
      metadata: tr('File', '文件'),
    };
    return labels[renderer] || tr('File', '文件');
  }
  function locatorView(locator) {
    const source = locator && typeof locator === 'object' ? locator : {};
    if (!source.pointer && (source.value === '' || source.value == null)) return '';
    return `<div class="gpi-evidence-locator"><div><span>${esc(tr('JSON pointer', 'JSON 指针'))}</span><code>${esc(text(source.pointer, 500) || '—')}</code></div><div><span>${esc(tr('Bound source value', '绑定源数值'))}</span><code>${esc(text(source.value, 500) || '—')}</code></div></div>`;
  }
  function relationLabel(relation) {
    const labels = {
      analysis_code: tr('Code that generated this record', '生成这条记录的代码'),
      input_data: tr('Declared input evidence', '登记的输入证据'),
      run_plan_authority: tr('Approved analysis plan', '运行所依据的分析计划'),
      run_research_context: tr('Research context', '研究问题与配置'),
      run_cohort_authority: tr('Cohort authority', '队列定义权威'),
    };
    return labels[String(relation || '')] || tr('Registered parent evidence', '已登记父证据');
  }
  function evidenceLink(row) {
    const item = row && typeof row === 'object' ? row : {};
    const evidenceId = text(item.evidence_id, 160);
    const sha256 = text(item.sha256, 64).toLowerCase();
    const status = String(item.status || 'unregistered');
    const label = relationLabel(item.relation);
    if (status !== 'registered' || !/^[A-Za-z0-9_.-]{1,160}$/.test(evidenceId) || !/^[a-f0-9]{64}$/.test(sha256)) {
      return `<div class="gpi-evidence-lineage-gap"><span>${esc(label)}</span><strong>${esc(evidenceId || tr('Not recorded', '未记录'))}</strong><small>${esc(status)}</small></div>`;
    }
    const fileName = text(item.display_name || evidenceId, 180);
    return `<button type="button" class="gpi-evidence-lineage-link" data-gpi-evidence-open data-evidence-id="${esc(evidenceId)}" data-evidence-sha256="${esc(sha256)}" data-evidence-kind="${esc(text(item.kind || 'artifact', 80))}" data-evidence-label="${esc(fileName)}"><span>${esc(label)}</span><strong>${esc(fileName)}</strong></button>`;
  }
  function recordView(payload) {
    const p = payload && typeof payload === 'object' ? payload : {};
    return `<section class="gpi-evidence-record"><div class="gpi-evidence-record-head"><span>${esc(tr('Registered source record', '真实登记文件'))}</span><strong>${esc(text(p.display_name || p.evidence_id, 180))}</strong></div><p class="gpi-evidence-record-status">${esc(tr('Registry digest verified', '登记摘要已核对'))}</p></section>`;
  }
  function reproductionPathView(payload, rows) {
    const p = payload && typeof payload === 'object' ? payload : {};
    const stages = [p.kind === 'code' ? tr('Code', '代码') : tr('Result', '结果')];
    if (rows.some(row => row && row.relation === 'analysis_code' && row.status === 'registered')) stages.push(tr('Code', '代码'));
    if (rows.some(row => row && row.relation === 'input_data' && row.status === 'registered')) stages.push(tr('Input', '输入'));
    if (p.run_authority && p.run_authority.status === 'recorded') stages.push(tr('Run', '运行'));
    return `<div class="gpi-evidence-path" aria-label="${esc(tr('Reproduction path', '复现路径'))}">${stages.map((stage, index) => `${index ? '<b aria-hidden="true">→</b>' : ''}<span>${esc(stage)}</span>`).join('')}</div>`;
  }
  function fileAuditView(payload, locator) {
    const p = payload && typeof payload === 'object' ? payload : {};
    const facts = [
      [tr('Run-relative file', '运行内真实文件'), p.relative_path],
      [tr('Analysis step', '生成步骤'), p.role],
      [tr('Producer', '生成 owner'), p.producer],
      [tr('Generation mode', '生成方式'), p.generation_mode],
    ].filter(([, value]) => value !== '' && value != null);
    const rows = Array.isArray(p.declared_lineage) ? p.declared_lineage : [];
    const hasCode = rows.some(row => row && row.relation === 'analysis_code' && row.status === 'registered');
    const rawJson = p.renderer === 'json'
      ? `<details class="gpi-evidence-json-details"><summary>${esc(tr('Open raw JSON for audit', '展开原始 JSON 审计内容'))}</summary>${jsonView(p)}</details>` : '';
    return `<details class="gpi-evidence-audit-details"><summary>${esc(tr('Full file audit', '完整文件审计'))}</summary><div class="gpi-evidence-audit-body">${locatorView(locator)}${p.description ? `<p>${esc(text(p.description, 500))}</p>` : ''}<dl>${facts.map(([label, value]) => `<div><dt>${esc(label)}</dt><dd><code>${esc(text(value, 500))}</code></dd></div>`).join('')}</dl><div class="gpi-evidence-record-identity"><span>${esc(text(p.kind || 'artifact', 80))}</span><code>${esc(text(p.evidence_id, 160))}</code><code>${esc(text(p.sha256, 64))}</code></div>${hasCode ? `<p class="gpi-evidence-line-note">${esc(tr('The run records the complete generating script, but not a field-to-line span. Open the script to inspect its numbered source.', '当前运行登记了完整生成脚本，但未登记字段到具体代码行的映射；点击后可查看带行号的只读源码。'))}</p>` : ''}${rawJson}</div></details>`;
  }
  function declaredLineageView(payload) {
    const rows = Array.isArray(payload && payload.declared_lineage) ? payload.declared_lineage : [];
    return `<section class="gpi-evidence-lineage"><div class="gpi-evidence-section-head"><span>${esc(tr('Reproduction path', '复现路径'))}</span></div>${reproductionPathView(payload, rows)}${rows.length ? `<div class="gpi-evidence-lineage-list">${rows.map(evidenceLink).join('')}</div>` : `<p>${esc(tr('This record has no direct parent evidence registered.', '该文件没有登记直接父证据。'))}</p>`}</section>`;
  }
  function runAuthorityView(payload) {
    const authority = payload && payload.run_authority && typeof payload.run_authority === 'object' ? payload.run_authority : {};
    if (authority.status !== 'recorded') {
      return `<details class="gpi-evidence-authority is-missing"><summary>${esc(tr('Upstream run provenance', '上层运行溯源'))}</summary><p>${esc(tr('No valid run manifest was recorded for this evidence.', '该证据没有可用的运行清单记录。'))}</p></details>`;
    }
    const facts = [
      [tr('Run', '运行'), authority.run_id],
      ['Git', authority.git_sha ? `${authority.git_sha}${authority.git_dirty === true ? ' · dirty' : ''}` : ''],
      [tr('Runner image', '执行镜像'), authority.runner_image_digest],
      [tr('Environment identity', '环境身份'), authority.environment_identity_sha256],
      [tr('Prompt pack', '提示包'), authority.prompt_pack_sha256 || authority.prompt_pack_version],
      [tr('Execution identity', '执行身份'), authority.execution_identity_sha256],
    ].filter(([, value]) => value !== '' && value != null);
    const links = Array.isArray(authority.links) ? authority.links : [];
    return `<details class="gpi-evidence-authority"><summary>${esc(tr('Continue to upstream run provenance', '继续查看上层运行溯源'))}</summary><div class="gpi-evidence-authority-body"><p>${esc(tr('These are run-level authority anchors, not inferred direct parents of the current file.', '这些是同一运行的权威锚点，不会被冒充为当前文件的直接父证据。'))}</p><dl>${facts.map(([label, value]) => `<div><dt>${esc(label)}</dt><dd><code>${esc(text(value, 500))}</code></dd></div>`).join('')}</dl>${links.length ? `<div class="gpi-evidence-lineage-list">${links.map(evidenceLink).join('')}</div>` : `<p>${esc(tr('No clickable run-authority evidence was registered.', '没有登记可点击的运行权威证据。'))}</p>`}</div></details>`;
  }
  function codeView(payload) {
    const lines = text(payload.text, 600000).split('\n');
    return `<div class="gpi-evidence-code" role="region" aria-label="${esc(tr('Read-only evidence code', '只读证据代码'))}" tabindex="0"><ol>${lines.map(line => `<li><code>${esc(line || ' ')}</code></li>`).join('')}</ol></div>`;
  }
  function jsonView(payload) {
    let value = '';
    try { value = JSON.stringify(payload.value, null, 2); } catch (_error) { value = tr('JSON preview unavailable.', 'JSON 预览不可用。'); }
    return `<pre class="gpi-preview-code gpi-evidence-json" tabindex="0"><code>${esc(value)}</code></pre>`;
  }
  function scalar(value, limit) {
    if (value == null || value === '') return '—';
    if (typeof value === 'number' && Number.isFinite(value)) return value.toLocaleString(undefined, { maximumFractionDigits: 3 });
    if (Array.isArray(value)) return value.slice(0, 8).map(item => scalar(item, 80)).join(', ');
    return text(value, limit || 180);
  }
  function percent(value) {
    const numeric = Number(value);
    return Number.isFinite(numeric) ? `${numeric.toFixed(Math.abs(numeric) >= 10 ? 1 : 2)}%` : '—';
  }
  function confidenceInterval(low, high) {
    if (low == null || high == null) return '—';
    return `${percent(low)}–${percent(high)}`;
  }
  function statisticView(payload) {
    const value = payload && payload.value && typeof payload.value === 'object' && !Array.isArray(payload.value)
      ? payload.value : {};
    const report = value.reportable_descriptive_results && typeof value.reportable_descriptive_results === 'object'
      ? value.reportable_descriptive_results : value;
    const overall = report.overall_outcome && typeof report.overall_outcome === 'object'
      ? report.overall_outcome : {};
    const facts = [];
    const nTotal = value.n_total != null ? value.n_total : overall.n;
    if (nTotal != null) facts.push([tr('Sample', '样本量'), scalar(nTotal)]);
    const outcome = overall.outcome || value.outcome;
    if (outcome) facts.push([tr('Outcome', '结局'), scalar(outcome)]);
    if (overall.event_n != null || value.outcome_event_n != null) facts.push([tr('Events', '事件数'), scalar(overall.event_n != null ? overall.event_n : value.outcome_event_n)]);
    if (overall.risk_pct != null) facts.push([tr('Overall risk', '总体风险'), `${percent(overall.risk_pct)} (${confidenceInterval(overall.risk_ci_low_pct, overall.risk_ci_high_pct)})`]);
    if (value.estimate != null) facts.push([tr('Estimate', '估计值'), scalar(value.estimate)]);
    if (Array.isArray(value.ci) && value.ci.length >= 2) facts.push([tr('95% CI', '95% CI'), `${scalar(value.ci[0])}–${scalar(value.ci[1])}`]);
    if (Array.isArray(value.exposure_columns) && value.exposure_columns.length) facts.push([tr('Exposure', '暴露'), scalar(value.exposure_columns)]);
    const factCards = facts.map(([label, fact]) => `<div class="gpi-evidence-statistic-card"><span>${esc(label)}</span><strong>${esc(fact)}</strong></div>`).join('');
    const method = value.method || report.method || '';
    const exposures = Array.isArray(report.exposures) ? report.exposures.slice(0, 12) : [];
    const exposureSections = exposures.map(exposure => {
      const groups = Array.isArray(exposure && exposure.groups) ? exposure.groups.slice(0, 24) : [];
      const distribution = exposure && exposure.continuous_distribution && typeof exposure.continuous_distribution === 'object'
        ? exposure.continuous_distribution : null;
      const groupTable = groups.length ? `<div class="gpi-evidence-table-wrap"><table class="gpi-evidence-table"><thead><tr><th>${esc(tr('Group', '分组'))}</th><th>${esc(tr('N', '例数'))}</th><th>${esc(tr('Events / N', '事件 / N'))}</th><th>${esc(tr('Risk', '风险'))}</th><th>${esc(tr('95% CI', '95% CI'))}</th></tr></thead><tbody>${groups.map(group => `<tr><td>${esc(scalar(group.label || group.group_value))}</td><td>${esc(scalar(group.n))}</td><td>${esc(`${scalar(group.outcome_event_n)} / ${scalar(group.outcome_n)}`)}</td><td>${esc(percent(group.outcome_risk_pct))}</td><td>${esc(confidenceInterval(group.outcome_risk_ci_low_pct, group.outcome_risk_ci_high_pct))}</td></tr>`).join('')}</tbody></table></div>` : '';
      const distributionTable = distribution ? `<div class="gpi-evidence-table-wrap"><table class="gpi-evidence-table"><thead><tr><th>${esc(tr('N', '例数'))}</th><th>${esc(tr('Median', '中位数'))}</th><th>${esc(tr('IQR', '四分位距'))}</th><th>${esc(tr('Range', '范围'))}</th></tr></thead><tbody><tr><td>${esc(scalar(distribution.n))}</td><td>${esc(scalar(distribution.median))}</td><td>${esc(`${scalar(distribution.q25)}–${scalar(distribution.q75)}`)}</td><td>${esc(`${scalar(distribution.minimum)}–${scalar(distribution.maximum)}`)}</td></tr></tbody></table></div>` : '';
      if (!groupTable && !distributionTable) return '';
      return `<section class="gpi-evidence-statistic-section"><h3>${esc(scalar(exposure.exposure || tr('Exposure', '暴露')))}${exposure.unit ? ` <small>${esc(scalar(exposure.unit, 80))}</small>` : ''}</h3>${groupTable}${distributionTable}</section>`;
    }).join('');
    const readable = factCards || method || exposureSections
      ? `<div class="gpi-evidence-statistic"><div class="gpi-evidence-statistic-head"><span>${esc(tr('Readable result', '结果摘要'))}</span><strong>${esc(tr('Registered statistic', '已登记统计量'))}</strong></div>${factCards ? `<div class="gpi-evidence-statistic-facts">${factCards}</div>` : ''}</div>` : '';
    const details = method || exposureSections
      ? `<details class="gpi-evidence-result-details"><summary>${esc(tr('Method and complete result tables', '方法与完整结果表'))}</summary><div>${method ? `<p class="gpi-evidence-statistic-method"><strong>${esc(tr('Method', '方法'))}</strong>${esc(scalar(method, 500))}</p>` : ''}${exposureSections}</div></details>` : '';
    return `${readable}${details}`;
  }
  function tableView(payload) {
    const headers = Array.isArray(payload.headers) ? payload.headers.slice(0, 24) : [];
    const rows = Array.isArray(payload.rows) ? payload.rows.slice(0, 100) : [];
    return `<div class="gpi-evidence-table-wrap" tabindex="0"><table class="gpi-evidence-table"><thead><tr>${headers.map(value => `<th>${esc(text(value, 160))}</th>`).join('')}</tr></thead><tbody>${rows.map(row => `<tr>${headers.map((_, index) => `<td>${esc(text(Array.isArray(row) ? row[index] : '', 1000))}</td>`).join('')}</tr>`).join('')}</tbody></table>${payload.rows_truncated || payload.columns_truncated ? `<p>${esc(tr('Preview is bounded; additional rows or columns are not shown.', '预览有界；其余行或列未展示。'))}</p>` : ''}</div>`;
  }
  function metadataView(payload) {
    const reasons = {
      patient_level_rows_withheld: tr('Patient-level cohort rows are withheld. File identity and digest remain visible.', '患者级队列行已隐藏；文件身份与摘要仍可核对。'),
      direct_identifier_columns_withheld: tr('This table contains direct identifier columns, so row preview is blocked.', '该表含直接标识符列，因此禁止预览数据行。'),
      non_result_table_rows_withheld: tr('Only result-owned aggregate tables can expose rows in this view.', '此处只允许预览分析结果所属的聚合表。'),
      preview_size_limit: tr('The file exceeds the bounded preview size.', '文件超过有界预览大小。'),
      evidence_preview_host_path_detected: tr('The text contains a host path and was withheld.', '文本包含主机路径，已隐藏。'),
      evidence_preview_encoding_unsupported: tr('The text encoding is not supported for safe preview.', '文本编码不符合安全预览要求。'),
      unsupported_evidence_type: tr('This registered evidence type has no browser renderer yet.', '该已登记证据类型暂没有浏览器渲染器。'),
    };
    const reason = String(payload.withheld_reason || 'unsupported_evidence_type');
    return `<div class="gpi-evidence-withheld"><strong>${esc(tr('Metadata-only preview', '仅元数据预览'))}</strong><p>${esc(reasons[reason] || reason)}</p>${payload.bytes != null ? `<span>${esc(String(payload.bytes))} bytes</span>` : ''}</div>`;
  }
  function render(payload, locator) {
    const p = payload && typeof payload === 'object' ? payload : {};
    const isStatistic = p.previewable && p.renderer === 'json' && p.kind === 'statistic';
    let body = metadataView(p);
    if (p.previewable && p.renderer === 'code') body = codeView(p);
    else if (isStatistic) body = statisticView(p);
    else if (p.previewable && p.renderer === 'json') body = jsonView(p);
    else if (p.previewable && p.renderer === 'table') body = tableView(p);
    return `<div class="gpi-evidence-view">${isStatistic ? body : ''}${recordView(p)}${declaredLineageView(p)}${fileAuditView(p, locator)}${runAuthorityView(p)}${isStatistic ? '' : body}<p class="gpi-evidence-readonly">${esc(tr('Read-only preview. Code is displayed, never executed; raw patient rows and absolute host paths remain outside the browser boundary.', '只读预览。代码只展示、不执行；原始患者行和主机绝对路径不会进入浏览器边界。'))}</p></div>`;
  }

  window.EU_GUIDED_PI_EVIDENCE_PREVIEW = { render, title, kindLabel };
})();
