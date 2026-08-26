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
    if (!source.pointer && source.value === '') return '';
    return `<div class="gpi-evidence-locator"><div><span>${esc(tr('JSON pointer', 'JSON 指针'))}</span><code>${esc(text(source.pointer, 500) || '—')}</code></div><div><span>${esc(tr('Bound source value', '绑定源数值'))}</span><code>${esc(text(source.value, 500) || '—')}</code></div></div>`;
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
    let body = metadataView(p);
    if (p.previewable && p.renderer === 'code') body = codeView(p);
    else if (p.previewable && p.renderer === 'json') body = jsonView(p);
    else if (p.previewable && p.renderer === 'table') body = tableView(p);
    return `<div class="gpi-evidence-view">${locatorView(locator)}<div class="gpi-evidence-meta"><span>${esc(text(p.kind || 'artifact', 80))}</span><code>${esc(text(p.evidence_id, 160))}</code><code>${esc(text(p.sha256, 64))}</code></div>${body}<p class="gpi-evidence-readonly">${esc(tr('Read-only preview. Code is displayed, never executed; raw patient rows and host paths remain outside the browser boundary.', '只读预览。代码只展示、不执行；原始患者行和主机路径不会进入浏览器边界。'))}</p></div>`;
  }

  window.EU_GUIDED_PI_EVIDENCE_PREVIEW = { render, title, kindLabel };
})();
