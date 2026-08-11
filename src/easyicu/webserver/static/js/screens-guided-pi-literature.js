/* Guided Pi literature evidence renderer owner.
   Retrieval remains owned by Idea Mining / Research Agent. This module renders
   only host-projected metadata and never infers citations from prose. */
(function () {
  'use strict';

  function tr(en, zh) { return window.EU_LANG === 'zh' ? zh : en; }
  function esc(value) {
    return String(value == null ? '' : value)
      .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;').replace(/'/g, '&#39;');
  }
  function safeUrl(value) {
    try {
      const parsed = new URL(String(value || ''));
      if (parsed.protocol !== 'https:' || !parsed.hostname || parsed.username || parsed.password) return '';
      return parsed.href;
    } catch (_) { return ''; }
  }
  function sourceMeta(row) {
    return [row.venue, row.year, row.pmid ? `PMID ${row.pmid}` : '', row.doi ? `DOI ${row.doi}` : '']
      .filter(Boolean).map(esc).join(' · ');
  }
  function articleCard(row, indexByKey) {
    const url = safeUrl(row.source_url || row.url);
    const title = row.title || row.label || row.key || tr('Untitled source', '未命名文献');
    const relevance = row.relevance || '';
    const key = row.key || '';
    if (key && indexByKey) indexByKey.set(String(key), row);
    return `<article class="gpi-lit-card">
      <div class="gpi-lit-card-head"><span class="gpi-lit-kind">${esc(tr('Literature', '文献'))}</span>${key ? `<code>${esc(key)}</code>` : ''}</div>
      <h4>${esc(title)}</h4>
      ${sourceMeta(row) ? `<div class="gpi-lit-meta">${sourceMeta(row)}</div>` : ''}
      ${relevance ? `<p>${esc(relevance)}</p>` : ''}
      ${url ? `<a href="${esc(url)}" target="_blank" rel="noopener noreferrer">${esc(tr('Open source record', '打开来源页面'))}<span aria-hidden="true">↗</span></a>` : `<span class="gpi-lit-no-link">${esc(tr('No verified source link in this artifact', '该产物没有已核验的来源链接'))}</span>`}
    </article>`;
  }
  function searchSummary(payload) {
    const search = payload.search && typeof payload.search === 'object' ? payload.search : {};
    const searched = !!search.search_conducted;
    const title = searched
      ? tr('Retrieval was performed', '已执行真实检索')
      : tr('Curated references only', '仅使用预置参考文献');
    const detail = search.note || (searched
      ? tr('The listed retrieval sources returned this bundle.', '下列检索来源返回了这个文献包。')
      : tr('No search was performed; do not describe this as a systematic search.', '本次没有执行检索，不得描述为系统检索。'));
    const sources = Array.isArray(search.sources_returning) ? search.sources_returning : [];
    const prisma = search.prisma && typeof search.prisma === 'object' ? search.prisma : null;
    return `<section class="gpi-lit-search ${searched ? 'searched' : 'curated'}">
      <div><span class="gpi-lit-status-dot" aria-hidden="true"></span><strong>${esc(title)}</strong></div>
      <p>${esc(detail)}</p>
      ${sources.length ? `<small>${esc(tr('Sources: ', '来源：'))}${sources.map(esc).join(', ')}</small>` : ''}
      ${prisma ? `<small>${esc(tr('Flow: ', '检索流：'))}${esc(JSON.stringify(prisma))}</small>` : ''}
    </section>`;
  }
  function planMap(payload, indexByKey) {
    const rows = Array.isArray(payload.step_citation_map) ? payload.step_citation_map : [];
    if (!rows.length) return '';
    const boundCount = rows.filter(row => Array.isArray(row.citation_keys) && row.citation_keys.length).length;
    const auxiliaryCount = rows.filter(row => {
      const keys = Array.isArray(row.citation_keys) ? row.citation_keys : [];
      return !keys.length && String(row.planned_analysis_role || '') === 'auxiliary';
    }).length;
    return `<section class="gpi-lit-map">
      <header><h3>${esc(tr('Plan decisions and supporting articles', '计划决策与支持文献'))}</h3><span>${esc(tr(`${boundCount} evidence-bound · ${auxiliaryCount} auxiliary`, `${boundCount} 个科学决策已绑定 · ${auxiliaryCount} 个辅助步骤`))}</span></header>
      <div class="gpi-lit-map-list">${rows.map(row => {
        const keys = Array.isArray(row.citation_keys) ? row.citation_keys : [];
        const sources = keys.map(key => indexByKey.get(String(key))).filter(Boolean);
        const auxiliary = String(row.planned_analysis_role || '') === 'auxiliary';
        return `<article class="gpi-lit-step ${keys.length ? 'bound' : 'unbound'}">
          <div><code>${esc(row.step_id || '')}</code>${row.planned_analysis_role ? `<span>${esc(row.planned_analysis_role)}</span>` : ''}</div>
          <p>${esc(row.intent || '')}</p>
          ${sources.length ? `<ul>${sources.map(source => {
            const url = safeUrl(source.source_url || source.url);
            const label = esc(source.title || source.key);
            return `<li>${url ? `<a href="${esc(url)}" target="_blank" rel="noopener noreferrer">${label}<span aria-hidden="true">↗</span></a>` : `<strong>${label}</strong>`}<small>${esc(source.key || '')}</small></li>`;
          }).join('')}</ul>` : `<div class="gpi-lit-unbound">${esc(auxiliary
            ? tr('Auxiliary execution or rendering step; it inherits the governed scientific plan and is not counted as a missing literature decision.', '辅助执行或呈现步骤；它继承受治理的科学计划，不计作文献决策缺口。')
            : tr('A scientific decision has no bound citation and requires review.', '该科学决策没有绑定文献，需要审阅。'))}</div>`}
        </article>`;
      }).join('')}</div>
    </section>`;
  }
  function renderArtifact(payload) {
    const p = payload && typeof payload === 'object' ? payload : {};
    const citations = Array.isArray(p.citations) ? p.citations : [];
    const indexByKey = new Map();
    const cards = citations.map(row => articleCard(row, indexByKey)).join('');
    return `<div class="gpi-lit-view">
      ${searchSummary(p)}
      <div class="gpi-lit-boundary" role="note"><strong>${esc(tr('Evidence boundary', '证据边界'))}</strong><span>${esc(p.evidence_boundary || tr('Literature supports design rationale; patient/result evidence is governed separately.', '文献支持设计依据；患者与结果证据由另一条证据链治理。'))}</span></div>
      ${planMap(p, indexByKey)}
      <section class="gpi-lit-library"><header><h3>${esc(tr('Article library', '文献库'))}</h3><span>${esc(String(citations.length))}</span></header>${cards || `<div class="gpi-lit-empty">${esc(tr('No projected articles are available.', '没有可预览的文献。'))}</div>`}</section>
    </div>`;
  }
  function renderSource(resource) {
    const row = {
      title: resource.title || resource.label,
      year: resource.year,
      venue: resource.venue,
      relevance: resource.relevance,
      doi: resource.doi,
      pmid: resource.pmid,
      source_url: resource.url,
    };
    return `<div class="gpi-lit-view source-only">
      <section class="gpi-lit-search searched"><div><span class="gpi-lit-status-dot" aria-hidden="true"></span><strong>${esc(tr('PubMed search result', 'PubMed 检索结果'))}</strong></div><p>${esc(tr('This metadata came from the user-authorized Idea Mining search receipt.', '此元数据来自用户授权的 Idea Mining 检索回执。'))}</p></section>
      ${articleCard(row)}
      <div class="gpi-lit-boundary" role="note"><strong>${esc(tr('Metadata / abstract evidence', '元数据 / 摘要证据'))}</strong><span>${esc(tr('Open and appraise the source before treating it as scientific authority. No full text was stored.', '在把它作为科学依据前仍需打开并审阅原文；系统未保存全文。'))}</span></div>
    </div>`;
  }

  window.EU_GUIDED_PI_LITERATURE = { renderArtifact, renderSource, safeUrl };
})();
