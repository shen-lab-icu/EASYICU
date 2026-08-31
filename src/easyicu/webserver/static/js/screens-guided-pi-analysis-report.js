/* Guided Copilot full analysis-report owner.
   It composes governed projections only; it does not rerun models or raise
   the run's analysis-only scientific authority. */
(function () {
  'use strict';

  const { esc } = window.EU_HTML;
  function tr(en, zh) { return window.EU_LANG === 'zh' ? zh : en; }

  const SOURCE_ARTIFACTS = [
    'run_context.json',
    'source_run_manifest.json',
    'manuscript_provenance.json',
    'quality_gate.json',
    'figure_gallery.json',
  ];

  async function load(api, projectId, runId) {
    if (!api || typeof api.loadPiCopilotResearchArtifact !== 'function') {
      throw new Error(tr('The research artifact API is unavailable.', '研究产物接口不可用。'));
    }
    const loaded = await Promise.all(SOURCE_ARTIFACTS.map(async name => {
      try {
        return [name, await api.loadPiCopilotResearchArtifact(projectId, runId, name)];
      } catch (error) {
        if (name === 'figure_gallery.json') return [name, null];
        throw error;
      }
    }));
    const rows = Object.fromEntries(loaded);
    const provenance = rows['manuscript_provenance.json'];
    return {
      payload: {
        schema_version: 'easyicu.web-full-analysis-report/1',
        run_context: (rows['run_context.json'] && rows['run_context.json'].payload) || {},
        source_manifest: (rows['source_run_manifest.json'] && rows['source_run_manifest.json'].payload) || {},
        manuscript_provenance: (provenance && provenance.payload) || {},
        quality_gate: (rows['quality_gate.json'] && rows['quality_gate.json'].payload) || {},
        figure_gallery: (rows['figure_gallery.json'] && rows['figure_gallery.json'].payload) || {},
      },
      governance: (provenance && provenance.governance) || null,
    };
  }

  function claims(payload) {
    const provenance = payload && payload.manuscript_provenance;
    return provenance && Array.isArray(provenance.claims) ? provenance.claims : [];
  }
  function findClaim(rows, patterns) {
    const rules = Array.isArray(patterns) ? patterns : [patterns];
    return rows.find(row => rules.some(rule => rule.test(String(row && row.source_field || '')))) || null;
  }
  function display(row, fallback) {
    return row && row.display_value != null ? String(row.display_value) : (fallback || '—');
  }
  function evidenceAttributes(row) {
    const evidence = row && row.evidence && typeof row.evidence === 'object' ? row.evidence : {};
    const evidenceId = String(evidence.evidence_id || '');
    const sha256 = String(evidence.sha256 || '').toLowerCase();
    if (!/^[A-Za-z0-9_.-]{1,160}$/.test(evidenceId) || !/^[a-f0-9]{64}$/.test(sha256)) return '';
    return ` data-gpi-evidence-open data-evidence-id="${esc(evidenceId)}" data-evidence-sha256="${esc(sha256)}" data-evidence-kind="${esc(evidence.kind || 'statistic')}" data-evidence-label="${esc(evidence.description || evidenceId)}" data-evidence-pointer="${esc(row.source_json_pointer || '')}" data-evidence-source-value="${esc(row.source_value || row.display_value || '')}"`;
  }
  function metric(label, value, note, row) {
    const tag = evidenceAttributes(row) ? 'button' : 'article';
    return `<${tag} class="gpi-analysis-metric"${evidenceAttributes(row)}><span>${esc(label)}</span><strong>${esc(value)}</strong><small>${esc(note)}</small></${tag}>`;
  }
  function evidenceValue(label, row, suffix) {
    return `<button type="button" class="gpi-analysis-evidence-value"${evidenceAttributes(row)}><span>${esc(label)}</span><strong>${esc(display(row))}${esc(suffix || '')}</strong><small>${esc(tr('Open exact registered source', '打开准确登记来源'))}</small></button>`;
  }
  function artifactButton(name, label) {
    return `<button type="button" data-gpi-report-artifact="${esc(name)}" data-gpi-report-label="${esc(label)}">${esc(label)}</button>`;
  }
  function cleanArticleText(value) {
    return String(value || '')
      .replace(/\s*\[(?!@)[A-Za-z_][A-Za-z0-9_.-]*\]/g, '')
      .replace(/\[@[^\]]+\]/g, '')
      .replace(/\*\*/g, '')
      .replace(/\s+([,.;:)])/g, '$1')
      .trim();
  }
  function sectionParagraphs(provenance, sectionName, limit) {
    const blocks = provenance && Array.isArray(provenance.article_blocks) ? provenance.article_blocks : [];
    const output = [];
    let active = false;
    for (const block of blocks) {
      const text = (Array.isArray(block && block.segments) ? block.segments : [])
        .map(segment => String(segment && segment.text || '')).join('');
      if (block && block.kind === 'heading' && Number(block.level || 2) === 2) {
        active = text.trim().toLowerCase() === sectionName.toLowerCase();
        continue;
      }
      if (active && block && block.kind === 'paragraph') {
        const cleaned = cleanArticleText(text);
        if (cleaned) output.push(cleaned);
        if (output.length >= limit) break;
      }
    }
    return output;
  }

  function render(payload) {
    const p = payload && typeof payload === 'object' ? payload : {};
    const context = p.run_context && typeof p.run_context === 'object' ? p.run_context : {};
    const provenance = p.manuscript_provenance && typeof p.manuscript_provenance === 'object' ? p.manuscript_provenance : {};
    const rows = claims(p);
    const sourceN = findClaim(rows, [/^cohort\.n_stays$/]);
    const eligibleN = findClaim(rows, [/^n_total$/]);
    const completeN = findClaim(rows, [/^n_complete_case$/, /complete_case_n$/]);
    const eventN = findClaim(rows, [/^n_events$/]);
    const descriptiveEvents = findClaim(rows, [/overall_outcome\.event_n$/]);
    const overallRisk = findClaim(rows, [/overall_outcome\.risk_pct$/]);
    const effect = findClaim(rows, [/^primary_or$/]);
    const low = findClaim(rows, [/^primary_or_ci\[0\]$/, /^primary_or_ci_low$/]);
    const high = findClaim(rows, [/^primary_or_ci\[1\]$/, /^primary_or_ci_high$/]);
    const discussion = sectionParagraphs(provenance, 'Discussion', 2);
    const limitations = sectionParagraphs(provenance, 'Limitations', 2);
    const gallery = window.AGENT_RENDER && typeof window.AGENT_RENDER.figureGallery === 'function'
      ? window.AGENT_RENDER.figureGallery(p.figure_gallery || {}) : '';
    const presentation = !!(p.figure_gallery && p.figure_gallery.presentation_variant);
    const interpretation = [
      effect && low && high
        ? tr(
          `The registered primary estimate is OR ${display(effect)} (95% CI ${display(low)}–${display(high)}). Its exposure, comparator, outcome and time window retain the definitions recorded by the run.`,
          `已登记的主要估计为 OR ${display(effect)}（95% CI ${display(low)}–${display(high)}）。暴露、对照、结局和时间窗均沿用本次运行中登记的定义。`,
        ) : '',
      tr(
        'Every displayed number and figure is projected from the registered run artifacts. This Web report does not recalculate estimates or infer a causal effect.',
        '所有展示的数值和图件均投影自已登记的运行产物；本 Web 报告不重新计算估计值，也不推断因果效应。',
      ),
      tr(
        'Primary and sensitivity estimates may answer different estimand questions. Their numerical values must be interpreted using their registered definitions rather than treated as interchangeable.',
        '主要估计与敏感性估计可能回答不同的 estimand 问题；必须依据各自登记的定义解读，不能把数值视为可以互换。',
      ),
    ].filter(Boolean);
    return `<div class="gpi-analysis-report ag-artifact-readable">
      <header class="gpi-analysis-hero"><div><span>${esc(tr('Complete analysis report', '完整分析报告'))}</span><h2>${esc(context.question || tr('Research question not recorded', '尚未记录研究问题'))}</h2><p>${esc(tr('Results, interpretation, robustness and limitations assembled only from governed run artifacts.', '仅根据受治理运行产物组成的结果、解读、稳健性和局限性报告。'))}</p></div><em>ANALYSIS ONLY</em></header>
      <section class="gpi-analysis-summary"><div><small>${esc(tr('Executive summary', '执行摘要'))}</small><h3>${esc(tr('What this analysis supports', '这项分析支持什么'))}</h3></div><p>${esc(tr('This report summarizes only claims registered by the run. The study population, exposure, comparator, outcome, time window and adjustment set retain their evidence-bound definitions; this presentation layer does not add a clinical or causal conclusion.', '本报告只汇总本次运行登记的结论。研究人群、暴露、对照、结局、时间窗和调整集均沿用证据绑定的定义；展示层不会新增临床或因果结论。'))}</p></section>
      <section class="gpi-analysis-metrics" aria-label="${esc(tr('Key registered results', '核心登记结果'))}">
        ${metric(tr('Source ICU stays', '来源 ICU stay'), display(sourceN), tr('Before eligibility filtering', '纳入条件筛选前'), sourceN)}
        ${metric(tr('Eligible stays', '符合条件 stay'), display(eligibleN), tr('Registered denominator', '已登记分母'), eligibleN)}
        ${metric(tr('Complete-case model', '完整病例模型'), display(completeN), eventN ? `${display(eventN)} ${tr('model events', '个模型事件')}` : tr('Primary model sample', '主要模型样本'), completeN)}
        ${metric(tr('Overall outcome risk', '总体结局风险'), display(overallRisk), descriptiveEvents ? `${display(descriptiveEvents)} / ${display(eligibleN)}` : tr('Descriptive result', '描述性结果'), overallRisk)}
      </section>
      <section class="gpi-analysis-section"><div class="gpi-analysis-section-head"><span>01</span><div><small>${esc(tr('Design and population', '设计与研究人群'))}</small><h3>${esc(tr('Registered analysis frame', '已登记分析框架'))}</h3></div></div><div class="gpi-analysis-two-col"><p>${esc(tr('The exact study design and population definitions remain bound to the approved plan and run context. This report displays registered denominators without inferring a different cohort, exposure, outcome, time window or adjustment set.', '准确的研究设计与人群定义继续绑定已批准计划和运行上下文。本报告只展示已登记分母，不推断不同的队列、暴露、结局、时间窗或调整集。'))}</p><div class="gpi-analysis-values">${evidenceValue(tr('Source cohort', '来源队列'), sourceN)}${evidenceValue(tr('Eligible analysis set', '符合条件的分析集'), eligibleN)}${evidenceValue(tr('Complete cases', '完整病例'), completeN)}</div></div></section>
      <section class="gpi-analysis-section"><div class="gpi-analysis-section-head"><span>02</span><div><small>${esc(tr('Results', '分析结果'))}</small><h3>${esc(tr('Primary association and absolute-risk context', '主要关联与绝对风险背景'))}</h3></div></div>${presentation ? `<div class="gpi-analysis-presentation-note"><strong>${esc(tr('Digest-verified presentation figures', '摘要核验后的展示图'))}</strong><span>${esc(tr('Re-rendered from registered source tables; original run figures and digests are unchanged.', '根据已登记源数据表重新排版；原始运行图件及其摘要保持不变。'))}</span></div>` : ''}${gallery || `<p>${esc(tr('No embedded figure is available.', '暂无可嵌入图件。'))}</p>`}</section>
      <section class="gpi-analysis-section is-interpretation"><div class="gpi-analysis-section-head"><span>03</span><div><small>${esc(tr('Result interpretation', '结果解读'))}</small><h3>${esc(tr('Clinical and statistical meaning', '临床与统计含义'))}</h3></div></div><ol>${interpretation.map(value => `<li>${esc(value)}</li>`).join('')}</ol>${discussion.length ? `<details><summary>${esc(tr('Show evidence-bound discussion text', '展开证据绑定的 Discussion 文本'))}</summary>${discussion.map(value => `<p>${esc(value)}</p>`).join('')}</details>` : ''}</section>
      <section class="gpi-analysis-section"><div class="gpi-analysis-section-head"><span>04</span><div><small>${esc(tr('Robustness and data quality', '稳健性与数据质量'))}</small><h3>${esc(tr('What was checked—and how to read it', '检查了什么，以及应如何理解'))}</h3></div></div><ul><li>${esc(tr('Displayed denominators and estimates come from registered evidence; unavailable values remain unavailable.', '展示的分母和估计值来自已登记证据；不可用的数值继续保持不可用。'))}</li><li>${esc(tr('Measurement opportunity, missingness and applicability must be interpreted using the run-specific audit artifacts.', '测量机会、缺失性和适用性必须依据本次运行的审计产物解读。'))}</li><li>${esc(tr('Primary and sensitivity rows must not be treated as independent or equivalent unless the registered analysis says so.', '除非已登记分析明确说明，否则不得把主要分析与敏感性分析视为相互独立或等价。'))}</li></ul></section>
      <section class="gpi-analysis-section is-limit"><div class="gpi-analysis-section-head"><span>05</span><div><small>${esc(tr('Limitations', '局限性'))}</small><h3>${esc(tr('What this report cannot prove', '这份报告不能证明什么'))}</h3></div></div>${limitations.length ? limitations.map(value => `<p>${esc(value)}</p>`).join('') : `<p>${esc(tr('Interpretation is limited to the design, data, estimand and evidence scope registered by this run. This report alone cannot establish causation, clinical validity or external generalizability.', '解释范围受本次运行登记的设计、数据、estimand 和证据边界限制；仅凭本报告不能确立因果关系、临床有效性或外部可推广性。'))}</p>`}</section>
      <nav class="gpi-analysis-links" aria-label="${esc(tr('Traceable report artifacts', '可追溯报告产物'))}"><span>${esc(tr('Open underlying artifact', '打开底层产物'))}</span>${artifactButton('result_tables.json', tr('Result tables', '结果表'))}${artifactButton('figure_gallery.json', tr('Figure gallery', '图件画廊'))}${artifactButton('manuscript_provenance.json', tr('Evidence-bound article', '证据绑定文章'))}${artifactButton('quality_gate.json', tr('Quality gate', '质量闸门'))}</nav>
    </div>`;
  }

  window.EU_GUIDED_PI_ANALYSIS_REPORT = { load, render };
})();
