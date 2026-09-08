/* Guided Copilot full analysis-report owner.
   It composes governed projections only; it does not rerun models or raise
   the run's analysis-only scientific authority. */
(function () {
  'use strict';

  const { esc } = window.EU_HTML;
  function tr(en, zh) { return window.EU_LANG === 'zh' ? zh : en; }

  const SOURCE_ARTIFACTS = [
    'agent_plan.json',
    'run_context.json',
    'source_run_manifest.json',
    'manuscript_provenance.json',
    'quality_gate.json',
    'figure_gallery.json',
    'result_tables.json',
  ];

  async function load(api, projectId, runId, resource) {
    if (!api || typeof api.loadPiCopilotResearchArtifact !== 'function') {
      throw new Error(tr('The research artifact API is unavailable.', '研究产物接口不可用。'));
    }
    const loader = window.EasyICU.guidedPi.require('reportArtifacts');
    if (!loader || typeof loader.load !== 'function') throw new Error('The report artifact loader is unavailable.');
    const rows = await loader.load(api, projectId, runId, SOURCE_ARTIFACTS, resource, ['figure_gallery.json']);
    const provenance = rows['manuscript_provenance.json'];
    return {
      payload: {
        schema_version: 'easyicu.web-full-analysis-report/1',
        plan: (rows['agent_plan.json'] && rows['agent_plan.json'].payload) || {},
        run_context: (rows['run_context.json'] && rows['run_context.json'].payload) || {},
        source_manifest: (rows['source_run_manifest.json'] && rows['source_run_manifest.json'].payload) || {},
        manuscript_provenance: (provenance && provenance.payload) || {},
        quality_gate: (rows['quality_gate.json'] && rows['quality_gate.json'].payload) || {},
        figure_gallery: (rows['figure_gallery.json'] && rows['figure_gallery.json'].payload) || {},
        result_tables: (rows['result_tables.json'] && rows['result_tables.json'].payload) || {},
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
    const sourceManifest = p.source_manifest && typeof p.source_manifest === 'object' ? p.source_manifest : {};
    const provenance = p.manuscript_provenance && typeof p.manuscript_provenance === 'object' ? p.manuscript_provenance : {};
    const manuscriptReady = !!(sourceManifest.readiness && sourceManifest.readiness.manuscript_ready === true);
    const resultSummary = window.EasyICU.guidedPi.optional('resultSummary');
    const registeredSummary = resultSummary
      ? resultSummary.summarize(p.result_tables || {}, p.plan)
      : { claims: [], exposureLevels: [] };
    const rows = claims(p).concat(registeredSummary.claims || []);
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
    const figureCount = Array.isArray(p.figure_gallery && p.figure_gallery.figures)
      ? p.figure_gallery.figures.length : 0;
    const planSteps = Array.isArray(p.plan && p.plan.steps) ? p.plan.steps : [];
    const groupRows = registeredSummary.exposureLevels || [];
    const count = value => value == null ? '—' : Number(value).toLocaleString('en-US');
    const pct = value => value == null ? '—' : `${Number(value).toFixed(2)}%`;
    const groupTable = groupRows.length ? `<div class="gpi-analysis-table-scroll"><table><caption>${esc(registeredSummary.outcomeLabel || tr('Recorded outcome', '已登记结局'))}</caption><thead><tr>${[
      tr('Group', '分组'), tr('Records', '记录数'), tr('Cohort share', '占队列比例'),
      tr('Outcome events / observed records', '结局事件数 / 有结局记录数'), tr('Observed proportion', '结局比例'),
    ].map(label => `<th scope="col">${esc(label)}</th>`).join('')}</tr></thead><tbody>${groupRows.map(row => `<tr><th scope="row">${esc(row.label)}</th><td>${esc(count(row.n))}</td><td>${esc(pct(row.sharePct))}</td><td>${esc(count(row.events))} / ${esc(count(row.denominator))}</td><td>${esc(pct(row.outcomeRatePct))}</td></tr>`).join('')}</tbody></table></div>` : '';
    const article = manuscriptReady && window.AGENT_RENDER && typeof window.AGENT_RENDER.manuscriptProvenanceView === 'function'
      ? window.AGENT_RENDER.manuscriptProvenanceView(provenance) : '';
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
      effect ? tr(
        'Primary and sensitivity estimates may answer different estimand questions. Their numerical values must be interpreted using their registered definitions rather than treated as interchangeable.',
        '主要估计与敏感性估计可能回答不同的 estimand 问题；必须依据各自登记的定义解读，不能把数值视为可以互换。',
      ) : tr('The table describes observed groups. It does not estimate an adjusted association or establish an exposure effect.', '上表描述各组实际观察到的分布，没有估计调整后的关联，也不能证明暴露造成了结局差异。'),
    ].filter(Boolean);
    return `<div class="gpi-analysis-report ag-artifact-readable">
      <header class="gpi-analysis-hero"><div><span>${esc(tr('Complete analysis report', '完整分析报告'))}</span><h2>${esc(context.question || tr('Research question not recorded', '尚未记录研究问题'))}</h2><p>${esc(tr('Results, interpretation, robustness and limitations assembled only from governed run artifacts.', '仅根据受治理运行产物组成的结果、解读、稳健性和局限性报告。'))}</p></div><em>ANALYSIS ONLY</em></header>
      <section class="gpi-analysis-summary"><div><small>${esc(tr('Results at a glance', '先看结果'))}</small><h3>${esc(tr('What was observed', '实际观察到了什么'))}</h3></div><div>${groupTable || `<p>${esc(tr('Read the registered result tables and manuscript below; no compatible summary table is available.', '下方提供已登记的结果表与文章；当前没有可直接汇总的分组表。'))}</p>`}<p>${esc(tr('These are analysis records, not necessarily independent patients. Percentages describe observed data; no significance test or causal conclusion is implied.', '这里统计的是分析记录，不一定是相互独立的患者。比例描述实际数据，不代表显著性检验或因果结论。'))}</p></div></section>
      <section class="gpi-analysis-metrics" aria-label="${esc(tr('Key registered results', '核心登记结果'))}">
        ${metric(tr('Source ICU stays', '来源 ICU stay'), display(sourceN), tr('Before eligibility filtering', '纳入条件筛选前'), sourceN)}
        ${metric(tr('Eligible stays', '符合条件 stay'), display(eligibleN), tr('Registered denominator', '已登记分母'), eligibleN)}
        ${metric(tr('Observed outcome events', '观察到的结局事件'), display(eventN), tr('Recorded event count', '已登记事件数'), eventN)}
        ${metric(tr('Overall outcome risk', '总体结局风险'), display(overallRisk), descriptiveEvents ? `${display(descriptiveEvents)} / ${display(eligibleN)}` : tr('Descriptive result', '描述性结果'), overallRisk)}
      </section>
      <section class="gpi-analysis-section"><div class="gpi-analysis-section-head"><span>01</span><div><small>${esc(tr('Design and population', '设计与研究人群'))}</small><h3>${esc(tr('The registered plan', '本次采用的研究计划'))}</h3></div></div><p>${esc(tr('Data source: ', '数据源：'))}${esc(context.source && (context.source.label || context.source.database) || '—')} · ${artifactButton('agent_plan.json', tr('Full plan and definitions', '完整计划与定义'))}</p><ol>${planSteps.map(step => `<li>${esc(step.intent || step.method || '')}</li>`).join('')}</ol>${completeN ? `<p>${esc(tr('Model-complete records: ', '模型完整记录：'))}${esc(display(completeN))}</p>` : ''}</section>
      <section class="gpi-analysis-section"><div class="gpi-analysis-section-head"><span>02</span><div><small>${esc(tr('Results', '分析结果'))}</small><h3>${esc(effect ? tr('Primary association and absolute-risk context', '主要关联与绝对风险背景') : tr('Exposure distribution and outcome-risk context', '暴露分布与结局风险背景'))}</h3></div></div>${presentation ? `<div class="gpi-analysis-presentation-note"><strong>${esc(tr('Digest-verified presentation figures', '摘要核验后的展示图'))}</strong><span>${esc(tr('Re-rendered from registered source tables; original run figures and digests are unchanged.', '根据已登记源数据表重新排版；原始运行图件及其摘要保持不变。'))}</span></div>` : ''}${gallery || `<p>${esc(tr('No embedded figure is available.', '暂无可嵌入图件。'))}</p>`}</section>
      <section class="gpi-analysis-section is-interpretation"><div class="gpi-analysis-section-head"><span>03</span><div><small>${esc(tr('Result interpretation', '结果解读'))}</small><h3>${esc(tr('Clinical and statistical meaning', '临床与统计含义'))}</h3></div></div><ol>${interpretation.map(value => `<li>${esc(value)}</li>`).join('')}</ol>${discussion.length ? `<details><summary>${esc(tr('Show evidence-bound discussion text', '展开证据绑定的 Discussion 文本'))}</summary>${discussion.map(value => `<p>${esc(value)}</p>`).join('')}</details>` : ''}</section>
      <section class="gpi-analysis-section"><div class="gpi-analysis-section-head"><span>04</span><div><small>${esc(tr('Robustness and data quality', '稳健性与数据质量'))}</small><h3>${esc(tr('What was checked—and how to read it', '检查了什么，以及应如何理解'))}</h3></div></div><ul><li>${esc(tr('Displayed denominators and estimates come from registered evidence; unavailable values remain unavailable.', '展示的分母和估计值来自已登记证据；不可用的数值继续保持不可用。'))}</li><li>${esc(tr('Measurement opportunity, missingness and applicability must be interpreted using the run-specific audit artifacts.', '测量机会、缺失性和适用性必须依据本次运行的审计产物解读。'))}</li><li>${esc(tr('Primary and sensitivity rows must not be treated as independent or equivalent unless the registered analysis says so.', '除非已登记分析明确说明，否则不得把主要分析与敏感性分析视为相互独立或等价。'))}</li></ul></section>
      <section class="gpi-analysis-section is-limit"><div class="gpi-analysis-section-head"><span>05</span><div><small>${esc(tr('Limitations', '局限性'))}</small><h3>${esc(tr('What this report cannot prove', '这份报告不能证明什么'))}</h3></div></div>${limitations.length ? limitations.map(value => `<p>${esc(value)}</p>`).join('') : `<p>${esc(tr('Interpretation is limited to the design, data, estimand and evidence scope registered by this run. This report alone cannot establish causation, clinical validity or external generalizability.', '解释范围受本次运行登记的设计、数据、estimand 和证据边界限制；仅凭本报告不能确立因果关系、临床有效性或外部可推广性。'))}</p>`}</section>
      ${article ? `<section class="gpi-analysis-section"><div class="gpi-analysis-section-head"><span>06</span><div><small>${esc(tr('Manuscript', '文章'))}</small><h3>${esc(tr('Full text, tables and references', '完整正文、结果表与参考文献'))}</h3></div></div>${article}</section>` : ''}
      <nav class="gpi-analysis-links" aria-label="${esc(tr('Traceable report artifacts', '可追溯报告产物'))}"><span>${esc(tr('Supporting files', '补充文件'))}</span>${artifactButton('result_tables.json', tr('Result tables', '结果表'))}${figureCount > 0 ? artifactButton('figure_gallery.json', tr('Figure gallery', '图件画廊')) : ''}${artifactButton('manuscript_provenance.json', manuscriptReady ? tr('Evidence-bound article', '证据绑定文章') : tr('Manuscript generation diagnosis', '稿件生成诊断'))}${artifactButton('quality_gate.json', tr('Quality gate', '质量闸门'))}</nav>
    </div>`;
  }

  window.EasyICU.guidedPi.declare('analysisReport', { load, render });
})();
