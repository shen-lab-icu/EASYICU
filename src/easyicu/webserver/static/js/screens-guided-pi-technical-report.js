/* Guided Copilot technical-report projection owner.
   Composes already-governed run artifacts into a concise human report. It
   never recalculates scientific estimates or changes their claim authority. */
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
        schema_version: 'easyicu.web-technical-report/1',
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

  function claimRows(payload) {
    const provenance = payload && payload.manuscript_provenance;
    return provenance && Array.isArray(provenance.claims) ? provenance.claims : [];
  }
  function claimByField(claims, patterns) {
    const rules = Array.isArray(patterns) ? patterns : [patterns];
    return claims.find(row => rules.some(rule => rule.test(String(row && row.source_field || '')))) || null;
  }
  function numeric(row) {
    if (!row || row.canonical_value == null) return null;
    const value = Number(row && row.canonical_value);
    return Number.isFinite(value) ? value : null;
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
    return `<${tag} class="gpi-tech-metric"${evidenceAttributes(row)}><span>${esc(label)}</span><strong>${esc(value)}</strong><small>${esc(note)}</small></${tag}>`;
  }
  function bar(label, row, total) {
    const value = numeric(row);
    const denominator = Number(total);
    const width = value != null && denominator > 0 ? Math.max(2, Math.min(100, value / denominator * 100)) : 0;
    return `<div class="gpi-tech-bar"><div><span>${esc(label)}</span><strong>${esc(display(row))}</strong></div><i><b style="width:${width.toFixed(2)}%"></b></i></div>`;
  }
  function riskBar(label, row, maxValue) {
    const value = numeric(row);
    const width = value != null && maxValue > 0 ? Math.max(2, Math.min(100, value / maxValue * 100)) : 0;
    return `<button class="gpi-tech-risk"${evidenceAttributes(row)}><span>${esc(label)}</span><i><b style="width:${width.toFixed(2)}%"></b></i><strong>${esc(display(row))}</strong></button>`;
  }
  function artifactButton(name, label) {
    return `<button type="button" data-gpi-report-artifact="${esc(name)}" data-gpi-report-label="${esc(label)}">${esc(label)}</button>`;
  }
  function effectPlot(effect, low, high) {
    const point = numeric(effect);
    const lower = numeric(low);
    const upper = numeric(high);
    if (point == null || lower == null || upper == null) return '';
    const min = Math.min(0.8, lower * 0.9);
    const max = Math.max(2, upper * 1.1);
    const position = value => Math.max(0, Math.min(100, (value - min) / (max - min) * 100));
    const left = position(lower);
    const right = position(upper);
    const reference = position(1);
    return `<div class="gpi-tech-effect" aria-label="${esc(tr('Effect estimate and confidence interval', '效应估计与置信区间'))}">
      <div><span>${esc(tr('Lower', '下限'))} ${esc(display(low))}</span><strong>OR ${esc(display(effect))}</strong><span>${esc(tr('Upper', '上限'))} ${esc(display(high))}</span></div>
      <i><em style="left:${reference.toFixed(2)}%"></em><b style="left:${left.toFixed(2)}%;width:${Math.max(1, right - left).toFixed(2)}%"></b><u style="left:${position(point).toFixed(2)}%"></u></i>
      <small>${esc(tr('Reference OR = 1. Every displayed value comes from the registered result evidence.', '参照线 OR＝1；显示值均来自已登记结果证据。'))}</small>
    </div>`;
  }

  function render(payload) {
    const p = payload && typeof payload === 'object' ? payload : {};
    const context = p.run_context && typeof p.run_context === 'object' ? p.run_context : {};
    const manifest = p.source_manifest && typeof p.source_manifest === 'object' ? p.source_manifest : {};
    const resultSummary = window.EasyICU.guidedPi.optional('resultSummary');
    const registeredSummary = resultSummary
      ? resultSummary.summarize(p.result_tables || {})
      : { claims: [], exposureLevels: [] };
    const claims = claimRows(p).concat(registeredSummary.claims || []);
    const total = claimByField(claims, [/^cohort\.n_stays$/, /^n_total$/]);
    const measured = claimByField(claims, [/plausibility_audit\..*\.compared_n$/, /input_bindings\[0\]\.row_count$/, /^exposure_distribution\.n_total$/]);
    const complete = claimByField(claims, [/^n_complete_case$/, /complete_case_n$/]);
    const risk = claimByField(claims, [/overall_outcome\.risk_pct$/]);
    const effect = claimByField(claims, [/^primary_or$/]);
    const effectLow = claimByField(claims, [/^primary_or_ci\[0\]$/, /^primary_or_ci_low$/]);
    const effectHigh = claimByField(claims, [/^primary_or_ci\[1\]$/, /^primary_or_ci_high$/]);
    const measuredRisk = claimByField(claims, [/exposures\[0\]\.groups\[0\]\.outcome_risk_pct$/]);
    const unmeasuredRisk = claimByField(claims, [/exposures\[0\]\.groups\[1\]\.outcome_risk_pct$/]);
    const registeredLevels = Array.isArray(registeredSummary.exposureLevels)
      ? registeredSummary.exposureLevels : [];
    const totalN = numeric(total);
    const maxRisk = Math.max(numeric(measuredRisk) || 0, numeric(unmeasuredRisk) || 0, 1);
    const readiness = manifest.readiness && typeof manifest.readiness === 'object' ? manifest.readiness : {};
    const status = readiness.manuscript_ready ? tr('Manuscript ready · publication not authorized', '稿件就绪 · 尚未获发表授权')
      : readiness.analysis_validated ? tr('Analysis validated · review pending', '分析已验证 · 等待审阅')
      : tr('Analysis-only · review required', '仅供分析 · 需要审阅');
    const findings = [
      total ? tr(`The registered cohort contains ${display(total)} ICU stays.`, `已登记研究队列包含 ${display(total)} 次 ICU 住院。`) : '',
      risk ? tr(`The registered overall outcome risk is ${display(risk)}.`, `已登记的总体结局风险为 ${display(risk)}。`) : '',
      effect && effectLow && effectHigh ? tr(`The primary registered contrast is OR ${display(effect)} (95% CI ${display(effectLow)}–${display(effectHigh)}).`, `主要登记对比为 OR ${display(effect)}（95% CI ${display(effectLow)}–${display(effectHigh)}）。`) : '',
    ].filter(Boolean);
    const limitations = [
      effect
        ? tr('The registered association does not establish causation.', '已登记的关联不能据此认定因果关系。')
        : tr('This report summarizes registered outputs and does not establish causation or clinical validity.', '本报告汇总已登记产物，不能据此认定因果关系或临床有效性。'),
      measuredRisk && unmeasuredRisk ? (registeredLevels.length
        ? tr('Differences across registered exposure levels are descriptive and do not establish a causal effect.', '登记暴露层级之间的差异仅为描述性结果，不能据此认定因果效应。')
        : tr('Outcome risk differs between measured and unmeasured source states; selective measurement matters for interpretation.', '有测量记录与无测量记录的结局风险不同，选择性测量会影响解释。')) : '',
      complete && total ? (effect
        ? tr(`The primary model uses ${display(complete)} complete cases from the registered cohort.`, `主要模型使用已登记队列中的 ${display(complete)} 个完整病例。`)
        : tr(`${display(complete)} rows are complete for every requested variable; this is a data-completeness count, not a fitted-model sample.`, `${display(complete)} 条记录的所有请求变量完整；这是数据完整性计数，不是拟合模型样本。`)) : '',
    ].filter(Boolean);
    const gallery = window.AGENT_RENDER && typeof window.AGENT_RENDER.figureGallery === 'function'
      ? window.AGENT_RENDER.figureGallery(p.figure_gallery || {}) : '';
    const presentationNote = p.figure_gallery && p.figure_gallery.presentation_variant
      ? `<div class="gpi-tech-presentation-note"><strong>${esc(tr('Digest-verified presentation view', '摘要核验后的展示视图'))}</strong><span>${esc(tr('Re-rendered from registered source tables; original run figures remain unchanged.', '根据已登记源数据表重新排版；原始运行图件保持不变。'))}</span></div>`
      : '';
    const figureCount = Array.isArray(p.figure_gallery && p.figure_gallery.figures)
      ? p.figure_gallery.figures.length : 0;
    return `<div class="gpi-tech-report ag-artifact-readable">
      <header class="gpi-tech-hero"><div><span>${esc(tr('Technical analysis report', '技术分析报告'))}</span><h2>${esc(context.question || tr('Research question not recorded', '尚未记录研究问题'))}</h2><p>${esc(tr('A concise Web view assembled from registered run artifacts; no estimate is recalculated here.', '由已登记运行产物组合形成的简明 Web 视图；此处不重新计算任何估计值。'))}</p></div><em>${esc(status)}</em></header>
      <section class="gpi-tech-metrics" aria-label="${esc(tr('Key results', '核心结果'))}">
        ${metric(tr('Cohort', '研究队列'), display(total), tr('ICU stays', '次 ICU 住院'), total)}
        ${metric(tr('Complete-data rows', '完整变量行'), display(complete), tr('All requested variables complete', '所有请求变量完整'), complete)}
        ${metric(tr('Overall outcome risk', '总体结局风险'), display(risk), tr('Registered descriptive result', '已登记描述性结果'), risk)}
        ${metric(tr('Primary association', '主要关联'), effect ? `OR ${display(effect)}` : tr('Descriptive only', '仅描述性'), effectLow && effectHigh ? `95% CI ${display(effectLow)}–${display(effectHigh)}` : tr('No adjusted effect was authorized', '未授权调整后效应估计'), effect)}
      </section>
      <div class="gpi-tech-grid">
        <section class="gpi-tech-panel"><div class="gpi-tech-section-head"><span>01</span><div><small>${esc(tr('Population', '样本构成'))}</small><h3>${esc(tr('How the analysis denominator narrows', '分析分母如何收窄'))}</h3></div></div>${bar(tr('Registered cohort', '登记队列'), total, totalN)}${bar(registeredLevels.length ? tr('Rows in registered distribution', '进入登记分布表的记录') : tr('Exposure measured', '具有暴露测量'), measured, totalN)}${bar(effect ? tr('Complete-case model', '完整病例模型') : tr('Complete-data rows', '完整变量行'), complete, totalN)}</section>
        <section class="gpi-tech-panel"><div class="gpi-tech-section-head"><span>02</span><div><small>${esc(tr('Outcome context', '结局背景'))}</small><h3>${esc(registeredLevels.length ? tr('Outcome rates by registered exposure level', '按登记暴露层级展示结局率') : tr('Measured versus unmeasured source states', '有测量与无测量记录比较'))}</h3></div></div>${measuredRisk && unmeasuredRisk ? `${riskBar(registeredLevels[0] ? tr(`Exposure level ${registeredLevels[0].level}`, `暴露层级 ${registeredLevels[0].level}`) : tr('Measured', '有测量记录'), measuredRisk, maxRisk)}${riskBar(registeredLevels[1] ? tr(`Exposure level ${registeredLevels[1].level}`, `暴露层级 ${registeredLevels[1].level}`) : tr('Not measured', '无测量记录'), unmeasuredRisk, maxRisk)}` : `<p class="gpi-tech-empty">${esc(tr('This run did not register a two-state outcome comparison.', '本次运行未登记两种状态的结局比较。'))}</p>`}</section>
      </div>
      <section class="gpi-tech-panel is-effect"><div class="gpi-tech-section-head"><span>03</span><div><small>${esc(tr('Primary estimate', '主要估计'))}</small><h3>${esc(tr('Association and uncertainty', '关联强度及不确定性'))}</h3></div></div>${effectPlot(effect, effectLow, effectHigh) || `<p class="gpi-tech-empty">${esc(tr('A bounded primary effect was not available.', '没有可展示的有界主要效应估计。'))}</p>`}</section>
      <section class="gpi-tech-findings"><div><small>${esc(tr('What the run found', '本次分析发现'))}</small><h3>${esc(tr('Core findings', '核心发现'))}</h3></div><ol>${findings.map(value => `<li>${esc(value)}</li>`).join('')}</ol></section>
      ${gallery ? `<section class="gpi-tech-figures"><div><small>${esc(tr('Registered figures', '已登记图件'))}</small><h3>${esc(tr('Visual results from this run', '本次运行的可视化结果'))}</h3></div>${presentationNote}${gallery}</section>` : ''}
      <section class="gpi-tech-limit"><div><small>${esc(tr('Interpretation boundary', '解释边界'))}</small><h3>${esc(tr('Read before using the result', '使用结果前请注意'))}</h3></div><ul>${limitations.map(value => `<li>${esc(value)}</li>`).join('')}</ul></section>
      <nav class="gpi-tech-links" aria-label="${esc(tr('Detailed run artifacts', '详细运行产物'))}"><span>${esc(tr('Continue to details', '继续查看详情'))}</span>${artifactButton('result_tables.json', tr('Result tables', '结果表'))}${figureCount > 0 ? artifactButton('figure_gallery.json', tr('Figures', '图件')) : ''}${artifactButton('manuscript_provenance.json', tr('Evidence-bound article', '证据绑定文章'))}${artifactButton('quality_gate.json', tr('Validation gate', '验证闸门'))}</nav>
    </div>`;
  }

  window.EasyICU.guidedPi.declare('technicalReport', { load, render });
})();
