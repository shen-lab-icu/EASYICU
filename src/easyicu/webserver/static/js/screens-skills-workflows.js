/* Owner: read-only research workflow list, graph and hub-start intent. */
(function () {
  'use strict';
  const { esc } = window.EU_HTML;
  const tr = (en, zh) => window.t(en, zh);
  const pick = (row, key) => window.EU_LANG === 'zh' && row[`${key}_zh`] || row[key] || '';
  // null until the host answers, so "loading" and "failed" never read as an empty catalogue.
  let catalog = null, failed = false, selected = '', pending = null, loadedAt = 0;
  const rows = () => (catalog && catalog.items) || [];
  const find = id => rows().find(row => row.id === id);
  const label = id => window.AGENT_READER_VOCAB?.tableTitle(id)
    || (/^table:/.test(id) ? tr('Analysis table', '分析结果表')
      : /^figure:/.test(id) ? tr('Analysis figure', '分析图') : id);
  const bulletList = values => `<ul>${values.filter(Boolean).map(value => `<li>${esc(value)}</li>`).join('')}</ul>`;
  const roles = () => ({ analysis_unit: tr('Study unit', '分析单位'), time_zero: tr('Time origin', '时间起点'),
    event_time: tr('Event time', '事件时间'), censor_time: tr('End of follow-up', '随访终点'),
    exposure: tr('Exposure', '暴露'), outcome: tr('Outcome', '结局'), observation_window: tr('Observation window', '观察窗'),
    followup: tr('Follow-up', '随访'),
    observation_timing: tr('Measurement times', '测量时间'), treatment_timing: tr('Treatment times', '治疗时间'),
    concept: tr('Variables', '变量') });
  const kinds = () => ({ categorical: tr('Categorical', '分类'), continuous: tr('Continuous', '连续'),
    binary: tr('Binary', '二分类'), time_to_event: tr('Time to event', '时间到事件') });
  // A workflow needs roles; which variables fill them comes from the researcher's
  // question, so example concepts are not listed as requirements.
  function inputLabels(inputs) {
    const labels = inputs.map(row => row.role === 'concept'
      ? tr('Study variables · from your question and data', '研究变量 · 由问题与数据决定')
      : [roles()[row.role] || row.role, kinds()[row.kind]].filter(Boolean).join(' · '));
    return [...new Set(labels)];
  }
  const nodeNames = () => ({ cohort_definition: tr('Cohort', '队列'), table_one: 'Table 1',
    measurement_audit: tr('Measurement audit', '测量核查'), adjusted_association: tr('Adjusted model', '调整后模型'),
    absolute_risk_context: tr('Absolute risk', '绝对风险'), robustness_replay: tr('Sensitivity', '敏感性分析'),
    visualization: tr('Figure', '图形'), report: tr('Report', '报告'),
    feature_functional_form: tr('Functional form', '函数形式检验'),
    primary_survival_suite: tr('Survival models', '生存分析模型'), candidate_selection: tr('Model selection', '候选模型选择'),
    cohort_accounting: tr('Cohort accounting', '纳排与队列'), baseline_context: tr('Baseline table', '基线描述'),
    time_varying_cox: tr('Time-varying Cox', '时变 Cox 模型'), target_trial_suite: tr('Clone-censor-weight', '克隆-删失-加权'),
    g_method_history: tr('Treatment and confounder histories', '治疗与混杂因素历史'),
    longitudinal_msm: tr('Marginal structural model', '边际结构模型'),
    landmark_features: tr('Landmark features', 'Landmark 特征'), horizon_labels: tr('Horizon outcomes', '预测时窗结局'),
    patient_split: tr('Patient split', '按患者划分'), model_fit: tr('Model fit', '模型拟合'),
    landmark_performance: tr('Performance by landmark', '各 landmark 表现') });
  // Fallback when a step has no short name: its typed kind, never the long step instruction.
  const kindNames = () => ({ cohort: tr('Cohort', '队列'), preflight: tr('Data check', '数据核查'),
    model: tr('Primary analysis', '主分析'), diagnostic: tr('Diagnostic', '诊断检验'),
    sensitivity: tr('Sensitivity', '敏感性分析'), figure: tr('Figure', '图形'), report: tr('Report', '报告') });
  function nodeLabel(node) {
    const names = nodeNames(), module = node.module_id === 'custom_analysis' ? '' : node.module_id;
    const named = names[node.id] || names[module || node.kind];
    if (named) return named;
    if (window.EU_LANG === 'zh' && node.label_zh) return node.label_zh;
    const vocab = window.AGENT_READER_VOCAB?.tableTitle(node.id);
    if (vocab && vocab !== node.id) return vocab;
    const label = node.label || '';
    return (window.EU_LANG === 'zh' || label.length > 40) && kindNames()[node.kind] || label || node.id;
  }
  // Graph geometry: columns are dependency depth; rows are separated by channels
  // that edges skipping a column travel along, so no edge crosses a step box.
  const BOX_W = 176, BOX_H = 56, COL = 236, ROW = 96, PAD = 36, GAP_X = COL - BOX_W;
  function reachesWithout(out, source, target) {
    const seen = new Set(), stack = out.get(source).filter(id => id !== target);
    while (stack.length) {
      const id = stack.pop();
      if (id === target) return true;
      if (seen.has(id)) continue;
      seen.add(id); stack.push(...out.get(id));
    }
    return false;
  }
  function layout(row) {
    const nodes = row.steps || [], ids = new Set(nodes.map(node => node.id));
    const all = (row.edges || []).filter(edge => ids.has(edge.source) && ids.has(edge.target) && edge.source !== edge.target);
    const out = new Map(nodes.map(node => [node.id, []]));
    all.forEach(edge => out.get(edge.source).push(edge.target));
    // An edge implied by a longer path adds no ordering; drawing it only adds crossings.
    const edges = all.filter(edge => !reachesWithout(out, edge.source, edge.target));
    const depths = new Map(), visiting = new Set();
    function depth(id) {
      if (depths.has(id)) return depths.get(id);
      if (visiting.has(id)) return 0;
      visiting.add(id);
      const parents = all.filter(edge => edge.target === id);
      const value = parents.length ? 1 + Math.max(...parents.map(edge => depth(edge.source))) : 0;
      visiting.delete(id); depths.set(id, value); return value;
    }
    const levels = [];
    nodes.forEach(node => { const level = depth(node.id); (levels[level] = levels[level] || []).push(node.id); });
    const index = new Map();
    levels.forEach(level => (level || []).forEach((id, i) => index.set(id, i)));
    for (let pass = 0; pass < 2; pass += 1) levels.forEach((level, l) => {
      if (!l || !level) return;
      const center = id => {
        const parents = edges.filter(edge => edge.target === id).map(edge => index.get(edge.source));
        return parents.length ? parents.reduce((a, b) => a + b, 0) / parents.length : index.get(id);
      };
      level.sort((a, b) => center(a) - center(b) || index.get(a) - index.get(b));
      level.forEach((id, i) => index.set(id, i));
    });
    const positions = new Map();
    levels.forEach((level, l) => (level || []).forEach((id, i) => positions.set(id, { x: PAD + l * COL, y: PAD + i * ROW })));
    const lanes = Math.max(1, ...levels.map(level => (level || []).length));
    return { nodes, edges, positions, levels: levels.length, lanes };
  }
  function ports(edges, positions, key, other) {
    const result = new Map();
    const groups = new Map();
    edges.forEach(edge => { const id = edge[key]; groups.set(id, [...(groups.get(id) || []), edge]); });
    groups.forEach((list, id) => {
      const box = positions.get(id), step = Math.min(10, (BOX_H - 16) / Math.max(1, list.length - 1));
      list.sort((a, b) => positions.get(a[other]).y - positions.get(b[other]).y || positions.get(a[other]).x - positions.get(b[other]).x)
        .forEach((edge, i) => result.set(edge, box.y + BOX_H / 2 + (i - (list.length - 1) / 2) * step));
    });
    return result;
  }
  // Rounded right-angle path through the given points (each corner is a bend).
  function orthogonal(points) {
    let d = `M${points[0][0]},${points[0][1]}`;
    for (let i = 1; i < points.length - 1; i += 1) {
      const [px, py] = points[i - 1], [x, y] = points[i], [nx, ny] = points[i + 1];
      const r = Math.min(10, Math.hypot(x - px, y - py) / 2, Math.hypot(nx - x, ny - y) / 2);
      const ux = Math.sign(x - px), uy = Math.sign(y - py), vx = Math.sign(nx - x), vy = Math.sign(ny - y);
      d += ` L${x - ux * r},${y - uy * r} Q${x},${y} ${x + vx * r},${y + vy * r}`;
    }
    const [lx, ly] = points[points.length - 1];
    return `${d} L${lx},${ly}`;
  }
  function edgePaths(model) {
    const { edges, positions } = model;
    const starts = ports(edges, positions, 'source', 'target'), ends = ports(edges, positions, 'target', 'source');
    // Edges that skip a column run along the channel above the row nearest both ends.
    const channelY = g => PAD + g * ROW - (ROW - BOX_H) / 2, used = new Map(), routes = new Map();
    edges.forEach(edge => {
      const sx = positions.get(edge.source).x + BOX_W, tx = positions.get(edge.target).x;
      const sy = starts.get(edge), ty = ends.get(edge);
      if (tx - sx <= GAP_X) { routes.set(edge, { verticals: [[sx, sy, ty]] }); return; }
      let best = 0;
      for (let g = 0; g <= model.lanes; g += 1) {
        if (Math.abs(channelY(g) - sy) + Math.abs(channelY(g) - ty) < Math.abs(channelY(best) - sy) + Math.abs(channelY(best) - ty)) best = g;
      }
      const n = used.get(best) || 0; used.set(best, n + 1);
      const cy = channelY(best) + (n % 2 ? 1 : -1) * Math.ceil(n / 2) * 6;
      routes.set(edge, { cy, verticals: [[sx, sy, cy], [tx - GAP_X, cy, ty]] });
    });
    // Every vertical in one column gap turns at its own x. Downward runs starting
    // lower turn first and upward runs starting higher turn first, so fans nest.
    const byGap = new Map();
    routes.forEach(route => route.verticals.forEach(vertical => byGap.set(vertical[0], [...(byGap.get(vertical[0]) || []), vertical])));
    byGap.forEach(list => {
      const rank = ([, from, to]) => to > from ? -from : to < from ? 1e6 + from : 5e5;
      list.sort((a, b) => rank(a) - rank(b))
        .forEach((vertical, i) => vertical.push(vertical[0] + GAP_X * (list.length > 1 ? 0.2 + 0.6 * i / (list.length - 1) : 0.5)));
    });
    return edges.map(edge => {
      const route = routes.get(edge), tx = positions.get(edge.target).x;
      const points = [[route.verticals[0][0], starts.get(edge)]];
      route.verticals.forEach(([, from, to, x]) => { points.push([x, from], [x, to]); });
      points.push([tx, ends.get(edge)]);
      const d = orthogonal(points.filter((point, i) => !i || point[0] !== points[i - 1][0] || point[1] !== points[i - 1][1]));
      return `<path data-wf-edge="${esc(edge.source)}:${esc(edge.target)}" d="${d}" class="${edge.optional ? 'is-optional' : ''}" marker-end="url(#wf-arrow)"><title>${esc((edge.products || []).map(label).join(', '))}</title></path>`;
    }).join('');
  }
  function graph(row) {
    const model = layout(row);
    const width = Math.max(360, PAD * 2 + (model.levels - 1) * COL + BOX_W);
    const height = PAD + model.lanes * ROW - (ROW - BOX_H) + PAD;
    const boxes = model.nodes.map(node => {
      const point = model.positions.get(node.id), text = nodeLabel(node);
      const chars = Array.from(text), limit = /[㐀-鿿]/.test(text) ? 12 : 22;
      const lines = [chars.slice(0, limit).join(''), chars.length > limit ? chars.slice(limit, limit * 2 - 1).join('') + (chars.length >= limit * 2 ? '…' : '') : ''].filter(Boolean);
      return `<g data-wf-node="${esc(node.id)}" transform="translate(${point.x},${point.y})" class="${node.optional ? 'is-optional' : ''}" tabindex="0"><title>${esc(text)}</title><rect width="${BOX_W}" height="${BOX_H}" rx="7"/><text x="12" y="${lines.length > 1 ? 23 : 33}">${lines.map((line, i) => `<tspan x="12" dy="${i ? 18 : 0}">${esc(line)}</tspan>`).join('')}</text>${node.optional ? `<text class="wf-optional" x="${BOX_W - 8}" y="14" text-anchor="end">${tr('Optional', '可选')}</text>` : ''}</g>`;
    }).join('');
    return `<div class="eusk-workflow-graph" tabindex="0" role="region" aria-label="${tr('Workflow graph', '流程图')}"><svg width="${width}" height="${height}" viewBox="0 0 ${width} ${height}" role="img" aria-label="${esc(pick(row, 'title'))}"><defs><marker id="wf-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10z"/></marker></defs>${edgePaths(model)}${boxes}</svg></div>`;
  }
  function databaseReasonCodes(row) {
    const decision = row.decision || {};
    return [...(decision.substitutions || []).map(item => item.reason_code),
      ...(decision.missing || []).map(item => item.code), ...(decision.unassessed_concepts || [])].filter(Boolean);
  }
  function databaseReason(row) {
    const codes = databaseReasonCodes(row);
    if (codes.includes('numeric_threshold_aligned_to_band')) return tr('Values are recorded in bands; thresholds must align with those bands.', '数值按区间记录，纳入阈值需与区间边界对齐。');
    if (codes.includes('mortality_meaning_differs')) return tr('Mortality definitions differ; confirm the outcome and follow-up in the study plan.', '死亡结局定义存在差异，需在研究计划中确认结局与随访。');
    if (codes.length) return tr('Some required fields or definitions need review in the study plan.', '部分所需字段或定义需要在研究计划中核对。');
    return '';
  }
  function outputLabel(item) {
    const scoped = {
      'table:table_one': tr('Baseline table · study cohort', '基线特征表（研究队列）'),
      'table:landmark_table_one': tr('Baseline table · landmark risk set', '基线特征表（Landmark 风险集）'),
      'table:cohort_flow': tr('Cohort exclusions · table', '队列纳排明细表'),
      'figure:cohort_flow': tr('Cohort flow · figure', '队列纳排流程图'),
    };
    if (Object.prototype.hasOwnProperty.call(scoped, item.id)) return scoped[item.id];
    const text = label(item.id), kind = String(item.id).split(':', 1)[0];
    const formats = { table: tr('table', '表'), figure: tr('figure', '图'),
      artifact: tr('data file', '数据文件'), manifest: tr('manifest', '清单') };
    return formats[kind] ? `${text}${tr(` · ${formats[kind]}`, `（${formats[kind]}）`)}` : text;
  }
  function outputGroups(row) {
    const groups = { main: [], design: [], checks: [] }, seen = new Set();
    const producers = new Map((row.steps || []).map(step => [step.id, step]));
    for (const item of row.outputs || []) {
      // Different risk sets can share a translated label. Only identical
      // product IDs are duplicates; retain and name their distinct scopes.
      if (!item.id || seen.has(item.id)) continue;
      seen.add(item.id);
      const producer = producers.get(item.producer) || {};
      const kind = item.producer_kind || producer.kind;
      const optional = item.optional === true || producer.optional === true;
      const design = optional || kind === 'sensitivity' || /(?:sensitivity|robustness|rmst)/.test(item.id);
      const checks = /^(?:log|manifest):/.test(item.id)
        || /(?:audit|diagnostic|measurement_|receipt|checklist)/.test(item.id);
      groups[design ? 'design' : checks ? 'checks' : 'main'].push(outputLabel(item));
    }
    const section = (title, values) => values.length ? `<h3>${title}</h3>${bulletList(values)}` : '';
    return section(tr('Main example products', '示例主要产物'), groups.main)
      + section(tr('Design-specific and sensitivity results', '设计相关与敏感性结果'), groups.design)
      + (groups.checks.length ? `<details class="eusk-workflow-checks"><summary>${tr('Checks and execution records', '核查与运行记录')}</summary>${bulletList(groups.checks)}</details>` : '');
  }
  function illustrationNote(row) {
    const landmark = row.id === 'survival-time-to-event'
      && (row.example_coordinates?.landmark_hours != null
        || (row.outputs || []).some(item => /^table:landmark_/.test(item.id)));
    return `<p class="eusk-workflow-note">${landmark
      ? tr('This illustration uses a fixed-landmark survival design. Whether to use a landmark and how to define time zero depend on the research question and data.', '当前展示固定 Landmark 生存研究示例。是否采用 Landmark、如何定义时间起点，由研究问题与数据条件决定。')
      : tr('This is an example structure. Variables, time windows, branches and deliverables are determined in the complete study plan.', '这是研究结构示例。具体变量、时间窗、分支和交付物由完整研究计划确定。')}</p>`;
  }
  function databaseAcceptance(row) {
    return row.real_data_acceptance === 'not_verified'
      ? tr('Real-data study acceptance not verified', '真实研究未验收')
      : tr('Real-data acceptance requires a receipt', '真实验收记录待核对');
  }
  function sections(row) {
    if (!row) return '';
    const statuses = { supported: tr('Field conditions supported', '字段条件支持'), degraded: tr('Definitions need review', '定义需核对'), unavailable: tr('Required conditions missing', '所需条件不足') };
    const inputs = row.inputs || [], family = inputs.filter(item => item.scope !== 'canonical_example');
    const example = inputs.filter(item => item.scope === 'canonical_example');
    const familyLabels = inputLabels(family), exampleLabels = inputLabels(example).filter(value => !familyLabels.includes(value));
    return `${illustrationNote(row)}<div class="eusk-workflow-products"><section><h2>${tr('Required inputs', '所需输入')}</h2><p class="eusk-workflow-note">${tr('Describe your question and data in conversation. The plan checks these requirements.', '在对话中说明研究问题和数据，计划会核对这些条件。')}</p>${bulletList(familyLabels)}${exampleLabels.length ? `<details><summary>${tr('Additional inputs in this example', '此示例的补充条件')}</summary>${bulletList(exampleLabels)}</details>` : ''}</section><section><h2>${tr('Example outputs', '示例输出')}</h2><p class="eusk-workflow-note">${tr('The approved plan determines which products are required.', '实际需要的产物以审阅后的完整计划为准。')}</p>${outputGroups(row)}</section></div>
      <section><h2>${tr('Example dependency graph', '示例依赖图')}</h2><p class="eusk-workflow-note">${tr('Lines summarize prerequisite and product dependencies in this example, not the execution status of your study.', '连线概括示例中的前置条件与产物依赖，不代表你的研究执行状态。')}${(row.steps || []).some(step => step.optional) ? ` ${tr('Dashed branches are selected by the study plan.', '虚线分支按研究计划选择。')}` : ''}</p>${graph(row)}<ol class="eusk-workflow-gates"><li>${tr('Check data conditions', '核对数据条件')}</li><li>${tr('Review the complete plan', '审阅完整方案')}</li><li>${tr('Run analyses and diagnostics', '执行分析与诊断')}</li><li>${tr('Verify results and deliver', '核验结果后交付')}</li></ol><p class="eusk-workflow-note">${tr('Unresolved data or diagnostic checks must be addressed in the plan before drawing conclusions.', '数据条件或诊断检查未通过时，需要先处理问题或在方案中明确限制，再作结论。')}</p></section>
      <section><h2>${tr('Database conditions and acceptance', '数据库条件与验收')}</h2><p class="eusk-workflow-note">${tr('Database-level field support is separate from real-data study acceptance. Your project data is checked in its plan.', '数据库级字段支持与真实研究验收分别列出，当前项目的数据仍需在计划中核对。')}</p><div class="eusk-workflow-databases">${(row.databases || []).map(db => {
        const reason = databaseReason(db) || statuses[db.status], id = `wf-reason-${row.id}-${db.id}`;
        return `<span class="eusk-workflow-database ${esc(db.status)}" tabindex="0" title="${esc(reason)}" aria-describedby="${esc(id)}"><strong>${esc(db.label)}</strong> ${esc(statuses[db.status] || tr('Conditions require review', '条件待核对'))}<small>${esc(databaseAcceptance(db))}</small><span class="eusk-workflow-reason" role="tooltip" id="${esc(id)}">${esc(reason)}</span></span>`;
      }).join('')}</div></section>`;
  }
  function questionEntry() {
    return `<section class="eusk-workflow-entry"><div><h2>${tr('Start with your research question', '先描述研究问题')}</h2><p>${tr('Tell EasyICU what you want to study, where the data is, and what you need. It will recommend a design; the library below is available for comparison and adjustment.', '说明想研究什么、数据在哪里、希望得到什么。EasyICU 再推荐研究设计，下方流程库供你比较和调整。')}</p></div><button type="button" class="eusk-use" data-wf-question>${tr('Describe my question', '描述研究问题')}</button></section>`;
  }
  function render(payload, selectedId, skills = []) {
    const entry = questionEntry();
    // Before the host answers there is nothing to call empty yet.
    if (!payload && failed) return `${entry}<div class="eusk-empty" role="alert"><strong>${tr('Research workflows could not be loaded', '研究流程读取失败')}</strong><p>${tr('The local service did not answer. Try again.', '本机服务没有响应，请重试。')}</p><button type="button" data-wf-retry>${tr('Retry', '重试')}</button></div>`;
    if (!payload) return `${entry}<div class="eusk-empty" role="status">${tr('Loading research workflows…', '正在读取研究流程…')}</div>`;
    const items = payload.items || [], custom = window.EU_SKILL_CUSTOM_WORKFLOW;
    const customActive = !!custom && (selectedId || selected) === custom.ID;
    const row = customActive ? null : items.find(item => item.id === (selectedId || selected)) || items[0];
    if (!row && !customActive) return `${entry}<div class="eusk-empty" role="status">${tr('No workflows available', '暂无研究流程')}</div>`;
    const nav = `<nav class="eusk-workflow-list" aria-label="${tr('Research workflows', '研究流程')}">${items.map(item => `<button type="button" data-wf-select="${esc(item.id)}" aria-pressed="${!customActive && item.id === row?.id}"><strong>${esc(pick(item, 'title'))}</strong><span>${esc(pick(item, 'description'))}</span></button>`).join('')}${custom ? custom.navItem(customActive) : ''}</nav>`;
    if (customActive) return `${entry}<div class="eusk-workflows">${nav}${custom.render(skills, graph)}</div>`;
    return `${entry}<div class="eusk-workflows">${nav}
      <article class="eusk-workflow-detail"><header><h2>${esc(pick(row, 'title'))}</h2><p>${esc(pick(row, 'description'))}</p><details><summary>${tr('Technical details', '技术详情')}</summary><small>${esc(row.version)}</small>${bulletList((row.databases || []).map(db => { const codes = databaseReasonCodes(db); return codes.length ? `${db.label}: ${codes.join(', ')}` : ''; }))}</details></header>
      <div class="eusk-workflow-actions"><button type="button" class="eusk-use" data-wf-start="${esc(row.id)}" ${row.enabled ? '' : 'disabled'}>${tr('Start with this workflow', '用此流程开始')}</button><button type="button" data-sk-open="builtin:${esc(row.id)}">${tr('Docs', '文档')}</button></div>${sections(row)}</article></div>`;
  }
  async function load() {
    if (Date.now() - loadedAt < 10000) return catalog;
    if (!pending && window.EU_API?.getJSON) pending = window.EU_API.getJSON('/api/capabilities/research-workflows')
      .then(payload => { catalog = payload; failed = false; loadedAt = Date.now(); return catalog; })
      .catch(error => { if (!catalog) failed = true; throw error; }).finally(() => { pending = null; });
    return pending;
  }
  // The request names the workflow and asks for the study's own inputs; the
  // researcher edits or sends it, and nothing runs before the plan is approved.
  function startRequest(skill, row) {
    const title = pick(row || {}, 'title') || skill.title || skill.name;
    return tr(`Use the "${title}" workflow as a starting point. Assess its fit to my question and data, explain the recommended design, and ask only for missing information that affects that design. Put the cohort, timing, outcome, methods and deliverables in one complete plan for my review; do not run analyses before that review. My research question:`,
      `请以「${title}」流程为研究起点，根据我的问题和数据判断是否适用，说明推荐的研究设计，只补问会影响设计的缺失信息。把人群、时间、结局、方法和交付要求放进一份完整计划供我审阅，审阅前不运行分析。我的研究问题是：`);
  }
  function startQuestion() {
    window.sessionStorage.removeItem('easyicu.skillHub.method');
    window.sessionStorage.setItem('easyicu.skillHub.question', tr(
      'Recommend a study design for my research question and available data, explain why, and ask only for missing information that affects the design. Prepare one complete plan for my review before running analyses. My research question:',
      '请根据我的研究问题和可用数据推荐合适的研究设计，说明理由，只补问会影响设计的缺失信息，并给出一份完整计划供我审阅，审阅前不运行分析。我的研究问题是：',
    ));
    location.hash = '#guided';
  }
  function start(skill, row) {
    window.sessionStorage.setItem('easyicu.skillHub.method', JSON.stringify({
      id: skill.name || skill.id, title: skill.title, kind: skill.kind || 'method',
      layer: skill.skillLayer || 'research_workflow', capability_id: skill.capabilityId || '',
      action_ids: skill.actionIds || [], included_module_ids: skill.includedModuleIds || [],
      method_family: skill.methodFamily || '', method_key: skill.methodKey || '', claim_ceiling: skill.claimCeiling,
    }));
    window.sessionStorage.setItem('easyicu.skillHub.question', startRequest(skill, row));
    location.hash = '#guided';
  }
  function handle(target, skills) {
    if (target.hasAttribute && target.hasAttribute('data-wf-question')) { startQuestion(); return true; }
    if (target.hasAttribute && target.hasAttribute('data-wf-retry')) {
      failed = false; loadedAt = 0;
      load().catch(() => {}).finally(() => { if (window.__euRender) window.__euRender(); });
      return true;
    }
    if (target.dataset.wfSelect) { selected = target.dataset.wfSelect; return true; }
    if (window.EU_SKILL_CUSTOM_WORKFLOW?.handle(target, skills)) return true;
    if (!target.dataset.wfStart) return false;
    const row = find(target.dataset.wfStart), skill = skills.find(item => item.name === row?.id);
    if (row?.enabled && skill) start(skill, row);
    return true;
  }
  // These three sections are replaced by structured owner data for workflows.
  // Components without those specifics omit the old repeated generic text.
  function overviewMarkdown(content) {
    return String(content || '').replace(/^## (?:Outputs|输出|Steps and their checks|步骤与检查|Per-database notes|各数据库说明)\s*\n[\s\S]*?(?=^## |$(?![\s\S]))/gm, '');
  }
  if (typeof document !== 'undefined') document.addEventListener('click', event => {
    const button = event.target.closest('[data-sk-mode]');
    if (button) window.SCREENS?.skills?.selectTab(button.dataset.skMode);
  }, true);
  window.EU_SKILL_WORKFLOWS = Object.freeze({ render, sections, graph, start, startQuestion, handle, load, find,
    overviewMarkdown, payload: () => catalog });
})();
