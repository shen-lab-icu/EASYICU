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
  const label = id => window.AGENT_READER_VOCAB?.tableTitle(id) || id;
  const bulletList = values => `<ul>${values.filter(Boolean).map(value => `<li>${esc(value)}</li>`).join('')}</ul>`;
  const roles = () => ({ analysis_unit: tr('Study unit', '分析单位'), time_zero: tr('Time origin', '时间起点'),
    event_time: tr('Event time', '事件时间'), censor_time: tr('End of follow-up', '随访终点'),
    exposure: tr('Exposure', '暴露'), outcome: tr('Outcome', '结局'), observation_window: tr('Observation window', '观察窗'),
    followup: tr('Follow-up', '随访'),
    observation_timing: tr('Measurement times', '测量时间'), treatment_timing: tr('Treatment times', '治疗时间'),
    concept: tr('Variables', '变量') });
  const kinds = () => ({ categorical: tr('Categorical', '分类'), continuous: tr('Continuous', '连续'),
    binary: tr('Binary', '二分类') });
  // A workflow needs roles; which variables fill them comes from the researcher's
  // question, so example concepts are not listed as requirements.
  function inputLabels(inputs) {
    const labels = inputs.map(row => row.role === 'concept'
      ? tr('Adjustment variables · from your question', '调整变量 · 由研究问题决定')
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
      return `<path data-wf-edge="${esc(edge.source)}:${esc(edge.target)}" d="${d}" marker-end="url(#wf-arrow)"><title>${esc((edge.products || []).map(label).join(', '))}</title></path>`;
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
  function databaseReason(row) {
    const decision = row.decision || {};
    return [...(decision.substitutions || []).map(item => item.reason_code),
      ...(decision.missing || []).map(item => item.code), ...(decision.unassessed_concepts || [])].join(' · ');
  }
  function sections(row) {
    if (!row) return '';
    const statuses = { supported: tr('Supported', '支持'), degraded: tr('Limited', '受限'), unavailable: tr('Unavailable', '不可用') };
    return `<div class="eusk-workflow-products"><section><h2>${tr('Required inputs', '所需输入')}</h2>${bulletList(inputLabels(row.inputs || []))}</section><section><h2>${tr('Generated outputs', '生成的输出')}</h2>${bulletList((row.outputs || []).map(item => label(item.id)))}</section></div>
      <section><h2>${tr('Workflow graph', '流程图')}</h2>${graph(row)}</section>
      <section><h2>${tr('Databases', '各数据库')}</h2><div class="eusk-workflow-databases">${(row.databases || []).map(db => {
        const reason = databaseReason(db) || statuses[db.status], id = `wf-reason-${row.id}-${db.id}`;
        return `<span class="eusk-workflow-database ${esc(db.status)}" tabindex="0" title="${esc(reason)}" aria-describedby="${esc(id)}"><strong>${esc(db.label)}</strong> ${esc(statuses[db.status])}<span class="eusk-workflow-reason" role="tooltip" id="${esc(id)}">${esc(reason)}</span></span>`;
      }).join('')}</div></section>`;
  }
  function render(payload, selectedId, skills = []) {
    // Before the host answers there is nothing to call empty yet.
    if (!payload && failed) return `<div class="eusk-empty" role="alert"><strong>${tr('Research workflows could not be loaded', '研究流程读取失败')}</strong><p>${tr('The local service did not answer. Try again.', '本机服务没有响应，请重试。')}</p><button type="button" data-wf-retry>${tr('Retry', '重试')}</button></div>`;
    if (!payload) return `<div class="eusk-empty" role="status">${tr('Loading research workflows…', '正在读取研究流程…')}</div>`;
    const items = payload.items || [], custom = window.EU_SKILL_CUSTOM_WORKFLOW;
    const customActive = !!custom && (selectedId || selected) === custom.ID;
    const row = customActive ? null : items.find(item => item.id === (selectedId || selected)) || items[0];
    if (!row && !customActive) return `<div class="eusk-empty" role="status">${tr('No workflows available', '暂无研究流程')}</div>`;
    const nav = `<nav class="eusk-workflow-list" aria-label="${tr('Research workflows', '研究流程')}">${custom ? custom.navItem(customActive) : ''}${items.map(item => `<button type="button" data-wf-select="${esc(item.id)}" aria-pressed="${!customActive && item.id === row.id}"><strong>${esc(pick(item, 'title'))}</strong><span>${esc(pick(item, 'description'))}</span></button>`).join('')}</nav>`;
    if (customActive) return `<div class="eusk-workflows">${nav}${custom.render(skills, graph)}</div>`;
    return `<div class="eusk-workflows">${nav}
      <article class="eusk-workflow-detail"><header><h2>${esc(pick(row, 'title'))}</h2><p>${esc(pick(row, 'description'))}</p><small>${esc(row.version)}</small></header>
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
    return tr(`Help me run a study with the "${title}" workflow. Follow the workflow guide, ask me for the research question, data source and analysis goal, and confirm the configuration before generating the plan.`,
      `请用「${title}」流程帮我开展一项研究：按流程指南先问我研究问题、数据来源和分析目标，确认配置后再生成研究计划。`);
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
  window.EU_SKILL_WORKFLOWS = Object.freeze({ render, sections, graph, start, handle, load, find,
    overviewMarkdown, payload: () => catalog });
})();
