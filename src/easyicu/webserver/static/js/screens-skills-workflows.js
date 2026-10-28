/* Owner: read-only research workflow list, graph and hub-start intent. */
(function () {
  'use strict';
  const { esc } = window.EU_HTML;
  const tr = (en, zh) => window.t(en, zh);
  const pick = (row, key) => window.EU_LANG === 'zh' && row[`${key}_zh`] || row[key] || '';
  let catalog = { items: [] }, selected = '', pending = null, loadedAt = 0;
  const rows = () => catalog.items || [];
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
  function inputLabel(row) {
    const concept = (window.EU_CATALOG?.dict || {})[row.concept];
    const term = pick(row, 'label') || (concept ? tr(concept[0], concept[1]) : row.concept);
    return [roles()[row.role] || row.role, term || kinds()[row.kind]].filter(Boolean).join(' · ');
  }
  const nodeNames = () => ({ cohort_definition: tr('Cohort', '队列'), table_one: 'Table 1',
    measurement_audit: tr('Measurement audit', '测量核查'), adjusted_association: tr('Adjusted model', '调整后模型'),
    absolute_risk_context: tr('Absolute risk', '绝对风险'), robustness_replay: tr('Sensitivity', '敏感性分析'),
    visualization: tr('Figure', '图形'), report: tr('Report', '报告') });
  function nodeLabel(node) {
    const named = nodeNames()[node.module_id || node.kind];
    return named || window.AGENT_READER_VOCAB?.tableTitle(node.id) || pick(node, 'label') || node.id;
  }
  function graph(row) {
    const nodes = row.steps || [], edges = row.edges || [];
    const ids = new Set(nodes.map(node => node.id));
    const depths = new Map(), visiting = new Set();
    function depth(id) {
      if (depths.has(id)) return depths.get(id);
      if (visiting.has(id)) return 0;
      visiting.add(id);
      const parents = edges.filter(edge => edge.target === id && ids.has(edge.source));
      const value = parents.length ? 1 + Math.max(...parents.map(edge => depth(edge.source))) : 0;
      visiting.delete(id); depths.set(id, value); return value;
    }
    const lanes = new Map(), positions = new Map();
    for (const node of nodes) {
      const level = depth(node.id), lane = lanes.get(level) || 0;
      lanes.set(level, lane + 1); positions.set(node.id, { x: 20 + level * 220, y: 20 + lane * 94 });
    }
    const width = Math.max(360, 40 + lanes.size * 220 - 30);
    const height = 40 + Math.max(1, ...lanes.values()) * 94;
    const links = edges.filter(edge => positions.has(edge.source) && positions.has(edge.target)).map(edge => {
      const from = positions.get(edge.source), to = positions.get(edge.target);
      const x = from.x + 180, y = from.y + 30, endY = to.y + 30, mid = (x + to.x) / 2;
      return `<path data-wf-edge="${esc(edge.source)}:${esc(edge.target)}" d="M${x},${y} C${mid},${y} ${mid},${endY} ${to.x},${endY}" marker-end="url(#wf-arrow)"><title>${esc((edge.products || []).map(label).join(', '))}</title></path>`;
    }).join('');
    const boxes = nodes.map(node => {
      const point = positions.get(node.id), text = nodeLabel(node);
      const chars = Array.from(text), limit = /[\u3400-\u9fff]/.test(text) ? 12 : 22;
      const lines = [chars.slice(0, limit).join(''), chars.length > limit ? chars.slice(limit, limit * 2 - 1).join('') + (chars.length >= limit * 2 ? '…' : '') : ''].filter(Boolean);
      return `<g data-wf-node="${esc(node.id)}" transform="translate(${point.x},${point.y})" class="${node.optional ? 'is-optional' : ''}" tabindex="0"><title>${esc(text)}</title><rect width="180" height="60" rx="7"/><text x="12" y="${lines.length > 1 ? 23 : 34}">${lines.map((line, i) => `<tspan x="12" dy="${i ? 18 : 0}">${esc(line)}</tspan>`).join('')}</text>${node.optional ? `<text class="wf-optional" x="12" y="78">${tr('Optional', '可选')}</text>` : ''}</g>`;
    }).join('');
    return `<div class="eusk-workflow-graph" tabindex="0" role="region" aria-label="${tr('Workflow graph', '流程图')}"><svg width="${width}" height="${height}" viewBox="0 0 ${width} ${height}" role="img" aria-label="${esc(pick(row, 'title'))}"><defs><marker id="wf-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10z"/></marker></defs>${links}${boxes}</svg></div>`;
  }
  function databaseReason(row) {
    const decision = row.decision || {};
    return [...(decision.substitutions || []).map(item => item.reason_code),
      ...(decision.missing || []).map(item => item.code), ...(decision.unassessed_concepts || [])].join(' · ');
  }
  function sections(row) {
    if (!row) return '';
    const statuses = { supported: tr('Supported', '支持'), degraded: tr('Limited', '受限'), unavailable: tr('Unavailable', '不可用') };
    return `<div class="eusk-workflow-products"><section><h2>${tr('Required inputs', '所需输入')}</h2>${bulletList((row.inputs || []).map(inputLabel))}</section><section><h2>${tr('Generated outputs', '生成的输出')}</h2>${bulletList((row.outputs || []).map(item => label(item.id)))}</section></div>
      <section><h2>${tr('Workflow graph', '流程图')}</h2>${graph(row)}</section>
      <section><h2>${tr('Databases', '各数据库')}</h2><div class="eusk-workflow-databases">${(row.databases || []).map(db => {
        const reason = databaseReason(db) || statuses[db.status], id = `wf-reason-${row.id}-${db.id}`;
        return `<span class="eusk-workflow-database ${esc(db.status)}" tabindex="0" title="${esc(reason)}" aria-describedby="${esc(id)}"><strong>${esc(db.label)}</strong> ${esc(statuses[db.status])}<span class="eusk-workflow-reason" role="tooltip" id="${esc(id)}">${esc(reason)}</span></span>`;
      }).join('')}</div></section>`;
  }
  function render(payload, selectedId) {
    // Before the host answers there is nothing to call empty yet.
    if (!payload) return `<div class="eusk-empty" role="status">${tr('Loading research workflows…', '正在读取研究流程…')}</div>`;
    const items = payload.items || [];
    const row = items.find(item => item.id === (selectedId || selected)) || items[0];
    if (!row) return `<div class="eusk-empty" role="status">${tr('No workflows available', '暂无研究流程')}</div>`;
    return `<div class="eusk-workflows"><nav class="eusk-workflow-list" aria-label="${tr('Research workflows', '研究流程')}">${items.map(item => `<button type="button" data-wf-select="${esc(item.id)}" aria-pressed="${item.id === row.id}"><strong>${esc(pick(item, 'title'))}</strong><span>${esc(pick(item, 'description'))}</span></button>`).join('')}</nav>
      <article class="eusk-workflow-detail"><header><h2>${esc(pick(row, 'title'))}</h2><p>${esc(pick(row, 'description'))}</p><small>${esc(row.version)}</small></header>
      <div class="eusk-workflow-actions"><button type="button" class="eusk-use" data-wf-start="${esc(row.id)}" ${row.enabled ? '' : 'disabled'}>${tr('Start with this workflow', '用此流程开始')}</button><button type="button" data-sk-open="builtin:${esc(row.id)}">${tr('Docs', '文档')}</button></div>${sections(row)}</article></div>`;
  }
  async function load() {
    if (Date.now() - loadedAt < 10000) return catalog;
    if (!pending && window.EU_API?.getJSON) pending = window.EU_API.getJSON('/api/capabilities/research-workflows')
      .then(payload => { catalog = payload; loadedAt = Date.now(); return catalog; }).finally(() => { pending = null; });
    return pending;
  }
  function start(skill, row) {
    const starter = window.EasyICU?.guidedPi?.optional('starters');
    const example = row?.example && starter?.exampleQuestion(row.example, tr);
    window.sessionStorage.setItem('easyicu.skillHub.method', JSON.stringify({
      id: skill.name || skill.id, title: skill.title, kind: skill.kind || 'method',
      layer: skill.skillLayer || 'research_workflow', capability_id: skill.capabilityId || '',
      action_ids: skill.actionIds || [], included_module_ids: skill.includedModuleIds || [],
      method_family: skill.methodFamily || '', method_key: skill.methodKey || '', claim_ceiling: skill.claimCeiling,
    }));
    window.sessionStorage.setItem('easyicu.skillHub.question', example?.prompt
      || (skill.skillLayer !== 'research_workflow' && skill.prompt) || tr('Help me plan this study.', '请帮我规划这项研究。'));
    location.hash = '#guided';
  }
  function handle(target, skills) {
    if (target.dataset.wfSelect) { selected = target.dataset.wfSelect; return true; }
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
