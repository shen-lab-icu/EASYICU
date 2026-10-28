/* Custom workflow builder on the Skills research-workflow tab.
   The researcher picks analysis modules in order; the composition becomes a
   conversation request. The planner owns the real plan and its checks, and
   nothing runs before that plan is approved. */
(function () {
  'use strict';
  const tr = (en, zh) => (window.EU_LANG === 'zh' ? zh : en);
  const esc = value => String(value == null ? '' : value).replace(/[&<>"']/g, ch => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[ch]));
  const ID = '__custom';
  const KEY = 'easyicu.skillHub.customWorkflow';
  let chosen = [];
  let query = '';
  try { chosen = JSON.parse(window.sessionStorage.getItem(KEY) || '[]').filter(item => typeof item === 'string'); } catch (_) { chosen = []; }
  const save = () => { try { window.sessionStorage.setItem(KEY, JSON.stringify(chosen)); } catch (_) {} };

  const modules = skills => skills.filter(item => item.kind === 'method_component' && item.enabled);
  const byName = (skills, name) => modules(skills).find(item => item.name === name);
  const matches = item => !query || `${item.title} ${item.category} ${item.description}`.toLowerCase().includes(query);

  // The host always compiles the cohort first and the report last.
  function previewRow(skills) {
    const picked = chosen.map(name => byName(skills, name)).filter(Boolean);
    const steps = [
      { id: 'cohort_definition', kind: 'cohort', module_id: 'cohort_definition', label: 'Cohort' },
      ...picked.map((item, index) => ({ id: `custom_${index}`, kind: 'model', module_id: '', label: item.title, label_zh: item.title })),
      { id: 'report', kind: 'report', module_id: 'report', label: 'Report' },
    ];
    const edges = steps.slice(1).map((step, index) => ({ source: steps[index].id, target: step.id, products: [] }));
    return { id: ID, title: tr('Custom workflow', '自定义流程'), title_zh: '自定义流程', steps, edges };
  }

  function navItem(active) {
    return `<button type="button" class="eusk-workflow-custom" data-wf-select="${ID}" aria-pressed="${active}"><strong>${tr('Custom workflow', '自定义流程')}</strong><span>${tr('Combine analysis modules in your own order.', '按你的顺序组合分析模块。')}</span></button>`;
  }

  function render(skills, graph) {
    const all = modules(skills);
    const groups = new Map();
    all.forEach(item => { if (!groups.has(item.category)) groups.set(item.category, []); groups.get(item.category).push(item); });
    const list = [...groups].map(([category, items]) => `<section data-wfc-group><h3>${esc(category)}</h3>${items.map(item => {
      const added = chosen.includes(item.name);
      return `<div class="eusk-wfc-module" data-wfc-row="${esc(item.name)}" ${matches(item) ? '' : 'hidden'}><div><strong>${esc(item.title)}</strong><span>${esc(item.description)}</span></div><button type="button" data-wfc-add="${esc(item.name)}" ${added ? 'disabled' : ''}>${added ? tr('Added', '已加入') : tr('Add', '加入')}</button></div>`;
    }).join('')}</section>`).join('');
    const picked = chosen.map(name => byName(skills, name)).filter(Boolean);
    const order = picked.length ? `<ol class="eusk-wfc-order">${picked.map((item, index) => `<li><span>${esc(item.title)}</span><button type="button" data-wfc-move="${index}:-1" aria-label="${tr('Move up', '上移')}" ${index ? '' : 'disabled'}>↑</button><button type="button" data-wfc-move="${index}:1" aria-label="${tr('Move down', '下移')}" ${index < picked.length - 1 ? '' : 'disabled'}>↓</button><button type="button" data-wfc-remove="${index}" aria-label="${tr('Remove', '移除')}">×</button></li>`).join('')}</ol>`
      : `<p class="eusk-wfc-empty">${tr('Add modules from the list below.', '从下方列表加入模块。')}</p>`;
    return `<article class="eusk-workflow-detail eusk-wfc"><header><h2>${tr('Custom workflow', '自定义流程')}</h2><p>${tr('Pick the analysis steps you want and their order. EasyICU defines the cohort first and writes the report last; when you start, it asks for your question and data, checks each step against them, and shows the full plan before anything runs.', '选择你要的分析步骤和顺序。EasyICU 会先定义队列、最后写报告；开始后会先问你的研究问题和数据，逐步核对每一步是否适用，并在运行前给出完整计划。')}</p></header>
      <div class="eusk-workflow-actions"><button type="button" class="eusk-use" data-wfc-start ${picked.length ? '' : 'disabled'}>${tr('Start with this combination', '用此组合开始')}</button>${picked.length ? `<button type="button" data-wfc-clear>${tr('Clear', '清空')}</button>` : ''}</div>
      <section><h2>${tr('Selected steps', '已选步骤')}</h2>${order}</section>
      <section><h2>${tr('Workflow graph', '流程图')}</h2>${graph(previewRow(skills))}</section>
      <section><h2>${tr('Modules', '模块')}</h2><input type="search" class="eusk-wfc-search" data-wfc-search value="${esc(query)}" placeholder="${tr('Search modules', '搜索模块')}" aria-label="${tr('Search modules', '搜索模块')}"><div class="eusk-wfc-modules">${list}</div></section></article>`;
  }

  function startRequest(skills) {
    const names = chosen.map(name => byName(skills, name)).filter(Boolean).map(item => item.title);
    const steps = names.map((name, index) => `${index + 1}. ${name}`);
    return tr(`Help me run a study with this custom workflow: define the cohort, then ${steps.join('; ')}; then write the report. Ask me for the research question, data source and analysis goal first. If a step does not fit the question, explain why and propose an alternative, and confirm the configuration before generating the plan.`,
      `请按这个自定义流程帮我开展一项研究：先定义队列，然后依次做 ${steps.join('；')}，最后写报告。先问我研究问题、数据来源和分析目标；如果某一步不适合这个问题，请说明原因并给出替代，确认配置后再生成研究计划。`);
  }

  // Returns true when the click belonged to the builder (the caller rerenders).
  function handle(target, skills) {
    if (target.dataset.wfcAdd) { if (!chosen.includes(target.dataset.wfcAdd)) chosen.push(target.dataset.wfcAdd); save(); return true; }
    if (target.dataset.wfcRemove) { chosen.splice(Number(target.dataset.wfcRemove), 1); save(); return true; }
    if (target.dataset.wfcMove) {
      const [index, step] = target.dataset.wfcMove.split(':').map(Number), next = index + step;
      if (next >= 0 && next < chosen.length) [chosen[index], chosen[next]] = [chosen[next], chosen[index]];
      save(); return true;
    }
    if (target.hasAttribute('data-wfc-clear')) { chosen = []; save(); return true; }
    if (target.hasAttribute('data-wfc-start')) {
      if (!chosen.length) return true;
      window.sessionStorage.setItem('easyicu.skillHub.question', startRequest(skills));
      location.hash = '#guided';
      return true;
    }
    return false;
  }

  // Filter in place so typing keeps the caret in the search box.
  if (typeof document !== 'undefined') document.addEventListener('input', event => {
    const input = event.target.closest && event.target.closest('[data-wfc-search]');
    if (!input) return;
    query = input.value.trim().toLowerCase();
    const root = input.closest('.eusk-wfc');
    root.querySelectorAll('[data-wfc-row]').forEach(row => { row.hidden = !!query && !row.textContent.toLowerCase().includes(query); });
    root.querySelectorAll('[data-wfc-group]').forEach(group => { group.hidden = !group.querySelector('[data-wfc-row]:not([hidden])'); });
  });

  window.EU_SKILL_CUSTOM_WORKFLOW = Object.freeze({ ID, navItem, render, handle, startRequest, select: () => ID });
})();
