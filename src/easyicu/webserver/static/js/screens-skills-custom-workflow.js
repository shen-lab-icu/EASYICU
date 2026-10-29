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

  // This is a proposed composition. Input requirements and real ordering are
  // reviewed in the study plan; some modules use aggregates rather than patients.
  function previewRow(skills) {
    const picked = chosen.map(name => byName(skills, name)).filter(Boolean);
    const steps = [
      { id: 'research_inputs', kind: 'preflight', label: 'Research inputs', label_zh: '研究输入核对' },
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
    return `<article class="eusk-workflow-detail eusk-wfc"><header><h2>${tr('Custom workflow', '自定义流程')}</h2><p>${tr('Choose the methods and preferred order. EasyICU checks their fit and input requirements against your question and data, then prepares one complete plan for review before running analyses.', '选择希望使用的方法及顺序。EasyICU 会根据你的问题和数据核对适用性与前置条件，整合成一份完整计划供审阅，审阅前不运行分析。')}</p></header>
      <div class="eusk-workflow-actions"><button type="button" class="eusk-use" data-wfc-start ${picked.length ? '' : 'disabled'}>${tr('Start with this combination', '用此组合开始')}</button>${picked.length ? `<button type="button" data-wfc-clear>${tr('Clear', '清空')}</button>` : ''}</div>
      <section><h2>${tr('Selected steps', '已选步骤')}</h2>${order}</section>
      <section><h2>${tr('Proposed composition', '组合示意（待计划核对）')}</h2><p class="eusk-workflow-note">${tr('The plan determines the actual inputs, dependencies and order. A patient cohort is required only for methods that use patient-level data.', '实际输入、依赖和顺序由完整计划确定；仅使用患者级数据的方法才需要患者队列。')}</p>${graph(previewRow(skills))}</section>
      <section><h2>${tr('Modules', '模块')}</h2><input type="search" class="eusk-wfc-search" data-wfc-search value="${esc(query)}" placeholder="${tr('Search modules', '搜索模块')}" aria-label="${tr('Search modules', '搜索模块')}"><div class="eusk-wfc-modules">${list}</div></section></article>`;
  }

  function startRequest(skills) {
    const names = chosen.map(name => byName(skills, name)).filter(Boolean).map(item => item.title);
    const steps = names.map((name, index) => `${index + 1}. ${name}`);
    return tr(`Use this custom composition as a proposal: ${steps.join('; ')}. Reuse the question and data already provided, check each method's fit, prerequisites and order, and ask only for missing information that changes the design. Explain unsuitable steps and alternatives. Combine data preparation, methods, diagnostics and deliverables in one complete plan for my review; do not run analyses before that review.`,
      `请参考这个自定义组合准备研究方案：${steps.join('；')}。复用已提供的问题和数据，核对各方法的适用性、前置条件和顺序，只集中补问会改变设计的缺失信息；不适用时说明理由和替代。将数据准备、方法、诊断和交付要求合并成一份完整计划供我审阅，审阅前不运行分析。`);
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
      if (!chosen.some(name => byName(skills, name))) return true;
      window.sessionStorage.removeItem('easyicu.skillHub.method');
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
