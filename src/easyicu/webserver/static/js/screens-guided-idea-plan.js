/* Guided Copilot idea-mining plan/replan panel.
   Owner: Guided Idea Mining pre-Agent planning widget. */
(function () {
  'use strict';

  function list(items, fallback, cls, esc) {
    const rows = Array.isArray(items) ? items.filter(Boolean) : [];
    if (!rows.length) return fallback || '';
    return `<div class="${cls || 'gdi-feature-list'}">${rows.map(item => {
      const label = typeof item === 'object' ? (item.title || item.pattern || '') : String(item || '');
      const useFor = typeof item === 'object' ? item.use_for : '';
      const guardrail = typeof item === 'object' ? item.guardrail : '';
      return `<div class="gdi-feature-row one"><div><strong>${esc(label)}</strong>${useFor ? `<small>${esc(useFor)}</small>` : ''}${guardrail ? `<small>${esc(guardrail)}</small>` : ''}</div></div>`;
    }).join('')}</div>`;
  }

  function render(ctx) {
    const t = ctx.t;
    const esc = ctx.esc;
    const attr = ctx.attr;
    const icon = ctx.icon;
    const guidedIdea = ctx.getIdea();
    if (!guidedIdea || !guidedIdea.result) return '';
    if (!guidedIdea.dataContextConfirmed) {
      return `
        <div class="gdi-plan">
          <div class="gdi-ledger-title">
            <div><span class="gdx-label">${t('Step 4 · Plan / replan', '第 4 步 · 计划 / replan')}</span><strong>${t('Locked until data context is confirmed', '确认数据上下文前保持锁定')}</strong></div>
            <span class="pill warn">${t('not ready', '未就绪')}</span>
          </div>
          <p>${esc(t('Idea Mining has only proposed a research question. It has not completed the cohort, feature modules, or analysis setup. Confirm the local export/cohort/module context before drafting an Agent plan.', 'Idea Mining 现在只是提出研究问题，还没有完成队列、特征模块或分析设置。请先确认本地导出、队列和模块上下文，再生成 Agent 计划草案。'))}</p>
        </div>`;
    }
    const draft = guidedIdea.planDraft || guidedIdea.result.idea_plan || null;
    const plan = draft && draft.plan ? draft.plan : null;
    if (!plan) {
      return `
        <div class="gdi-plan">
          <div class="gdi-ledger-title">
            <div><span class="gdx-label">${t('Step 4 · Plan / replan', '第 4 步 · 计划 / replan')}</span><strong>${t('Create a study plan before Agent handoff', '交给 Agent 前先生成研究计划')}</strong></div>
            <span class="pill warn">${t('plan required', '需要计划')}</span>
          </div>
          <p>${esc(t('The next step is an explicit planning pass: clinical question, feasibility risks, cohort/module confirmations, analysis family, and reporting boundary. This still does not create an Agent run.', '下一步是显式计划步骤：整理临床问题、可行性风险、队列/模块确认、分析类型和报告边界。这仍然不会创建 Agent run。'))}</p>
          <div class="gdx-actions">
            <button type="button" class="btn primary" data-gi-plan ${guidedIdea.planning ? 'disabled' : ''}>${guidedIdea.planning ? '<span class="spin"></span>' : icon('agent', 13)} ${t('Generate study plan', '生成研究计划')}</button>
          </div>
        </div>`;
    }
    const steps = Array.isArray(plan.analysis_plan) ? plan.analysis_plan : [];
    const patterns = Array.isArray(plan.reference_analysis_patterns) ? plan.reference_analysis_patterns : [];
    const constraints = Array.isArray(plan.clinical_icu_constraints) ? plan.clinical_icu_constraints : [];
    const confirmations = Array.isArray(plan.required_user_confirmations) ? plan.required_user_confirmations : [];
    const appliedNotes = plan.human_plan_notes || '';
    return `
      <div class="gdi-plan">
        <div class="gdi-ledger-title">
          <div><span class="gdx-label">${t('Step 4 · Plan / replan before Agent', '第 4 步 · Agent 前计划 / replan')}</span><strong>${esc(plan.research_question || t('Confirm the plan before Agent run', '先确认计划，再进入 Agent run'))}</strong></div>
          <span class="pill warn">${t('draft locked', '草稿锁定')}</span>
        </div>
        <div class="gdx-status ok">
          <span>${icon('check', 12)}</span>
          <div><strong>${esc(plan.plan_status || t('draft plan requires review', '计划草案需要审阅'))}</strong><small>${esc((plan.agent_boundary && plan.agent_boundary.reason) || t('Planning is not an Agent execution and does not unlock manuscript claims.', '计划不是 Agent 执行，也不会解锁稿件结论。'))}</small></div>
        </div>
        ${steps.length ? `<ol>${steps.map(row => `<li>${esc(row)}</li>`).join('')}</ol>` : ''}
        ${patterns.length ? `<details class="gdi-plan-details" open><summary>${icon('book', 13)} ${t('Reference method patterns', '参考方法套路')}</summary>${list(patterns, '', 'gdi-feature-list', esc)}</details>` : ''}
        ${constraints.length ? `<details class="gdi-plan-details"><summary>${icon('shield', 13)} ${t('ICU constraints', 'ICU 场景约束')}</summary>${list(constraints, '', 'gdi-feature-list', esc)}</details>` : ''}
        ${confirmations.length ? `<div class="gdr-note"><strong>${t('Still needs user confirmation', '仍需用户确认')}</strong><br>${confirmations.map(esc).join(' · ')}</div>` : ''}
        ${appliedNotes ? `<div class="gdr-note"><strong>${t('Applied replan note', '已应用的 replan 说明')}</strong><br>${esc(appliedNotes)}</div>` : ''}
        <label class="gdi-field wide">
          <span>${t('Plan / replan notes', '计划 / replan 说明')}</span>
          <textarea rows="3" data-gi-field="planEdits" placeholder="${attr(t('e.g. restrict to first ICU stay; add a missingness sensitivity; compare dose groups only after timing is confirmed.', '例如：限制首次 ICU；增加缺失敏感性；只有确认时序后再比较剂量分组。'))}">${esc(guidedIdea.planEdits || '')}</textarea>
        </label>
        <div class="gdx-actions">
          <button type="button" class="btn" data-gi-replan ${guidedIdea.planning ? 'disabled' : ''}>${guidedIdea.planning ? '<span class="spin"></span>' : icon('refresh', 13)} ${t('Replan from notes', '根据说明重规划')}</button>
          <button type="button" class="btn primary" data-gi-handoff ${guidedIdea.handoffing ? 'disabled' : ''}>${icon('lock', 13)} ${t('Freeze handoff for Agent', '冻结交接给 Agent')}</button>
          <button type="button" class="btn" data-gi-project ${!guidedIdea.handoff || guidedIdea.projectCreating ? 'disabled' : ''}>${icon('agent', 13)} ${t('Create Agent project', '创建 Agent 项目')}</button>
          ${guidedIdea.project ? `<button type="button" class="btn" data-open="agent">${t('Open Agent Projects', '打开 Agent Projects')}</button>` : ''}
        </div>
      </div>`;
  }

  window.EU_GUIDED_IDEA_PLAN = { render };
})();
