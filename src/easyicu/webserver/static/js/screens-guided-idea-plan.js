/* Guided Copilot idea-mining plan/replan panel.
   Owner: Guided Idea Mining pre-Agent planning widget.

   DENSITY. Walking the demo flow, this one card measured 5,890 characters in a
   single chat bubble — 89% of the entire conversation up to that point, which
   was 6,640 characters across 9 messages after the reader had done two things.
   Nobody reads that to approve a research plan.

   Where it went: the plan step <ol> alone was 2,192 characters, of which 300
   were the seven step TITLES and 1,879 were their action/output/guardrail
   prose. So the titles — the part you scan to decide whether the plan is
   right — were 14% of the block they were buried in.

   The step list therefore shows titles, with the detail behind ONE toggle
   rather than seven, and the two reference sections are closed by default with
   their counts in the summary (patterns was `open`, contributing another 228).
   Nothing is removed: every character is one click away, and the plan is still
   the thing the reader must approve before any handoff. */
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

  function normalizePlanStep(row, i, t) {
    if (row && typeof row === 'object') return row;
    const text = String(row || '').trim();
    const lower = text.toLowerCase();
    const has = (needle) => lower.includes(needle);
    const mk = (title, action, output, guardrail) => ({ title, action, output, guardrail });
    if (has('lock the clinical question')) {
      return mk(
        t('Freeze the clinical question and estimand', '锁定临床问题和估计目标'),
        t('Confirm population, exposure or index time, comparator, outcome, and analysis window before reading effect estimates.', '先确认人群、暴露或时间零点、比较组、结局和分析窗口，再看效应估计。'),
        t('One locked PICOT-style question.', '一条锁定的 PICOT 风格问题。'),
        t('Idea Mining proposes the question; it has not completed cohort selection or analysis.', 'Idea 挖掘只提出问题，还没有完成队列筛选或分析。')
      );
    }
    if (has('confirm the active easyicu export') || has('cohort denominator')) {
      return mk(
        t('Confirm export, cohort, and modules', '确认导出、队列和模块'),
        t('Confirm the real local export, denominator, required modules, and concept mappings with the user.', '和用户确认真实本地导出、分母、所需模块和概念映射。'),
        t('Export/cohort/module contract for Agent Projects.', '交给研究项目的导出/队列/模块契约。'),
        t('MOCK or demo exports are UI rehearsal only.', 'MOCK 或演示导出只能用于界面演练。')
      );
    }
    if (has('outcome-blind feasibility') || has('missingness structure')) {
      return mk(
        t('Run outcome-blind feasibility assessment', '运行不看结局效应的可行性评估'),
        t('Check availability, joint completeness, time-index support, missingness, and event rate before modeling.', '建模前检查可用性、联合完整度、时间索引、缺失结构和事件率。'),
        t('Feasibility table with denominators and blockers.', '包含分母和阻断项的可行性表。'),
        t('Do not present feasibility as a clinical finding.', '不能把可行性检查当成临床结论。')
      );
    }
    if (has('treatment-strategy comparison') || has('timing anchors')) {
      return mk(
        t('Translate the article into an ICU treatment-strategy question', '把文章转译成 ICU 治疗策略问题'),
        t('Define vasopressor/fluid timing anchors, exposure summaries, comparator groups, and eligible windows.', '定义升压药/补液的时间锚点、暴露摘要、比较组和入组窗口。'),
        t('Treatment-strategy contrast ready for review.', '可审阅的治疗策略对照。'),
        t('Flag confounding by indication and immortal-time risk before modeling.', '建模前标记适应症混杂和 immortal-time 风险。')
      );
    }
    if (has('balance and sensitivity') || has('sensitivity checks')) {
      return mk(
        t('Predefine balance and sensitivity checks', '预先定义平衡性和敏感性检查'),
        t('Compare severity, missingness, exposure timing, and alternative dose/window definitions.', '比较严重程度、缺失、暴露时序，以及替代剂量/窗口定义。'),
        t('Sensitivity checklist for replan.', '用于 replan 的敏感性清单。'),
        t('Keep claims exploratory unless assumptions are audited.', '除非假设被审计，否则结论保持探索性。')
      );
    }
    if (has('prior-art') || has('literature')) {
      return mk(
        t('Use existing literature as an inspiration map', '把已有文献当成启发地图'),
        t('Check whether prior studies answer the same question or suggest better comparators, subgroups, timing, or outcomes.', '检查既有研究是否回答同一问题，或提示更好的比较组、亚组、时序和结局。'),
        t('Already answered, partially answered, or new exploratory angle.', '已回答、部分回答，或新的探索角度。'),
        t('Prior work shapes novelty; it does not automatically kill the idea.', '既有研究塑造创新点，不会自动否定 idea。')
      );
    }
    if (has('agent projects') || has('handoff')) {
      return mk(
        t('Create a project seed only after confirmation', '确认后再创建研究项目种子'),
        t('Send the locked question, feasibility table, literature interpretation, and analysis steps to Agent Projects.', '把锁定问题、可行性表、文献解释和分析步骤交给研究项目。'),
        t('Metadata-only project seed.', '仅元数据的项目种子。'),
        t('Evidence checks and human sign-off remain required.', '仍然需要证据核验和人工签署。')
      );
    }
    return mk(text || `${t('Plan step', '计划步骤')} ${i + 1}`, '', '', t('Review this legacy planning note before Agent handoff.', '交给 Agent 前需要审阅这条历史计划说明。'));
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
        ${steps.length ? `
        <div class="gdi-plan-steps ${guidedIdea.planDetail ? 'show-detail' : ''}">
          <div class="gdi-plan-steps-head">
            <span>${t('Plan', '研究计划')} · ${steps.length} ${t('steps', '步')}</span>
            <button type="button" class="gdi-plan-toggle" data-gi-plandetail
              >${guidedIdea.planDetail ? t('Hide detail', '收起细节') : t('Show detail', '展开细节')}</button>
          </div>
          <ol>${steps.map((row, i) => {
            const obj = normalizePlanStep(row, i, t);
            const title = obj.title || obj.action || t('Plan step', '计划步骤');
            const detail = [obj.action, obj.output ? `${t('Output', '产物')}: ${obj.output}` : '', obj.guardrail ? `${t('Guardrail', '约束')}: ${obj.guardrail}` : ''].filter(Boolean).join(' · ');
            return `<li><strong>${esc(title)}</strong>${detail ? `<small class="gdi-plan-detail">${esc(detail)}</small>` : ''}</li>`;
          }).join('')}</ol>
        </div>` : ''}
        ${patterns.length ? `<details class="gdi-plan-details"><summary>${icon('book', 13)} ${t('Reference method patterns', '参考方法套路')} · ${patterns.length}</summary>${list(patterns, '', 'gdi-feature-list', esc)}</details>` : ''}
        ${constraints.length ? `<details class="gdi-plan-details"><summary>${icon('shield', 13)} ${t('ICU constraints', 'ICU 场景约束')} · ${constraints.length}</summary>${list(constraints, '', 'gdi-feature-list', esc)}</details>` : ''}
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
