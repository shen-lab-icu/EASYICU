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

  function normalizePlanStep(row, i, t) {
    const original = row && typeof row === 'object' ? row : null;
    const text = original
      ? [original.title, original.action, original.output, original.guardrail].filter(Boolean).join(' ')
      : String(row || '').trim();
    const lower = text.toLowerCase();
    const has = (needle) => lower.includes(needle);
    const mk = (title, action, output, guardrail) => ({ title, action, output, guardrail });
    if (has('lock the clinical question') || has('freeze the clinical question')) {
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
        t('Export/cohort/module contract for the governed Research Agent handoff.', '交给受治理 Research Agent 的导出/队列/模块契约。'),
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
    if (has('descriptive association') || has('crude contrasts')) {
      return mk(
        t('Start with a descriptive association', '先进行描述性关联分析'),
        t('Summarize denominators, exposure and outcome distributions, crude contrasts, and missingness before adjusted models.', '先汇总分母、暴露与结局分布、粗比较和缺失情况，再考虑调整模型。'),
        t('Descriptive cohort comparison package.', '描述性队列比较材料。'),
        t('Use adjusted or time-to-event models only after covariates and assumptions are confirmed.', '只有确认协变量和模型假设后，才进入调整模型或时间结局模型。')
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
    if ((has('prior-art') || has('literature')) && !has('agent projects') && !has('project seed')) {
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
        t('Store the locked question, feasibility table, literature interpretation, and analysis steps as a governed handoff.', '把锁定问题、可行性表、文献解释和分析步骤存为受治理的交接对象。'),
        t('Metadata-only project seed.', '仅包含元数据的项目种子。'),
        t('Evidence checks and human sign-off remain required.', '仍然需要证据核验和人工签署。')
      );
    }
    return original || mk(text || `${t('Plan step', '计划步骤')} ${i + 1}`, '', '', t('Review this legacy planning note before Agent handoff.', '交给 Agent 前需要审阅这条历史计划说明。'));
  }

  function localizedStatus(value, t) {
    const key = String(value || '').trim();
    const labels = {
      draft_plan_requires_user_review: t('Draft requires researcher review', '草案等待研究者审阅'),
      searched_no_hits: t('Search completed; no candidates retained', '检索完成，未保留候选文献'),
      searched: t('Search completed', '检索已完成'),
      search_failed: t('Search did not complete', '检索未完成'),
      blocked: t('Not yet established', '尚未验证'),
      partial: t('Partially established', '部分成立'),
      ready: t('Ready for review', '可供审阅'),
      needs_review: t('Needs construct review', '需要构念审阅'),
    };
    return labels[key] || key || t('not established', '尚未成立');
  }

  function localizedRole(value, t) {
    const key = String(value || 'feature').trim();
    const labels = {
      exposure: t('Exposure', '暴露'),
      predictor: t('Predictor', '预测因素'),
      outcome: t('Outcome', '结局'),
      feature: t('Feature', '候选特征'),
      eligibility_or_episode: t('Population or episode construction', '人群或通气阶段构建'),
    };
    return labels[key] || key;
  }

  function localizedVariableLabel(value, t) {
    const key = String(value || '').trim();
    const labels = {
      'Total IV Fluid Input': t('Total IV Fluid Input', '静脉液体总输入'),
      'Cumulative Fluid Balance': t('Cumulative Fluid Balance', '累积液体平衡'),
      'Hourly Fluid Balance': t('Hourly Fluid Balance', '小时液体平衡'),
      'Mechanical Ventilation Mode': t('Mechanical Ventilation Mode', '机械通气模式'),
    };
    return labels[key] || key;
  }

  function localizedRequirement(value, t) {
    const key = String(value || '').trim();
    const labels = {
      'prepare or register a usable EasyICU export': t('Prepare or register a usable EasyICU export', '准备或登记可用的 EasyICU 数据导出'),
      'prepare or select a real EasyICU export': t('Prepare or select a real EasyICU export', '准备或选择真实的 EasyICU 数据导出'),
      'local export / database source': t('Local export or database source', '本地导出或数据库来源'),
      'cohort denominator and inclusion/exclusion criteria': t('Cohort denominator and inclusion/exclusion criteria', '队列分母和纳入排除标准'),
      'feature modules and mapped concepts': t('Feature modules and measurable concepts', '特征模块和可测量概念'),
      'analysis family and reporting boundary': t('Analysis family and reporting boundary', '分析类型和报告边界'),
      'resolve idea feasibility before Agent execution': t('Resolve idea feasibility before execution', '执行前完成研究想法的可行性验证'),
    };
    return labels[key] || key;
  }

  function localizedResolution(value, t) {
    const labels = {
      direct_observed: t('Directly available', '可直接提取'),
      validated_derived: t('Registered derived feature', '已注册派生特征'),
      event_reconstructable: t('Clinical event reconstruction required', '需要重建临床事件'),
      proxy_only: t('Proxy only', '只有代理信号'),
      unavailable: t('Unavailable', '当前不可获得'),
    };
    return labels[String(value || '')] || t('Requires review', '需要审阅');
  }

  function renderConstructAnswerability(plan, t, esc, icon) {
    const summary = plan.source_feasibility_summary && typeof plan.source_feasibility_summary === 'object'
      ? plan.source_feasibility_summary : {};
    const rows = Array.isArray(summary.construct_answerability)
      ? summary.construct_answerability.filter(row => row && typeof row === 'object') : [];
    if (!rows.length) return '';
    const design = summary.design_answerability && typeof summary.design_answerability === 'object'
      ? summary.design_answerability : {};
    const ready = summary.status === 'ready' && rows.every(row => row.verdict === 'ready');
    const joint = Number.isFinite(Number(design.joint_observed_entities))
      ? Number(design.joint_observed_entities) : null;
    const source = summary.source && typeof summary.source === 'object' ? summary.source : {};
    const why = ready
      ? t(
        'The real export contains the required concepts, timing can be reconstructed, and every clinical construct passed its typed check. Final confirmation still does not start analysis.',
        '真实导出已包含所需概念，时间顺序可以重建，且所有临床构念均通过类型化检查；最终确认仍不会自动启动分析。'
      )
      : t(
        'At least one clinical construct still needs a definition, materialization receipt, or a better data source. It cannot be called executable yet.',
        '至少一个临床构念仍缺少定义、物化回执或更合适的数据源，因此现在还不能称为可执行。'
      );
    return `
      <details class="gdi-plan-details" open>
        <summary>${icon('shield', 13)} ${t('Clinical construct and data answerability', '临床构念与数据可回答性')}</summary>
        <div class="gdi-feature-list">
          ${rows.map(row => {
            const facing = row.user_facing && typeof row.user_facing === 'object' ? row.user_facing : {};
            const explanation = facing.explanation || row.rationale_zh || '';
            return `<div class="gdi-feature-row one"><div><strong>${esc(row.label_zh || row.requested_term || t('Clinical construct', '临床构念'))}</strong><small>${esc(`${facing.label || localizedStatus(row.verdict, t)} · ${localizedResolution(row.resolution_kind, t)}`)}</small>${explanation ? `<small>${esc(explanation)}</small>` : ''}${row.semantic_warning ? `<small>${esc(row.semantic_warning)}</small>` : ''}</div></div>`;
          }).join('')}
        </div>
        <div class="gdr-note"><strong>${ready ? t('Why it is executable now', '为什么现在可执行') : t('Why it is not executable yet', '为什么现在还不可执行')}</strong><br>${esc(why)}${source.label ? `<br>${esc(`${t('Confirmed source', '已确认数据源')}: ${source.label}${joint === null ? '' : ` · ${t('Jointly usable', '联合可用')} ${joint}`}`)}` : ''}</div>
      </details>`;
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
        ${steps.length ? `<ol>${steps.map((row, i) => {
          const obj = normalizePlanStep(row, i, t);
          const title = obj.title || obj.action || t('Plan step', '计划步骤');
          const detail = [obj.action, obj.output ? `${t('Output', '产物')}: ${obj.output}` : '', obj.guardrail ? `${t('Guardrail', '约束')}: ${obj.guardrail}` : ''].filter(Boolean).join(' · ');
          return `<li><strong>${esc(title)}</strong>${detail ? `<br><small>${esc(detail)}</small>` : ''}</li>`;
        }).join('')}</ol>` : ''}
        ${patterns.length ? `<details class="gdi-plan-details" open><summary>${icon('book', 13)} ${t('Reference method patterns', '参考方法套路')}</summary>${list(patterns, '', 'gdi-feature-list', esc)}</details>` : ''}
        ${constraints.length ? `<details class="gdi-plan-details"><summary>${icon('shield', 13)} ${t('ICU constraints', 'ICU 场景约束')}</summary>${list(constraints, '', 'gdi-feature-list', esc)}</details>` : ''}
        ${renderConstructAnswerability(plan, t, esc, icon)}
        ${confirmations.length ? `<div class="gdr-note"><strong>${t('Still needs user confirmation', '仍需用户确认')}</strong><br>${confirmations.map(esc).join(' · ')}</div>` : ''}
        ${appliedNotes ? `<div class="gdr-note"><strong>${t('Applied replan note', '已应用的 replan 说明')}</strong><br>${esc(appliedNotes)}</div>` : ''}
        <label class="gdi-field wide">
          <span>${t('Plan / replan notes', '计划 / replan 说明')}</span>
          <textarea rows="3" data-gi-field="planEdits" placeholder="${attr(t('e.g. restrict to first ICU stay; add a missingness sensitivity; compare dose groups only after timing is confirmed.', '例如：限制首次 ICU；增加缺失敏感性；只有确认时序后再比较剂量分组。'))}">${esc(guidedIdea.planEdits || '')}</textarea>
        </label>
        <div class="gdx-actions">
          <button type="button" class="btn" data-gi-replan ${guidedIdea.planning ? 'disabled' : ''}>${guidedIdea.planning ? '<span class="spin"></span>' : icon('refresh', 13)} ${t('Replan from notes', '根据说明重规划')}</button>
          <button type="button" class="btn primary" data-gi-handoff ${guidedIdea.handoffing ? 'disabled' : ''}>${icon('lock', 13)} ${t('Freeze handoff for Agent', '冻结交接给 Agent')}</button>
          <button type="button" class="btn" data-gi-project ${!guidedIdea.handoff || guidedIdea.projectCreating ? 'disabled' : ''}>${icon('agent', 13)} ${t('Create project seed', '创建项目种子')}</button>
          ${guidedIdea.project ? `<button type="button" class="btn" data-open="agent">${t('Open Project Monitor', '打开项目监控')}</button>` : ''}
        </div>
      </div>`;
  }

  function renderArtifact(payload, deps) {
    const plan = payload && payload.plan ? payload.plan : (payload || {});
    const t = deps.tr;
    const esc = deps.esc;
    const icon = deps.icon;
    const steps = Array.isArray(plan.analysis_plan) ? plan.analysis_plan : [];
    const variables = Array.isArray(plan.variables) ? plan.variables : [];
    const confirmations = Array.isArray(plan.required_user_confirmations) ? plan.required_user_confirmations : [];
    const prior = plan.prior_art_review && typeof plan.prior_art_review === 'object'
      ? plan.prior_art_review : {};
    const gate = plan.execution_gate && typeof plan.execution_gate === 'object'
      ? plan.execution_gate : {};
    const blockers = Array.isArray(gate.blockers) ? gate.blockers : [];
    const confirmed = plan.confirmed_plan_fields && typeof plan.confirmed_plan_fields === 'object'
      ? plan.confirmed_plan_fields : {};
    const appliedNotes = String(plan.human_plan_notes || '').trim();
    const feasibility = plan.source_feasibility_summary && typeof plan.source_feasibility_summary === 'object'
      ? plan.source_feasibility_summary : {};
    return `
      <section class="gdi-plan gdi-plan-artifact">
        <div class="gdi-ledger-title">
          <div><span class="gdx-label">${t('Idea Mining plan preview', 'Idea Mining 方案预览')}</span><strong>${esc(plan.research_question || t('Research question pending', '研究问题待确认'))}</strong></div>
          <span class="pill warn">${esc(localizedStatus(plan.plan_status, t))}</span>
        </div>
        <div class="gdi-ledger-grid">
          <div><span>${t('Population', '研究人群')}</span><strong>${esc((plan.cohort && plan.cohort.default) || t('Pending', '待确认'))}</strong></div>
          <div><span>${t('Exposure', '暴露')}</span><strong>${esc(plan.exposure || confirmed.exposure || t('Pending', '待确认'))}</strong></div>
          <div><span>${t('Outcome', '结局')}</span><strong>${esc(plan.outcome || confirmed.outcome || t('Pending', '待确认'))}</strong></div>
          <div><span>${t('Time window', '时间窗口')}</span><strong>${esc(plan.time_window || confirmed.time_window || t('Pending', '待确认'))}</strong></div>
          <div><span>${t('Literature review', '文献调研')}</span><strong>${esc(localizedStatus(prior.status || t('not checked', '尚未检索'), t))}</strong><small>${esc(`${Number(prior.result_count || 0)} ${t('metadata candidate(s)', '篇元数据候选')}`)}</small></div>
          <div><span>${t('Feasibility', '数据可行性')}</span><strong>${esc(localizedStatus(feasibility.status || (plan.pre_experiment_summary && plan.pre_experiment_summary.status), t))}</strong><small>${t('No patient rows or effect estimates', '不展示患者行或效应估计')}</small></div>
          <div><span>${t('Execution', '执行状态')}</span><strong>${gate.agent_run_ready_after_human_confirmation ? t('eligible after confirmation', '确认后才可能执行') : t('blocked', '当前阻断')}</strong><small>${t('This preview never starts analysis', '本预览不会启动分析')}</small></div>
        </div>
        ${appliedNotes ? `<div class="gdr-note"><strong>${t('Researcher-confirmed constraints', '研究者已确认的约束')}</strong><br>${esc(appliedNotes)}</div>` : ''}
        ${variables.length ? `<details class="gdi-plan-details" open><summary>${icon('grid', 13)} ${t('Variable roles', '变量角色')}</summary><div class="gdi-feature-list">${variables.map(row => `<div class="gdi-feature-row one"><div><strong>${esc(localizedVariableLabel(row.label || row.concept_id || '', t))}</strong><small>${esc(localizedRole(row.role, t))}</small></div></div>`).join('')}</div></details>` : ''}
        ${steps.length ? `<details class="gdi-plan-details" open><summary>${icon('list', 13)} ${t('Planned steps', '计划步骤')}</summary><ol>${steps.map((row, i) => {
          const obj = normalizePlanStep(row, i, t);
          const detail = [obj.action, obj.output ? `${t('Output', '产物')}: ${obj.output}` : '', obj.guardrail ? `${t('Guardrail', '约束')}: ${obj.guardrail}` : ''].filter(Boolean).join(' · ');
          return `<li><strong>${esc(obj.title || obj.action || t('Plan step', '计划步骤'))}</strong>${detail ? `<br><small>${esc(detail)}</small>` : ''}</li>`;
        }).join('')}</ol></details>` : ''}
        ${renderConstructAnswerability(plan, t, esc, icon)}
        ${confirmations.length ? `<div class="gdr-note"><strong>${t('Still needs researcher confirmation', '仍需研究者确认')}</strong><br>${confirmations.map(value => esc(localizedRequirement(value, t))).join(' · ')}</div>` : ''}
        ${blockers.length ? `<div class="gdr-note"><strong>${t('Why execution remains blocked', '为什么当前不能执行')}</strong><br>${blockers.map(value => esc(localizedRequirement(value, t))).join(' · ')}</div>` : ''}
      </section>`;
  }

  window.EU_GUIDED_IDEA_PLAN = { render, renderArtifact };
})();
