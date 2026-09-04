/* Guided Copilot workflow-authority panel owner.

   Owner: projecting the bound 7-stage workflow into the right-hand panel --
   current stage, its reason, progress, and the full stage list. Split out of
   screens-guided-pi.js, which was hundreds of lines past its size ratchet.

   Read-only by contract: it renders host state into #gdStudyAside and never
   mutates it. The legacy Guided shell must not write this panel while the
   Copilot shell is mounted; screens-guided.js guards its own renderAside for
   exactly that reason. */
(function () {
  'use strict';

  function create(host) {
    const tr = host.tr;
    const esc = host.esc;
    const iconHtml = host.iconHtml;
    const projectId = host.projectId;
    const displayProjectTitle = host.displayProjectTitle;

    function syncProjectWorkflowAside() {
      const demo = host.demoMode() && window.EasyICU.guidedPi.optional('demo');
      const workflow = demo && typeof demo.workflow === 'function' ? demo.workflow() : host.workflow();
      if (host.shell() !== 'pi' || (!host.demoMode() && !projectId())) return;
      const aside = document.getElementById('gdStudyAside');
      const body = document.getElementById('gdAsideBody');
      const head = aside && aside.querySelector('.gd-aside-head');
      if (!aside || !body || !head) return;
      if (!workflow) {
        const receipt = host.project() && host.project().binding_receipt;
        const revision = receipt && Number.isInteger(receipt.study_context_revision)
          ? ` · r${receipt.study_context_revision}` : '';
        head.innerHTML = `<div class="at">${tr('Research progress', '研究进度')}</div><div class="asub">${tr('Loading this project’s saved progress.', '正在读取当前项目的进度。')}</div>`;
        body.innerHTML = `<div class="gd-pipeline-summary" data-gpi-project-workflow-loading role="status" aria-live="polite">
          <div class="gd-pipeline-summary-head"><div><strong>${esc(displayProjectTitle(host.project() && host.project().title, projectId()))}${esc(revision)}</strong><div class="gd-pipeline-value">${tr('Loading project progress…', '正在读取项目进度…')}</div></div></div>
        </div>`;
        return;
      }
      const stages = Array.isArray(workflow.stages) ? workflow.stages : [];
      const reviewerDemo = workflow.kind === 'reviewer_validation_demo';
      const names = {
        question: reviewerDemo ? tr('Reviewer protocol', '审稿协议') : tr('Scientific question', '科学问题'),
        idea: reviewerDemo ? tr('Validation scope', '验证范围') : tr('Idea mining', '想法发掘'),
        setup: reviewerDemo ? tr('Data contract', '数据合同') : tr('Study setup', '研究配置'),
        extraction: reviewerDemo ? tr('Safe projection', '安全投影') : tr('Feature extraction', '特征提取'),
        plan: tr('Analysis plan', '分析计划'), analysis: tr('Analysis and validation', '分析与验证'),
        interpretation: tr('Result interpretation', '结果解读'), manuscript: reviewerDemo ? tr('Reviewer dossier', '审稿报告') : tr('Manuscript', '稿件'),
      };
      const reasons = {
        question_bound: tr('Question is bound to this project', '科学问题已绑定到当前项目'),
        idea_handoff_accepted: tr('Selected idea is digest-bound', '所选想法已用摘要绑定'),
        prior_art_authority_not_established: tr('Prior-art authority and novelty are not established', '先前研究权限与新颖性未成立'),
        idea_feasibility_refresh_required: tr('Recheck feasibility against the current data source', '需要按当前数据源重新核验可行性'),
        study_setup_complete: tr('Required study setup is complete', '必需研究配置已完成'),
        approved_plan_setup_receipt: tr('The approved plan records the study setup used for this analysis', '已批准的计划记录了本次分析采用的研究配置'),
        active_export_ready: tr('A matching EasyICU export is ready', '同一项目的 EasyICU 数据包已就绪'),
        plan_ready: tr('Ready to create the analysis plan', '可以生成分析计划'),
        provider_ready_to_generate_plan: tr('Question and data source are ready; generate a candidate plan for review', '问题和数据源已就绪，可以生成候选计划供审阅'),
        agent_plan_ready: tr('The digest-bound analysis plan is ready', '摘要绑定分析计划已就绪'),
        operator_plan_approval_required: tr('Review and approve the digest-bound plan before analysis', '请在分析前审核并批准摘要绑定的计划'),
        plan_execution_upgrade_required: tr('Generate one package-bound plan before analysis', '需要先生成一份与数据包绑定的可执行计划'),
        plan_scientific_changes_required: tr('The scientific plan review requires a new study/plan version before analysis', '科学计划审阅要求先形成新的研究/计划版本，当前不能继续分析'),
        plan_configuration_superseded: tr('The study configuration changed; the old plan is superseded and cannot be approved', '研究配置已变化；旧计划已失效，不能再批准'),
        plan_review_not_resumable: tr('The old plan no longer has a live resume authority and must be regenerated', '旧计划的可恢复执行权限已失效，必须重新生成'),
        scientific_plan_review_policy_stale: tr('The scientific review policy changed; regenerate the plan while keeping the prepared data', '科学审阅规则已更新；保留已准备数据并重新生成计划'),
        operator_plan_approved: tr('Digest-bound plan approved by the user', '摘要绑定计划已由用户批准'),
        analysis_ready: tr('Ready for analysis after plan approval', '计划确认后可以执行分析'),
        validated_analysis_required: tr('Validated analysis is required first', '需要先完成并验证分析'),
        validated_analysis_complete: tr('Analysis, validation, and numeric checks are complete', '分析、验证与数值核验已完成'),
        validated_analysis_ready: tr('Analysis, validation, and numeric checks are complete', '分析、验证与数值核验已完成'),
        evidence_bound_interpretation_ready: tr('Review the evidence-bound result interpretation', '请审阅证据约束的结果解读'),
        manuscript_draft_ready_for_review: tr('Review the evidence-bound manuscript draft', '请审阅证据绑定的稿件草稿'),
        interpretation_complete: tr('Evidence-bounded interpretation is complete', '证据约束的结果解读已完成'),
        human_review_required: tr('Draft is locked pending clinical and methods review', '初稿已锁定，等待临床与方法学审阅'),
        source_population_scope_open: tr('Prepared data are traceable, but source-population scope is open', '准备数据可追踪，但来源人群范围未闭合'),
        publication_analysis_incomplete: tr('The executable plan is not a complete publication analysis', '可执行计划不是完整投稿分析'),
        paper_authority_not_granted: tr('Draft generated; publication authority was not granted', '初稿已生成；未授予论文发表权限'),
        full_agent_manuscript_required: tr('A governed Agent manuscript is required', '需要由受治理的 Agent 生成稿件'),
        reviewer_protocol_bound: tr('Six reviewer criteria were bound before results', '已在查看结果前绑定 6 项审稿标准'),
        bounded_validation_objective_selected: tr('The systems-validation objective is explicit', '系统验证目标已明确'),
        prepared_data_contract_verified: tr('The prepared-data and descriptive claim contracts are verified', '准备后数据与描述性结论合同已核验'),
        aggregate_projection_verified: tr('The aggregate-only browser projection passed', '仅聚合浏览器投影已通过'),
        exact_plan_reviewed: tr('The exact six-step plan was reviewed', '精确六步计划已审阅'),
        six_of_six_steps_complete: tr('All six required steps completed', '6 个必需步骤全部完成'),
        descriptive_ceiling_preserved: tr('Interpretation stayed within the descriptive ceiling', '结果解读保持在描述性上限内'),
        reviewer_dossier_complete: tr('The reviewer HTML and PDF dossier are complete', '审稿 HTML 与 PDF 报告已完整生成'),
      };
      const reasonText = stage => reasons[stage && stage.reason_code]
        || tr('Waiting for the preceding governed stage', '等待前一受治理阶段完成');
      const done = Number(workflow.completed_required_stages || 0);
      const total = Math.max(1, Number(workflow.required_stage_count || 7));
      const pct = Math.max(0, Math.min(100, Math.round(done / total * 100)));
      const current = stages.find(stage => stage.id === workflow.current_stage)
        || stages.find(stage => stage.status !== 'complete') || stages[stages.length - 1];
      const currentIndex = Math.max(0, stages.indexOf(current));
      const next = stages.slice(currentIndex + 1).find(stage => stage.status !== 'complete');
      const nextIsActionable = next && ['ready', 'running', 'review_required'].includes(next.status);
      const nextCaption = nextIsActionable
        ? tr('Next step', '下一步')
        : tr('Later stage', '后续阶段');
      head.innerHTML = `<div class="at">${host.demoMode() ? tr('Reviewer demonstration', '审稿人演示') : tr('Research progress', '研究进度')}</div><div class="asub">${host.demoMode() ? tr('Read-only view of one registered run.', '一个已登记运行的只读预览。') : tr('Question, data, plan, and results stay together in this project.', '问题、数据、计划与结果都保存在当前项目中。')}</div>`;
      body.innerHTML = `<div class="gd-pipeline-summary" data-gpi-project-workflow-aside>
        <div class="gd-pipeline-summary-head"><div><div class="eyebrow">${tr('Current stage', '当前阶段')}</div><strong>${esc(names[current && current.id] || (current && current.label) || tr('Ready', '就绪'))}</strong><div class="gd-pipeline-value">${esc(reasonText(current))}</div></div></div>
        <div class="gd-pipeline-bar" aria-label="${tr('EasyICU project progress', 'EasyICU 项目进度')}"><span style="width:${pct}%;"></span></div>
        <div class="gd-pipeline-meta"><span><strong>${done}/${total}</strong> ${tr('stages complete', '个阶段已完成')}</span></div>
        ${next ? `<div class="gd-pipeline-next"><span>${nextCaption}</span><strong>${esc(names[next.id] || next.label || next.id)}</strong></div>` : ''}
      </div>
      <details class="gd-pipeline-disclosure" open><summary><span>${tr('All research stages', '全部研究阶段')}</span><small>${stages.length}</small></summary><div class="gd-pipeline-list" data-gpi-project-workflow-list>${stages.map(stage => {
        const status = stage.status === 'complete' ? 'done' : stage.status === 'ready' || stage.status === 'running' || stage.status === 'review_required' ? 'active' : 'locked';
        const marker = status === 'done' ? iconHtml('check', 11) : status === 'locked' ? iconHtml('lock', 10) : iconHtml('dot', 10);
        return `<div class="study-item ${status}"><span class="si-dot">${marker}</span><div class="si-txt"><div class="si-t">${esc(names[stage.id] || stage.label || stage.id)}</div></div></div>`;
      }).join('')}</div></details>`;
    }

    return { syncProjectWorkflowAside };
  }

  window.EasyICU.guidedPi.declare('aside', { create });
})();
