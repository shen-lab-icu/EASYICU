/* Guided Copilot workflow-confirmation owner.
   Owner: which confirmation one workflow state requires, and how that card
   renders. Split out of screens-guided-pi.js, which was 620 lines past its
   size ratchet and mixed this catalogue with session lifecycle, transcript
   projection and event wiring.

   This owner is read-only by contract: it never mutates host state, never
   sends a message, and never grants an action. It returns the confirmation
   descriptor -- including the grants the host should attach if the user
   approves -- and the host remains the only place a turn is actually sent. */
(function () {
  'use strict';

  function create(host) {
    const tr = host.tr;
    const esc = host.esc;
    const iconHtml = host.iconHtml;
    const resourceButton = host.resourceButton;
    const sessionIsStale = host.sessionIsStale;

    function preparedDataStatus() {
      const authorization = (host.session() && host.session().data_source_authorization) || {};
      const scope = String(authorization.extraction_scope || '');
      const source = authorization.source || {};
      const sourceLabel = String(source.label || source.reference_release || source.database || '').trim();
      const titles = {
        reuse_prepared_full: tr('Prepared source data and planned variables are available', '已复用现有数据源，计划变量可用'),
        all_supported: tr('Source data and planned variables are available', '数据源与计划变量可用'),
        study_required: tr('Study data source and planned variables are available', '本研究数据源与计划变量可用'),
      };
      return {
        title: titles[scope] || tr('The source data and planned variables are available', '数据源与计划变量可用'),
        sourceLabel,
        detail: tr(
          'Approval reuses this source. The final analytic cohort, preprocessing, and models are still pending and run only after approval.',
          '批准后复用该数据源；最终分析队列、数据预处理和模型仍待执行，不会重复提取已有数据。',
        ),
      };
    }

    function workflowConfirmation() {
      const workflow = host.workflow() || {};
      const code = String(workflow.next_action_code || '');
      const archivedJobs = Array.isArray(host.session() && host.session().archived_child_jobs)
        ? host.session().archived_child_jobs : [];
      const latestFailedAgentJob = archivedJobs.slice().reverse().find(job => (
        job && job.kind === 'agent-run' && job.status === 'failed'
      ));
      const latestReviewableAgentJob = archivedJobs.slice().reverse().find(job => (
        job && job.kind === 'agent-run' && job.status === 'done'
        && Array.isArray(job.artifact_refs)
        && job.artifact_refs.some(ref => ref && ref.artifact === 'agent_plan.json')
        && job.artifact_refs.some(ref => ref && ref.artifact === 'literature_evidence.json')
      ));
      const historicalRunId = String(
        latestReviewableAgentJob && latestReviewableAgentJob.run_id
        || latestReviewableAgentJob && latestReviewableAgentJob.artifact_refs
          && latestReviewableAgentJob.artifact_refs.find(ref => ref && ref.run_id)?.run_id
        || ''
      );
      const planContractExhausted = Boolean(
        latestFailedAgentJob
        && latestFailedAgentJob.error_code === 'research_pipeline_plan_contract_exhausted'
      );
      if (code === 'extraction_ready') return {
        code, grants: ['extract'],
        message: tr('I confirm the current study setup. Start data extraction and quality review.', '我确认当前研究配置，请开始数据提取和质量审阅。'),
        title: tr('Study setup is complete. Start data extraction and quality review?', '研究配置已完成，开始数据提取和质量审阅吗？'),
        note: tr('This creates a governed export with denominator, missingness, provenance, and extraction receipts.', '这会生成带分母、缺失率、来源和提取回执的受治理数据包。'),
        approve: tr('Confirm extraction', '确认提取'),
      };
      if (code === 'plan_ready') return {
        code, grants: ['run'],
        message: tr('Run the local executability preflight for the approved research brief. Do not generate or claim a formal plan yet.', '请对已确认的研究简报运行本地可执行性预检；此时不要生成或声称已有正式计划。'),
        title: tr('Study setup is ready. Run the local executability preflight?', '研究设定已就绪，是否运行本地可执行性预检？'),
        note: tr('This is a deterministic local check. A separate confirmation will authorize the Research Agent Planner only after preflight passes.', '这是确定性的本地检查；预检通过后，才会另行确认是否授权 Research Agent Planner 生成候选计划。'),
        approve: tr('Run local preflight', '运行本地预检'),
      };
      if (code === 'provider_ready_to_generate_plan') return {
        code, grants: ['provider_run', 'literature'],
        message: tr(
          'Start generating the candidate research plan.',
          '开始生成候选研究计划。',
        ),
        title: tr('Generate a candidate research plan now?', '现在生成候选研究计划吗？'),
        note: tr(
          'The plan first decides which data are needed without reading patient rows. EasyICU then prepares or reuses those data, lets you review the analysis package, and asks you to approve the package-bound executable plan before analysis.',
          '计划会先决定需要哪些数据，不读取患者行；随后 EasyICU 按计划准备或复用数据，让你审阅分析数据包，并在分析前再次审核与数据包绑定的可执行计划。',
        ),
        flowSteps: [
          tr('Generate candidate plan', '生成候选计划'),
          tr('Prepare or reuse data', '按计划准备或复用数据'),
          tr('Review data readiness', '审阅数据准备情况'),
          tr('Approve executable plan and analyse', '审核可执行计划并开始分析'),
        ],
        flowTitle: tr('What happens next', '接下来会发生什么'),
        flowHint: tr('Process preview · no action needed yet', '流程预览 · 现在无需操作'),
        flowCurrent: tr('Current: waiting to generate the candidate plan', '当前：等待生成候选计划'),
        approve: tr('Start plan generation', '开始生成研究计划'),
      };
      if (code === 'plan_configuration_superseded' || code === 'plan_review_not_resumable' || code === 'scientific_plan_review_policy_stale') return {
        code, grants: ['provider_run', 'literature'],
        message: tr(
          'I confirm that the old plan must not be reused. Start a fresh Research Agent planning run from the current study configuration, and pause again for my review before analysis.',
          '我确认旧计划不能复用。请按当前研究配置启动一次全新的 Research Agent 规划，并在分析前再次停下让我审核。',
        ),
        title: code === 'scientific_plan_review_policy_stale'
          ? tr('The scientific review policy was updated. Regenerate the plan?', '科学审阅规则已更新，是否重新生成计划？')
          : tr('The study changed. Generate a fresh analysis plan?', '研究配置已更新，是否重新生成分析计划？'),
        note: code === 'scientific_plan_review_policy_stale'
          ? tr('The prepared data remain available. Only the old review and plan execution authority are stale; no extraction or analysis will be repeated automatically.', '已准备的数据仍可继续使用；只有旧审阅和计划执行权限已过期，不会自动重复提取或开始分析。')
          : tr('The old run stays as history. The new run receives a new id and current configuration digest.', '旧 run 仅保留为历史；新 run 使用新的标识和当前配置摘要。'),
        approve: tr('Generate fresh plan', '重新生成计划'),
      };
      if (code === 'failed_pipeline_execution_retry_available') return {
        code, grants: ['provider_run'],
        message: tr(
          'Retry analysis from the failed step using the exact approved plan and unchanged study configuration. Reuse completed steps and do not run the Planner again.',
          '研究配置未改变。请复用精确批准的计划和已完成步骤，从失败步骤重新执行分析；不要再次运行 Planner。',
        ),
        title: tr('The approved analysis stopped during execution', '已批准的分析在执行阶段停止'),
        note: tr(
          'The failed attempt remains in the audit history. This retry does not regenerate or silently change the approved plan.',
          '失败尝试仍保留在审计历史中；本次重试不会重新生成或暗中修改已批准计划。',
        ),
        approve: tr('Retry analysis from failed step', '从失败步骤重试分析'),
      };
      if (code === 'failed_pipeline_requires_fresh_plan') return {
        code, grants: ['provider_run', 'literature'],
        message: tr(
          'Keep the failed planning attempt and the previous candidate plan as history. Start a fresh Research Agent planning run from the current study configuration, and pause for my review before analysis.',
          '保留失败的规划尝试和上一版候选计划作为历史。请按当前研究配置启动一次新的 Research Agent 规划，并在分析前停下让我审核。',
        ),
        title: planContractExhausted
          ? tr('The revised plan did not pass the scientific contract', '修订版计划未通过科学合同')
          : tr('The previous research task did not complete', '上次科研任务未完成'),
        note: planContractExhausted
          ? tr(
            'Four draft attempts were rejected before analysis. The previous plan and its “0 direct matches” literature snapshot are historical evidence only. Generate a fresh plan; no patient analysis has run.',
            '系统在分析前连续否决了 4 版草案。上一版计划及其“直接匹配 0 篇”的文献页仅为历史记录；需要重新生成计划，患者分析尚未开始。下方只读产物仍属未验证状态，不能签署或发表。',
          )
          : tr(
            'The failed task remains immutable history. Generate a fresh plan before any analysis.',
            '失败任务保留为不可变历史；任何分析开始前都必须重新生成计划。',
          ),
        approve: tr('Generate a fresh plan', '重新生成新计划'),
        reviewMaterialsTitle: tr('Read-only outputs from the failed-closed run', '失败关闭运行的只读产物'),
        reviewResources: historicalRunId ? [
          { kind: 'research_artifact', run_id: historicalRunId, artifact: 'agent_plan.json', label: tr('Preview the previous candidate plan', '预览上一版候选计划'), media_type: 'application/json' },
          { kind: 'research_artifact', run_id: historicalRunId, artifact: 'literature_evidence.json', label: tr('View the previous literature snapshot', '查看上一版文献快照'), media_type: 'application/json' },
          { kind: 'research_artifact', run_id: historicalRunId, artifact: 'result_tables.json', label: tr('Preview unvalidated result tables', '预览未验证结果表'), media_type: 'application/json' },
          { kind: 'research_artifact', run_id: historicalRunId, artifact: 'figure_gallery.json', label: tr('Preview unvalidated figures', '预览未验证图件'), media_type: 'application/json' },
          { kind: 'research_artifact', run_id: historicalRunId, artifact: 'manuscript_provenance.json', label: tr('Preview evidence-bound article', '预览证据绑定文章'), media_type: 'application/json' },
          { kind: 'research_artifact', run_id: historicalRunId, artifact: 'quality_gate.json', label: tr('Review validation gate', '查看验证闸门'), media_type: 'application/json' },
        ] : [],
      };
      if (code === 'planner_checkpoint_resume_available') return {
        code, grants: ['provider_run', 'literature'],
        message: tr(
          'Keep the bounded Planner run as immutable history. Continue formal plan generation from its validated checkpoint for the unchanged study configuration, and pause for my review before analysis.',
          '保留受预算限制的 Planner 运行为不可变历史。请基于未改变研究配置的已验证 checkpoint 继续生成正式计划，并在分析前停下让我审核。',
        ),
        title: tr('The Planner saved a validated checkpoint. Continue?', 'Planner 已保存验证检查点，是否继续？'),
        note: tr('The validated prefix is reused under a new bounded provider turn. No analysis has run.', '已验证前缀会在新的受限模型轮次中复用；尚未运行任何分析。'),
        approve: tr('Continue plan generation', '继续生成计划'),
      };
      if (code === 'operator_plan_approval_required') return {
        code, grants: ['provider_run'],
        message: tr(
          'I approve this exact evidence-bound plan without changing the study configuration. Decline optional study-authority additions for this run, preserve every open scientific finding as a limitation, and resume the current plan.',
          '我批准当前这份证据绑定的计划，不修改研究配置。本轮不新增可选的科学设定；请把所有未闭合的科学问题保留为局限，并继续执行当前计划。',
        ),
        title: tr('The plan and pre-analysis data check are ready', '计划与分析前数据检查已准备好'),
        note: tr('Review the plan, data readiness, and references. Approval then builds the final cohort, preprocesses the data, and runs the analysis.', '请查看计划、数据准备情况和文献依据；批准后才会生成最终分析队列、完成预处理并运行分析。'),
        approve: tr('Approve and start analysis', '批准并开始分析'),
        compactApproval: true,
        dataStatus: preparedDataStatus(),
        reject: tr('Reject this plan', '拒绝当前计划'),
        rejectMessage: tr(
          'I reject this exact current plan and keep it as immutable history. Do not execute it and do not change the study configuration. Submit only the rejected review decision; a replacement plan must be started as a separate governed action.',
          '我拒绝当前这份计划，并将其保留为不可变历史。不要执行该计划，也不要修改研究配置。本次只提交“拒绝”审核决定；新的替代计划必须作为下一次独立受治理操作启动。',
        ),
        reviewMaterialsTitle: tr('Quick review', '快速审阅'),
        reviewResources: [
          { kind: 'research_artifact', run_id: String((host.session() && host.session().binding && host.session().binding.run_id) || ''), artifact: 'agent_plan.json', label: tr('Research plan', '研究计划'), media_type: 'application/json' },
          { kind: 'research_artifact', run_id: String((host.session() && host.session().binding && host.session().binding.run_id) || ''), artifact: 'literature_evidence.json', label: tr('Literature evidence', '文献依据'), media_type: 'application/json' },
        ],
      };
      if (code === 'plan_execution_upgrade_required') return {
        code, grants: ['configure', 'extract'],
        message: tr(
          'I confirm this candidate plan as the basis for data preparation. Keep the plan, prepare only the cohort, variables, and time windows it requires, then generate a package-bound execution plan and pause again for my review before analysis.',
          '我确认以这份候选计划作为数据准备依据。请保留计划，只准备计划所需的队列、变量和时间窗；数据包就绪后再生成与数据包绑定的可执行计划，并在分析前再次停下让我审核。',
        ),
        title: tr(
          'Candidate plan ready; confirm data preparation next',
          '候选研究计划已生成，请确认数据准备',
        ),
        note: tr(
          'This plan used metadata only; it does not mean a prepared package already exists. EasyICU will prepare the required data first and will not start analysis yet.',
          '这份计划只使用了能力目录，并不代表数据包已经准备好。EasyICU 将先按计划准备所需数据，此时不会开始分析。',
        ),
        approve: tr('Confirm plan and prepare data', '确认方案并准备数据'),
        reviewMaterialsTitle: tr('Preview-only plan kept as history', '仅预览计划保留为历史'),
        reviewResources: [
          { kind: 'research_artifact', run_id: String((host.session() && host.session().binding && host.session().binding.run_id) || ''), artifact: 'agent_plan.json', label: tr('Preview the current plan', '预览当前计划'), media_type: 'application/json' },
          { kind: 'research_artifact', run_id: String((host.session() && host.session().binding && host.session().binding.run_id) || ''), artifact: 'literature_evidence.json', label: tr('View its literature evidence', '查看当前文献依据'), media_type: 'application/json' },
        ],
      };
      if (code === 'plan_scientific_changes_required') return {
        code, grants: ['provider_run', 'literature'],
        nonApprovable: Boolean(
          workflow.plan_review_summary
          && Array.isArray(workflow.plan_review_summary.authorization_questions)
          && workflow.plan_review_summary.authorization_questions.length
        ),
        message: tr(
          'Keep the current candidate plan as immutable review evidence. Let EasyICU apply the Planner-owned review findings and generate a revised candidate plan before any extraction or analysis.',
          '保留当前候选计划作为不可变审阅证据。请由 EasyICU 处理 Planner 负责的审阅项，并在任何数据提取或分析前生成修订版候选计划。',
        ),
        title: tr('The candidate plan needs one revision before it can run', '候选计划还需要修订，暂不能执行'),
        note: tr(
          'The plan is saved and analysis has not started. EasyICU should propose the endpoint and sensitivity design inside the revised plan; you review the complete plan instead of designing those details here.',
          '计划已经保存，分析尚未开始。结局定义和敏感性分析应由 EasyICU 在修订版计划中提出；你只需审阅完整计划，不必在这里替系统设计细节。',
        ),
        approve: tr('Generate revised candidate plan', '生成修订版候选计划'),
        reviewMaterialsTitle: tr('View the plan and review evidence', '查看计划与审阅依据'),
        reviewResources: [
          { kind: 'research_artifact', run_id: String((host.session() && host.session().binding && host.session().binding.run_id) || ''), artifact: 'agent_plan.json', label: tr('Open the complete plan', '打开完整计划'), media_type: 'application/json' },
          { kind: 'research_artifact', run_id: String((host.session() && host.session().binding && host.session().binding.run_id) || ''), artifact: 'literature_evidence.json', label: tr('View literature evidence', '查看文献依据'), media_type: 'application/json' },
          { kind: 'research_artifact', run_id: String((host.session() && host.session().binding && host.session().binding.run_id) || ''), artifact: 'scientific_plan_review.json', label: tr('View review details', '查看审阅详情'), media_type: 'application/json' },
        ],
      };
      return null;
    }

    function workflowConfirmationHtml() {
      const confirmation = workflowConfirmation();
      if (host.busy() || sessionIsStale() || !confirmation) return '';
      const reviewResourceButtons = (confirmation.reviewResources || [])
        .filter(resource => resource.run_id)
        .map(resource => resourceButton(resource, resource.label));
      const reviewResources = reviewResourceButtons.join('');
      const review = host.workflow() && host.workflow().plan_review_summary;
      const planPreview = host.workflow() && host.workflow().plan_conversation_preview;
      const decisionItems = confirmation.code === 'plan_scientific_changes_required'
        && review && Array.isArray(review.authorization_questions)
        ? review.authorization_questions.slice(0, 4)
        : [];
      const remediationCounts = confirmation.code === 'plan_scientific_changes_required'
        && review && review.remediation_buckets && typeof review.remediation_buckets === 'object'
        ? {
            automatic: (Array.isArray(review.remediation_buckets.agent_plan_revision) ? review.remediation_buckets.agent_plan_revision.length : 0),
            evidence: (Array.isArray(review.remediation_buckets.external_evidence) ? review.remediation_buckets.external_evidence.length : 0)
              + (Array.isArray(review.remediation_buckets.independent_review) ? review.remediation_buckets.independent_review.length : 0),
          }
        : { automatic: 0, evidence: 0 };
      const decisionCount = decisionItems.length;
      const firstDecisionItem = decisionCount ? decisionItems[0] : null;
      const firstDecisionCopy = firstDecisionItem
        ? localizedAuthorizationDecisionCopy(firstDecisionItem)
        : null;
      const reviewStatus = confirmation.code === 'plan_scientific_changes_required'
        ? `<div class="gpi-confirmation-review-status"><strong>${esc(decisionCount
          ? firstDecisionCopy.recommendation
          : tr('EasyICU will revise the candidate plan before analysis', 'EasyICU 将在分析前修订候选计划'))}</strong><span>${esc(decisionCount
            ? firstDecisionCopy.reason
            : tr(
              `${remediationCounts.automatic + remediationCounts.evidence} plan and evidence items are system-owned; no scientific setup answer is required here.`,
              `${remediationCounts.automatic + remediationCounts.evidence} 项计划修订与补证由系统负责；这里不需要回答科学设定问题。`,
            ))}</span></div>`
        : '';
      const dataStatus = confirmation.dataStatus
        ? confirmation.compactApproval
          ? `<div class="gpi-confirmation-data-status is-compact"><strong>${esc(confirmation.dataStatus.title)}${confirmation.dataStatus.sourceLabel ? ` · ${esc(confirmation.dataStatus.sourceLabel)}` : ''}</strong><small>${esc(confirmation.dataStatus.detail)}</small></div>`
          : `<div class="gpi-confirmation-data-status"><strong>${esc(confirmation.dataStatus.title)}</strong>${confirmation.dataStatus.sourceLabel ? `<span>${esc(confirmation.dataStatus.sourceLabel)}</span>` : ''}<small>${esc(confirmation.dataStatus.detail)}</small></div>`
        : '';
      const flowSteps = Array.isArray(confirmation.flowSteps) && confirmation.flowSteps.length
        ? `<div class="gpi-confirmation-flow-overview">
            <div class="gpi-confirmation-flow-heading"><strong>${esc(confirmation.flowTitle || tr('What happens next', '接下来会发生什么'))}</strong><span>${esc(confirmation.flowHint || tr('Process preview · no action needed yet', '流程预览 · 现在无需操作'))}</span></div>
            <ol class="gpi-confirmation-flow">${confirmation.flowSteps.map((step, index) => `<li${index === 0 ? ' class="is-current"' : ''}><span>${index + 1}</span><div><b>${esc(step)}</b>${index === 0 && confirmation.flowCurrent ? `<em>${esc(confirmation.flowCurrent)}</em>` : ''}</div></li>`).join('')}</ol>
          </div>`
        : '';
      const reviewMaterials = reviewResources
        ? confirmation.code === 'plan_scientific_changes_required'
          ? `<details class="gpi-confirmation-resources"><summary>${esc(confirmation.reviewMaterialsTitle || tr('View the plan and review evidence', '查看计划与审阅依据'))}</summary><div>${reviewResources}</div></details>`
          : `<div class="gpi-confirmation-resources is-expanded"><strong>${esc(confirmation.reviewMaterialsTitle || (confirmation.compactApproval ? tr('Quick review', '快速审阅') : tr('View the plan and references', '查看计划与依据')))}</strong><div>${reviewResourceButtons[0] || ''}${confirmation.compactApproval ? `<button class="gpi-resource-link" type="button" data-gpi-confirm-preview-data>${esc(tr('Data readiness check', '数据准备检查'))}</button>` : ''}${reviewResourceButtons.slice(1).join('')}</div></div>`
        : '';
      const compactOtherAction = confirmation.compactApproval && confirmation.rejectMessage
        ? `<details class="gpi-confirmation-more"><summary>${esc(tr('Other actions', '其他操作'))}</summary><button type="button" data-gpi-confirm-reject>${esc(confirmation.reject)}</button></details>`
        : '';
      const planConversation = planPreview && confirmation.compactApproval
        ? planConversationHtml(planPreview)
        : '';
      const displayedTitle = firstDecisionCopy ? firstDecisionCopy.cardTitle : confirmation.title;
      const displayedNote = firstDecisionCopy ? firstDecisionCopy.context : confirmation.note;
      const decisionActions = firstDecisionCopy && Array.isArray(firstDecisionCopy.options)
        ? firstDecisionCopy.options.map((option, index) => `<button class="btn ${index === 0 ? 'primary ' : ''}sm" type="button" data-gpi-next-choice="${esc(option.message)}">${esc(option.label)}</button>`).join('')
        : '';
      return `${planConversation}<section class="gpi-confirmation${confirmation.code === 'plan_scientific_changes_required' ? ' is-science-review' : ''}${confirmation.compactApproval ? ' is-plan-approval' : ''}" aria-label="${tr('Workflow confirmation required', '需要确认科研流程')}">
        <span class="gpi-confirmation-icon" aria-hidden="true">${iconHtml('shield', 17)}</span>
        <div><strong>${esc(displayedTitle)}</strong><small>${esc(displayedNote)}</small>${flowSteps}${dataStatus}${reviewStatus}${reviewMaterials}${compactOtherAction}</div>
        <div class="gpi-confirmation-actions">
          ${confirmation.dataStatus && !confirmation.compactApproval ? `<button class="btn sm" type="button" data-gpi-confirm-preview-data>${esc(tr('Preview analysis data', '先预览分析数据'))}</button>` : ''}
          ${decisionActions || (confirmation.code === 'plan_scientific_changes_required' && !decisionCount ? '' : `<button class="btn ${confirmation.code === 'plan_scientific_changes_required' ? 'primary ' : ''}sm" type="button" data-gpi-confirm-edit>${confirmation.code === 'plan_scientific_changes_required' ? tr('Answer this question', '回答这个问题') : confirmation.code === 'provider_ready_to_generate_plan' ? tr('Add research requirements', '我想先补充研究要求') : confirmation.code === 'failed_pipeline_execution_retry_available' ? tr('Generate a fresh research plan', '重新生成研究计划') : confirmation.compactApproval ? tr('Change plan', '修改计划') : tr('Request changes', '提出修改')}</button>`)}
          ${confirmation.rejectMessage && !confirmation.compactApproval ? `<button class="btn sm" type="button" data-gpi-confirm-reject>${esc(confirmation.reject)}</button>` : ''}
          ${confirmation.nonApprovable ? '' : `<button class="btn primary sm" type="button" data-gpi-confirm-action>${esc(confirmation.approve)}</button>`}
        </div>
      </section>`;
    }

    function planConversationHtml(preview) {
      const rows = Array.isArray(preview && preview.items) ? preview.items.slice(0, 6) : [];
      if (!rows.length) return '';
      const labels = {
        population_and_unit: tr('Population and unit', '研究人群与分析单位'),
        exposure_and_timing: tr('Exposure and timing', '暴露定义与时间窗'),
        outcome_and_followup: tr('Outcome and follow-up', '结局与随访'),
        adjustment_and_model: tr('Adjustment and model', '调整变量与模型'),
        missing_data: tr('Missing data', '缺失数据处理'),
        sensitivity_and_feasibility: tr('Checks and sensitivity', '执行前检查与敏感性分析'),
      };
      const counts = [
        preview.analysis_step_count
          ? `${preview.analysis_step_count} ${tr('analysis steps', '个分析步骤')}`
          : preview.step_count ? `${preview.step_count} ${tr('steps', '个步骤')}` : '',
        preview.output_step_count ? `${preview.output_step_count} ${tr('output steps', '个产物生成步骤')}` : '',
        preview.table_count ? `${preview.table_count} ${tr('tables', '张表')}` : '',
        preview.figure_count ? `${preview.figure_count} ${tr('figures', '张图')}` : '',
      ].filter(Boolean).join(' · ');
      const design = preview && preview.design && typeof preview.design === 'object' ? preview.design : {};
      const designRows = [
        [tr('Estimand', '要估计什么'), design.estimand],
        [tr('Analysis time zero', '分析时间零点'), design.time_zero],
        [tr('Follow-up', '随访范围'), design.observation_window],
        [tr('Primary method', '主要方法'), design.primary_method],
      ].filter(row => String(row[1] || '').trim());
      const designHtml = designRows.length
        ? `<section class="gpi-plan-exact-settings"><strong>${esc(tr('Exact settings proposed by the plan', '计划建议的具体设定'))}</strong>${designRows.map(row => `<div><b>${esc(row[0])}</b><span>${esc(row[1])}</span></div>`).join('')}${Array.isArray(design.required_variables) && design.required_variables.length ? `<small>${esc(tr('Required variables: ', '需要的变量：'))}${esc(design.required_variables.join('、'))}</small>` : ''}</section>`
        : '';
      const rowHtml = row => `<li><span>${rows.indexOf(row) + 1}</span><div><b>${esc(labels[String(row && row.key || '')] || tr('Plan item', '计划内容'))}</b><em>${esc(row && row.text || '')}</em></div></li>`;
      const primaryRows = rows.slice(0, 4);
      const supportingRows = rows.slice(4);
      return `<article class="gpi-plan-conversation" aria-label="${esc(tr('Candidate research plan summary', '候选研究计划摘要'))}">
        <p><strong>${esc(tr('I have generated a candidate plan from your research question.', '我已经根据你的研究问题生成了一份候选计划。'))}</strong><span>${esc(tr('Here is the core plan for your review. No analysis has started yet.', '核心方案直接列在下面，方便你先审阅；目前还没有开始分析。'))}</span></p>
        <ol>${primaryRows.map(rowHtml).join('')}</ol>
        ${supportingRows.length || designHtml ? `<details class="gpi-plan-conversation-more"><summary>${esc(tr('View data handling and full settings', '查看数据处理与完整设定'))}</summary>${supportingRows.length ? `<ol>${supportingRows.map(rowHtml).join('')}</ol>` : ''}${designHtml}</details>` : ''}
        ${counts ? `<footer>${esc(counts)} · ${esc(tr('Open the full plan below for details and evidence bindings.', '完整步骤与文献绑定可在下方继续查看。'))}</footer>` : ''}
      </article>`;
    }

    function localizedAuthorizationQuestion(item) {
      const code = String((item && item.code) || '');
      const known = {
        OUTCOME_DEFINITION_UNRESOLVED: tr(
          'Which available clinical endpoint and time horizon should this study use?',
          '这项研究应使用哪个当前数据可支持的临床结局及时间范围？',
        ),
        POST_BASELINE_EXPOSURE_TIMING_NOT_CLOSED: tr(
          'Should the revised study use a prespecified landmark/time-varying design, or remain descriptive?',
          '修订后的研究应采用预先设定的 landmark／时变设计，还是仅保留描述性分析？',
        ),
        ADJUSTMENT_SET_NOT_USER_CONFIRMED: tr(
          'Do you approve the proposed baseline adjustment set and its pre-time-zero rationale?',
          '你是否批准建议的基线调整变量及其时间零点前的选择依据？',
        ),
        REPEATED_STAY_IDENTITY_UNAVAILABLE: tr(
          'The same patient may have more than one ICU stay. Choose either: A. keep only the first ICU stay per patient (recommended), or B. keep all stays and account for repeated records in the model.',
          '同一患者可能有多次 ICU 入院。请选择：A. 每位患者只保留第一次 ICU 入院（推荐）；B. 保留全部入院，并在模型中处理同一患者的重复记录。',
        ),
        ROBUSTNESS_AUTHORITY_NOT_PRESPECIFIED: tr(
          'Which executable sensitivity analyses should be prespecified for this study?',
          '这项研究需要预先设定哪些可执行的敏感性分析？',
        ),
      };
      return known[code] || String((item && (item.question || item.code)) || '');
    }

    function localizedAuthorizationDecisionCopy(item) {
      const code = String((item && item.code) || '');
      if (code === 'REPEATED_STAY_IDENTITY_UNAVAILABLE') return {
        cardTitle: tr(
          'The plan is ready, but repeated ICU stays need one rule',
          '计划已生成，但还需确认重复入院规则',
        ),
        context: tr(
          'The same patient may appear more than once in the ICU data. Choose one option below and EasyICU will revise the plan automatically; analysis has not started.',
          '同一患者可能在 ICU 数据中出现多次。请选择下方一种方案，EasyICU 会自动修订计划；目前尚未开始分析。',
        ),
        recommendation: tr(
          'Recommended: analyse each patient once',
          '推荐：每位患者只分析一次',
        ),
        reason: tr(
          'Keep only the first ICU stay for each patient to avoid counting the same patient repeatedly.',
          '每位患者仅保留第一次 ICU 入院，避免同一患者被重复计入。',
        ),
        options: [
          {
            label: tr('Use first ICU stay only (recommended)', '采用推荐方案：仅保留首次入院'),
            message: tr(
              'Use only each patient\'s first ICU stay, then revise and resubmit the candidate research plan with this rule. Keep analysis paused.',
              '每位患者仅保留第一次 ICU 入院，请按这一规则修订并重新提交候选研究计划；分析保持暂停。',
            ),
          },
          {
            label: tr('Keep all stays and account for repeats', '保留全部入院并处理重复记录'),
            message: tr(
              'Keep all ICU stays and account for repeated records from the same patient in the model, then revise and resubmit the candidate research plan. Keep analysis paused.',
              '保留全部 ICU 入院，并在模型中处理同一患者的重复记录；请按这一规则修订并重新提交候选研究计划，分析保持暂停。',
            ),
          },
        ],
      };
      return {
        cardTitle: tr('The plan needs one answer before it can be revised', '计划还需回答 1 个问题才能修订'),
        context: localizedAuthorizationQuestion(item),
        recommendation: tr('Your answer will be saved in the revised plan', '你的回答会写入修订后的计划'),
        reason: tr('Answer this question in the conversation; analysis will remain paused.', '请在对话中回答这个问题；分析仍会保持暂停。'),
      };
    }


    // The shell still needs the localizer for the "one open question at a
    // time" prompt it composes when the user asks to continue the review.
    return {
      workflowConfirmation,
      workflowConfirmationHtml,
      localizedAuthorizationQuestion,
    };
  }

  window.EU_GUIDED_PI_CONFIRMATION = { create };
})();
