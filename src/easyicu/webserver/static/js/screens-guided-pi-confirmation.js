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
        note: tr('This is a deterministic local check. A separate confirmation will authorize the Research Agent Planner only after preflight passes.', '这是确定性的本地检查；预检通过后，才会另行确认是否授权 Research Agent Planner 生成正式计划。'),
        approve: tr('Run local preflight', '运行本地预检'),
      };
      if (code === 'provider_ready_to_generate_plan') return {
        code, grants: ['provider_run', 'literature'],
        message: tr(
          'Start generating the formal research plan.',
          '开始生成正式研究计划。',
        ),
        title: tr('Generate a formal research plan now?', '现在生成正式研究计划吗？'),
        note: tr('A prepared package is optional at this stage. EasyICU can plan from its database capability catalog without reading patient rows; if a package already exists, it can also use its bounded feasibility metadata. Extraction and analysis remain blocked.', '这一阶段不强制要求已准备的数据包。EasyICU 可以只依据数据库能力目录出计划且不读取患者行；若已有数据包，也可利用其中受限的可行性元数据。数据提取和分析仍保持阻断。'),
        approve: tr('Start plan generation', '开始生成研究计划'),
      };
      if (code === 'plan_configuration_superseded' || code === 'plan_review_not_resumable') return {
        code, grants: ['provider_run', 'literature'],
        message: tr(
          'I confirm that the old plan must not be reused. Start a fresh Research Agent planning run from the current study configuration, and pause again for my review before analysis.',
          '我确认旧计划不能复用。请按当前研究配置启动一次全新的 Research Agent 规划，并在分析前再次停下让我审核。',
        ),
        title: tr('The study changed. Generate a fresh analysis plan?', '研究配置已更新，是否重新生成分析计划？'),
        note: tr('The old run stays as history. The new run receives a new id and current configuration digest.', '旧 run 仅保留为历史；新 run 使用新的标识和当前配置摘要。'),
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
        title: tr('The formal Agent Plan is ready. Review the plan and its literature before analysis.', '正式研究计划已生成；请先审阅计划及其文献依据。'),
        note: tr('Open both review materials below. The literature view shows retrieval provenance, screening rationale, and exact plan-step citation bindings.', '请先打开下方两份审阅材料；文献依据会显示检索来源、筛选理由和每个计划步骤的精确引用绑定。'),
        approve: tr('Approve and continue', '批准并继续'),
        reject: tr('Reject this plan', '拒绝当前计划'),
        rejectMessage: tr(
          'I reject this exact current plan and keep it as immutable history. Do not execute it and do not change the study configuration. Submit only the rejected review decision; a replacement plan must be started as a separate governed action.',
          '我拒绝当前这份计划，并将其保留为不可变历史。不要执行该计划，也不要修改研究配置。本次只提交“拒绝”审核决定；新的替代计划必须作为下一次独立受治理操作启动。',
        ),
        reviewMaterialsTitle: tr('Required review materials', '计划审阅材料'),
        reviewResources: [
          { kind: 'research_artifact', run_id: String((host.session() && host.session().binding && host.session().binding.run_id) || ''), artifact: 'agent_plan.json', label: tr('Preview formal Agent Plan', '预览正式研究计划'), media_type: 'application/json' },
          { kind: 'research_artifact', run_id: String((host.session() && host.session().binding && host.session().binding.run_id) || ''), artifact: 'literature_evidence.json', label: tr('View literature evidence for this plan', '查看该计划的文献依据'), media_type: 'application/json' },
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
        title: tr('Scientific plan review requires changes', '科学计划审阅要求修改'),
        note: tr(
          'The plan is saved and analysis has not started. EasyICU should propose the endpoint and sensitivity design inside the revised plan; you review the complete plan instead of designing those details here.',
          '计划已经保存，分析尚未开始。结局定义和敏感性分析应由 EasyICU 在修订版计划中提出；你只需审阅完整计划，不必在这里替系统设计细节。',
        ),
        approve: tr('Generate revised candidate plan', '生成修订版候选计划'),
        reviewMaterialsTitle: tr('Plan, literature, and scientific review', '计划、文献依据与科学审阅'),
        reviewResources: [
          { kind: 'research_artifact', run_id: String((host.session() && host.session().binding && host.session().binding.run_id) || ''), artifact: 'scientific_plan_review.json', label: tr('Open scientific review', '打开科学审阅'), media_type: 'application/json' },
          { kind: 'research_artifact', run_id: String((host.session() && host.session().binding && host.session().binding.run_id) || ''), artifact: 'agent_plan.json', label: tr('Preview proposed Agent Plan', '预览候选研究计划'), media_type: 'application/json' },
          { kind: 'research_artifact', run_id: String((host.session() && host.session().binding && host.session().binding.run_id) || ''), artifact: 'literature_evidence.json', label: tr('View literature evidence for this plan', '查看该计划的文献依据'), media_type: 'application/json' },
        ],
      };
      return null;
    }

    function workflowConfirmationHtml() {
      const confirmation = workflowConfirmation();
      if (host.busy() || sessionIsStale() || !confirmation) return '';
      const reviewResources = (confirmation.reviewResources || [])
        .filter(resource => resource.run_id)
        .map(resource => resourceButton(resource, resource.label))
        .join('');
      const review = host.workflow() && host.workflow().plan_review_summary;
      const authorizationQuestions = confirmation.code === 'plan_scientific_changes_required'
        && review && Array.isArray(review.authorization_questions)
        ? review.authorization_questions.slice(0, 4).map(item => `<li>${esc(localizedAuthorizationQuestion(item))}</li>`).join('')
        : '';
      const remediationCounts = confirmation.code === 'plan_scientific_changes_required'
        && review && review.remediation_buckets && typeof review.remediation_buckets === 'object'
        ? {
            automatic: (Array.isArray(review.remediation_buckets.agent_plan_revision) ? review.remediation_buckets.agent_plan_revision.length : 0),
            evidence: (Array.isArray(review.remediation_buckets.external_evidence) ? review.remediation_buckets.external_evidence.length : 0)
              + (Array.isArray(review.remediation_buckets.independent_review) ? review.remediation_buckets.independent_review.length : 0),
          }
        : { automatic: 0, evidence: 0 };
      const decisionCount = authorizationQuestions
        ? Math.min(4, review.authorization_questions.length)
        : 0;
      const firstDecision = decisionCount
        ? localizedAuthorizationQuestion(review.authorization_questions[0])
        : '';
      const reviewStatus = confirmation.code === 'plan_scientific_changes_required'
        ? `<div class="gpi-confirmation-review-status"><strong>${esc(decisionCount
          ? tr(`Answer this first: ${firstDecision}`, `现在先回答：${firstDecision}`)
          : tr('EasyICU will revise the candidate plan before analysis', 'EasyICU 将在分析前修订候选计划'))}</strong><span>${esc(decisionCount
            ? tr(
              `${Math.max(0, decisionCount - 1)} more decision follows. EasyICU will handle ${remediationCounts.automatic + remediationCounts.evidence} plan and evidence items.`,
              `保存后还有 ${Math.max(0, decisionCount - 1)} 个决定；其余 ${remediationCounts.automatic + remediationCounts.evidence} 项计划修订与补证由 EasyICU 处理。`,
            )
            : tr(
              `${remediationCounts.automatic + remediationCounts.evidence} plan and evidence items are system-owned; no scientific setup answer is required here.`,
              `${remediationCounts.automatic + remediationCounts.evidence} 项计划修订与补证由系统负责；这里不需要回答科学设定问题。`,
            ))}</span></div>`
        : '';
      return `<section class="gpi-confirmation${confirmation.code === 'plan_scientific_changes_required' ? ' is-science-review' : ''}" aria-label="${tr('Workflow confirmation required', '需要确认科研流程')}">
        <span class="gpi-confirmation-icon" aria-hidden="true">${iconHtml('shield', 17)}</span>
        <div><strong>${esc(confirmation.title)}</strong><small>${esc(confirmation.note)}</small>${reviewStatus}${reviewResources ? `<details class="gpi-confirmation-resources"><summary>${esc(tr('View the plan and references', '查看候选计划与依据'))}</summary><div>${reviewResources}</div></details>` : ''}</div>
        <div class="gpi-confirmation-actions">
          ${confirmation.code === 'plan_scientific_changes_required' && !decisionCount ? '' : `<button class="btn ${confirmation.code === 'plan_scientific_changes_required' ? 'primary ' : ''}sm" type="button" data-gpi-confirm-edit>${confirmation.code === 'plan_scientific_changes_required' ? tr('Answer decision 1', '回答第 1 项') : confirmation.code === 'provider_ready_to_generate_plan' ? tr('Add research requirements', '我想先补充研究要求') : tr('Request changes', '提出修改')}</button>`}
          ${confirmation.rejectMessage ? `<button class="btn sm" type="button" data-gpi-confirm-reject>${esc(confirmation.reject)}</button>` : ''}
          ${confirmation.nonApprovable ? '' : `<button class="btn primary sm" type="button" data-gpi-confirm-action>${esc(confirmation.approve)}</button>`}
        </div>
      </section>`;
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
        ROBUSTNESS_AUTHORITY_NOT_PRESPECIFIED: tr(
          'Which executable sensitivity analyses should be prespecified for this study?',
          '这项研究需要预先设定哪些可执行的敏感性分析？',
        ),
      };
      return known[code] || String((item && (item.question || item.code)) || '');
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
