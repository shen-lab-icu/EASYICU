/* Guided Copilot transcript projection owner.

   Owner: turning one persisted Pi session into the renderable message
   timeline -- user turns, assistant turns, their activity rows, and the
   lifecycle replay merged onto them. Split out of screens-guided-pi.js, which
   was several hundred lines past its size ratchet and mixed this projection
   with session lifecycle, event wiring and rendering.

   Pure by contract: it reads a session and returns a new array. It never
   touches host state, never sends, and never renders. */
(function () {
  'use strict';

  const CHILD_JOB_SUBMISSION_CODES = new Set([
    'easyicu_extraction_submitted',
    'easyicu_run_submitted',
    'easyicu_full_run_submitted',
    'easyicu_review_submitted',
  ]);
  const IDEA_EXPLORATION_TOOLS = new Set([
    'easyicu_mine_ideas',
    'easyicu_search_literature',
  ]);
  const PLAN_HOST_ACTION_CODES = new Set([
    'auto_generate_plan',
    'auto_revise_plan',
    'generate_plan',
  ]);
  // Host receipts prove that EasyICU ran a plan task; they are not authored
  // conversation turns. A manual click is already shown immediately by the
  // UI, while replay must never manufacture an editable user quotation.
  const SYSTEM_ONLY_HOST_ACTION_CODES = new Set([
    'auto_generate_plan',
    'auto_revise_plan',
    'generate_plan',
  ]);
  const PLAN_ARTIFACTS = [
    ['agent_plan.json', 'Open full plan', '打开完整计划'],
    ['scientific_plan_review.json', 'Open scientific plan review', '打开计划科学审阅'],
    ['scientific_readiness.json', 'Open scientific readiness review', '打开科学就绪审阅'],
    ['literature_evidence.json', 'Open literature evidence', '打开文献依据'],
  ];
  const HOST_ACTION_ARTIFACTS = {
    auto_generate_plan: PLAN_ARTIFACTS,
    auto_revise_plan: PLAN_ARTIFACTS,
    generate_plan: PLAN_ARTIFACTS,
    prepare_analysis_data: [
      ['cohort_summary.json', 'Open cohort summary', '打开队列摘要'],
      ['source_run_manifest.json', 'Open source manifest', '打开数据来源清单'],
      ['run_context.json', 'Open run context', '打开运行上下文'],
      ['agent_plan.json', 'Open executable plan', '打开可执行计划'],
    ],
    execute_plan: [
      ['result_tables.json', 'Open result tables', '打开结果表'],
      ['figure_gallery.json', 'Open figure gallery', '打开图表'],
      ['manuscript_provenance.json', 'Open evidence-bound article', '打开证据绑定文章'],
      ['evidence_ledger.json', 'Open evidence ledger', '打开证据台账'],
      ['scientific_readiness.json', 'Open scientific review', '打开科学审阅'],
    ],
    retry_analysis: [
      ['result_tables.json', 'Open refreshed result tables', '打开更新后的结果表'],
      ['figure_gallery.json', 'Open refreshed figures', '打开更新后的图表'],
      ['evidence_ledger.json', 'Open evidence ledger', '打开证据台账'],
      ['scientific_readiness.json', 'Open scientific review', '打开科学审阅'],
    ],
    review_prepared_data: [
      ['cohort_summary.json', 'Open cohort summary', '打开队列摘要'],
      ['source_run_manifest.json', 'Open source manifest', '打开数据来源清单'],
      ['run_context.json', 'Open run context', '打开运行上下文'],
    ],
    review_results: [
      ['result_tables.json', 'Open result tables', '打开结果表'],
      ['figure_gallery.json', 'Open figure gallery', '打开图表'],
      ['manuscript_provenance.json', 'Open evidence-bound article', '打开证据绑定文章'],
      ['evidence_ledger.json', 'Open evidence ledger', '打开证据台账'],
    ],
    review_result_tables: [
      ['result_tables.json', 'Open result tables', '打开结果表'],
      ['evidence_ledger.json', 'Open evidence ledger', '打开证据台账'],
    ],
    review_figures: [
      ['figure_gallery.json', 'Open figure gallery', '打开图表'],
      ['result_tables.json', 'Open source result tables', '打开图表来源结果表'],
    ],
    review_manuscript: [
      ['manuscript_provenance.json', 'Open evidence-bound article', '打开证据绑定文章'],
      ['evidence_ledger.json', 'Open evidence ledger', '打开证据台账'],
      ['scientific_readiness.json', 'Open scientific review', '打开科学审阅'],
    ],
    review_scientific_review: [
      ['scientific_readiness.json', 'Open scientific review', '打开科学审阅'],
      ['evidence_ledger.json', 'Open evidence ledger', '打开证据台账'],
    ],
  };

  function latestTurnCompletedIdeaExploration(timeline) {
    const rows = Array.isArray(timeline) ? timeline : [];
    let latestUserIndex = -1;
    rows.forEach((row, index) => {
      if (row && row.role === 'user') latestUserIndex = index;
    });
    if (latestUserIndex < 0) return false;
    return rows.slice(latestUserIndex + 1).some(row => (
      row && row.role === 'activity' && Array.isArray(row.steps)
      && row.steps.some(step => (
        step && step.kind === 'tool' && step.status === 'complete'
        && IDEA_EXPLORATION_TOOLS.has(String(step.toolName || ''))
      ))
    ));
  }

  function create(host) {
    const tr = typeof host.tr === 'function' ? host.tr : (en => en);
    const ACTIVITY = host.activity;
    const upsertActivityStep = host.upsertActivityStep;
    const timeMs = host.timeMs;
    const resourceKey = host.resourceKey;
    const modelErrorText = host.modelErrorText;
    const activityHasCompletedAction = host.activityHasCompletedAction || (() => false);
    const workflowActionCode = host.workflowActionCode;

    function hostActionFailed(actionCode, job, status) {
      const normalizedStatus = String(status || '');
      if (['failed', 'cancelled', 'interrupted'].includes(normalizedStatus)) return true;
      if (!PLAN_HOST_ACTION_CODES.has(String(actionCode || '')) || normalizedStatus !== 'done') return false;
      const refs = Array.isArray(job && job.artifact_refs) ? job.artifact_refs : [];
      return !refs.some(ref => ref && String(ref.artifact || '') === 'agent_plan.json');
    }

    function hostActionCopy(actionCode, job, status, turn) {
      const normalizedStatus = String(status || '');
      const interrupted = normalizedStatus === 'interrupted';
      const failed = hostActionFailed(actionCode, job, normalizedStatus);
      const copies = {
        auto_generate_plan: {
          user: '',
          running: tr('EasyICU is automatically generating the research plan', 'EasyICU 正在自动生成研究计划'),
          done: tr('EasyICU automatically generated the plan. Open the complete plan, scientific review, and literature evidence below; analysis has not started.', 'EasyICU 已自动生成计划。可在下方打开完整计划、科学审阅和文献依据；分析尚未开始。'),
          failed: tr('EasyICU could not complete the automatic research plan. Open the execution details before retrying.', 'EasyICU 未能完成自动研究计划，请查看执行明细后重试。'),
        },
        auto_revise_plan: {
          user: '',
          running: tr('EasyICU is automatically revising the research plan', 'EasyICU 正在自动修订研究计划'),
          done: tr('EasyICU automatically revised the plan after its scientific checks. Open the current plan and review evidence below; analysis has not started.', 'EasyICU 已根据科学检查自动修订计划。可在下方打开当前计划和审阅依据；分析尚未开始。'),
          failed: tr('EasyICU could not complete the automatic plan revision. Open the execution details before retrying.', 'EasyICU 未能完成自动计划修订，请查看执行明细后重试。'),
        },
        generate_plan: {
          user: tr('Generate the candidate research plan', '生成候选研究计划'),
          running: tr('Generating the research plan', '正在生成研究计划'),
          done: tr('The plan is ready. Open the complete plan, scientific review, and literature evidence below; analysis has not started.', '计划已生成。可在下方打开完整计划、科学审阅和文献依据；分析尚未开始。'),
          failed: tr('The research plan was not completed. Open the execution details before retrying.', '研究计划未生成完成，请查看执行明细后重试。'),
        },
        prepare_analysis_data: {
          user: tr('Confirm the plan and prepare analysis data', '确认方案并准备分析数据'),
          running: tr('Preparing the analysis data and executable plan', '正在准备分析数据和可执行计划'),
          done: tr('The analysis dataset and executable plan are ready. Review distributions and missingness in the EasyICU Data Workbench first, then verify the cohort summary and source manifest below. Analysis has not started.', '分析数据与可执行计划已准备完成。请先在 EasyICU 数据工作台审阅分布与缺失情况，再核对下方队列摘要和数据来源清单；分析尚未开始。'),
          failed: tr('The analysis data or executable plan could not be prepared. Review the failed step before retrying.', '分析数据或可执行计划未准备完成，请查看失败步骤后重试。'),
        },
        execute_plan: {
          user: tr('Approve the plan and start analysis', '批准计划并开始分析'),
          running: tr('Running the approved research plan', '正在执行已批准的研究计划'),
          done: tr('Analysis and numeric validation are complete. Open the result tables, figures, evidence-bound article, evidence ledger, and scientific review below. These are analysis artifacts, not publication approval.', '分析与数值校验已完成。可在下方逐项打开结果表、图表、证据绑定文章、证据台账和科学审阅；这些是分析产物，不代表论文或发表获批。'),
          failed: tr('The approved analysis did not complete. Review the failed step before retrying.', '已批准的分析未完成，请查看失败步骤后重试。'),
        },
        retry_analysis: {
          user: tr('Retry the incomplete analysis', '重试未完成的分析'),
          running: tr('Retrying the approved research run', '正在重试已批准的科研任务'),
          done: tr('The retry is complete. Review the refreshed results and validation evidence.', '重试已完成，请审阅更新后的结果和校验证据。'),
          failed: tr('The retry did not complete. Review the execution details before trying again.', '重试仍未完成，请查看执行明细后再试。'),
        },
        review_prepared_data: {
          user: tr('Open EasyICU data review and visualization', '打开 EasyICU 数据审阅与可视化'),
          running: tr('Preparing the data review', '正在准备数据审阅'),
          done: tr('The native EasyICU Data Workbench is open. It shows the prepared cohort, selected-feature distributions, coverage, and missingness; the bound receipts below prove the data source and run context, not a causal claim.', 'EasyICU 原生数据工作台已打开，可查看准备队列、已选特征分布、覆盖率与缺失情况；下方绑定回执证明数据来源与运行上下文，不证明因果关系。'),
          failed: tr('The source-data review could not be opened.', '源数据审阅未能打开。'),
        },
        review_results: {
          user: tr('Review the analysis results and figures', '审阅分析结果和图表'),
          running: tr('Preparing the result review', '正在准备结果审阅'),
          done: tr('The governed result view is open. Reopen the result tables, figures, article, and evidence ledger from the verified links below.', '受治理的结果视图已打开。可从下方已绑定链接重新打开结果表、图表、文章和证据台账。'),
          failed: tr('The result review could not be opened.', '结果审阅未能打开。'),
        },
        review_result_tables: {
          user: tr('View the analysis result tables', '查看分析结果表'),
          running: tr('Preparing the result tables', '正在准备分析结果表'),
          done: tr('The result tables are open in reader order: core estimates first, supporting audits second. Display rounding improves readability; JSON retains the exact values and lineage.', '结果表已按阅读顺序打开：先看核心估计，再看支持性审计。可读视图只调整显示精度，JSON 仍保留原始数值与溯源。'),
          failed: tr('The analysis result tables could not be opened.', '分析结果表未能打开。'),
        },
        review_figures: {
          user: tr('Review the analysis figures', '审阅分析图表'),
          running: tr('Preparing the figure review', '正在准备图表审阅'),
          done: tr('The figures regenerated from this run’s registered result tables are open. The primary publication figure is shown at full width; supporting figures remain available below it.', '由本轮登记结果表重新生成的图件已打开。主要投稿图按整栏优先展示，辅助图保留在其下方。'),
          failed: tr('The analysis figures could not be opened.', '分析图表未能打开。'),
        },
        review_manuscript: {
          user: tr('Preview the evidence-bound article', '预览证据绑定文章'),
          running: tr('Preparing the article preview', '正在准备文章预览'),
          done: tr('The evidence-bound article generated after this run’s tables and figures is open. Use the bound scientific review to check which statements remain analysis-only.', '在本轮结果表和图件之后重新生成的证据绑定文章已打开；请结合绑定的科学审阅核对哪些表述仍只属于分析级结论。'),
          failed: tr('The evidence-bound article could not be opened.', '证据绑定文章未能打开。'),
        },
        review_scientific_review: {
          user: tr('View the scientific review', '查看科学审阅'),
          running: tr('Preparing the scientific review', '正在准备科学审阅'),
          done: tr('The scientific readiness review is open. Use the links below to inspect each claim limit and its evidence ledger.', '科学就绪审阅已打开。可用下方链接核对每项结论边界及其证据台账。'),
          failed: tr('The scientific review could not be opened.', '科学审阅未能打开。'),
        },
      };
      const copy = copies[String(actionCode || '')] || null;
      if (!copy) return null;
      const resultsVerified = Boolean(job && job.analysis_results_available
        && job.analysis_validated && job.numeric_verified);
      const progress = Array.isArray(job && job.progress) ? job.progress : [];
      const completedSteps = progress.map(row => {
        const match = /Step\s+(\d+)\/(\d+)\s+complete/i.exec(String(row && row.label || ''));
        return match ? [Number(match[1]), Number(match[2])] : null;
      }).filter(Boolean).sort((left, right) => right[0] - left[0])[0];
      const actionArtifact = String(turn && turn.action_key || '').split(':').slice(1).join(':');
      let doneCopy = copy.done;
      if (actionCode === 'execute_plan' && resultsVerified) {
        const stepText = completedSteps
          ? tr(`${completedSteps[0]}/${completedSteps[1]} approved steps`, `${completedSteps[0]}/${completedSteps[1]} 个批准步骤`)
          : tr('the approved steps', '批准的分析步骤');
        doneCopy = tr(
          `EasyICU completed ${stepText}, numeric validation, figure regeneration, and evidence-bound article generation in this run. Review each output below; publication approval remains separate.`,
          `EasyICU 已在本轮完成 ${stepText}、数值校验、图件重绘和证据绑定文章生成。请在下方逐项审阅；投稿授权仍需另行完成。`,
        );
      } else if (actionCode === 'review_results' && actionArtifact === 'full_analysis_report.json') {
        doneCopy = tr('The complete analysis report is open. It connects the approved plan, executed steps, results, figures, and evidence boundaries.', '完整分析报告已打开，按批准计划、执行步骤、结果、图件和证据边界组织本轮分析。');
      } else if (actionCode === 'review_results' && actionArtifact === 'technical_report.json') {
        doneCopy = tr('The technical analysis report is open. It contains method and validation detail for audit, rather than the reader-facing article narrative.', '技术分析报告已打开，包含供审计的方法与校验细节，不与面向读者的文章叙事混在一起。');
      } else if (actionCode === 'review_manuscript' && actionArtifact === 'manuscript_scaffold.pdf') {
        doneCopy = tr('The LaTeX manuscript PDF generated by this run is open. Its claims remain bounded by the same evidence and scientific review.', '本轮生成的 LaTeX 论文 PDF 已打开；其中论断仍受同一证据与科学审阅约束。');
      }
      return {
        user: copy.user,
        running: copy.running,
        assistant: interrupted
          ? tr('The background task stopped when the EasyICU service restarted. No result was claimed; retry this step.', 'EasyICU 服务重启时后台任务已中断，未生成或认领结果；请重试这一步。')
          : failed ? copy.failed
          : (actionCode === 'execute_plan' && !resultsVerified && job
            ? tr('Plan execution finished. Review the run status and available evidence before continuing.', '计划执行已结束，请先审阅运行状态和现有证据再继续。')
            : doneCopy),
      };
    }

    function runIdFromActionKey(actionKey) {
      const match = String(actionKey || '').match(/^(run_[^:]+)/);
      return match ? match[1] : '';
    }

    function hostActionHistoryKey(turn) {
      const actionCode = String(turn && turn.action_code || '');
      if (!actionCode) return '';
      if (PLAN_HOST_ACTION_CODES.has(actionCode)) return 'plan';
      if (actionCode === 'execute_plan' || actionCode === 'retry_analysis') {
        return 'analysis_execution';
      }
      const artifact = String(turn && turn.action_key || '')
        .split(':').slice(1).join(':');
      return actionCode === 'review_results' && artifact
        ? `${actionCode}:${artifact}`
        : actionCode;
    }

    function sourceJobForHostAction(turn, childJob, session, archivedJobs) {
      if (childJob) {
        return Array.isArray(childJob.artifact_refs) && childJob.artifact_refs.length ? childJob : null;
      }
      const boundRunId = String(session && session.binding && session.binding.run_id || '');
      const runId = runIdFromActionKey(turn && turn.action_key) || boundRunId;
      if (!runId) return null;
      return archivedJobs.slice().reverse().find(job => (
        job && String(job.run_id || '') === runId
        && Array.isArray(job.artifact_refs) && job.artifact_refs.length
        && (job.analysis_results_available || job.analysis_validated || job.numeric_verified)
      )) || archivedJobs.slice().reverse().find(job => (
        job && String(job.run_id || '') === runId
        && Array.isArray(job.artifact_refs) && job.artifact_refs.length
      )) || null;
    }

    function hostActionResources(actionCode, job) {
      const requested = HOST_ACTION_ARTIFACTS[String(actionCode || '')] || [];
      const refs = Array.isArray(job && job.artifact_refs) ? job.artifact_refs : [];
      const seen = new Set();
      return requested.map(([artifact, en, zh]) => {
        const ref = refs.find(row => row && String(row.artifact || '') === artifact);
        if (!ref || seen.has(artifact)) return null;
        seen.add(artifact);
        return { ...ref, conversation_label: tr(en, zh) };
      }).filter(Boolean);
    }

    function hostJobProgressSteps(actionCode, job, copy, startedAt) {
      const steps = [{
        id: 'pipeline-submitted', kind: 'pipeline', step: 'submitted',
        label: copy.running, status: 'complete', at: startedAt, owner: 'EasyICU',
      }];
      const progress = Array.isArray(job && job.progress) ? job.progress : [];
      const refs = Array.isArray(job && job.artifact_refs) ? job.artifact_refs : [];
      const hasPreparedData = refs.some(row => row && ['cohort_summary.json', 'source_run_manifest.json'].includes(String(row.artifact || '')));
      if (actionCode === 'prepare_analysis_data' && hasPreparedData) {
        steps.push({
          id: 'pipeline-inputs', kind: 'pipeline', step: 'inputs', status: 'complete', at: startedAt + 1,
          text: tr('The prepared cohort and source manifest were registered before analysis.', '分析前已登记准备队列和数据来源清单。'),
        });
      }
      const completedSteps = progress.map(row => {
        const match = /Step\s+(\d+)\/(\d+)\s+complete/i.exec(String(row && row.label || ''));
        return match ? [Number(match[1]), Number(match[2])] : null;
      }).filter(Boolean).sort((left, right) => right[0] - left[0])[0];
      const stageDetails = {
        planning: tr('Candidate drafts are contract-checked; unsuccessful drafts remain in the immutable run receipt.', '候选草案会经过结构校验；未通过的草案仍保留在不可变运行回执中。'),
        plan: tr('The complete candidate plan and its scientific review were registered for researcher review.', '完整候选计划及其科学审阅已登记，供研究者审阅。'),
        step: completedSteps ? tr(`${completedSteps[0]}/${completedSteps[1]} approved steps completed.`, `已完成 ${completedSteps[0]}/${completedSteps[1]} 个批准步骤。`) : '',
        coder: completedSteps ? tr(`${completedSteps[0]}/${completedSteps[1]} approved steps completed.`, `已完成 ${completedSteps[0]}/${completedSteps[1]} 个批准步骤。`) : '',
        runner: completedSteps ? tr(`${completedSteps[0]}/${completedSteps[1]} approved steps completed.`, `已完成 ${completedSteps[0]}/${completedSteps[1]} 个批准步骤。`) : '',
        visual_qa: tr('Figure geometry was checked after rendering; the registered result tables remain the numeric source of truth.', '图件渲染后已核查版式；登记结果表仍是数值事实来源。'),
        figure: tr('The publication figure bundle was regenerated from registered run evidence without refitting the model.', '投稿图件包已从本轮登记证据重新生成，图件渲染器不会重新拟合模型。'),
        writer: tr('The article scaffold was regenerated after the result and figure artifacts were available.', '结果与图件产物就绪后，文章骨架已重新生成。'),
        latex: tr('LaTeX, bibliography, and PDF exports were rendered from the regenerated manuscript bundle.', 'LaTeX、参考文献和 PDF 已由重新生成的稿件包导出。'),
      };
      progress.forEach((row, index) => {
        const step = String(row && row.step || '').trim();
        if (!step || step === 'run') return;
        steps.push({
          id: `pipeline-${step}`, kind: 'pipeline', step,
          status: String(job && job.status || '') === 'done' || row.status === 'complete' ? 'complete' : (row.status === 'failed' ? 'error' : 'running'),
          at: startedAt + index + 2,
          text: stageDetails[step] || '',
        });
      });
      return steps;
    }

    function transcriptMessages(session) {
      const rows = Array.isArray(session && session.transcript) ? session.transcript : [];
      const messages = [];
      const tools = new Map();
      const assistantByActivity = new Map();
      const childJobHandoffReplies = new Set();
      let activity = null;
      let turnResources = [];
      let lastTimestamp = Date.now();
      function addTurnResources(resources) {
        (Array.isArray(resources) ? resources : []).forEach(resource => {
          const key = resourceKey(resource);
          if (key && !turnResources.some(item => resourceKey(item) === key)) turnResources.push(resource);
        });
      }
      function closeHistoryActivity(at) {
        if (!activity || activity.status !== 'running') return;
        const endedAt = Number(at || lastTimestamp || activity.startedAt);
        upsertActivityStep(activity, { id: 'terminal', kind: 'settled', status: 'complete', at: endedAt });
        activity.status = 'complete'; activity.endedAt = endedAt;
      }
      rows.forEach((row, index) => {
        const rowAt = timeMs(row.timestamp);
        lastTimestamp = rowAt;
        const parts = Array.isArray(row.content) ? row.content : [];
        const text = parts.filter(p => p && p.type === 'text').map(p => p.text || '').join('');
        if (text && row.role === 'user') {
          closeHistoryActivity(rowAt);
          turnResources = [];
          messages.push({
            id: 'history-' + index, role: 'user', text, complete: true,
            entryId: String(row.entry_id || ''),
            timelineAt: rowAt, timelineOrder: index * 10,
          });
          activity = {
            id: 'history-activity-' + index, role: 'activity', status: 'running',
            startedAt: rowAt, steps: [], timelineAt: rowAt, timelineOrder: index * 10 + 1,
          };
          upsertActivityStep(activity, { id: 'submitted', kind: 'submitted', status: 'complete', at: rowAt });
          messages.push(activity);
        }
        parts.filter(p => p && p.type === 'tool_call').forEach((tool, partIndex) => {
          const id = tool.tool_call_id || `history-tool-${index}-${partIndex}`;
          const toolStep = {
            id: 'tool-' + id, kind: 'tool', toolName: tool.tool_name,
            status: 'running', at: rowAt, startedAt: rowAt,
            resource: tool.resource || null,
          };
          tools.set(id, toolStep);
          if (activity) upsertActivityStep(activity, {
            ...toolStep,
          });
        });
        parts.filter(p => p && p.type === 'tool_result').forEach((receipt, partIndex) => {
          const id = receipt.tool_call_id || `history-result-${index}-${partIndex}`;
          let toolStep = tools.get(id);
          if (!toolStep) {
            toolStep = {
              id: 'tool-' + id, kind: 'tool', toolName: receipt.tool_name,
              startedAt: rowAt,
            };
            tools.set(id, toolStep);
          }
          Object.assign(toolStep, {
            status: receipt.is_error ? 'error' : 'complete', text: receipt.summary || '',
            code: receipt.code || '', owner: receipt.owner || '',
            resource: receipt.resource || toolStep.resource || null,
            resources: Array.isArray(receipt.resources) ? receipt.resources : [],
            endedAt: rowAt,
          });
          addTurnResources([toolStep.resource].concat(toolStep.resources || []));
          if (activity) {
            upsertActivityStep(activity, toolStep);
            if (receipt.job_id && CHILD_JOB_SUBMISSION_CODES.has(String(receipt.code || ''))) {
              activity.childJobHandoff = String(receipt.job_id);
            }
          }
        });
        if (text && row.role !== 'user') {
          const message = {
            id: 'history-' + index, role: row.role || 'assistant', text, complete: true,
            errorCode: row.error_code || '',
            resources: row.role === 'assistant' ? turnResources.slice(0, 24) : [],
            timelineAt: rowAt, timelineOrder: index * 10 + 2,
          };
          messages.push(message);
          if (row.role === 'assistant' && activity) {
            assistantByActivity.set(activity, message);
            if (activity.childJobHandoff) childJobHandoffReplies.add(message.id);
          }
          if (row.role === 'assistant' && activity && !parts.some(p => p && p.type === 'tool_call')) {
            closeHistoryActivity(rowAt);
          }
        } else if (row.role === 'assistant' && row.error_code) {
          messages.push({
            id: 'history-' + index, role: 'assistant',
            text: modelErrorText(row.error_code, activityHasCompletedAction(activity)),
            complete: true, errorCode: row.error_code,
            timelineAt: rowAt, timelineOrder: index * 10 + 2,
          });
          closeHistoryActivity(rowAt);
        }
      });
      closeHistoryActivity(lastTimestamp);
      const replayOwner = window.EasyICU.guidedPi.require('replay');
      const lifecycleTurns = replayOwner && typeof replayOwner.lifecycleTurns === 'function'
        ? replayOwner.lifecycleTurns(session) : [];
      const replayTurns = lifecycleTurns.filter(turn => turn && turn.kind !== 'host_action');
      const hostTurns = lifecycleTurns.filter(turn => turn && turn.kind === 'host_action');
      const historyActivities = messages.filter(row => row.role === 'activity' && !row.childJobId);
      const replayOffset = Math.max(0, historyActivities.length - replayTurns.length);
      replayTurns.forEach((turn, turnIndex) => {
        const replay = Array.isArray(turn && turn.events) ? turn.events : [];
        if (!replay.length) return;
        // Replay turn timestamps are receipt-level seconds, while lifecycle event
        // timestamps retain milliseconds. Use the event envelope for the visible
        // wall clock so the total reconciles with the exclusive phase durations.
        const replayStarted = timeMs((replay[0] && replay[0].at) || (turn && turn.started_at));
        const replayEnded = timeMs((replay[replay.length - 1] && replay[replay.length - 1].at) || (turn && turn.ended_at));
        let replayActivity = historyActivities[replayOffset + turnIndex];
        const isNewReplayActivity = !replayActivity;
        if (!replayActivity) replayActivity = { id: 'saved-activity-' + String((turn && turn.job_id) || replayStarted), role: 'activity', steps: [], expanded: false, timelineAt: replayStarted, timelineOrder: turnIndex * 10 + 1 };
        const turnStatus = String((turn && turn.status) || session.last_turn_status || 'done');
        const durablePlanState = typeof workflowActionCode === 'function'
          ? String(workflowActionCode() || '') : '';
        const wrapupRecovered = turnIndex === replayTurns.length - 1
          && ['failed', 'interrupted'].includes(turnStatus)
          && [
            'operator_plan_approval_required',
            'planner_checkpoint_resume_available',
          ].includes(durablePlanState);
        replayActivity.status = wrapupRecovered ? 'complete'
          : turnStatus === 'running' ? 'running'
          : (['failed', 'interrupted'].includes(turnStatus) ? 'error'
            : (turnStatus === 'cancelled' ? 'cancelled' : 'complete'));
        replayActivity.startedAt = replayActivity.startedAt
          ? Math.min(Number(replayActivity.startedAt), replayStarted)
          : replayStarted;
        replayActivity.endedAt = replayEnded;
        replayActivity.allowedActions = Array.isArray(turn && turn.allowed_actions) ? turn.allowed_actions.slice() : [];
        const submittedChildJob = replay.some(event => (
          event && event.type === 'tool_end' && event.job_id
          && CHILD_JOB_SUBMISSION_CODES.has(String(event.code || ''))
        ));
        const handoffReply = assistantByActivity.get(replayActivity);
        if (submittedChildJob && handoffReply) childJobHandoffReplies.add(handoffReply.id);
        replay.forEach(event => {
          const at = timeMs(event && event.at);
          if (event.type === 'run_start') upsertActivityStep(replayActivity, { id: 'agent', kind: 'agent', status: 'complete', at });
          else if (event.type === 'turn_start') ACTIVITY.startTurn(replayActivity, at);
          else if (event.type === 'turn_end') ACTIVITY.finishTurn(replayActivity, at);
          else if (event.type === 'assistant_start') {
            const phase = replayActivity.steps.filter(item => item.kind === 'assistant').length + 1;
            upsertActivityStep(replayActivity, { id: 'assistant-' + phase, kind: 'assistant', phase, status: 'running', at, startedAt: at });
          } else if (event.type === 'message_end') {
            const phase = replayActivity.steps.slice().reverse().find(item => item.kind === 'assistant' && item.status === 'running');
            if (phase) { phase.status = event.error_code ? 'error' : 'complete'; phase.endedAt = at; phase.stopReason = event.stop_reason || ''; }
          } else if (event.type === 'tool_start') {
            upsertActivityStep(replayActivity, { id: 'tool-' + event.tool_call_id, kind: 'tool', toolName: event.tool_name, status: 'running', at, startedAt: at, resource: event.resource || null });
          } else if (event.type === 'tool_progress') {
            upsertActivityStep(replayActivity, { id: 'tool-' + event.tool_call_id, kind: 'tool', toolName: event.tool_name, status: 'running', at });
          } else if (event.type === 'tool_end') {
            upsertActivityStep(replayActivity, { id: 'tool-' + event.tool_call_id, kind: 'tool', toolName: event.tool_name, status: event.is_error ? 'error' : 'complete', code: event.code || '', owner: event.owner || '', jobId: event.job_id || '', at, endedAt: at, resource: event.resource || null, resources: Array.isArray(event.resources) ? event.resources : [] });
          } else if (event.type === 'retry') upsertActivityStep(replayActivity, { id: 'retry-' + event.attempt, kind: 'retry', status: 'complete', attempt: event.attempt, maxAttempts: event.max_attempts, at, startedAt: at, endedAt: at });
          else if (event.type === 'compaction_start') upsertActivityStep(replayActivity, { id: 'compaction', kind: 'compaction', status: 'running', at, startedAt: at });
          else if (event.type === 'compaction_end') upsertActivityStep(replayActivity, { id: 'compaction', kind: 'compaction', status: event.aborted ? 'error' : 'complete', at, endedAt: at });
        });
        replayActivity.steps.sort((left, right) => Number(left.at || 0) - Number(right.at || 0));
        replayActivity.steps.forEach(step => {
          if (step.status === 'running' && replayActivity.status !== 'running') {
            step.status = replayActivity.status === 'complete' ? 'complete' : 'error';
            step.endedAt = replayActivity.endedAt;
          }
        });
        if (isNewReplayActivity && replayActivity.steps.length) messages.push(replayActivity);
      });
      const archivedJobRows = Array.isArray(session && session.archived_child_jobs)
        ? session.archived_child_jobs : [];
      const archivedJobs = new Map(archivedJobRows.map(job => [String(job && job.job_id || ''), job]));
      const planTurnIndexes = [];
      const planRunIds = new Map();
      const latestHostTurnByHistoryKey = new Map();
      hostTurns.forEach((turn, turnIndex) => {
        const historyKey = hostActionHistoryKey(turn);
        if (historyKey) latestHostTurnByHistoryKey.set(historyKey, turnIndex);
        if (!PLAN_HOST_ACTION_CODES.has(String(turn && turn.action_code || ''))) return;
        planTurnIndexes.push(turnIndex);
        const childJobId = String(turn && turn.child_job_id || '');
        const job = archivedJobs.get(childJobId) || null;
        const sourceJob = sourceJobForHostAction(turn, job, session, archivedJobRows);
        const planRunId = String(sourceJob && sourceJob.run_id || '');
        if (planRunId) planRunIds.set(turnIndex, planRunId);
      });
      const supersededPlanTurns = new Set(
        planTurnIndexes.filter(turnIndex => (
          latestHostTurnByHistoryKey.get('plan') !== turnIndex
        )),
      );
      const supersededPlanRunIds = new Set(
        Array.from(supersededPlanTurns).map(turnIndex => planRunIds.get(turnIndex)).filter(Boolean),
      );
      let previousPassiveReview = null;
      hostTurns.forEach((turn, turnIndex) => {
        const actionCode = String(turn && turn.action_code || '');
        // The replay store keeps every immutable attempt. The main conversation
        // projects only the latest attempt for each user-visible workflow
        // action, so one natural-language question does not turn into dozens of
        // repeated "prepare" or "retry" exchanges after a repair session.
        const historyKey = hostActionHistoryKey(turn);
        if (historyKey && latestHostTurnByHistoryKey.get(historyKey) !== turnIndex) return;
        const childJobId = String(turn && turn.child_job_id || '');
        const job = archivedJobs.get(childJobId) || null;
        const sourceJob = sourceJobForHostAction(turn, job, session, archivedJobRows);
        const actionRunId = runIdFromActionKey(turn && turn.action_key);
        if (
          !PLAN_HOST_ACTION_CODES.has(actionCode)
          && actionRunId && supersededPlanRunIds.has(actionRunId)
          && !(sourceJob && sourceJob.analysis_results_available)
        ) return;
        const status = String((job && job.status) || (turn && turn.status) || 'done');
        const copy = hostActionCopy(
          actionCode,
          sourceJob || job,
          status,
          turn,
        );
        if (!copy) return;
        const startedAt = timeMs(turn.started_at);
        const endedAt = timeMs(turn.ended_at || turn.started_at);
        const legacyDuplicate = !childJobId && actionCode === 'review_results'
          && previousPassiveReview
          && startedAt - previousPassiveReview < 5 * 60 * 1000;
        if (legacyDuplicate) return;
        previousPassiveReview = !childJobId && actionCode === 'review_results'
          ? startedAt : null;
        const actionId = String(turn.job_id || `${actionCode}-${startedAt}`);
        if (!SYSTEM_ONLY_HOST_ACTION_CODES.has(actionCode)) {
          messages.push({
            id: 'host-user-' + actionId, role: 'user', text: copy.user, complete: true,
            hostActionCode: actionCode, timelineAt: startedAt, timelineOrder: turnIndex * 10,
          });
        }
        if (childJobId) {
          const activityStatus = status === 'running' ? 'running'
            : hostActionFailed(actionCode, job || sourceJob, status) ? 'error'
            : (status === 'done' ? 'complete' : (status === 'cancelled' ? 'cancelled' : 'error'));
          const hostActivity = {
            id: 'easyicu-job-' + childJobId, role: 'activity', status: activityStatus,
            startedAt, endedAt: status === 'running' ? null : endedAt,
            childJobId, runningTitle: copy.running, steps: [], expanded: status === 'running',
            timelineAt: startedAt, timelineOrder: turnIndex * 10 + 1,
          };
          hostJobProgressSteps(
            actionCode,
            job || sourceJob,
            copy,
            startedAt,
          ).forEach(step => upsertActivityStep(hostActivity, step));
          messages.push(hostActivity);
        }
        if (status !== 'running') {
          messages.push({
            id: 'host-assistant-' + actionId, role: 'assistant', text: copy.assistant,
            resources: hostActionResources(actionCode, sourceJob),
            hostActionCode: actionCode,
            complete: true, timelineAt: endedAt, timelineOrder: turnIndex * 10 + 2,
          });
        }
      });
      messages.sort((left, right) => {
        const at = Number(left.timelineAt || left.startedAt || 0) - Number(right.timelineAt || right.startedAt || 0);
        return at || Number(left.timelineOrder || 0) - Number(right.timelineOrder || 0);
      });
      return ACTIVITY.focusLatest(messages.filter(row => (
        (row.text || row.role === 'activity') && !childJobHandoffReplies.has(row.id)
      )));
    }

    return { transcriptMessages, latestTurnCompletedIdeaExploration };
  }

  window.EasyICU.guidedPi.declare('transcript', {
    create, latestTurnCompletedIdeaExploration,
  });
})();
