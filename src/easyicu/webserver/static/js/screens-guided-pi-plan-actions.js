/* Governed Plan action owner for Guided Copilot.
   Owns authority calculation and the complete user-action lifecycle from one
   explicit click/edit through review, retry, fresh planning, and terminal job
   handoff. Conversation rendering and HTTP transport remain injected adapters. */
(function () {
  'use strict';

  const FRESH_PLAN_CODES = new Set([
    'provider_ready_to_generate_plan',
    'plan_scientific_changes_required',
    'failed_pipeline_requires_fresh_plan',
    'plan_configuration_superseded',
    'plan_review_not_resumable',
    'scientific_plan_review_policy_stale',
  ]);
  const REPLAY_GRANT_INTENTS = new Set([
    'user_edited_message',
    'replace_plan_response_preserve_study',
  ]);
  const FRESH_REPLAY_CODES = new Set([
    'failed_pipeline_requires_fresh_plan',
    'plan_configuration_superseded',
    'plan_review_not_resumable',
    'scientific_plan_review_policy_stale',
    'plan_scientific_changes_required',
  ]);

  function create(host) {
    const tr = host.tr;
    const regeneration = host.regeneration;
    const nextActions = host.nextActions;
    const replay = host.replay;
    let automaticScientificRevisionStarted = false;

    function unavailable() {
      return !host.session() || host.busy() || host.sessionIsStale();
    }

    function workflowCode() {
      return String((host.workflow() && host.workflow().next_action_code) || '');
    }

    function setPending(value) {
      host.setBusy(Boolean(value));
      host.setError('');
      host.render();
    }

    function generationRequest(reasonCode) {
      const retryExecution = reasonCode === 'failed_pipeline_execution_retry_available';
      const executionUpgrade = reasonCode === 'plan_execution_upgrade_required';
      const resumeCheckpoint = reasonCode === 'planner_checkpoint_resume_available';
      const fresh = reasonCode !== 'provider_ready_to_generate_plan'
        && !resumeCheckpoint
        && !retryExecution;
      return {
        text: retryExecution
          ? tr('Retry analysis from the failed step', '从失败步骤重试分析')
          : executionUpgrade
            ? tr('Confirm the plan and prepare analysis data', '确认方案并准备分析数据')
          : resumeCheckpoint
            ? tr('Continue generating the candidate research plan', '继续生成候选研究计划')
          : fresh
            ? tr('Generate a fresh research plan', '重新生成研究计划')
            : tr('Generate the candidate research plan', '生成候选研究计划'),
        grants: ['provider_run'],
        intent: fresh
          ? 'confirm_fresh_plan_generation'
          : reasonCode === 'provider_ready_to_generate_plan'
            ? 'confirm_formal_plan_generation'
            : '',
      };
    }

    async function startFormalPlanGeneration(reasonCode, options = {}) {
      if (unavailable()) return;
      const automatic = Boolean(options && options.automatic);
      const request = generationRequest(String(reasonCode || ''));
      const session = host.session() || {};
      const binding = session.binding || {};
      const provider = session.research_provider || {};
      const studyContextId = String(binding.study_context_id || '').trim();
      const revisingScientificPlan = reasonCode === 'plan_scientific_changes_required';
      const revisionSourceRunId = revisingScientificPlan
        ? String(binding.run_id || '').trim()
        : '';
      const api = host.api();
      if (
        !studyContextId
        || typeof api.loadStudyContext !== 'function'
        || typeof api.startAgentRun !== 'function'
      ) {
        host.setError(tr(
          'The prepared study is unavailable. Refresh this project and try again.',
          '当前已准备研究不可用，请刷新项目后重试。',
        ));
        host.render();
        return;
      }
      if (!automatic) {
        host.appendMessage({
          id: 'plan-generation-' + Date.now(), role: 'user', complete: true,
          text: request.text,
        });
      }
      setPending(true);
      try {
        const response = await api.loadStudyContext(studyContextId);
        const study = response && (response.context || response.study || response);
        const source = study && study.data_source;
        const sourcePath = String((source && source.path) || '').trim();
        if (!study || !sourcePath) throw new Error('prepared_data_source_unavailable');
        const payload = await api.startAgentRun({
          path: sourcePath,
          study_id: studyContextId,
          study_context_id: studyContextId,
          question: study.question,
          run_type: 'full',
          llm_provider: String(provider.provider || ''),
          credential_source: String(provider.credential_source || ''),
          external_llm_opt_in: true,
          // The visible planning confirmation authorizes literature support as
          // well as the provider turn.  Dropping this flag made every Web
          // revision repeat the same "no direct evidence search" finding.
          literature_search_authorized: true,
          engine: 'research_agent_pipeline',
          // A scientific revision is not a fresh, context-free plan.  Bind the
          // immutable review that requested changes so Planner can repair its
          // own findings instead of rediscovering them on every user click.
          planner_start_mode: reasonCode === 'planner_checkpoint_resume_available'
            ? 'resume_checkpoint'
            : revisingScientificPlan
              ? 'auto'
              : 'fresh',
          plan_revision_source_run_id: revisionSourceRunId,
        });
        await host.recordHostAction(
          automatic
            ? 'auto_revise_plan'
            : reasonCode === 'plan_execution_upgrade_required'
            ? 'prepare_analysis_data'
            : 'generate_plan',
          String(payload.job_id || ''),
          String(payload.job_id || ''),
        );
        host.setBusy(false);
        host.watchChildJob(String(payload.job_id || ''), 'easyicu_full_run_submitted');
      } catch (error) {
        host.setBusy(false);
        host.setError(host.errorText(error));
        host.render();
      }
    }

    async function continueSystemOwnedPlanRevision() {
      const workflow = host.workflow() || {};
      const questions = workflow.plan_review_summary
        && Array.isArray(workflow.plan_review_summary.authorization_questions)
        ? workflow.plan_review_summary.authorization_questions
        : [];
      if (
        String(workflow.next_action_code || '') !== 'plan_scientific_changes_required'
        || questions.length
        || automaticScientificRevisionStarted
      ) return false;
      automaticScientificRevisionStarted = true;
      await startFormalPlanGeneration(
        'plan_scientific_changes_required', {automatic: true},
      );
      return true;
    }

    function regenerationAuthority(text, regenerationIntent) {
      const code = workflowCode();
      const planGrants = REPLAY_GRANT_INTENTS.has(String(regenerationIntent || ''))
        && nextActions && typeof nextActions.governedPlanGrants === 'function'
        ? nextActions.governedPlanGrants(text, code)
        : [];
      const grants = Array.from(new Set([
        ...host.turnGrants().filter(action => action === 'configure'),
        ...planGrants,
      ]));
      const intent = planGrants.includes('provider_run')
        && code === 'provider_ready_to_generate_plan'
        ? 'confirm_formal_plan_generation'
        : planGrants.includes('provider_run')
          && code === 'planner_checkpoint_resume_available'
          ? 'confirm_planner_checkpoint_resume'
          : planGrants.includes('provider_run') && FRESH_REPLAY_CODES.has(code)
            ? 'confirm_fresh_plan_generation'
            : '';
      return Object.freeze({ grants: Object.freeze(grants), intent });
    }

    function resubmitHostGenerated(row, text) {
      const id = String((row && row.id) || '');
      const original = String((row && row.text) || '').trim();
      const message = String(text || '').trim();
      const isPlanAction = value => regeneration
        && typeof regeneration.isPlanActionText === 'function'
        && regeneration.isPlanActionText(value);
      if (!id.startsWith('plan-generation-') && !isPlanAction(original)) return false;
      if (!isPlanAction(message)) return false;
      const code = workflowCode();
      const grants = nextActions && typeof nextActions.governedPlanGrants === 'function'
        ? nextActions.governedPlanGrants(message, code)
        : [];
      if (!grants.includes('provider_run')) return false;
      host.truncateMessagesAt(id);
      void startFormalPlanGeneration(
        code === 'failed_pipeline_execution_retry_available'
          ? 'failed_pipeline_requires_fresh_plan'
          : code,
      );
      return true;
    }

    async function submitReview(decision) {
      if (unavailable()) return;
      const session = host.session() || {};
      const binding = session.binding || {};
      const runId = String(binding.run_id || '').trim();
      const studyContextId = String(binding.study_context_id || '').trim();
      const api = host.api();
      if (!runId || !studyContextId || typeof api.submitAgentRunReview !== 'function') {
        host.setError(tr(
          'The current plan review coordinates are unavailable. Refresh this project and try again.',
          '当前计划的审核坐标不可用，请刷新该项目后重试。',
        ));
        host.render();
        return;
      }
      const approved = decision === 'approved';
      host.appendMessage({
        id: 'plan-review-' + Date.now(), role: 'user', complete: true,
        text: approved
          ? tr('Approve plan and start analysis', '批准计划并开始分析')
          : tr('Reject this plan', '拒绝当前计划'),
      });
      setPending(true);
      try {
        const payload = await api.submitAgentRunReview({
          run_id: runId,
          study_context_id: studyContextId,
          decision,
          external_llm_opt_in: true,
        });
        await host.recordHostAction(
          'execute_plan', String(payload.job_id || ''), String(payload.job_id || ''),
        );
        host.setBusy(false);
        host.watchChildJob(String(payload.job_id || ''), 'easyicu_review_submitted');
      } catch (error) {
        host.setBusy(false);
        host.setError(host.errorText(error));
        host.render();
      }
    }

    async function retryFailedExecution(reason) {
      if (unavailable()) return;
      if (!replay || typeof replay.retryFailedExecution !== 'function') {
        host.setError(tr(
          'The failed run coordinates are unavailable. Refresh this project and try again.',
          '失败运行的恢复坐标不可用，请刷新当前项目后重试。',
        ));
        host.render();
        return;
      }
      const validationRepair = reason === 'validation_repair';
      host.appendMessage({
        id: 'execution-retry-' + Date.now(), role: 'user', complete: true,
        text: validationRepair
          ? tr('Repair the remaining validation item', '修复剩余校验项')
          : tr('Retry analysis from the failed step', '从失败步骤重试分析'),
      });
      setPending(true);
      try {
        const payload = await replay.retryFailedExecution({
          api: host.api(), session: host.session(),
        });
        await host.recordHostAction(
          'retry_analysis', String(payload.job_id || ''), String(payload.job_id || ''),
        );
        host.setBusy(false);
        host.watchChildJob(
          String(payload.job_id || ''),
          validationRepair
            ? 'easyicu_full_run_report_resume_submitted'
            : 'easyicu_full_run_resume_submitted',
        );
      } catch (error) {
        host.setBusy(false);
        host.setError(host.errorText(error));
        host.render();
      }
    }

    async function confirmWorkflow(confirmation) {
      if (!confirmation) return;
      if (confirmation.code === 'operator_plan_approval_required') {
        await submitReview('approved');
        return;
      }
      if (confirmation.code === 'plan_execution_upgrade_required') {
        await startFormalPlanGeneration(confirmation.code);
        return;
      }
      if (confirmation.code === 'failed_pipeline_execution_retry_available') {
        await retryFailedExecution();
        return;
      }
      if (FRESH_PLAN_CODES.has(confirmation.code)) {
        await startFormalPlanGeneration(confirmation.code);
        return;
      }
      if (confirmation.code === 'planner_checkpoint_resume_available') {
        await startFormalPlanGeneration(confirmation.code);
        return;
      }
      await host.sendText(confirmation.message, confirmation.grants);
    }

    async function rejectWorkflow(confirmation) {
      if (!confirmation || !confirmation.rejectMessage) return;
      if (confirmation.code === 'operator_plan_approval_required') {
        await submitReview('rejected');
        return;
      }
      await host.sendText(confirmation.rejectMessage, confirmation.grants);
    }

    async function confirmDecision(selection) {
      if (!selection || unavailable()) return;
      const session = host.session() || {};
      const binding = session.binding || {};
      const expectedRevision = Number(binding.study_revision || 0);
      const runId = String(binding.run_id || '').trim();
      const api = host.api();
      if (!expectedRevision || !runId || typeof api.confirmPiCopilotPlanDecision !== 'function') return;
      setPending(true);
      try {
        const payload = await api.confirmPiCopilotPlanDecision(session.session_id, {
          project_id: host.projectId(),
          decision_code: selection.decision_code,
          option_id: selection.option_id,
          expected_revision: expectedRevision,
          run_id: runId,
        });
        await host.refreshSession(true);
        await host.loadWorkflow();
        host.setBusy(false);
        host.render();
        if (payload && payload.next_action === 'replan') {
          await startFormalPlanGeneration('plan_configuration_superseded');
        } else if (payload && payload.next_action === 'reextract') {
          host.setError(tr(
            'This option needs a new timestamped extraction. EasyICU has saved the choice and kept analysis paused.',
            '该方案需要重新提取带时间戳的数据；EasyICU 已保存选择并保持分析暂停。',
          ));
          host.render();
        }
      } catch (error) {
        host.setBusy(false);
        host.setError(host.errorText(error));
        host.render();
      }
    }

    function governedNextChoiceGrants(element, message) {
      const projected = String((element && element.dataset && element.dataset.gpiNextGrants) || '')
        .split(',').map(value => value.trim()).filter(Boolean);
      const projectedAllowlist = new Set(['extract', 'configure']);
      if (projected.length && projected.every(value => projectedAllowlist.has(value))) return projected;
      const planGrants = nextActions && typeof nextActions.governedPlanGrants === 'function'
        ? nextActions.governedPlanGrants(message, workflowCode())
        : [];
      return planGrants.length ? planGrants : null;
    }

    return Object.freeze({
      confirmDecision,
      confirmWorkflow,
      continueSystemOwnedPlanRevision,
      governedNextChoiceGrants,
      regenerationAuthority,
      rejectWorkflow,
      resubmitHostGenerated,
      retryFailedExecution,
      startFormalPlanGeneration,
      submitReview,
    });
  }

  window.EU_GUIDED_PI_PLAN_ACTIONS = Object.freeze({ create });
})();
