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
  const AUTOMATIC_PROVIDER_RUN_CODES = new Set([
    'provider_ready_to_generate_plan',
    'plan_execution_upgrade_required',
    'scientific_plan_review_policy_stale',
    'plan_configuration_superseded',
    'plan_scientific_changes_required',
  ]);
  const BARE_CONTINUATION = /^(?:(?:请|麻烦)?\s*(?:继续|开始|往下做|接着做)(?:一下|吧|做|执行|推进)?|(?:please\s+)?(?:continue|proceed|go\s+ahead))(?:[。.!！]?)$/i;

  function create(host) {
    const tr = host.tr;
    const regeneration = host.regeneration;
    const nextActions = host.nextActions;
    const replay = host.replay;
    const startedTransitions = new Set();

    function unavailable() {
      return !host.session() || host.busy() || host.sessionIsStale();
    }

    function workflowCode() {
      return String((host.workflow() && host.workflow().next_action_code) || '');
    }

    function transitionKey(reasonCode) {
      const session = host.session() || {};
      const binding = session.binding || {};
      // Host-action persistence may advance the study revision even when the
      // scientific configuration and reviewed run are unchanged.  A failed
      // candidate-to-package upgrade must therefore stay deduplicated by its
      // exact source run; otherwise each bookkeeping revision starts the same
      // failed job again. Initial plan generation still uses the revision
      // because it has no source run coordinate yet.
      const revisionCoordinate = reasonCode === 'provider_ready_to_generate_plan'
        || reasonCode === 'agent_plan_configuration_required'
        ? String(binding.study_revision || '')
        : '';
      return [
        String(session.session_id || ''),
        String(binding.study_context_id || ''),
        revisionCoordinate,
        String(binding.run_id || ''),
        String(reasonCode || ''),
      ].join(':');
    }

    function latestArchivedAgentRunFailed() {
      const session = host.session() || {};
      const rows = Array.isArray(session.archived_child_jobs)
        ? session.archived_child_jobs : [];
      let latest = null;
      rows.forEach(row => {
        if (!row || String(row.kind || '') !== 'agent-run') return;
        if (!latest || Number(row.created_at_epoch || 0) >= Number(latest.created_at_epoch || 0)) {
          latest = row;
        }
      });
      return Boolean(latest && ['failed', 'cancelled'].includes(String(latest.status || '')));
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
        && !retryExecution
        && !executionUpgrade;
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
      if (unavailable()) return false;
      const automatic = Boolean(options && options.automatic);
      const request = generationRequest(String(reasonCode || ''));
      const session = host.session() || {};
      const binding = session.binding || {};
      const provider = session.research_provider || {};
      const studyContextId = String(binding.study_context_id || '').trim();
      const revisingScientificPlan = reasonCode === 'plan_scientific_changes_required';
      const executionUpgrade = reasonCode === 'plan_execution_upgrade_required';
      const retryingFailedPlan = reasonCode === 'failed_pipeline_requires_fresh_plan';
      const staleScientificPolicy = reasonCode === 'scientific_plan_review_policy_stale';
      // Both a plan-owned revision and a candidate-to-package upgrade must be
      // bound to the exact reviewed run.  The server distinguishes the two by
      // the digest-verified scientific review: non-approvable reviews produce
      // a bounded repair contract, while approvable metadata-only plans grant
      // only their exact materialization roster.
      const revisionSourceRunId = revisingScientificPlan || executionUpgrade
        ? String(binding.run_id || '').trim()
        : '';
      // A user- or agent-initiated transition consumes only this exact
      // session/revision/run coordinate. A page can host several studies, so a
      // process-wide boolean would incorrectly suppress later conversations.
      const guardedTransition = [
        'provider_ready_to_generate_plan',
        'plan_scientific_changes_required',
        'plan_execution_upgrade_required',
        'scientific_plan_review_policy_stale',
      ].includes(String(reasonCode || ''));
      const guardKey = transitionKey(reasonCode);
      if (guardedTransition) startedTransitions.add(guardKey);
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
        if (guardedTransition) startedTransitions.delete(guardKey);
        return false;
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
          // The natural research request authorizes evidence gathering for the
          // candidate plan. Dropping this flag made every Web revision repeat
          // the same "no direct evidence search" finding.
          literature_search_authorized: true,
          engine: 'research_agent_pipeline',
          // A scientific revision is not a fresh, context-free plan.  Bind the
          // immutable review that requested changes so Planner can repair its
          // own findings instead of rediscovering them on every user click.
          planner_start_mode: reasonCode === 'planner_checkpoint_resume_available'
            ? 'resume_checkpoint'
            : revisingScientificPlan || executionUpgrade || retryingFailedPlan || staleScientificPolicy
              ? 'auto'
              : 'fresh',
          plan_revision_source_run_id: revisionSourceRunId,
        });
        await host.recordHostAction(
          automatic && !executionUpgrade
            ? reasonCode === 'provider_ready_to_generate_plan'
              ? 'auto_generate_plan'
              : 'auto_revise_plan'
            : executionUpgrade
            ? 'prepare_analysis_data'
            : 'generate_plan',
          String(payload.job_id || ''),
          String(payload.job_id || ''),
        );
        host.setBusy(false);
        host.watchChildJob(
          String(payload.job_id || ''),
          executionUpgrade
            ? 'easyicu_full_run_upgrade_submitted'
            : 'easyicu_full_run_submitted',
        );
        return true;
      } catch (error) {
        if (guardedTransition) startedTransitions.delete(guardKey);
        host.setBusy(false);
        host.setError(host.errorText(error));
        host.render();
        return false;
      }
    }

    async function compileAgentPlanConfiguration() {
      if (unavailable()) return false;
      const actionCode = 'agent_plan_configuration_required';
      const guardKey = transitionKey(actionCode);
      if (startedTransitions.has(guardKey)) return false;
      const session = host.session() || {};
      const binding = session.binding || {};
      const expectedRevision = Number(binding.study_revision || 0);
      const runId = String(binding.run_id || '').trim();
      const api = host.api();
      if (
        !expectedRevision
        || !runId
        || typeof api.applyPiCopilotAgentPlanConfiguration !== 'function'
      ) return false;
      startedTransitions.add(guardKey);
      setPending(true);
      try {
        const payload = await api.applyPiCopilotAgentPlanConfiguration(
          session.session_id,
          {
            project_id: host.projectId(),
            expected_revision: expectedRevision,
            run_id: runId,
          },
        );
        await host.refreshSession(true);
        await host.loadWorkflow();
        host.setBusy(false);
        host.render();
        if (payload && payload.next_action === 'fresh_plan') {
          return startFormalPlanGeneration(
            'plan_configuration_superseded', {automatic: true},
          );
        }
        return true;
      } catch (error) {
        startedTransitions.delete(guardKey);
        host.setBusy(false);
        host.setError(host.errorText(error));
        host.render();
        return false;
      }
    }

    async function continueSystemOwnedPlanProgression(options = {}) {
      const workflow = host.workflow() || {};
      const actionCode = String(workflow.next_action_code || '');
      if (unavailable()) return false;
      // Opening, restoring, or rebinding a conversation is a read operation.
      // It must never be treated as fresh user authority to start another
      // Provider-backed plan (or a deterministic hop that immediately starts
      // one). Active message/job completion calls omit this passive flag and
      // may continue the already-authorized workflow once.
      if (Boolean(options && options.passive)) return false;
      if (actionCode === 'agent_plan_configuration_required') {
        return compileAgentPlanConfiguration();
      }
      // Automatic continuation is allowed once after the natural request, but
      // never as an automatic retry.  All Provider-backed plan paths share the
      // same failure boundary so reopening a project cannot bounce between
      // several reason-code branches and repeatedly spend on the same question.
      if (
        AUTOMATIC_PROVIDER_RUN_CODES.has(actionCode)
        && latestArchivedAgentRunFailed()
      ) return false;
      if (actionCode === 'provider_ready_to_generate_plan') {
        if (startedTransitions.has(transitionKey(actionCode))) return false;
        return startFormalPlanGeneration(actionCode, {automatic: true});
      }
      if (actionCode === 'plan_execution_upgrade_required') {
        const session = host.session() || {};
        const binding = session.binding || {};
        const reviewedRunId = String(binding.run_id || '').trim();
        const studyContextId = String(binding.study_context_id || '').trim();
        if (
          startedTransitions.has(transitionKey(actionCode))
          // A failed or cancelled package-bound attempt is durable session
          // evidence.  Do not turn a reload/rebind into an unbounded automatic
          // retry loop; an explicit retry remains available after the runtime
          // or architecture defect is repaired.
          || !reviewedRunId
          || !studyContextId
        ) return false;
        return startFormalPlanGeneration(actionCode, {automatic: true});
      }
      if (actionCode === 'scientific_plan_review_policy_stale') {
        if (startedTransitions.has(transitionKey(actionCode))) return false;
        return startFormalPlanGeneration(actionCode, {automatic: true});
      }
      if (actionCode === 'plan_configuration_superseded') {
        if (startedTransitions.has(transitionKey(actionCode))) return false;
        return startFormalPlanGeneration(actionCode, {automatic: true});
      }
      const questions = workflow.plan_review_summary
        && Array.isArray(workflow.plan_review_summary.authorization_questions)
        ? workflow.plan_review_summary.authorization_questions
        : [];
      const plannerOwnedFindings = workflow.plan_review_summary
        && workflow.plan_review_summary.remediation_buckets
        && Array.isArray(workflow.plan_review_summary.remediation_buckets.agent_plan_revision)
        ? workflow.plan_review_summary.remediation_buckets.agent_plan_revision
        : [];
      if (
        actionCode !== 'plan_scientific_changes_required'
        || questions.length
        || !plannerOwnedFindings.length
        || startedTransitions.has(transitionKey(actionCode))
      ) return false;
      return startFormalPlanGeneration(
        'plan_scientific_changes_required', {automatic: true},
      );
    }

    async function continueUserRequestedSystemProgression(text) {
      const message = String(text || '').trim();
      const workflow = host.workflow() || {};
      const questions = workflow.plan_review_summary
        && Array.isArray(workflow.plan_review_summary.authorization_questions)
        ? workflow.plan_review_summary.authorization_questions
        : [];
      const repairs = workflow.plan_review_summary
        && workflow.plan_review_summary.remediation_buckets
        && Array.isArray(workflow.plan_review_summary.remediation_buckets.agent_plan_revision)
        ? workflow.plan_review_summary.remediation_buckets.agent_plan_revision
        : [];
      if (
        !BARE_CONTINUATION.test(message)
        || String(workflow.next_action_code || '') !== 'plan_scientific_changes_required'
        || questions.length
        || !repairs.length
      ) return false;
      host.appendMessage({
        id: 'user-' + Date.now(), role: 'user', text: message, complete: true,
      });
      if (typeof host.setDraft === 'function') host.setDraft('');
      host.render();
      return continueSystemOwnedPlanProgression();
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
        // An edited fallback question may still be present in the composer
        // when the researcher answers through the structured decision card.
        // The card response is already the authoritative answer, so retaining
        // that draft suggests the same decision still needs to be sent.
        if (typeof host.setDraft === 'function') host.setDraft('');
        await host.refreshSession(true);
        await host.loadWorkflow();
        host.setBusy(false);
        host.render();
        if (payload && payload.next_action === 'replan') {
          await startFormalPlanGeneration(
            'plan_configuration_superseded', {automatic: true},
          );
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
      continueSystemOwnedPlanProgression,
      continueUserRequestedSystemProgression,
      governedNextChoiceGrants,
      regenerationAuthority,
      rejectWorkflow,
      resubmitHostGenerated,
      retryFailedExecution,
      startFormalPlanGeneration,
      submitReview,
    });
  }

  window.EasyICU.guidedPi.declare('planActions', { create });
})();
