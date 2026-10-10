/* Owner: Guided Pi plan-action widget. */
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
    // routes/agent.py keeps a plan generated from these metadata-only.
    'population_inclusion_not_applied',
    'question_requirement_not_covered',
    'question_requirement_capability_gap',
    'question_requirements_unreadable',
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
  // Both states describe a failed approved execution; they differ only in
  // whether retrying the failed step could change its outcome.
  const FAILED_EXECUTION_WORKFLOW_CODES = new Set([
    'failed_pipeline_execution_retry_available',
    'failed_pipeline_execution_retry_futile',
  ]);
  const AUTOMATIC_PROVIDER_RUN_CODES = new Set([
    'provider_ready_to_generate_plan',
    'scientific_plan_review_policy_stale',
    'plan_configuration_superseded',
    'plan_scientific_changes_required',
  ]);
  const BARE_CONTINUATION = /^(?:(?:请|麻烦)?\s*(?:继续|开始|往下做|接着做)(?:一下|吧|做|执行|推进)?|(?:please\s+)?(?:continue|proceed|go\s+ahead))(?:[。.!！]?)$/i;

  // Shared by the read-only card and the action boundary. A stopped automatic
  // loop is not approval, nor permission to discard the reviewed source.
  function canRetryStoppedPlan(workflow) {
    const summary = (workflow || {}).plan_review_summary || {};
    const repairs = (summary.remediation_buckets || {}).agent_plan_revision;
    return (workflow || {}).next_action_code === 'agent_plan_revision_nonconvergent'
      && Boolean(String(summary.run_id || '').trim())
      && Array.isArray(summary.authorization_questions)
      && summary.authorization_questions.length === 0
      && Array.isArray(summary.automatic_revision_blockers)
      && summary.automatic_revision_blockers.length === 0
      && Array.isArray(repairs) && repairs.length > 0;
  }

  function create(host) {
    const tr = host.tr;
    const regeneration = host.regeneration;
    const nextActions = host.nextActions;
    const replay = host.replay;
    const startedTransitions = new Set();

    function unavailable() {
      return !host.session() || host.busy() || host.sessionIsStale()
        || (typeof host.researchSourceReady === 'function' && !host.researchSourceReady());
    }

    function workflowCode() {
      return String((host.workflow() && host.workflow().next_action_code) || '');
    }

    // The decision a job-starting button answers, as the host's projection
    // offers it. The host starts one job per decision of the study, refuses a
    // stale one, and records the click in this conversation itself
    // (webserver/host_action_jobs.py); a page without the projection submits
    // without it, as before.
    function hostAction(actionCode, family) {
      const session = host.session() || {};
      if (!session.session_id || typeof host.workflow !== 'function') return null;
      const decisions = (host.workflow() || {}).host_decisions || {};
      const decision = decisions[family];
      if (!decision) return null;
      return {
        session_id: String(session.session_id),
        project_id: String((typeof host.projectId === 'function' && host.projectId()) || ''),
        action_code: actionCode,
        decision,
      };
    }

    // A reopened page can meet a decision whose job the host is still
    // starting; follow the projection until the job exists or the start ends.
    // A repaint replaces the log, closing what the researcher opened and
    // moving the view, so a poll repaints only when the start moved on.
    const STARTING_REFRESH_MS = 3000;
    let startingRefresh = null;
    function startingState() {
      const workflow = host.workflow() || {};
      return `${workflowCode()}|${String(workflow.starting && workflow.starting.started_at || '')}`;
    }
    function followStartingDecision() {
      if (startingRefresh || typeof host.loadWorkflow !== 'function') return;
      startingRefresh = setTimeout(async () => {
        startingRefresh = null;
        const before = startingState();
        try { await host.loadWorkflow(); } catch (error) { return; }
        if (startingState() !== before) host.render(true);
        if (workflowCode() === 'starting') followStartingDecision();
      }, STARTING_REFRESH_MS);
    }

    // The host's answer to a repeated or stale decision is the study's state,
    // not a failure of this click: reload the projection so the page shows
    // it. A decision still starting needs no banner.
    function followHostDecisionRefusal(error) {
      const code = String(error && error.code || '');
      if (!['host_action_in_progress', 'study_job_running', 'host_action_decision_stale'].includes(code)) return false;
      if (typeof host.loadWorkflow === 'function') {
        void Promise.resolve(host.loadWorkflow()).then(() => {
          host.render(true);
          if (workflowCode() === 'starting') followStartingDecision();
        }).catch(() => null);
      }
      if (code !== 'host_action_in_progress') return false;
      host.render();
      return true;
    }

    function automaticRevisionBlocked() {
      const summary = (host.workflow() || {}).plan_review_summary || {};
      return Array.isArray(summary.automatic_revision_blockers)
        && summary.automatic_revision_blockers.length > 0;
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
        String(reasonCode === 'agent_plan_revision_nonconvergent'
          ? ((host.workflow() || {}).plan_review_summary || {}).run_id || ''
          : binding.run_id || ''),
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

    function retrySourceRunId() {
      // The retry offer is computed from the authoritative latest run, so the
      // retry action must name that same run. The session binding is only a
      // fallback: it keeps the coordinate of the reviewed candidate plan, and
      // the server's retry owner refuses a source whose gate reason is not a
      // failed approved execution.
      const latest = typeof host.latestRun === 'function' ? host.latestRun() : null;
      const projected = String((latest && latest.run_id) || '').trim();
      if (projected) return projected;
      const session = host.session() || {};
      return String((session.binding && session.binding.run_id) || '').trim();
    }

    function generationRequest(reasonCode) {
      const retryExecution = reasonCode === 'failed_pipeline_execution_retry_available';
      const executionUpgrade = reasonCode === 'plan_execution_upgrade_required';
      const resumeCheckpoint = reasonCode === 'planner_checkpoint_resume_available';
      // A causal study whose target trial is approved plans on its data; the
      // request is the ordinary formal generation (routes/agent.py decides).
      const trialPlan = reasonCode === 'target_trial_plan_ready';
      const fresh = reasonCode !== 'provider_ready_to_generate_plan'
        && !trialPlan
        && !resumeCheckpoint
        && !retryExecution
        && !executionUpgrade;
      return {
        text: reasonCode === 'agent_plan_revision_nonconvergent'
          ? tr('Replan once after repair', '修复后重新规划一次')
          : retryExecution
          ? tr('Retry analysis from the failed step', '从失败步骤重试分析')
          : executionUpgrade
            ? tr('Confirm the plan and prepare analysis data', '确认方案并准备分析数据')
          : resumeCheckpoint
            ? tr('Continue generating the candidate research plan', '继续生成候选研究计划')
          : trialPlan
            ? tr('Generate the plan on this study’s data', '按本研究数据生成计划')
          : fresh
            ? tr('Generate a fresh research plan', '重新生成研究计划')
            : tr('Generate the candidate research plan', '生成候选研究计划'),
        grants: ['provider_run'],
        intent: fresh
          ? 'confirm_fresh_plan_generation'
          : reasonCode === 'provider_ready_to_generate_plan' || trialPlan
            ? 'confirm_formal_plan_generation'
            : '',
      };
    }

    async function startFormalPlanGeneration(reasonCode, options = {}) {
      if (unavailable()) return false;
      const automatic = Boolean(options && options.automatic);
      const retryingStoppedPlan = reasonCode === 'agent_plan_revision_nonconvergent';
      if (retryingStoppedPlan && (automatic || !canRetryStoppedPlan(host.workflow()))) return false;
      // A newly generated candidate is not a reviewed plan. The existing
      // confirmation action owns this transition; job completion cannot
      // manufacture approval to prepare its data package.
      if (automatic && reasonCode === 'plan_execution_upgrade_required') return false;
      if (automatic && reasonCode === 'plan_scientific_changes_required'
        && automaticRevisionBlocked()) return false;
      const request = generationRequest(String(reasonCode || ''));
      const session = host.session() || {};
      const expectedProjectId = host.projectId();
      const expectedSessionId = session.session_id;
      const selectionRevision = host.selectionRevision ? host.selectionRevision() : null;
      const isCurrent = () => host.projectId() === expectedProjectId
        && host.session() && host.session().session_id === expectedSessionId
        && (!host.selectionRevision || host.selectionRevision() === selectionRevision);
      const binding = session.binding || {};
      const provider = session.research_provider || {};
      const studyContextId = String(binding.study_context_id || '').trim();
      const revisingScientificPlan = reasonCode === 'plan_scientific_changes_required'
        || retryingStoppedPlan;
      const executionUpgrade = reasonCode === 'plan_execution_upgrade_required';
      const retryingFailedPlan = reasonCode === 'failed_pipeline_requires_fresh_plan';
      const replanningFailedExecution = retryingFailedPlan
        && FAILED_EXECUTION_WORKFLOW_CODES.has(workflowCode());
      const staleScientificPolicy = reasonCode === 'scientific_plan_review_policy_stale';
      // Both a plan-owned revision and a candidate-to-package upgrade must be
      // bound to the exact reviewed run.  The server distinguishes the two by
      // the digest-verified scientific review: non-approvable reviews produce
      // a bounded repair contract, while approvable metadata-only plans grant
      // only their exact materialization roster.
      const revisionSourceRunId = retryingStoppedPlan
        ? String(host.workflow().plan_review_summary.run_id).trim()
        : revisingScientificPlan || executionUpgrade || replanningFailedExecution
          ? String(binding.run_id || '').trim()
          : '';
      if (replanningFailedExecution && !revisionSourceRunId) {
        host.setError(tr(
          'The failed plan source is unavailable. Refresh this project before generating a new plan.',
          '失败计划的来源不可用，请刷新项目后再生成新计划。',
        ));
        host.render();
        return false;
      }
      // A user- or agent-initiated transition consumes only this exact
      // session/revision/run coordinate. A page can host several studies, so a
      // process-wide boolean would incorrectly suppress later conversations.
      const guardedTransition = [
        'provider_ready_to_generate_plan',
        'target_trial_plan_ready',
        'plan_scientific_changes_required',
        'plan_execution_upgrade_required',
        'scientific_plan_review_policy_stale',
        'agent_plan_revision_nonconvergent',
      ].includes(String(reasonCode || ''));
      const guardKey = transitionKey(reasonCode);
      if (retryingStoppedPlan && startedTransitions.has(guardKey)) return false;
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
      // Until the host answers with a job the page has nothing to follow or
      // stop; the composer says this plan is starting (session-view).
      const starting = { sessionId: expectedSessionId, startedAt: Date.now() };
      const endStarting = () => {
        if (host.planStarting && host.planStarting() === starting) host.setPlanStarting(null);
      };
      if (host.setPlanStarting) host.setPlanStarting(starting);
      setPending(true);
      let jobStarted = false;
      const stillCurrent = () => {
        if (isCurrent()) return true;
        endStarting();
        // A stale request that has not crossed the job-creation boundary did
        // not consume this transition. Let the same session retry if reopened.
        if (guardedTransition && !jobStarted) startedTransitions.delete(guardKey);
        return false;
      };
      try {
        const response = await api.loadStudyContext(studyContextId);
        if (!stillCurrent()) return false;
        const study = response && (response.context || response.study || response);
        const source = study && study.data_source;
        const sourcePath = String((source && source.path) || '').trim();
        if (!study || !sourcePath) throw new Error('prepared_data_source_unavailable');
        const decision = hostAction(
          automatic && !executionUpgrade
            ? reasonCode === 'provider_ready_to_generate_plan'
              ? 'auto_generate_plan'
              : 'auto_revise_plan'
            : executionUpgrade
            ? 'prepare_analysis_data'
            : 'generate_plan',
          'plan_transition',
        );
        const payload = await api.startAgentRun({
          path: sourcePath,
          study_id: studyContextId,
          study_context_id: studyContextId,
          question: study.question,
          run_type: 'full',
          llm_provider: String(provider.provider || ''),
          credential_source: String(provider.credential_source || ''),
          external_llm_opt_in: true,
          ...(decision ? { host_action: decision } : {}),
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
        jobStarted = true;
        if (!stillCurrent()) return false;
        endStarting();
        host.setBusy(false);
        host.watchChildJob(
          String(payload.job_id || ''),
          executionUpgrade
            ? 'easyicu_full_run_upgrade_submitted'
            : 'easyicu_full_run_submitted',
        );
        return true;
      } catch (error) {
        if (!stillCurrent()) return false;
        if (guardedTransition) startedTransitions.delete(guardKey);
        endStarting();
        host.setBusy(false);
        if (followHostDecisionRefusal(error)) return false;
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
      if (typeof host.setPlanConfigurationError === 'function') host.setPlanConfigurationError('');
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
        // A refused configuration is a researcher decision, not a banner:
        // keep the code so the confirmation card can explain and offer it.
        if (typeof host.setPlanConfigurationError === 'function') {
          host.setPlanConfigurationError(String(error && error.code || 'agent_plan_configuration_failed'));
        } else {
          host.setError(host.errorText(error));
        }
        host.render();
        return false;
      }
    }

    async function continueSystemOwnedPlanProgression(options = {}) {
      const workflow = host.workflow() || {};
      const actionCode = String(workflow.next_action_code || '');
      if (actionCode === 'starting') {
        followStartingDecision();
        return false;
      }
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
        return false;
      }
      if (actionCode === 'scientific_plan_review_policy_stale') {
        if (startedTransitions.has(transitionKey(actionCode))) return false;
        return startFormalPlanGeneration(actionCode, {automatic: true});
      }
      if (actionCode === 'plan_configuration_superseded') {
        if (startedTransitions.has(transitionKey(actionCode))) return false;
        return startFormalPlanGeneration(actionCode, {automatic: true});
      }
      if (!plannerRevisionCanStart()) return false;
      return startFormalPlanGeneration(
        'plan_scientific_changes_required', {automatic: true},
      );
    }

    // Every condition under which the planner-owned revision of a reviewed
    // plan starts. A bare continuation claims the researcher's message only
    // when the revision can start; otherwise (an unconfirmed data source, a
    // failed last run, a revision already started) the message goes to the
    // conversation once, as any other message does.
    function plannerRevisionCanStart() {
      const actionCode = 'plan_scientific_changes_required';
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
      return String(workflow.next_action_code || '') === actionCode
        && !unavailable()
        && !latestArchivedAgentRunFailed()
        && !automaticRevisionBlocked()
        && !questions.length
        && repairs.length > 0
        && !startedTransitions.has(transitionKey(actionCode));
    }

    async function continueUserRequestedSystemProgression(text) {
      const message = String(text || '').trim();
      if (!BARE_CONTINUATION.test(message) || !plannerRevisionCanStart()) return false;
      host.appendMessage({
        id: 'user-' + Date.now(), role: 'user', text: message, complete: true,
      });
      if (typeof host.setDraft === 'function') host.setDraft('');
      host.render();
      // The message is claimed and shown once. A start that then fails reports
      // its own error; it must not also reach the conversation as a copy.
      await continueSystemOwnedPlanProgression();
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
        FAILED_EXECUTION_WORKFLOW_CODES.has(code)
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
      const hostDecision = hostAction('execute_plan', 'plan_review');
      setPending(true);
      try {
        const payload = await api.submitAgentRunReview({
          run_id: runId,
          study_context_id: studyContextId,
          decision,
          external_llm_opt_in: true,
          ...(hostDecision && hostDecision.decision.run_id === runId ? { host_action: hostDecision } : {}),
        });
        host.setBusy(false);
        host.watchChildJob(String(payload.job_id || ''), 'easyicu_review_submitted');
      } catch (error) {
        host.setBusy(false);
        if (followHostDecisionRefusal(error)) return;
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
      const reportOnly = reason === 'report_only';
      const restore = reason === 'restore';
      const validationRepair = reason === 'validation_repair' || reportOnly || restore;
      // A workflow/session refresh can settle while the report request is in
      // flight. Its fallback must retain this action's approved run and route.
      const session = host.session() || {};
      const resumeRunId = retrySourceRunId();
      const hostDecision = hostAction('retry_analysis', 'execution_retry');
      const retryOptions = {
        api: host.api(),
        session: {
          ...session,
          binding: { ...session.binding },
          research_provider: { ...session.research_provider },
        },
        resumeRunId,
        // The decision names the run the projection reads; a retry of any
        // other run submits without it.
        hostAction: hostDecision && hostDecision.decision.source_run_id === resumeRunId
          ? hostDecision : null,
      };
      host.appendMessage({
        id: 'execution-retry-' + Date.now(), role: 'user', complete: true,
        text: restore
          ? tr('Restore the report and its checks using the approved study', '按已批准的研究恢复报告与校验')
          : reportOnly
          ? tr('Repair only the report from sealed results; do not rerun analysis', '只使用封存结果修订报告，不重跑分析')
          : validationRepair
          ? tr('Repair the remaining validation item', '修复剩余校验项')
          : tr('Retry analysis from the failed step', '从失败步骤重试分析'),
      });
      setPending(true);
      try {
        let payload;
        try {
          payload = await replay.retryFailedExecution({
            ...retryOptions, reportOnly: reportOnly || restore,
          });
        } catch (error) {
          // A general Restore action may revalidate the same approved run
          // when old report inputs were not sealed or their aggregate
          // reporting contract needs refresh. Explicit report-only
          // requests never widen scope, and all other failures stay closed.
          const restoreCode = String(error && (error.code || error.message) || '');
          if (!restore || ![
            'WRITER_ONLY_REGISTERED_INPUT_CHANGED',
            'WRITER_ONLY_REPORT_PROJECTION_REFRESH_REQUIRED',
          ].includes(restoreCode)) throw error;
          host.appendMessage({
            id: 'report-recovery-' + Date.now(), role: 'assistant', complete: true,
            text: tr(
              'The saved report checks need restoration. I am revalidating the existing run against its approved plan before continuing the report; the research question and data source stay unchanged.',
              '报告的历史校验记录需要恢复。正在按原批准方案重新核验现有运行，再继续报告；研究问题和数据来源不变。',
            ),
          });
          payload = await replay.retryFailedExecution({
            ...retryOptions, reportOnly: false,
          });
        }
        host.setBusy(false);
        host.watchChildJob(
          String(payload.job_id || ''),
          validationRepair
            ? 'easyicu_full_run_report_resume_submitted'
            : 'easyicu_full_run_resume_submitted',
        );
      } catch (error) {
        host.setBusy(false);
        if (followHostDecisionRefusal(error)) return;
        host.setError(host.errorText(error));
        host.render();
      }
    }

    async function confirmWorkflow(confirmation) {
      if (!confirmation) return;
      if (confirmation.code === 'agent_plan_revision_nonconvergent') {
        if (confirmation.retryPlanRevision) await startFormalPlanGeneration(confirmation.code);
        return;
      }
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
      if (confirmation.code === 'failed_pipeline_execution_retry_futile') {
        // The retry is not offered; the card's only action is the same fresh
        // plan the retry card offers as its alternative.
        await startFormalPlanGeneration('failed_pipeline_requires_fresh_plan');
        return;
      }
      if (confirmation.code === 'agent_plan_configuration_required') {
        // Explicit researcher authority for the deterministic host compile;
        // a passive page open never applies it by itself.
        await compileAgentPlanConfiguration();
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
      if (confirmation.code === 'target_trial_plan_ready') {
        await startFormalPlanGeneration(confirmation.code);
        return;
      }
      if (confirmation.approveTrial) {
        await approveTargetTrial();
        return;
      }
      await host.sendText(confirmation.message, confirmation.grants);
    }

    // The researcher's click on an approvable target trial card, every line
    // ticked: the host records it against the record the card shows
    // (webserver/target_trial_card.py) and writes the conversation's row.
    async function approveTargetTrial() {
      if (unavailable()) return;
      const owner = window.EasyICU.guidedPi.optional('targetTrial');
      const session = host.session() || {};
      const body = owner ? owner.approvalRequest(host.workflow(), session, host.projectId()) : null;
      const api = host.api();
      if (!body || typeof api.approvePiCopilotTargetTrial !== 'function') return;
      setPending(true);
      try {
        await api.approvePiCopilotTargetTrial(session.session_id, body);
        await host.refreshSession(true);
        await host.loadWorkflow();
        host.setBusy(false);
        host.render();
      } catch (error) {
        host.setBusy(false);
        // A stale card is the study's state, not this click's failure: show
        // the card the host has now, with why the click did not approve.
        await Promise.resolve(host.loadWorkflow()).catch(() => null);
        host.setError((owner && owner.refusalText(error, tr)) || host.errorText(error));
        host.render();
      }
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

  window.EasyICU.guidedPi.declare('planActions', { create, canRetryStoppedPlan });
})();
