/* Owner: the conversation's block for a run that stopped before finishing.

   One block per run, at the place the run-files owner gave that run in the
   timeline (screens-guided-pi-run-files.js projectTimeline lists each run
   once). The block says why the run stopped in one readable sentence from
   the error-text owner. Typed stop choices enter existing workflow owners;
   they never create execution authority. Historical stops retain their
   receipt and diagnostics. The current failed launch can have choices
   before it has a run record. */
(function () {
  'use strict';

  const { esc } = window.EU_HTML;
  const STOP_STATUSES = new Set(['failed', 'cancelled', 'interrupted']);
  // A blocked gate that waits for review is not a stop (aside.runFacts).
  const FAILURE_GATE = /^research_pipeline_|^data_foundation_blocked$|^research_agent_pipeline_failed_closed$/;

  const PLAN_RETRY_CODES = new Set([
    'failed_pipeline_requires_fresh_plan',
    'provider_ready_to_generate_plan',
    'planner_checkpoint_resume_available',
    'plan_execution_upgrade_required',
  ]);

  function gateReason(run) {
    return String(run && (run.gate_reason || run.gate_reason_code) || '');
  }
  function stopped(run) {
    if (!run || typeof run !== 'object') return false;
    const status = String(run.run_status || '').toLowerCase();
    if (STOP_STATUSES.has(status)) return true;
    return (status === 'blocked' || String(run.gate_status || '') === 'blocked') && FAILURE_GATE.test(gateReason(run));
  }

  // Shared by rendering and click-time validation. A stale control cannot
  // act on the new run that has replaced the one it was rendered for.
  function actionRoute(action, workflow, job = null) {
    const params = action && action.params || {};
    const kind = action && action.kind;
    if (!action || !/^[a-z_]+_[a-f0-9]{16}$/.test(action.action_id || '')
      || !String(action.action_id).startsWith(`${kind}_`)
      || action.label_key !== `stop_next_actions.${kind}`) return '';
    const code = String(workflow && workflow.next_action_code || '');
    if (kind === 'retry' && params.route === 'retry') {
      if (!job && params.stage === 'execution' && code === 'failed_pipeline_execution_retry_available') return 'retry';
      if (params.stage !== 'planning') return '';
      if (job && code !== 'provider_ready_to_generate_plan') return 'draft';
      return PLAN_RETRY_CODES.has(code) || code === 'failed_pipeline_execution_retry_futile' ? 'confirm' : '';
    }
    if (['edit_plan', 'rerun_without_element', 'use_alternative_concept', 'drop_variable'].includes(kind)
      && params.route === 'plan_edit' && params.intent === 'candidate_plan'
      && params.planner_start_mode === 'fresh' && params.requires_explicit_choice === true
      && params.requires_plan_approval === true) {
      if (kind !== 'edit_plan' || params.concept || params.element || params.step_id || params.change_question === true) return 'draft';
      if (job && code.startsWith('failed_pipeline_')) return 'draft';
      return ['failed_pipeline_requires_fresh_plan', 'failed_pipeline_execution_retry_futile'].includes(code) ? 'confirm' : 'edit';
    }
    if (kind === 'edit_study' && params.route === 'study_edit'
      && ['cohort', 'outcome', 'source', 'source_or_question'].includes(params.field)) return 'draft';
    if (kind === 'rephrase_in_scope' && params.route === 'question_edit'
      && params.scope_owner === 'easyicu.webserver.study_family_scope') return 'draft';
    if (kind === 'check_connection' && params.route === 'model_connection'
      && !window.EasyICU.guidedPi.optional('hosted')?.active()) return 'connect';
    if (kind === 'reexport_data' && params.route === 'data_extraction' && params.requires_explicit_choice === true) return 'data';
    return '';
  }

  function currentRun(run, workflow, latest) {
    const authoritative = (workflow && workflow.runs || []).find(row => row && row.authoritative === true);
    const id = String(authoritative && authoritative.run_id || latest && latest.run_id || '');
    return Boolean(id && run && run.run_id === id);
  }

  function offeredActions(run, workflow) {
    return Array.isArray(workflow && workflow.next_actions) ? workflow.next_actions
      : Array.isArray(run && run.next_actions) ? run.next_actions : [];
  }

  function jobActions(job, workflow) {
    const reason = gateReason(job) || job.error_code;
    // A matching workflow stop may withdraw or refine the job's offer. A
    // prior run's actions belong to that run and cannot replace this job's.
    const sameStop = workflow?.active_job?.job_id === job.job_id && reason === workflow.next_action_code;
    return offeredActions(job, sameStop ? workflow : null);
  }

  function archivedJobTime(session, job) {
    const replay = session.conversation_replay;
    const turn = replay?.turns?.find(row => row?.kind === 'host_action'
      && row.child_job_id === job.job_id && STOP_STATUSES.has(row.status)
      && Number.isFinite(Date.parse(row.ended_at)));
    // A replay write bounds the age of a bare receipt without its own turn.
    return Date.parse(turn?.ended_at ?? replay?.updated_at);
  }

  function latestJobStop(session, workflow, { credentialVerifiedAt, codexAuthVerified } = {}) {
    if (session?.stale?.stale || workflow?.active_job?.status === 'running'
      || workflow?.next_action_code === 'starting') return null;
    const job = session?.archived_child_jobs?.at(-1);
    if (!job?.job_id || job.run_id || !STOP_STATUSES.has(job.status) || !['agent-run', 'extract'].includes(job.kind)) return null;
    const study = workflow?.study_setup_receipt;
    const binding = session.binding;
    if (binding?.study_context_id && study?.study_context_id
      && (binding.study_context_id !== study.study_context_id || binding.study_revision !== study.revision)) return null;
    const startEpoch = job.created_at_epoch ?? job.finished_at_epoch;
    const started = startEpoch == null ? archivedJobTime(session, job) : startEpoch * 1000;
    const finished = job.finished_at_epoch != null ? job.finished_at_epoch * 1000 : started;
    const newerThan = (date, epoch) => {
      const updated = Date.parse(date);
      return Number.isFinite(updated) && Number.isFinite(epoch) && updated > epoch;
    };
    const apiCredentialChanged = job.kind === 'agent-run' && session.research_provider?.provider !== 'codex'
      && newerThan(credentialVerifiedAt, started);
    const codexAuthRestored = job.kind === 'agent-run' && session.research_provider?.provider === 'codex'
      && (gateReason(job) || job.error_code) === 'research_pipeline_codex_user_auth_required'
      && codexAuthVerified === true;
    if (codexAuthRestored || newerThan(session.created_at, started) || newerThan(study?.updated_at, finished)
      || newerThan(workflow?.bound_export_updated_at, finished) || apiCredentialChanged
      || workflow?.runs?.some(run => run.run_id && newerThan(run.updated_at, started))) return null;
    return job;
  }

  function actionLabel(action, route, workflow, tr) {
    if (route === 'draft') return window.EasyICU.guidedPi.optional('confirmation')?.stopActionCopy(action, tr)?.label;
    const code = workflow?.next_action_code;
    const labels = {
      retry: 'retry',
      data: 'reexport_data',
      connect: 'check_connection',
      confirm: action.kind === 'retry' && PLAN_RETRY_CODES.has(code) ? code : 'edit_plan',
      edit: code === 'failed_pipeline_execution_retry_available' ? 'edit_plan' : 'plan_edit',
    };
    return window.EU_PRODUCT_LABELS.stopAction(labels[route], tr);
  }

  function create(deps) {
    const tr = deps.tr;
    const runFailureText = typeof deps.runFailureText === 'function' ? deps.runFailureText : () => '';

    function actionButtons(actions, workflow, attribute, coordinate, job = null) {
      const controls = [];
      const seen = new Set();
      for (const action of actions) {
        const route = actionRoute(action, workflow, job);
        const key = route === 'draft' ? action.action_id : route;
        if (!route || seen.has(key)) continue;
        const label = actionLabel(action, route, workflow, tr);
        if (!label) continue;
        seen.add(key);
        controls.push(`<button class="btn sm" type="button" data-gpi-stop-action="${esc(action.action_id)}" ${attribute}="${esc(coordinate)}" data-gpi-stop-workflow="${esc(workflow?.next_action_code || '')}">${esc(label)}</button>`);
      }
      return controls;
    }

    function actions(run, hasRecovery) {
      const labels = window.EU_PRODUCT_LABELS;
      if (!labels || typeof labels.stopAction !== 'function') return '';
      const workflow = typeof deps.workflow === 'function' ? deps.workflow() : {};
      const latest = typeof deps.latestRun === 'function' ? deps.latestRun() : null;
      const live = currentRun(run, workflow, latest) && deps.canAct?.();
      const controls = live ? actionButtons(offeredActions(run, workflow), workflow, 'data-gpi-stop-run', run.run_id) : [];
      if (live && !controls.length && hasRecovery) {
        controls.push(`<button class="btn sm" type="button" data-gpi-stop-recovery>${esc(labels.stopAction('review', tr))}</button>`);
      }
      const receipt = (run.artifact_refs || []).find(ref => ref && ref.run_id === run.run_id && ref.artifact === 'source_run_manifest.json')
        || ((run.artifact_names || []).includes('source_run_manifest.json') ? { kind: 'research_artifact', run_id: run.run_id, artifact: 'source_run_manifest.json', media_type: 'application/json' } : null);
      if (receipt && typeof deps.resourceButton === 'function' && typeof deps.canPreview === 'function' && deps.canPreview()) {
        // A button like the stop's other choices, not the inline link style.
        controls.push(deps.resourceButton(receipt, labels.stopAction('receipt', tr), { className: 'btn sm' }));
      }
      // A local diagnostic zip the researcher can send to support (runTrust).
      const registry = window.EasyICU && window.EasyICU.guidedPi;
      const trust = registry && typeof registry.optional === 'function' ? registry.optional('runTrust') : null;
      if (trust && run.run_ref) controls.push(trust.problemButtonHtml(run.run_ref));
      return controls.filter(Boolean).length ? `<div class="gpi-run-stop-actions">${controls.join('')}</div>` : '';
    }

    function cause(run) {
      const status = String(run.run_status || run.status || '').toLowerCase();
      const code = gateReason(run) || String(run.error_code || '')
        || (status === 'cancelled' ? (run.kind === 'extract' ? 'extract_cancelled' : 'research_pipeline_cancelled') : '');
      const detail = run.gate_detail && typeof run.gate_detail === 'object' ? run.gate_detail : {};
      const reason = detail.reason_code || run.gate_detail_code;
      const workflow = deps.workflow?.();
      const coordinateStop = workflow?.next_action_code;
      if (['study_coordinate_unavailable', 'study_coordinate_unrecognized'].includes(coordinateStop)
        && currentRun(run, workflow, deps.latestRun?.())) {
        const missing = workflow.missing_concepts ?? detail.missing_concepts ?? run.gate_missing_concepts;
        const currentCause = runFailureText(coordinateStop, { missing });
        if (currentCause) return currentCause;
      }
      if (reason === 'provider_http_error' || reason === 'provider_timeout'
        || code === 'research_pipeline_provider_timeout') {
        const message = tr(
          'The model service returned an error, so the run stopped. Retry once the connection or service is back.',
          '模型服务返回错误，本次运行已停止。网络或服务恢复后可直接重试。',
        );
        return message + ((detail.runner_kind || run.gate_detail_runner_kind) === 'not_started'
          ? tr(' Analysis has not started.', '分析尚未开始。') : '');
      }
      const text = code ? String(runFailureText(code, {
        code: detail.reason_code || run.gate_detail_code,
        missing: detail.missing_concepts || run.gate_missing_concepts,
        cause: detail.cause_code || run.gate_detail_cause_code,
        mismatchFields: detail.mismatch_fields || run.gate_detail_mismatch_fields,
        statedVariable: detail.stated_variable,
        step: detail.step_id || run.gate_detail_step_id,
        stepTitle: detail.step_title || run.gate_detail_step_title,
        transportRetries: detail.transport_retries ?? run.gate_transport_retries,
      }) || '').trim() : '';
      return text || tr('This run stopped before it finished.', '这次运行在完成前停止。');
    }

    function renderJob(session) {
      const workflow = deps.workflow?.();
      const job = latestJobStop(session, workflow, deps.connectionChecks?.());
      if (!deps.canAct?.() || !job || !window.EU_PRODUCT_LABELS?.stopAction) return '';
      const controls = actionButtons(jobActions(job, workflow), workflow, 'data-gpi-stop-job', job.job_id, job);
      if (!controls.length) return '';
      return `<div class="gpi-run-stop" role="note" data-gpi-job-stop><p>${esc(cause(job))}</p><div class="gpi-run-stop-actions">${controls.join('')}</div></div>`;
    }

    function render(row, { hasRecovery = false } = {}) {
      const runs = (Array.isArray(row && row.savedRuns) ? row.savedRuns : []).filter(stopped);
      return runs.map(run => `<div class="gpi-run-stop" role="note" data-gpi-run-stop="${esc(run.run_ref || '')}"><p>${esc(cause(run))}</p>${actions(run, hasRecovery)}</div>`).join('');
    }

    return Object.freeze({ render, renderJob, stopped });
  }

  window.EasyICU.guidedPi.declare('runStop', { create, stopped, actionRoute, currentRun, offeredActions, jobActions, latestJobStop });
})();
