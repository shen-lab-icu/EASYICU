/* Owner: the conversation's block for a run that stopped before finishing.

   One block per run, at the place the run-files owner gave that run in the
   timeline (screens-guided-pi-run-files.js projectTimeline lists each run
   once). The block says why the run stopped in one readable sentence from
   the error-text owner. Typed stop choices enter existing workflow owners;
   they never create execution authority. Historical stops offer only their
   receipt. The run id stays in the run record; this block names the
   run only by its opaque run_ref. */
(function () {
  'use strict';

  const { esc } = window.EU_HTML;
  const STOP_STATUSES = new Set(['failed', 'cancelled', 'interrupted']);
  // A blocked gate that waits for review is not a stop (aside.runFacts).
  const FAILURE_GATE = /^research_pipeline_|^data_foundation_blocked$|^research_agent_pipeline_failed_closed$/;

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
  function actionRoute(action, workflow) {
    const params = action && action.params || {};
    const kind = action && action.kind;
    if (!action || !/^[a-z_]+_[a-f0-9]{16}$/.test(action.action_id || '')
      || !String(action.action_id).startsWith(`${kind}_`)
      || action.label_key !== `stop_next_actions.${kind}`) return '';
    const code = String(workflow && workflow.next_action_code || '');
    if (kind === 'retry' && params.route === 'retry') {
      return code === 'failed_pipeline_execution_retry_available' ? 'retry' : '';
    }
    if (['edit_plan', 'rerun_without_element', 'use_alternative_concept', 'drop_variable'].includes(kind)
      && params.route === 'plan_edit' && params.intent === 'candidate_plan'
      && params.planner_start_mode === 'fresh' && params.requires_explicit_choice === true
      && params.requires_plan_approval === true) {
      if (kind === 'edit_plan' && ['failed_pipeline_requires_fresh_plan', 'failed_pipeline_execution_retry_futile'].includes(code)) return 'confirm';
      return 'edit';
    }
    if (kind === 'reexport_data' && params.route === 'data_extraction' && params.requires_explicit_choice === true) return 'data';
    if (kind === 'open_provider_settings' && params.route === 'provider_settings') return 'settings';
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

  function create(deps) {
    const tr = deps.tr;
    const runFailureText = typeof deps.runFailureText === 'function' ? deps.runFailureText : () => '';

    function actions(run) {
      const labels = window.EU_PRODUCT_LABELS;
      if (!labels || typeof labels.stopAction !== 'function') return '';
      const workflow = typeof deps.workflow === 'function' ? deps.workflow() : {};
      const latest = typeof deps.latestRun === 'function' ? deps.latestRun() : null;
      const live = currentRun(run, workflow, latest) && typeof deps.canAct === 'function' && deps.canAct();
      const controls = [];
      if (live) {
        const seen = new Set();
        for (const action of offeredActions(run, workflow)) {
          const route = actionRoute(action, workflow);
          if (!route || seen.has(route)) continue;
          seen.add(route);
          if (route === 'data' || route === 'settings') {
            controls.push(`<a class="btn sm" href="#${route === 'data' ? 'extraction' : 'settings'}">${esc(labels.stopAction(action.kind, tr))}</a>`);
          } else {
            const label = route === 'retry' ? 'retry' : action.kind === 'edit_plan' && (route === 'confirm' || workflow.next_action_code === 'failed_pipeline_execution_retry_available') ? 'edit_plan' : 'plan_edit';
            controls.push(`<button class="btn sm" type="button" data-gpi-stop-action="${esc(action.action_id)}" data-gpi-stop-run="${esc(run.run_id)}">${esc(labels.stopAction(label, tr))}</button>`);
          }
        }
        if (!controls.length && typeof deps.hasRecovery === 'function' && deps.hasRecovery()) {
          controls.push(`<button class="btn sm" type="button" data-gpi-stop-recovery>${esc(labels.stopAction('review', tr))}</button>`);
        }
      }
      const receipt = (run.artifact_refs || []).find(ref => ref && ref.run_id === run.run_id && ref.artifact === 'source_run_manifest.json')
        || ((run.artifact_names || []).includes('source_run_manifest.json') ? { kind: 'research_artifact', run_id: run.run_id, artifact: 'source_run_manifest.json', media_type: 'application/json' } : null);
      if (receipt && typeof deps.resourceButton === 'function' && typeof deps.canPreview === 'function' && deps.canPreview()) {
        controls.push(deps.resourceButton(receipt, labels.stopAction('receipt', tr)));
      }
      return controls.filter(Boolean).length ? `<div class="gpi-run-stop-actions">${controls.join('')}</div>` : '';
    }

    function cause(run) {
      const status = String(run.run_status || '').toLowerCase();
      const code = gateReason(run) || String(run.error_code || '')
        || (status === 'cancelled' ? 'research_pipeline_cancelled' : '');
      const detail = run.gate_detail && typeof run.gate_detail === 'object' ? run.gate_detail : {};
      const text = code ? String(runFailureText(code, {
        code: detail.reason_code || run.gate_detail_code,
        missing: detail.missing_concepts || run.gate_missing_concepts,
        cause: detail.cause_code || run.gate_detail_cause_code,
        step: detail.step_id || run.gate_detail_step_id,
        stepTitle: detail.step_title || run.gate_detail_step_title,
        transportRetries: detail.transport_retries ?? run.gate_transport_retries,
      }) || '').trim() : '';
      return text || tr('This run stopped before it finished.', '这次运行在完成前停止。');
    }

    function render(row) {
      const runs = (Array.isArray(row && row.savedRuns) ? row.savedRuns : []).filter(stopped);
      return runs.map(run => `<div class="gpi-run-stop" role="note" data-gpi-run-stop="${esc(run.run_ref || '')}"><p>${esc(cause(run))}</p>${actions(run)}</div>`).join('');
    }

    return Object.freeze({ render, stopped });
  }

  window.EasyICU.guidedPi.declare('runStop', { create, stopped, actionRoute, currentRun, offeredActions });
})();
