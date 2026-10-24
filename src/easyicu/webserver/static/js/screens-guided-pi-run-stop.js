/* Owner: the conversation's block for a run that stopped before finishing.

   One block per run, at the place the run-files owner gave that run in the
   timeline (screens-guided-pi-run-files.js projectTimeline lists each run
   once). The block says why the run stopped in one readable sentence from
   the error-text owner. It offers no button: the action that works (retry,
   a fresh plan, P0-15's checkpoint continuation) is the workflow
   confirmation card's, which offers every choice and whose answer leaves a
   decision record. The run id stays in the run record; this block names the
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

  function create(deps) {
    const tr = deps.tr;
    const runFailureText = typeof deps.runFailureText === 'function' ? deps.runFailureText : () => '';

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
      }) || '').trim() : '';
      return text || tr('This run stopped before it finished.', '这次运行在完成前停止。');
    }

    function render(row) {
      const runs = (Array.isArray(row && row.savedRuns) ? row.savedRuns : []).filter(stopped);
      return runs.map(run => `<div class="gpi-run-stop" role="note" data-gpi-run-stop="${esc(run.run_ref || '')}"><p>${esc(cause(run))}</p></div>`).join('');
    }

    return Object.freeze({ render, stopped });
  }

  window.EasyICU.guidedPi.declare('runStop', { create, stopped });
})();
