/* Owner: Guided Copilot live turn stream.
   Projects the Pi SSE events of one running turn into conversation state
   (activity steps, streamed reply text, the model's reasoning rows, tool
   receipts and host follow-ups) and patches streamed text in place so a long
   transcript is not rebuilt for every token. It never authorizes anything:
   host grants and workflow decisions stay with the shell and its owners. */
(function () {
  'use strict';

  function create(host) {
    const {
      state, timeMs, ensureActivity, upsertActivityStep, finishActivity,
      assistantRow, addAssistantResources, completeLatestAssistant,
      activityHasCompletedAction, modelErrorText, activity: ACTIVITY,
      modules: MODULES, projectId, loadWorkflow, render, watchChildJob,
      hostJobs: HOST_JOBS, assistantTextHtml, publicAssistantText, followUps: FOLLOW_UPS,
    } = host;

    function handlePiEvent(event) {
      if (!event || typeof event !== 'object') return;
      const at = timeMs(event.at);
      const activity = ensureActivity(event.at);
      if (event.type === 'run_start') {
        state.currentTurnResources = [];
        upsertActivityStep(activity, { id: 'agent', kind: 'agent', status: 'complete', at });
      } else if (event.type === 'turn_start') {
        ACTIVITY.startTurn(activity, at);
      } else if (event.type === 'assistant_start') {
        const phase = activity.steps.filter(item => item.kind === 'assistant').length + 1;
        upsertActivityStep(activity, { id: 'assistant-' + phase, kind: 'assistant', phase, status: 'running', at, startedAt: at });
      } else if (event.type === 'thinking_start' || event.type === 'thinking_delta' || event.type === 'thinking_end') {
        // The model's reasoning summary is part of the turn's trace: one row
        // per model phase that grows while the model thinks.
        const phase = Math.max(1, activity.steps.filter(item => item.kind === 'assistant').length);
        const id = 'thinking-' + phase;
        let step = activity.steps.find(item => item.id === id);
        if (!step) {
          step = { id, kind: 'thinking', status: 'running', at, startedAt: at, text: '' };
          activity.steps.push(step);
        }
        if (event.type === 'thinking_delta') {
          step.text = (String(step.text || '') + String(event.delta || '')).slice(0, 4000);
          if (patchStreamingThinking(step)) return;
        } else if (event.type === 'thinking_end') {
          step.status = 'complete'; step.endedAt = at;
        }
      } else if (event.type === 'text_delta') {
        const delta = String(event.delta || '');
        const streamingRow = assistantRow();
        // Remember how many tool calls preceded this text segment: the turn
        // layout places interim narration between the trace rows it follows.
        if (streamingRow.afterSteps == null) {
          streamingRow.afterSteps = activity.steps.filter(item => item.kind === 'tool').length;
        }
        streamingRow.text += delta; ACTIVITY.appendPublicDelta(activity, delta);
        // Streamed text is patched in place: rebuilding the whole conversation
        // for every token made long transcripts stutter and reset selections.
        if (patchStreamingMessage(streamingRow)) return;
      } else if (event.type === 'message_end') {
        let row = state.messages.slice().reverse().find(item => item.role === 'assistant' && !item.complete);
        if (event.error_code) {
          row = row || assistantRow();
          row.errorCode = String(event.error_code);
          if (!row.text) row.text = modelErrorText(
            row.errorCode, activityHasCompletedAction(activity),
          );
        }
        completeLatestAssistant(event.stop_reason);
        const step = activity.steps.slice().reverse().find(item => item.kind === 'assistant' && item.status === 'running');
        if (step) { step.status = event.error_code ? 'error' : 'complete'; step.endedAt = at; step.stopReason = event.stop_reason || ''; }
      } else if (event.type === 'tool_start') {
        const assistant = activity.steps.slice().reverse().find(item => item.kind === 'assistant' && item.status === 'running');
        if (assistant) assistant.status = 'complete';
        const thinking = activity.steps.slice().reverse().find(item => item.kind === 'thinking' && item.status === 'running');
        if (thinking) { thinking.status = 'complete'; thinking.endedAt = at; }
        upsertActivityStep(activity, {
          id: 'tool-' + event.tool_call_id, kind: 'tool', toolName: event.tool_name,
          status: 'running', at, startedAt: at, resource: event.resource || null,
        });
      } else if (event.type === 'tool_progress') {
        upsertActivityStep(activity, { id: 'tool-' + event.tool_call_id, kind: 'tool', toolName: event.tool_name, status: 'running', at });
      }
      else if (event.type === 'tool_end') {
        const toolResources = [event.resource].concat(Array.isArray(event.resources) ? event.resources : []).filter(Boolean);
        upsertActivityStep(activity, {
          id: 'tool-' + event.tool_call_id, kind: 'tool', toolName: event.tool_name,
          status: event.is_error ? 'error' : 'complete', code: event.code || '',
          owner: event.owner || '', text: event.summary || '', at, endedAt: at,
          jobId: event.job_id || '',
          resource: event.resource || null,
          resources: Array.isArray(event.resources) ? event.resources : [],
        });
        addAssistantResources(toolResources);
        const localWorkspace = !event.is_error && String(event.code || '') === 'easyicu_local_source_workspace_ready'
          ? toolResources.find(resource => resource && resource.kind === 'native_workspace')
          : null;
        const preview = MODULES.optional('preview');
        if (localWorkspace && preview && preview.open) {
          preview.open(localWorkspace, projectId());
        }
        if (event.host_rebind_after_turn === true || ['study_context_updated', 'easyicu_extraction_submitted', 'easyicu_run_submitted', 'easyicu_full_run_submitted', 'easyicu_report_repair_submitted'].includes(String(event.code || ''))) {
          state.pendingAuthorityRebind = true;
        }
        if (/^(easyicu_(research_workflow_projected|idea_|active_export_reused|extraction_|run_|full_run_|report_repair_|result_|manuscript_))/.test(String(event.code || ''))) {
          loadWorkflow().then(render).catch(() => {});
        }
        if (event.job_id && ['easyicu_extraction_submitted', 'easyicu_run_submitted', 'easyicu_full_run_submitted', 'easyicu_report_repair_submitted'].includes(String(event.code || ''))) {
          watchChildJob(String(event.job_id), String(event.code || ''));
        }
        HOST_JOBS.noteToolResult(event);
      } else if (event.type === 'turn_end') {
        ACTIVITY.finishTurn(activity, at);
      } else if (event.type === 'retry') {
        upsertActivityStep(activity, { id: 'retry-' + event.attempt, kind: 'retry', status: 'running', attempt: event.attempt, maxAttempts: event.max_attempts, at, startedAt: at });
      } else if (event.type === 'compaction_start') {
        upsertActivityStep(activity, { id: 'compaction', kind: 'compaction', status: 'running', at, startedAt: at });
      } else if (event.type === 'compaction_end') {
        upsertActivityStep(activity, { id: 'compaction', kind: 'compaction', status: event.aborted ? 'error' : 'complete', at, endedAt: at });
      } else if (event.type === 'agent_cycle_end' && event.will_retry) {
        const retry = activity.steps.slice().reverse().find(item => item.kind === 'retry' && item.status === 'running');
        if (retry) { retry.status = 'complete'; retry.endedAt = at; }
      } else if (event.type === 'run_end') {
        finishActivity('complete', event.at, 'settled');
      }
      render();
    }
    function patchStreamingThinking(step) {
      if (!state.host || !step || step.status !== 'running') return false;
      const node = state.host.querySelector(`[data-gpi-thinking-stream="${step.id}"]`);
      if (!node) return false;
      node.innerHTML = ACTIVITY.reasoningHtml(step.text);
      const box = node.closest('.gpi-activity-body, .gpi-activity-running');
      const scroller = box ? box.querySelector('ol') : null;
      if (scroller && scroller.scrollHeight - scroller.scrollTop - scroller.clientHeight < 120) scroller.scrollTop = scroller.scrollHeight;
      const log = state.host.querySelector('[data-gpi-log]');
      if (log && log.scrollHeight - log.scrollTop - log.clientHeight < 160) log.scrollTop = log.scrollHeight;
      return true;
    }
    function patchStreamingMessage(row) {
      if (!state.host || !row || row.role !== 'assistant' || row.complete) return false;
      const article = state.host.querySelector(`[data-gpi-streaming-message][data-gpi-message-id="${row.id}"]`);
      if (!article) return false;
      const body = article.querySelector('.gpi-message-body');
      if (!body) return false;
      let text = body.querySelector(':scope > .gpi-text');
      const placeholder = body.querySelector(':scope > .gpi-streaming');
      if (!text) {
        text = document.createElement('div');
        text.className = 'gpi-text';
        if (placeholder) placeholder.replaceWith(text); else body.prepend(text);
      }
      const visible = publicAssistantText(row.text);
      text.innerHTML = assistantTextHtml(FOLLOW_UPS ? FOLLOW_UPS.split(visible).text : visible);
      const log = state.host.querySelector('[data-gpi-log]');
      if (log && log.scrollHeight - log.scrollTop - log.clientHeight < 160) log.scrollTop = log.scrollHeight;
      return true;
    }

    return Object.freeze({ handlePiEvent, patchStreamingMessage });
  }

  window.EasyICU.guidedPi.declare('liveStream', { create });
})();
