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
  ]);

  function create(host) {
    const ACTIVITY = host.activity;
    const upsertActivityStep = host.upsertActivityStep;
    const timeMs = host.timeMs;
    const resourceKey = host.resourceKey;
    const modelErrorText = host.modelErrorText;
    const workflowActionCode = host.workflowActionCode;

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
          });
          activity = {
            id: 'history-activity-' + index, role: 'activity', status: 'running',
            startedAt: rowAt, steps: [],
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
          messages.push({ id: 'history-' + index, role: 'assistant', text: modelErrorText(row.error_code), complete: true, errorCode: row.error_code });
          closeHistoryActivity(rowAt);
        }
      });
      closeHistoryActivity(lastTimestamp);
      const replayOwner = window.EU_GUIDED_PI_REPLAY;
      const replayTurns = replayOwner && typeof replayOwner.lifecycleTurns === 'function'
        ? replayOwner.lifecycleTurns(session) : [];
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
        if (!replayActivity) replayActivity = { id: 'saved-activity-' + String((turn && turn.job_id) || replayStarted), role: 'activity', steps: [], expanded: false };
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
      return ACTIVITY.focusLatest(messages.filter(row => (
        (row.text || row.role === 'activity') && !childJobHandoffReplies.has(row.id)
      )));
    }

    return { transcriptMessages };
  }

  window.EU_GUIDED_PI_TRANSCRIPT = { create };
})();
