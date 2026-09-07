/* Guided Copilot child-job monitor owner.

   Owner: watching one background EasyICU job (extraction, preflight, or a
   Research Agent run) and projecting its lifecycle onto the conversation's
   activity timeline. Split out of screens-guided-pi.js, which was hundreds of
   lines past its size ratchet.

   The shared mutable state this cluster reaches crosses the seam as an
   explicit contract rather than as the host's state object: it reads the
   message list and the bound session, and owns exactly two fields -- the
   watched job id and its EventSource. It never sends a conversation turn. */
(function () {
  'use strict';

  function create(host) {
    const tr = host.tr;
    const ACTIVITY = host.activity;
    const upsertActivityStep = host.upsertActivityStep;
    const render = host.render;
    const api = host.api;
    const loadWorkflow = host.loadWorkflow;
    const sessionIsStale = host.sessionIsStale;
    const rebind = host.rebind;
    const refreshSession = host.refreshSession;
    const archiveChildJob = host.archiveChildJob;

    function closeChildSource() {
      if (host.childSource()) { host.childSource().close(); host.setChildSource(null); }
      host.setChildJobId('');
    }
    function runningJobTitle(code) {
      const value = String(code || '').toLowerCase();
      if (value.includes('extraction')) return tr('Extracting and validating study data', '正在提取并验证研究数据');
      if (value.includes('review')) return tr('Running the approved research plan', '正在执行已批准的研究计划');
      if (value.includes('full_run_report_resume')) return tr('Restoring the manuscript and evidence checks', '正在恢复稿件与证据校验');
      if (value.includes('full_run_resume')) return tr('Resuming the approved research run', '正在续跑已批准的科研任务');
      if (value.includes('full_run_upgrade')) return tr('Preparing analysis data and binding the approved plan', '正在准备分析数据并绑定已审阅计划');
      if (value.includes('full_run')) return tr('Generating the research plan', '正在生成研究计划');
      return tr('EasyICU research task is running', 'EasyICU 科研任务正在运行');
    }
    function childActivity(jobId, code) {
      let activity = host.messages().find(row => row.role === 'activity' && row.childJobId === jobId);
      if (activity) return activity;
      const startedAt = Date.now();
      activity = {
        id: 'easyicu-job-' + jobId, role: 'activity', status: 'running',
        startedAt, childJobId: jobId, runningTitle: runningJobTitle(code), steps: [], expanded: true,
      };
      const label = code === 'easyicu_extraction_submitted'
        ? tr('EasyICU data extraction submitted', 'EasyICU 数据提取任务已提交')
        : String(code || '').includes('easyicu_full_run')
          ? String(code || '').includes('report_resume')
            ? tr('Manuscript and evidence validation resumed', '已恢复稿件与证据校验')
            : String(code || '').includes('resume')
            ? tr('Approved research run resumed', '已续跑获批的科研任务')
            : String(code || '').includes('upgrade')
              ? tr('Analysis data preparation submitted', '已提交分析数据准备任务')
            : tr('Research Agent planning submitted', 'Research Agent 规划任务已提交')
          : code === 'easyicu_review_submitted'
            ? tr('Approved plan submitted for analysis', '已批准计划已提交分析')
          : tr('EasyICU preflight submitted', 'EasyICU 预检任务已提交');
      upsertActivityStep(activity, {
        id: 'pipeline-submitted', kind: 'pipeline', step: 'submitted', label,
        status: 'complete', at: startedAt, code: jobId, owner: 'EasyICU',
      });
      host.messages().push(activity);
      return activity;
    }
    function completeRunningPipelineSteps(activity) {
      activity.steps.forEach(step => {
        if (step.kind === 'pipeline' && step.status === 'running') step.status = 'complete';
      });
    }
    function isPlanAttempt(job) {
      if (!job || String(job.kind || '') !== 'agent-run') return false;
      if (job.human_review_pending === true) return true;
      if (String(job.gate_reason_code || '') === 'human_plan_review_required') return true;
      return String(job.status || '') === 'failed'
        && (!Array.isArray(job.artifact_refs) || job.artifact_refs.length === 0);
    }
    function supersedeEarlierPlanAttempts(job) {
      if (!isPlanAttempt(job) || job.human_review_pending !== true) return;
      const createdAt = Number(job.created_at_epoch);
      if (!Number.isFinite(createdAt)) return;
      const messages = host.messages();
      for (let index = messages.length - 1; index >= 0; index -= 1) {
        const row = messages[index];
        if (
          row && row.role === 'activity' && row.childJobPlanAttempt === true
          && String(row.childJobId || '') !== String(job.job_id || '')
          && Number(row.childJobCreatedAt) < createdAt
        ) messages.splice(index, 1);
      }
    }
    function childEventLabel(event) {
      return ACTIVITY.pipelineEventLabel(event);
    }
    function childEventKind(event) {
      const message = String(event && (event.message || event.label) || '').toLowerCase();
      return String(event && event.step || '').toLowerCase() === 'planning'
        && (String(event && event.retry_phase || '') === 'rejected'
          || message.includes('did not satisfy the scientific contract'))
        ? 'retry' : 'pipeline';
    }
    function childEventId(step, kind, event) {
      if (kind !== 'retry') return 'pipeline-' + step;
      const attempt = Number(event && event.current);
      const unit = String(event && event.planning_unit || 'plan').replace(/[^a-z0-9_-]/gi, '').slice(0, 24) || 'plan';
      return `pipeline-plan-retry-${unit}-${Number.isFinite(attempt) ? attempt : 'current'}`;
    }
    function handleChildJobEvent(jobId, code, event) {
      if (!event || typeof event !== 'object' || host.childJobId() !== jobId) return;
      const activity = childActivity(jobId, code);
      if (event.type === 'end') {
        completeRunningPipelineSteps(activity);
        const gate = event.result && event.result.gate;
        const pending = Boolean(event.result && event.result.human_review_pending);
        const replayOwner = window.EasyICU.guidedPi.require('replay');
        const errorCode = String(event.error || '').split(':', 1)[0].trim();
        const presentation = replayOwner && typeof replayOwner.childJobPresentation === 'function'
          ? replayOwner.childJobPresentation({
            status: event.status,
            error_code: errorCode,
            gate_status: gate && gate.status,
            gate_reason_code: gate && gate.reason,
            human_review_pending: pending,
          }, tr) : {};
        const failed = event.status === 'failed' || event.status === 'cancelled';
        const blocked = Boolean(presentation.blocked);
        const label = presentation.terminalLabel || (event.status === 'cancelled'
          ? tr('EasyICU research task cancelled', 'EasyICU 科研任务已取消')
          : event.status === 'failed'
            ? tr('EasyICU research task failed', 'EasyICU 科研任务失败')
            : pending
              ? tr('Analysis paused for plan review', '分析已暂停，等待计划审核')
              : gate && gate.reportable === false
                ? tr('Analysis finished; the scientific gate remains locked', '分析已结束；科学闸门仍保持锁定')
                : tr('EasyICU research task completed', 'EasyICU 科研任务已完成'));
        upsertActivityStep(activity, {
          id: 'pipeline-terminal', kind: 'pipeline', step: 'terminal', label,
          status: failed || blocked ? 'error' : 'complete', at: Date.now(),
          code: String((gate && (gate.reason || gate.status)) || errorCode || event.status || ''),
          owner: String((event.result && event.result.run_id) || ''),
        });
        activity.status = blocked ? 'blocked' : (failed ? (event.status === 'cancelled' ? 'cancelled' : 'error') : 'complete');
        if (presentation.title) activity.displayTitle = presentation.title;
        activity.endedAt = Date.now();
        closeChildSource();
        archiveChildJob(jobId)
          .catch(() => null)
          .then(() => refreshSession(true))
          .then(async () => {
            if (host.session() && sessionIsStale()) await rebind();
            await loadWorkflow();
            // Successful plan stages may advance automatically. A failed or
            // cancelled background job must stop here: immediately replaying
            // the same transition hides the failure and can consume provider
            // calls indefinitely.
            const continued = event.status === 'done'
              && typeof host.continueSystemOwnedPlanProgression === 'function'
              ? await host.continueSystemOwnedPlanProgression()
              : false;
            if (!continued) render();
          })
          .catch(() => render());
        return;
      }
      if (!['start', 'progress', 'gate', 'artifact', 'cancel_requested'].includes(String(event.type || ''))) return;
      completeRunningPipelineSteps(activity);
      const step = String(event.step || event.type || 'pipeline').slice(0, 80);
      const kind = childEventKind(event);
      upsertActivityStep(activity, {
        // One row per pipeline step, updated in place. Keying on `seq` gave a
        // fresh row for every event, so a single step reported four times
        // (started / generating / running / complete) and 13 steps filled the
        // timeline with 52 near-identical lines.
        id: childEventId(step, kind, event),
        kind, step, label: childEventLabel(event), status: 'running',
        at: Date.now(), code: step,
        owner: String(event.run_id || '').slice(0, 160),
      });
      render();
    }
    function watchChildJob(jobId, code) {
      if (host.childJobId() === jobId && host.childSource()) return;
      closeChildSource();
      host.setChildJobId(jobId);
      childActivity(jobId, code);
      host.setChildSource(new EventSource('/api/jobs/' + encodeURIComponent(jobId) + '/events'));
      let ended = false;
      host.childSource().onmessage = event => {
        let row = null; try { row = JSON.parse(event.data); } catch (e) { return; }
        if (row.type === 'end') ended = true;
        handleChildJobEvent(jobId, code, row);
      };
      host.childSource().onerror = () => {
        if (ended || host.childJobId() !== jobId) return;
        const activity = childActivity(jobId, code);
        completeRunningPipelineSteps(activity);
        upsertActivityStep(activity, {
          id: 'pipeline-stream-error', kind: 'pipeline', step: 'event_stream',
          label: tr('Live progress connection stopped; the server task may still be running', '实时进度连接已中断；服务端任务可能仍在运行'),
          status: 'error', at: Date.now(),
        });
        activity.status = 'error'; activity.endedAt = Date.now();
        closeChildSource(); render();
      };
      render();
    }
    async function cancelChildJob(jobId) {
      const activeId = String(host.childJobId() || '');
      const requestedId = String(jobId || '');
      if (!requestedId || requestedId !== activeId) return false;
      const activity = childActivity(requestedId, '');
      if (activity.cancelRequested) return false;
      activity.cancelRequested = true;
      render();
      try {
        const result = await api().cancelJob(requestedId, 'user_requested_from_copilot');
        if (result && result.cancel_request_accepted === false && result.status === 'running') {
          throw new Error(tr('This task could not be stopped.', '当前任务无法停止。'));
        }
        return true;
      } catch (error) {
        activity.cancelRequested = false;
        render();
        throw error;
      }
    }
    function hydrateProjectedJob(job) {
      if (!job || !job.present || !job.job_id || !host.session()) return;
      const jobId = String(job.job_id);
      supersedeEarlierPlanAttempts(job);
      const activity = childActivity(jobId, String(job.kind || ''));
      activity.childJobPlanAttempt = isPlanAttempt(job);
      activity.childJobCreatedAt = Number(job.created_at_epoch);
      const replayOwner = window.EasyICU.guidedPi.require('replay');
      const presentation = replayOwner && typeof replayOwner.childJobPresentation === 'function'
        ? replayOwner.childJobPresentation(job, tr) : {};
      activity.expanded = activity.status === 'running' || Boolean(presentation.expanded);
      activity.durationKnown = Boolean(presentation.durationKnown);
      if (presentation.startedAt != null) activity.startedAt = presentation.startedAt;
      if (presentation.endedAt != null) activity.endedAt = presentation.endedAt;
      if (presentation.title) activity.displayTitle = presentation.title;
      const progress = Array.isArray(job.progress) ? job.progress : [];
      progress.forEach(event => {
        if (String(event.type || '') === 'end') return;
        const step = String(event.step || event.type || 'pipeline').slice(0, 80);
        const kind = childEventKind(event);
        const count = event.current != null && event.total != null ? `${event.current}/${event.total}` : '';
        const reason = String(event.reason_code || '');
        upsertActivityStep(activity, {
          id: childEventId(step, kind, event),
          kind, step,
          // Replays must use the same safe public projection as live SSE.
          // Persisted runner prose is an audit receipt, not conversation copy.
          label: childEventLabel(event),
          status: ['failed', 'cancelled', 'error'].includes(String(event.status || '')) ? 'error' : 'complete',
          at: Date.now(), code: [count, reason].filter(Boolean).join(' · '), owner: String(job.kind || 'EasyICU'),
        });
      });
      const settled = ['done', 'failed', 'cancelled'].includes(String(job.status || ''));
      if (settled) {
        const blocked = Boolean(presentation.blocked);
        completeRunningPipelineSteps(activity);
        upsertActivityStep(activity, {
          // Match the live event id so replay replaces the row instead of
          // appending a second terminal state after session refresh.
          id: 'pipeline-terminal', kind: 'pipeline', step: 'terminal',
          label: presentation.terminalLabel || (job.status === 'done'
            ? tr('EasyICU research task completed', 'EasyICU 科研任务已完成')
            : job.status === 'cancelled'
              ? tr('EasyICU research task cancelled', 'EasyICU 科研任务已取消')
              : tr('EasyICU research task failed', 'EasyICU 科研任务失败')),
          status: job.status === 'done' && !blocked ? 'complete' : 'error', at: Date.now(),
          code: String(job.gate_reason_code || job.gate_status || job.error_code || job.status || ''), owner: String(job.kind || 'EasyICU'),
          resources: Array.isArray(job.artifact_refs) ? job.artifact_refs : [],
        });
        activity.status = blocked ? 'blocked' : (job.status === 'done' ? 'complete' : (job.status === 'cancelled' ? 'cancelled' : 'error'));
        if (presentation.endedAt == null) activity.endedAt = Date.now();
      }
    }

    return {
      closeChildSource,
      childActivity,
      handleChildJobEvent,
      watchChildJob,
      cancelChildJob,
      hydrateProjectedJob,
    };
  }

  window.EasyICU.guidedPi.declare('childJob', { create });
})();
