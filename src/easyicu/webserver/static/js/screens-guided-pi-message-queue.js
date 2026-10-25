/* Owner: bounded, host-owned follow-ups while a conversation turn is running. */
(function () {
  'use strict';
  function render(queue, { tr, esc }) {
    const items = (queue && queue.items || []).filter(row => ['queued', 'dispatching', 'interrupted'].includes(row.status));
    if (!items.length) return '';
    return `<div class="gpi-followup-queue" role="status" aria-live="polite">${items.map(row => `<div><span><strong>${row.status === 'interrupted' ? tr('Queue interrupted — check the conversation before resending', '排队已中断，请核对对话后重新发送') : tr('Queued for the next reply', '已排队，将在当前回复后发送')}</strong><small>${esc(row.message)}</small></span><button type="button" class="btn sm" data-gpi-cancel-followup="${esc(row.id)}" ${row.status === 'dispatching' ? 'disabled' : ''}>${tr('Remove', '移除')}</button></div>`).join('')}</div>`;
  }
  function create({ state, api, projectId, render: repaint, refreshSession, watchJob, errorText }) {
    let timer = null;
    let checking = false;
    let submitting = false;
    const seen = new Set();
    function stop() { if (timer) clearTimeout(timer); timer = null; }
    function sync() {
      stop();
      const rows = state.session && state.session.followup_queue && state.session.followup_queue.items || [];
      if (state.host && rows.some(row => ['queued', 'dispatching', 'submitted'].includes(row.status) && !seen.has(row.id))) timer = setTimeout(reconcile, 900);
    }
    async function reconcile() {
      if (checking || !state.host || !state.session) return;
      checking = true;
      const sessionId = state.session.session_id;
      const project = projectId();
      const current = () => state.host && state.session && state.session.session_id === sessionId && projectId() === project;
      try {
        const payload = await api().loadPiCopilotFollowups(sessionId, project);
        if (!current()) return;
        const changed = JSON.stringify(state.session.followup_queue) !== JSON.stringify(payload.queue);
        state.session.followup_queue = payload.queue;
        if (!state.busy) {
          const submitted = (payload.queue.items || []).filter(row => row.status === 'submitted' && !seen.has(row.id));
          for (const row of submitted) {
            let job;
            try { job = await api().loadJobSnapshot(row.job_id); }
            catch (error) {
              if (Number(error.status) !== 404) throw error;
              seen.add(row.id);
              await refreshSession(false);
              continue;
            }
            if (!current() || state.busy) return;
            seen.add(row.id);
            await refreshSession(false);
            if (!current()) return;
            if (job.status === 'running' || job.status === 'queued') {
              state.busy = true; state.jobId = row.job_id; watchJob(row.job_id);
              break;
            }
          }
        }
        if (changed || !state.busy) repaint();
      } catch (error) {
        if (current()) { state.error = errorText(error); repaint(); }
      } finally { checking = false; sync(); }
    }
    async function send(text) {
      if (submitting) return;
      submitting = true; state.queueSubmitting = true; repaint();
      const sessionId = state.session.session_id;
      const project = projectId();
      const current = () => state.session && state.session.session_id === sessionId && projectId() === project;
      try {
        const payload = await api().sendPiCopilotMessage(sessionId, {
          project_id: project, message: text, message_origin: 'typed', queue_if_busy: true,
        });
        if (!current()) return;
        if (String(state.draft || '').trim() === text) state.draft = '';
        if (payload.status === 'queued') state.session.followup_queue = payload.queue;
        else { await refreshSession(false); if (!current()) return; state.busy = true; state.jobId = payload.job_id; watchJob(payload.job_id); }
      } catch (error) { if (current()) throw error; }
      finally { submitting = false; state.queueSubmitting = false; repaint(); sync(); }
    }
    async function cancel(itemId) {
      if (!state.session) return;
      const sessionId = state.session.session_id;
      const project = projectId();
      try {
        const payload = await api().cancelPiCopilotFollowup(sessionId, itemId, { project_id: project });
        if (!state.session || state.session.session_id !== sessionId || projectId() !== project) return;
        state.session.followup_queue = payload.queue; repaint(); sync();
      } catch (error) {
        if (state.session && state.session.session_id === sessionId && projectId() === project) { state.error = errorText(error); repaint(); }
      }
    }
    return { send, cancel, reconcile, sync, stop };
  }
  window.EasyICU.guidedPi.declare('messageQueue', { create, render });
}());
