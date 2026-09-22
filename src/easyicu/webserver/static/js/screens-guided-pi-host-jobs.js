/* Owner: Guided Pi host-job follow-through (official demo preparation). */
/* When the model submits a host job that finishes outside the conversation
   turn -- today the official demo download/prepare/register pipeline -- the
   transcript only records "submitted". This owner watches that job, tells the
   researcher when the data are ready, and offers the one confirmation that
   lets the conversation continue: bind the registered export to the study
   and confirm it as this session's data source. It never sends model text
   and never grants analysis; source confirmation stays a host decision.
   When the opening question itself names one official demo ("eICU demo"),
   the data-source card offers that demo directly: one click prepares (or
   re-registers) it, then binds and confirms it -- the researcher's click is
   the source decision, so the model never has to ask which database. */
(function () {
  'use strict';

  const { esc } = window.EU_HTML;
  const NOTICE_ROLE = 'host_notice';
  const DEMO_PREP_TOOL = 'easyicu_prepare_demo_source';
  const DEMO_PREP_CODE = 'easyicu_demo_source_preparation_submitted';
  const DEMO_PREP_JOB_KIND = 'demo-source-prepare';
  const POLL_MS = 3000;
  const DEMO_WORD = '(?:demo|演示|示例)';

  // A catalog demo is named when its title's product token ("eICU",
  // "MIMIC-IV") sits next to a demo word in the researcher's own text.
  function demoNamePattern(demo) {
    const head = String(demo && demo.title || '').trim().split(/\s+/)[0] || '';
    const name = head.toLowerCase().replace(/[^a-z0-9]+/g, '[\\s-]*');
    if (!name) return null;
    return new RegExp(`${name}[^。.!！?？;；\\n]{0,16}${DEMO_WORD}|${DEMO_WORD}[^。.!！?？;；\\n]{0,8}${name}`);
  }

  function create(host) {
    const tr = host.tr;
    const api = host.api;
    const DATA_CONSENT = host.dataConsent;
    const watched = new Map();
    const autoUse = new Set();
    const catalog = { rows: null, loading: false };
    let offerPending = false;

    function sessionId() {
      return String(host.session() && host.session().session_id || '');
    }
    function noticeId(id) { return `demo-source-ready-${id || sessionId()}`; }
    function notices() {
      return host.workflowReceipts().filter(row => row && row.role === NOTICE_ROLE);
    }
    function upsertNotice(row) {
      host.setWorkflowReceipts(host.workflowReceipts().filter(item => item.id !== row.id).concat([row]));
    }
    function removeNotice(id) {
      host.setWorkflowReceipts(host.workflowReceipts().filter(item => item.id !== id));
    }
    function activeSource() {
      const source = window.EU_SOURCES && typeof window.EU_SOURCES.activeSource === 'function'
        ? window.EU_SOURCES.activeSource() : null;
      return source && source.ok !== false && source.path ? source : null;
    }
    async function reloadSources() {
      try {
        if (window.EU_SOURCES && typeof window.EU_SOURCES.reload === 'function') await window.EU_SOURCES.reload();
      } catch (_) { /* the notice falls back to the cached registry */ }
    }
    function requiresConfirmation() {
      return Boolean(host.session() && DATA_CONSENT && DATA_CONSENT.requiresConfirmation(host.session()));
    }

    async function announceReady(jobId) {
      const expectedSession = sessionId();
      await reloadSources();
      if (!expectedSession || sessionId() !== expectedSession) return false;
      const source = activeSource();
      if (!source || !requiresConfirmation()) return false;
      const summary = source.summary && typeof source.summary === 'object' ? source.summary : {};
      upsertNotice({
        id: noticeId(expectedSession), role: NOTICE_ROLE, notice: 'demo_source_ready',
        session_id: expectedSession, job_id: String(jobId || ''),
        source_id: String(source.id || ''), source_label: String(source.label || ''),
        database: String(source.database || ''),
        stays: Number(summary.stays) || 0, modules: Number(summary.modules) || 0,
        pending: false, timelineAt: Date.now(),
      });
      host.render();
      return true;
    }

    function stop(jobId) {
      const timer = watched.get(jobId);
      if (timer) clearInterval(timer);
      watched.delete(jobId);
    }
    function stopAll() {
      watched.forEach(timer => clearInterval(timer));
      watched.clear();
    }

    /* Poll the job snapshot instead of holding a second EventSource: the
       conversation may already stream its own turn, and the prepare job only
       needs its terminal state. */
    function watchDemoSourceJob(jobId) {
      const id = String(jobId || '').trim();
      if (!id || watched.has(id) || typeof api().loadJobSnapshot !== 'function') return;
      const expectedSession = sessionId();
      let ticking = false;
      const tick = async () => {
        if (ticking) return;
        if (sessionId() !== expectedSession) { stop(id); return; }
        ticking = true;
        let snapshot = null;
        try {
          snapshot = await api().loadJobSnapshot(id);
        } catch (_) {
          // An unknown job (server restarted) is resolved from the session's
          // archived jobs and the registry on the next sync instead.
          stop(id); ticking = false; return;
        }
        ticking = false;
        if (sessionId() !== expectedSession) { stop(id); return; }
        const status = String(snapshot && snapshot.status || '');
        if (status === 'done') {
          stop(id);
          const announced = await announceReady(id);
          // A demo the researcher chose from the question card is used as
          // soon as it is registered; the card click was the source decision.
          const row = autoUse.delete(id) && announced ? notices().find(item => item.id === noticeId(expectedSession)) : null;
          offerPending = false;
          if (row) await useDemoSource(row);
          Promise.resolve(host.loadWorkflow()).then(() => host.render()).catch(() => null);
          return;
        }
        if (status === 'failed' || status === 'cancelled') {
          stop(id);
          autoUse.delete(id);
          offerPending = false;
          host.setError(tr('Official demo preparation did not complete: ', '官方 Demo 数据准备未完成：')
            + String(snapshot && snapshot.error || status));
          host.render();
        }
      };
      watched.set(id, setInterval(() => { void tick(); }, POLL_MS));
      void tick();
    }

    function noteToolResult(event) {
      if (!event || event.is_error || String(event.code || '') !== DEMO_PREP_CODE || !event.job_id) return false;
      watchDemoSourceJob(String(event.job_id));
      return true;
    }

    function latestTranscriptPrepJob() {
      const rows = host.messages();
      for (let index = rows.length - 1; index >= 0; index -= 1) {
        const row = rows[index];
        if (!row || row.role !== 'activity' || !Array.isArray(row.steps)) continue;
        const step = row.steps.slice().reverse().find(item => item && item.kind === 'tool'
          && String(item.toolName || '') === DEMO_PREP_TOOL && item.status !== 'error' && item.jobId);
        if (step) return String(step.jobId);
      }
      return '';
    }

    /* Re-derive the follow-through from persisted session state, so a reload
       or a server restart does not lose the "data are ready" step. */
    function sync() {
      const session = host.session();
      if (!session) { stopAll(); return; }
      const id = noticeId(sessionId());
      if (!requiresConfirmation()) {
        if (host.workflowReceipts().some(row => row.id === id)) { removeNotice(id); host.render(); }
        return;
      }
      const jobs = Array.isArray(session.archived_child_jobs) ? session.archived_child_jobs : [];
      const prep = jobs.slice().reverse().find(job => job && job.kind === DEMO_PREP_JOB_KIND);
      if (prep && prep.status === 'done') {
        if (!host.workflowReceipts().some(row => row.id === id)) void announceReady(prep.job_id);
        return;
      }
      if (prep && (prep.status === 'running' || prep.status === 'queued')) {
        watchDemoSourceJob(prep.job_id);
        return;
      }
      const transcriptJob = latestTranscriptPrepJob();
      if (transcriptJob) watchDemoSourceJob(transcriptJob);
    }

    function sourceSnapshot(source) {
      return {
        source_id: String(source.id || ''),
        path: String(source.path || ''),
        label: String(source.label || ''),
        database: String(source.database || ''),
      };
    }

    async function useDemoSource(row) {
      const session = host.session();
      if (!row || !session || host.busy() || row.pending) return;
      const source = activeSource();
      const store = window.EU_STUDY_CONTEXT;
      if (!source || !store || typeof store.update !== 'function' || typeof store.persist !== 'function') {
        host.setError(tr('The prepared demo export is not registered any more. Choose the folder instead.', '已准备的 Demo 导出不再处于注册状态，请改为选择文件夹。'));
        host.render();
        return;
      }
      const expectedSession = sessionId();
      row.pending = true;
      host.setError('');
      host.render();
      try {
        const contextId = String(session.binding && session.binding.study_context_id || '');
        if (typeof store.hydrate === 'function') await store.hydrate({ force: true });
        const active = typeof store.active === 'function' ? store.active() : null;
        if (contextId && (!active || active.id !== contextId) && typeof store.activate === 'function') {
          await store.activate(contextId);
        }
        if (typeof store.refreshActiveFromServer === 'function') await store.refreshActiveFromServer();
        store.update({ data_source: sourceSnapshot(source) }, { persist: false, reason: 'demo-source-binding' });
        const saved = await store.persist();
        if (sessionId() !== expectedSession) return;
        if (DATA_CONSENT.selectionInProgress(host.session())) {
          await host.confirmDataSourceBinding({
            id: 'source-binding-' + Date.now(), receipt_kind: 'data_source_binding',
            database: source.database, source_label: source.label,
            study_context_id: String(saved && saved.id || ''),
            study_revision: Number(saved && saved.revision || 0),
          });
        } else {
          await host.rebind();
          if (sessionId() !== expectedSession) return;
          await host.authorizeDataSource('use_study_required_data');
        }
        if (sessionId() !== expectedSession) return;
        removeNotice(row.id);
        host.render();
      } catch (error) {
        row.pending = false;
        host.setError(host.errorText(error));
        host.render();
      }
    }

    async function loadCatalog() {
      const caller = api().loadOfficialDemoSources;
      if (catalog.rows || catalog.loading || typeof caller !== 'function') return;
      catalog.loading = true;
      try {
        const payload = await caller();
        catalog.rows = Array.isArray(payload && payload.sources) ? payload.sources : [];
      } catch (_) {
        catalog.rows = [];
      } finally {
        catalog.loading = false;
      }
      host.render();
    }

    /* The one official demo the opening question names, while this
       conversation still needs a data source; otherwise null. */
    function namedDemo() {
      if (!host.session() || !requiresConfirmation()) return null;
      if (!catalog.rows) { void loadCatalog(); return null; }
      const first = host.messages().find(row => row && row.role === 'user' && String(row.text || '').trim());
      const text = first ? String(first.text).normalize('NFKC').toLowerCase() : '';
      const hits = text ? catalog.rows.filter(demo => { const pattern = demoNamePattern(demo); return pattern && pattern.test(text); }) : [];
      if (hits.length !== 1) return null;
      const demo = hits[0];
      const status = demo.status && typeof demo.status === 'object' ? demo.status : {};
      return {
        id: String(demo.id || ''), label: [demo.title, demo.version ? `v${demo.version}` : ''].filter(Boolean).join(' '),
        prepared: status.export_ready === true, pending: offerPending,
      };
    }

    async function useNamedDemo(sourceId) {
      const offer = namedDemo();
      const caller = api().startOfficialDemoSourcePrepare;
      if (!offer || offer.id !== sourceId || offerPending || host.busy() || typeof caller !== 'function') return false;
      offerPending = true;
      host.setError('');
      host.render();
      try {
        const payload = await caller(sourceId);
        const jobId = String(payload && payload.job_id || '');
        if (!jobId) throw new Error(tr('The official demo preparation did not start.', '官方 Demo 数据准备未能启动。'));
        autoUse.add(jobId);
        watchDemoSourceJob(jobId);
        return true;
      } catch (error) {
        offerPending = false;
        host.setError(host.errorText(error));
        host.render();
        return false;
      }
    }

    async function handleAction(action, id) {
      const row = notices().find(item => item.id === id);
      if (!row) return false;
      if (action === 'use') { await useDemoSource(row); return true; }
      if (action === 'other') {
        removeNotice(row.id);
        await host.authorizeDataSource('begin_local_selection');
        return true;
      }
      if (action === 'dismiss') { removeNotice(row.id); host.render(); return true; }
      return false;
    }

    function renderNotice(row) {
      if (!row || row.role !== NOTICE_ROLE || row.session_id !== sessionId()) return '';
      const disabled = row.pending || host.busy() ? 'disabled' : '';
      const facts = [
        row.source_label,
        row.stays ? tr(`${row.stays.toLocaleString()} ICU stays`, `${row.stays.toLocaleString()} 例 ICU 住院`) : '',
        row.modules ? tr(`${row.modules} modules`, `${row.modules} 个模块`) : '',
      ].filter(Boolean).join(' · ');
      return `<article class="gpi-message assistant gpi-workflow-receipt gpi-host-notice" role="status" data-gpi-host-notice="${esc(row.id)}">
        <div class="gpi-message-body">
          <div class="gpi-workflow-receipt-head">${host.iconHtml('check', 15)}<div><strong>${esc(tr('Official demo data are prepared and registered', '官方 Demo 数据已准备并注册'))}</strong><span>${esc(tr('This is EasyICU state, not a model reply.', '这是 EasyICU 本地状态，不是模型回复。'))}</span></div></div>
          <p>${esc(facts)}</p>
          <p>${esc(tr(
            'Confirm it as this conversation’s data source and EasyICU proposes the research plan next. Confirming a source does not approve analysis.',
            '确认为本次会话的数据源后，EasyICU 会接着拟定研究计划；确认数据源不等于批准分析。',
          ))}</p>
          <div class="gpi-host-notice-actions">
            <button class="btn primary sm" type="button" data-gpi-host-notice-action="use" ${disabled}>${esc(row.pending ? tr('Confirming…', '正在确认…') : tr('Use it for this conversation', '用于本次会话'))}</button>
            <button class="btn sm" type="button" data-gpi-host-notice-action="other" ${disabled}>${esc(tr('Choose another folder', '选择其他文件夹'))}</button>
          </div>
        </div>
      </article>`;
    }

    return Object.freeze({ handleAction, namedDemo, noteToolResult, renderNotice, stopAll, sync, useNamedDemo, watchDemoSourceJob });
  }

  window.EasyICU.guidedPi.declare('hostJobs', { create });
})();
