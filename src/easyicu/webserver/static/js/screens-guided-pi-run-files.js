/* Copilot's in-message run files owner. Browsing never changes the
   active study or starts a model turn; all reads/downloads/reviews use the
   existing host contracts and the selected run's exact directory. */
(function () {
  'use strict';
  const modules = window.EasyICU.guidedPi;
  const tr = (en, zh) => window.EU_LANG === 'zh' ? zh : en;
  const esc = value => window.EU_HTML.esc(value == null ? '' : value);
  const rows = value => Array.isArray(value) ? value : [];
  const confirmations = [
    ['evidence_reviewed', 'I reviewed the evidence artifacts', '我已审阅证据产物'],
    ['claims_remain_locked', 'Claims remain locked / not reportable', '我确认论断仍不可报告'],
    ['no_patient_rows_persisted', 'No patient rows are persisted', '我确认未持久化患者行'],
  ];

  async function findConversation(api, drafts, studyId, current) {
    for (const draft of drafts) {
      const result = await api.loadPiCopilotSessions(100, draft.id);
      if (current && !current()) return null;
      const session = rows(result.sessions).find(row => row.binding && row.binding.study_context_id === studyId);
      if (session) return { draft, session };
    }
    return null;
  }

  function createController(api, changed) {
    const state = { projects: [], drafts: [], selected: null, runs: [], count: 0,
      review: null, artifact: null, loading: false, error: '', warning: '', signing: false };
    let revision = 0;
    function update() { if (changed) changed(state); }
    function checked(value) {
      if (!value || value.ok === false) throw new Error(value && (value.message || value.error) || 'History response unavailable');
      return value;
    }
    async function catalog(context) {
      const ticket = ++revision;
      state.loading = true; state.error = ''; state.warning = ''; state.review = null; state.artifact = null; update();
      const results = await Promise.allSettled([
        api.listStudyContexts(), api.loadIdeaAgentProjects({ limit: 100 }),
        api.loadGuidedDrafts({ limit: 100 }), api.loadAgentRunHistory({ limit: 200 }),
      ]);
      if (ticket !== revision) return;
      const payloads = results.map(result => {
        if (result.status === 'rejected') { state.warning += String(result.reason.message || result.reason) + ' '; return {}; }
        try { return checked(result.value); } catch (error) { state.warning += error.message + ' '; return {}; }
      });
      const projects = new Map();
      rows(payloads[0].contexts).forEach(study => projects.set(study.id, {
        id: study.id, title: study.title || study.question || study.id, study,
      }));
      rows(payloads[1].projects).forEach(seed => {
        const existing = projects.get(seed.study_id) || {};
        projects.set(seed.study_id, { ...existing, id: seed.study_id,
          title: existing.title || seed.title || seed.study_id, seed,
          seedDir: seed.project_dir, readOnly: Boolean(seed.seed_kind === 'canonical9_import' || seed.benchmark || seed.read_only_import || seed.read_only) });
      });
      rows(payloads[3].runs).forEach(run => {
        if (run.study_id && !projects.has(run.study_id)) projects.set(run.study_id, { id: run.study_id, title: run.study_id });
      });
      if (context && context.studyId) {
        projects.set(context.studyId, { ...(projects.get(context.studyId) || {}),
          id: context.studyId, title: context.title || context.studyId, projectId: context.projectId,
          readOnly: !Array.isArray(payloads[1].projects) || Boolean((projects.get(context.studyId) || {}).readOnly) });
      }
      state.drafts = rows(payloads[2].drafts);
      state.projects = Array.from(projects.values());
      if (Number(payloads[1].count) > rows(payloads[1].projects).length || Number(payloads[2].count) > state.drafts.length
        || Number(payloads[3].count) > rows(payloads[3].runs).length) {
        state.warning += tr('Some registry rows are outside the display limit. Open the original project folder to recover them.', '部分登记记录超出展示上限，可通过打开原项目文件夹恢复。');
      }
      state.loading = false;
      const preferred = state.projects.find(project => project.id === (context && context.studyId));
      if (preferred || state.projects.length) await select(preferred || state.projects[0]);
      else { state.selected = null; state.runs = []; state.count = 0; update(); }
    }
    async function select(project) {
      const ticket = ++revision;
      state.selected = project; state.runs = []; state.count = 0;
      state.review = null; state.artifact = null; state.error = ''; state.loading = true; update();
      try {
        // Read both roots: selecting a seed must not hide later pipeline runs.
        const requests = [{ study_id: project.id, limit: 200 }];
        if (project.seedDir) requests.push({ study_id: project.id, project_seed_dir: project.seedDir, limit: 200 });
        const results = await Promise.all(requests.map(request => api.loadAgentRunHistory(request).then(checked)));
        if (ticket !== revision) return;
        const byDir = new Map();
        results.forEach(result => rows(result.runs).forEach(run => { if (run.project_dir) byDir.set(run.project_dir, run); }));
        rows(project.seed && project.seed.runs).forEach(run => {
          if (run.project_dir && !byDir.has(run.project_dir)) byDir.set(run.project_dir, {
            ...run, study_id: project.id, run_id: run.run_id || run.label,
            run_label: run.label, run_status: run.status,
          });
        });
        state.runs = Array.from(byDir.values()).sort((a, b) => Number(b.updated_at_epoch || 0) - Number(a.updated_at_epoch || 0));
        state.count = state.runs.length;
        if (results.some(result => Number(result.count) > rows(result.runs).length)) {
          state.warning = tr('This project exceeds the history display limit; older records remain in its folder.', '本项目超出历史展示上限，更早记录仍保存在项目文件夹中。');
        }
      } catch (error) { if (ticket === revision) state.error = String(error.message || error); }
      finally { if (ticket === revision) { state.loading = false; update(); } }
    }
    async function openRun(run) {
      const ticket = ++revision;
      state.review = null; state.artifact = null; state.error = ''; state.loading = true; update();
      try {
        const review = checked(await api.loadAgentRunReview(run.project_dir));
        if (ticket !== revision) return;
        if (review.project_dir !== run.project_dir || (review.study_id && review.study_id !== state.selected.id)) throw new Error('Run identity mismatch');
        state.review = review;
      } catch (error) { if (ticket === revision) state.error = String(error.message || error); }
      finally { if (ticket === revision) { state.loading = false; update(); } }
    }
    async function artifact(name) {
      const review = state.review;
      if (!review || !rows(review.artifacts).some(item => item.name === name)) return;
      const ticket = ++revision;
      state.loading = true; state.error = ''; state.artifact = null; update();
      try {
        const data = checked(await api.loadAgentRunArtifact(review.project_dir, name));
        if (ticket === revision) state.artifact = { name, payload: data.payload };
      } catch (error) { if (ticket === revision) state.error = String(error.message || error); }
      finally { if (ticket === revision) { state.loading = false; update(); } }
    }
    async function download(name) {
      const review = state.review;
      if (!review || state.loading || state.signing) return;
      if (name && !rows(review.artifacts).some(item => item.name === name)) return;
      const ticket = revision;
      try {
        if (name) await api.downloadAgentRunArtifact(review.project_dir, name);
        else await api.downloadAgentRunBundle(review.project_dir);
      } catch (error) { if (ticket === revision) { state.error = String(error.message || error); update(); } }
    }
    async function sign(provided) {
      const review = state.review;
      if (!review || state.loading || state.signing || state.selected.readOnly || review.signed
        || !review.readiness || review.readiness.signable !== true
        || !confirmations.every(([id]) => rows(provided).includes(id))) return;
      const ticket = revision;
      state.signing = true; state.error = ''; update();
      try {
        const result = checked(await api.signoffAgentRun(review.project_dir, {
          reviewer: 'local_reviewer', confirmations: provided, note: 'Reviewed from EasyICU Copilot',
        }));
        if (ticket === revision) state.review = result;
      } catch (error) { if (ticket === revision) state.error = String(error.message || error); }
      finally { state.signing = false; if (ticket === revision) update(); }
    }
    function invalidate() { revision += 1; }
    return { state, catalog, select, openRun, artifact, download, sign, invalidate };
  }

  function runIds(row) {
    const resources = rows(row && row.resources).concat(row && row.resource || []);
    rows(row && row.steps).forEach(step => resources.push(...rows(step.resources), ...(step.resource ? [step.resource] : [])));
    return [...new Set(resources.filter(resource => /^research_(artifact|document|report)$/.test(resource.kind || '') && resource.run_id).map(resource => resource.run_id))];
  }

  // Keep saved receipts at their original resource-bearing message. Runs with
  // no surviving transcript reference are explicit host receipts, never replies
  // invented on behalf of the model or the user.
  function projectTimeline(timeline, runs) {
    const projected = timeline.map(row => ({ ...row, savedRuns: [] }));
    runs.forEach(run => {
      const matches = projected.filter(row => !row.childJobHandoff && runIds(row).includes(run.run_id));
      const target = matches.filter(row => row.role === 'assistant').pop() || matches.pop();
      if (target) { target.savedRuns.push(run); return; }
      const at = Date.parse(run.updated_at || '') || Number(run.updated_at_epoch || 0) * 1000;
      const receipt = { id: 'saved-run-' + run.project_dir, role: 'saved_run', timelineAt: at, savedRuns: [run] };
      const index = at ? projected.findIndex(row => Number(row.timelineAt || row.startedAt || 0) > at) : -1;
      if (index >= 0) projected.splice(index, 0, receipt); else projected.push(receipt);
    });
    return projected;
  }

  function create({ api, context, changed, resourceButton }) {
    let identity = '';
    let catalog = null;
    let pending = null;
    let generation = 0;
    const entries = new Map();
    function reset() {
      generation += 1;
      if (catalog) catalog.invalidate();
      entries.forEach(entry => entry.controller.invalidate());
      entries.clear(); catalog = null; identity = ''; pending = null;
    }
    function sync() {
      const ctx = context();
      const key = ctx.studyId ? [ctx.projectId, ctx.studyId, ctx.sessionId, ctx.runId, ctx.busy].join(':') : '';
      if (key === identity) return pending;
      reset(); identity = key;
      if (!key) return Promise.resolve();
      const ticket = generation;
      catalog = createController(api());
      pending = catalog.catalog(ctx).then(() => { if (ticket === generation) changed(); });
      return pending;
    }
    function timeline(value) {
      return projectTimeline(value, catalog ? catalog.state.runs : []);
    }
    function notice() {
      if (!catalog) return '';
      const state = catalog.state;
      const error = state.error || state.warning;
      return error ? `<div class="gpi-run-files-error" role="alert">${esc(tr('Some saved run files could not be read: ', '部分运行文件未能读取：') + error)} <button type="button" data-run-files-retry>${tr('Retry', '重试')}</button></div>` : '';
    }
    function detail(entry) {
      const state = entry.controller.state;
      const review = state.review;
      const disabled = state.loading || state.signing;
      const error = state.error ? `<p class="gpi-run-files-error" role="alert">${esc(state.error)} <button type="button" data-run-files-reload>${tr('Retry', '重试')}</button></p>` : '';
      if (!review) return error || `<p role="status">${tr('Reading files…', '正在读取文件…')}</p>`;
      const artifacts = rows(review.artifacts);
      const current = context();
      const file = (item, index) => {
        const resource = { kind: /\.(pdf|html)$/i.test(item.name) ? 'research_document' : 'research_artifact', run_id: review.run_id, artifact: item.name,
          label: window.AGENT_RENDER.artifactTitle(item.name), media_type: /\.pdf$/i.test(item.name) ? 'application/pdf' : /\.html$/i.test(item.name) ? 'text/html' : 'application/json' };
        const canPreview = current.projectId && review.engine === 'easyicu.research_agent.pipeline' && /\.(json|pdf|html)$/i.test(item.name);
        return `<div class="gpi-run-file">${canPreview ? resourceButton(resource) : /\.json$/i.test(item.name) ? `<button type="button" data-run-files-artifact="${index}" ${disabled ? 'disabled' : ''}>${esc(resource.label)}</button>` : `<span>${esc(resource.label)}</span>`}<small>${esc(item.name)}</small><button type="button" data-run-files-download="${index}" ${disabled ? 'disabled' : ''}>${tr('Download', '下载')}</button></div>`;
      };
      const primaryNames = ['manuscript_provenance.json', 'result_tables.json', 'figure_gallery.json', 'manuscript_scaffold.pdf', 'agent_plan.json', 'literature_evidence.json'];
      const primary = primaryNames.map(name => artifacts.findIndex(item => item.name === name)).filter(index => index >= 0).map(index => file(artifacts[index], index)).join('');
      const other = artifacts.map((item, index) => primaryNames.includes(item.name) ? '' : file(item, index)).join('');
      const signable = review.readiness && review.readiness.signable === true && !review.signed && !state.selected.readOnly;
      return `${error}<div class="gpi-run-files-actions"><button type="button" data-run-files-bundle ${disabled || !artifacts.length ? 'disabled' : ''}>${tr('Download all files', '下载全部文件')}</button><button type="button" data-run-files-reload ${disabled ? 'disabled' : ''}>${tr('Refresh status', '刷新状态')}</button></div>
        <div class="gpi-run-file-list">${primary}</div>${other ? `<details class="gpi-run-extra" ${entry.extraOpen ? 'open' : ''}><summary>${tr('More files & provenance', '更多文件与溯源')}</summary><div class="gpi-run-file-list">${other}</div></details>` : ''}
        ${review.signoff_stale ? `<p class="gpi-run-files-error">${tr('Review is stale: files changed or are missing.', '审阅已失效：文件发生变化或缺失。')}</p>` : review.signed ? `<p role="status">${tr('Local review recorded.', '已记录本地审阅。')}</p>` : ''}
        ${signable ? `<details class="gpi-run-signoff" ${entry.reviewOpen ? 'open' : ''}><summary>${tr('Record my review', '记录我的审阅')}</summary><p>${tr('Local review does not grant publication authority.', '本地审阅不授予发表权限。')}</p><fieldset>${confirmations.map(([id, en, zh]) => `<label><input type="checkbox" data-run-files-confirm="${id}" ${entry.checks.has(id) ? 'checked' : ''} ${disabled ? 'disabled' : ''}>${tr(en, zh)}</label>`).join('')}<button type="button" data-run-files-sign ${disabled || entry.checks.size !== confirmations.length ? 'disabled' : ''}>${tr('Sign reviewed run', '签署本次审阅')}</button></fieldset></details>` : ''}
        ${state.artifact ? `<article class="gpi-run-artifact"><h4>${esc(window.AGENT_RENDER.artifactTitle(state.artifact.name))}</h4>${window.AGENT_RENDER.artifactStructuredView(state.artifact.name, state.artifact.payload || {})}<details><summary>${tr('Source JSON', '来源 JSON')}</summary><pre>${esc(JSON.stringify(state.artifact.payload, null, 2))}</pre></details></article>` : ''}`;
    }
    function render(row) {
      return rows(row.savedRuns).map(run => {
        const entry = entries.get(run.project_dir);
        const status = run.run_status === 'human_review_pending' ? tr('Plan awaiting review', '计划待审阅') : window.AGENT_RENDER.runStatusLabel(run.run_status || run.readiness_status || 'unknown');
        return `<details class="gpi-run-files" data-run-files="${esc(run.project_dir)}" ${entry && entry.open ? 'open' : ''}><summary><span>${tr('Run files & review', '本次运行文件与审阅')}</span><small>${esc(run.run_id)} · ${esc(tr('Original run: ', '原运行：') + status)}</small></summary>${entry && entry.open ? detail(entry) : ''}</details>`;
      }).join('');
    }
    async function open(dir) {
      if (!catalog) return;
      const run = catalog.state.runs.find(item => item.project_dir === dir);
      if (!run) return;
      const ticket = generation;
      let entry = entries.get(dir);
      if (!entry) {
        entry = { open: true, checks: new Set(), reviewOpen: false };
        entry.controller = createController(api(), () => { if (ticket === generation) changed(); });
        entry.controller.state.selected = catalog.state.selected;
        entries.set(dir, entry);
      }
      entry.open = true; entry.checks.clear();
      await entry.controller.openRun(run);
    }
    function handleClick(event) {
      const target = event.target;
      if (target.closest('[data-run-files-retry]')) { reset(); sync(); changed(); return true; }
      const section = target.closest('[data-run-files]');
      if (!section) return false;
      const dir = section.dataset.runFiles;
      const entry = entries.get(dir);
      const reference = target.closest('[data-gpi-reference]');
      if (reference) {
        event.preventDefault();
        const number = reference.dataset.gpiReference;
        const node = /^[1-9][0-9]*$/.test(number) ? section.querySelector('#gpi-reference-' + number) : null;
        if (node) node.scrollIntoView({ block: 'nearest' });
        return true;
      }
      const claim = target.closest('[data-gpi-claim]');
      if (claim || target.closest('[data-gpi-claim-close]')) {
        const id = claim ? claim.dataset.gpiClaim : '';
        section.querySelectorAll('[data-gpi-claim-panel]').forEach(panel => { panel.hidden = panel.dataset.gpiClaimPanel !== id; });
        section.querySelectorAll('[data-gpi-claim]').forEach(button => button.setAttribute('aria-expanded', String(button.dataset.gpiClaim === id)));
        const drawer = section.querySelector('.gpi-claim-drawer'); if (drawer) drawer.classList.toggle('is-active', Boolean(id));
        const layout = section.querySelector('[data-gpi-manuscript-layout]'); if (layout) layout.classList.toggle('has-claim-drawer', Boolean(id));
        return true;
      }
      const summary = target.closest('summary');
      if (summary && summary.parentElement === section) {
        event.preventDefault();
        if (entry && entry.open) { entry.open = false; changed(); } else open(dir);
        return true;
      }
      if (!entry) return true;
      if (summary && summary.parentElement.classList.contains('gpi-run-signoff')) { entry.reviewOpen = !summary.parentElement.open; return true; }
      if (summary && summary.parentElement.classList.contains('gpi-run-extra')) { entry.extraOpen = !summary.parentElement.open; return true; }
      const artifact = target.closest('[data-run-files-artifact]');
      if (artifact && !artifact.disabled) {
        const item = rows(entry.controller.state.review && entry.controller.state.review.artifacts)[Number(artifact.dataset.runFilesArtifact)];
        if (item) entry.controller.artifact(item.name);
        return true;
      }
      if (target.closest('[data-run-files-reload]')) { open(dir); return true; }
      if (target.closest('[data-run-files-bundle]')) { entry.controller.download(); return true; }
      const download = target.closest('[data-run-files-download]');
      if (download && !download.disabled) {
        const item = rows(entry.controller.state.review && entry.controller.state.review.artifacts)[Number(download.dataset.runFilesDownload)];
        if (item) entry.controller.download(item.name);
        return true;
      }
      if (target.closest('[data-run-files-sign]')) { entry.controller.sign([...entry.checks]); return true; }
      // Resource preview events continue through the existing conversation owner.
      return false;
    }
    function handleChange(event) {
      const input = event.target.closest('[data-run-files-confirm]');
      if (!input) return false;
      const entry = entries.get(input.closest('[data-run-files]').dataset.runFiles);
      if (!entry) return true;
      if (input.checked) entry.checks.add(input.dataset.runFilesConfirm); else entry.checks.delete(input.dataset.runFilesConfirm);
      const button = input.closest('fieldset').querySelector('[data-run-files-sign]');
      button.disabled = entry.checks.size !== confirmations.length;
      return true;
    }
    return { sync, reset, timeline, render, notice, handleClick, handleChange, open };
  }
  modules.declare('runFiles', { createController, findConversation, projectTimeline, runIds, create });
})();
