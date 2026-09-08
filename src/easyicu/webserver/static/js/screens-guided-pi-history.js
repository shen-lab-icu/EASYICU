/* Copilot's persisted history and results owner. Browsing never changes the
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
          id: context.studyId, title: context.title || context.studyId, projectId: context.projectId });
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

  let dialog = null;
  let controller = null;
  let opener = null;
  function close() {
    if (controller) controller.invalidate();
    if (dialog) { dialog.close(); dialog.remove(); dialog = null; }
    if (opener && opener.isConnected) opener.focus();
  }
  function paint(state) {
    if (!dialog) return;
    const render = window.AGENT_RENDER;
    const review = state.review;
    const disabled = state.loading || state.signing;
    const selected = state.selected;
    const artifacts = rows(review && review.artifacts);
    const signable = review && review.readiness && review.readiness.signable === true && !review.signed && !selected.readOnly;
    const primaryNames = ['manuscript_provenance.json', 'result_tables.json', 'figure_gallery.json', 'manuscript_scaffold.pdf', 'agent_plan.json', 'literature_evidence.json'];
    const fileRow = (item, index) => `<div><button type="button" data-history-artifact="${index}" ${disabled || !/\.(json|pdf|html)$/i.test(item.name) ? 'disabled' : ''}>${esc(render.artifactTitle(item.name))}</button><small>${esc(item.name)}</small><button type="button" data-history-download="${index}" ${disabled ? 'disabled' : ''}>${tr('Download', '下载')}</button></div>`;
    const primaryFiles = primaryNames.map(name => artifacts.findIndex(item => item.name === name)).filter(index => index >= 0).map(index => fileRow(artifacts[index], index)).join('');
    const otherFiles = artifacts.map((item, index) => primaryNames.includes(item.name) ? '' : fileRow(item, index)).join('');
    dialog.innerHTML = `<header><div><h2>${tr('History & results', '历史与成果')}</h2><p>${tr('Saved runs and files. Closing this panel returns to your conversation.', '查看已保存的运行和文件，关闭后回到原对话。')}</p></div><button type="button" data-history-close aria-label="${tr('Close history', '关闭历史与成果')}">×</button></header>
      <div class="gpi-history-controls"><label>${tr('Project', '项目')}<select data-history-project ${disabled ? 'disabled' : ''}>${state.projects.map((project, index) => `<option value="${index}" ${selected && selected.id === project.id ? 'selected' : ''}>${esc(project.title)}</option>`).join('')}</select></label><button type="button" data-history-refresh ${disabled ? 'disabled' : ''}>${tr('Refresh', '刷新')}</button>${selected ? `<button type="button" data-history-continue ${disabled ? 'disabled' : ''}>${tr('Return to this research', '回到这项研究')}</button>` : ''}</div>
      ${state.warning ? `<p class="gpi-history-error" role="status">${tr('Some history could not be loaded: ', '部分历史未能完整加载：')}${esc(state.warning)}</p>` : ''}
      ${state.error ? `<p class="gpi-history-error" role="alert">${tr('Could not load or complete this action: ', '无法读取或完成操作：')}${esc(state.error)}</p>` : ''}
      ${state.loading ? `<p role="status">${tr('Reading saved records…', '正在读取已保存记录…')}</p>` : ''}
      <div class="gpi-history-layout"><nav aria-label="${tr('Saved runs', '已保存运行')}"><p>${state.count} ${tr('runs', '次运行')}</p>${state.runs.map((run, index) => `<button type="button" data-history-run="${index}" ${disabled ? 'disabled' : ''} aria-pressed="${Boolean(review && review.project_dir === run.project_dir)}"><strong>${esc(run.run_label || run.run_id)}</strong><span>${esc(tr('Original run: ', '原运行：') + (run.run_status === 'human_review_pending' ? tr('Plan awaiting review', '计划待审阅') : render.runStatusLabel(run.run_status || run.readiness_status || run.gate_status || 'unknown')))}</span><small>${esc(run.updated_at || '')} · ${Number(run.artifact_count || 0)} ${tr('files', '个文件')}</small></button>`).join('')}${!state.loading && !state.error && !state.runs.length ? `<p>${tr('No saved runs found for this selection.', '此选择下尚未找到保存的运行。')}</p>` : ''}</nav>
      <section class="gpi-history-detail">${review ? `<h3>${esc(review.run_id)}</h3><p>${tr('Local review does not grant publication authority.', '本地审阅不授予发表权限。')}</p>${review.signoff_stale ? `<p class="gpi-history-error">${tr('Sign-off is stale: files changed or are missing.', '签署已失效：文件发生变化或缺失。')}</p>` : review.signed ? `<p role="status">${tr('Local review recorded.', '已记录本地审阅。')}</p>` : ''}<button type="button" data-history-bundle ${disabled || !artifacts.length ? 'disabled' : ''}>${tr('Download all files', '下载全部文件')}</button>
      <div class="gpi-history-files">${primaryFiles}</div>${otherFiles ? `<details><summary>${tr('More files & provenance', '更多文件与溯源')}</summary><div class="gpi-history-files">${otherFiles}</div></details>` : ''}
      ${signable ? `<fieldset><legend>${tr('Record my local review', '记录我的本地审阅')}</legend>${confirmations.map(([id, en, zh]) => `<label><input type="checkbox" data-history-confirm="${id}" ${disabled ? 'disabled' : ''}>${tr(en, zh)}</label>`).join('')}<button type="button" data-history-sign disabled>${state.signing ? tr('Saving…', '正在保存…') : tr('Sign reviewed run', '签署本次审阅')}</button></fieldset>` : ''}
      ${state.artifact ? `<article class="gpi-history-preview"><h3>${esc(render.artifactTitle(state.artifact.name))}</h3>${render.artifactStructuredView(state.artifact.name, state.artifact.payload || {})}<details><summary>${tr('Source JSON', '来源 JSON')}</summary><pre>${esc(JSON.stringify(state.artifact.payload, null, 2))}</pre></details></article>` : ''}` : `<p>${tr('Select a run to inspect its files and review status.', '选择一次运行，查看对应文件和审阅状态。')}</p>`}</section></div>`;
  }
  async function continueResearch() {
    const currentController = controller;
    const currentDialog = dialog;
    const state = currentController.state;
    const selected = state.selected;
    if (!selected) return;
    state.loading = true; paint(state);
    try {
      const shell = modules.require('shell');
      const current = shell.historyContext();
      if (current.busy) throw new Error(tr('Wait for the current action to settle before switching research projects.', '当前操作完成后再切换研究项目。'));
      if (current.studyId === selected.id) { close(); return; }
      // Resolve the original conversation from exact saved bindings, never from titles.
      const stillCurrent = () => controller === currentController && dialog === currentDialog;
      const match = await findConversation(window.EU_API, state.drafts, selected.id, stillCurrent);
      if (!stillCurrent()) return;
      if (match) {
        const url = new URL(location.href);
        url.searchParams.set('pi_project', match.draft.id);
        url.searchParams.set('pi_session', match.session.session_id);
        url.hash = 'guided';
        if ((match.session.language === 'zh' ? 'zh' : 'en') !== window.EU_LANG && window.setLang) window.setLang(match.session.language === 'zh' ? 'zh' : 'en');
        location.assign(url.href);
        return;
      }
      const adapter = window.EU_AGENT_STUDY_CONTEXT;
      await adapter.prepareGuidedHandoff({ id: selected.id, name: [selected.title, selected.title], studyContext: selected.study });
      close();
      window.__euRender();
    } catch (error) { if (controller === currentController && dialog === currentDialog) { state.loading = false; state.error = String(error.message || error); paint(state); } }
  }
  async function open(context) {
    close();
    opener = document.activeElement;
    dialog = document.createElement('dialog');
    dialog.className = 'gpi-history';
    dialog.setAttribute('aria-label', tr('History & results', '历史与成果'));
    document.body.appendChild(dialog);
    dialog.addEventListener('cancel', event => { event.preventDefault(); close(); });
    controller = createController(window.EU_API, paint);
    const currentController = controller;
    const currentDialog = dialog;
    dialog.addEventListener('change', event => {
      if (event.target.matches('[data-history-project]')) controller.select(controller.state.projects[Number(event.target.value)]);
      if (event.target.matches('[data-history-confirm]')) {
        dialog.querySelector('[data-history-sign]').disabled = !Array.from(dialog.querySelectorAll('[data-history-confirm]')).every(input => input.checked);
      }
    });
    dialog.addEventListener('click', event => {
      const reference = event.target.closest('[data-gpi-reference]');
      if (reference) { event.preventDefault(); const number = reference.dataset.gpiReference; const target = /^[1-9][0-9]*$/.test(number) ? dialog.querySelector('#gpi-reference-' + number) : null; if (target) target.scrollIntoView({ block: 'nearest' }); return; }
      const claim = event.target.closest('[data-gpi-claim]');
      if (claim || event.target.closest('[data-gpi-claim-close]')) {
        const id = claim ? claim.dataset.gpiClaim : '';
        dialog.querySelectorAll('[data-gpi-claim-panel]').forEach(panel => { panel.hidden = panel.dataset.gpiClaimPanel !== id; });
        dialog.querySelectorAll('[data-gpi-claim]').forEach(button => button.setAttribute('aria-expanded', String(button.dataset.gpiClaim === id)));
        const drawer = dialog.querySelector('.gpi-claim-drawer'); if (drawer) drawer.classList.toggle('is-active', Boolean(id));
        const layout = dialog.querySelector('[data-gpi-manuscript-layout]'); if (layout) layout.classList.toggle('has-claim-drawer', Boolean(id));
        return;
      }
      const button = event.target.closest('button');
      if (!button || button.disabled) return;
      const state = controller.state;
      if (button.hasAttribute('data-history-close')) { close(); return; }
      if (button.hasAttribute('data-history-refresh')) { controller.catalog(context); return; }
      if (button.hasAttribute('data-history-continue')) { continueResearch(); return; }
      if (button.hasAttribute('data-history-run')) { controller.openRun(state.runs[Number(button.dataset.historyRun)]); return; }
      if (button.hasAttribute('data-history-bundle')) { controller.download(); return; }
      const artifacts = rows(state.review && state.review.artifacts);
      if (button.hasAttribute('data-history-artifact')) {
        const name = artifacts[Number(button.dataset.historyArtifact)].name;
        const current = modules.require('shell').historyContext();
        if (current.projectId && current.studyId === state.selected.id && state.review.engine === 'easyicu.research_agent.pipeline') {
          const resource = { kind: /\.(pdf|html)$/i.test(name) ? 'research_document' : 'research_artifact', run_id: state.review.run_id, artifact: name, label: window.AGENT_RENDER.artifactTitle(name), media_type: /\.pdf$/i.test(name) ? 'application/pdf' : /\.html$/i.test(name) ? 'text/html' : 'application/json' };
          close(); modules.require('preview').open(resource, current.projectId, { currentRunId: current.runId || '' });
        } else if (/\.json$/i.test(name)) controller.artifact(name);
        else controller.download(name);
      }
      if (button.hasAttribute('data-history-download')) controller.download(artifacts[Number(button.dataset.historyDownload)].name);
      if (button.hasAttribute('data-history-sign')) controller.sign(Array.from(dialog.querySelectorAll('[data-history-confirm]:checked')).map(input => input.dataset.historyConfirm));
    });
    dialog.showModal();
    context = context || {};
    if (!context.projectId) context.projectId = new URL(location.href).searchParams.get('pi_project') || '';
    if (context && context.projectId && !context.studyId) {
      try {
        const listed = await window.EU_API.loadPiCopilotSessions(100, context.projectId);
        const session = rows(listed.sessions).find(row => row.binding && row.binding.study_context_id);
        if (session) context = { ...context, studyId: session.binding.study_context_id };
      } catch (error) { context.resolutionError = String(error.message || error); }
    }
    if (controller !== currentController || dialog !== currentDialog) return;
    await currentController.catalog(context);
    if (context.resolutionError && controller === currentController && dialog === currentDialog) {
      currentController.state.warning += tr('The original conversation binding could not be read: ', '原对话绑定未能读取：') + context.resolutionError;
      paint(currentController.state);
    }
  }
  document.addEventListener('click', event => {
    if (event.target.closest('[data-gpi-history]')) open(modules.require('shell').historyContext());
  });
  window.addEventListener('hashchange', () => { if (location.hash !== '#guided') close(); });
  modules.declare('history', { createController, findConversation, open, close });
})();
