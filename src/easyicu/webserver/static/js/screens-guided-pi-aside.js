/* Guided Copilot workflow-authority panel owner.

   Owner: projecting 7 required stages plus optional idea mining into the panel --
   current stage, its reason, progress, and the full stage list. Split out of
   screens-guided-pi.js, which was hundreds of lines past its size ratchet.

   Read-only by contract: it renders host state into #gdStudyAside and never
   mutates it. The legacy Guided shell must not write this panel while the
   Copilot shell is mounted; screens-guided.js guards its own renderAside for
   exactly that reason. */
(function () {
  'use strict';

  function create(host) {
    const tr = host.tr;
    const esc = host.esc;
    const iconHtml = host.iconHtml;
    const projectId = host.projectId;
    const displayProjectTitle = host.displayProjectTitle;
    let resultQuery = '';
    let resultProjectId = '';
    const LAYOUT_KEY = 'easyicu.pi.workspacePanels.v1';
    // Three panels, each backed by an EasyICU owner: the governed workflow
    // stages, the run's artifacts, and per-project notes. A "Compute" panel
    // was dropped: EasyICU executes locally (remote compute is a disabled
    // capability) and has no machines, background workers, or pipelines to
    // list; the run facts it carried live on the current to-do row.
    const panelKeys = ['progress', 'results', 'notes'];
    let visiblePanels = { progress: true, results: true, notes: true };
    try {
      const saved = JSON.parse(window.localStorage.getItem(LAYOUT_KEY) || '{}');
      panelKeys.forEach(key => { if (typeof saved[key] === 'boolean') visiblePanels[key] = saved[key]; });
    } catch (_) {}
    let resultSort = 'recommended';
    let resultView = 'list';

    function togglePanel(key) {
      if (!panelKeys.includes(key)) return false;
      visiblePanels[key] = !visiblePanels[key];
      try { window.localStorage.setItem(LAYOUT_KEY, JSON.stringify(visiblePanels)); } catch (_) {}
      syncProjectWorkflowAside();
      return visiblePanels[key];
    }

    function layoutOptions() { return { ...visiblePanels }; }

    // Project notes are project memory: they live as `project_notes.md` in
    // the project's local folder (guided_sessions owns the file) so they
    // survive browsers and machines. The browser copy remains the fallback
    // for a project without a registered folder, and a browser-only note
    // from before this owner is offered into the folder once.
    const NOTES_SAVE_DELAY_MS = 800;
    let notes = { projectId: '', text: '', available: null, loading: false, saving: false, savedAt: '', error: '', timer: null };
    function notesKey(id) { return `easyicu.pi.projectNotes.v1.${id}`; }
    function localNotes(id) { try { return window.localStorage.getItem(notesKey(id)) || ''; } catch (_) { return ''; } }
    function rememberLocalNotes(id, text) { try { window.localStorage.setItem(notesKey(id), text); } catch (_) {} }
    function notesApi() { return host.api ? host.api() : (window.EU_API || {}); }
    function ensureNotesLoaded() {
      const id = projectId();
      if (!id || notes.projectId === id) return;
      if (notes.timer) { clearTimeout(notes.timer); notes.timer = null; }
      notes = { projectId: id, text: localNotes(id), available: null, loading: true, saving: false, savedAt: '', error: '', timer: null };
      const client = notesApi();
      if (typeof client.loadGuidedProjectNotes !== 'function') { notes.available = false; notes.loading = false; return; }
      void client.loadGuidedProjectNotes(id).then(payload => {
        if (notes.projectId !== id) return;
        notes.loading = false;
        notes.available = Boolean(payload && payload.available);
        notes.savedAt = String(payload && payload.updated_at || '');
        if (notes.available) {
          if (payload.present) notes.text = String(payload.text || '');
          else if (notes.text.trim()) scheduleNotesSave();
        }
        paintNotes();
      }).catch(error => {
        if (notes.projectId !== id) return;
        notes.loading = false; notes.available = false;
        notes.error = String((error && error.message) || tr('Could not read the project notes.', '读取项目笔记失败。')).slice(0, 200);
        paintNotes();
      });
    }
    function scheduleNotesSave() {
      if (notes.timer) clearTimeout(notes.timer);
      notes.timer = setTimeout(() => { notes.timer = null; void saveNotes(); }, NOTES_SAVE_DELAY_MS);
    }
    async function saveNotes() {
      const id = notes.projectId;
      const client = notesApi();
      if (!id || notes.available !== true || typeof client.saveGuidedProjectNotes !== 'function') return;
      const text = notes.text;
      notes.saving = true; notes.error = ''; paintNotes();
      try {
        const payload = await client.saveGuidedProjectNotes(id, text);
        if (notes.projectId !== id) return;
        notes.savedAt = String(payload && payload.updated_at || '');
        if (notes.text === text) try { window.localStorage.removeItem(notesKey(id)); } catch (_) {}
      } catch (error) {
        if (notes.projectId !== id) return;
        notes.error = String((error && error.message) || tr('Could not save the project notes.', '保存项目笔记失败。')).slice(0, 200);
        rememberLocalNotes(id, text);
      } finally {
        if (notes.projectId === id) { notes.saving = false; paintNotes(); }
      }
    }
    function notesStatusText() {
      if (notes.loading) return tr('Reading the project folder…', '正在读取项目文件夹……');
      if (notes.error) return notes.error;
      if (notes.available === false) return tr('This project has no local folder; notes stay in this browser.', '此项目没有本地文件夹，笔记仅保存在此浏览器。');
      if (notes.saving) return tr('Saving to the project folder…', '正在保存到项目文件夹……');
      if (notes.timer) return tr('Unsaved changes', '有未保存的改动');
      const contracts = window.EU_GUIDED_CONTRACTS || {};
      const when = notes.savedAt && typeof contracts.fmtRunTime === 'function' ? contracts.fmtRunTime(notes.savedAt) : '';
      return when
        ? `${tr('Saved in the project folder', '已保存到项目文件夹')} · ${when}`
        : tr('Saved as project_notes.md in the project folder.', '保存为项目文件夹里的 project_notes.md。');
    }
    function paintNotes() {
      const body = document.getElementById('gdAsideBody');
      const status = body && body.querySelector('[data-gpi-notes-status]');
      if (status) status.textContent = notesStatusText();
      const area = body && body.querySelector('[data-gpi-project-notes]');
      if (area && document.activeElement !== area && area.value !== notes.text) area.value = notes.text;
    }
    function notesPanel() {
      ensureNotesLoaded();
      return `<div class="gpi-project-notes"><textarea data-gpi-project-notes aria-label="${tr('Project notes', '项目笔记')}" placeholder="${tr('Add notes for this project…', '记录当前项目的备忘……')}"${notes.loading ? ' readonly' : ''}>${esc(notes.text)}</textarea><small data-gpi-notes-status>${esc(notesStatusText())}</small></div>`;
    }
    function handleNotesInput(target) {
      const id = projectId();
      notes.projectId = notes.projectId || id;
      notes.text = target.value;
      if (notes.available === true) { rememberLocalNotes(id, notes.text); scheduleNotesSave(); }
      else rememberLocalNotes(id, notes.text);
      paintNotes();
    }

    // The two run facts the current to-do row shows under its task text:
    // the stage sentence of the job EasyICU is running now, and — once the
    // previous governed run stopped — the cause, named with its run id.
    function runFacts(workflow) {
      const job = workflow && workflow.active_job && workflow.active_job.present
        ? workflow.active_job : null;
      const running = job && job.status === 'running';
      const progress = running && Array.isArray(job.progress) ? job.progress.slice(-1)[0] : null;
      // The raw lifecycle step ("planning · 1/4") reads like a progress bar
      // but 1/4 is a validation attempt count; show the researcher-facing
      // stage sentence instead.
      const stage = progress && progress.step && typeof host.progressLabel === 'function'
        ? String(host.progressLabel(progress) || '') : '';
      const last = host.latestRun && host.latestRun();
      const row = !running && last && last.present ? last : null;
      // A saved run row carries its gate reason rather than an error code; the
      // workflow projection names it `gate_reason_code` with `gate_status`.
      const blocked = row && ['failed', 'blocked'].includes(String(row.run_status || row.status || row.gate_status || ''));
      const gateReason = String(row && (row.gate_reason_code || row.gate_reason) || '');
      // A blocked gate that is merely waiting for review is not a failure.
      const failureGate = blocked && /^research_pipeline_|^data_foundation_blocked$/.test(gateReason) ? gateReason : '';
      const failureCode = String(row && (row.error_code || failureGate) || '');
      const failure = failureCode
        ? String(typeof host.runFailureText === 'function' ? host.runFailureText(failureCode) || failureCode : failureCode)
        : '';
      return { stage, failure, failedRunId: failure ? String(row.run_id || '') : '' };
    }

    // The project's run record: every governed run bound to this study,
    // newest first, as facts (type, outcome, time, file count, cause) — the
    // results shelf above shows only the run that currently owns the
    // workflow. Rows are informational; historical artifacts stay closed.
    function runStatus(run) {
      const status = String(run.run_status || '');
      const gate = String(run.gate_status || '');
      if (status === 'human_review_pending') return { key: 'review', label: tr('Awaiting review', '待审阅') };
      if (status === 'failed') return { key: 'failed', label: tr('Failed', '失败') };
      if (status === 'cancelled') return { key: 'cancelled', label: tr('Cancelled', '已取消') };
      if (status === 'running') return { key: 'running', label: tr('Running', '运行中') };
      if (status === 'blocked' || gate === 'blocked') return { key: 'blocked', label: tr('Blocked', '阻断') };
      if (status === 'pass' || gate === 'pass') return { key: 'done', label: tr('Passed', '已通过') };
      return { key: 'recorded', label: tr('Recorded', '已有记录') };
    }
    function runHistoryOpen(body, fallback) {
      const previous = body && body.querySelector && body.querySelector('[data-gpi-run-history]');
      return previous ? Boolean(previous.open) : Boolean(fallback);
    }
    function runHistoryHtml(workflow, open) {
      const runs = Array.isArray(workflow && workflow.runs) ? workflow.runs : [];
      if (!runs.length) return '';
      const contracts = window.EU_GUIDED_CONTRACTS || {};
      const when = typeof contracts.fmtRunTime === 'function' ? contracts.fmtRunTime : value => String(value || '');
      const kind = run => run.run_type === 'full'
        ? tr('Full analysis', '完整分析') : run.plan_available ? tr('Candidate plan', '候选计划') : tr('Run', '运行');
      const rows = runs.map(run => {
        const status = runStatus(run);
        const cause = ['failed', 'blocked'].includes(status.key) && run.gate_reason_code && typeof host.runFailureText === 'function'
          ? String(host.runFailureText(run.gate_reason_code) || '') : '';
        const count = Number(run.artifact_count || 0);
        return `<li class="gpi-run-row is-${status.key}${run.authoritative ? ' is-current' : ''}"><span class="gpi-run-mark" aria-hidden="true"></span><div class="gpi-run-copy"><strong>${esc(kind(run))} · ${esc(status.label)}${run.authoritative ? `<span class="gpi-run-current">${tr('current', '当前')}</span>` : ''}</strong><small>${esc(when(run.updated_at))}${count ? ` · ${count} ${tr('files', '个文件')}` : ''}</small>${cause ? `<small class="gpi-run-cause">${esc(cause)}</small>` : ''}<code>${esc(run.run_id)}</code></div></li>`;
      }).join('');
      return `<details class="gpi-run-history" data-gpi-run-history${open ? ' open' : ''}><summary>${tr('Run record', '运行记录')}<small class="gpi-aside-count">${runs.length}</small></summary><ol>${rows}</ol></details>`;
    }

    function syncProjectWorkflowAside() {
      const demo = host.demoMode() && window.EasyICU.guidedPi.optional('demo');
      const workflow = demo && typeof demo.workflow === 'function' ? demo.workflow() : host.workflow();
      if (host.shell() !== 'pi' || (!host.demoMode() && !projectId())) return;
      if (resultProjectId !== projectId()) { resultProjectId = projectId(); resultQuery = ''; }
      const aside = document.getElementById('gdStudyAside');
      const body = document.getElementById('gdAsideBody');
      const head = aside && aside.querySelector('.gd-aside-head');
      if (!aside || !body || !head) return;
      const section = (key, title, content) => {
        if (!visiblePanels[key]) return '';
        const previousSection = body.querySelector && body.querySelector(`[data-gpi-aside-section="${key}"]`);
        const open = !previousSection || previousSection.open;
        return `<details class="gpi-aside-section" data-gpi-aside-section="${key}"${open ? ' open' : ''}><summary>${title}</summary><div>${content}</div></details>`;
      };
      const emptyResults = `<p class="gpi-aside-empty">${tr('Results will appear here when available.', '生成的成果将在这里集中展示。')}</p>`;
      head.innerHTML = `<div class="at">${host.demoMode() ? tr('Reviewer demonstration', '审稿人演示') : tr('Research workspace', '研究工作区')}</div>`;
      if (!workflow) {
        const error = host.workflowError && host.workflowError();
        body.innerHTML = section('progress', tr('To-dos', '待办'), `<div class="gd-pipeline-summary" data-gpi-project-workflow-loading role="status" aria-live="polite"><div class="gd-pipeline-value">${esc(error || tr('Loading project progress…', '正在读取项目进度…'))}</div></div>`)
          + section('results', tr('Results', '成果'), emptyResults)
          + section('notes', tr('Notes', '笔记'), notesPanel())
          + (!panelKeys.some(key => visiblePanels[key]) ? `<p class="gpi-aside-empty gpi-panels-empty">${tr('Choose panels from Layout above.', '可从顶部「布局」重新显示面板。')}</p>` : '');
        body.oninput = event => {
          if (event.target.matches && event.target.matches('[data-gpi-project-notes]')) handleNotesInput(event.target);
        };
        body.onclick = null;
        body.onchange = null;
        return;
      }
      const stages = Array.isArray(workflow.stages) ? workflow.stages : [];
      const reviewerDemo = workflow.kind === 'reviewer_validation_demo';
      const names = {
        question: reviewerDemo ? tr('Reviewer protocol', '审稿协议') : tr('Scientific question', '科学问题'),
        idea: reviewerDemo ? tr('Validation scope', '验证范围') : tr('Idea mining', '想法发掘'),
        setup: reviewerDemo ? tr('Data contract', '数据合同') : tr('Study setup', '研究配置'),
        extraction: reviewerDemo ? tr('Safe projection', '安全投影') : tr('Research data preparation', '研究数据准备'),
        plan: tr('Analysis plan', '分析计划'), analysis: tr('Analysis and validation', '分析与验证'),
        interpretation: tr('Result interpretation', '结果解读'), manuscript: reviewerDemo ? tr('Reviewer dossier', '审稿报告') : tr('Manuscript', '稿件'),
      };
      const reasons = {
        question_bound: tr('Question is bound to this project', '科学问题已绑定到当前项目'),
        idea_handoff_accepted: tr('Selected idea is digest-bound', '所选想法已用摘要绑定'),
        prior_art_authority_not_established: tr('Prior-art authority and novelty are not established', '先前研究权限与新颖性未成立'),
        idea_feasibility_refresh_required: tr('Recheck feasibility against the current data source', '需要按当前数据源重新核验可行性'),
        study_setup_complete: tr('Required study setup is complete', '必需研究配置已完成'),
        approved_plan_setup_receipt: tr('The approved plan records the study setup used for this analysis', '已批准的计划记录了本次分析采用的研究配置'),
        active_export_ready: tr('A matching EasyICU export is ready', '同一项目的 EasyICU 数据包已就绪'),
        approved_analysis_input_receipt: tr('The completed analysis records its prepared input', '已完成的分析记录了本次使用的研究输入'),
        bound_research_input_prepared: tr('The bound research input is prepared; scientific validation remains separate', '本次研究输入已备妥；科学验证仍需单独完成'),
        metadata_only_input_not_prepared: tr('This candidate uses metadata only; research data are not prepared yet', '当前候选计划仅使用元数据，本次研究数据尚未备妥'),
        research_input_preparation_required: tr('The source is registered; prepare this question’s research input', '数据源已登记，仍需准备本次研究输入'),
        plan_ready: tr('Ready to create the analysis plan', '可以生成分析计划'),
        provider_ready_to_generate_plan: tr('Question and data source are ready; generate a candidate plan for review', '问题和数据源已就绪，可以生成候选计划供审阅'),
        agent_plan_ready: tr('The digest-bound analysis plan is ready', '摘要绑定分析计划已就绪'),
        operator_plan_approval_required: tr('Review and approve the digest-bound plan before analysis', '请在分析前审核并批准摘要绑定的计划'),
        plan_execution_upgrade_required: tr('Generate one package-bound plan before analysis', '需要先生成一份与数据包绑定的可执行计划'),
        plan_scientific_changes_required: tr('The scientific plan review requires a new study/plan version before analysis', '科学计划审阅要求先形成新的研究/计划版本，当前不能继续分析'),
        plan_configuration_superseded: tr('The study configuration changed; the old plan is superseded and cannot be approved', '研究配置已变化；旧计划已失效，不能再批准'),
        plan_review_not_resumable: tr('The old plan no longer has a live resume authority and must be regenerated', '旧计划的可恢复执行权限已失效，必须重新生成'),
        failed_pipeline_requires_fresh_plan: tr('The previous run failed. Review the record before creating a fresh plan.', '上次任务未完成，请查看记录后重新生成计划。'),
        scientific_plan_review_policy_stale: tr('The scientific review policy changed; regenerate the plan while keeping the prepared data', '科学审阅规则已更新；保留已准备数据并重新生成计划'),
        operator_plan_approved: tr('Digest-bound plan approved by the user', '摘要绑定计划已由用户批准'),
        analysis_ready: tr('Ready for analysis after plan approval', '计划确认后可以执行分析'),
        research_planning_running: tr('The research task is running; no analysis execution progress is available yet', '科研任务运行中；尚无分析执行进度回执'),
        analysis_running: tr('The approved analysis is running', '已批准的分析正在执行'),
        report_repair_running: tr('Revising the report from sealed evidence; analysis is not being rerun', '正在用封存证据修订报告；不会重跑分析'),
        planner_checkpoint_resume_available: tr('Continue planning from a preserved checkpoint after validating its binding', '校验绑定后，从保留的检查点继续生成计划'),
        validated_analysis_required: tr('Validated analysis is required first', '需要先完成并验证分析'),
        validated_analysis_complete: tr('Analysis, validation, and numeric checks are complete', '分析、验证与数值核验已完成'),
        validated_analysis_ready: tr('Analysis, validation, and numeric checks are complete', '分析、验证与数值核验已完成'),
        evidence_bound_interpretation_ready: tr('Review the evidence-bound result interpretation', '请审阅证据约束的结果解读'),
        manuscript_draft_ready_for_review: tr('Review the evidence-bound manuscript draft', '请审阅证据绑定的稿件草稿'),
        interpretation_complete: tr('Evidence-bounded interpretation is complete', '证据约束的结果解读已完成'),
        human_review_required: tr('Draft is locked pending clinical and methods review', '初稿已锁定，等待临床与方法学审阅'),
        source_population_scope_open: tr('Prepared data are traceable, but source-population scope is open', '准备数据可追踪，但来源人群范围未闭合'),
        publication_analysis_incomplete: tr('The executable plan is not a complete publication analysis', '可执行计划不是完整投稿分析'),
        paper_authority_not_granted: tr('Draft generated; publication authority was not granted', '初稿已生成；未授予论文发表权限'),
        full_agent_manuscript_required: tr('A governed Agent manuscript is required', '需要由受治理的 Agent 生成稿件'),
        report_revision_ready_for_review: tr('Current report revision is ready for review', '当前报告修订可供审阅'),
        reviewer_protocol_bound: tr('Six reviewer criteria were bound before results', '已在查看结果前绑定 6 项审稿标准'),
        bounded_validation_objective_selected: tr('The systems-validation objective is explicit', '系统验证目标已明确'),
        prepared_data_contract_verified: tr('The prepared-data and descriptive claim contracts are verified', '准备后数据与描述性结论合同已核验'),
        aggregate_projection_verified: tr('The aggregate-only browser projection passed', '仅聚合浏览器投影已通过'),
        exact_plan_reviewed: tr('The exact six-step plan was reviewed', '精确六步计划已审阅'),
        six_of_six_steps_complete: tr('All six required steps completed', '6 个必需步骤全部完成'),
        descriptive_ceiling_preserved: tr('Interpretation stayed within the descriptive ceiling', '结果解读保持在描述性上限内'),
        reviewer_dossier_complete: tr('The reviewer HTML and PDF dossier are complete', '审稿 HTML 与 PDF 报告已完整生成'),
      };
      const reasonText = stage => reasons[stage && stage.reason_code]
        || tr('Waiting for the preceding governed stage', '等待前一受治理阶段完成');
      const done = Number(workflow.completed_required_stages || 0);
      const total = Math.max(1, Number(workflow.required_stage_count || 7));
      const current = stages.find(stage => stage.id === workflow.current_stage)
        || stages.find(stage => stage.status !== 'complete') || stages[stages.length - 1];
      const currentIndex = Math.max(0, stages.indexOf(current));
      const next = stages.slice(currentIndex + 1).find(stage => stage.status !== 'complete');
      const nextIsActionable = next && ['ready', 'running', 'review_required'].includes(next.status);
      const nextCaption = nextIsActionable
        ? tr('Next step', '下一步')
        : tr('Later stage', '后续阶段');
      const results = !host.demoMode() && host.resultsHtml
        ? host.resultsHtml(resultQuery, { sort: resultSort, view: resultView }) : '';
      const pending = !host.demoMode() && host.hasPendingReview && host.hasPendingReview();
      const reviewAction = !host.demoMode() && host.reviewActionHtml ? host.reviewActionHtml() : '';
      // The to-do list reads like a task checklist: every stage is a row, and
      // the current row says what is needed now in the words of the decision
      // card the conversation is showing (or what EasyICU is doing), not in
      // gate vocabulary. The stage reason is the fallback.
      const activeTask = !host.demoMode() && typeof host.activeTaskTitle === 'function'
        ? String(host.activeTaskTitle() || '') : '';
      const decision = !host.demoMode() && typeof host.pendingDecisionTitle === 'function'
        ? String(host.pendingDecisionTitle() || '') : '';
      const currentText = activeTask || decision || reasonText(current);
      const facts = host.demoMode() ? { stage: '', failure: '', failedRunId: '' } : runFacts(workflow);
      const stageLine = activeTask && facts.stage && facts.stage !== currentText
        ? `<div class="si-s">${esc(facts.stage)}</div>` : '';
      const failureLine = !activeTask && facts.failure
        ? `<div class="si-s gpi-run-failure">${facts.failedRunId ? `<code>${esc(facts.failedRunId)}</code> ` : ''}${esc(facts.failure)}</div>` : '';
      const currentAction = pending
        ? `<button type="button" class="btn sm gpi-study-pending" data-gpi-aside-pending>${tr('View pending decision', '查看待确认事项')}</button>`
        : !results && reviewAction ? `<div class="gpi-study-pending">${reviewAction}</div>` : '';
      const progress = `
      <div class="gd-pipeline-summary gd-pipeline-checklist" data-gpi-project-workflow-aside>
      <ol class="gd-pipeline-list" data-gpi-project-workflow-list aria-label="${tr('Research to-dos', '研究待办')}">${stages.map(stage => {
        const optional = stage.required_for_completion === false;
        const isCurrent = stage === current && stage.status !== 'complete';
        const status = stage.status === 'complete' ? 'done' : stage.status === 'optional' ? 'optional' : stage.status === 'ready' || stage.status === 'running' || stage.status === 'review_required' ? 'active' : 'locked';
        const running = isCurrent && Boolean(activeTask);
        const marker = status === 'done' ? iconHtml('check', 11)
          : running ? '<span class="gpi-running-spinner" aria-hidden="true"></span>'
          : status === 'locked' ? iconHtml('lock', 10) : iconHtml('dot', 10);
        const caption = stage === next && !results
          ? `<span class="gd-pipeline-next">${nextCaption}</span>` : '';
        const detail = isCurrent
          ? `<div class="si-v">${esc(currentText)}</div>${stageLine}${failureLine}${currentAction}` : '';
        return `<li class="study-item ${status}${isCurrent ? ' current' : ''}"${isCurrent ? ' aria-current="step"' : ''}><span class="si-dot">${marker}</span><div class="si-txt"><div class="si-t">${esc(names[stage.id] || stage.label || stage.id)}${optional ? tr(' · Optional', ' · 可选') : ''}${caption}</div>${detail}</div></li>`;
      }).join('')}</ol>
      <div class="gd-pipeline-meta"><span><strong>${done}/${total}</strong> ${tr('required stages complete', '个必需阶段已完成')}</span></div>
      </div>`;
      body.innerHTML = section('progress', `${tr('To-dos', '待办')}<small class="gpi-aside-count">${done}/${total}</small>`, progress)
        + section('results', tr('Results', '成果'), (results || emptyResults) + runHistoryHtml(workflow, runHistoryOpen(body, !results)))
        + section('notes', tr('Notes', '笔记'), notesPanel())
        + (!panelKeys.some(key => visiblePanels[key]) ? `<p class="gpi-aside-empty gpi-panels-empty">${tr('Choose panels from Layout above.', '可从顶部「布局」重新显示面板。')}</p>` : '');
      body.oninput = event => {
        if (event.target.matches && event.target.matches('[data-gpi-project-notes]')) { handleNotesInput(event.target); return; }
        if (!event.target.matches || !event.target.matches('[data-gpi-results-search]')) return;
        resultQuery = event.target.value;
        const needle = resultQuery.trim().toLocaleLowerCase();
        let visible = 0;
        body.querySelectorAll('[data-gpi-result-file]').forEach(row => {
          row.hidden = !String(row.dataset.gpiResultFile || '').toLocaleLowerCase().includes(needle);
          if (!row.hidden) visible++;
        });
        const empty = body.querySelector('[data-gpi-results-empty]');
        if (empty) empty.hidden = visible !== 0;
      };
      body.onclick = event => {
        const view = event.target.closest('[data-gpi-results-view]');
        if (view) {
          resultView = view.dataset.gpiResultsView === 'details' ? 'details' : 'list';
          syncProjectWorkflowAside();
          return;
        }
        const resource = event.target.closest('[data-gpi-resource-kind]');
        if (resource && host.openResource) { host.openResource(resource); return; }
        if (event.target.closest('[data-gpi-aside-pending]') && host.revealPendingReview) host.revealPendingReview();
      };
      body.onchange = event => {
        if (!event.target.matches('[data-gpi-results-sort]')) return;
        resultSort = ['recommended', 'name', 'size'].includes(event.target.value)
          ? event.target.value : 'recommended';
        syncProjectWorkflowAside();
      };
    }

    return { syncProjectWorkflowAside, togglePanel, layoutOptions };
  }

  window.EasyICU.guidedPi.declare('aside', { create });
})();
