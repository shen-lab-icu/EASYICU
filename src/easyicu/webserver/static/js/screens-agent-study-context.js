/* Agent Projects StudyContext adapter.
   Owner: persisted StudyContext -> Agent project projection, activation,
   bound-source resolution, and run-persistence ordering. */
(function () {
  'use strict';

  const jobContexts = new Map();
  const contextRuns = new Map();
  let runTokenSequence = 0;
  let guidedHandoff = null;

  function createRunChannel() {
    let current = null;

    function make(fields, tokenId) {
      const value = fields || {};
      return Object.freeze({
        token_id: tokenId || `run-${Date.now().toString(36)}-${++runTokenSequence}`,
        surface: String(value.surface || ''),
        study_id: String(value.study_id || ''),
        context_id: String(value.context_id || ''),
        context_revision: Number.isInteger(value.context_revision) ? value.context_revision : null,
        job_id: String(value.job_id || ''),
        question: String(value.question || ''),
        source_path: String(value.source_path || ''),
        study_mode: String(value.study_mode || 'analysis'),
        run_type: String(value.run_type || 'preflight'),
        provider: String(value.provider || 'mock'),
        project_seed_dir: String(value.project_seed_dir || ''),
      });
    }

    function same(left, right) {
      return !!(left && right && left.token_id === right.token_id);
    }

    return {
      start(fields) {
        current = make(fields);
        return current;
      },
      bind(token, fields) {
        const next = make(Object.assign({}, token || {}, fields || {}), token && token.token_id);
        if (same(current, token)) current = next;
        return next;
      },
      isCurrent(token) { return same(current, token); },
      clear(token) {
        if (!token || same(current, token)) current = null;
      },
      current() { return current; },
    };
  }

  function createJobMemory(storage, key, maxAgeMs) {
    const target = storage;
    const storageKey = String(key || 'easyicu.agent.activeJob.v1');
    const maxAge = Number(maxAgeMs) > 0 ? Number(maxAgeMs) : 24 * 60 * 60 * 1000;

    function read() {
      try {
        const raw = target.getItem(storageKey);
        const parsed = raw ? JSON.parse(raw) : null;
        if (!parsed) return {};
        if (parsed.job_id) {
          const legacyKey = parsed.study_id || parsed.study_context_id || '_legacy';
          return { [legacyKey]: parsed };
        }
        return parsed.jobs && typeof parsed.jobs === 'object' ? parsed.jobs : {};
      } catch (_) {
        return {};
      }
    }

    function write(jobs) {
      try {
        if (Object.keys(jobs || {}).length) target.setItem(storageKey, JSON.stringify({ version: 2, jobs }));
        else target.removeItem(storageKey);
      } catch (_) {}
    }

    return {
      remember(meta) {
        const value = Object.assign({ created_at: Date.now() }, meta || {});
        const studyId = value.study_id || value.study_context_id;
        if (!studyId || !value.job_id) return;
        const jobs = read();
        jobs[studyId] = value;
        write(jobs);
      },
      get(studyId) {
        const jobs = read();
        const meta = jobs[studyId] || null;
        if (!meta || !meta.job_id) return null;
        if (Date.now() - Number(meta.created_at || 0) > maxAge) {
          delete jobs[studyId];
          write(jobs);
          return null;
        }
        return meta;
      },
      clear(jobId, studyId) {
        const jobs = read();
        Object.keys(jobs).forEach(contextKey => {
          const meta = jobs[contextKey];
          if ((!studyId || contextKey === studyId) && (!jobId || !meta || meta.job_id === jobId)) delete jobs[contextKey];
        });
        write(jobs);
      },
    };
  }

  function store() { return window.EU_STUDY_CONTEXT || null; }

  function matchingLastRun(context) {
    const last = context && context.id ? (contextRuns.get(context.id) || window.EU_AGENT_LAST_RUN) : window.EU_AGENT_LAST_RUN;
    if (!last || !context) return null;
    const binding = last.context_binding || {};
    return last.study_context_id === context.id || binding.study_context_id === context.id ? last : null;
  }

  function runRows(context) {
    const last = matchingLastRun(context);
    if (!last) return [];
    const label = last.run_label || ('run ' + String(last.run_id || '').slice(-6));
    const kind = last.run_type === 'full' ? 'provider scaffold' : 'registry-backed preflight';
    const status = last.gate && last.gate.status === 'blocked' ? 'blocked' : 'complete';
    const duration = last.duration_sec == null ? '—' : `${last.duration_sec}s`;
    const when = last.completed_at || last.created_at || 'local';
    return [[label, [kind, kind], status, duration, [when, when]]];
  }

  function project(context) {
    if (!context || !context.id) return null;
    const source = context.data_source || {};
    const crossdbSelection = context.crossdb_selection || {};
    const selectedSources = Array.isArray(crossdbSelection.sources) ? crossdbSelection.sources : [];
    const cohort = context.cohort || {};
    const currentStage = String(context.current_stage || 'plan');
    const lastRun = matchingLastRun(context);
    const gate = lastRun && lastRun.gate && typeof lastRun.gate === 'object' ? lastRun.gate : null;
    const running = !!context.active_job_id || currentStage === 'analyze';
    const review = currentStage === 'review';
    const reviewBlocked = currentStage === 'review_blocked' || !!(gate && gate.status === 'blocked');
    const failed = currentStage === 'agent_failed' || currentStage === 'agent_cancelled';
    const status = running ? 'running' : (reviewBlocked ? 'review_blocked' : (review ? 'gate' : (failed ? 'idle' : 'idle')));
    const stage = running ? 2 : (review || reviewBlocked ? 3 : 0);
    const title = context.title || context.question || 'StudyContext project';
    const question = context.question || context.analysis_goal || title;
    const sourceLabel = selectedSources.length
      ? `${crossdbSelection.source_count || selectedSources.length} selected exports`
      : (source.label || source.database || 'StudyContext');
    return {
      id: context.id,
      name: [title, title],
      mode: 'analysis', status, stage,
      cohort: cohort.label || cohort.preset || cohort.review || 'configured cohort',
      source: [`${sourceLabel} · ${(context.modules || []).length} modules`, `${sourceLabel} · ${(context.modules || []).length} 个模块`],
      question: [question, question],
      runs: runRows(context),
      signed: false,
      planOnly: currentStage === 'crossdb_plan_only' && !!(context.confirmations && context.confirmations.crossdb_plan_only),
      gate,
      projectKind: 'study_context',
      studyContext: context,
    };
  }

  function projects() {
    const api = store();
    const rows = api && api.all ? api.all() : [];
    return (rows || []).map(project).filter(Boolean);
  }

  function activeId() {
    const api = store();
    const context = api && api.active ? api.active() : null;
    return context && context.id;
  }

  function has(id) {
    return projects().some(row => row.id === id);
  }

  function activate(id) {
    const api = store();
    return api && api.activate ? api.activate(id) : Promise.resolve(null);
  }

  function sourceFor(study, fallback) {
    // A Cross-DB receipt is the source boundary. Falling back to the active
    // registry export here made a two-source plan look as if it were bound to
    // one arbitrary export (including that export's path and stay count).
    const selection = study && study.studyContext && study.studyContext.crossdb_selection;
    const selectedSources = Array.isArray(selection && selection.sources) ? selection.sources : [];
    if (study && (study.planOnly || selectedSources.length > 1)) return null;
    const bound = study && study.studyContext && study.studyContext.data_source;
    if (!bound || !bound.path) return fallback || null;
    const registry = window.EU_WORKSPACE_REGISTRY || {};
    return (registry.sources || []).find(row => row.path === bound.path) || bound;
  }

  function persistForRun(study) {
    const api = store();
    if (!study || !study.studyContext || !api) return Promise.resolve(null);
    const current = api.active ? api.active() : null;
    if (!current || current.id !== study.studyContext.id) return activate(study.studyContext.id);
    return api.persist ? api.persist() : Promise.resolve(current);
  }

  async function prepareGuidedHandoff(study) {
    const projectId = String(study && (study.id || study.studyContext && study.studyContext.id) || '').trim();
    if (!projectId) {
      throw new Error('A research project is required for Guided Copilot handoff.');
    }
    const initialTitle = String(study.name && study.name[0] || projectId).trim();
    if (!study.studyContext || !study.studyContext.id) {
      guidedHandoff = Object.freeze({
        schema_version: 'easyicu.guided-project-handoff/1',
        project_id: projectId,
        project_title: initialTitle.slice(0, 160) || projectId,
        binding_receipt: null,
      });
      return guidedHandoff;
    }
    const context = await persistForRun(study);
    if (!context || !context.id || !Number.isInteger(context.revision)) {
      throw new Error('The StudyContext handoff could not be persisted.');
    }
    const title = String(context.title || study.name && study.name[0] || context.id).trim();
    const bindingReceipt = Object.freeze({
      schema_version: 'easyicu.pi-project-binding-handoff/1',
      project_id: String(context.id),
      project_title: title.slice(0, 160) || String(context.id),
      study_context_id: String(context.id),
      study_context_revision: context.revision,
    });
    guidedHandoff = Object.freeze({
      schema_version: 'easyicu.guided-project-handoff/1',
      project_id: bindingReceipt.project_id,
      project_title: bindingReceipt.project_title,
      binding_receipt: bindingReceipt,
    });
    return guidedHandoff;
  }

  function takeGuidedHandoff() {
    const receipt = guidedHandoff;
    guidedHandoff = null;
    return receipt;
  }

  /* Returns null, or a typed refusal: {code, reason}. The code is what
     gate-remedy.js turns into "what clears this" — the reason alone stated our
     implementation limit and left the user with no next step. */
  function runBlocker(study) {
    if (!study || !study.planOnly) return null;
    const t = window.t || ((en) => en);
    return {
      code: 'crossdb_plan_only',
      reason: t(
        'This Cross-DB handoff is plan-only. The current Agent runner consumes one export and cannot execute the aggregate multi-database payload yet.',
        '当前 Cross-DB 交接仅用于创建计划。Agent runner 目前只消费单个导出，尚不能执行跨库聚合载荷。',
      ),
    };
  }

  function markContextStage(contextId, stage, activeJobId, expectedJobId, contextRevision) {
    const api = store();
    if (!contextId || !api || !api.patchContext) return null;
    const options = { dirty: false, reason: 'agent-stage' };
    if (expectedJobId !== undefined) options.expectedActiveJobId = expectedJobId;
    const patch = {
      current_stage: stage,
      last_route: 'agent',
      active_job_id: activeJobId || null,
    };
    if (Number.isInteger(contextRevision)) patch.revision = contextRevision;
    return api.patchContext(contextId, patch, options);
  }

  function markContextRunning(contextId, jobId, contextRevision) {
    if (!contextId || !jobId) return;
    jobContexts.set(jobId, contextId);
    markContextStage(contextId, 'analyze', jobId, undefined, contextRevision);
  }

  function markRunning(study, jobId) {
    if (study && study.studyContext) markContextRunning(study.studyContext.id, jobId);
  }

  function terminalStage(status, result) {
    if (status === 'done') return result && result.gate && result.gate.status === 'blocked' ? 'review_blocked' : 'review';
    return status === 'cancelled' ? 'agent_cancelled' : 'agent_failed';
  }

  function markContextFinished(contextId, status, result, jobId, contextRevision) {
    const boundId = contextId || jobContexts.get(jobId);
    if (!boundId || !jobId) return null;
    const revision = Number.isInteger(contextRevision)
      ? contextRevision
      : (result && Number.isInteger(result.study_context_revision) ? result.study_context_revision : null);
    const updated = markContextStage(boundId, terminalStage(status, result), null, jobId, revision);
    jobContexts.delete(jobId);
    if (!updated) return null;
    if (status === 'done' && result) {
      const lastRun = Object.assign({}, result, { study_context_id: boundId });
      contextRuns.set(boundId, lastRun);
      const active = store() && store().active ? store().active() : null;
      if (active && active.id === boundId) window.EU_AGENT_LAST_RUN = lastRun;
    }
    return updated;
  }

  function markFinished(study, status, jobId, result) {
    const contextId = study && study.studyContext && study.studyContext.id;
    return markContextFinished(contextId, status, result, jobId);
  }

  function bindingNote(study) {
    if (!study || !study.studyContext) return '';
    const t = window.t || ((en) => en);
    const icon = window.icon || (() => '');
    const detail = study.planOnly
      ? t('Plan-only Cross-DB context: review and refine the multi-database question here, but execution stays blocked until Agent can consume the aggregate Cross-DB payload.', '仅计划的 Cross-DB 上下文：可以在这里审阅和完善跨库问题；在 Agent 能消费跨库聚合载荷前，执行保持拦截。')
      : t('Applied when the run starts: export path and research question. Informational until the analysis pipeline consumes them: cohort, modules, outcome, time window, and comparator.', '运行启动时会实际应用：导出路径与研究问题。队列、模块、结局、时间窗和比较组在分析 pipeline 使用前均明确标为信息性上下文。');
    return `<div class="note info mt-12"><div class="ico">${icon('shield', 13)}</div><div class="body"><div class="t">${t('StudyContext bound to this project', 'StudyContext 已绑定到此项目')}</div><div class="d">${detail}</div></div></div>`;
  }

  function submissionWarning(response) {
    const payload = response || {};
    const t = window.t || ((en) => en);
    const warnings = [];
    // There is deliberately no "the pointer did not sync but the job runs
    // anyway" warning: the server now refuses to start a run it could not
    // record against the StudyContext, so that state cannot reach this screen.
    if (payload.audit_warning) {
      warnings.push(t(
        'The job was accepted, but its submission audit event was not recorded. Treat provenance as incomplete until reviewed.',
        '任务已接受，但提交审计事件未成功记录。完成复核前，应将溯源视为不完整。',
      ));
    }
    return warnings.join(' · ');
  }

  function warningNote(warning) {
    if (!warning) return '';
    const t = window.t || ((en) => en);
    const icon = window.icon || (() => '');
    const esc = value => String(value || '').replace(/[&<>]/g, char => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' }[char]));
    return `<div class="note warn mt-12"><div class="ico">${icon('alert', 13)}</div><div class="body"><div class="t">${t('Nonfatal submission warning', '非致命提交警告')}</div><div class="d">${esc(warning)}</div></div></div>`;
  }

  function subscribe(callback) {
    window.addEventListener('easyicu:study-context', event => {
      if (!event.detail || event.detail.reason === 'sync') return;
      callback(event.detail.context || null, event.detail.reason || 'update');
    });
  }

  function hydrate() {
    const api = store();
    return api && api.hydrate ? api.hydrate() : Promise.resolve(null);
  }

  window.EU_AGENT_STUDY_CONTEXT = {
    activeId, activate, bindingNote, createJobMemory, createRunChannel, has, hydrate, markContextFinished, markContextRunning,
    markFinished, markRunning, persistForRun, prepareGuidedHandoff, projects, runBlocker, sourceFor, takeGuidedHandoff,
    submissionWarning, subscribe, warningNote,
  };
})();
