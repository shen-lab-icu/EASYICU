/* Agent Science Workbench submodule.
   Owner: Agent Projects Science tab. Keeps public science-workbench-inspired
   artifact history, reviewer gate, reusable protocol cards, and ICU-native renderers
   out of the already-large screens-agent.js owner file. */
(function () {
  'use strict';

  const state = {
    projectDir: null,
    loading: false,
    error: null,
    data: null,
    artifact: null,
    tab: 'Review',
    module: 'overview',
    renderer: 'concept_coverage_matrix',
    saved: {},
  };

  function esc(v) {
    return String(v == null ? '' : v).replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
  }
  function icon(name, size) {
    return window.icon ? window.icon(name, size || 13) : '';
  }
  function bi(en, zh) {
    return window.t ? window.t(en, zh) : en;
  }
  function settingEnabled(key, fallback) {
    const settings = window.EU_SETTINGS || {};
    if (Object.prototype.hasOwnProperty.call(settings, key)) return !!settings[key];
    return !!fallback;
  }
  function policyOf(data) {
    return (data && data.capability_policy) || window.EU_CAPABILITIES || {};
  }
  function policySetting(data, key, fallback) {
    const settings = (policyOf(data) && policyOf(data).settings) || {};
    if (Object.prototype.hasOwnProperty.call(settings, key)) return !!settings[key];
    return settingEnabled(key, fallback);
  }
  function capOf(data, id) {
    const caps = (policyOf(data) && policyOf(data).capabilities) || {};
    return caps[id] || {};
  }
  function repaint(cb) {
    if (typeof cb === 'function') cb();
    else if (window.__euRender) window.__euRender();
  }
  function loadSaved() {
    try {
      const raw = localStorage.getItem('easyicu.agent.science.savedProtocols.v1');
      state.saved = raw ? JSON.parse(raw) : {};
    } catch (_) {
      state.saved = {};
    }
  }
  function persistSaved() {
    try { localStorage.setItem('easyicu.agent.science.savedProtocols.v1', JSON.stringify(state.saved || {})); } catch (_) {}
  }
  function request(projectDir, onDone) {
    const dir = projectDir || '';
    if (state.projectDir === dir && (state.loading || state.data || state.error)) return;
    state.projectDir = dir;
    state.loading = true;
    state.error = null;
    state.data = null;
    state.artifact = null;
    // Request token: the workbench endpoint can be slow (it probes the
    // Zotero connector), so a fetch for a previously selected project can
    // resolve AFTER the current one and would otherwise overwrite it.
    const seq = (state.seq = (state.seq || 0) + 1);
    if (!window.EU_API || !window.EU_API.loadAgentScienceWorkbench) {
      state.loading = false;
      state.error = 'Science Workbench API is not available.';
      repaint(onDone);
      return;
    }
    window.EU_API.loadAgentScienceWorkbench(dir ? { project_dir: dir } : {}).then(data => {
      if (seq !== state.seq) return;
      state.loading = false;
      state.error = null;
      state.data = data;
      if (data && data.capability_policy) window.EU_CAPABILITIES = data.capability_policy;
      const items = data && data.artifact_history && Array.isArray(data.artifact_history.items)
        ? data.artifact_history.items
        : [];
      state.artifact = items[0] ? items[0].name : null;
      repaint(onDone);
    }).catch(err => {
      if (seq !== state.seq) return;
      state.loading = false;
      state.error = err.message || String(err);
      state.data = null;
      repaint(onDone);
    });
  }
  function artifactItems() {
    return state.data && state.data.artifact_history && Array.isArray(state.data.artifact_history.items)
      ? state.data.artifact_history.items
      : [];
  }
  function currentArtifact() {
    const items = artifactItems();
    return items.find(row => row.name === state.artifact) || items[0] || null;
  }
  function statusTone(status) {
    const s = String(status || '');
    if (s === 'passed' || s === 'not_applicable') return 'ok';
    if (s === 'failed') return 'bad';
    if (s === 'needs_review') return 'warn';
    return 'info';
  }
  function statusText(row) {
    const explicit = row && row.status_label;
    if (explicit) return explicit;
    const s = String((row && row.status) || '');
    if (s === 'passed') return 'passed / 已通过';
    if (s === 'needs_review') return 'needs review / 待审阅';
    if (s === 'waiting_for_run') return 'waiting / 等待运行';
    if (s === 'failed') return 'failed / 未通过';
    if (s === 'not_applicable') return 'not applicable / 不适用';
    if (s === 'unavailable') return 'unavailable / 不可用';
    return s || 'unknown / 未知';
  }
  function referenceCard(data) {
    const ref = (data && data.visual_reference) || {};
    const cues = Array.isArray(ref.design_cues) ? ref.design_cues : [];
    return `
      <div class="card pad ag-sci-reference">
        <div class="ag-sci-ref-grid">
          <div>
            <div class="eyebrow">${bi('Visual reference', '视觉参考')}</div>
            <div class="panel-title" style="font-size:15px;margin-top:4px;">${bi('Artifact preview with code, environment, and review tabs', 'Artifact 预览 + 代码、环境、审阅标签')}</div>
            <div class="panel-sub">${bi('Mapped from the public Claude Science screenshot, without loading remote media inside the local app.', '参考公开 Claude Science 截图的信息架构；本地应用内不加载远程媒体。')}</div>
            <div class="ag-sci-cues">${cues.map(c => `<span class="ag-sci-cue">${esc(c)}</span>`).join('')}</div>
          </div>
          <a class="btn sm" href="${esc(ref.article_url || '#')}" target="_blank" rel="noopener">${icon('file', 12)} ${bi('Open source', '打开来源')}</a>
        </div>
      </div>`;
  }
  function capabilityStackSection(data) {
    const skillsOn = policySetting(data, 'science_skills_enabled', true);
    const pubmedOn = policySetting(data, 'connector_pubmed_enabled', true);
    const zoteroOn = policySetting(data, 'connector_zotero_enabled', false);
    const mcpOn = policySetting(data, 'mcp_tools_enabled', false);
    const promptsOn = policySetting(data, 'prompt_contracts_enabled', true);
    const auditOn = policySetting(data, 'tool_audit_enabled', true);
    const remoteComputeOn = policySetting(data, 'remote_compute_enabled', false);
    const zotero = capOf(data, 'zotero_connector');
    const mcp = capOf(data, 'mcp_tools');
    const prompts = capOf(data, 'prompt_contracts');
    const audit = capOf(data, 'tool_audit');
    const remote = capOf(data, 'remote_compute');
    const allowedTools = Array.isArray(mcp.allowed_tools) ? mcp.allowed_tools.length : 0;
    const blockedTools = Array.isArray(mcp.blocked_tools) ? mcp.blocked_tools.length : 0;
    const promptRules = Array.isArray(prompts.rules) ? prompts.rules.length : 0;
    const connectorCount = [pubmedOn, zoteroOn].filter(Boolean).length;
    const rows = [
      {
        icon: 'layers',
        title: 'Skills / 技能',
        status: skillsOn ? 'on / 开启' : 'off / 关闭',
        tone: skillsOn ? 'ok' : 'warn',
        desc: skillsOn
          ? 'Reusable ICU workflows are available as local protocols. / 可复用 ICU 工作流已作为本地 protocol 可用。'
          : 'Reusable protocol shortcuts are hidden until Skills is enabled in Settings. / Settings 启用 Skills 前会隐藏 protocol 快捷能力。',
        chips: ['Idea to Agent', 'Evidence checks', 'Figure review'],
      },
      {
        icon: 'db',
        title: 'Connectors / 连接器',
        status: connectorCount ? `${connectorCount} enabled / ${connectorCount} 已启用` : 'off / 关闭',
        tone: connectorCount ? 'ok' : 'warn',
        desc: pubmedOn
          ? 'PubMed metadata can be used after each source-level opt-in. / 每个来源单独 opt-in 后可使用 PubMed 元数据。'
          : 'PubMed is disabled globally, so Idea Mining stays local-only. / PubMed 已全局关闭，Idea Mining 保持仅本地。',
        chips: [`PubMed ${pubmedOn ? 'on' : 'off'}`, `Zotero ${zotero.status || (zoteroOn ? 'checking' : 'off')}`, 'source opt-in'],
      },
      {
        icon: 'globe',
        title: 'MCP tools / MCP 工具',
        status: mcpOn ? `${allowedTools} allowed / ${allowedTools} 允许` : 'off / 关闭',
        tone: mcpOn ? 'info' : 'warn',
        desc: mcpOn
          ? 'External tools must pass scope and allowlist checks. / 外部工具必须通过作用域和白名单检查。'
          : 'The standard MCP boundary is visible but disabled until adapters are configured. / 标准 MCP 边界已展示，适配器配置前保持关闭。',
        chips: [`${allowedTools} allowed`, `${blockedTools} blocked`, 'tool scope'],
      },
      {
        icon: 'file',
        title: 'Prompt contracts / 提示词契约',
        status: promptsOn ? 'on / 开启' : 'off / 关闭',
        tone: promptsOn ? 'ok' : 'warn',
        desc: promptsOn
          ? 'Global prompts stay case-neutral; project rules live in protocols. / 全局提示词保持 case-neutral，项目规则写入 protocol。'
          : 'Prompt contract reminders are hidden, but existing project protocols remain unchanged. / 提示词契约提醒会隐藏，已有项目 protocol 不改变。',
        chips: [`${promptRules} active rules`, 'rubrics', 'no prompt pile-up'],
      },
      {
        icon: 'shield',
        title: 'Tool audit / 工具审计',
        status: auditOn ? 'on / 开启' : 'off / 关闭',
        tone: auditOn ? 'ok' : 'warn',
        desc: auditOn
          ? 'Claims, citations, calculations, hashes, and tool use are checked before draft release. / 草稿放行前检查论断、引用、计算、哈希和工具使用。'
          : 'Reviewer-check evidence remains visible, but tool-audit controls are disabled. / 审阅检查证据仍可见，但工具审计控制已关闭。',
        chips: [`${Number(audit.event_count || 0)} events`, 'sign-off', 'hash check'],
      },
      {
        icon: 'gear',
        title: 'Compute / 计算环境',
        status: remoteComputeOn ? `${remote.status || 'checking'} / 远程控制` : 'local only / 仅本地',
        tone: remoteComputeOn ? 'info' : 'ok',
        desc: remoteComputeOn
          ? 'Remote or HPC execution still requires credentials and artifact-return rules. / 远程或 HPC 执行仍需凭证和产物回传规则。'
          : 'Runs stay local-first unless remote compute is explicitly enabled. / 未显式启用远程计算时，运行保持本地优先。',
        chips: ['local first', remote.reason || 'backend policy', 'HPC control'],
      },
    ];
    return `
      <div class="ag-sci-section ag-sci-capstack">
        <div class="ag-sci-check-head">
          <div>
            <div class="eyebrow">Research tool stack / 研究工具栈</div>
            <div class="panel-sub">Current capability switches from Settings. / 这里显示 Settings 中当前启用的研究能力。</div>
          </div>
          <button class="btn sm" data-ag-sci-open-settings>${icon('gear', 12)} ${bi('Open Settings', '打开 Settings')}</button>
        </div>
        <div class="ag-sci-capgrid">
          ${rows.map(row => `
            <div class="ag-sci-capcard ${esc(row.tone)}">
              <div class="ag-sci-caphead">
                <span class="ag-sci-capicon">${icon(row.icon, 14)}</span>
                <div>
                  <b>${esc(row.title)}</b>
                  <em>${esc(row.status)}</em>
                </div>
              </div>
              <p>${esc(row.desc)}</p>
              <div class="ag-sci-card-meta">${row.chips.map(x => `<span class="chip">${esc(x)}</span>`).join('')}</div>
            </div>`).join('')}
        </div>
      </div>`;
  }
  function fmtMetric(value) {
    if (value == null) return '—';
    if (typeof value === 'number') return Number.isFinite(value) ? value.toLocaleString() : '—';
    return esc(value);
  }
  function stageStatusOf(status) {
    const s = String(status || '');
    if (s === 'passed' || s === 'reportable' || s.indexOf('ready_for_') === 0) return 'passed';
    if (s === 'failed' || s === 'idea_mining_unavailable') return 'failed';
    if (s === 'waiting_for_run' || s === 'waiting_for_idea_run') return 'waiting_for_run';
    return 'needs_review';
  }
  function moduleRows(data) {
    const summary = (data && data.run_summary) || {};
    const checklist = (data && data.fig5_checklist) || {};
    const pipeline = (data && data.discovery_pipeline) || {};
    const alignment = (data && data.feature_alignment) || {};
    const alignItems = Array.isArray(alignment.items) ? alignment.items : [];
    const artifacts = artifactItems();
    const protocols = data && Array.isArray(data.reusable_protocols) ? data.reusable_protocols : [];
    const renderers = data && Array.isArray(data.native_renderers) ? data.native_renderers : [];
    const readyRenderers = renderers.filter(row => row.can_render).length;
    const alignPassed = alignItems.filter(row => ['passed', 'not_applicable'].includes(String(row.status || ''))).length;
    const alignNeeds = alignItems.length - alignPassed;
    const checksPassed = checklist.passed_count || 0;
    const checksTotal = checklist.applicable_count || 0;
    return [
      {
        id: 'overview',
        icon: 'list',
        label: 'Overview / 总览',
        kicker: 'Current status / 当前状态',
        status: stageStatusOf(summary.status),
        metric: `${fmtMetric((summary.kpis || []).length || 0)} KPIs`,
        navMetric: `${fmtMetric((summary.kpis || []).length || 0)} KPIs`,
        detail: summary.next_action || bi('No active local Agent run.', '尚未打开本地 Agent run。'),
      },
      {
        id: 'discovery',
        icon: 'target',
        label: 'Discovery / 发现',
        kicker: 'Idea pipeline / 想法流程',
        status: stageStatusOf(pipeline.status),
        metric: pipeline.latest_run_id ? bi('local run', '本地 run') : bi('no run', '无 run'),
        navMetric: pipeline.latest_run_id ? 'run' : 'no run',
        detail: pipeline.status_label || bi('Waiting for Idea Mining run.', '等待 Idea Mining run。'),
      },
      {
        id: 'evidence',
        icon: 'shield',
        label: 'Evidence / 证据',
        kicker: 'Review package / 审阅包',
        status: checksTotal && checksPassed >= checksTotal ? 'passed' : 'needs_review',
        metric: `${fmtMetric(checksPassed)}/${fmtMetric(checksTotal)} checks`,
        navMetric: `${fmtMetric(checksPassed)}/${fmtMetric(checksTotal)}`,
        detail: checklist.next_action || bi('Review evidence artifacts before claims move forward.', '推进论断前先审阅证据产物。'),
      },
      {
        id: 'coverage',
        icon: 'check',
        label: 'Coverage / 覆盖',
        kicker: 'Workbench views / 工作台视图',
        status: alignItems.length ? (alignNeeds ? 'needs_review' : 'passed') : 'waiting_for_run',
        metric: `${fmtMetric(alignPassed)}/${fmtMetric(alignItems.length)} views`,
        navMetric: `${fmtMetric(alignPassed)}/${fmtMetric(alignItems.length)}`,
        detail: alignNeeds ? `${alignNeeds} ${bi('views need review', '个视图待审阅')}` : bi('All populated views are clear.', '已填充视图均已通过。'),
      },
      {
        id: 'resources',
        icon: 'file',
        label: 'Resources / 资源',
        kicker: 'Tool stack / 工具栈',
        status: renderers.length && readyRenderers === renderers.length ? 'passed' : 'needs_review',
        metric: `${fmtMetric(protocols.length)} skills · ${fmtMetric(readyRenderers)}/${fmtMetric(renderers.length)} previews`,
        navMetric: `${fmtMetric(protocols.length)} · ${fmtMetric(readyRenderers)}/${fmtMetric(renderers.length)}`,
        detail: 'Skills, connectors, MCP tools, prompt contracts, ICU previews. / 技能、连接器、MCP 工具、提示词契约和 ICU 预览。',
      },
    ];
  }
  function moduleNav(data) {
    const rows = moduleRows(data || {});
    if (!rows.some(row => row.id === state.module)) state.module = 'overview';
    return `
      <div class="ag-sci-module-nav" role="tablist" aria-label="${bi('Science Workbench modules', '科学工作台模块')}">
        ${rows.map(row => `
          <button id="ag-sci-module-tab-${esc(row.id)}" class="ag-sci-module-tab ${state.module === row.id ? 'on' : ''} ${esc(row.status)}" role="tab" aria-selected="${state.module === row.id ? 'true' : 'false'}" aria-controls="ag-sci-module-panel" data-ag-sci-module="${esc(row.id)}">
            <span class="ag-sci-module-icon">${icon(row.icon, 14)}</span>
            <span class="ag-sci-module-copy"><b>${esc(row.label)}</b></span>
            <small>${esc(row.navMetric || row.metric)}</small>
          </button>`).join('')}
      </div>`;
  }
  function overviewSection(data) {
    const rows = moduleRows(data || {}).filter(row => row.id !== 'overview');
    return `
      ${summaryCard(data || {})}
      <div class="ag-sci-section ag-sci-overview">
        <div class="ag-sci-check-head">
          <div>
            <div class="eyebrow">Module map / 模块地图</div>
            <div class="panel-sub">Current discovery, evidence, coverage, and reusable-resource status. / 当前发现、证据、覆盖范围与复用资源状态。</div>
          </div>
        </div>
        <div class="ag-sci-module-cards">
          ${rows.map(row => `
            <button class="ag-sci-module-card ${esc(row.status)}" data-ag-sci-module="${esc(row.id)}">
              <span class="ag-sci-module-card-top"><b>${esc(row.label)}</b><em>${esc(statusText(row))}</em></span>
              <strong>${esc(row.metric)}</strong>
              <small>${esc(row.detail)}</small>
            </button>`).join('')}
        </div>
      </div>`;
  }
  function scienceModuleBody(data) {
    const safe = data || {};
    if (state.module === 'discovery') return discoveryPipelineSection(safe);
    if (state.module === 'evidence') return checklistSection(safe) + workbenchLayout(safe);
    if (state.module === 'coverage') return featureAlignmentSection(safe);
    if (state.module === 'resources') return capabilityStackSection(safe) + protocolsSection(safe) + renderersSection(safe) + referenceCard(safe);
    return overviewSection(safe);
  }
  function summaryCard(data) {
    const summary = (data && data.run_summary) || {};
    const scope = summary.workflow_scope || (data && data.workflow_scope) || {};
    const kpis = Array.isArray(summary.kpis) ? summary.kpis : [];
    return `
      <div class="ag-sci-summary">
        <div class="ag-sci-summary-main">
          <div class="eyebrow">${bi('Run summary', '运行摘要')}</div>
          <div class="ag-sci-summary-title">${esc(summary.title || bi('No active local Agent run', '尚未打开本地 Agent run'))}</div>
          <div class="ag-sci-summary-sub">${esc(summary.source_label || '')}</div>
          <div class="row gap-8 wrap mt-8">
            <span class="pill ${summary.status === 'reportable' ? 'ok' : 'warn'}"><span class="dot"></span>${esc(summary.status_label || bi('Review locked', '审阅锁定'))}</span>
            <span class="pill ${esc(scope.tone || 'info')}"><span class="dot"></span>${esc(scope.label || bi('Workflow unclassified', '工作流未分类'))}</span>
            <span class="pill info"><span class="dot"></span>${bi('local artifacts only', '仅本地产物')}</span>
          </div>
          <div class="ag-sci-next">${esc(summary.next_action || '')}</div>
        </div>
        <div class="ag-sci-kpi-grid">
          ${kpis.map(row => `
            <div class="ag-sci-kpi">
              <span>${esc(row.label)}</span>
              <strong>${fmtMetric(row.value)}</strong>
              <em>${esc(row.detail || '')}</em>
            </div>`).join('')}
        </div>
      </div>`;
  }
  function checklistSection(data) {
    const checklist = (data && data.fig5_checklist) || {};
    const items = Array.isArray(checklist.items) ? checklist.items : [];
    const artifactNames = new Set(artifactItems().map(row => row.name));
    const progress = Math.max(0, Math.min(100, Number(checklist.progress || 0) * 100));
    return `
      <div class="ag-sci-section ag-sci-checklist">
        <div class="ag-sci-check-head">
          <div>
            <div class="eyebrow">${esc(checklist.title || bi('Evidence readiness checklist', '证据就绪清单'))}</div>
            <div class="panel-sub">${esc(checklist.description || bi('Checks whether this run has enough local evidence for review.', '检查当前 run 是否已有足够的本地证据进入审阅。'))}</div>
          </div>
          <div class="row gap-8 wrap" style="justify-content:flex-end;">
            <span class="pill ${checklist.candidate_for_fig5 ? 'ok' : 'info'}"><span class="dot"></span>${checklist.candidate_for_fig5 ? 'Fig5 candidate / Figure 5 候选' : 'not Fig5 candidate / 非 Figure 5 候选'}</span>
            <span class="pill ${progress >= 100 ? 'ok' : 'warn'}"><span class="dot"></span>${fmtMetric(checklist.passed_count || 0)}/${fmtMetric(checklist.applicable_count || 0)}</span>
          </div>
        </div>
        <div class="ag-sci-progress" aria-label="${bi('Checklist progress', '清单进度')}">
          <div class="ag-sci-progress-fill" style="width:${progress.toFixed(0)}%;"></div>
        </div>
        <div class="ag-sci-ready-list">
          ${items.map(row => {
            const canFocus = row.focus_artifact && artifactNames.has(row.focus_artifact);
            return `
              <div class="ag-sci-ready-item ${esc(row.status)}">
                <span class="ag-sci-dot"></span>
                <div>
                  <b>${esc(row.label)}</b>
                  <span>${esc(row.next_action)}</span>
                  <div class="ag-sci-ready-foot">
                    <span class="mono">${esc(row.evidence)}</span>
                    ${canFocus ? `<button class="btn sm" data-ag-sci-focus-art="${esc(row.focus_artifact)}">${icon('search', 12)} ${bi('Focus evidence', '聚焦证据')}</button>` : `<span class="chip">${bi('waiting for artifact', '等待产物')}</span>`}
                  </div>
                </div>
              </div>`;
          }).join('')}
        </div>
        <div class="ag-sci-next">${esc(checklist.next_action || '')}</div>
      </div>`;
  }
  function featureAlignmentSection(data) {
    const alignment = (data && data.feature_alignment) || {};
    const rows = Array.isArray(alignment.items) ? alignment.items : [];
    const artifactNames = new Set(artifactItems().map(row => row.name));
    return `
      <div class="ag-sci-section ag-sci-align">
        <div class="ag-sci-check-head">
          <div>
            <div class="eyebrow">Workbench coverage / 工作台覆盖</div>
            <div class="panel-sub">Shows which local evidence views are populated for this Agent or Idea Mining run. / 显示当前 Agent 或 Idea Mining run 已填充哪些本地证据视图。</div>
          </div>
          <span class="pill info"><span class="dot"></span>${rows.length}</span>
        </div>
        <div class="ag-sci-align-grid">
          ${rows.map(row => {
            const focus = row.focus_artifact && artifactNames.has(row.focus_artifact);
            return `
              <div class="ag-sci-align-item ${esc(row.status)}">
                <div>
                  <b>${esc(row.label)}</b>
                  <span>${esc(row.evidence || '')}</span>
                </div>
                ${focus ? `<button class="btn sm" data-ag-sci-focus-art="${esc(row.focus_artifact)}">${icon('search', 12)} ${bi('Check', '检查')}</button>` : `<span class="chip">${esc(statusText(row))}</span>`}
              </div>`;
          }).join('')}
        </div>
      </div>`;
  }
  function discoveryPipelineSection(data) {
    const pipeline = (data && data.discovery_pipeline) || {};
    const stages = Array.isArray(pipeline.stages) ? pipeline.stages : [];
    const ready = !!(pipeline.source_data_review_ready || pipeline.fig5_candidate_ready);
    const source = pipeline.source || {};
    const hasRun = !!pipeline.latest_run_id;
    return `
      <div class="ag-sci-section ag-sci-discovery">
        <div class="ag-sci-check-head">
          <div>
            <div class="eyebrow">Discovery pipeline / 发现流程</div>
            <div class="ag-sci-discovery-title">${esc(pipeline.title || bi('No local discovery candidate yet', '尚无本地 discovery 候选'))}</div>
            <div class="panel-sub">${hasRun ? esc([source.title, source.journal, source.year].filter(Boolean).join(' · ')) : 'Run Idea Mining to connect source signal, prior-art review, feasibility, plan, and Agent handoff. / 运行 Idea Mining 后串联来源线索、既有研究审阅、可行性、计划和 Agent 交接。'}</div>
          </div>
          <div class="row gap-8 wrap" style="justify-content:flex-end;">
            <span class="pill ${ready ? 'ok' : 'warn'}"><span class="dot"></span>${esc(pipeline.status_label || bi('Needs review', '待审阅'))}</span>
            <button class="btn sm" data-ag-sci-open-ideas>${icon('arrow', 12)} ${bi('Open Idea Mining', '打开 Idea Mining')}</button>
          </div>
        </div>
        <div class="ag-sci-discovery-stats">
          <div><span>${bi('Concepts', '概念')}</span><strong>${fmtMetric(pipeline.mapped_concept_count)}</strong></div>
          <div><span>${bi('Feature stats', '特征统计')}</span><strong>${fmtMetric(pipeline.feature_stat_count)}</strong></div>
          <div><span>${bi('Cohort n', '队列 n')}</span><strong>${fmtMetric(pipeline.cohort_entities)}</strong></div>
          <div><span>${bi('Run', '运行')}</span><strong class="mono">${esc(pipeline.latest_run_id || '—')}</strong></div>
        </div>
        <div class="ag-sci-discovery-lane">
          ${stages.map(row => `
            <div class="ag-sci-discovery-step ${esc(row.status)}">
              <span class="ag-sci-dot"></span>
              <div>
                <b>${esc(row.label)}</b>
                <em>${esc(row.evidence || '')}</em>
                <span>${esc(row.next_action || '')}</span>
              </div>
            </div>`).join('')}
        </div>
        <div class="ag-sci-next">${esc(pipeline.next_action || '')}</div>
      </div>`;
  }
  function workbenchLayout(data) {
    const items = artifactItems();
    const artifact = currentArtifact();
    const tabs = artifact && artifact.history_tabs ? Object.keys(artifact.history_tabs) : ['Code', 'Execution Log', 'Messages', 'Environment', 'Review'];
    const gate = data && data.reviewer_gate ? data.reviewer_gate : {};
    const checks = Array.isArray(gate.checks) ? gate.checks : [];
    return `
      <div class="ag-sci-layout">
        <div class="ag-sci-rail">
          <div class="ag-sci-rail-title">${bi('Artifacts', '产物')}</div>
          <div class="ag-sci-art-list">
            ${items.length ? items.map(row => `
              <button class="ag-sci-art-btn ${state.artifact === row.name ? 'on' : ''}" data-ag-sci-art="${esc(row.name)}">
                <b>${esc(row.title || row.name)}</b>
                <span class="mono" style="display:block;margin-top:3px;color:var(--ink-4);font-size:10px;">${esc(row.sha256_short || row.category || '')}</span>
              </button>`).join('') : `<div class="panel-sub">${bi('Open a local Agent run to populate artifact history.', '打开本地 Agent run 后会显示产物历史。')}</div>`}
          </div>
        </div>
        <div class="ag-sci-preview">
          ${artifact ? `
            <div class="ag-sci-art-head">
              <div>
                <div class="ag-sci-art-title">${esc(artifact.title || artifact.name)}</div>
                <div class="ag-sci-art-sub mono">${esc(artifact.name)}${artifact.sha256_short ? ' · ' + esc(artifact.sha256_short) : ''}</div>
              </div>
              <span class="pill ${artifact.provenance_complete ? 'ok' : 'warn'}"><span class="dot"></span>${artifact.provenance_complete ? bi('provenance complete', '溯源完整') : bi('needs provenance', '需补溯源')}</span>
            </div>
            <div class="ag-sci-tabbar">
              ${tabs.map(tab => `<button class="${state.tab === tab ? 'on' : ''}" data-ag-sci-tab="${esc(tab)}">${esc(tab)}</button>`).join('')}
            </div>
            <div class="ag-sci-tabpanel">${esc((artifact.history_tabs || {})[state.tab] || '')}</div>
            <div class="ag-sci-mini-grid">
              <div class="ag-sci-mini"><span>${bi('Inputs', '输入')}</span><strong>${(artifact.inputs || []).length}</strong></div>
              <div class="ag-sci-mini"><span>${bi('Evidence IDs', '证据 ID')}</span><strong>${(artifact.evidence_ids || []).length}</strong></div>
              <div class="ag-sci-mini"><span>${bi('Size', '大小')}</span><strong>${artifact.bytes == null ? '—' : Number(artifact.bytes || 0).toLocaleString() + ' B'}</strong></div>
            </div>` : `
            <div class="state-hero empty-state" style="min-height:270px;">
              <div class="glyph">${icon('file', 28)}</div>
              <div class="st-t">${bi('Artifact history waiting for a run', '等待运行产物历史')}</div>
              <div class="st-d">${bi('The same panel will show Code, Execution Log, Messages, Environment, and Review for each local artifact.', '同一面板会为每个本地产物展示 Code、Execution Log、Messages、Environment 和 Review。')}</div>
            </div>`}
        </div>
        <div class="ag-sci-side">
        <div class="ag-sci-side-title">${bi('Reviewer checks', '审阅检查')}</div>
          <div class="row gap-8 wrap mt-8">
            <span class="pill ${gate.reportable ? 'ok' : 'warn'}"><span class="dot"></span>${gate.reportable ? bi('reportable', '可报告') : bi('not reportable', '不可报告')}</span>
            <span class="pill ${gate.signed ? 'ok' : 'info'}"><span class="dot"></span>${gate.signed ? bi('signed', '已签署') : bi('unsigned', '未签署')}</span>
          </div>
          <div class="ag-sci-checks">
            ${checks.map(row => `
              <div class="ag-sci-check ${esc(row.status)}">
                <span class="ag-sci-dot"></span>
                <div><b>${esc(row.label || row.id)}</b><span>${esc(row.detail || row.evidence || row.status)}</span></div>
              </div>`).join('')}
          </div>
        </div>
      </div>`;
  }
  function protocolsSection(data) {
    const rows = data && Array.isArray(data.reusable_protocols) ? data.reusable_protocols : [];
    const skillsOn = policySetting(data, 'science_skills_enabled', true);
    return `
      <div class="ag-sci-section">
        <div class="row" style="justify-content:space-between;align-items:baseline;">
          <div><div class="eyebrow">${bi('Reusable ICU protocols / skills', '可复用 ICU protocol / skill')}</div><div class="panel-sub">${bi('Successful handoffs can be saved as repeatable local protocols without changing global prompts.', '成功交接可保存成可复用本地 protocol，不改全局 prompt。')}</div></div>
          <span class="pill info"><span class="dot"></span>${rows.length}</span>
        </div>
        <div class="ag-sci-card-grid">
          ${rows.length ? rows.map(row => {
            const saved = !!state.saved[row.id];
            return `
            <div class="ag-sci-protocol">
              <div class="ag-sci-card-head">
                <div><div class="ag-sci-card-title">${esc(row.title)}</div><div class="ag-sci-card-desc">${esc(row.scope)}</div></div>
                <span class="pill ${saved ? 'ok' : 'info'}"><span class="dot"></span>${saved ? bi('saved', '已保存') : esc(row.stage)}</span>
              </div>
              <div class="ag-sci-card-meta">
                ${(row.outputs || []).slice(0, 3).map(x => `<span class="chip">${esc(x)}</span>`).join('')}
              </div>
              <div class="ag-sci-save">
                <span class="mono" style="font-size:10.5px;color:var(--ink-4);">${esc(row.id)}</span>
                <button class="btn sm" data-ag-sci-save="${esc(row.id)}">${saved ? icon('check', 12) + ' ' + bi('Saved', '已保存') : icon('download', 12) + ' ' + bi('Save protocol', '保存 protocol')}</button>
              </div>
            </div>`;
          }).join('') : `
            <div class="state-hero empty-state ag-sci-protocol-empty">
              <div class="glyph">${icon('layers', 24)}</div>
              <div class="st-t">${skillsOn ? bi('No local protocols returned', '暂无本地 protocol') : bi('Skills disabled in Settings', 'Settings 已关闭 Skills')}</div>
              <div class="st-d">${skillsOn ? bi('Protocols appear here after the backend registry is available.', '后端 registry 可用后会在这里显示 protocol。') : bi('Turn on Research skills in Settings to show reusable ICU workflow protocols.', '在 Settings 打开 Research skills 后，会显示可复用 ICU 工作流 protocol。')}</div>
            </div>`}
        </div>
      </div>`;
  }
  function rendererPreview(row) {
    const p = row.preview || {};
    if (row.id === 'concept_coverage_matrix') {
      const rows = Array.isArray(p.rows) ? p.rows : [];
      return `<div class="ag-sci-bars">${rows.length ? rows.map(r => {
        const v = Math.max(0, Math.min(100, Number(r.value || 0)));
        return `<div class="ag-sci-bar"><span>${esc(r.label)}</span><div class="ag-sci-track"><div class="ag-sci-fill" style="width:${v}%;"></div></div><b>${Number.isFinite(v) ? v.toFixed(0) + '%' : '—'}</b></div>`;
      }).join('') : `<div class="panel-sub">${bi('Coverage rows will render after a run.', '运行后会渲染覆盖率行。')}</div>`}</div>`;
    }
    if (row.id === 'claim_evidence_graph') {
      const claims = Array.isArray(p.claims) ? p.claims : [];
      return `<div class="ag-sci-graph">${claims.length ? claims.map(c => `<div class="ag-sci-edge"><b>${esc(c.label)}</b><span>${esc((c.evidence_ids || []).join(' + ') || 'no evidence id')}</span></div>`).join('') : `<div class="panel-sub">${bi('Claim graph appears when a locked draft exists.', '存在锁定草稿时会显示论断图谱。')}</div>`}</div>`;
    }
    if (row.id === 'cohort_attrition') {
      const groups = Array.isArray(p.groups) ? p.groups : [];
      return `<div class="ag-sci-graph"><div class="ag-sci-edge"><b>${bi('Denominator', '分母')}</b><span>${p.denominator == null ? '—' : Number(p.denominator || 0).toLocaleString()}</span></div>${groups.map(g => `<div class="ag-sci-edge"><b>${esc(g.label || g.id)}</b><span>${Number(g.entities || 0).toLocaleString()}</span></div>`).join('')}</div>`;
    }
    const events = Array.isArray(p.events) ? p.events : [];
    return `<div class="ag-sci-graph">${events.length ? events.map(e => `<div class="ag-sci-edge"><b>${esc(e.status)}</b><span>${esc(e.label)}</span></div>`).join('') : `<div class="panel-sub">${bi('Run phase lane appears after review checks are available.', '存在审阅检查后会显示运行阶段时间线。')}</div>`}</div>`;
  }
  function renderersSection(data) {
    const rows = data && Array.isArray(data.native_renderers) ? data.native_renderers : [];
    return `
      <div class="ag-sci-section">
        <div><div class="eyebrow">${bi('ICU-native views', 'ICU 原生视图')}</div><div class="panel-sub">${bi('Views built for ICU objects — concepts, cohorts, time lanes, and claim graphs — instead of generic charts.', '为 ICU 对象打造的视图 —— 概念、队列、时间线、论断图谱 —— 而非通用图表。')}</div></div>
        <div class="ag-sci-card-grid">
          ${rows.map(row => `
            <div class="ag-sci-renderer">
              <div class="ag-sci-card-head">
                <div><div class="ag-sci-card-title">${esc(row.title)}</div><div class="ag-sci-card-desc">${esc(row.description)}</div></div>
                <span class="pill ${row.can_render ? 'ok' : 'warn'}"><span class="dot"></span>${row.can_render ? bi('ready', '可渲染') : bi('waiting', '等待数据')}</span>
              </div>
              <div class="ag-sci-card-meta">${(row.available_artifacts || row.artifact_names || []).slice(0, 3).map(x => `<span class="chip">${esc(x)}</span>`).join('')}</div>
              ${rendererPreview(row)}
            </div>`).join('')}
        </div>
      </div>`;
  }
  function render(ctx) {
    loadSaved();
    const live = ctx && ctx.live;
    const data = state.data;
    if (live && live.project_dir) request(live.project_dir, ctx && ctx.repaint);
    if (!live || !live.project_dir) request('', ctx && ctx.repaint);
    return `
      <div class="card pad">
        <div class="row" style="justify-content:space-between;align-items:baseline;margin-bottom:12px;">
          <div>
            <div class="eyebrow">${bi('Science Workbench · advanced view', '科学工作台 · 进阶视图')}</div>
            <div class="panel-title" style="font-size:15px;margin-top:4px;">${bi('The same run, opened at the evidence level', '同一次运行 · 展开到证据层')}</div>
            <div class="panel-sub">${bi('This is not a separate tool: it re-opens the run for this study (the one in Runs / Outputs) at the artifact + evidence level, so you can audit what each claim rests on. Optional — Overview and Draft cover most reviews.', '这不是独立工具：它把本研究在“运行/产出”里的同一次运行，展开到产物 + 证据层，方便你核查每条结论的依据。可选 —— 多数审阅看“概览”和“草稿”就够了。')}</div>
          </div>
          <button class="btn sm" data-ag-sci-refresh>${icon('history', 12)} ${bi('Refresh', '刷新')}</button>
        </div>
        ${state.loading ? `<div class="note info"><div class="ico">${icon('file', 16)}</div><div class="body"><span class="t">${bi('Loading science workbench', '正在加载科学工作台')}</span><span class="d">${bi('Reading bounded local review artifacts only.', '仅读取有界本地审阅产物。')}</span></div></div>` : ''}
        ${state.error ? `<div class="note warn"><div class="ico">${icon('alert', 16)}</div><div class="body"><span class="t">${bi('Science workbench failed', '科学工作台加载失败')}</span><span class="d">${esc(state.error)}</span></div></div>` : ''}
        ${moduleNav(data || {})}
        <div id="ag-sci-module-panel" class="ag-sci-module-panel" role="tabpanel" aria-labelledby="ag-sci-module-tab-${esc(state.module)}">
          ${scienceModuleBody(data || {})}
        </div>
      </div>`;
  }
  function wire(root, ctx) {
    const host = root.querySelector('#agHost');
    if (!host) return;
    host.querySelectorAll('[data-ag-sci-refresh]').forEach(btn => btn.addEventListener('click', () => {
      const live = ctx && ctx.live;
      state.projectDir = null;
      request(live && live.project_dir ? live.project_dir : '', ctx && ctx.repaint);
    }));
    host.querySelectorAll('[data-ag-sci-module]').forEach(btn => btn.addEventListener('click', () => {
      state.module = btn.dataset.agSciModule || 'overview';
      repaint(ctx && ctx.repaint);
    }));
    host.querySelectorAll('[data-ag-sci-art]').forEach(btn => btn.addEventListener('click', () => {
      state.artifact = btn.dataset.agSciArt || null;
      repaint(ctx && ctx.repaint);
    }));
    host.querySelectorAll('[data-ag-sci-tab]').forEach(btn => btn.addEventListener('click', () => {
      state.tab = btn.dataset.agSciTab || 'Review';
      repaint(ctx && ctx.repaint);
    }));
    host.querySelectorAll('[data-ag-sci-focus-art]').forEach(btn => btn.addEventListener('click', () => {
      state.module = 'evidence';
      state.artifact = btn.dataset.agSciFocusArt || null;
      state.tab = 'Review';
      repaint(ctx && ctx.repaint);
    }));
    host.querySelectorAll('[data-ag-sci-open-ideas]').forEach(btn => btn.addEventListener('click', () => {
      location.hash = '#ideas';
    }));
    host.querySelectorAll('[data-ag-sci-open-settings]').forEach(btn => btn.addEventListener('click', () => {
      location.hash = '#settings';
    }));
    host.querySelectorAll('[data-ag-sci-save]').forEach(btn => btn.addEventListener('click', () => {
      const id = btn.dataset.agSciSave;
      if (!id) return;
      state.saved[id] = { saved_at: new Date().toISOString(), source: state.projectDir || 'empty_state' };
      persistSaved();
      repaint(ctx && ctx.repaint);
    }));
  }

  window.EU_AGENT_SCIENCE = { request, render, wire };
})();
