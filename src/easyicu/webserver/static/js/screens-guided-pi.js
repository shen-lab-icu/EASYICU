/* Pi AgentSession client for Guided Copilot.
   Owner: Pi chat/session/tool UX only. Study cards and scientific execution
   remain in their existing EasyICU owners. */
(function () {
  'use strict';

  const state = {
    host: null, conv: null, runtime: null, sessions: [], session: null,
    messages: [], loading: true, creating: false, busy: false, jobId: '',
    source: null, childSource: null, childJobId: '', error: '', shell: 'pi', draft: '', setupSaving: false,
    showSetup: false, availableModels: [], project: null,
    projectInitialization: null, workflow: null,
    agentMode: 'research', accessMode: 'assist', pendingAuthorityRebind: false,
    demoMode: false, demoScrollTopPending: false, currentTurnResources: [],
  };

  const ACCESS_MODE_GRANTS = Object.freeze({
    ask: Object.freeze([]),
    assist: Object.freeze(['idea', 'literature', 'configure', 'run', 'workspace_write', 'mcp_read']),
    full: Object.freeze(['idea', 'literature', 'configure', 'extract', 'run', 'provider_run', 'cancel', 'workspace_write', 'mcp_read']),
  });

  function tr(en, zh) { return window.EU_LANG === 'zh' ? zh : en; }
  function esc(value) {
    return String(value == null ? '' : value)
      .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;').replace(/'/g, '&#39;');
  }
  function assistantTextHtml(value) {
    const renderer = window.EU_GUIDED_PI_MARKDOWN;
    return renderer && typeof renderer.render === 'function'
      ? renderer.render(value)
      : esc(value).replace(/\n/g, '<br>');
  }
  function api() { return window.EU_API || {}; }
  function isStaticPreview() { return window.location && window.location.protocol === 'file:'; }
  function runtimeReady() { return !!(state.runtime && state.runtime.status === 'ready'); }
  function projectId() { return String((state.project && state.project.id) || '').trim(); }
  function agentMode() {
    return (state.session && state.session.agent_mode) || state.agentMode || 'research';
  }
  function accessModeLabel(mode) {
    if (mode === 'ask') return tr('Ask first', '请求访问');
    if (mode === 'full') return tr('Full access', '完全访问');
    return tr('Auto-approve', '自动审批');
  }
  function turnGrants() {
    const grants = ACCESS_MODE_GRANTS[state.accessMode] || ACCESS_MODE_GRANTS.assist;
    return grants.filter(action => action !== 'workspace_write' || agentMode() === 'workspace');
  }
  function setShell(shell) {
    state.shell = shell === 'pi' ? 'pi' : 'legacy';
    if (state.conv) state.conv.classList.toggle('pi-active', state.shell === 'pi');
    render();
  }
  function rememberSession(id) {
    const key = projectId() ? 'easyicu_pi_copilot_session:' + encodeURIComponent(projectId()) : '';
    if (!key) return;
    try {
      if (id) localStorage.setItem(key, id);
      else localStorage.removeItem(key);
    } catch (e) {}
  }
  function rememberedSession() {
    const key = projectId() ? 'easyicu_pi_copilot_session:' + encodeURIComponent(projectId()) : '';
    if (!key) return '';
    try { return localStorage.getItem(key) || ''; } catch (e) { return ''; }
  }
  function errorText(error) {
    if (!error) return '';
    if (error.code === 'pi_session_authority_stale') {
      return tr('The study binding changed after this conversation was saved. Rebind it before continuing.', '这段对话保存后研究绑定发生了变化，请先重新绑定再继续。');
    }
    if (error.code === 'pi_provider_auth_failed') {
      return tr('The model service rejected this API credential.', '模型服务拒绝了这个 API 凭据，请检查后重试。');
    }
    if (error.code === 'pi_provider_model_unavailable') {
      return tr('The selected model was not reported by this service.', '该服务没有返回所选模型，请从下方发现的模型中选择。');
    }
    if (error.code === 'pi_provider_connection_failed') {
      return tr('EasyICU could not reach the model service.', 'EasyICU 无法连接到模型服务，请检查地址和服务状态。');
    }
    if (error.code === 'pi_session_project_mismatch') {
      return tr('That Pi conversation belongs to another research project.', '该 Pi 对话属于另一个研究项目，不能在当前项目中打开。');
    }
    if (isStaticPreview() && String(error.message || '').includes('Failed to fetch')) {
      return tr('This is a static preview without the EasyICU backend. Start EasyICU and open http://127.0.0.1:8765/#guided.', '这是不带 EasyICU 后端的静态预览。请启动 EasyICU，再打开 http://127.0.0.1:8765/#guided。');
    }
    return String(error.message || error.code || error);
  }

  function providerPreset(config, runtime) {
    const transport = config.api_transport || runtime.api_transport || 'openai-completions';
    const base = String(config.base_url || '').toLowerCase();
    if (transport === 'anthropic-messages') return 'anthropic';
    if (transport === 'google-generative-ai') return 'google';
    if (base.includes('api.openai.com')) return 'openai';
    if (base.includes('127.0.0.1:8317') || base.includes('localhost:8317')) return 'cliproxyapi';
    return 'custom-openai';
  }

  function option(value, selected, label) {
    return `<option value="${value}" ${value === selected ? 'selected' : ''}>${label}</option>`;
  }
  function sessionIsStale() {
    return !!(state.session && state.session.stale && state.session.stale.stale);
  }

  function timeMs(value) {
    const parsed = Date.parse(String(value || ''));
    return Number.isFinite(parsed) ? parsed : Date.now();
  }
  function durationText(startedAt, endedAt) {
    const elapsed = Math.max(0, Number(endedAt || Date.now()) - Number(startedAt || Date.now()));
    if (elapsed < 1000) return `${elapsed} ms`;
    const seconds = elapsed / 1000;
    if (seconds < 60) {
      const value = seconds < 10 ? seconds.toFixed(1) : Math.round(seconds);
      return tr(`${value}s`, `${value} 秒`);
    }
    const minutes = Math.floor(seconds / 60);
    const remainder = Math.round(seconds % 60);
    return tr(
      remainder ? `${minutes}m ${remainder}s` : `${minutes}m`,
      remainder ? `${minutes} 分 ${remainder} 秒` : `${minutes} 分`,
    );
  }
  function iconHtml(name, size) {
    return typeof window.icon === 'function' ? window.icon(name, size || 16, 1.55) : '';
  }
  function toolIcon(name) {
    const tool = String(name || '');
    if (/preview/.test(tool)) return 'globe';
    if (/read_project|write_project/.test(tool)) return 'file';
    if (/edit_project/.test(tool)) return 'edit';
    if (/check_project/.test(tool)) return 'check';
    if (/list_project/.test(tool)) return 'folder';
    if (/load_skill/.test(tool)) return 'wand';
    if (/update|replan/.test(tool)) return 'edit';
    if (/run$|extraction/.test(tool)) return 'play';
    if (/resume/.test(tool)) return 'refresh';
    if (/cancel/.test(tool)) return 'stop';
    if (/literature|evidence|validation|blocker|interpretation/.test(tool)) return 'shield';
    if (/workflow|manuscript|idea/.test(tool)) return 'list';
    if (/artifact|plan|step/.test(tool)) return 'list';
    if (/workspace|capability/.test(tool)) return 'db';
    if (/context/.test(tool)) return 'file';
    return 'spark';
  }
  function activityIcon(step) {
    if (!step) return 'spark';
    if (step.kind === 'submitted') return 'arrow';
    if (step.kind === 'turn' || step.kind === 'retry') return 'refresh';
    if (step.kind === 'assistant') return 'wand';
    if (step.kind === 'tool') return toolIcon(step.toolName);
    if (step.kind === 'pipeline') {
      if (/artifact|report|manuscript/.test(step.step || '')) return 'file';
      if (/gate|valid|audit|evidence/.test(step.step || '')) return 'shield';
      if (/plan/.test(step.step || '')) return 'list';
      return 'play';
    }
    if (step.kind === 'compaction') return 'layers';
    if (step.kind === 'failed') return 'alert';
    if (step.kind === 'cancelled') return 'stop';
    if (step.kind === 'settled') return 'check';
    return 'spark';
  }
  function resourceName(resource) {
    return resource && String(resource.label || resource.artifact || resource.file || '').trim();
  }
  function resourceKey(resource) {
    if (!resource) return '';
    if (resource.kind === 'literature_source') return `literature:${resource.pmid || resource.doi || resource.url || ''}`;
    if (resource.kind === 'demo_artifact') return `demo:${resource.artifact || ''}`;
    if (resource.kind === 'data_package_review') return `data-package:${resource.study_context_id || ''}:${resource.study_revision || 0}:${resource.review_sha256 || ''}`;
    return resource.kind === 'research_artifact' || resource.kind === 'research_document'
      ? `research:${resource.run_id || ''}:${resource.artifact || ''}`
      : `${resource.kind || 'file'}:${resource.file || ''}`;
  }
  function resourceLabel(resource) {
    if (!resource) return '';
    if (resource.kind === 'demo_artifact' && window.EU_GUIDED_PI_DEMO && typeof window.EU_GUIDED_PI_DEMO.artifactLabel === 'function') {
      return window.EU_GUIDED_PI_DEMO.artifactLabel(resource.artifact || resource.label || '');
    }
    if (resource.kind === 'research_artifact' && window.AGENT_RENDER && typeof window.AGENT_RENDER.artifactTitle === 'function') {
      return window.AGENT_RENDER.artifactTitle(resource.artifact || resource.label || '');
    }
    return resourceName(resource);
  }
  function resourceButton(resource, label) {
    if (!resource) return '';
    const kind = resource.kind === 'demo_artifact' ? 'demo_artifact' : (resource.kind === 'data_package_review' ? 'data_package_review' : (resource.kind === 'research_document' ? 'research_document' : (resource.kind === 'research_artifact' ? 'research_artifact' : (resource.kind === 'literature_source' ? 'literature_source' : (resource.kind === 'webpage' ? 'webpage' : 'file')))));
    return `<button class="gpi-resource-link" type="button"
      data-gpi-resource-kind="${esc(kind)}"
      data-gpi-resource-file="${esc(resource.file || '')}"
      data-gpi-resource-run="${esc(resource.run_id || '')}"
      data-gpi-resource-artifact="${esc(resource.artifact || '')}"
      data-gpi-resource-label="${esc(resource.kind === 'demo_artifact' ? (resource.title || resourceLabel(resource)) : (resource.label || resource.artifact || resource.file || ''))}"
      data-gpi-resource-media="${esc(resource.media_type || 'text/plain')}"
      data-gpi-resource-url="${esc(resource.url || '')}"
      data-gpi-resource-title="${esc(resource.title || resource.label || '')}"
      data-gpi-resource-year="${esc(resource.year || '')}"
      data-gpi-resource-venue="${esc(resource.venue || '')}"
      data-gpi-resource-relevance="${esc(resource.relevance || '')}"
      data-gpi-resource-doi="${esc(resource.doi || '')}"
      data-gpi-resource-pmid="${esc(resource.pmid || '')}"
      data-gpi-resource-study="${esc(resource.study_context_id || '')}"
      data-gpi-resource-revision="${esc(resource.study_revision == null ? '' : resource.study_revision)}"
      data-gpi-resource-digest="${esc(resource.review_sha256 || resource.sha256 || '')}">${esc(label || resourceLabel(resource))}</button>`;
  }
  function toolLabel(name, resource) {
    const labels = {
      easyicu_workspace_status: tr('Check workspace status', '检查工作区状态'),
      easyicu_list_data_sources: tr('List registered data sources', '列出已登记数据源'),
      easyicu_inspect_data_package: tr('Review data package', '审阅数据包'),
      easyicu_inspect_workflow: tr('Inspect research workflow', '检查科研流程'),
      easyicu_inspect_context: tr('Inspect study context', '读取研究配置'),
      easyicu_inspect_plan: tr('Inspect scientific plan', '读取科学计划'),
      easyicu_inspect_literature: tr('Inspect literature evidence', '读取文献证据'),
      easyicu_inspect_capability: tr('Inspect capabilities', '检查可用能力'),
      easyicu_inspect_run: tr('Inspect run status', '读取运行状态'),
      easyicu_inspect_step: tr('Inspect plan step', '读取计划步骤'),
      easyicu_inspect_validation: tr('Inspect validation', '读取验证状态'),
      easyicu_list_artifacts: tr('List run artefacts', '列出运行产物'),
      easyicu_inspect_evidence: tr('Inspect evidence', '读取证据状态'),
      easyicu_explain_blocker: tr('Explain blocker', '解释阻断原因'),
      easyicu_inspect_interpretation: tr('Interpret validated results', '解读已验证结果'),
      easyicu_inspect_manuscript: tr('Inspect manuscript draft', '读取论文草稿'),
      easyicu_update_study_context: tr('Save study setup', '保存研究配置'),
      easyicu_mine_ideas: tr('Mine research ideas', '发掘研究想法'),
      easyicu_search_literature: tr('Search PubMed literature', '检索 PubMed 文献'),
      easyicu_prepare_idea_handoff: tr('Prepare idea plan', '准备想法计划'),
      easyicu_accept_idea_handoff: tr('Accept selected idea', '接受所选想法'),
      easyicu_start_extraction: tr('Start feature extraction', '启动特征提取'),
      easyicu_run: tr('Start EasyICU run', '启动 EasyICU 运行'),
      easyicu_resume: tr('Resume EasyICU work', '恢复 EasyICU 任务'),
      easyicu_cancel: tr('Cancel EasyICU job', '取消 EasyICU 任务'),
      easyicu_request_replan: tr('Request replan', '请求重新规划'),
      easyicu_load_skill: tr('Load web-prototype skill', '加载网页原型技能'),
      easyicu_list_extensions: tr('List frozen extensions', '列出固化扩展'),
      easyicu_call_mcp_tool: tr('Call allowlisted MCP tool', '调用白名单 MCP 工具'),
      easyicu_list_project_files: tr('List project files', '列出项目文件'),
      easyicu_read_project_file: tr('Read project file', '读取项目文件'),
      easyicu_write_project_file: tr('Write project file', '写入项目文件'),
      easyicu_edit_project_file: tr('Edit project file', '编辑项目文件'),
      easyicu_check_project_file: tr('Check project file', '检查项目文件'),
      easyicu_preview_project_file: tr('Prepare web preview', '准备网页预览'),
    };
    const label = labels[String(name || '')] || String(name || tr('EasyICU tool', 'EasyICU 工具'));
    const file = resourceName(resource);
    return file ? `${label} · ${file}` : label;
  }
  function completedToolLabel(name, resource) {
    const labels = {
      easyicu_workspace_status: tr('Checked workspace status', '已检查工作区状态'),
      easyicu_list_data_sources: tr('Listed registered data sources', '已列出已登记数据源'),
      easyicu_inspect_workflow: tr('Read research workflow', '已读取科研流程'),
      easyicu_inspect_context: tr('Read study setup', '已读取研究配置'),
      easyicu_inspect_plan: tr('Read scientific plan', '已读取科学计划'),
      easyicu_inspect_literature: tr('Read literature evidence', '已读取文献证据'),
      easyicu_inspect_capability: tr('Checked capabilities', '已检查可用能力'),
      easyicu_inspect_run: tr('Read run status', '已读取运行状态'),
      easyicu_inspect_step: tr('Read plan step', '已读取计划步骤'),
      easyicu_inspect_validation: tr('Read validation', '已读取验证状态'),
      easyicu_list_artifacts: tr('Listed run artefacts', '已列出运行产物'),
      easyicu_inspect_evidence: tr('Read evidence', '已读取证据状态'),
      easyicu_explain_blocker: tr('Read blocker details', '已读取阻断原因'),
      easyicu_inspect_interpretation: tr('Organized evidence-bound interpretation', '已整理证据约束的结果解读'),
      easyicu_inspect_manuscript: tr('Read manuscript draft', '已读取论文草稿'),
      easyicu_update_study_context: tr('Saved study setup', '已保存研究配置'),
      easyicu_mine_ideas: tr('Mined research ideas', '已发掘研究想法'),
      easyicu_search_literature: tr('Searched PubMed literature', '已检索 PubMed 文献'),
      easyicu_prepare_idea_handoff: tr('Prepared idea plan', '已准备想法计划'),
      easyicu_accept_idea_handoff: tr('Accepted selected idea', '已接受所选想法'),
      easyicu_start_extraction: tr('Started feature extraction', '已启动特征提取'),
      easyicu_run: tr('Started EasyICU run', '已启动 EasyICU 运行'),
      easyicu_resume: tr('Resumed EasyICU work', '已恢复 EasyICU 任务'),
      easyicu_cancel: tr('Cancelled EasyICU job', '已取消 EasyICU 任务'),
      easyicu_request_replan: tr('Requested replan', '已请求重新规划'),
      easyicu_load_skill: tr('Loaded web-prototype skill', '已加载网页原型技能'),
      easyicu_list_extensions: tr('Listed frozen extensions', '已列出固化扩展'),
      easyicu_call_mcp_tool: tr('Called allowlisted MCP tool', '已调用白名单 MCP 工具'),
      easyicu_list_project_files: tr('Listed project files', '已列出项目文件'),
      easyicu_read_project_file: tr('Read project file', '已读取项目文件'),
      easyicu_write_project_file: tr('Wrote project file', '已写入项目文件'),
      easyicu_edit_project_file: tr('Edited project file', '已编辑项目文件'),
      easyicu_check_project_file: tr('Checked project file', '已检查项目文件'),
      easyicu_preview_project_file: tr('Prepared web preview', '已准备网页预览'),
    };
    const label = labels[String(name || '')] || tr(`Used ${toolLabel(name)}`, `已使用 ${toolLabel(name)}`);
    const file = resourceName(resource);
    return file ? `${label} · ${file}` : label;
  }

  function activeActivity() {
    return state.messages.slice().reverse().find(row => row.role === 'activity' && !row.childJobId && row.status === 'running');
  }
  function ensureActivity(at) {
    let row = activeActivity();
    if (!row) {
      const startedAt = timeMs(at);
      row = { id: 'activity-' + startedAt, role: 'activity', status: 'running', startedAt, steps: [] };
      state.messages.push(row);
    }
    return row;
  }
  function upsertActivityStep(activity, step) {
    if (!activity) return;
    const found = activity.steps.find(item => item.id === step.id);
    if (found) Object.assign(found, step);
    else activity.steps.push(step);
  }
  function finishActivity(status, at, terminalKind) {
    const activity = activeActivity();
    if (!activity) return;
    const endedAt = timeMs(at);
    activity.steps.forEach(step => {
      if (step.status === 'running') step.status = status === 'complete' ? 'complete' : 'error';
    });
    if (terminalKind) {
      upsertActivityStep(activity, {
        id: 'terminal', kind: terminalKind,
        status: status === 'complete' ? 'complete' : 'error', at: endedAt,
      });
    }
    activity.status = status;
    activity.endedAt = endedAt;
  }

  function activityStepLabel(step) {
    const done = step.status === 'complete';
    const failed = step.status === 'error';
    if (step.kind === 'submitted') return tr('Message submitted to Pi AgentSession', '消息已提交给 Pi AgentSession');
    if (step.kind === 'agent') return tr('Pi agent loop started', 'Pi Agent 循环已启动');
    if (step.kind === 'turn') return done
      ? tr(`Model turn ${step.turn + 1} finished`, `模型回合 ${step.turn + 1} 已结束`)
      : tr(`Model turn ${step.turn + 1} is running`, `模型回合 ${step.turn + 1} 进行中`);
    if (step.kind === 'assistant') return done
      ? tr(`Response phase ${step.phase} finished`, `回复阶段 ${step.phase} 已结束`)
      : tr(`Generating response phase ${step.phase}`, `正在生成回复阶段 ${step.phase}`);
    if (step.kind === 'tool') return failed
      ? tr(`${toolLabel(step.toolName, step.resource)} returned an error`, `${toolLabel(step.toolName, step.resource)} 返回错误`)
      : done
        ? completedToolLabel(step.toolName, step.resource)
        : tr(`Calling ${toolLabel(step.toolName, step.resource)}`, `正在调用 ${toolLabel(step.toolName, step.resource)}`);
    if (step.kind === 'pipeline') return String(step.label || tr('EasyICU research pipeline updated', 'EasyICU 科研流程已更新'));
    if (step.kind === 'retry') return tr(`Retrying (${step.attempt}/${step.maxAttempts})`, `正在重试（${step.attempt}/${step.maxAttempts}）`);
    if (step.kind === 'compaction') return done ? tr('Context compaction finished', '上下文整理已完成') : tr('Compacting context', '正在整理上下文');
    if (step.kind === 'cancelled') return tr('This turn was stopped', '本轮已停止');
    if (step.kind === 'failed') return tr('This turn failed', '本轮失败');
    if (step.kind === 'settled') return tr('This turn completed', '本轮已完成');
    return tr('Agent activity updated', 'Agent 状态已更新');
  }

  function transcriptMessages(session) {
    const rows = Array.isArray(session && session.transcript) ? session.transcript : [];
    const messages = [];
    const tools = new Map();
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
        messages.push({ id: 'history-' + index, role: 'user', text, complete: true });
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
        if (activity) upsertActivityStep(activity, toolStep);
      });
      if (text && row.role !== 'user') {
        messages.push({
          id: 'history-' + index, role: row.role || 'assistant', text, complete: true,
          errorCode: row.error_code || '',
          resources: row.role === 'assistant' ? turnResources.slice(0, 24) : [],
        });
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
      const replayStarted = timeMs((turn && turn.started_at) || (replay[0] && replay[0].at));
      let replayActivity = historyActivities[replayOffset + turnIndex];
      const isNewReplayActivity = !replayActivity;
      if (!replayActivity) replayActivity = { id: 'saved-activity-' + String((turn && turn.job_id) || replayStarted), role: 'activity', steps: [], expanded: false };
      const turnStatus = String((turn && turn.status) || session.last_turn_status || 'done');
      replayActivity.status = turnStatus === 'running' ? 'running'
        : (['failed', 'interrupted'].includes(turnStatus) ? 'error'
          : (turnStatus === 'cancelled' ? 'cancelled' : 'complete'));
      replayActivity.startedAt = replayActivity.startedAt || replayStarted;
      replayActivity.endedAt = timeMs((turn && turn.ended_at) || (replay[replay.length - 1] && replay[replay.length - 1].at));
      replayActivity.allowedActions = Array.isArray(turn && turn.allowed_actions) ? turn.allowed_actions.slice() : [];
      replay.forEach(event => {
        const at = timeMs(event && event.at);
        if (event.type === 'run_start') upsertActivityStep(replayActivity, { id: 'agent', kind: 'agent', status: 'complete', at });
        else if (event.type === 'turn_start') upsertActivityStep(replayActivity, { id: 'turn-' + event.turn_index, kind: 'turn', turn: Number(event.turn_index || 0), status: 'running', at });
        else if (event.type === 'turn_end') {
          const turn = replayActivity.steps.find(item => item.id === 'turn-' + event.turn_index);
          if (turn) turn.status = 'complete';
        } else if (event.type === 'assistant_start') {
          const phase = replayActivity.steps.filter(item => item.kind === 'assistant').length + 1;
          upsertActivityStep(replayActivity, { id: 'assistant-' + phase, kind: 'assistant', phase, status: 'running', at });
        } else if (event.type === 'message_end') {
          const phase = replayActivity.steps.slice().reverse().find(item => item.kind === 'assistant' && item.status === 'running');
          if (phase) phase.status = event.error_code ? 'error' : 'complete';
        } else if (event.type === 'tool_start' || event.type === 'tool_progress') {
          upsertActivityStep(replayActivity, { id: 'tool-' + event.tool_call_id, kind: 'tool', toolName: event.tool_name, status: 'running', at, resource: event.resource || null });
        } else if (event.type === 'tool_end') {
          upsertActivityStep(replayActivity, { id: 'tool-' + event.tool_call_id, kind: 'tool', toolName: event.tool_name, status: event.is_error ? 'error' : 'complete', code: event.code || '', owner: event.owner || '', jobId: event.job_id || '', at, endedAt: at, resource: event.resource || null, resources: Array.isArray(event.resources) ? event.resources : [] });
        } else if (event.type === 'retry') upsertActivityStep(replayActivity, { id: 'retry-' + event.attempt, kind: 'retry', status: 'complete', attempt: event.attempt, maxAttempts: event.max_attempts, at });
        else if (event.type === 'compaction_start' || event.type === 'compaction_end') upsertActivityStep(replayActivity, { id: 'compaction', kind: 'compaction', status: event.type === 'compaction_end' && !event.aborted ? 'complete' : 'running', at });
      });
      replayActivity.steps.forEach(step => { if (step.status === 'running' && replayActivity.status !== 'running') step.status = replayActivity.status === 'complete' ? 'complete' : 'error'; });
      if (isNewReplayActivity && replayActivity.steps.length) messages.push(replayActivity);
    });
    return messages.filter(row => row.text || row.role === 'activity');
  }

  function statusBanner() {
    if (state.loading) {
      return `<div class="gpi-inline"><span class="gpi-dot waiting"></span>${tr('Checking Pi Copilot…', '正在检查 Pi Copilot…')}</div>`;
    }
    if (!runtimeReady()) {
      const blockers = (state.runtime && state.runtime.blockers) || [];
      const reason = blockers.includes('api_key_configured')
        ? tr('Connect and verify your model service before entering Pi Copilot.', '请先连接并验证模型服务，再进入 Pi Copilot。')
        : blockers.includes('provider_connection_unverified')
          ? tr('Verify the saved model service before entering Pi Copilot.', '请先验证已保存的模型服务，再进入 Pi Copilot。')
        : blockers.includes('easyicu_ai_opt_in_disabled')
          ? tr('Confirm external AI use before entering Pi Copilot.', '请先确认允许使用外部 AI，再进入 Pi Copilot。')
          : tr('Pi is not ready on this machine. The local Guided workflow remains available.', '这台电脑上的 Pi 尚未就绪，仍可使用本地研究引导流程。');
      return `<div class="gpi-inline unavailable"><span class="gpi-dot"></span><span>${esc(reason)}</span><button class="gpi-link" type="button" data-gpi-setup>${tr('Set up', '开始配置')}</button></div>`;
    }
    if (state.shell === 'legacy') {
      return `<div class="gpi-inline ready"><span class="gpi-dot"></span><span>${tr('Pi AgentSession is ready with EasyICU-only tools.', 'Pi AgentSession 已就绪，仅开放 EasyICU 工具。')}</span><button class="gpi-link" type="button" data-gpi-open>${tr('Open Pi shell', '打开 Pi 交互壳')}</button></div>`;
    }
    return '';
  }

  function setupPanel() {
    const runtime = state.runtime || {};
    const config = runtime.configuration || {};
    const blockers = runtime.blockers || [];
    const runtimeMissing = blockers.filter(code => [
      'node_available', 'node_version_supported', 'entrypoint_available',
      'dependency_installed', 'lockfile_present', 'base_url_configured',
    ].includes(code));
    const savedCredential = !!config.credential_present;
    const canCancel = runtimeReady();
    const staticPreview = isStaticPreview();
    const preset = providerPreset(config, runtime);
    const transport = config.api_transport || runtime.api_transport || 'openai-completions';
    const discovered = state.availableModels.map(model => `<option value="${esc(model)}"></option>`).join('');
    return `
      <div class="gpi-setup-wrap">
        <form class="gpi-setup" data-gpi-provider-form autocomplete="off">
          <div class="gpi-kicker">PI COPILOT · FIRST-USE SETUP</div>
          <h2>${tr('Connect your model service', '连接你的模型服务')}</h2>
          <p>${tr('Like signing in to Codex or Claude Code, this one-time check must succeed before the conversation opens. The API credential is saved only in EasyICU’s private local credential file and is never returned to this page.', '就像登录 Codex 或 Claude Code 一样，只有这次连接验证成功后才会开放对话。API 凭据只保存在 EasyICU 本机私有凭据文件中，不会回传到页面。')}</p>
          <div class="gpi-setup-grid">
            <label><span>${tr('Service type', '服务类型')}</span><select data-gpi-provider-preset>${option('cliproxyapi', preset, 'CLIProxyAPI / Local proxy')}${option('custom-openai', preset, 'OpenAI-compatible gateway')}${option('openai', preset, 'OpenAI API')}${option('anthropic', preset, 'Anthropic API')}${option('google', preset, 'Google Gemini API')}</select></label>
            <label><span>${tr('Provider ID', '提供方标识')}</span><input name="provider" maxlength="80" value="${esc(config.provider || runtime.provider || 'easyicu-local')}" required></label>
            <label class="wide"><span>${tr('Service address', '服务地址')}</span><input name="base_url" maxlength="2048" value="${esc(config.base_url || 'http://127.0.0.1:8317/v1')}" inputmode="url" spellcheck="false" required></label>
            <label><span>${tr('Compatibility protocol', '兼容协议')}</span><select name="api_transport">${option('openai-completions', transport, 'OpenAI Chat Completions')}${option('openai-responses', transport, 'OpenAI Responses')}${option('anthropic-messages', transport, 'Anthropic Messages')}${option('google-generative-ai', transport, 'Google Generative AI')}</select></label>
            <label><span>${tr('Model', '模型')}</span><input name="model" list="gpi-model-options" maxlength="256" value="${esc(config.model || runtime.model || 'gpt-5.6-luna')}" spellcheck="false" required><datalist id="gpi-model-options">${discovered}</datalist></label>
            <label><span>${tr('API credential', 'API 凭据')}</span><input name="api_key" type="password" maxlength="8192" autocomplete="new-password" placeholder="${savedCredential ? tr('Re-enter to verify or replace', '重新输入以验证或更换') : tr('Paste once; it will not be shown again', '仅粘贴一次，之后不再显示')}" required></label>
          </div>
          <div class="gpi-config-note">${tr('Pi supports many provider brands. Service type selects the provider; compatibility protocol selects its wire API. For CLIProxyAPI on port 8317, use OpenAI Chat Completions.', 'Pi 支持很多模型提供方。“服务类型”表示接入对象，“兼容协议”表示实际通信格式；CLIProxyAPI 的 8317 端口请选择 OpenAI Chat Completions。')}</div>
          ${state.availableModels.length ? `<div class="gpi-config-note ok"><span class="gpi-dot"></span>${tr('Models reported by this service:', '该服务返回的可用模型：')} ${esc(state.availableModels.slice(0, 12).join(', '))}</div>` : ''}
          ${savedCredential ? `<div class="gpi-config-note ok"><span class="gpi-dot"></span>${tr('A private credential is saved, but a newly entered credential is still required to verify or change this connection.', '本机已有私有凭据；为验证或更换连接，仍需重新输入一次凭据。')}</div>` : ''}
          ${runtimeMissing.length ? `<div class="gpi-config-note warn">${tr('The Pi runtime also needs attention before chat can open:', '聊天开放前还需要处理 Pi 运行环境：')} ${esc(runtimeMissing.join(', '))}</div>` : ''}
          ${staticPreview ? `<div class="gpi-config-note warn"><strong>${tr('Static preview only.', '当前只是静态预览。')}</strong>&nbsp;${tr('Start EasyICU, then open http://127.0.0.1:8765/#guided. This file:// page cannot verify any credential.', '请启动 EasyICU，再打开 http://127.0.0.1:8765/#guided；当前 file:// 页面无法验证任何凭据。')}</div>` : ''}
          <label class="gpi-optin"><input name="enable_ai" type="checkbox" required> <span>${tr('I authorize this verification request and external AI use for Pi Copilot. Chat text, PHI-safe summaries, and workspace file contents may be sent to this configured service. Do not place PHI, patient rows, credentials, or private clinical data in the workspace.', '我授权本次连接验证，并允许 Pi Copilot 使用外部 AI；对话文字、经 PHI 安全投影的摘要和工作区文件内容可能发送到所配置的服务。请勿在工作区放置 PHI、患者行级数据、凭据或私密临床数据。')}</span></label>
          ${state.error ? `<div class="gpi-error inline">${esc(state.error)}</div>` : ''}
          <div class="gpi-setup-actions">
            ${canCancel ? `<button class="btn" type="button" data-gpi-cancel-setup>${tr('Back to conversation', '返回对话')}</button>` : `<button class="gpi-link" type="button" data-gpi-legacy>${tr('Use local Guided workflow', '使用本地研究引导流程')}</button>`}
            <button class="btn primary" type="submit" ${state.setupSaving || staticPreview ? 'disabled' : ''}>${state.setupSaving ? tr('Verifying…', '正在验证…') : tr('Verify and enter Copilot', '验证并进入 Copilot')}</button>
          </div>
          <div class="gpi-consent">${tr('Verification calls only the service model-list endpoint. The credential is never written to project files, browser storage, logs, or Pi session history.', '验证仅调用服务的模型列表接口。凭据不会写入项目文件、浏览器存储、日志或 Pi 会话历史。')}</div>
        </form>
      </div>`;
  }

  function activatePanel() {
    const saved = state.sessions.map(row => `
      <button class="gpi-session-row" type="button" data-gpi-session="${esc(row.session_id)}">
        <span><strong>${esc(row.title || 'Pi Copilot')}</strong><small>${esc(row.updated_at || '')}</small></span>
        <span>${row.agent_mode === 'workspace' ? tr('Workspace', '工作区') : tr('Research', '研究')}</span>
      </button>`).join('');
    return `
      <div class="gpi-activate">
        <div class="gpi-kicker">PI AGENTSESSION · EASYICU GATEWAY</div>
        <h2>${tr('Start a conversation in this project', '在当前项目中开始对话')}</h2>
        <div class="gpi-config-note ok"><span class="gpi-dot"></span>${tr('Research project', '研究项目')}: <strong>${esc((state.project && state.project.title) || projectId())}</strong></div>
        <button class="btn gpi-demo-launch" type="button" data-gpi-demo>${iconHtml('play', 16)} ${tr('View the complete research workflow demo', '查看完整科研流程演示')}</button>
        ${state.projectInitialization && state.projectInitialization.required ? `<div class="gpi-config-note warn"><strong>${tr('Study setup confirmation required.', '需要确认研究配置初始化。')}</strong> ${tr('No complete saved setup was found. Activating Pi will create an explicitly acknowledged empty StudyContext and collect the missing fields here in conversation.', '未找到完整的已保存配置。启用 Pi 后会在你明确确认下创建空的 StudyContext，并在当前对话中继续收集缺失字段。')}</div>` : ''}
        <p>${tr('Choose the tool boundary for this conversation. Research mode works with study configuration and evidence; Workspace mode also creates, edits, checks, and previews artifacts inside this project’s isolated folder.', '请选择这段对话的工具边界。研究模式处理研究配置与证据；项目工作区模式还可以在当前项目的隔离目录中创建、编辑、检查并预览产物。')}</p>
        <div class="gpi-mode-picker" role="radiogroup" aria-label="${tr('Agent mode', 'Agent 模式')}">
          <button type="button" role="radio" data-gpi-mode-choice="research" aria-checked="${state.agentMode === 'research'}">
            ${iconHtml('shield', 17)}<span><strong>${tr('Research workflow', '科研流程')}</strong><small>${tr('Question, setup, run, evidence, and results', '问题、配置、运行、证据与结果')}</small></span>
          </button>
          <button type="button" role="radio" data-gpi-mode-choice="workspace" aria-checked="${state.agentMode === 'workspace'}">
            ${iconHtml('folder', 17)}<span><strong>${tr('Artifact workspace', '产物工作区')}</strong><small>${tr('Real file tools and web preview', '真实文件工具与网页预览')}</small></span>
          </button>
        </div>
        <button class="btn primary" type="button" data-gpi-create ${state.creating ? 'disabled' : ''}>
          ${state.creating ? tr('Starting…', '正在启动…') : tr('I agree — activate Pi Copilot', '我同意——启用 Pi Copilot')}
        </button>
        <div class="gpi-consent">${tr('Workspace tools can see only this project’s isolated artifact folder. They cannot browse the EasyICU repository, patient rows, credentials, or arbitrary host files.', '工作区工具只能看到当前项目的隔离产物目录，不能浏览 EasyICU 仓库、患者行级数据、凭据或任意本机文件。')}</div>
        ${saved ? `<div class="gpi-saved"><div class="gpi-section-title">${tr('Pi conversations in this project', '当前项目中的 Pi 对话')}</div>${saved}</div>` : ''}
        <button class="gpi-link" type="button" data-gpi-legacy>${tr('Use the local Guided workflow', '使用本地研究引导流程')}</button>
      </div>`;
  }

  function projectRequiredPanel() {
    return `
      <div class="gpi-activate">
        <div class="gpi-kicker">EASYICU PROJECT · PI CONVERSATIONS</div>
        <h2>${tr('Select a research project first', '请先选择研究项目')}</h2>
        <p>${tr('Use the Research projects list on the left, or create a new project. EasyICU keeps study setup, runs, and evidence in that project; Pi keeps its conversation history.', '请从左侧“研究项目”中选择一个项目，或新建项目。EasyICU 在项目中保存研究配置、运行和证据；Pi 保存自己的对话历史。')}</p>
        <button class="btn primary gpi-demo-launch" type="button" data-gpi-demo>${iconHtml('play', 16)} ${tr('View the complete research workflow demo', '查看完整科研流程演示')}</button>
        <button class="gpi-link" type="button" data-gpi-legacy>${tr('Use the local Guided workflow', '使用本地研究引导流程')}</button>
      </div>`;
  }

  function activityStepPrimary(step) {
    const label = activityStepLabel(step);
    if (!step.resource) return `<strong>${esc(label)}</strong>`;
    return resourceButton(step.resource, label);
  }
  function activityStepResources(step) {
    const primary = resourceKey(step.resource);
    const resources = (Array.isArray(step.resources) ? step.resources : [])
      .filter(resource => resourceKey(resource) && resourceKey(resource) !== primary);
    if (!resources.length) return '';
    return `<div class="gpi-resource-list" aria-label="${tr('Run artifacts', '运行产物')}">${resources.map(resource => resourceButton(resource)).join('')}</div>`;
  }
  function activityStepRow(step) {
    return `<li class="${esc(step.status || 'complete')}">
      <span class="gpi-activity-step-icon" aria-hidden="true">${iconHtml(activityIcon(step), 15)}</span>
      <span class="gpi-activity-step-copy">${activityStepPrimary(step)}${step.text ? `<span>${esc(step.text)}</span>` : ''}${activityStepResources(step)}${step.code ? `<small>${esc([step.code, step.owner].filter(Boolean).join(' · '))}</small>` : ''}</span>
      <span class="gpi-status-pip" aria-hidden="true"></span>
    </li>`;
  }

  function messageHtml(row) {
    if (row.role === 'activity') {
      const visibleSteps = row.steps.filter(step => ['submitted', 'agent', 'turn', 'assistant', 'tool', 'pipeline', 'retry', 'compaction'].includes(step.kind));
      const latest = visibleSteps[visibleSteps.length - 1] || row.steps[row.steps.length - 1];
      const running = row.status === 'running';
      const failed = row.status === 'error' || row.status === 'cancelled';
      if (running) {
        const title = latest && latest.kind !== 'submitted'
          ? activityStepLabel(latest)
          : tr('Pi is preparing the next action', 'Pi 正在准备下一步');
        const liveSteps = visibleSteps.slice(-20);
        return `<div class="gpi-activity-running" role="status">
          <div class="gpi-activity-live">
            <span class="gpi-activity-glyph" aria-hidden="true">${iconHtml(activityIcon(latest), 15)}</span>
            <span class="gpi-activity-title">${esc(title)}</span>
            <span class="gpi-status-pip" aria-hidden="true"></span>
          </div>
          ${liveSteps.length ? `<ol>${liveSteps.map(activityStepRow).join('')}</ol>` : ''}
        </div>`;
      }
      const toolSteps = visibleSteps.filter(step => step.kind === 'tool');
      const pipelineSteps = visibleSteps.filter(step => step.kind === 'pipeline');
      const completedTitle = row.displayTitle || (row.childJobId
        ? tr('EasyICU research task finished', 'EasyICU 科研任务已结束')
        : toolSteps.length === 1
        ? activityStepLabel(toolSteps[0])
        : toolSteps.length === 2
          ? toolSteps.map(step => completedToolLabel(step.toolName, step.resource)).join(tr(' and ', '、'))
          : toolSteps.length > 2
            ? tr(`Used ${toolSteps.length} EasyICU tools`, `已使用 ${toolSteps.length} 个 EasyICU 工具`)
            : tr('Answered without using tools', '仅回答，未执行操作'));
      const title = failed ? tr('This turn needs attention', '本轮需要处理') : completedTitle;
      const steps = visibleSteps.map(activityStepRow).join('');
      return `<details class="gpi-activity ${failed ? 'error' : 'complete'}" ${failed || row.expanded ? 'open' : ''}>
        <summary>
          <span class="gpi-activity-glyph" aria-hidden="true">${iconHtml(failed ? 'alert' : activityIcon(toolSteps[0] || pipelineSteps[0] || latest), 15)}</span>
          <span class="gpi-disclosure" aria-hidden="true">${iconHtml('chevron', 14)}</span>
          <span class="gpi-activity-title">${esc(title)}</span>
          <span class="gpi-activity-meta">${esc(tr(`${visibleSteps.length} steps`, `${visibleSteps.length} 个步骤`))}${row.durationKnown === false ? '' : ` · ${esc(durationText(row.startedAt, row.endedAt))}`}</span>
        </summary>
        <div class="gpi-activity-body">
          ${steps ? `<ol>${steps}</ol>` : ''}
          <p>${tr('Lifecycle facts and EasyICU receipts only — private chain-of-thought is never displayed.', '这里只显示生命周期事实和 EasyICU 回执，不展示模型的私有思维链。')}</p>
        </div>
      </details>`;
    }
    const cls = row.role === 'user' ? 'user' : 'assistant';
    const preferredArtifacts = [
      'result_tables.json', 'figure_gallery.json', 'manuscript_scaffold.pdf',
      'manuscript_draft.json', 'agent_plan.json', 'literature_evidence.json',
      'evidence_ledger.json', 'quality_gate.json',
    ];
    const messageResources = (Array.isArray(row.resources) ? row.resources : [])
      .filter(resource => resourceKey(resource))
      .filter((resource, index, rows) => rows.findIndex(item => resourceKey(item) === resourceKey(resource)) === index)
      .sort((left, right) => {
        const leftRank = preferredArtifacts.indexOf(String(left.artifact || ''));
        const rightRank = preferredArtifacts.indexOf(String(right.artifact || ''));
        return (leftRank < 0 ? preferredArtifacts.length : leftRank) - (rightRank < 0 ? preferredArtifacts.length : rightRank);
      })
      .slice(0, 8);
    return `<article class="gpi-message ${cls}">
      <div class="gpi-message-body">
        ${row.text ? `<div class="gpi-text${row.errorCode ? ' gpi-model-error' : ''}">${row.role === 'assistant' ? assistantTextHtml(row.text) : esc(row.text)}</div>` : `<div class="gpi-streaming"><i></i><i></i><i></i></div>`}
        ${messageResources.length ? `<div class="gpi-message-resources" aria-label="${tr('Referenced run artifacts', '本轮引用的运行产物')}"><span>${tr('Open evidence and artifacts', '打开证据和产物')}</span><div class="gpi-resource-list">${messageResources.map(resource => resourceButton(resource)).join('')}</div></div>` : ''}
      </div>
    </article>`;
  }

  function workflowHtml(workflowOverride) {
    const workflow = workflowOverride || state.workflow || {};
    const stages = Array.isArray(workflow.stages) ? workflow.stages : [];
    if (!stages.length) return '';
    const names = {
      question: tr('Question', '问题'), idea: tr('Ideas + literature', '选题与文献'),
      setup: tr('Study design', '研究设计'), extraction: tr('Extract + review', '提取与审阅'),
      plan: tr('Plan + evidence', '计划与证据'), analysis: tr('Analyze + figures', '分析与图表'),
      interpretation: tr('Interpret', '结果解读'), manuscript: tr('Paper', '论文'),
    };
    return `<nav class="gpi-workflow" aria-label="${tr('EasyICU research workflow', 'EasyICU 科研流程')}">
      <div class="gpi-workflow-meta"><strong>${tr('Research workflow', '科研流程')}</strong><span>${esc(workflow.completed_required_stages || 0)}/${esc(workflow.required_stage_count || 7)}</span></div>
      <ol>${stages.map(stage => `<li class="${esc(stage.status || 'blocked')}" title="${esc(stage.reason_code || '')}" aria-current="${stage.id === workflow.current_stage ? 'step' : 'false'}"><i></i><span>${esc(names[stage.id] || stage.label || stage.id)}</span></li>`).join('')}</ol>
    </nav>`;
  }

  function workflowConfirmation() {
    const workflow = state.workflow || {};
    const code = String(workflow.next_action_code || '');
    if (code === 'extraction_ready') return {
      code, grants: ['extract'],
      message: tr('I confirm the current study setup. Start data extraction and quality review.', '我确认当前研究配置，请开始数据提取和质量审阅。'),
      title: tr('Study setup is complete. Start data extraction and quality review?', '研究配置已完成，开始数据提取和质量审阅吗？'),
      note: tr('This creates a governed export with denominator, missingness, provenance, and extraction receipts.', '这会生成带分母、缺失率、来源和提取回执的受治理数据包。'),
      approve: tr('Confirm extraction', '确认提取'),
    };
    if (code === 'plan_ready') return {
      code, grants: ['run', 'provider_run'],
      message: tr('Use the current data package to prepare an evidence-bound analysis plan.', '请基于当前数据包生成证据绑定的分析计划。'),
      title: tr('The data package is ready. Prepare the evidence-bound analysis plan?', '数据包已就绪，是否生成证据绑定的分析计划？'),
      note: tr('The Agent may run a local preflight first and will pause again before executing the approved plan.', 'Agent 可以先运行本地预检，并会在执行计划前再次暂停等待批准。'),
      approve: tr('Prepare plan', '生成计划'),
    };
    if (code === 'provider_plan_ready') return {
      code, grants: ['provider_run'],
      message: tr(
        'I approve the completed local preflight. Start the full Research Agent only to generate the evidence-bound plan, then pause for my plan review.',
        '我确认本地预检结果。请启动完整 Research Agent 生成证据绑定的分析计划，并停在计划人工审阅门。',
      ),
      title: tr('Local preflight passed. Generate the real Research Agent plan?', '本地预检已完成，生成真实 Research Agent 计划吗？'),
      note: tr('This authorizes one provider planning run. Execution must pause again for your plan approval.', '这会授权一次模型规划；真正执行前必须再次停下等待你的计划批准。'),
      approve: tr('Generate Agent plan', '生成 Agent 计划'),
    };
    if (code === 'plan_configuration_superseded' || code === 'plan_review_not_resumable') return {
      code, grants: ['provider_run'],
      message: tr(
        'I confirm that the old plan must not be reused. Start a fresh Research Agent planning run from the current study configuration, and pause again for my review before analysis.',
        '我确认旧计划不能复用。请按当前研究配置启动一次全新的 Research Agent 规划，并在分析前再次停下让我审核。',
      ),
      title: tr('The study changed. Generate a fresh analysis plan?', '研究配置已更新，是否重新生成分析计划？'),
      note: tr('The old run stays as history. The new run receives a new id and current configuration digest.', '旧 run 仅保留为历史；新 run 使用新的标识和当前配置摘要。'),
      approve: tr('Generate fresh plan', '重新生成计划'),
    };
    if (code === 'failed_pipeline_requires_fresh_plan') return {
      code, grants: ['provider_run'],
      message: tr(
        'Keep the failed run as history. Start a fresh Research Agent planning run from the current study configuration, and pause for my review before analysis.',
        '保留上次失败运行为历史。请按当前研究配置启动一次新的 Research Agent 规划，并在分析前停下让我审核。',
      ),
      title: tr('The previous analysis failed closed. Generate a fresh plan?', '上次分析未通过验证，是否重新生成干净计划？'),
      note: tr('The previous evidence remains immutable. The new run receives a new id and must pass plan review again.', '旧证据保持不变；新 run 使用新的标识，并且仍须经过计划人工审核。'),
      approve: tr('Generate fresh plan', '重新生成计划'),
    };
    if (code === 'operator_plan_approval_required') return {
      code, grants: ['provider_run'],
      message: tr(
        'I approve this exact evidence-bound plan without changing the study configuration. Decline optional study-authority additions for this run, preserve every open scientific finding as a limitation, and resume the current plan.',
        '我批准当前这份证据绑定的计划，不修改研究配置。本轮不新增可选的科学设定；请把所有未闭合的科学问题保留为局限，并继续执行当前计划。',
      ),
      title: tr('The evidence-bound plan is ready. Continue to analysis?', '证据绑定的计划已准备好，是否继续分析？'),
      note: tr('Approving resumes this plan only. Scientific and human-review gates remain in force.', '批准只会继续当前计划；科学闸门和人工审阅门禁仍然有效。'),
      approve: tr('Approve and continue', '批准并继续'),
      reviewResources: [
        { kind: 'research_artifact', run_id: String((state.session && state.session.binding && state.session.binding.run_id) || ''), artifact: 'agent_plan.json', label: tr('Open analysis plan', '打开分析计划'), media_type: 'application/json' },
        { kind: 'research_artifact', run_id: String((state.session && state.session.binding && state.session.binding.run_id) || ''), artifact: 'literature_evidence.json', label: tr('Open literature bindings', '打开文献绑定'), media_type: 'application/json' },
      ],
    };
    if (code === 'plan_scientific_changes_required') return {
      code, grants: ['provider_run'], nonApprovable: true,
      message: tr(
        'The current plan is scientifically non-approvable. Keep it as evidence, apply the review findings in a new study version, and generate a fresh plan before analysis.',
        '当前计划未达到科学可批准标准。请保留本次审阅证据，在新的研究版本中处理这些问题，再重新生成计划。',
      ),
      title: tr('Scientific plan review requires changes', '科学计划审阅要求修改'),
      note: tr(
        'The score and findings are deterministic and digest-bound. They cannot be waived by approving this pause; changes that alter the estimand or study authority still need your explicit confirmation.',
        '评分与问题清单均已确定性计算并绑定摘要，不能通过“批准本次暂停”来绕过；涉及估计目标或研究权威的修改仍需你明确确认。',
      ),
      approve: tr('Review required changes', '查看并处理问题'),
      reviewResources: [
        { kind: 'research_artifact', run_id: String((state.session && state.session.binding && state.session.binding.run_id) || ''), artifact: 'scientific_plan_review.json', label: tr('Open scientific review', '打开科学审阅'), media_type: 'application/json' },
        { kind: 'research_artifact', run_id: String((state.session && state.session.binding && state.session.binding.run_id) || ''), artifact: 'agent_plan.json', label: tr('Open proposed plan', '打开候选计划'), media_type: 'application/json' },
        { kind: 'research_artifact', run_id: String((state.session && state.session.binding && state.session.binding.run_id) || ''), artifact: 'literature_evidence.json', label: tr('Open literature evidence', '打开文献证据'), media_type: 'application/json' },
      ],
    };
    return null;
  }

  function workflowConfirmationHtml() {
    const confirmation = workflowConfirmation();
    if (state.busy || sessionIsStale() || !confirmation) return '';
    const reviewResources = (confirmation.reviewResources || [])
      .filter(resource => resource.run_id)
      .map(resource => resourceButton(resource, resource.label))
      .join('');
    const review = state.workflow && state.workflow.plan_review_summary;
    const scorecard = confirmation.code === 'plan_scientific_changes_required'
      && review && review.dimension_scores && typeof review.dimension_scores === 'object'
      ? `<div class="gpi-confirmation-scorecard"><strong>${esc(tr(`Scientific review ${review.score || 0}/100`, `科学审阅 ${review.score || 0}/100`))}</strong><span>${Object.entries(review.dimension_scores).map(([key, value]) => `${esc(key)} ${esc(value)}`).join(' · ')}</span></div>`
      : '';
    const authorizationQuestions = confirmation.code === 'plan_scientific_changes_required'
      && review && Array.isArray(review.authorization_questions)
      ? review.authorization_questions.slice(0, 4).map(item => `<li>${esc(item.question || item.code || '')}</li>`).join('')
      : '';
    const remediationLanes = confirmation.code === 'plan_scientific_changes_required'
      && review && review.remediation_buckets && typeof review.remediation_buckets === 'object'
      ? [
          ['agent_plan_revision', tr('Agent can repair in a fresh plan', 'Agent 可在全新计划中自动修复')],
          ['study_authority_change', tr('Needs your scientific decision', '需要你的科学设定决定')],
          ['external_evidence', tr('Needs more external evidence', '需要继续补充外部证据')],
          ['independent_review', tr('Needs independent novelty review', '需要独立创新性审阅')],
        ].map(([key, label]) => {
          const codes = Array.isArray(review.remediation_buckets[key]) ? review.remediation_buckets[key] : [];
          return codes.length ? `<li><strong>${esc(label)}</strong><span>${esc(codes.join(' · '))}</span></li>` : '';
        }).filter(Boolean).join('')
      : '';
    return `<section class="gpi-confirmation" aria-label="${tr('Workflow confirmation required', '需要确认科研流程')}">
      <span class="gpi-confirmation-icon" aria-hidden="true">${iconHtml('shield', 17)}</span>
      <div><strong>${esc(confirmation.title)}</strong><small>${esc(confirmation.note)}</small>${scorecard}${remediationLanes ? `<ul class="gpi-confirmation-questions">${remediationLanes}</ul>` : ''}${authorizationQuestions ? `<ul class="gpi-confirmation-questions">${authorizationQuestions}</ul>` : ''}${reviewResources ? `<div class="gpi-confirmation-resources">${reviewResources}</div>` : ''}</div>
      <div class="gpi-confirmation-actions">
        <button class="btn sm" type="button" data-gpi-confirm-edit>${confirmation.code === 'plan_scientific_changes_required' ? tr('Answer next scientific question', '回答下一个科学问题') : tr('Request changes', '提出修改')}</button>
        ${confirmation.nonApprovable ? '' : `<button class="btn primary sm" type="button" data-gpi-confirm-action>${esc(confirmation.approve)}</button>`}
      </div>
    </section>`;
  }

  function accessModeHtml() {
    const modes = [
      ['ask', tr('Ask before every tool action', '每次工具操作前都询问')],
      ['assist', tr('Auto-approve low-risk setup and inspection; ask before extraction and full analysis', '自动批准低风险配置与检查；提取和完整分析前仍询问')],
      ['full', tr('Allow all available tools; explicit scientific confirmation gates still apply', '允许所有可用工具；明确的科学确认门禁仍然有效')],
    ];
    return `<details class="gpi-access-menu">
      <summary>${iconHtml(state.accessMode === 'full' ? 'unlock' : 'shield', 15)}<span>${esc(accessModeLabel(state.accessMode))}</span><span class="gpi-access-chevron" aria-hidden="true">${iconHtml('chevron', 13)}</span></summary>
      <div class="gpi-access-popover" role="group" aria-label="${tr('Agent access level', 'Agent 访问级别')}">
        ${modes.map(([mode, description]) => `<button type="button" data-gpi-access-mode="${mode}" aria-pressed="${state.accessMode === mode}"><span><strong>${esc(accessModeLabel(mode))}</strong><small>${esc(description)}</small></span>${state.accessMode === mode ? iconHtml('check', 15) : ''}</button>`).join('')}
        <p>${tr('Access levels never reveal credentials, patient rows, or arbitrary host files.', '任何访问级别都不会开放凭据、患者行级数据或任意本机文件。')}</p>
      </div>
    </details>`;
  }

  function sessionPanel() {
    const session = state.session || {};
    const model = session.model || {};
    const messages = state.messages.map(messageHtml).join('');
    const stale = sessionIsStale();
    const workspace = agentMode() === 'workspace';
    return `
      <div class="gpi-panel">
        <header class="gpi-head">
          <div><div class="gpi-kicker">PI AGENTSESSION · ${esc((state.project && state.project.title) || projectId())}</div><div class="gpi-title">${esc(session.title || 'Pi Copilot')} <span class="gpi-live" role="status" aria-live="polite">${state.busy ? tr('working', '工作中') : tr('ready', '就绪')}</span></div></div>
          <div class="gpi-head-meta">
            <div class="gpi-mode-switch" role="group" aria-label="${tr('Agent mode', 'Agent 模式')}">
              <button type="button" data-gpi-mode-switch="research" aria-pressed="${!workspace}">${tr('Research', '研究')}</button>
              <button type="button" data-gpi-mode-switch="workspace" aria-pressed="${workspace}">${tr('Workspace', '工作区')}</button>
            </div>
            <span>${esc(model.id || (state.runtime && state.runtime.model) || 'model')}</span>
            <button class="btn sm gpi-demo-launch" type="button" data-gpi-demo>${iconHtml('play', 14)} ${tr('Full demo', '完整演示')}</button>
            <button class="gpi-link" type="button" data-gpi-presentation-pin aria-pressed="${session.pinned_for_presentation ? 'true' : 'false'}">${session.pinned_for_presentation ? tr('Saved for presentation', '已保留演示') : tr('Save for presentation', '保留演示')}</button>
            <button class="gpi-link" type="button" data-gpi-config>${tr('Model service', '模型服务')}</button>
            <button class="gpi-link" type="button" data-gpi-new>${tr('New', '新会话')}</button>
            <button class="gpi-link" type="button" data-gpi-study-setup>${tr('Study setup', '研究配置')}</button>
          </div>
        </header>
        ${workflowHtml()}
        ${stale ? `<div class="gpi-stale"><strong>${tr('Authority changed', '权威状态已变化')}</strong><span>${tr('The EasyICU study binding, revision, or active run changed. Rebind before continuing.', 'EasyICU 研究绑定、版本或活动运行已变化，请先重新绑定。')}</span><button class="btn sm" type="button" data-gpi-rebind>${tr('Rebind current state', '重新绑定当前状态')}</button></div>` : ''}
        <div class="gpi-log" data-gpi-log>
          ${messages || (workspace
            ? `<div class="gpi-empty"><strong>${tr('Build something in this project', '在当前项目中创建产物')}</strong><span>${tr('Pi can read, write, edit, check, and preview files in this project’s isolated workspace, while retaining EasyICU research tools.', 'Pi 可以在当前项目的隔离工作区中读取、写入、编辑、检查并预览文件，同时保留 EasyICU 研究工具。')}</span></div>`
            : `<div class="gpi-empty"><strong>${tr('Ask about the current study', '询问当前研究')}</strong><span>${tr('Pi can inspect context, plans, runs, validation, artefacts, evidence, and blockers through bounded EasyICU tools.', 'Pi 可通过受限的 EasyICU 工具检查上下文、计划、运行、验证、产物、证据和阻断原因。')}</span></div>`)}
          ${workflowConfirmationHtml()}
        </div>
        ${state.error ? `<div class="gpi-error">${esc(state.error)}</div>` : ''}
        <div class="gpi-compose">
          <textarea data-gpi-input rows="3" maxlength="12000" placeholder="${workspace ? tr('Ask Pi to create or edit a project artifact — do not paste patient rows or identifiers.', '让 Pi 创建或编辑当前项目产物——请勿粘贴患者行级数据或标识符。') : tr('Ask Pi about this EasyICU study — do not paste patient rows or identifiers.', '向 Pi 询问这个 EasyICU 研究——请勿粘贴患者行级数据或标识符。')}" ${state.busy || stale ? 'disabled' : ''}>${esc(state.draft)}</textarea>
          <div class="gpi-actions">
            ${accessModeHtml()}
            ${state.busy ? `<button class="btn danger" type="button" data-gpi-stop>${tr('Stop', '停止')}</button>` : `<button class="btn primary" type="button" data-gpi-send ${stale ? 'disabled' : ''}>${tr('Send', '发送')}</button>`}
          </div>
        </div>
    </div>`;
  }

  function demoPanel() {
    const demo = window.EU_GUIDED_PI_DEMO;
    if (!demo || typeof demo.messages !== 'function') {
      return `<div class="gpi-activate"><h2>${tr('Demo unavailable', '演示暂不可用')}</h2><button class="btn" type="button" data-gpi-demo-exit>${tr('Back', '返回')}</button></div>`;
    }
    const messages = demo.messages().map(messageHtml).join('');
    const workflow = demo.workflow();
    return `<div class="gpi-panel gpi-demo-panel">
      <header class="gpi-head">
        <div><div class="gpi-kicker">PI COPILOT · COMPLETE RESEARCH WORKFLOW</div><div class="gpi-title">${tr('Complete research workflow demo', '完整科研流程演示')} <span class="gpi-live">${tr('read-only', '只读')}</span></div></div>
        <div class="gpi-head-meta"><span>${tr('Experimental SOFA-2 sensitivity · MIMIC-IV', '实验性 SOFA-2 敏感性 · MIMIC-IV')}</span><button class="gpi-link" type="button" data-gpi-demo-exit>${tr('Back to my project', '返回我的项目')}</button></div>
      </header>
      ${workflowHtml(workflow)}
      <div class="gpi-demo-note" role="note">${iconHtml('shield', 16)}<span><strong>${tr('Interactive product demo.', '可交互产品演示。')}</strong> ${tr('The transcript is read-only. Numeric artifacts come from a real historical E1 engineering canary using the experimental sep3_sofa2_max sensitivity indicator—not standard Sepsis-3—and are not formal paper evidence. Current live projects map generic Sepsis-3 to SOFA-1 and use SOFA-2 only when explicitly requested.', '对话为只读演示。数值产物来自真实历史 E1 工程试跑，使用的是实验性 sep3_sofa2_max 敏感性指标，而不是标准 Sepsis-3；这些内容不是正式论文证据。当前真实项目会把通用 Sepsis-3 映射到 SOFA-1，只有用户明确要求时才使用 SOFA-2。')}</span></div>
      <div class="gpi-log" data-gpi-log>${messages}</div>
      <footer class="gpi-demo-footer"><span>${tr('Open any underlined article or artifact in the conversation to inspect it on the right.', '点击对话中带下划线的文章或产物，可在右侧直接审阅。')}</span><button class="btn primary" type="button" data-gpi-demo-exit>${tr('Start my own research', '开始我自己的研究')}</button></footer>
    </div>`;
  }

  function openDemo() {
    const demo = window.EU_GUIDED_PI_DEMO;
    if (!demo || typeof demo.messages !== 'function') return;
    if (window.EU_GUIDED_PI_PREVIEW && window.EU_GUIDED_PI_PREVIEW.close) window.EU_GUIDED_PI_PREVIEW.close();
    state.demoMode = true;
    state.demoScrollTopPending = true;
    state.error = '';
    setShell('pi');
  }
  function closeDemo() {
    state.demoMode = false;
    state.demoScrollTopPending = false;
    if (window.EU_GUIDED_PI_PREVIEW && window.EU_GUIDED_PI_PREVIEW.close) window.EU_GUIDED_PI_PREVIEW.close();
    render();
  }

  function syncProjectWorkflowAside() {
    const demo = state.demoMode && window.EU_GUIDED_PI_DEMO;
    const workflow = demo && typeof demo.workflow === 'function' ? demo.workflow() : state.workflow;
    if (state.shell !== 'pi' || (!state.demoMode && !projectId())) return;
    const aside = document.getElementById('gdStudyAside');
    const body = document.getElementById('gdAsideBody');
    const head = aside && aside.querySelector('.gd-aside-head');
    if (!aside || !body || !head) return;
    if (!workflow) {
      const receipt = state.project && state.project.binding_receipt;
      const revision = receipt && Number.isInteger(receipt.study_context_revision)
        ? ` · r${receipt.study_context_revision}` : '';
      head.innerHTML = `<div class="eyebrow">${tr('One EasyICU workflow', '统一 EasyICU 科研流程')}</div><div class="at">${tr('Project authority', '项目权威状态')}</div><div class="asub">${tr('Loading the bound StudyContext workflow.', '正在读取已绑定的 StudyContext 流程。')}</div>`;
      body.innerHTML = `<div class="gd-pipeline-summary" data-gpi-project-workflow-loading role="status" aria-live="polite">
        <div class="gd-pipeline-summary-head"><div><div class="eyebrow">${tr('Bound project', '已绑定项目')}</div><strong>${esc((state.project && state.project.title) || projectId())}${esc(revision)}</strong><div class="gd-pipeline-value">${tr('Loading authoritative configuration…', '正在读取权威配置…')}</div></div></div>
      </div>`;
      return;
    }
    const stages = Array.isArray(workflow.stages) ? workflow.stages : [];
    const names = {
      question: tr('Scientific question', '科学问题'), idea: tr('Idea mining', '想法发掘'),
      setup: tr('Study setup', '研究配置'), extraction: tr('Feature extraction', '特征提取'),
      plan: tr('Analysis plan', '分析计划'), analysis: tr('Analysis and validation', '分析与验证'),
      interpretation: tr('Result interpretation', '结果解读'), manuscript: tr('Manuscript', '稿件'),
    };
    const reasons = {
      question_bound: tr('Question is bound to this project', '科学问题已绑定到当前项目'),
      idea_handoff_accepted: tr('Selected idea is digest-bound', '所选想法已用摘要绑定'),
      prior_art_authority_not_established: tr('Prior-art authority and novelty are not established', '先前研究权限与新颖性未成立'),
      idea_feasibility_refresh_required: tr('Recheck feasibility against the current data source', '需要按当前数据源重新核验可行性'),
      study_setup_complete: tr('Required study setup is complete', '必需研究配置已完成'),
      active_export_ready: tr('A matching EasyICU export is ready', '同一项目的 EasyICU 数据包已就绪'),
      plan_ready: tr('Ready to create the analysis plan', '可以生成分析计划'),
      agent_plan_ready: tr('The digest-bound analysis plan is ready', '摘要绑定分析计划已就绪'),
      operator_plan_approval_required: tr('Review and approve the digest-bound plan before analysis', '请在分析前审核并批准摘要绑定的计划'),
      plan_scientific_changes_required: tr('The scientific plan review requires a new study/plan version before analysis', '科学计划审阅要求先形成新的研究/计划版本，当前不能继续分析'),
      plan_configuration_superseded: tr('The study configuration changed; the old plan is superseded and cannot be approved', '研究配置已变化；旧计划已失效，不能再批准'),
      plan_review_not_resumable: tr('The old plan no longer has a live resume authority and must be regenerated', '旧计划的可恢复执行权限已失效，必须重新生成'),
      operator_plan_approved: tr('Digest-bound plan approved by the user', '摘要绑定计划已由用户批准'),
      analysis_ready: tr('Ready for analysis after plan approval', '计划确认后可以执行分析'),
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
    };
    const reasonText = stage => reasons[stage && stage.reason_code]
      || tr('Waiting for the preceding governed stage', '等待前一受治理阶段完成');
    const done = Number(workflow.completed_required_stages || 0);
    const total = Math.max(1, Number(workflow.required_stage_count || 7));
    const pct = Math.max(0, Math.min(100, Math.round(done / total * 100)));
    const current = stages.find(stage => stage.id === workflow.current_stage)
      || stages.find(stage => stage.status !== 'complete') || stages[stages.length - 1];
    head.innerHTML = `<div class="eyebrow">${tr('One EasyICU workflow', '统一 EasyICU 科研流程')}</div><div class="at">${state.demoMode ? tr('Product demo', '产品演示') : tr('Project authority', '项目权威状态')}</div><div class="asub">${state.demoMode ? tr('Read-only walkthrough backed by aggregate engineering-canary artifacts.', '由工程试跑聚合产物支撑的只读流程演示。') : tr('Conversation, extraction, analysis, and evidence share this projection.', '对话、提取、分析与证据共用这一份状态。')}</div>`;
    body.innerHTML = `<div class="gd-pipeline-summary" data-gpi-project-workflow-aside>
      <div class="gd-pipeline-summary-head"><div><div class="eyebrow">${tr('Current stage', '当前阶段')}</div><strong>${esc(names[current && current.id] || (current && current.label) || tr('Ready', '就绪'))}</strong><div class="gd-pipeline-value">${esc(reasonText(current))}</div></div></div>
      <div class="gd-pipeline-bar" aria-label="${tr('EasyICU project progress', 'EasyICU 项目进度')}"><span style="width:${pct}%;"></span></div>
      <div class="gd-pipeline-meta"><span><strong>${done}/${total}</strong> ${tr('required stages complete', '个必需阶段完成')}</span><span>${tr('One project', '同一项目')}</span></div>
    </div>
    <div class="gd-pipeline-list open" data-gpi-project-workflow-list>${stages.map(stage => {
      const status = stage.status === 'complete' ? 'done' : stage.status === 'ready' || stage.status === 'running' || stage.status === 'review_required' ? 'active' : 'locked';
      const marker = status === 'done' ? iconHtml('check', 11) : status === 'locked' ? iconHtml('lock', 10) : iconHtml('dot', 10);
      return `<div class="study-item ${status}" title="${esc(stage.reason_code || '')}"><span class="si-dot">${marker}</span><div class="si-txt"><div class="si-t">${esc(names[stage.id] || stage.label || stage.id)}</div><div class="si-v">${esc(reasonText(stage))}</div></div></div>`;
    }).join('')}</div>`;
  }

  function render() {
    if (!state.host) return;
    state.host.hidden = false;
    state.host.innerHTML = state.shell === 'legacy'
      ? statusBanner()
      : (state.demoMode ? demoPanel() : ((state.showSetup || !runtimeReady()) ? setupPanel() : (!projectId() ? projectRequiredPanel() : (state.session ? sessionPanel() : activatePanel()))));
    syncProjectWorkflowAside();
    requestAnimationFrame(() => {
      const log = state.host && state.host.querySelector('[data-gpi-log]');
      if (log) {
        log.scrollTop = state.demoScrollTopPending ? 0 : log.scrollHeight;
        state.demoScrollTopPending = false;
      }
    });
  }

  async function loadStatus() {
    state.loading = true; state.error = ''; render();
    if (isStaticPreview()) {
      state.runtime = { status: 'unavailable', blockers: ['static_preview_no_backend'] };
      state.showSetup = true; state.loading = false; render(); return;
    }
    try {
      const payload = await api().loadPiCopilotStatus();
      state.runtime = payload && payload.runtime;
      if (projectId()) {
        await prepareProject();
      }
      if (!runtimeReady()) {
        state.showSetup = true;
      }
    } catch (error) {
      state.runtime = { status: 'unavailable', blockers: ['status_request_failed'] };
      state.error = errorText(error);
    } finally {
      state.loading = false; render();
    }
  }

  async function configureProvider(form) {
    if (state.setupSaving || !form) return;
    const data = new FormData(form);
    const apiKey = String(data.get('api_key') || '').trim();
    const keyInput = form.querySelector('[name="api_key"]');
    if (keyInput) keyInput.value = '';
    state.setupSaving = true; state.error = '';
    const submit = form.querySelector('[type="submit"]');
    if (submit) { submit.disabled = true; submit.textContent = tr('Verifying…', '正在验证…'); }
    try {
      const payload = await api().savePiCopilotProviderConfig({
        provider: String(data.get('provider') || '').trim(),
        api_key: apiKey,
        base_url: String(data.get('base_url') || '').trim(),
        model: String(data.get('model') || '').trim(),
        api_transport: String(data.get('api_transport') || 'openai-completions'),
        enable_ai: data.get('enable_ai') === 'on',
      });
      state.runtime = payload && payload.runtime;
      if (!runtimeReady()) {
        state.error = tr('The model connection was saved, but the Pi runtime is not ready yet.', '模型连接已保存，但 Pi 运行环境尚未就绪。');
        return;
      }
      state.showSetup = false;
      if (projectId()) {
        await prepareProject();
        await createSession();
      }
    } catch (error) {
      state.availableModels = Array.isArray(error && error.details && error.details.available_models)
        ? error.details.available_models.map(String) : [];
      state.error = errorText(error);
    } finally {
      state.setupSaving = false; render();
    }
  }

  async function createSession() {
    if (state.creating || !projectId()) return;
    state.creating = true; state.error = ''; state.pendingAuthorityRebind = false; render();
    try {
      const payload = await api().createPiCopilotSession({
        project_id: projectId(),
        title: `${(state.project && state.project.title) || tr('Research project', '研究项目')} · ${state.agentMode === 'workspace' ? tr('Workspace', '工作区') : tr('Research', '研究')}`,
        agent_mode: state.agentMode,
        language: window.EU_LANG === 'zh' ? 'zh' : 'en',
        thinking_level: 'off', external_llm_opt_in: true,
      });
      state.session = payload.session; state.messages = transcriptMessages(state.session);
      state.agentMode = state.session.agent_mode || state.agentMode;
      hydrateProjectedJob(state.workflow && state.workflow.active_job);
      state.projectInitialization = null;
      rememberSession(state.session.session_id);
      state.sessions = [state.session].concat(state.sessions.filter(row => row.session_id !== state.session.session_id));
    } catch (error) { state.error = errorText(error); }
    finally { state.creating = false; render(); }
  }

  async function openSession(sessionId) {
    const expectedProjectId = projectId();
    if (!expectedProjectId) return;
    closeChildSource();
    state.error = '';
    state.pendingAuthorityRebind = false;
    try {
      const payload = await api().loadPiCopilotSession(sessionId, expectedProjectId);
      if (expectedProjectId !== projectId()) return;
      const replayOwner = window.EU_GUIDED_PI_REPLAY;
      state.session = replayOwner && typeof replayOwner.hydrate === 'function'
        ? await replayOwner.hydrate(api(), payload.session, expectedProjectId)
        : payload.session;
      if (expectedProjectId !== projectId()) return;
      state.messages = transcriptMessages(state.session);
      state.agentMode = state.session.agent_mode || 'research';
      (Array.isArray(state.session.archived_child_jobs) ? state.session.archived_child_jobs : []).forEach(hydrateProjectedJob);
      const activeMessageJob = String(state.session.active_message_job_id || '').trim();
      if (activeMessageJob) {
        state.busy = true;
        state.jobId = activeMessageJob;
        watchJob(activeMessageJob);
      }
      reconcileSettledSession();
      hydrateProjectedJob(state.workflow && state.workflow.active_job);
      rememberSession(sessionId); setShell('pi');
      await loadWorkflow();
    } catch (error) { rememberSession(''); state.error = errorText(error); render(); }
  }

  function assistantRow() {
    let row = state.messages[state.messages.length - 1];
    if (!row || row.role !== 'assistant' || row.complete) {
      row = {
        id: 'live-' + Date.now(), role: 'assistant', text: '', complete: false,
        resources: state.currentTurnResources.slice(0, 24),
      };
      state.messages.push(row);
    }
    return row;
  }
  function addAssistantResources(resources) {
    const existing = state.currentTurnResources;
    (Array.isArray(resources) ? resources : []).forEach(resource => {
      const key = resourceKey(resource);
      if (key && !existing.some(item => resourceKey(item) === key)) existing.push(resource);
    });
    state.currentTurnResources = existing.slice(0, 24);
    const row = state.messages.slice().reverse().find(item => item.role === 'assistant' && !item.complete);
    if (row) row.resources = state.currentTurnResources.slice();
  }
  function completeLatestAssistant(stopReason) {
    const row = state.messages.slice().reverse().find(item => item.role === 'assistant' && !item.complete);
    if (row) { row.complete = true; row.stopReason = stopReason || ''; }
  }
  function modelErrorText(code) {
    const value = String(code || '');
    if (value === 'pi_shell_token_budget_exhausted' || value === 'pi_shell_session_provider_call_budget_exhausted') {
      return tr(
        'This conversation reached its bounded safety budget. Start a new conversation in the same research project; the StudyContext, literature, data source, runs, and evidence remain bound to the project.',
        '本会话已达到安全预算。请在同一研究项目中新建后续对话；StudyContext、文献、数据源、运行和证据仍保留在项目中。'
      );
    }
    if (value === 'pi_model_context_limit') return tr('The model context limit was reached. Start a new conversation or shorten the request.', '模型上下文已达到上限，请新建会话或缩短请求。');
    if (value === 'pi_model_rate_limited') return tr('The model service is temporarily rate-limited. No EasyICU action was executed; retry shortly.', '模型服务暂时限流。本轮没有执行 EasyICU 操作，请稍后重试。');
    if (value === 'pi_model_provider_unavailable') return tr('The model service connection was interrupted. No EasyICU action was executed; retry after connectivity recovers.', '模型服务连接中断。本轮没有执行 EasyICU 操作，连接恢复后可直接重试。');
    return tr('The model service could not complete this turn. No EasyICU action should be assumed.', '模型服务未能完成本轮，不能据此认为任何 EasyICU 操作已经执行。');
  }
  async function switchMode(mode) {
    const next = mode === 'research' ? 'research' : 'workspace';
    if (state.busy || next === agentMode()) return;
    closeSource();
    closeChildSource();
    state.agentMode = next;
    state.session = null;
    state.messages = [];
    state.error = '';
    state.pendingAuthorityRebind = false;
    rememberSession('');
    if (window.EU_GUIDED_PI_PREVIEW && window.EU_GUIDED_PI_PREVIEW.close) {
      window.EU_GUIDED_PI_PREVIEW.close();
    }
    await createSession();
  }
  function handlePiEvent(event) {
    if (!event || typeof event !== 'object') return;
    const at = timeMs(event.at);
    const activity = ensureActivity(event.at);
    if (event.type === 'run_start') {
      state.currentTurnResources = [];
      upsertActivityStep(activity, { id: 'agent', kind: 'agent', status: 'complete', at });
    } else if (event.type === 'turn_start') {
      upsertActivityStep(activity, { id: 'turn-' + event.turn_index, kind: 'turn', turn: Number(event.turn_index || 0), status: 'running', at });
    } else if (event.type === 'assistant_start') {
      const phase = activity.steps.filter(item => item.kind === 'assistant').length + 1;
      upsertActivityStep(activity, { id: 'assistant-' + phase, kind: 'assistant', phase, status: 'running', at });
    } else if (event.type === 'text_delta') {
      assistantRow().text += String(event.delta || '');
    } else if (event.type === 'message_end') {
      let row = state.messages.slice().reverse().find(item => item.role === 'assistant' && !item.complete);
      if (event.error_code) {
        row = row || assistantRow();
        row.errorCode = String(event.error_code);
        if (!row.text) row.text = modelErrorText(row.errorCode);
      }
      completeLatestAssistant(event.stop_reason);
      const step = activity.steps.slice().reverse().find(item => item.kind === 'assistant' && item.status === 'running');
      if (step) step.status = event.error_code ? 'error' : 'complete';
    } else if (event.type === 'tool_start') {
      const assistant = activity.steps.slice().reverse().find(item => item.kind === 'assistant' && item.status === 'running');
      if (assistant) assistant.status = 'complete';
      upsertActivityStep(activity, {
        id: 'tool-' + event.tool_call_id, kind: 'tool', toolName: event.tool_name,
        status: 'running', at, startedAt: at, resource: event.resource || null,
      });
    } else if (event.type === 'tool_progress') {
      upsertActivityStep(activity, { id: 'tool-' + event.tool_call_id, kind: 'tool', toolName: event.tool_name, status: 'running', at });
    }
    else if (event.type === 'tool_end') {
      upsertActivityStep(activity, {
        id: 'tool-' + event.tool_call_id, kind: 'tool', toolName: event.tool_name,
        status: event.is_error ? 'error' : 'complete', code: event.code || '',
        owner: event.owner || '', text: event.summary || '', at, endedAt: at,
        jobId: event.job_id || '',
        resource: event.resource || null,
        resources: Array.isArray(event.resources) ? event.resources : [],
      });
      addAssistantResources([event.resource].concat(Array.isArray(event.resources) ? event.resources : []));
      if (event.host_rebind_after_turn === true || ['study_context_updated', 'easyicu_extraction_submitted', 'easyicu_run_submitted', 'easyicu_full_run_submitted'].includes(String(event.code || ''))) {
        state.pendingAuthorityRebind = true;
      }
      if (/^(easyicu_(research_workflow_projected|idea_|active_export_reused|extraction_|run_|full_run_|result_|manuscript_))/.test(String(event.code || ''))) {
        loadWorkflow().then(render).catch(() => {});
      }
      if (event.job_id && ['easyicu_extraction_submitted', 'easyicu_run_submitted', 'easyicu_full_run_submitted'].includes(String(event.code || ''))) {
        watchChildJob(String(event.job_id), String(event.code || ''));
      }
    } else if (event.type === 'turn_end') {
      const turn = activity.steps.find(item => item.id === 'turn-' + event.turn_index);
      if (turn) turn.status = 'complete';
    } else if (event.type === 'retry') {
      upsertActivityStep(activity, { id: 'retry-' + event.attempt, kind: 'retry', status: 'running', attempt: event.attempt, maxAttempts: event.max_attempts, at });
    } else if (event.type === 'compaction_start') {
      upsertActivityStep(activity, { id: 'compaction', kind: 'compaction', status: 'running', at });
    } else if (event.type === 'compaction_end') {
      upsertActivityStep(activity, { id: 'compaction', kind: 'compaction', status: event.aborted ? 'error' : 'complete', at });
    } else if (event.type === 'agent_cycle_end' && event.will_retry) {
      const retry = activity.steps.slice().reverse().find(item => item.kind === 'retry' && item.status === 'running');
      if (retry) retry.status = 'complete';
    } else if (event.type === 'run_end') {
      finishActivity('complete', event.at, 'settled');
    }
    render();
  }
  function closeSource() { if (state.source) { state.source.close(); state.source = null; } }
  function reconcileSettledSession() {
    if (state.session && state.session.active_message_job_id) return;
    if (!state.session || state.session.streaming !== false || !state.busy) return;
    closeSource();
    state.busy = false;
    state.jobId = '';
    finishActivity('complete', null, 'settled');
  }
  function closeChildSource() {
    if (state.childSource) { state.childSource.close(); state.childSource = null; }
    state.childJobId = '';
  }
  function childActivity(jobId, code) {
    let activity = state.messages.find(row => row.role === 'activity' && row.childJobId === jobId);
    if (activity) return activity;
    const startedAt = Date.now();
    activity = {
      id: 'easyicu-job-' + jobId, role: 'activity', status: 'running',
      startedAt, childJobId: jobId, steps: [],
    };
    const label = code === 'easyicu_extraction_submitted'
      ? tr('EasyICU data extraction submitted', 'EasyICU 数据提取任务已提交')
      : code === 'easyicu_full_run_submitted'
        ? tr('Research Agent planning submitted', 'Research Agent 规划任务已提交')
        : tr('EasyICU preflight submitted', 'EasyICU 预检任务已提交');
    upsertActivityStep(activity, {
      id: 'pipeline-submitted', kind: 'pipeline', step: 'submitted', label,
      status: 'complete', at: startedAt, code: jobId, owner: 'EasyICU',
    });
    state.messages.push(activity);
    return activity;
  }
  function completeRunningPipelineSteps(activity) {
    activity.steps.forEach(step => {
      if (step.kind === 'pipeline' && step.status === 'running') step.status = 'complete';
    });
  }
  function childEventLabel(event) {
    if (event.label) return String(event.label);
    if (event.type === 'start') return tr('EasyICU research pipeline started', 'EasyICU 科研流程已启动');
    if (event.type === 'cancel_requested') return tr('Cancellation requested', '已请求取消任务');
    return tr('EasyICU research pipeline updated', 'EasyICU 科研流程已更新');
  }
  function handleChildJobEvent(jobId, code, event) {
    if (!event || typeof event !== 'object' || state.childJobId !== jobId) return;
    const activity = childActivity(jobId, code);
    if (event.type === 'end') {
      completeRunningPipelineSteps(activity);
      const gate = event.result && event.result.gate;
      const pending = Boolean(event.result && event.result.human_review_pending);
      const failed = event.status === 'failed' || event.status === 'cancelled';
      const label = event.status === 'cancelled'
        ? tr('EasyICU research task cancelled', 'EasyICU 科研任务已取消')
        : event.status === 'failed'
          ? tr('EasyICU research task failed', 'EasyICU 科研任务失败')
          : pending
            ? tr('Analysis paused for plan review', '分析已暂停，等待计划审核')
            : gate && gate.reportable === false
              ? tr('Analysis finished; the scientific gate remains locked', '分析已结束；科学闸门仍保持锁定')
              : tr('EasyICU research task completed', 'EasyICU 科研任务已完成');
      upsertActivityStep(activity, {
        id: 'pipeline-terminal', kind: 'pipeline', step: 'terminal', label,
        status: failed ? 'error' : 'complete', at: Date.now(),
        code: String((gate && gate.status) || event.status || ''),
        owner: String((event.result && event.result.run_id) || ''),
      });
      activity.status = failed ? (event.status === 'cancelled' ? 'cancelled' : 'error') : 'complete';
      activity.endedAt = Date.now();
      closeChildSource();
      archiveChildJob(jobId)
        .catch(() => null)
        .then(() => refreshSession(true))
        .then(async () => {
          if (state.session && sessionIsStale()) await rebind();
          await loadWorkflow();
          render();
        })
        .catch(() => render());
      return;
    }
    if (!['start', 'progress', 'gate', 'artifact', 'cancel_requested'].includes(String(event.type || ''))) return;
    completeRunningPipelineSteps(activity);
    const step = String(event.step || event.type || 'pipeline').slice(0, 80);
    upsertActivityStep(activity, {
      id: 'pipeline-' + String(event.seq == null ? step : event.seq),
      kind: 'pipeline', step, label: childEventLabel(event), status: 'running',
      at: Date.now(), code: step,
      owner: String(event.run_id || '').slice(0, 160),
    });
    render();
  }
  function watchChildJob(jobId, code) {
    if (state.childJobId === jobId && state.childSource) return;
    closeChildSource();
    state.childJobId = jobId;
    childActivity(jobId, code);
    state.childSource = new EventSource('/api/jobs/' + encodeURIComponent(jobId) + '/events');
    let ended = false;
    state.childSource.onmessage = event => {
      let row = null; try { row = JSON.parse(event.data); } catch (e) { return; }
      if (row.type === 'end') ended = true;
      handleChildJobEvent(jobId, code, row);
    };
    state.childSource.onerror = () => {
      if (ended || state.childJobId !== jobId) return;
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
  function hydrateProjectedJob(job) {
    if (!job || !job.present || !job.job_id || !state.session) return;
    const jobId = String(job.job_id);
    const activity = childActivity(jobId, String(job.kind || ''));
    const replayOwner = window.EU_GUIDED_PI_REPLAY;
    const presentation = replayOwner && typeof replayOwner.childJobPresentation === 'function'
      ? replayOwner.childJobPresentation(job, tr) : {};
    activity.expanded = Boolean(presentation.expanded);
    activity.durationKnown = Boolean(presentation.durationKnown);
    if (presentation.startedAt != null) activity.startedAt = presentation.startedAt;
    if (presentation.endedAt != null) activity.endedAt = presentation.endedAt;
    if (presentation.title) activity.displayTitle = presentation.title;
    const progress = Array.isArray(job.progress) ? job.progress : [];
    progress.forEach((event, index) => {
      const step = String(event.step || event.type || 'pipeline').slice(0, 80);
      const count = event.current != null && event.total != null ? `${event.current}/${event.total}` : '';
      const reason = String(event.reason_code || '');
      upsertActivityStep(activity, {
        id: 'projected-' + String(event.seq == null ? index : event.seq),
        kind: 'pipeline', step,
        label: String(event.label || step.replace(/[_-]+/g, ' ')),
        status: ['failed', 'cancelled', 'error'].includes(String(event.status || '')) ? 'error' : 'complete',
        at: Date.now(), code: [count, reason].filter(Boolean).join(' · '), owner: String(job.kind || 'EasyICU'),
      });
    });
    const settled = ['done', 'failed', 'cancelled'].includes(String(job.status || ''));
    if (settled) {
      completeRunningPipelineSteps(activity);
      upsertActivityStep(activity, {
        id: 'projected-terminal', kind: 'pipeline', step: 'terminal',
        label: presentation.terminalLabel || (job.status === 'done'
          ? tr('EasyICU research task completed', 'EasyICU 科研任务已完成')
          : job.status === 'cancelled'
            ? tr('EasyICU research task cancelled', 'EasyICU 科研任务已取消')
            : tr('EasyICU research task failed', 'EasyICU 科研任务失败')),
        status: job.status === 'done' ? 'complete' : 'error', at: Date.now(),
        code: String(job.gate_status || job.error_code || job.status || ''), owner: String(job.kind || 'EasyICU'),
        resources: Array.isArray(job.artifact_refs) ? job.artifact_refs : [],
      });
      activity.status = job.status === 'done' ? 'complete' : (job.status === 'cancelled' ? 'cancelled' : 'error');
      if (presentation.endedAt == null) activity.endedAt = Date.now();
    }
  }
  async function refreshSession(preserveTimeline) {
    if (!state.session || !projectId()) return;
    try {
      const payload = await api().loadPiCopilotSession(state.session.session_id, projectId());
      const replayOwner = window.EU_GUIDED_PI_REPLAY;
      state.session = !preserveTimeline && replayOwner && typeof replayOwner.hydrate === 'function'
        ? await replayOwner.hydrate(api(), payload.session, projectId())
        : payload.session;
      if (!preserveTimeline) state.messages = transcriptMessages(state.session);
      (Array.isArray(state.session.archived_child_jobs) ? state.session.archived_child_jobs : []).forEach(hydrateProjectedJob);
      reconcileSettledSession();
    } catch (e) {}
  }

  async function loadProjectSessions() {
    const expectedProjectId = projectId();
    if (!runtimeReady() || !expectedProjectId) return;
    const listed = await api().loadPiCopilotSessions(30, expectedProjectId);
    if (expectedProjectId !== projectId()) return;
    state.sessions = (listed && listed.sessions) || [];
    const remembered = rememberedSession();
    if (remembered && state.sessions.some(row => row.session_id === remembered)) {
      await openSession(remembered);
    } else if (state.sessions.length) {
      // Recover the latest project conversation even on another browser/device
      // where no local navigation hint exists.
      await openSession(state.sessions[0].session_id);
    }
  }

  async function loadWorkflow() {
    const expectedProjectId = projectId();
    if (!expectedProjectId || !api().loadPiCopilotProjectWorkflow) return;
    try {
      const payload = await api().loadPiCopilotProjectWorkflow(expectedProjectId);
      if (expectedProjectId !== projectId()) return;
      state.workflow = payload && payload.workflow ? payload.workflow : null;
      if (state.workflow && payload && payload.active_job) state.workflow.active_job = payload.active_job;
      hydrateProjectedJob(payload && payload.active_job);
      const activeJob = payload && payload.active_job;
      if (activeJob && activeJob.present && activeJob.status === 'running' && activeJob.job_id) {
        const kind = String(activeJob.kind || '');
        const code = /extract/i.test(kind)
          ? 'easyicu_extraction_submitted'
          : (/research|agent/i.test(kind) ? 'easyicu_full_run_submitted' : 'easyicu_run_submitted');
        watchChildJob(String(activeJob.job_id), code);
      }
    } catch (error) {
      if (expectedProjectId === projectId()) state.workflow = null;
    }
  }

  async function prepareProject() {
    const expectedProjectId = projectId();
    if (!expectedProjectId) return;
    const bindingReceipt = state.project && state.project.binding_receipt;
    try {
      const initialized = await api().initializePiCopilotProject({
        project_id: expectedProjectId,
        title: (state.project && state.project.title) || expectedProjectId,
        confirm_initialization: false,
        binding_receipt: state.project.binding_receipt || undefined,
      });
      if (expectedProjectId !== projectId()) return;
      state.projectInitialization = initialized || { status: 'ready' };
      if (bindingReceipt && initialized && initialized.binding_receipt) {
        state.project = { ...state.project, binding_receipt: null };
      }
      await loadWorkflow();
      if (runtimeReady()) await loadProjectSessions();
    } catch (error) {
      if (expectedProjectId !== projectId()) return;
      if (error && error.code === 'pi_project_initialization_required') {
        state.projectInitialization = {
          required: true,
          missingRequired: (error.details && error.details.missing_required) || [],
        };
        state.error = '';
        render();
        return;
      }
      throw error;
    }
  }

  function bindProject(project) {
    const next = project && String(project.id || '').trim()
      ? {
          id: String(project.id).trim(),
          title: String(project.title || project.id).trim(),
          binding_receipt: project.binding_receipt || null,
        }
      : null;
    const sameProject = projectId() === String((next && next.id) || '');
    const currentReceipt = state.project && state.project.binding_receipt;
    if (sameProject && JSON.stringify(currentReceipt || null) === JSON.stringify((next && next.binding_receipt) || null)) return;
    closeSource();
    closeChildSource();
    state.demoMode = false;
    state.demoScrollTopPending = false;
    state.project = next;
    state.session = null;
    state.sessions = [];
    state.messages = [];
    state.busy = false;
    state.jobId = '';
    state.error = '';
    state.projectInitialization = null;
    state.workflow = null;
    state.pendingAuthorityRebind = false;
    if (window.EU_GUIDED_PI_PREVIEW && window.EU_GUIDED_PI_PREVIEW.clearProject) {
      window.EU_GUIDED_PI_PREVIEW.clearProject();
    }
    render();
    if (next) {
      // Project selection happens after the initial Pi status request in the
      // common path. prepareProject() updates the authoritative workflow and
      // saved-session lists asynchronously, so render again when it settles;
      // otherwise the legacy 0/8 aside and empty-session panel remain visible
      // even though the server returned the bound StudyContext revision.
      prepareProject()
        .then(() => { if (projectId() === next.id) render(); })
        .catch(error => { if (projectId() === next.id) { state.error = errorText(error); render(); } });
    }
  }

  function isActive() { return state.shell === 'pi'; }
  function watchJob(jobId) {
    closeSource();
    state.source = new EventSource('/api/jobs/' + encodeURIComponent(jobId) + '/events');
    state.source.onmessage = async event => {
      let row = null; try { row = JSON.parse(event.data); } catch (e) { return; }
      if (row.type === 'pi_event') handlePiEvent(row.event);
      if (row.type === 'end') {
        closeSource(); state.busy = false;
        if (row.status === 'failed') {
          finishActivity('error', null, 'failed');
          state.error = /^pi_model_/.test(String(row.error || ''))
            ? modelErrorText(row.error)
            : String(row.error || tr('Pi message failed.', 'Pi 消息失败。'));
        } else if (row.status === 'cancelled') {
          finishActivity('cancelled', null, 'cancelled');
          state.error = tr('Pi message stopped.', 'Pi 消息已停止。');
        } else {
          finishActivity('complete', null, 'settled');
        }
        await refreshSession(true);
        if (state.pendingAuthorityRebind && state.session && sessionIsStale()) {
          await rebind();
        }
        await loadWorkflow();
        state.pendingAuthorityRebind = false;
        render();
      }
    };
    state.source.onerror = () => { if (!state.busy) closeSource(); };
  }
  async function sendText(text, grantsOverride) {
    if (!state.session || state.busy || sessionIsStale()) return;
    text = String(text || '').trim();
    if (!text) return;
    const grants = Array.isArray(grantsOverride) ? grantsOverride : turnGrants();
    const submittedAt = Date.now();
    state.currentTurnResources = [];
    state.messages.push({ id: 'user-' + submittedAt, role: 'user', text, complete: true });
    const activity = ensureActivity(new Date(submittedAt).toISOString());
    upsertActivityStep(activity, { id: 'submitted', kind: 'submitted', status: 'complete', at: submittedAt });
    state.draft = ''; state.busy = true; state.error = ''; render();
    try {
      const payload = await api().sendPiCopilotMessage(state.session.session_id, {
        project_id: projectId(), message: text, allowed_actions: grants,
      });
      state.jobId = payload.job_id; watchJob(payload.job_id);
    } catch (error) {
      state.busy = false; finishActivity('error', null, 'failed');
      state.error = errorText(error); render();
    }
  }
  async function sendMessage() {
    if (!state.session || state.busy || sessionIsStale()) return;
    const input = state.host.querySelector('[data-gpi-input]');
    const text = String((input && input.value) || state.draft || '').trim();
    await sendText(text);
  }
  async function confirmWorkflowAction() {
    const confirmation = workflowConfirmation();
    if (!confirmation) return;
    await sendText(confirmation.message, confirmation.grants);
  }
  function editWorkflow() {
    const workflow = state.workflow || {};
    const review = workflow.plan_review_summary || {};
    const questions = Array.isArray(review.authorization_questions)
      ? review.authorization_questions.filter(item => item && (item.question || item.code))
      : [];
    const nextQuestion = questions.length ? String(questions[0].question || questions[0].code) : '';
    state.draft = nextQuestion || tr(
      'Please ask me the next unresolved scientific decision and save my answer in the typed study configuration.',
      '请一次只问我一个尚未解决的科学设定问题，并把我的回答保存到结构化研究配置。',
    );
    render();
    requestAnimationFrame(() => {
      const input = state.host && state.host.querySelector('[data-gpi-input]');
      if (input) { input.focus(); input.setSelectionRange(input.value.length, input.value.length); }
    });
  }
  function studySetupReviewPrompt(workflow) {
    const receipt = workflow && workflow.study_setup_receipt;
    const missing = workflow && workflow.missing_setup_fields;
    const missingText = Array.isArray(missing) && missing.length
      ? missing.join(', ')
      : 'none';
    const receiptText = JSON.stringify(receipt || {
      study_context_id: '',
      revision: 0,
      configured_fields: [],
      configuration: {},
    });
    return tr(
      `Review this existing project's study configuration in this conversation. Treat the following path-free Study Setup Receipt as the authoritative starting state: ${receiptText}. Preserve study_context_id and revision; do not create a new project or reset configured fields. Current missing fields: ${missingText}. Summarize the configured values first, then ask which field I want to edit.`,
      `请在当前对话中审阅这个已有项目的研究配置。以下不含本地路径的 Study Setup Receipt 是权威起始状态：${receiptText}。保留 study_context_id 和 revision；不要新建项目，也不要重置已配置字段。当前缺失字段：${missingText}。请先概括已有配置，再询问我要修改哪个字段。`,
    );
  }
  async function openStudySetupInConversation() {
    if (!state.session || state.busy || sessionIsStale()) return;
    setShell('pi');
    state.showSetup = false;
    state.error = '';
    await loadWorkflow();
    const prompt = studySetupReviewPrompt(state.workflow);
    await sendText(prompt, ['configure']);
  }
  async function stopMessage() {
    if (!state.session || !state.busy) return;
    try {
      await api().abortPiCopilotSession(state.session.session_id, {
        project_id: projectId(), message_job_id: state.jobId || null,
      });
    }
    catch (error) { state.error = errorText(error); render(); }
  }
  async function rebind() {
    if (!state.session) return;
    try {
      const payload = await api().rebindPiCopilotSession(
        state.session.session_id,
        { project_id: projectId() },
      );
      state.session = payload.session; state.error = ''; render();
    } catch (error) { state.error = errorText(error); render(); }
  }

  async function archiveChildJob(jobId) {
    if (!state.session || !jobId || !api().archivePiCopilotChildJob) return null;
    return api().archivePiCopilotChildJob(
      state.session.session_id,
      jobId,
      { project_id: projectId() },
    );
  }

  async function togglePresentationPin() {
    if (!state.session || !api().pinPiCopilotPresentation) return;
    try {
      const pinned = !Boolean(state.session.pinned_for_presentation);
      const payload = await api().pinPiCopilotPresentation(
        state.session.session_id,
        { project_id: projectId(), pinned },
      );
      state.session.pinned_for_presentation = Boolean(
        payload && payload.session && payload.session.pinned_for_presentation,
      );
      state.error = '';
    } catch (error) {
      state.error = errorText(error);
    }
    render();
  }

  function wire() {
    if (!state.host) return;
    state.host.addEventListener('click', event => {
      const session = event.target.closest('[data-gpi-session]');
      if (session) { openSession(session.dataset.gpiSession); return; }
      if (event.target.closest('[data-gpi-demo-exit]')) { closeDemo(); return; }
      if (event.target.closest('[data-gpi-demo]')) { openDemo(); return; }
      const resource = event.target.closest('[data-gpi-resource-kind]');
      if (resource) {
        if (window.EU_GUIDED_PI_PREVIEW && window.EU_GUIDED_PI_PREVIEW.open) {
          window.EU_GUIDED_PI_PREVIEW.open({
            file: resource.dataset.gpiResourceFile,
            kind: resource.dataset.gpiResourceKind,
            run_id: resource.dataset.gpiResourceRun,
            artifact: resource.dataset.gpiResourceArtifact,
            label: resource.dataset.gpiResourceLabel,
            media_type: resource.dataset.gpiResourceMedia,
            url: resource.dataset.gpiResourceUrl,
            title: resource.dataset.gpiResourceTitle,
            year: resource.dataset.gpiResourceYear,
            venue: resource.dataset.gpiResourceVenue,
            relevance: resource.dataset.gpiResourceRelevance,
            doi: resource.dataset.gpiResourceDoi,
            pmid: resource.dataset.gpiResourcePmid,
            study_context_id: resource.dataset.gpiResourceStudy,
            study_revision: resource.dataset.gpiResourceRevision,
            review_sha256: resource.dataset.gpiResourceDigest,
            sha256: resource.dataset.gpiResourceDigest,
          }, projectId());
        }
        return;
      }
      const modeChoice = event.target.closest('[data-gpi-mode-choice]');
      if (modeChoice) { state.agentMode = modeChoice.dataset.gpiModeChoice === 'research' ? 'research' : 'workspace'; render(); return; }
      const modeSwitch = event.target.closest('[data-gpi-mode-switch]');
      if (modeSwitch) { switchMode(modeSwitch.dataset.gpiModeSwitch); return; }
      const accessMode = event.target.closest('[data-gpi-access-mode]');
      if (accessMode) { state.accessMode = accessMode.dataset.gpiAccessMode || 'assist'; render(); return; }
      if (event.target.closest('[data-gpi-retry]')) { loadStatus(); return; }
      if (event.target.closest('[data-gpi-setup]')) { state.showSetup = true; setShell('pi'); return; }
      if (event.target.closest('[data-gpi-open]')) { setShell('pi'); return; }
      if (event.target.closest('[data-gpi-study-setup]')) { openStudySetupInConversation(); return; }
      if (event.target.closest('[data-gpi-legacy]')) { setShell('legacy'); return; }
      if (event.target.closest('[data-gpi-create]')) { createSession(); return; }
      if (event.target.closest('[data-gpi-confirm-action]')) { confirmWorkflowAction(); return; }
      if (event.target.closest('[data-gpi-confirm-edit]')) { editWorkflow(); return; }
      if (event.target.closest('[data-gpi-send]')) { sendMessage(); return; }
      if (event.target.closest('[data-gpi-stop]')) { stopMessage(); return; }
      if (event.target.closest('[data-gpi-rebind]')) { rebind(); return; }
      if (event.target.closest('[data-gpi-presentation-pin]')) { togglePresentationPin(); return; }
      if (event.target.closest('[data-gpi-config]')) { state.showSetup = true; state.error = ''; render(); return; }
      if (event.target.closest('[data-gpi-cancel-setup]')) { state.showSetup = false; state.error = ''; render(); return; }
      if (event.target.closest('[data-gpi-new]')) { state.session = null; state.messages = []; rememberSession(''); render(); }
    });
    state.host.addEventListener('input', event => {
      if (event.target.matches('[data-gpi-input]')) state.draft = event.target.value;
    });
    state.host.addEventListener('change', event => {
      if (!event.target.matches('[data-gpi-provider-preset]')) return;
      const form = event.target.closest('[data-gpi-provider-form]');
      if (!form) return;
      const presets = {
        cliproxyapi: { provider: 'easyicu-local', base_url: 'http://127.0.0.1:8317/v1', api_transport: 'openai-completions', model: 'gpt-5.6-luna' },
        'custom-openai': { provider: 'custom-openai', base_url: 'https://example.com/v1', api_transport: 'openai-completions', model: '' },
        openai: { provider: 'openai', base_url: 'https://api.openai.com/v1', api_transport: 'openai-responses', model: 'gpt-5.6-luna' },
        anthropic: { provider: 'anthropic', base_url: 'https://api.anthropic.com/v1', api_transport: 'anthropic-messages', model: 'claude-sonnet-4-6' },
        google: { provider: 'google', base_url: 'https://generativelanguage.googleapis.com/v1beta', api_transport: 'google-generative-ai', model: 'gemini-3.5-flash' },
      };
      const selected = presets[event.target.value];
      if (!selected) return;
      Object.keys(selected).forEach(name => {
        const field = form.elements.namedItem(name);
        if (field) field.value = selected[name];
      });
      state.availableModels = [];
      const modelList = form.querySelector('#gpi-model-options');
      if (modelList) modelList.replaceChildren();
    });
    state.host.addEventListener('keydown', event => {
      if (event.target.matches('[data-gpi-input]') && event.key === 'Enter' && (event.metaKey || event.ctrlKey)) {
        event.preventDefault(); sendMessage();
      }
    });
    state.host.addEventListener('submit', event => {
      const form = event.target.closest('[data-gpi-provider-form]');
      if (!form) return;
      event.preventDefault(); configureProvider(form);
    });
  }
  function mount(host) {
    if (!host || state.host === host) return;
    closeSource(); closeChildSource(); state.host = host; state.conv = host.closest('.gd-conv'); state.shell = 'pi';
    if (state.conv) state.conv.classList.add('pi-active');
    wire(); loadStatus();
  }
  function unmount() {
    closeSource(); closeChildSource(); state.host = null; state.conv = null; state.busy = false; state.jobId = '';
  }
  window.EU_GUIDED_PI = { mount, unmount, setShell, bindProject, isActive };
})();
