/* Owner: Guided Copilot conversation, workflow, and session panel rendering. */
(function () {
  function create(deps) {
    const {
      state, modules: MODULES, dataConsent: DATA_CONSENT, starters: STARTERS,
      ideaSource: IDEA_SOURCE, header: HEADER, regeneration: REGENERATION,
      studyWorkspace: STUDY_WORKSPACE, activity: ACTIVITY, runFiles: RUN_FILES,
      resourceOwner: RESOURCE_OWNER, messageActions: MESSAGE_ACTIONS,
      transcript: TRANSCRIPT, runOutcome: RUN_OUTCOME, aside: ASIDE,
      cohortEligibility: COHORT_ELIGIBILITY, tr, esc, iconHtml, projectId,
      publicAssistantText, assistantTextHtml, sessionIsStale, agentMode,
      accessModeLabel, projectTitle, navigationSessionTitle,
      workflowConfirmationHtml, hostJobs: HOST_JOBS, followUps: FOLLOW_UPS,
      effortMenu: EFFORT_MENU,
    } = deps;

    function messageHtml(row, options) {
      if (row.childJobHandoff) return '';
      if (row.role === 'activity') return ACTIVITY.render(row, options && options.trace) + RUN_FILES.render(row);
      if (row.role === 'host_notice') return HOST_JOBS && typeof HOST_JOBS.renderNotice === 'function' ? HOST_JOBS.renderNotice(row) : '';
      if (row.role === 'saved_run') return `<article class="gpi-message assistant gpi-saved-run"><div class="gpi-message-body"><p>${tr('Saved run synchronized from this research project.', '已从本研究同步保存的运行记录。')}</p>${RUN_FILES.render(row)}</div></article>`;
      if (row.role === 'workflow_receipt') {
        const rows = row.total_rows == null ? Number.NaN : Number(row.total_rows);
        const files = Number(row.data_file_count);
        const supports = Number(row.support_file_count);
        const isResult = row.receipt_kind === 'extraction_result';
        return `<article class="gpi-message assistant gpi-workflow-receipt" role="status">
          <div class="gpi-message-body">
            <div class="gpi-workflow-receipt-head">${iconHtml('check', 15)}<div><strong>${isResult ? tr('Extraction result synchronized', '抽取结果已同步') : tr('Extraction setup saved', '抽取配置已保存')}</strong><span>${tr('This is EasyICU state, not a model reply.', '这是 EasyICU 本地状态，不是模型回复。')}</span></div></div>
            <div class="gpi-workflow-receipt-grid">
              <span>${tr('Database', '数据库')}<b>${esc(row.database || row.source_label || '—')}</b></span>
              ${isResult ? `<span>${tr('Output', '产物')}<b>${Number.isFinite(files) ? files : 0} + ${Number.isFinite(supports) ? supports : 0} ${tr('files', '个文件')}</b></span>` : ''}
              ${isResult ? `<span>${tr('Rows', '行数')}<b>${Number.isFinite(rows) ? rows.toLocaleString() : '—'}</b></span>` : ''}
              <span>${tr('Cohort', '队列')}<b>${esc(row.cohort_summary || tr('Current confirmed cohort', '当前确认队列'))}</b></span>
              <span>${tr('Modules', '模块')}<b>${esc((row.modules || []).join(', ') || '—')}</b></span>
              <span>${tr('Format', '格式')}<b>${esc(String(row.export_format || '—').toUpperCase())}</b></span>
              <span>StudyContext<b>${esc(row.study_context_id || '—')} · rev ${Number(row.study_revision || 0)}</b></span>
            </div>
            ${row.output_dir ? `<div class="gpi-workflow-receipt-path"><span>${tr('Local output folder', '本机输出文件夹')}</span><code>${esc(row.output_dir)}</code></div>` : ''}
            <p>${isResult
              ? tr('Copilot now reads this completed local extraction result. The absolute local path is shown only in this host UI and is not inserted as model-authored text.', 'Copilot 现在可以读取这次已完成的本地抽取结果。本机绝对路径只显示在当前宿主界面，不会被伪装成模型生成的文字。')
              : tr('Copilot now reads this database, cohort, feature-module, time-window, and export-format setup from the saved StudyContext. No extraction has been claimed yet.', 'Copilot 现在会从已保存的 StudyContext 读取数据库、队列、特征模块、时间窗和导出格式；此时尚未声称已经完成抽取。')}</p>
          </div>
        </article>`;
      }
      const cls = row.role === 'user' ? 'user' : 'assistant';
      const messageView = STUDY_WORKSPACE.messageView(row, projectId());
      const messageSkillHtml = messageView.requestedSkill
        ? `<div class="gpi-message-skill">✧ ${tr('Requested Skill', '指定技能')} · ${esc(messageView.requestedSkill)}</div>` : '';
      const messageResourcesHtml = RESOURCE_OWNER.renderForMessage(messageView, 8);
      const historicalDataConsentHtml = options && options.historicalDataConsent
        && DATA_CONSENT && typeof DATA_CONSENT.renderPast === 'function'
        ? DATA_CONSENT.renderPast(state.session, { tr, esc, icon: iconHtml })
        : '';
      // The model's follow-up suggestions are lifted out of the reply text;
      // only the latest reply offers them, as clickable questions.
      const lifted = row.role === 'assistant' && FOLLOW_UPS && row.complete !== false
        ? FOLLOW_UPS.split(publicAssistantText(row.text)) : null;
      const publicRow = row.role === 'assistant'
        ? { ...row, text: lifted ? lifted.text : publicAssistantText(row.text) } : row;
      const nextOwner = MODULES.require('nextActions');
      const nextStep = row.role === 'assistant' && row.complete !== false
        && !['auto_generate_plan', 'auto_revise_plan', 'generate_plan']
          .includes(String(row.hostActionCode || ''))
        && nextOwner && typeof nextOwner.project === 'function'
        ? nextOwner.project(publicRow.text) : null;
      const interactive = Boolean(options && options.interactive);
      const visibleText = nextStep ? nextOwner.bodyText(nextStep, { live: interactive }) : row.role === 'user' ? messageView.text : publicRow.text;
      const historicalChoiceAnswered = Boolean(options && options.historicalChoiceAnswered);
      const nextStepHtml = !nextStep
        ? ''
        : !interactive
          ? (historicalChoiceAnswered ? '' : nextOwner.renderPast(nextStep, window.EU_LANG))
          : typeof nextOwner.render === 'function'
        ? nextOwner.render(nextStep, {
          language: window.EU_LANG,
          disabled: state.busy || sessionIsStale(),
          dataSourceAuthorization: DATA_CONSENT && DATA_CONSENT.authorization(state.session),
          workflowActionCode: String((state.workflow && state.workflow.next_action_code) || ''),
          suppressFallback: String((state.workflow && state.workflow.next_action_code) || '') === 'provider_ready_to_generate_plan',
        }) : '';
      const messageActions = MESSAGE_ACTIONS.render(publicRow, {
        editing: state.editingMessageId === row.id,
        allowEdit: Boolean(options && options.allowEdit),
        canEdit: Boolean(options && options.canEdit),
        canRetry: Boolean(options && options.canRetry),
        retryUserEntryId: String(options && options.retryUserEntryId || ''),
      });
      const contentHtml = messageActions.editorHtml
        || (visibleText ? `<div class="gpi-text${row.errorCode ? ' gpi-model-error' : ''}">${row.role === 'assistant' ? assistantTextHtml(visibleText) : esc(visibleText)}</div>` : `<div class="gpi-streaming"><i></i><i></i><i></i></div>`);
      const streaming = row.role === 'assistant' && row.complete === false && !messageActions.editorHtml;
      const followUpsHtml = interactive && lifted && lifted.questions.length
        ? FOLLOW_UPS.render(lifted.questions, { tr, iconHtml, disabled: state.busy || sessionIsStale() }) : '';
      return `<article class="gpi-message ${cls}${messageActions.actionsHtml ? ' has-actions' : ''}${streaming ? ' is-streaming' : ''}" data-gpi-message-id="${esc(row.id || '')}"${streaming ? ' data-gpi-streaming-message' : ''}>
        <div class="gpi-message-body">
          ${messageSkillHtml}${contentHtml}
          ${messageResourcesHtml}
          ${RUN_FILES.render(row)}
          ${historicalDataConsentHtml}
          ${nextStepHtml}
          ${messageActions.actionsHtml}
          ${followUpsHtml}
        </div>
      </article>`;
    }

    // A refused automatic plan configuration shows the plan card and the
    // cohort decision together; otherwise the cohort decision, when the plan
    // needs one, replaces the confirmation card.
    function continuationCardsHtml() {
      const eligibility = COHORT_ELIGIBILITY.render();
      const confirmation = workflowConfirmationHtml();
      const refused = String((state.workflow && state.workflow.next_action_code) || '') === 'agent_plan_configuration_required'
        && Boolean(state.planConfigurationError);
      if (refused && eligibility && confirmation) return confirmation + eligibility;
      return eligibility || confirmation;
    }

    function workflowHtml(workflowOverride) {
      const workflow = workflowOverride || state.workflow || {};
      const stages = Array.isArray(workflow.stages) ? workflow.stages : [];
      if (!stages.length) return '';
      const reviewerDemo = workflow.kind === 'reviewer_validation_demo';
      const names = {
        question: reviewerDemo ? tr('Protocol', '审稿协议') : tr('Question', '问题'),
        idea: reviewerDemo ? tr('Validation scope', '验证范围') : tr('Ideas + literature', '选题与文献'),
        setup: reviewerDemo ? tr('Data contract', '数据合同') : tr('Study design', '研究设计'),
        extraction: reviewerDemo ? tr('Projection', '安全投影') : tr('Extract + review', '提取与审阅'),
        plan: tr('Plan + evidence', '计划与证据'), analysis: tr('Analyze + figures', '分析与图表'),
        interpretation: tr('Interpret', '结果解读'), manuscript: reviewerDemo ? tr('Dossier', '审稿报告') : tr('Paper', '论文'),
      };
      return `<nav class="gpi-workflow" aria-label="${tr('EasyICU research workflow', 'EasyICU 科研流程')}">
        <div class="gpi-workflow-meta"><strong>${reviewerDemo ? tr('Reviewer workflow', '审稿流程') : tr('Research workflow', '科研流程')}</strong><span class="shell-sr-only">${esc(workflow.completed_required_stages || 0)}/${esc(workflow.required_stage_count || 7)} ${tr('required stages complete', '个必需阶段已完成')}</span></div>
        <ol>${stages.map(stage => `<li class="${esc(stage.status || 'blocked')}" title="${esc(names[stage.id] || stage.label || stage.id)}" aria-current="${stage.id === workflow.current_stage ? 'step' : 'false'}"><i></i><span>${esc(names[stage.id] || stage.label || stage.id)}</span></li>`).join('')}</ol>
      </nav>`;
    }

    // What this conversation can reach beyond EasyICU's own tools: the MCP
    // servers and user Skills frozen into it, plus the settings switches the
    // literature connectors depend on. Read from the session record and the
    // page-level settings; no extra request.
    function composerExtensions(session) {
      const settings = window.EU_SETTINGS || {};
      const frozen = (session && session.extension_activation) || {};
      const servers = Array.isArray(frozen.mcp_servers) ? frozen.mcp_servers : [];
      const skills = Array.isArray(frozen.skills) ? frozen.skills : [];
      return {
        mcpEnabled: settings.mcp_tools_enabled === true,
        pubmedEnabled: settings.connector_pubmed_enabled !== false,
        zoteroEnabled: settings.connector_zotero_enabled === true,
        mcpServers: servers.map(row => ({ name: String(row.name || ''), tools: Array.isArray(row.allowed_tools) ? row.allowed_tools.length : 0 })),
        skills: skills.map(row => ({ name: String(row.name || '') })),
      };
    }
    function sessionPanel() {
      const session = state.session || {};
      const model = session.model || {};
      const research = session.research_provider || {};
      const connection = session.model_connection || null;
      const stale = sessionIsStale();
      const workspace = agentMode() === 'workspace';
      const freshTask = !state.messages.length;
      const ideaExplorationTurn = !workspace
        && TRANSCRIPT.latestTurnCompletedIdeaExploration(state.messages);
      const showProjectContinuationCards = !workspace && !freshTask && !ideaExplorationTurn;
      const dataConsentRequired = showProjectContinuationCards
        && DATA_CONSENT && DATA_CONSENT.requiresConfirmation(session);
      const fullTimeline = freshTask ? [] : RUN_FILES.timeline(state.messages.concat(state.workflowReceipts));
      const timeline = state.regenerating && REGENERATION
        ? REGENERATION.visibleRows(fullTimeline, state.regeneration)
        : fullTimeline;
      const activeChild = timeline.slice().reverse().find(row => row.role === 'activity' && row.childJobId && row.status === 'running');
      const interactionLocked = state.busy || Boolean(activeChild) || state.projectLoading;
      // The latest reply keeps its suggestions and choices even when a
      // finished host receipt (a review click, a plan job) was recorded
      // after it; only a reply or a still-running turn supersedes it.
      const latestAssistant = timeline.slice().reverse().find(row => row.role === 'assistant'
        || (row.role === 'activity' && row.status === 'running'));
      const answeredAssistantIds = new Set();
      let pendingAssistantId = '';
      timeline.forEach(row => {
        if (row && row.role === 'assistant') {
          if (pendingAssistantId) answeredAssistantIds.add(pendingAssistantId);
          pendingAssistantId = String(row.id || '');
        } else if (row && row.role === 'user' && pendingAssistantId) {
          answeredAssistantIds.add(pendingAssistantId);
          pendingAssistantId = '';
        }
      });
      let precedingUserText = '';
      let precedingUserEntryId = '';
      let historicalDataConsentProjected = false;
      const projectRow = row => (state.regenerating && REGENERATION
        ? REGENERATION.project(row, state.regeneration) : row);
      // Text-only segments of a traced turn: the opening sentence above the
      // traces and the interim narration between trace rows. Actions, next
      // steps, and run files stay on the turn's final answer.
      const renderSegment = (row, kind) => {
        const displayRow = projectRow(row);
        if (!displayRow || displayRow.childJobHandoff) return '';
        const text = publicAssistantText(String(displayRow.text || ''));
        if (!text) return '';
        if (kind === 'intro') {
          return `<article class="gpi-message assistant gpi-turn-intro" data-gpi-message-id="${esc(displayRow.id || '')}"><div class="gpi-message-body"><div class="gpi-text">${assistantTextHtml(text)}</div></div></article>`;
        }
        return `<div class="gpi-activity-narration-text" data-gpi-message-id="${esc(displayRow.id || '')}">${assistantTextHtml(text)}</div>`;
      };
      const messages = ACTIVITY.renderTimeline(timeline, (row, trace) => {
        const displayRow = projectRow(row);
        const historicalDataConsent = !historicalDataConsentProjected
          && row.role === 'assistant'
          && DATA_CONSENT && typeof DATA_CONSENT.matchesSourceSelection === 'function'
          && DATA_CONSENT.matchesSourceSelection(state.session, precedingUserText);
        const html = messageHtml(displayRow, {
          interactive: row === latestAssistant && !interactionLocked && !stale,
          allowEdit: true,
          canEdit: !interactionLocked && !stale,
          canRetry: row.role === 'assistant' && !interactionLocked && !stale,
          retryText: row.role === 'assistant' ? precedingUserText : '',
          retryUserEntryId: row.role === 'assistant' ? precedingUserEntryId : '',
          historicalDataConsent,
          historicalChoiceAnswered: row.role === 'assistant'
            && answeredAssistantIds.has(String(row.id || '')),
          trace,
        });
        if (historicalDataConsent) historicalDataConsentProjected = true;
        if (row.role === 'user') {
          precedingUserText = String(row.text || '');
          precedingUserEntryId = String(row.entryId || '');
        }
        return html;
      }, renderSegment);
      const outcome = showProjectContinuationCards ? RUN_OUTCOME.render(state.latestRun, state.workflow) : '';
      const emptyResearch = !workspace && !messages && !outcome;
      const resultsView = Boolean(outcome) && !dataConsentRequired;
      const conversation = messages;
      const dataConsentHtml = dataConsentRequired
        ? DATA_CONSENT.render(session, { tr, esc, icon: iconHtml, namedDemo: HOST_JOBS && typeof HOST_JOBS.namedDemo === 'function' ? HOST_JOBS.namedDemo() : null })
        : '';
      const headerOptions = {
        tr, esc, icon: iconHtml,
        projectTitle: projectTitle(),
        sessionTitle: navigationSessionTitle(session),
        busy: interactionLocked,
        workspace,
        pinned: Boolean(session.pinned_for_presentation),
        layout: ASIDE.layoutOptions(),
        connectionLabel: connection
          ? ([connection.provider, connection.model].filter(Boolean).join(' · ') || 'model')
          : ([model.id || (state.runtime && state.runtime.model), research.provider, research.model].filter(Boolean).join(' / ') || 'legacy model binding'),
        connectionTitle: connection
          ? tr('One model connection for conversation and analysis', '对话与分析共用的一套模型连接')
          : tr('Legacy conversation and analysis bindings', '旧会话的对话与分析绑定'),
      };
      const composerCardHtml = `<div class="gpi-compose-card${activeChild ? ' is-running' : ''}">
        ${activeChild ? `<div class="gpi-compose-running" role="status" aria-live="polite" aria-busy="true">
          <span class="gpi-running-spinner" aria-hidden="true"></span>
          <span><strong>${esc(activeChild.cancelRequested ? tr('Stopping the research task', '正在停止科研任务') : (activeChild.runningTitle || tr('EasyICU research task is running', 'EasyICU 科研任务正在运行')))}</strong><small>${activeChild.cancelRequested ? tr('The cancellation request was sent. Waiting for the current safe checkpoint.', '已发送停止请求，正在等待当前安全检查点结束。') : tr('New messages are paused until this task finishes or asks for confirmation.', '任务完成或需要你确认后，才可继续发送消息。')}</small></span>
          <time data-gpi-live-elapsed="${Number(activeChild.startedAt || Date.now())}">${esc(ACTIVITY.durationText ? ACTIVITY.durationText(activeChild.startedAt) : '')}</time>
          <button class="btn danger sm" type="button" data-gpi-cancel-child-job="${esc(activeChild.childJobId)}" ${activeChild.cancelRequested ? 'disabled' : ''}>${activeChild.cancelRequested ? tr('Stopping…', '正在停止…') : tr('Stop generation', '停止生成')}</button>
        </div>` : `${!workspace && IDEA_SOURCE ? IDEA_SOURCE.status({ tr, esc }) : ''}${STUDY_WORKSPACE.renderReference(projectId(), session.session_id)}${STUDY_WORKSPACE.renderSkillReference(projectId(), session)}<textarea data-gpi-input rows="2" maxlength="12000" placeholder="${state.projectLoading ? tr('Research status is syncing. Continue when it finishes…', '正在同步研究状态，完成后可继续提问……') : workspace ? tr('Ask EasyICU Copilot to create or edit a project artifact — do not paste patient rows or identifiers.', '让 EasyICU 研究助手创建或编辑当前项目产物——请勿粘贴患者行级数据或标识符。') : tr('What ICU research question would you like to study?', '你想研究什么 ICU 科学问题？')}" ${interactionLocked || stale ? 'disabled' : ''}>${esc(state.draft)}</textarea>
          <div class="gpi-actions">
            <div class="gpi-action-leading">${IDEA_SOURCE ? IDEA_SOURCE.controls({ tr, esc, icon: iconHtml, disabled: interactionLocked || stale, allowIdeaSources: !workspace, extensions: composerExtensions(session) }) : ''}${STUDY_WORKSPACE.renderMaterials(RUN_OUTCOME.collection(state.latestRun, state.workflow), projectId(), session.session_id, interactionLocked || stale || Boolean(state.workflowError))}${STUDY_WORKSPACE.renderSkillPicker(projectId(), session, interactionLocked || stale)}${STUDY_WORKSPACE.renderAccessMode(state.accessMode, accessModeLabel, iconHtml)}</div>
            <div class="gpi-action-trailing">${EFFORT_MENU ? EFFORT_MENU.render({ iconHtml, level: session.thinking_level, disabled: interactionLocked || stale }) : ''}${HEADER.renderModelControl(headerOptions)}
            ${state.busy ? `<button class="btn danger" type="button" data-gpi-stop>${tr('Stop', '停止')}</button>` : `<button class="btn primary" type="button" data-gpi-send aria-label="${tr('Send', '发送')}" title="${tr('Send', '发送')}" ${interactionLocked || stale ? 'disabled' : ''}>${iconHtml('arrow', 15)}</button>`}</div>
          </div>`}
      </div>`;
      const emptyResearchHtml = STARTERS && typeof STARTERS.render === 'function'
        ? STARTERS.render({ tr, disabled: interactionLocked || stale,
          composer: `<div class="gpi-compose gpi-entry-compose">${composerCardHtml}</div>` })
        : `<div class="gpi-empty"><strong>${tr('Start with the research question', '先描述研究问题')}</strong></div>`;
      return `
        <div class="gpi-panel${emptyResearch ? ' gpi-empty-session' : ''}${resultsView ? ' gpi-results-session' : ''}">
          ${HEADER.render(headerOptions)}
          <div class="gpi-context-strip">
          ${state.workflowError ? `<div class="gpi-stale" role="alert">${esc(state.workflowError)}<button type="button" data-gpi-refresh-status>${tr('Retry', '重试')}</button></div>` : workflowHtml()}
          ${!workspace && DATA_CONSENT && typeof DATA_CONSENT.renderSelectedSource === 'function'
            ? DATA_CONSENT.renderSelectedSource(session, { tr, esc, icon: iconHtml }) : ''}
          </div>
          ${stale ? `<div class="gpi-stale"><strong>${tr('Authority changed', '权威状态已变化')}</strong><span>${tr('The EasyICU study binding, revision, or active run changed. Rebind before continuing.', 'EasyICU 研究绑定、版本或活动运行已变化，请先重新绑定。')}</span><button class="btn sm" type="button" data-gpi-rebind>${tr('Rebind current state', '重新绑定当前状态')}</button></div>` : ''}
          <div class="gpi-log${messages ? '' : ' gpi-log-start'}" data-gpi-log>
            ${RUN_FILES.notice()}
            ${conversation || (resultsView ? '' : workspace
                ? `<div class="gpi-empty"><strong>${tr('Build something in this project', '在当前项目中创建产物')}</strong><span>${tr('EasyICU Copilot can read, write, edit, check, and preview files in this project’s isolated workspace, while retaining EasyICU research tools.', 'EasyICU 研究助手可以在当前项目的隔离工作区中读取、写入、编辑、检查并预览文件，同时保留 EasyICU 研究工具。')}</span></div>`
                : emptyResearchHtml)}
            ${outcome}
            ${showProjectContinuationCards ? dataConsentHtml : ''}
            ${showProjectContinuationCards && !dataConsentRequired ? continuationCardsHtml() : ''}
          </div>
          ${state.error ? `<div class="gpi-error" role="alert"><span>${esc(state.error)}</span><button type="button" class="gpi-error-close" data-gpi-dismiss-error aria-label="${tr('Dismiss', '关闭提示')}">×</button></div>` : ''}
          ${emptyResearch ? '' : `<div class="gpi-compose">${composerCardHtml}</div>`}
      </div>`;
    }

    function demoPanel() {
      const demo = MODULES.optional('demo');
      if (!demo || typeof demo.messages !== 'function') {
        return `<div class="gpi-activate"><h2>${tr('Demo unavailable', '演示暂不可用')}</h2><button class="btn" type="button" data-gpi-demo-exit>${tr('Back', '返回')}</button></div>`;
      }
      const messages = demo.messages().map(row => messageHtml(row, { interactive: false })).join('');
      const workflow = demo.workflow();
      return `<div class="gpi-panel gpi-demo-panel">
        <header class="gpi-head">
          <div><div class="gpi-kicker">${tr('EASYICU COPILOT · REVIEWER DEMONSTRATION', 'EASYICU COPILOT · 审稿人演示')}</div><div class="gpi-title">${tr('Complete governed workflow', '完整受治理科研流程')} <span class="gpi-live">${tr('complete', '已完成')}</span></div></div>
          <div class="gpi-head-meta"><span>${tr('Registered source run · 94,458 ICU stays', '登记 source run · 94,458 ICU stays')}</span><button class="gpi-link" type="button" data-gpi-demo-exit>${tr('Back to my project', '返回我的项目')}</button></div>
        </header>
        ${workflowHtml(workflow)}
        <div class="gpi-demo-note" role="note">${iconHtml('shield', 16)}<span><strong>${tr('Read-only reviewer walkthrough.', '只读审稿人演示。')}</strong> ${tr('The transcript and dossier are a bounded projection derived from one registered source run, not live artifact transport. Aggregate results use the explicitly requested experimental first-24-hour SOFA-2 phenotype and a descriptive-only claim ceiling; they are not a clinical manuscript.', '对话与报告是从同一个登记 source run 派生的有界投影，不是 live artifact transport。聚合结果使用用户明确要求的入 ICU 后 24 小时实验性 SOFA-2 表型，结论上限为仅描述；它们不是临床论文。')}</span></div>
        <div class="gpi-log" data-gpi-log>${messages}</div>
        <footer class="gpi-demo-footer"><span>${tr('The reviewer dossier opens automatically. Select any underlined receipt to inspect its bounded source view.', '审稿报告会自动打开；点击任意带下划线的回执可检查其有界来源视图。')}</span><button class="btn primary" type="button" data-gpi-demo-exit>${tr('Start my own research', '开始我自己的研究')}</button></footer>
      </div>`;
    }

    return Object.freeze({ messageHtml, workflowHtml, sessionPanel, demoPanel });
  }

  window.EasyICU.guidedPi.declare('sessionView', { create });
})();
