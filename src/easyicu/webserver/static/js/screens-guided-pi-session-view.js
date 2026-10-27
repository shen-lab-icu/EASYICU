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
      effortMenu: EFFORT_MENU, workflowConfirmation, runFailureText, help: HELP,
      sessionMatchesUiLanguage, displaySessionTitle, sessionTraceLabel, providerBindingSummary,
      quickAnswers: QUICK_ANSWERS, multiDatabase: MULTI_DATABASE,
    } = deps;
    const modelMenu = MODULES.optional('modelMenu');
    if (modelMenu) modelMenu.bind({
      session: () => state.session, projectId,
      applySession: session => {
        state.session = { ...state.session, ...session };
        state.sessions = state.sessions.map(row => row.session_id === session.session_id ? { ...row, ...session } : row);
      },
      locked: () => Boolean(state.busy || state.projectLoading || state.planStarting
        || (state.session && state.session.streaming)),
      hosted: () => { const owner = MODULES.optional('hosted'); return Boolean(owner && owner.active()); },
    });
    const usageOwner = MODULES.optional('usage');
    if (usageOwner) usageOwner.bind({ session: () => state.session, projectId });
    // A stopped run's block (owner: screens-guided-pi-run-stop.js).
    const stopOwner = MODULES.optional('runStop');
    const RUN_STOP = stopOwner ? stopOwner.create({ tr, runFailureText,
      workflow: () => state.workflow, latestRun: () => state.latestRun,
      canAct: () => Boolean(state.session) && !state.busy && !state.childJobId && !sessionIsStale(),
      canPreview: () => Boolean(MODULES.optional('preview')),
      hasRecovery: () => typeof workflowConfirmation === 'function' && Boolean(workflowConfirmation()),
      resourceButton: RESOURCE_OWNER.button,
    }) : null;

    // Entry views project titles through the shared label owner, retaining the
    // bounded fallback for bundles that have not loaded that owner yet.
    function displayProjectTitle(value, fallback) { return window.EU_PRODUCT_LABELS?.projectTitle?.(value, fallback) ?? String(value ?? fallback ?? '').slice(0, 200); }
    // A saved message keeps its transcript entry id so search can reveal it.
    function entryAttr(row) {
      const id = String((row && row.entryId) || '');
      return id ? ` data-gpi-entry-id="${esc(id)}"` : '';
    }

    function activatePanel() {
      const saved = state.sessions.filter(sessionMatchesUiLanguage).map(row => `
      <button class="gpi-session-row" type="button" data-gpi-session="${esc(row.session_id)}">
        <span><strong>${esc(displaySessionTitle(row.title))}</strong><small>${esc(sessionTraceLabel(row))}</small></span>
        <span>${row.agent_mode === 'workspace' ? tr('Workspace', '工作区') : tr('Research', '研究')}</span>
      </button>`).join('');
      if (state.projectIssue === 'pi_project_study_context_missing') {
        return `
        <div class="gpi-activate gpi-project-recovery">
          <div class="gpi-kicker">${tr('EASYICU COPILOT · PROJECT RECOVERY', 'EASYICU COPILOT · 项目恢复')}</div>
          <h2>${tr('This old project can no longer be opened', '这个旧项目已无法继续打开')}</h2>
          <div class="gpi-config-note ok"><span class="gpi-dot"></span>${tr('Research project', '研究项目')}: <strong>${esc(displayProjectTitle(state.project && state.project.title, projectId()))}</strong></div>
          ${providerBindingSummary()}
          <div class="gpi-recovery-card" role="alert">
            <span class="gpi-recovery-icon">${iconHtml('folder', 20)}</span>
            <div>
              <strong>${tr('The saved research setup is no longer available', '关联的研究配置已经失效')}</strong>
              <p>${tr('The project shortcut still exists, but its authoritative study setup was removed. EasyICU will not silently create or attach a different setup.', '项目快捷记录仍然存在，但它原来绑定的权威研究配置已经被移除。EasyICU 不会静默创建或绑定另一份配置。')}</p>
            </div>
          </div>
          <div class="gpi-recovery-actions">
            <button class="btn primary" type="button" data-project-new>${tr('New project', '新建项目')}</button>
            <button class="btn" type="button" data-refreshdrafts>${tr('Refresh project list', '刷新项目列表')}</button>
          </div>
          <div class="gpi-consent">${tr('You can also choose another existing project from the list on the left. Rebinding this old project remains an explicit recovery operation.', '也可以直接从左侧列表选择其他已有项目。若要恢复当前旧项目，仍需执行明确的重新绑定操作。')}</div>
        </div>`;
      }
      return `
      <div class="gpi-activate">
        <div class="gpi-kicker">${tr('EASYICU COPILOT · RESEARCH WORKSPACE', 'EASYICU COPILOT · 科研工作区')}</div>
        <h2>${tr('Start a conversation in this project', '在当前项目中开始对话')}</h2>
        <div class="gpi-config-note ok"><span class="gpi-dot"></span>${tr('Research project', '研究项目')}: <strong>${esc(displayProjectTitle(state.project && state.project.title, projectId()))}</strong></div>
        ${providerBindingSummary()}
        ${state.error ? `<div class="gpi-error" role="alert">${esc(state.error)}</div>` : ''}
        <button class="btn primary" type="button" data-gpi-create ${state.creating ? 'disabled' : ''}>
          ${state.creating ? tr('Starting…', '正在启动…') : tr('Start research conversation', '开始研究对话')}
        </button>
        <div class="gpi-consent">${iconHtml('shield', 13)}<span>${tr('Study progress is saved automatically. File access can be enabled later and remains limited to this project folder — never patient rows, credentials, or files outside the project.', '研究进度会自动保存；文件操作可稍后开启，且只能访问当前项目目录，不包括患者行级数据、凭据或项目以外的文件。')}</span></div>
        ${saved ? `<div class="gpi-saved"><div class="gpi-section-title">${tr('Copilot conversations in this project', '当前项目中的研究助手对话')}</div>${saved}</div>` : ''}
        <div class="gpi-secondary-actions">
          <button class="gpi-link" type="button" data-gpi-demo>${tr('View workflow demo', '查看流程演示')}</button>
        </div>
      </div>`;
    }

    function projectRequiredPanel() {
      return `
      <div class="gpi-activate">
        <div class="gpi-kicker">EASYICU PROJECT · COPILOT CONVERSATIONS</div>
        <h2>${tr('Select a research project first', '请先选择研究项目')}</h2>
        <p>${tr('Use the Research projects list on the left, or create a new project. EasyICU keeps study setup, runs, evidence, and conversation history in that project.', '请从左侧“研究项目”中选择一个项目，或新建项目。EasyICU 会在项目中保存研究配置、运行、证据和对话历史。')}</p>
        <button class="btn primary gpi-demo-launch" type="button" data-gpi-demo>${iconHtml('play', 16)} ${tr('Demo: from question to result', '演示：从提问到结果')}</button>
      </div>`;
    }

    function restoringPanel() {
      return `
      <div class="gpi-panel gpi-restoring" aria-busy="true">
        <header class="gpi-head"><div class="gpi-head-title"><div class="gpi-kicker">EasyICU</div><div class="gpi-title">${esc(displayProjectTitle(state.project && state.project.title, projectId()))}</div></div></header>
        <div class="gpi-log" data-gpi-log><div class="gpi-restore-message" role="status" aria-live="polite"><span class="gpi-running-spinner" aria-hidden="true"></span>${tr('Loading this project’s conversations…', '正在读取当前项目的对话…')}</div></div>
        <div class="gpi-compose"><div class="gpi-compose-card"><textarea disabled aria-label="${tr('Message', '消息')}" placeholder="${tr('Continue when the conversation loads', '对话加载后可继续提问')}"></textarea></div></div>
      </div>`;
    }
    function messageHtml(row, options) {
      if (row.childJobHandoff) return '';
      if (row.role === 'decision_record') {
        const records = MODULES.optional('decisionRecords');
        return records ? records.render(row, { tr, esc }) : '';
      }
      const stopHtml = RUN_STOP ? RUN_STOP.render(row) : '';
      if (row.role === 'activity' && stopHtml) {
        // The run record owns the stop's typed cause. The activity's job code
        // alone would add a second, sometimes contradictory explanation.
        return ACTIVITY.render(row, { ...(options && options.trace), failureExplained: true }) + stopHtml + RUN_FILES.render(row);
      }
      if (row.role === 'activity') return ACTIVITY.render(row, options && options.trace) + RUN_FILES.render(row);
      if (row.role === 'quick_answer') return QUICK_ANSWERS ? QUICK_ANSWERS.render(row) : '';
      if (row.role === 'multi_database') return MULTI_DATABASE ? MULTI_DATABASE.render(row) : '';
      if (row.role === 'host_notice') return HOST_JOBS && typeof HOST_JOBS.renderNotice === 'function' ? HOST_JOBS.renderNotice(row) : '';
      if (row.role === 'saved_run') {
        // A saved run with no message of its own says what became of it: a
        // stopped run its cause and the action that works.
        const note = stopHtml || `<p>${tr('Saved run synchronized from this research project.', '已从本研究同步保存的运行记录。')}</p>`;
        return `<article class="gpi-message assistant gpi-saved-run"><div class="gpi-message-body">${note}${RUN_FILES.render(row)}</div></article>`;
      }
      // The demo's record of a host control that was pressed, not typed.
      if (row.role === 'demo_action') {
        return `<article class="gpi-message assistant gpi-workflow-receipt" role="status"><div class="gpi-message-body"><div class="gpi-workflow-receipt-head">${iconHtml('check', 15)}<div><strong>${esc(row.label)}</strong><span>${esc(row.note)}</span></div></div></div></article>`;
      }
      if (row.role === 'workflow_receipt') {
        const rows = row.total_rows == null ? Number.NaN : Number(row.total_rows);
        const files = Number(row.data_file_count);
        const supports = Number(row.support_file_count);
        const isResult = row.receipt_kind === 'extraction_result';
        // A hosted instance shows no server path (screens-guided-pi-hosted.js).
        const hostedOwner = MODULES.optional('hosted');
        const showPath = Boolean(row.output_dir) && !(hostedOwner && hostedOwner.active());
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
            ${showPath ? `<div class="gpi-workflow-receipt-path"><span>${tr('Output folder', '输出文件夹')}</span><code>${esc(row.output_dir)}</code></div>` : ''}
            <p>${isResult
              ? (showPath
                ? tr('Copilot now reads this completed extraction result. The folder path is shown only here and is never inserted as model-authored text.', 'Copilot 现在可以读取这次已完成的抽取结果。文件夹路径只显示在这里，不会被当作模型生成的文字。')
                : tr('Copilot now reads this completed extraction result.', 'Copilot 现在可以读取这次已完成的抽取结果。'))
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
          namedDatabase: HOST_JOBS && typeof HOST_JOBS.namedDatabase === 'function' ? HOST_JOBS.namedDatabase() : null,
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
      return `<article class="gpi-message ${cls}${messageActions.actionsHtml ? ' has-actions' : ''}${streaming ? ' is-streaming' : ''}" data-gpi-message-id="${esc(row.id || '')}"${entryAttr(row)}${streaming ? ' data-gpi-streaming-message' : ''}>
        <div class="gpi-message-body">
          ${messageSkillHtml}${contentHtml}
          ${messageResourcesHtml}
          ${stopHtml}${RUN_FILES.render(row)}
          ${historicalDataConsentHtml}
          ${nextStepHtml}
          ${messageActions.actionsHtml}
          ${followUpsHtml}
        </div>
      </article>`;
    }

    // Each answered gate leaves its record line where it was decided
    // (owner: screens-guided-pi-decision-records.js). A source label the
    // researcher typed already shows its past choice under that reply.
    function withDecisionRecords(timeline) {
      const records = MODULES.optional('decisionRecords');
      if (!records) return timeline;
      const typedSource = timeline.some(row => row && row.role === 'user'
        && DATA_CONSENT && typeof DATA_CONSENT.matchesSourceSelection === 'function'
        && DATA_CONSENT.matchesSourceSelection(state.session, row.text));
      return records.place(timeline, records.rows(state.session)
        .filter(record => !(typedSource && record.code === 'data_chosen')));
    }

    // Each quick-answer or multi-database card follows the turn that
    // produced it (owners: screens-guided-pi-quick-answers.js and
    // screens-guided-pi-multi-database.js).
    function withQuickAnswers(timeline) {
      const placed = QUICK_ANSWERS ? QUICK_ANSWERS.place(timeline, QUICK_ANSWERS.rows(state.session)) : timeline;
      return MULTI_DATABASE ? MULTI_DATABASE.place(placed, MULTI_DATABASE.rows(state.session)) : placed;
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
      const names = {
        question: tr('Question', '问题'),
        idea: tr('Ideas + literature', '选题与文献'),
        setup: tr('Study design', '研究设计'),
        extraction: tr('Extract + review', '提取与审阅'),
        plan: tr('Plan + evidence', '计划与证据'), analysis: tr('Analyze + figures', '分析与图表'),
        interpretation: tr('Interpret', '结果解读'), manuscript: tr('Paper', '论文'),
      };
      return `<nav class="gpi-workflow" aria-label="${tr('EasyICU research workflow', 'EasyICU 科研流程')}">
        <div class="gpi-workflow-meta"><strong>${tr('Research workflow', '科研流程')}</strong><span class="shell-sr-only">${esc(workflow.completed_required_stages || 0)}/${esc(workflow.required_stage_count || 7)} ${tr('required stages complete', '个必需阶段已完成')}</span></div>
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
      const researchLabel = [(connection || research).provider, (connection || research).model].filter(Boolean).join(' · ');
      const separateModels = !connection || (model.id && model.id !== connection.model);
      const stale = sessionIsStale();
      const workspace = agentMode() === 'workspace';
      const freshTask = !state.messages.length;
      const ideaExplorationTurn = !workspace
        && TRANSCRIPT.latestTurnCompletedIdeaExploration(state.messages);
      const showProjectContinuationCards = !workspace && !freshTask && !ideaExplorationTurn;
      const dataConsentRequired = showProjectContinuationCards
        && DATA_CONSENT && DATA_CONSENT.requiresConfirmation(session);
      const fullTimeline = freshTask ? [] : withQuickAnswers(withDecisionRecords(RUN_FILES.timeline(state.messages.concat(state.workflowReceipts))));
      const timeline = state.regenerating && REGENERATION
        ? REGENERATION.visibleRows(fullTimeline, state.regeneration)
        : fullTimeline;
      const activeChild = timeline.slice().reverse().find(row => row.role === 'activity' && row.childJobId && row.status === 'running');
      const interactionLocked = state.busy || Boolean(activeChild) || state.projectLoading;
      // A study job (data preparation, a plan) moves the binding as a turn
      // does and is rebound when it ends; it shows no activity row while it runs.
      const studyJob = state.workflow && state.workflow.active_job;
      const studyJobRunning = Boolean(studyJob && studyJob.present);
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
          return `<article class="gpi-message assistant gpi-turn-intro" data-gpi-message-id="${esc(displayRow.id || '')}"${entryAttr(displayRow)}><div class="gpi-message-body"><div class="gpi-text">${assistantTextHtml(text)}</div></div></article>`;
        }
        return `<div class="gpi-activity-narration-text" data-gpi-message-id="${esc(displayRow.id || '')}"${entryAttr(displayRow)}>${assistantTextHtml(text)}</div>`;
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
      // Workflow completion may arrive before the separate run-file catalog.
      // Use its authoritative stop until that run has a stop in the timeline.
      const pendingStop = showProjectContinuationCards && RUN_STOP
        ? (state.workflow && Array.isArray(state.workflow.runs) ? state.workflow.runs : []).find(run => (
          run.authoritative && RUN_STOP.stopped(run)
          && !timeline.some(row => (row.savedRuns || []).some(saved => (
            saved.run_id === run.run_id && RUN_STOP.stopped(saved)
          )))
        )) : null;
      const conversation = messages + (pendingStop ? RUN_STOP.render({ savedRuns: [pendingStop] }) : '');
      const dataConsentHtml = dataConsentRequired
        ? DATA_CONSENT.render(session, {
          tr, esc, icon: iconHtml,
          namedDemo: HOST_JOBS && typeof HOST_JOBS.namedDemo === 'function' ? HOST_JOBS.namedDemo() : null,
          namedDatabase: HOST_JOBS && typeof HOST_JOBS.namedDatabase === 'function' ? HOST_JOBS.namedDatabase() : null,
        })
        : '';
      const hostedOwner = MODULES.optional('hosted');
      const headerOptions = {
        tr, esc, icon: iconHtml,
        hosted: Boolean(hostedOwner && hostedOwner.active()),
        helpHtml: HELP ? HELP.render() : '',
        projectTitle: projectTitle(),
        sessionTitle: navigationSessionTitle(session),
        busy: interactionLocked,
        // The top bar's 运行中 · elapsed, while a server job runs.
        runningSince: activeChild && Number.isFinite(Number(activeChild.startedAt)) ? Number(activeChild.startedAt) : null,
        runningElapsed: activeChild && ACTIVITY.durationText ? ACTIVITY.durationText(activeChild.startedAt) : '',
        workspace,
        modelBindingKey: modelMenu ? modelMenu.bindingKey(session, projectId()) : '',
        layout: ASIDE.layoutOptions(),
        connectionLabel: (separateModels
          ? [model.id ? tr(`Conversation: ${model.id}`, `对话：${model.id}`) : '',
            researchLabel ? tr(`Analysis: ${researchLabel}`, `分析：${researchLabel}`) : ''].filter(Boolean).join(' / ')
          : researchLabel) || tr('Model connection unavailable', '模型连接暂不可用'),
        connectionTitle: connection
          ? tr('Conversation model', '对话模型')
          : tr('Legacy conversation and analysis bindings', '旧会话的对话与分析绑定'),
      };
      // A plan start the host has not answered with a job yet: nothing to stop.
      const starting = !activeChild && state.planStarting && state.planStarting.sessionId === session.session_id
        ? state.planStarting : null;
      const startingHtml = starting ? `<div class="gpi-compose-running" role="status" aria-live="polite" aria-busy="true" data-gpi-plan-starting>
          <span class="gpi-running-spinner" aria-hidden="true"></span>
          <span><strong>${starting.preparesData ? tr('Preparing the analysis data', '正在准备分析数据') : tr('Starting the research plan', '正在启动研究计划生成')}</strong><small>${starting.preparesData ? tr('EasyICU prepares the data this plan needs and checks the plan against it; a large package takes a few minutes. Analysis has not started.', 'EasyICU 按已确认的方案准备所需数据，并据此复核计划；数据较大时需要几分钟。分析尚未开始。') : tr('EasyICU checks the selected data package first; a large one takes a minute or two.', 'EasyICU 先核对所选数据包，再开始生成；数据较大时需要一两分钟。')}</small></span>
          <time data-gpi-live-elapsed="${Number(starting.startedAt || Date.now())}">${esc(ACTIVITY.durationText ? ACTIVITY.durationText(starting.startedAt) : '')}</time>
        </div>` : '';
      const queueOwner = MODULES.optional('messageQueue');
      const composerCardHtml = `<div class="gpi-compose-card${activeChild || starting ? ' is-running' : ''}">
        ${queueOwner ? queueOwner.render(session.followup_queue, { tr, esc }) : ''}
        ${startingHtml || (activeChild ? `<div class="gpi-compose-running" role="status" aria-live="polite" aria-busy="true">
          <span class="gpi-running-spinner" aria-hidden="true"></span>
          <span><strong>${esc(activeChild.cancelRequested ? tr('Stopping the research task', '正在停止科研任务') : (activeChild.runningTitle || tr('EasyICU research task is running', 'EasyICU 科研任务正在运行')))}</strong><small>${activeChild.cancelRequested ? tr('The cancellation request was sent. Waiting for the current safe checkpoint.', '已发送停止请求，正在等待当前安全检查点结束。') : tr('New messages are paused until this task finishes or asks for confirmation.', '任务完成或需要你确认后，才可继续发送消息。')}</small></span>
          <time data-gpi-live-elapsed="${Number(activeChild.startedAt || Date.now())}">${esc(ACTIVITY.durationText ? ACTIVITY.durationText(activeChild.startedAt) : '')}</time>
          <button class="btn danger sm" type="button" data-gpi-cancel-child-job="${esc(activeChild.childJobId)}" ${activeChild.cancelRequested ? 'disabled' : ''}>${activeChild.cancelRequested ? tr('Stopping…', '正在停止…') : tr('Stop generation', '停止生成')}</button>
        </div>` : `${!workspace && IDEA_SOURCE ? IDEA_SOURCE.status({ tr, esc }) : ''}${STUDY_WORKSPACE.renderReference(projectId(), session.session_id)}${STUDY_WORKSPACE.renderSkillReference(projectId(), session)}<textarea data-gpi-input rows="2" maxlength="12000" placeholder="${state.projectLoading ? tr('Research status is syncing. Continue when it finishes…', '正在同步研究状态，完成后可继续提问……') : workspace ? tr('Ask EasyICU Copilot to create or edit a project artifact — do not paste patient rows or identifiers.', '让 EasyICU 研究助手创建或编辑当前项目产物——请勿粘贴患者行级数据或标识符。') : tr('What ICU research question would you like to study?', '你想研究什么 ICU 科学问题？')}" ${state.projectLoading || stale ? 'disabled' : ''}>${esc(state.draft)}</textarea>
          <div class="gpi-actions">
            <div class="gpi-action-leading">${IDEA_SOURCE ? IDEA_SOURCE.controls({ tr, esc, icon: iconHtml, disabled: interactionLocked || stale, allowIdeaSources: !workspace, extensions: composerExtensions(session) }) : ''}${STUDY_WORKSPACE.renderMaterials(RUN_OUTCOME.collection(state.latestRun, state.workflow), projectId(), session.session_id, interactionLocked || stale || Boolean(state.workflowError), session)}${STUDY_WORKSPACE.renderSkillPicker(projectId(), session, interactionLocked || stale)}${STUDY_WORKSPACE.renderAccessMode(state.accessMode, accessModeLabel, iconHtml)}</div>
            <div class="gpi-action-trailing">${EFFORT_MENU ? EFFORT_MENU.render({ iconHtml, level: session.thinking_level, disabled: interactionLocked || stale }) : ''}${HEADER.renderModelControl(headerOptions)}
            ${state.busy ? `<button class="btn" type="button" data-gpi-send ${state.queueSubmitting || state.projectLoading || stale ? 'disabled' : ''}>${tr('Queue', '排队发送')}</button><button class="btn danger" type="button" data-gpi-stop>${tr('Stop', '停止')}</button>` : `<button class="btn primary" type="button" data-gpi-send aria-label="${tr('Send', '发送')}" title="${tr('Send', '发送')}" ${interactionLocked || stale ? 'disabled' : ''}>${iconHtml('arrow', 15)}</button>`}</div>
          </div>`)}
      </div>`;
      // A research job runs in the server process and outlives the page
      // (jobs.py JobManager.submit; the workflow re-watches it on return).
      const serverNoteHtml = `<p class="gpi-compose-server-note">${esc(tr('EasyICU runs on the server; you can switch conversations or close the page and come back to the progress.', 'EasyICU 在服务器上运行，可以切换对话或关闭页面，回来后继续显示进度。'))}</p>`;
      // A step the host held for the researcher's go-ahead asks above the composer.
      const accessOwner = MODULES.optional('access');
      const actionAskHtml = accessOwner && !workspace
        ? accessOwner.render(accessOwner.pending(state.messages), {
          disabled: interactionLocked || stale || Boolean(state.busy), mode: state.accessMode, agentMode: agentMode(),
        }) : '';
      const emptyResearchHtml = STARTERS && typeof STARTERS.render === 'function'
        ? STARTERS.render({ tr, disabled: interactionLocked || stale,
          composer: `<div class="gpi-compose gpi-entry-compose">${composerCardHtml}</div>` })
        : `<div class="gpi-empty"><strong>${tr('Start with the research question', '先描述研究问题')}</strong></div>`;
      // A turn or job of this conversation moves its own study binding and
      // rebinds when it ends; only a change found at rest asks for a rebind.
      return `
        <div class="gpi-panel${emptyResearch ? ' gpi-empty-session' : ''}${resultsView ? ' gpi-results-session' : ''}">
          ${HEADER.render(headerOptions)}
          <div class="gpi-context-strip">
          ${state.workflowError ? `<div class="gpi-stale" role="alert">${esc(state.workflowError)}<button type="button" data-gpi-refresh-status>${tr('Retry', '重试')}</button></div>` : workflowHtml()}
          ${!workspace && DATA_CONSENT && typeof DATA_CONSENT.renderSelectedSource === 'function'
            ? DATA_CONSENT.renderSelectedSource(session, { tr, esc, icon: iconHtml }) : ''}
          </div>
          ${stale && !interactionLocked && !studyJobRunning ? `<div class="gpi-stale"><strong>${tr('Authority changed', '权威状态已变化')}</strong><span>${tr('The EasyICU study binding, revision, or active run changed. Rebind before continuing.', 'EasyICU 研究绑定、版本或活动运行已变化，请先重新绑定。')}</span><button class="btn sm" type="button" data-gpi-rebind>${tr('Rebind current state', '重新绑定当前状态')}</button></div>` : ''}
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
          ${emptyResearch ? '' : `${activeChild ? serverNoteHtml : ''}<div class="gpi-compose">${actionAskHtml}${composerCardHtml}</div>`}
      </div>`;
    }

    function demoPanel() {
      const demo = MODULES.optional('demo');
      if (!demo || typeof demo.messages !== 'function') {
        return `<div class="gpi-activate"><h2>${tr('Demo unavailable', '演示暂不可用')}</h2><button class="btn" type="button" data-gpi-demo-exit>${tr('Back', '返回')}</button></div>`;
      }
      const button = (resource, label) => RESOURCE_OWNER.button(resource, label);
      // The walkthrough replays the run beat by beat; without it the whole
      // conversation shows at once.
      const walkthrough = MODULES.optional('demoWalkthrough');
      const body = walkthrough
        ? walkthrough.panelHtml({ messageHtml, workflowHtml, iconHtml, button, durationText: ACTIVITY.durationText, syncAside: () => ASIDE && ASIDE.syncProjectWorkflowAside() })
        : `<div class="gpi-log" data-gpi-log>${demo.messages().map(row => messageHtml(row, { interactive: false })).join('')}${typeof demo.resultHtml === 'function' ? demo.resultHtml({ tr, esc, button }) : ''}</div>
        <footer class="gpi-demo-footer"><span>${tr('Open any underlined file to see its record.', '点击带下划线的文件可查看原始记录。')}</span><button class="btn primary" type="button" data-gpi-demo-exit>${tr('Start my own research', '开始我自己的研究')}</button></footer>`;
      const workflow = walkthrough ? walkthrough.workflow() : demo.workflow();
      // A read-only walkthrough of one registered run: the header says what
      // it is in a few words and the note says what it will not touch.
      return `<div class="gpi-panel gpi-demo-panel"${walkthrough ? ' data-gpi-walk-root' : ''}>
        <header class="gpi-head">
          <div><div class="gpi-kicker">${tr('EASYICU COPILOT · DEMO', 'EASYICU COPILOT · 演示')}</div><div class="gpi-title">${tr('A research question, end to end', '一个研究问题的完整流程')}</div></div>
          <div class="gpi-head-meta"><span>${tr('MIMIC-IV · read-only', 'MIMIC-IV · 只读')}</span><button class="gpi-link" type="button" data-gpi-demo-exit>${tr('Back to my project', '返回我的项目')}</button></div>
        </header>
        <div data-gpi-walk-workflow>${workflowHtml(workflow)}</div>
        <div class="gpi-demo-note" role="note">${iconHtml('shield', 16)}<span>${tr('Read-only: the conversation and results come from one registered run. Nothing calls a model or changes your projects.', '只读演示：对话和结果来自一次已登记的真实运行，不会调用模型，也不会改动你的项目。')}</span></div>
        ${body}
      </div>`;
    }

    return Object.freeze({ messageHtml, workflowHtml, sessionPanel, demoPanel, activatePanel, projectRequiredPanel, restoringPanel });
  }

  window.EasyICU.guidedPi.declare('sessionView', { create });
})();
