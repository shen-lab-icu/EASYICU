/* Owner: Guided Pi DOM event wiring widget. */
/* Copilot-owned delegated DOM event wiring.
   The parent screen passes explicit state, owners, and actions; this module
   contains no scientific policy and does not own API transport. */
(function () {
  'use strict';

  function create(options) {
    const {
      state, RESOURCE_OWNER, RUN_FILES, MESSAGE_ACTIONS, STARTERS, IDEA_SOURCE, COHORT_ELIGIBILITY,
      DATA_CONSENT, RUN_OUTCOME, STUDY_WORKSPACE, ASIDE, render, projectId, previewWorkflowContext,
      openSession, closeDemo, openDemo, switchMode, loadCodexResearchStatus,
      openAuthorizationPopup, startCodexLogin, cancelCodexLogin, logoutCodex,
      loadCodexModels, tr, apiResearchReady, finishProviderSetup, loadStatus,
      setShell, openStudySetupInConversation, createSession, startEntry,
      previewApprovedPlanDataPackage, confirmWorkflowAction,
      retryFailedExecution,
      rejectWorkflowAction, editWorkflow, confirmCohortEligibility,
      confirmPlanDecision,
      authorizeDataSource, sendText, continueAfterDataSourceConfirmation,
      governedNextChoiceGrants, sendMessage, stopMessage, stopChildJob, rebind,
      togglePresentationPin, configureProvider, rememberSession, recordHostAction,
      HOST_JOBS, EFFORT_MENU, api, CONNECT, PROVIDER_CONTROL, HELP, MESSAGE_QUEUE, QUICK_ANSWERS, MULTI_DATABASE,
    } = options;

    // Outside presses, Escape, and opening another menu are handled for every
    // floating menu by popover-menus.js; this only closes a menu when an
    // action is chosen inside it. The shell listens in the capture phase, so
    // the menu is closed before the action repaints: a menu open at the
    // repaint is reopened.
    function dismissHeaderOverflow(event) {
      const action = event.target && event.target.closest
        ? event.target.closest('.gpi-head-overflow-menu button, .gpi-idea-source-popover button') : null;
      const menu = action && action.closest('.gpi-head-overflow[open], .gpi-idea-source-menu[open]');
      if (menu) menu.removeAttribute('open');
    }

    function reviewActionCode(descriptor) {
      const artifact = String((descriptor && descriptor.artifact) || '');
      if (artifact === 'result_tables.json') return 'review_result_tables';
      if (artifact === 'figure_gallery.json') return 'review_figures';
      if (['manuscript_provenance.json', 'manuscript_scaffold.pdf', 'manuscript_revision.pdf', 'article_report.json'].includes(artifact)) {
        return 'review_manuscript';
      }
      if (artifact === 'scientific_readiness.json') return 'review_scientific_review';
      if (String((descriptor && descriptor.kind) || '') === 'research_report') return 'review_results';
      return '';
    }

    function openResourceButton(resource) {
      openResource(RESOURCE_OWNER.fromButton(resource));
    }

    function openResource(descriptor) {
      const preview = window.EasyICU.guidedPi.optional('preview');
      if (!preview || !preview.open) return;
      if (preview.open(descriptor, projectId(), previewWorkflowContext()) !== true) return;
      const actionCode = reviewActionCode(descriptor);
      if (actionCode) void recordHostAction(actionCode,
        [String((descriptor && descriptor.run_id) || projectId()), String(descriptor.artifact || 'report')].join(':'));
    }

    function referenceResource(descriptor, expectedProjectId) {
      if (projectId() !== expectedProjectId || !state.session || state.busy || state.childJobId) return false;
      if (!STUDY_WORKSPACE.setReference(descriptor, projectId(), state.session.session_id)) return false;
      const input = state.host && state.host.querySelector('[data-gpi-input]');
      if (input) state.draft = input.value;
      STUDY_WORKSPACE.closeMaterials(state.host, false);
      const preview = window.EasyICU.guidedPi.optional('preview');
      if (preview) preview.close();
      render(true);
      requestAnimationFrame(() => {
        const composer = state.host && state.host.querySelector('[data-gpi-input]');
        if (composer) { composer.focus(); composer.scrollIntoView({ block: 'nearest' }); }
      });
      return true;
    }

    function revealPendingReview() {
      const card = state.host && state.host.querySelector('.gpi-confirmation');
      if (!card) return;
      card.scrollIntoView({ block: 'center', behavior: 'smooth' });
      card.setAttribute('tabindex', '-1');
      card.focus({ preventScroll: true });
    }

    function prepareEntryCompose(action) {
      if (action.method) STUDY_WORKSPACE.selectMethodTemplate(projectId(), state.session, action.method);
      state.draft = action.text;
      state.pendingEntryIntent = action.intent;
      state.followUpDraft = action.origin ? {
        origin: action.origin, projectId: projectId ? projectId() : '', sessionId: state.session.session_id,
        selectionRevision: state.sessionSelectionRevision,
      } : null;
      render();
      window.requestAnimationFrame(() => {
        const input = state.host && state.host.querySelector('[data-gpi-input]');
        if (!input) return;
        input.focus();
        input.setSelectionRange(input.value.length, input.value.length);
        input.scrollIntoView({ block: 'nearest' });
      });
    }

    function clearSessionSelection() {
      const preview = window.EasyICU.guidedPi.optional('preview');
      if (preview && preview.close) preview.close();
      state.sessionSelectionRevision += 1;
      state.session = null;
      state.messages = [];
      state.editingMessageId = '';
      state.pendingEntryIntent = '';
      state.followUpDraft = null;
      STUDY_WORKSPACE.removeReference();
      STUDY_WORKSPACE.removeSkill();
      STUDY_WORKSPACE.removeMethod();
      rememberSession('');
      const project = window.EasyICU.guidedPi.optional('project');
      if (project && project.syncLocation) project.syncLocation(projectId(), '');
    }

    async function prepareNewTaskFollowUp(prompt) {
      if (!prompt || !state.session || state.busy || state.childJobId || state.session.stale?.stale || !projectId()) return;
      const expectedProjectId = projectId();
      clearSessionSelection();
      render();
      const pending = createSession();
      const selectionRevision = state.sessionSelectionRevision;
      await pending;
      if (projectId() !== expectedProjectId || state.sessionSelectionRevision !== selectionRevision
        || !state.session || state.busy || state.childJobId || state.session.stale?.stale) return;
      prepareEntryCompose({ text: prompt, intent: '', origin: 'host_prompt' });
    }

    function wire() {
      if (!state.host) return;
      // Mentions (screens-guided-pi-mentions.js) finish host searches and lookups later.
      const MENTIONS = window.EasyICU.guidedPi.optional('mentions');
      if (MENTIONS) MENTIONS.onChange(() => render(true));
      state.host.addEventListener('click', event => {
        if (CONNECT && CONNECT.handleClick(event)) return;
        if (HELP && HELP.handleClick(event)) return;
        if (PROVIDER_CONTROL && PROVIDER_CONTROL.handleClick(event)) return;
        if (RUN_FILES && RUN_FILES.handleClick(event)) return;
        if (QUICK_ANSWERS && QUICK_ANSWERS.handleClick(event)) return;
        if (MULTI_DATABASE && MULTI_DATABASE.handleClick(event)) return;
        const append = event.target.closest('[data-gpi-result-append]');
        if (append) {
          if (!state.session || state.busy || state.childJobId || state.session.stale?.stale) return;
          append.disabled = true;
          void RUN_OUTCOME.proposeAppend(append.dataset.gpiResultAppend).then(changed => {
            if (changed) loadStatus();
            else append.disabled = false;
          });
          return;
        }
        if (event.target.closest('[data-gpi-append-owner]')) {
          if (!state.busy && !state.childJobId) openStudySetupInConversation();
          return;
        }
        if (event.target.closest('[data-gpi-stop-recovery]')) { revealPendingReview(); return; }
        const stopAction = event.target.closest('[data-gpi-stop-action]');
        if (stopAction) {
          if (!state.session || state.busy || state.childJobId || state.session.stale?.stale) return;
          const owner = window.EasyICU.guidedPi.optional('runStop');
          const workflow = state.workflow || {};
          const run = (workflow.runs || []).find(row => row.run_id === stopAction.dataset.gpiStopRun);
          if (!owner || !owner.currentRun(run, workflow, state.latestRun)) return;
          const action = owner.offeredActions(run, workflow).find(row => row.action_id === stopAction.dataset.gpiStopAction);
          const route = owner.actionRoute(action, workflow);
          if (route === 'retry' && typeof retryFailedExecution === 'function') retryFailedExecution();
          if (route === 'confirm' && typeof confirmWorkflowAction === 'function') confirmWorkflowAction();
          if (route === 'edit' && typeof editWorkflow === 'function') editWorkflow();
          return;
        }
        const layoutToggle = event.target.closest('[data-gpi-layout-toggle]');
        if (layoutToggle && ASIDE) {
          const enabled = ASIDE.togglePanel(layoutToggle.dataset.gpiLayoutToggle);
          layoutToggle.setAttribute('aria-pressed', String(enabled));
          layoutToggle.lastElementChild.textContent = enabled ? '✓' : '';
          return;
        }
        const newFollowUp = event.target.closest('[data-gpi-followup-new]');
        if (newFollowUp) {
          const prompt = RUN_OUTCOME.followUps(state.latestRun, state.workflow)[Number(newFollowUp.dataset.gpiFollowupNew)];
          if (prompt) void prepareNewTaskFollowUp(prompt);
          return;
        }
        const dismissFollowUp = event.target.closest('[data-gpi-followup-dismiss]');
        if (dismissFollowUp) {
          if (RUN_OUTCOME.dismissFollowUp(dismissFollowUp.dataset.gpiFollowupDismiss, state.latestRun, state.workflow)) render();
          return;
        }
        const modelFollowUp = event.target.closest('[data-gpi-model-followup]');
        if (modelFollowUp) {
          if (state.busy || state.childJobId || !state.session || state.session.stale?.stale) return;
          const question = String(modelFollowUp.dataset.gpiModelFollowup || '').trim();
          // Both suggestion owners draft first. Preserve the model origin
          // through editing and explicit send without adding approval grants.
          if (question) prepareEntryCompose({ text: question, intent: '', origin: 'model_option' });
          return;
        }
        const followUp = event.target.closest('[data-gpi-followup]');
        if (followUp) {
          if (state.busy || state.childJobId || !state.session || state.session.stale?.stale) return;
          const prompt = RUN_OUTCOME.followUps(state.latestRun, state.workflow)[Number(followUp.dataset.gpiFollowup)];
          if (prompt) prepareEntryCompose({ text: prompt, intent: '', origin: 'host_prompt' });
          return;
        }
        const reviewTab = event.target.closest('[data-gpi-review-tab]');
        if (reviewTab) {
          const review = reviewTab.closest('.gpi-review-summary');
          const index = Number(reviewTab.dataset.gpiReviewTab);
          if (!review || !RUN_OUTCOME.selectReviewTab(index)) return;
          review.querySelectorAll('[data-gpi-review-tab]').forEach(button => {
            button.setAttribute('aria-selected', String(button === reviewTab));
          });
          review.querySelectorAll('[data-gpi-review-panel]').forEach(panel => {
            panel.hidden = Number(panel.dataset.gpiReviewPanel) !== index;
          });
          return;
        }
        if (event.target.closest('[data-gpi-material-close]')) { STUDY_WORKSPACE.closeMaterials(state.host, true); return; }
        const mentionKind = event.target.closest('[data-gpi-mention-kind]');
        if ((mentionKind || event.target.closest('[data-gpi-mention-retry]')) && MENTIONS && state.session) {
          const kind = mentionKind ? mentionKind.dataset.gpiMentionKind : STUDY_WORKSPACE.mentionKind();
          const search = state.host.querySelector('[data-gpi-material-search]');
          STUDY_WORKSPACE.setMaterialCategory(`mention:${kind}`);
          MENTIONS.search(projectId(), state.session, kind, search ? search.value : '');
          render(true); return;
        }
        const mentionAdd = event.target.closest('[data-gpi-mention-add]');
        if (mentionAdd && MENTIONS && state.session && !state.busy && !state.childJobId) {
          void MENTIONS.add(projectId(), state.session, mentionAdd.dataset.gpiMentionAdd).then(added => {
            if (added) STUDY_WORKSPACE.closeMaterials(state.host, false);
            render(true);
          });
          return;
        }
        const mentionRemove = event.target.closest('[data-gpi-mention-remove]');
        if (mentionRemove && MENTIONS && state.session && !state.busy) {
          void MENTIONS.remove(projectId(), state.session, mentionRemove.dataset.gpiMentionRemove).then(() => render(true));
          return;
        }
        if (event.target.closest('[data-gpi-skill-close]')) { STUDY_WORKSPACE.closeSkills(state.host, true); return; }
        if (event.target.closest('[data-gpi-skill-remove]')) { STUDY_WORKSPACE.removeSkill(); render(true); return; }
        if (event.target.closest('[data-gpi-builder-remove]')) { STUDY_WORKSPACE.removeBuilder(); render(true); return; }
        if (event.target.closest('[data-gpi-method-remove]')) { STUDY_WORKSPACE.removeMethod(); render(true); return; }
        // Attached Skills live in the host's draft for the next message (screens-guided-pi-skill-attach.js).
        const attachOwner = window.EasyICU.guidedPi.optional('skillAttach');
        const attachedRemove = event.target.closest('[data-gpi-attached-remove]');
        if (attachedRemove && attachOwner && state.session && !state.busy) {
          void attachOwner.remove(projectId(), state.session, attachedRemove.dataset.gpiAttachedRemove).then(() => render(true));
          return;
        }
        const attachToggle = event.target.closest('[data-gpi-attach-toggle]');
        if (attachToggle && attachOwner && state.session && !state.busy && !state.childJobId) {
          const [source, ...rest] = String(attachToggle.dataset.gpiAttachToggle || '').split(':');
          void attachOwner.toggle(projectId(), state.session, source, rest.join(':')).then(() => render(true));
          return;
        }
        if (event.target.closest('[data-gpi-attached-defaults]') && attachOwner && state.session && !state.busy) {
          void attachOwner.saveDefaults(projectId(), state.session).then(() => render(true));
          return;
        }
        if (event.target.closest('[data-gpi-builder-hub]')) { location.hash = '#skills'; return; }
        if (event.target.closest('[data-gpi-method-hub]')) { location.hash = '#skills'; return; }
        if (event.target.closest('[data-gpi-skill-hub]')) { location.hash = '#skills'; return; }
        const manageExtensions = event.target.closest('[data-gpi-manage-extensions]');
        if (manageExtensions) {
          try { window.sessionStorage.setItem('easyicu.settings.openCapabilityTab', manageExtensions.dataset.gpiManageExtensions || 'overview'); } catch (_) {}
          location.hash = '#settings';
          return;
        }
        const pickerChoice = event.target.closest('[data-gpi-composer-picker]');
        if (pickerChoice) {
          if (state.busy || state.childJobId || !state.session) return;
          if (pickerChoice.dataset.gpiComposerPicker === 'materials') {
            openMentionDrawer();
          } else {
            STUDY_WORKSPACE.openSkills(projectId(), state.session);
            loadAttachable();
          }
          render(true);
          return;
        }
        const skillCatalog = event.target.closest('[data-gpi-skill-catalog]');
        if (skillCatalog) {
          if (STUDY_WORKSPACE.setSkillCatalog(skillCatalog.dataset.gpiSkillCatalog)) render(true);
          return;
        }
        const skillCategory = event.target.closest('[data-gpi-skill-category]');
        if (skillCategory) {
          STUDY_WORKSPACE.setSkillCategory(skillCategory.dataset.gpiSkillCategory);
          render(true); return;
        }
        const materialCategory = event.target.closest('[data-gpi-material-category]');
        if (materialCategory) {
          STUDY_WORKSPACE.setMaterialCategory(materialCategory.dataset.gpiMaterialCategory);
          render(true); return;
        }
        const catalogChoice = event.target.closest('[data-gpi-catalog-select]');
        if (catalogChoice) {
          if (state.busy || state.childJobId || !state.session) return;
          const action = STUDY_WORKSPACE.selectCatalog(catalogChoice);
          if (action) {
            STUDY_WORKSPACE.closeSkills(state.host, false);
            prepareEntryCompose(action);
          }
          return;
        }
        const skillChoice = event.target.closest('[data-gpi-skill-select]');
        if (skillChoice) {
          if (state.busy || state.childJobId || !state.session) return;
          if (STUDY_WORKSPACE.selectSkill(skillChoice, projectId(), state.session)) {
            STUDY_WORKSPACE.closeSkills(state.host, false);
            render(true);
          }
          return;
        }
        const material = event.target.closest('[data-gpi-material-preview], [data-gpi-material-reference]');
        if (material) {
          if (!state.session || state.busy || state.childJobId) return;
          const resource = STUDY_WORKSPACE.selectedMaterial(material, RUN_OUTCOME.collection(state.latestRun, state.workflow), projectId(), state.session.session_id);
          if (!resource) return;
          if (material.hasAttribute('data-gpi-material-reference')) referenceResource(resource, projectId());
          else { STUDY_WORKSPACE.closeMaterials(state.host, false); openResource(resource); }
          return;
        }
        if (event.target.closest('[data-gpi-reference-remove]')) { STUDY_WORKSPACE.removeReference(); render(true); return; }
        if (event.target.closest('[data-gpi-refresh-status]')) { loadStatus(); return; }
        if (IDEA_SOURCE && IDEA_SOURCE.handleClick(event, {
          host: () => state.host, render, tr,
        })) return;
        if (EFFORT_MENU && EFFORT_MENU.handleClick(event, {
          session: () => state.session, busy: () => Boolean(state.busy || state.childJobId),
          projectId, api, render, setSession: value => { state.session = value; },
          setError: message => { state.error = message; },
        })) return;
        const session = event.target.closest('[data-gpi-session]');
        if (session) { openSession(session.dataset.gpiSession); return; }
        const walkthrough = window.EasyICU.guidedPi.optional('demoWalkthrough');
        if (walkthrough && walkthrough.handleClick(event)) return;
        if (event.target.closest('[data-gpi-demo-exit]')) { closeDemo(); return; }
        if (event.target.closest('[data-gpi-demo]')) { openDemo(); return; }
        // A finished run's 运行记录 N scrolls to the run record in the right rail.
        if (event.target.closest('[data-gpi-open-run-record]')) { if (ASIDE && ASIDE.revealRunRecord) ASIDE.revealRunRecord(); return; }
        const traceExpand = event.target.closest('[data-gpi-trace-expand]');
        if (traceExpand) {
          event.preventDefault();
          const body = traceExpand.closest('.gpi-activity-body');
          const expanded = traceExpand.getAttribute('aria-expanded') !== 'true';
          if (body) body.classList.toggle('is-expanded', expanded);
          // A live trace re-renders on every event; its owner keeps the choice.
          const activityOwner = window.EasyICU.guidedPi.optional('activity');
          if (body && body.dataset.gpiTraceKey && activityOwner && activityOwner.traceExpanded) activityOwner.traceExpanded(body.dataset.gpiTraceKey, expanded);
          traceExpand.setAttribute('aria-expanded', String(expanded));
          const label = traceExpand.querySelector('span');
          if (label) label.textContent = expanded ? tr('Collapse', '收起') : tr('Expand all', '展开全部');
          return;
        }
        const operation = event.target.closest('[data-gpi-operation]');
        if (operation) {
          const sourceView = window.EasyICU.guidedPi.require('sourceView');
          const runId = String((state.latestRun && state.latestRun.run_id) || (state.workflow && state.workflow.run_id) || '');
          if (sourceView && sourceView.openOperation) sourceView.openOperation(operation, projectId(), runId, openResource);
          return;
        }
        const resource = event.target.closest('[data-gpi-resource-kind]');
        if (resource) {
          const evidence = resource.closest('[data-gpi-evidence-open]');
          const preview = window.EasyICU.guidedPi.optional('preview');
          if (evidence && preview && typeof preview.openRunEvidence === 'function') {
            preview.openRunEvidence(
              RESOURCE_OWNER.fromButton(resource), projectId(), evidence,
            );
            void recordHostAction('review_scientific_review', [
              String(resource.dataset.gpiResourceRun || projectId()),
              String(evidence.dataset.evidenceId || 'evidence'),
            ].join(':'));
          } else {
            openResourceButton(resource);
          }
          return;
        }
        const modeSwitch = event.target.closest('[data-gpi-mode-switch]');
        if (modeSwitch) { switchMode(modeSwitch.dataset.gpiModeSwitch); return; }
        const accessMode = event.target.closest('[data-gpi-access-mode]');
        if (accessMode) {
          const menu = accessMode.closest('details[data-popover-menu]');
          if (menu) menu.removeAttribute('open');
          state.accessMode = accessMode.dataset.gpiAccessMode || 'assist'; render(); return;
        }
        const researchProvider = event.target.closest('[data-gpi-research-provider]');
        if (researchProvider) {
          state.researchProvider = researchProvider.dataset.gpiResearchProvider === 'codex' ? 'codex' : 'api';
          state.error = '';
          if (state.researchProvider === 'codex') loadCodexResearchStatus(true);
          else render();
          return;
        }
        if (event.target.closest('[data-gpi-codex-login]')) {
          const popup = openAuthorizationPopup();
          startCodexLogin('browser', popup); return;
        }
        if (event.target.closest('[data-gpi-codex-device]')) {
          const popup = openAuthorizationPopup();
          startCodexLogin('device_code', popup); return;
        }
        if (event.target.closest('[data-gpi-codex-cancel]')) { cancelCodexLogin(); return; }
        if (event.target.closest('[data-gpi-codex-logout]')) { logoutCodex(); return; }
        if (event.target.closest('[data-gpi-codex-models]')) { loadCodexModels(true); return; }
        if (event.target.closest('[data-gpi-provider-done]')) {
          if (state.researchProvider === 'codex' && (!state.codexAuth || !state.codexAuth.authentication_verified || !state.researchModel)) {
            state.error = tr('Connect your ChatGPT account and select an account model first.', '请先连接 ChatGPT 账户并选择账户模型。');
            render(); return;
          }
          if (state.researchProvider === 'api' && !apiResearchReady()) {
            state.error = tr('Research Agent currently requires an OpenAI Chat Completions-compatible API connection.', 'Research Agent 当前需要 OpenAI Chat Completions 兼容 API 连接。');
            render(); return;
          }
          Promise.resolve(finishProviderSetup()).then(() => {
            if (!state.showSetup && !state.session && projectId()) void createSession();
          });
          return;
        }
        if (event.target.closest('[data-gpi-setup]')) { state.showSetup = true; setShell('pi'); return; }
        if (event.target.closest('[data-gpi-open]')) { setShell('pi'); return; }
        if (event.target.closest('[data-gpi-study-setup]')) { openStudySetupInConversation(); return; }
        if (event.target.closest('[data-gpi-create]')) { createSession(); return; }
        if (event.target.closest('[data-gpi-dismiss-error]')) { state.error = ''; render(); return; }
      const hostNoticeAction = event.target.closest('[data-gpi-host-notice-action]');
      if (hostNoticeAction && HOST_JOBS) {
        event.preventDefault();
        const notice = hostNoticeAction.closest('[data-gpi-host-notice]');
        void HOST_JOBS.handleAction(
          String(hostNoticeAction.dataset.gpiHostNoticeAction || ''),
          String(notice && notice.dataset.gpiHostNotice || ''),
        );
        return;
      }
        const previewPlanData = event.target.closest('[data-gpi-confirm-preview-data]');
        if (previewPlanData) { previewApprovedPlanDataPackage(previewPlanData); return; }
        const previewAnalysisData = event.target.closest('[data-gpi-run-outcome-data]');
        if (previewAnalysisData) { RUN_OUTCOME.openData(previewAnalysisData); return; }
        const reportRetry = event.target.closest('[data-gpi-run-outcome-retry]');
        if (reportRetry) {
          const reason = reportRetry.dataset.gpiRunOutcomeRetry;
          retryFailedExecution(['report_only', 'restore'].includes(reason) ? reason : 'validation_repair');
          return;
        }
        if (event.target.closest('[data-gpi-confirm-action]')) { confirmWorkflowAction(); return; }
        if (event.target.closest('[data-gpi-confirm-reject]')) { rejectWorkflowAction(); return; }
        if (event.target.closest('[data-gpi-confirm-edit]')) { editWorkflow(); return; }
        const cohortSelection = COHORT_ELIGIBILITY.actionFromEvent(event);
        if (cohortSelection) { confirmCohortEligibility(cohortSelection); return; }
        const planDecision = event.target.closest('[data-gpi-plan-decision-option]');
        if (planDecision) {
          confirmPlanDecision({
            decision_code: planDecision.dataset.gpiPlanDecisionCode,
            option_id: planDecision.dataset.gpiPlanDecisionOption,
          });
          return;
        }
        const namedDemo = event.target.closest('[data-gpi-named-demo]');
        if (namedDemo && HOST_JOBS && typeof HOST_JOBS.useNamedDemo === 'function') { void HOST_JOBS.useNamedDemo(namedDemo.dataset.gpiNamedDemo); return; }
        const dataSourceAction = DATA_CONSENT && DATA_CONSENT.actionFromEvent(event);
        if (dataSourceAction) {
          const database = typeof DATA_CONSENT.databaseFromEvent === 'function' ? DATA_CONSENT.databaseFromEvent(event) : '';
          const sourceId = typeof DATA_CONSENT.sourceIdFromEvent === 'function' ? DATA_CONSENT.sourceIdFromEvent(event) : '';
          authorizeDataSource(dataSourceAction, database || sourceId ? { database, sourceId } : undefined);
          return;
        }
        if (MESSAGE_ACTIONS.handleClick(event)) return;
        if (event.target.closest('[data-gpi-starter-browse]')) { location.hash = '#skills'; return; }
        const starterAction = STARTERS && STARTERS.actionFromEvent(event, tr);
        if (starterAction && starterAction.kind === 'send') {
          sendText(starterAction.text, [], starterAction.intent, true, 'host_prompt');
          return;
        }
        if (starterAction && starterAction.kind === 'compose') {
          prepareEntryCompose(starterAction);
          return;
        }
        if (event.target.closest('[data-gpi-data-source-continue]')) {
          continueAfterDataSourceConfirmation();
          return;
        }
        const nextChoice = event.target.closest('[data-gpi-next-choice]');
        if (nextChoice) {
          const localDatabase = String(nextChoice.dataset.gpiNextLocalDatabase || '').trim();
          if (localDatabase) {
            authorizeDataSource('begin_local_selection', { database: localDatabase });
            return;
          }
          const message = nextChoice.dataset.gpiNextChoice;
          const discoveryAction = STARTERS && STARTERS.actionFromDiscoveryChoice
            ? STARTERS.actionFromDiscoveryChoice(message, tr) : null;
          if (discoveryAction && discoveryAction.kind === 'compose') {
            prepareEntryCompose(discoveryAction);
            return;
          }
          if (discoveryAction && discoveryAction.kind === 'send') {
            sendText(discoveryAction.text, [], discoveryAction.intent, true, 'host_prompt');
            return;
          }
          const messageOrigin = nextChoice.dataset.gpiNextOrigin === 'model' ? 'model_option' : 'host_prompt';
          sendText(message, governedNextChoiceGrants(nextChoice, message), undefined, true, messageOrigin);
          return;
        }
        if (event.target.closest('[data-gpi-next-focus]')) {
          const input = state.host.querySelector('[data-gpi-input]');
          if (input) { input.focus(); input.scrollIntoView({ block: 'nearest' }); }
          return;
        }
        const queuedCancel = event.target.closest('[data-gpi-cancel-followup]');
        if (queuedCancel) { MESSAGE_QUEUE.cancel(queuedCancel.dataset.gpiCancelFollowup); return; }
        if (event.target.closest('[data-gpi-send]')) { sendMessage(); return; }
        if (event.target.closest('[data-gpi-stop]')) { stopMessage(); return; }
        const childStop = event.target.closest('[data-gpi-cancel-child-job]');
        if (childStop) { stopChildJob(childStop.dataset.gpiCancelChildJob); return; }
        if (event.target.closest('[data-gpi-rebind]')) { rebind(); return; }
        if (event.target.closest('[data-gpi-presentation-pin]')) { togglePresentationPin(); return; }
        if (event.target.closest('[data-gpi-config]')) { state.showSetup = true; state.error = ''; render(); return; }
        if (event.target.closest('[data-gpi-new]')) {
          clearSessionSelection();
          render();
          void startEntry();
        }
      });
      // "@" lists the project's results; with none yet it opens on variables.
      function openMentionDrawer() {
        STUDY_WORKSPACE.openMaterials(projectId(), state.session.session_id);
        if (!MENTIONS || STUDY_WORKSPACE.hasMaterials(RUN_OUTCOME.collection(state.latestRun, state.workflow))) return;
        STUDY_WORKSPACE.setMaterialCategory('mention:concept');
        MENTIONS.search(projectId(), state.session, 'concept', '');
      }
      // The picker marks what is attached at once; titles and project defaults arrive after one read.
      function loadAttachable() {
        const attachOwner = window.EasyICU.guidedPi.optional('skillAttach');
        if (!attachOwner || !state.session) return;
        const session = state.session;
        void Promise.all([attachOwner.load(projectId(), session), attachOwner.loadDefaults(projectId())])
          .then(() => { if (state.session === session) render(true); }, () => {});
      }
      function composerPickerTrigger(value) {
        const text = String(value || '');
        return /(^|\s)[@/]$/.test(text) ? text.slice(-1) : '';
      }
      state.host.addEventListener('input', event => {
        if (event.target.matches('[data-gpi-input]')) {
          state.draft = event.target.value;
          if (!String(state.draft || '').trim()) state.followUpDraft = null;
          // "@" opens the project-resource picker and "/" the Skill picker
          // when typed at the start of the message or after a space — the
          // same two drawers the + menu opens; the trigger character is
          // removed so it never reaches the model.
          const trigger = composerPickerTrigger(event.target.value);
          if (trigger && state.session && !state.busy && !state.childJobId && !event.isComposing) {
            event.target.value = event.target.value.slice(0, -1);
            state.draft = event.target.value;
            if (trigger === '@') openMentionDrawer();
            else { STUDY_WORKSPACE.openSkills(projectId(), state.session); loadAttachable(); }
            render(true);
            return;
          }
        }
        if (event.target.matches('[data-gpi-material-search]')) {
          STUDY_WORKSPACE.filterMaterials(state.host, event.target.value);
          const kind = STUDY_WORKSPACE.mentionKind();
          if (kind && MENTIONS && state.session) MENTIONS.search(projectId(), state.session, kind, event.target.value);
        }
        if (event.target.matches('[data-gpi-skill-search]')) STUDY_WORKSPACE.filterSkills(state.host, event.target.value);
      });
      state.host.addEventListener('change', event => {
        if (RUN_FILES && RUN_FILES.handleChange(event)) return;
        if (IDEA_SOURCE && IDEA_SOURCE.handleChange(event, {
          host: () => state.host, render, tr,
          onReady: () => {
            if (!String(state.draft || '').trim()) {
              state.draft = tr(
                'Mine candidate research innovations from this paper and review the supporting literature and EasyICU data boundary.',
                '请从这篇文章中发掘候选创新点，并审阅支持文献和 EasyICU 数据边界。',
              );
              state.pendingEntryIntent = 'idea_mining_entry';
            }
          },
        })) return;
        if (event.target.matches('[data-gpi-codex-model]')) {
          state.researchModel = String(event.target.value || '');
          state.error = ''; render(); return;
        }
        if (PROVIDER_CONTROL && PROVIDER_CONTROL.declaration.handleChange(event)) return;
        if (PROVIDER_CONTROL && PROVIDER_CONTROL.handleChange(event)) return;
      });
      state.host.addEventListener('keydown', event => {
        if (CONNECT && CONNECT.handleKeydown(event)) return;
        if (event.key === 'Escape') {
          if (state.host.querySelector('[data-gpi-material-picker]')) {
            event.preventDefault(); STUDY_WORKSPACE.closeMaterials(state.host, true); return;
          }
          if (state.host.querySelector('[data-gpi-skill-picker]')) {
            event.preventDefault(); STUDY_WORKSPACE.closeSkills(state.host, true); return;
          }
        }
        if (event.target.matches('[data-gpi-input]') && window.EU_COMPOSER_KEYBOARD.enterShouldSend(event)) {
          event.preventDefault(); sendMessage();
        }
      });
      state.host.addEventListener('submit', event => {
        if (MESSAGE_ACTIONS.handleSubmit(event)) return;
        const nextCustomForm = event.target.closest('[data-gpi-next-custom-form]');
        if (nextCustomForm) {
          event.preventDefault();
          const input = nextCustomForm.querySelector('[data-gpi-next-custom-input]');
          const message = String((input && input.value) || '').trim();
          if (message) sendText(message, governedNextChoiceGrants(null, message), undefined, true, 'typed');
          return;
        }
        if (PROVIDER_CONTROL && PROVIDER_CONTROL.declaration.handleSubmit(event)) return;
        const form = event.target.closest('[data-gpi-provider-form]');
        if (!form) return;
        event.preventDefault(); configureProvider(form);
      });
    }

    return Object.freeze({ dismissHeaderOverflow, wire, openResource, openResourceButton, referenceResource, revealPendingReview });
  }

  window.EasyICU.guidedPi.declare('events', { create });
})();
