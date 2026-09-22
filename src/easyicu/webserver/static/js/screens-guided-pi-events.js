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
      HOST_JOBS, EFFORT_MENU, api,
    } = options;

    // Outside presses, Escape, and opening another menu are handled for every
    // floating menu by popover-menus.js; this only closes a menu after an
    // action was chosen inside it.
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
      STUDY_WORKSPACE.removeReference();
      STUDY_WORKSPACE.removeSkill();
      STUDY_WORKSPACE.removeMethod();
      rememberSession('');
      const project = window.EasyICU.guidedPi.optional('project');
      if (project && project.syncLocation) project.syncLocation(projectId(), '');
    }

    async function prepareNewTaskFollowUp(prompt) {
      if (!prompt || state.busy || state.childJobId || !projectId()) return;
      clearSessionSelection();
      render();
      await createSession();
      if (state.session) prepareEntryCompose({ text: prompt, intent: '' });
    }

    function wire() {
      if (!state.host) return;
      state.host.addEventListener('click', event => {
        if (RUN_FILES && RUN_FILES.handleClick(event)) return;
        const filePill = event.target.closest('[data-gpi-file]');
        if (filePill) {
          event.preventDefault();
          event.stopPropagation();
          const filename = filePill.dataset.gpiFile;
          const collection = RUN_OUTCOME.collection(state.latestRun, state.workflow);
          const found = collection.find(r => r.artifact === filename || r.filename === filename || (r.title && r.title.includes(filename)));
          if (found) {
            openResource(found);
          } else {
            const preview = window.EasyICU.guidedPi.optional('preview');
            if (preview && preview.open) {
              preview.open({
                kind: filename.endsWith('.png') ? 'figure' : filename.endsWith('.csv') ? 'data_table' : 'document',
                artifact: filename,
                title: filename,
                filename: filename,
              }, projectId(), previewWorkflowContext());
            }
          }
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
          if (question) sendText(question, []);
          return;
        }
        const followUp = event.target.closest('[data-gpi-followup]');
        if (followUp) {
          if (state.busy || state.childJobId || !state.session || state.session.stale?.stale) return;
          const prompt = RUN_OUTCOME.followUps(state.latestRun, state.workflow)[Number(followUp.dataset.gpiFollowup)];
          if (prompt) prepareEntryCompose({ text: prompt, intent: '' });
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
        if (event.target.closest('[data-gpi-skill-close]')) { STUDY_WORKSPACE.closeSkills(state.host, true); return; }
        if (event.target.closest('[data-gpi-skill-remove]')) { STUDY_WORKSPACE.removeSkill(); render(true); return; }
        if (event.target.closest('[data-gpi-builder-remove]')) { STUDY_WORKSPACE.removeBuilder(); render(true); return; }
        if (event.target.closest('[data-gpi-method-remove]')) { STUDY_WORKSPACE.removeMethod(); render(true); return; }
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
            STUDY_WORKSPACE.openMaterials(projectId(), state.session.session_id);
          } else {
            STUDY_WORKSPACE.openSkills(projectId(), state.session);
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
        if (event.target.closest('[data-gpi-demo-exit]')) { closeDemo(); return; }
        if (event.target.closest('[data-gpi-demo]')) { openDemo(); return; }
        const traceExpand = event.target.closest('[data-gpi-trace-expand]');
        if (traceExpand) {
          event.preventDefault();
          const body = traceExpand.closest('.gpi-activity-body');
          const expanded = traceExpand.getAttribute('aria-expanded') !== 'true';
          if (body) body.classList.toggle('is-expanded', expanded);
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
        if (accessMode) { state.accessMode = accessMode.dataset.gpiAccessMode || 'assist'; render(); return; }
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
        if (event.target.closest('[data-gpi-legacy]')) { setShell('legacy'); return; }
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
        if (dataSourceAction) { authorizeDataSource(dataSourceAction); return; }
        if (MESSAGE_ACTIONS.handleClick(event)) return;
        if (event.target.closest('[data-gpi-starter-browse]')) { location.hash = '#skills'; return; }
        if (event.target.closest('[data-gpi-starter-shuffle]')) {
          if (STARTERS && STARTERS.shuffle) STARTERS.shuffle(state.host, tr);
          return;
        }
        const starterAction = STARTERS && STARTERS.actionFromEvent(event, tr);
        if (starterAction && starterAction.kind === 'send') {
          sendText(starterAction.text, [], starterAction.intent);
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
            sendText(discoveryAction.text, [], discoveryAction.intent);
            return;
          }
          sendText(message, governedNextChoiceGrants(nextChoice, message));
          return;
        }
        if (event.target.closest('[data-gpi-next-focus]')) {
          const input = state.host.querySelector('[data-gpi-input]');
          if (input) { input.focus(); input.scrollIntoView({ block: 'nearest' }); }
          return;
        }
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
      function composerPickerTrigger(value) {
        const text = String(value || '');
        return /(^|\s)[@/]$/.test(text) ? text.slice(-1) : '';
      }
      state.host.addEventListener('input', event => {
        if (event.target.matches('[data-gpi-input]')) {
          state.draft = event.target.value;
          // "@" opens the project-resource picker and "/" the Skill picker
          // when typed at the start of the message or after a space — the
          // same two drawers the + menu opens; the trigger character is
          // removed so it never reaches the model.
          const trigger = composerPickerTrigger(event.target.value);
          if (trigger && state.session && !state.busy && !state.childJobId && !event.isComposing) {
            event.target.value = event.target.value.slice(0, -1);
            state.draft = event.target.value;
            if (trigger === '@') STUDY_WORKSPACE.openMaterials(projectId(), state.session.session_id);
            else STUDY_WORKSPACE.openSkills(projectId(), state.session);
            render(true);
            return;
          }
        }
        if (event.target.matches('[data-gpi-starter-search]') && STARTERS && STARTERS.filter) STARTERS.filter(state.host, event.target.value);
        if (event.target.matches('[data-gpi-material-search]')) STUDY_WORKSPACE.filterMaterials(state.host, event.target.value);
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
        if (!event.target.matches('[data-gpi-provider-preset]')) return;
        const form = event.target.closest('[data-gpi-provider-form]');
        if (!form) return;
        const presets = {
          cliproxyapi: { provider: 'easyicu-local', base_url: 'http://127.0.0.1:8317/v1', api_transport: 'openai-completions', model: 'gpt-5.6-luna' },
          // D-P2-4: the custom gateway ships an EMPTY address on purpose. The
          // example domain is placeholder text only (see the input's
          // placeholder) and must never be a submittable value: submitting it
          // would carry the pasted API key to a stand-in host during
          // verification. configureProvider additionally refuses empty and
          // example.* addresses, and the backend rejects example.* outright.
          'custom-openai': { provider: 'custom-openai', base_url: '', api_transport: 'openai-completions', model: '' },
          openai: { provider: 'openai', base_url: 'https://api.openai.com/v1', api_transport: 'openai-responses', model: 'gpt-5.6-luna' },
          openrouter: { provider: 'openrouter', base_url: 'https://openrouter.ai/api/v1', api_transport: 'openai-completions', model: '' },
          deepseek: { provider: 'deepseek', base_url: 'https://api.deepseek.com/v1', api_transport: 'openai-completions', model: 'deepseek-chat' },
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
          if (message) sendText(message, governedNextChoiceGrants(null, message));
          return;
        }
        const form = event.target.closest('[data-gpi-provider-form]');
        if (!form) return;
        event.preventDefault(); configureProvider(form);
      });
    }

    return Object.freeze({ dismissHeaderOverflow, wire, openResource, openResourceButton, referenceResource, revealPendingReview });
  }

  window.EasyICU.guidedPi.declare('events', { create });
})();
