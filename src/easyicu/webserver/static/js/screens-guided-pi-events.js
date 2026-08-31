/* Copilot-owned delegated DOM event wiring.
   The parent screen passes explicit state, owners, and actions; this module
   contains no scientific policy and does not own API transport. */
(function () {
  'use strict';

  function create(options) {
    const {
      state, RESOURCE_OWNER, MESSAGE_ACTIONS, STARTERS, COHORT_ELIGIBILITY,
      DATA_CONSENT, RUN_OUTCOME, render, projectId, previewWorkflowContext,
      openSession, closeDemo, openDemo, switchMode, loadCodexResearchStatus,
      openAuthorizationPopup, startCodexLogin, cancelCodexLogin, logoutCodex,
      loadCodexModels, tr, apiResearchReady, finishProviderSetup, loadStatus,
      setShell, openStudySetupInConversation, createSession,
      previewApprovedPlanDataPackage, confirmWorkflowAction,
      retryFailedExecution,
      rejectWorkflowAction, editWorkflow, confirmCohortEligibility,
      confirmPlanDecision,
      authorizeDataSource, sendText, continueAfterDataSourceConfirmation,
      governedNextChoiceGrants, sendMessage, stopMessage, stopChildJob, rebind,
      togglePresentationPin, configureProvider,
    } = options;

    function dismissHeaderOverflow(event) {
      const menu = state.host && state.host.querySelector('.gpi-head-overflow[open]');
      if (!menu) return;
      const action = event.target && event.target.closest
        ? event.target.closest('.gpi-head-overflow-menu button') : null;
      if (!menu.contains(event.target) || action) menu.removeAttribute('open');
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
            window.EU_GUIDED_PI_PREVIEW.open(
              RESOURCE_OWNER.fromButton(resource), projectId(), previewWorkflowContext(),
            );
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
          finishProviderSetup(); return;
        }
        if (event.target.closest('[data-gpi-retry]')) { loadStatus(); return; }
        if (event.target.closest('[data-gpi-setup]')) { state.showSetup = true; setShell('pi'); return; }
        if (event.target.closest('[data-gpi-open]')) { setShell('pi'); return; }
        if (event.target.closest('[data-gpi-study-setup]')) { openStudySetupInConversation(); return; }
        if (event.target.closest('[data-gpi-legacy]')) { setShell('legacy'); return; }
        if (event.target.closest('[data-gpi-create]')) { createSession(); return; }
        const previewPlanData = event.target.closest('[data-gpi-confirm-preview-data]');
        if (previewPlanData) { previewApprovedPlanDataPackage(previewPlanData); return; }
        const previewAnalysisData = event.target.closest('[data-gpi-run-outcome-data]');
        if (previewAnalysisData) { RUN_OUTCOME.openData(previewAnalysisData); return; }
        if (event.target.closest('[data-gpi-run-outcome-retry]')) { retryFailedExecution('validation_repair'); return; }
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
        const dataSourceAction = DATA_CONSENT && DATA_CONSENT.actionFromEvent(event);
        if (dataSourceAction) { authorizeDataSource(dataSourceAction); return; }
        if (event.target.closest('[data-gpi-data-demo]')) {
          sendText(tr(
            'I do not have local data yet. Show only the official EasyICU demo datasets and explain their limits. Do not download or use one until I choose it. Offer only each exact demo or continuing study planning without data; do not offer a local full-database workflow.',
            '我还没有本地数据。请只列出 EasyICU 官方 Demo 数据并说明局限；在我选择前不要下载或使用。下一步只提供每个准确 Demo 或继续无数据规划，不要提供本地完整数据库工作流。',
          ));
          return;
        }
        if (MESSAGE_ACTIONS.handleClick(event)) return;
        const starterAction = STARTERS && STARTERS.actionFromEvent(event);
        if (starterAction && starterAction.kind === 'send') {
          sendText(starterAction.text, []);
          return;
        }
        if (starterAction && starterAction.kind === 'compose') {
          state.draft = starterAction.text;
          render();
          window.requestAnimationFrame(() => {
            const input = state.host && state.host.querySelector('[data-gpi-input]');
            if (!input) return;
            input.focus();
            input.setSelectionRange(input.value.length, input.value.length);
            input.scrollIntoView({ block: 'nearest' });
          });
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
        if (event.target.closest('[data-gpi-cancel-setup]')) { state.showSetup = false; state.error = ''; render(); return; }
        if (event.target.closest('[data-gpi-new]')) { state.sessionSelectionRevision += 1; state.session = null; state.messages = []; state.editingMessageId = ''; rememberSession(''); render(); }
      });
      state.host.addEventListener('input', event => {
        if (event.target.matches('[data-gpi-input]')) state.draft = event.target.value;
      });
      state.host.addEventListener('change', event => {
        if (event.target.matches('[data-gpi-codex-model]')) {
          state.researchModel = String(event.target.value || '');
          state.error = ''; render(); return;
        }
        if (!event.target.matches('[data-gpi-provider-preset]')) return;
        const form = event.target.closest('[data-gpi-provider-form]');
        if (!form) return;
        const presets = {
          cliproxyapi: { provider: 'easyicu-local', base_url: 'http://127.0.0.1:8317/v1', api_transport: 'openai-completions', model: 'gpt-5.6-luna' },
          'custom-openai': { provider: 'custom-openai', base_url: 'https://example.com/v1', api_transport: 'openai-completions', model: '' },
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
          const menu = state.host.querySelector('.gpi-head-overflow[open]');
          if (menu) {
            menu.removeAttribute('open');
            const summary = menu.querySelector('summary');
            if (summary) summary.focus();
            return;
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

    return Object.freeze({ dismissHeaderOverflow, wire });
  }

  window.EU_GUIDED_PI_EVENTS = Object.freeze({ create });
})();
