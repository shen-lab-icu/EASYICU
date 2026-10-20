/* Owner: Guided Pi provider/account control widget. */
/* Copilot-owned provider/account event controller.
   The parent screen passes its mutable session state and narrow callbacks;
   this owner handles only account login, model discovery, and API verification.
   An API connection is made in two steps: connect (the service lists its
   models), then choose a model and verify-and-save. */
(function () {
  'use strict';

  function create(options) {
    const {
      state, api, tr, render, runtimeReady, shellReady,
      connectionConfigured, connectionReady, errorText,
    } = options;

    function stopCodexPoll() {
      if (state.codexPoll) window.clearTimeout(state.codexPoll);
      state.codexPoll = null;
    }

    async function loadCodexModels(renderAfter) {
      if (!api().loadPiCopilotCodexModels) return;
      try {
        const payload = await api().loadPiCopilotCodexModels();
        state.codexModels = Array.isArray(payload && payload.models) ? payload.models : [];
        if (!state.researchModel || !state.codexModels.some(row => row.id === state.researchModel)) {
          const preferred = state.codexModels.find(row => row.is_default) || state.codexModels[0];
          state.researchModel = preferred ? String(preferred.id || '') : '';
        }
      } catch (error) {
        // Preserve the last verified catalog through a transient account probe.
        if (renderAfter) state.error = errorText(error);
      }
      if (renderAfter) render();
    }

    async function loadCodexResearchStatus(renderAfter) {
      if (!api().loadPiCopilotCodexStatus) return;
      try {
        const payload = await api().loadPiCopilotCodexStatus();
        state.codexAuth = payload && payload.auth ? payload.auth : null;
        if (state.codexAuth && state.codexAuth.authentication_verified) {
          stopCodexPoll();
          state.codexLogin = null;
          await loadCodexModels(false);
        } else if (state.codexAuth && state.codexAuth.account_session_status === 'codex_auth_login_pending') {
          stopCodexPoll();
          state.codexPoll = window.setTimeout(() => loadCodexResearchStatus(true), 1500);
        }
      } catch (error) {
        state.codexAuth = null;
        if (renderAfter) state.error = errorText(error);
      }
      if (renderAfter) render();
    }

    function safeAuthUrl(value) {
      // D-P1-3: same semantics as literature.safeUrl (https + hostname +
      // no userinfo) plus the Codex allowlist. Never navigate a backend URL
      // that fails this gate; report via errorText and close the popup.
      const registry = window.EasyICU && window.EasyICU.guidedPi
        && typeof window.EasyICU.guidedPi.optional === 'function'
        ? window.EasyICU.guidedPi.optional('literature') : null;
      if (registry && typeof registry.safeUrl === 'function') {
        const cleaned = registry.safeUrl(value);
        try {
          const parsed = new URL(cleaned);
          if (parsed.hostname.toLowerCase() !== 'auth.openai.com') return '';
          if (parsed.port) return '';
          return cleaned;
        } catch (_) { return ''; }
      }
      try {
        const parsed = new URL(String(value || ''));
        if (parsed.protocol !== 'https:' || !parsed.hostname || parsed.username || parsed.password) return '';
        if (parsed.port) return '';
        if (parsed.hostname.toLowerCase() !== 'auth.openai.com') return '';
        return parsed.href;
      } catch (_) { return ''; }
    }

    async function startCodexLogin(flow, popup) {
      if (state.codexBusy || !api().startPiCopilotCodexLogin) return;
      state.codexBusy = true; state.error = ''; render();
      try {
        if (api().saveSetting) await api().saveSetting('ai_enabled', true);
        const runtimePayload = await api().loadPiCopilotStatus();
        state.runtime = runtimePayload && runtimePayload.runtime;
        const payload = await api().startPiCopilotCodexLogin(flow || 'browser');
        state.codexAuth = payload && payload.auth ? payload.auth : state.codexAuth;
        state.codexLogin = payload && payload.login_started ? {
          auth_url: payload.auth_url || payload.verification_url || '',
          user_code: payload.user_code || '',
        } : null;
        const authUrl = String((payload && (payload.auth_url || payload.verification_url)) || '');
        if (authUrl) {
          const safeUrl = safeAuthUrl(authUrl);
          if (!safeUrl) {
            if (popup && !popup.closed) popup.close();
            state.error = errorText({ code: 'codex_auth_url_invalid' });
          } else if (popup && !popup.closed) popup.location.href = safeUrl;
          else window.open(safeUrl, '_blank', 'noopener,noreferrer');
        } else if (popup && !popup.closed) {
          popup.close();
        }
        await loadCodexResearchStatus(false);
      } catch (error) {
        if (popup && !popup.closed) popup.close();
        state.error = errorText(error);
      } finally {
        state.codexBusy = false; render();
      }
    }

    function openAuthorizationPopup() {
      const popup = window.open('about:blank', '_blank');
      if (popup) {
        try { popup.opener = null; } catch (error) {}
      }
      return popup;
    }

    async function cancelCodexLogin() {
      if (state.codexBusy || !api().cancelPiCopilotCodexLogin) return;
      state.codexBusy = true; state.error = ''; render();
      try {
        const payload = await api().cancelPiCopilotCodexLogin();
        stopCodexPoll();
        state.codexAuth = payload && payload.auth ? payload.auth : null;
        state.codexLogin = null;
      } catch (error) { state.error = errorText(error); }
      finally { state.codexBusy = false; render(); }
    }

    async function logoutCodex() {
      if (state.codexBusy || !api().logoutPiCopilotCodex) return;
      state.codexBusy = true; state.error = ''; render();
      try {
        const payload = await api().logoutPiCopilotCodex();
        stopCodexPoll();
        state.codexAuth = payload && payload.auth ? payload.auth : null;
        state.codexLogin = null; state.codexModels = []; state.researchModel = '';
      } catch (error) { state.error = errorText(error); }
      finally { state.codexBusy = false; render(); }
    }

    // D-P2-4: documentation-reserved example hosts (RFC 2606) are placeholder
    // text only and must never receive a verification probe with the key.
    function presetHostname(value) {
      try {
        return new URL(String(value || '')).hostname.toLowerCase().replace(/\.+$/, '') || '';
      } catch (_) {
        return '';
      }
    }

    function isExampleHostname(host) {
      // Includes the reserved .example top-level name the placeholder uses.
      return !!host && ['example', 'example.com', 'example.org', 'example.net', 'example.edu']
        .some(root => host === root || host.endsWith('.' + root));
    }

    // The credential typed into the open form lives here, in page memory,
    // only while that form is open: never in state, storage or markup. The
    // screen re-renders the whole form, so it is captured before and put
    // back after each render.
    let heldKey = '';

    function presets() {
      const owner = window.EasyICU.guidedPi.optional('provider');
      return (owner && owner.PRESETS) || {};
    }

    function signature(values) {
      const owner = window.EasyICU.guidedPi.optional('provider');
      return owner && owner.connectionSignature ? owner.connectionSignature(values) : '';
    }

    function providerForm(host) {
      return host && host.querySelector ? host.querySelector('[data-gpi-provider-form]') : null;
    }

    function connectionValues(form) {
      const data = new FormData(form);
      return {
        provider: String(data.get('provider') || '').trim(),
        base_url: String(data.get('base_url') || '').trim(),
        api_transport: String(data.get('api_transport') || 'openai-completions'),
      };
    }

    function captureForm(host) {
      const form = providerForm(host);
      // A closed form discards what it held; reopening shows the saved connection.
      if (!form) { heldKey = ''; state.apiDraft = null; state.modelCatalog = null; return; }
      const data = new FormData(form);
      const preset = form.querySelector('[data-gpi-provider-preset]');
      const previous = state.apiDraft || {};
      state.apiDraft = {
        ...connectionValues(form),
        preset: preset ? preset.value : previous.preset,
        model: data.has('model') ? String(data.get('model') || '') : previous.model,
      };
      heldKey = String(data.get('api_key') || '');
    }

    function restoreForm(host) {
      const form = providerForm(host);
      const input = form && form.querySelector('[name="api_key"]');
      if (input && heldKey) input.value = heldKey;
    }

    function catalogMatches(values) {
      return !!(state.modelCatalog && state.modelCatalog.models.length
        && state.modelCatalog.signature === signature(values));
    }

    function forgetCredential(form) {
      heldKey = '';
      const keyInput = form && form.querySelector('[name="api_key"]');
      if (keyInput) keyInput.value = '';
    }

    // A changed service clears the credential (it was typed for the other
    // service) and the model list; a changed address or protocol clears
    // only the list, which then belongs to another endpoint.
    function handleChange(event) {
      const form = event.target.closest && event.target.closest('[data-gpi-provider-form]');
      if (!form) return false;
      if (event.target.matches('[data-gpi-provider-preset]')) {
        const selected = presets()[event.target.value];
        if (!selected) return true;
        ['provider', 'base_url', 'api_transport'].forEach(name => {
          const field = form.elements.namedItem(name);
          if (field) field.value = selected[name];
        });
        forgetCredential(form);
        state.modelCatalog = null; state.error = '';
        if (state.apiDraft) delete state.apiDraft.model;
        render(); return true;
      }
      if (event.target.matches('[name="base_url"], [name="api_transport"], [name="provider"]')) {
        if (state.modelCatalog && !catalogMatches(connectionValues(form))) {
          state.modelCatalog = null; render();
        }
        return true;
      }
      return false;
    }

    function handleClick(event) {
      const button = event.target.closest && event.target.closest('[data-gpi-provider-rediscover]');
      if (!button) return false;
      const form = button.closest('[data-gpi-provider-form]');
      if (form) discoverModels(form);
      return true;
    }

    function addressRefused(form, baseUrl) {
      // D-P2-4: the custom-openai preset leaves the address empty and the
      // example domain is placeholder text only. Refuse both here — before
      // any request could carry the API key to a stand-in host — and
      // require the user to type a real gateway address. The backend
      // (validate_credential_endpoint) rejects example.* a second time.
      if (baseUrl && !isExampleHostname(presetHostname(baseUrl))) return false;
      forgetCredential(form);
      state.error = tr(
        'Enter your gateway address before connecting. The example address is only a placeholder and is never submitted.',
        '请先填写网关地址再连接。示例地址只是占位文本，不会提交。',
      );
      render();
      return true;
    }

    async function discoverModels(form) {
      if (state.modelDiscovering || state.setupSaving || !form || !api().discoverPiCopilotProviderModels) return;
      const values = connectionValues(form);
      if (addressRefused(form, values.base_url)) return;
      const apiKey = String(new FormData(form).get('api_key') || '').trim();
      state.modelDiscovering = true; state.error = ''; state.modelCatalog = null; render();
      try {
        const payload = await api().discoverPiCopilotProviderModels({ ...values, api_key: apiKey, enable_ai: true });
        const models = Array.isArray(payload && payload.models) ? payload.models.map(String) : [];
        const recommended = String((payload && payload.recommended) || '');
        state.modelCatalog = { signature: signature(values), models, recommended: models.includes(recommended) ? recommended : '' };
      } catch (error) {
        state.error = errorText(error);
      } finally {
        state.modelDiscovering = false; render();
      }
    }

    // Submitting the form connects first; once the service has listed its
    // models for exactly this address, it verifies and saves the choice.
    async function configureProvider(form) {
      if (state.setupSaving || state.modelDiscovering || !form) return;
      const values = connectionValues(form);
      if (!catalogMatches(values)) { discoverModels(form); return; }
      if (addressRefused(form, values.base_url)) return;
      const data = new FormData(form);
      const apiKey = String(data.get('api_key') || '').trim();
      state.setupSaving = true; state.error = '';
      const submit = form.querySelector('[type="submit"]');
      if (submit) { submit.disabled = true; submit.textContent = tr('Verifying…', '正在验证…'); }
      try {
        const payload = await api().savePiCopilotProviderConfig({
          ...values,
          api_key: apiKey,
          model: String(data.get('model') || '').trim(),
          enable_ai: true,
        });
        state.runtime = payload && payload.runtime;
        forgetCredential(providerForm(state.host));
        state.apiDraft = null; state.modelCatalog = null;
        if (!runtimeReady()) {
          state.error = tr('The model connection was saved, but the Copilot runtime is not ready yet.', '模型连接已保存，但研究助手运行环境尚未就绪。');
          return;
        }
        state.error = '';
      } catch (error) {
        const reported = error && error.details && error.details.available_models;
        if (Array.isArray(reported) && state.modelCatalog) state.modelCatalog = { ...state.modelCatalog, models: reported.map(String) };
        state.error = errorText(error);
      } finally {
        state.setupSaving = false; render();
      }
    }

    async function finishProviderSetup() {
      if (!connectionConfigured()) return;
      state.setupSaving = true; state.error = ''; render();
      try {
        if (!shellReady() && api().saveSetting) {
          await api().saveSetting('ai_enabled', true);
          const payload = await api().loadPiCopilotStatus();
          state.runtime = payload && payload.runtime;
        }
        if (!connectionReady()) {
          throw new Error(tr('The selected model connection is not ready yet.', '所选模型连接尚未就绪。'));
        }
        state.showSetup = false;
      } catch (error) {
        state.error = errorText(error);
      } finally {
        state.setupSaving = false; render();
      }
    }

    return Object.freeze({
      stopCodexPoll,
      loadCodexModels,
      loadCodexResearchStatus,
      startCodexLogin,
      openAuthorizationPopup,
      cancelCodexLogin,
      logoutCodex,
      configureProvider,
      finishProviderSetup,
      captureForm,
      restoreForm,
      handleChange,
      handleClick,
    });
  }

  window.EasyICU.guidedPi.declare('providerControl', { create });
})();
