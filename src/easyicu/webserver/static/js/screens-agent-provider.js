/* Agent Projects provider-panel owner.
   Keeps provider catalog and account/API status presentation out of the
   already-large screens-agent.js route module. This module is pure: the route
   passes immutable state and rendering helpers; no shared mutable closure
   state is copied across files. */
(function () {
  'use strict';

  const PROVIDERS = Object.freeze([
    Object.freeze({ id: 'codex', label: 'Codex account', mode: 'account' }),
    Object.freeze({ id: 'openai', label: 'OpenAI API', mode: 'api' }),
    Object.freeze({ id: 'openrouter', label: 'OpenRouter API', mode: 'api' }),
    Object.freeze({ id: 'deepseek', label: 'DeepSeek API', mode: 'api' }),
    Object.freeze({ id: 'anthropic', label: 'Claude API', mode: 'api' }),
    Object.freeze({ id: 'custom', label: 'Custom / compatible API', mode: 'api' }),
  ]);

  function render(options) {
    const {
      realData,
      currentStudy,
      source,
      contextBlocker,
      providerState,
      runState,
      t,
      icon,
      esc,
    } = options;
    if (!realData || currentStudy.empty || currentStudy.mode !== 'analysis') return '';
    const st = providerState.status || {};
    const limits = st.limits || {};
    const envFile = st.env_file || {};
    const missing = Array.isArray(st.missing) ? st.missing : [];
    const ready = !!st.ready;
    const accountMode = st.authentication_mode === 'account_session';
    const canRun = !!(source && ready && providerState.consent && !runState.active && !contextBlocker);
    const disabledReason = (contextBlocker && contextBlocker.reason) || (!source
      ? t('No active export source', '没有 active export 源')
      : !ready
      ? (missing.length ? missing.join(', ') : t('Provider not ready', 'provider 未就绪'))
      : !providerState.consent
      ? t('Per-run confirmation required', '需要逐次确认')
      : runState.active
      ? t('Run already in progress', '已有运行进行中')
      : '');
    const credentialOkay = accountMode
      ? !!st.account_session_present
      : !!st.credential_present;
    const credentialText = accountMode
      ? (st.authentication_verified === true
        ? t('account login verified', '账户登录已验证')
        : st.account_session_present
        ? t('Codex runtime ready; account checked on call', 'Codex 运行环境就绪；调用时验证账户')
        : t('account login unavailable', '账户登录不可用'))
      : (st.credential_present
        ? t('key env present', 'key env 已配置')
        : t('key env missing', 'key env 缺失'));
    const details = accountMode
      ? [
          [t('Account status', '账户状态'), st.account_session_status || '—'],
          [t('Model source', '模型来源'), st.model_source || 'account_default'],
          [t('Transport', '传输方式'), st.provider_identity || st.provider || '—'],
          [t('Private env file', '私有 env 文件'), t('not used for account login', '账户登录不读取 env 密钥文件')],
          [t('Budget', '预算'), `${Number(limits.max_external_calls_per_run || 1)} call · ${Number(limits.max_output_tokens || 1200)} max tokens`],
        ]
      : [
          [t('Credential source', '凭据来源'), st.credential_source || (st.credential_env_candidates || []).join(' / ') || '—'],
          [t('Model source', '模型来源'), st.model_source || (st.model_env_candidates || []).join(' / ') || '—'],
          [t('Base URL source', 'Base URL 来源'), st.base_url_source || (st.base_url_env_candidates || []).join(' / ') || '—'],
          [t('Private env file', '私有 env 文件'), envFile.status ? `${envFile.status}${Array.isArray(envFile.loaded_keys) && envFile.loaded_keys.length ? ' · ' + envFile.loaded_keys.join(' / ') : ''}` : '—'],
          [t('Budget', '预算'), `${Number(limits.max_external_calls_per_run || 1)} call · ${Number(limits.max_output_tokens || 1200)} max tokens`],
        ];
    return `
      <div class="card pad">
        <div class="row" style="justify-content:space-between;align-items:baseline;">
          <div>
            <div class="eyebrow">${t('External provider scaffold', '外部 provider 骨架')}</div>
            <div class="panel-sub" style="margin-top:4px;">${t('Uses this machine\'s signed-in Codex account or a configured API to generate a bounded plan and draft scaffold. It does not run a complete research analysis, and secrets are never shown or written to artifacts.', '使用本机已登录的 Codex 账户或已配置 API 生成受限计划与草稿骨架。这不是完整科研分析，登录凭据与密钥均不会显示或写入产物。')}</div>
          </div>
          <button class="btn sm ghost" data-ag-provider-refresh>${icon('refresh', 12)} ${t('Refresh', '刷新')}</button>
        </div>
        <div class="row wrap gap-6 mt-12">
          ${PROVIDERS.map(item => `<button class="btn sm ${providerState.provider === item.id ? 'primary' : 'ghost'}" data-ag-provider="${item.id}">${esc(item.label)}</button>`).join('')}
        </div>
        <div class="note-line mt-8" style="font-size:11px;color:var(--ink-4);">${icon('shield', 11)} ${t('Codex can reuse this machine\'s signed-in account. Claude and other API providers use server-owned environment configuration. EasyICU remains local-only by default.', 'Codex 可复用本机已登录账户；Claude 与其他 API provider 使用服务器自有环境配置。EasyICU 默认仍仅限本机访问。')}</div>
        ${providerState.loading ? `<div class="note info mt-12"><div class="ico">${icon('shield', 16)}</div><div class="body"><span class="t">${t('Checking provider readiness', '正在检查 provider 就绪状态')}</span><span class="d">${t('No research prompt is sent during this check.', '此检查不会发送科研提示词。')}</span></div></div>` : ''}
        ${providerState.error ? `<div class="note warn mt-12"><div class="ico">${icon('alert', 16)}</div><div class="body"><span class="t">${t('Provider status unavailable', 'provider 状态不可用')}</span><span class="d">${esc(providerState.error)}</span></div></div>` : ''}
        <div class="row wrap gap-6 mt-12">
          <span class="pill ${st.ai_enabled ? 'ok' : 'warn'}" style="height:22px;"><span class="dot"></span>AI ${st.ai_enabled ? t('enabled', '已开启') : t('off', '关闭')}</span>
          <span class="pill ${credentialOkay ? 'ok' : 'warn'}" style="height:22px;"><span class="dot"></span>${credentialText}</span>
          <span class="pill ${st.model_present ? 'ok' : 'warn'}" style="height:22px;"><span class="dot"></span>${st.model_present ? t('model available', '模型可用') : t('model missing', '模型缺失')}</span>
          <span class="pill ${ready ? 'ok' : 'warn'}" style="height:22px;"><span class="dot"></span>${ready ? t('ready', '就绪') : t('blocked', '受阻')}</span>
        </div>
        <div class="cols-2 mt-12" style="gap:8px;">
          ${details.map(([key, value]) => `
            <div style="padding:8px 10px;background:var(--surface-2);border-radius:var(--r-2);min-width:0;">
              <div class="eyebrow" style="font-size:9.5px;">${key}</div>
              <div class="mono" style="font-size:11.5px;color:var(--ink);margin-top:3px;overflow:hidden;text-overflow:ellipsis;">${esc(value)}</div>
            </div>`).join('')}
        </div>
        <label class="rtodo-row mt-12" style="background:var(--surface-2);">
          <input type="checkbox" data-ag-external-consent ${providerState.consent ? 'checked' : ''} />
          <span class="rtodo-t">${t('I authorize this external model call for this run only', '我只授权本次运行进行外部模型调用')}</span>
          <span class="rtodo-ref mono">per_run_opt_in</span>
        </label>
        <div class="row gap-8 mt-12">
          <button class="btn primary sm" data-ag-external-run aria-disabled="${canRun ? 'false' : 'true'}">${icon('file', 12)} ${t('Generate provider scaffold', '生成 provider 骨架')}</button>
          ${accountMode ? `<button class="btn ghost sm" data-ag-account-pipeline-run aria-disabled="${canRun ? 'false' : 'true'}">${icon('play', 12)} ${t('Start Research Agent planner canary', '启动 Research Agent Planner canary')}</button>` : ''}
          <span style="font-size:11px;color:var(--ink-4);align-self:center;">${canRun ? t('Will remain analysis_only unless STRICT evidence and human review pass.', '除非 STRICT evidence 与人工审阅通过，否则仍保持 analysis_only。') : esc(disabledReason)}</span>
        </div>
      </div>`;
  }

  window.AGENT_PROVIDER_PANEL = Object.freeze({ PROVIDERS, render });
})();
