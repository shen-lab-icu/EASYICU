/* Owner: Guided Pi data-consent gate widget. */
/* Conversation-level data-source confirmation for Guided Copilot.
   StudyContext remains the scientific owner; this module only renders and
   decodes the per-session consent gate. */
(function () {
  'use strict';

  function authorization(session) {
    return session && session.data_source_authorization
      ? session.data_source_authorization
      : { status: 'legacy_confirmed' };
  }

  function requiresConfirmation(session) {
    const status = String(authorization(session).status || '');
    return status === 'pending' || status === 'selection_in_progress';
  }

  function selectionInProgress(session) {
    return authorization(session).status === 'selection_in_progress';
  }

  function sourceLabel(current) {
    const source = current && current.source ? current.source : {};
    const label = String(source.label || '').trim();
    const release = String(source.reference_release || '').trim();
    if (!release || label.toLowerCase().endsWith(`v${release}`.toLowerCase())) return label;
    return [label, `v${release}`].filter(Boolean).join(' ');
  }

  function matchesSourceSelection(session, text) {
    const current = authorization(session);
    if (current.status !== 'confirmed' || current.confirmation_mode !== 'reuse_project_source') {
      return false;
    }
    const expected = sourceLabel(current).replace(/\s+/g, ' ').trim().toLowerCase();
    const actual = String(text || '').replace(/\s+/g, ' ').trim().toLowerCase();
    return Boolean(expected && actual === expected);
  }

  function selectedScopeAction(current) {
    if (current.extraction_scope === 'all_supported') return 'begin_full_data_selection';
    if (current.extraction_scope === 'reuse_prepared_full') return 'reuse_project_source';
    return 'use_study_required_data';
  }

  function renderSelectedSource(session, ctx) {
    const current = authorization(session);
    // This is current host state, not a reconstructed user message or approval.
    if (current.status !== 'confirmed' || current.confirmation_mode !== 'select_local_source') return '';
    const label = sourceLabel(current);
    if (!label) return '';
    return `<details class="gpi-data-consent" aria-label="${ctx.tr('Confirmed study data source', '本研究已确认的数据源')}">
      <summary>${ctx.tr('Data source confirmed: ', '已确认数据源：')}${ctx.esc(label)}</summary>
      <div class="gpi-data-consent-body">
        <p>${ctx.tr('This source was selected in the local data picker and confirmed for this conversation.', '这份数据已在本地数据选择器中选定，并确认用于本次会话。')}</p>
        ${current.confirmed_at ? `<p>${ctx.tr('Confirmed at: ', '确认时间：')}${ctx.esc(current.confirmed_at)}</p>` : ''}
        <p>${ctx.tr('Source confirmation does not approve analysis. The research plan and prepared data are reviewed separately.', '确认数据源不等于批准分析。研究计划和准备后的数据仍需分别审阅。')}</p>
      </div>
    </details>`;
  }

  function renderPast(session, ctx) {
    const current = authorization(session);
    if (current.status === 'confirmed' && current.confirmation_mode === 'agent_default_study_required') {
      return `<section class="gpi-data-consent" aria-label="${ctx.tr('Automatic data-preparation policy', '自动数据准备策略')}">
        <span class="gpi-data-consent-icon">${ctx.icon('shield', 16)}</span>
        <div class="gpi-data-consent-body">
          <strong>${ctx.tr('EasyICU will prepare only the data required by the reviewed plan', 'EasyICU 将只准备审阅后计划所需的数据')}</strong>
          <p>${ctx.esc(sourceLabel(current) || ctx.tr('Validated project source', '已验证的项目数据源'))}</p>
          <small>${ctx.tr('This is an automatic system policy, not a choice the researcher had to configure.', '这是系统自动策略，不是要求研究者配置的选项。')}</small>
        </div>
      </section>`;
    }
    if (current.status !== 'confirmed' || current.confirmation_mode !== 'reuse_project_source') return '';
    const selected = selectedScopeAction(current);
    const decision = selected === 'begin_full_data_selection'
      ? ctx.tr('All supported data was selected', '已选择全部支持数据')
      : selected === 'reuse_project_source'
        ? ctx.tr('The existing project package was reused', '已使用项目中已有的数据包')
        : ctx.tr('Only study-required data was selected', '已选择只准备研究所需数据');
    return `<section class="gpi-past-decision" aria-label="${ctx.tr('Historical data-preparation choice', '历史数据准备选择')}">
      <span aria-hidden="true">${ctx.icon('check', 14)}</span>
      <strong>${ctx.tr('Data source confirmed', '已确认数据源')}</strong>
      <span>${ctx.esc(sourceLabel(current) || ctx.tr('Validated project source', '已验证的项目数据源'))} · ${ctx.esc(decision)}</span>
    </section>`;
  }

  function render(session, ctx) {
    const current = authorization(session);
    if (current.status === 'pending' && current.reason === 'project_source_confirmation_required') {
      const source = current.source || {};
      const label = [source.label, source.reference_release ? `v${source.reference_release}` : '']
        .filter(Boolean).join(' ');
      return `<section class="gpi-data-consent" aria-label="${ctx.tr('Confirm data source', '确认数据源')}">
        <span class="gpi-data-consent-icon">${ctx.icon('shield', 16)}</span>
        <div class="gpi-data-consent-body">
          <strong>${ctx.tr('Which data should this study use?', '这项研究使用哪份数据？')}</strong>
          <p>${ctx.esc(label || ctx.tr('Validated project source', '已验证的项目数据源'))}</p>
          <div class="gpi-data-consent-actions">
            <button class="btn primary" type="button" data-gpi-data-source-action="use_study_required_data">${ctx.tr('Confirm this source', '确认使用这份数据')}</button>
            <button class="btn" type="button" data-gpi-data-source-action="begin_local_selection">${ctx.tr('Choose another source', '选择其他数据源')}</button>
          </div>
          <small>${ctx.tr('After you confirm the source, EasyICU will propose a plan and prepare only its required data. Source selection does not approve analysis.', '确认数据源后，EasyICU 会拟定计划，只准备计划所需的数据。选择数据源不等于批准分析。')}</small>
        </div>
      </section>`;
    }
    if (current.status === 'pending') {
      return `<section class="gpi-data-consent" aria-label="${ctx.tr('Bind data source', '绑定数据源')}"><div class="gpi-data-consent-body">
        <strong>${ctx.tr('Next, choose the data for this question', '接下来，请为这个问题选择数据源')}</strong>
        <p>${ctx.tr('Select and confirm a local dataset. EasyICU will then propose the research plan; no analysis starts yet.', '选择并确认本地数据后，EasyICU 会据此拟定研究计划，此时不会开始分析。')}</p>
        <button class="btn primary" type="button" data-gpi-data-source-action="begin_local_selection">${ctx.tr('Choose data source', '选择数据源')}</button>
      </div></section>`;
    }
    if (!selectionInProgress(session)) return '';
    return `<section class="gpi-data-consent" aria-label="${ctx.tr('Local data selection', '本地数据选择')}">
      <span class="gpi-data-consent-icon">${ctx.icon('shield', 16)}</span>
      <div class="gpi-data-consent-body">
        <strong>${ctx.tr('Local data selection is open', '已打开本地数据选择')}</strong>
        <p>${ctx.tr('Data tools remain locked until EasyICU validates and saves the selected source.', 'EasyICU 验证并保存所选来源之前，数据工具保持锁定。')}</p>
        <div class="gpi-data-consent-actions">
          <button class="btn primary" type="button" data-gpi-data-source-action="begin_local_selection">${ctx.tr('Return to local folder selection', '返回本地目录选择')}</button>
        </div>
        <small>${ctx.tr('Paths remain in the EasyICU host UI and are never sent to the model.', '目录路径只保留在 EasyICU 本机界面，不会发送给模型。')}</small>
      </div>
    </section>`;
  }

  function actionFromEvent(event) {
    const target = event && event.target && event.target.closest
      ? event.target.closest('[data-gpi-data-source-action]')
      : null;
    return target ? String(target.dataset.gpiDataSourceAction || '') : '';
  }

  window.EasyICU.guidedPi.declare('dataConsent', {
    authorization,
    requiresConfirmation,
    selectionInProgress,
    matchesSourceSelection,
    renderSelectedSource,
    renderPast,
    render,
    actionFromEvent,
  });
})();
