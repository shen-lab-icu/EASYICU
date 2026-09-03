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

  function renderPast(session, ctx) {
    const current = authorization(session);
    if (current.status !== 'confirmed' || current.confirmation_mode !== 'reuse_project_source') return '';
    const selected = selectedScopeAction(current);
    const choice = (action, en, zh) => {
      const isSelected = action === selected;
      const selectedText = isSelected ? ` · ${ctx.tr('Selected', '已选择')}` : '';
      return `<span class="btn${isSelected ? ' primary' : ''}" role="listitem"${isSelected ? ' aria-current="true"' : ''}>${ctx.tr(en, zh)}${selectedText}</span>`;
    };
    return `<section class="gpi-data-consent" aria-label="${ctx.tr('Historical data-preparation choice', '历史数据准备选择')}">
      <span class="gpi-data-consent-icon">${ctx.icon('shield', 16)}</span>
      <div class="gpi-data-consent-body">
        <strong>${ctx.tr('Options provided at the time', '当时提供的选项')}</strong>
        <p>${ctx.esc(sourceLabel(current) || ctx.tr('Validated project source', '已验证的项目数据源'))}</p>
        <div class="gpi-data-consent-actions" role="list">
          ${choice('use_study_required_data', 'Prepare only study-required data (recommended)', '只准备本研究需要的数据（推荐）')}
          ${choice('begin_full_data_selection', 'Extract all supported data', '提取全部支持数据')}
          ${choice('reuse_project_source', 'Reuse the previous complete package', '使用之前的完整数据包')}
        </div>
        <small>${ctx.tr('The recommended option plans first and materializes only named concepts; full extraction opens the local extraction owner; reuse starts no new extraction. This history record is read-only.', '推荐项会先生成计划，再只物化计划点名的概念；全量提取会打开本地提取功能；复用不会启动新的提取。这是只读历史记录。')}</small>
      </div>
    </section>`;
  }

  function render(session, ctx) {
    const current = authorization(session);
    if (current.status === 'pending' && current.reason === 'project_source_confirmation_required') {
      const source = current.source || {};
      const label = [source.label, source.reference_release ? `v${source.reference_release}` : '']
        .filter(Boolean).join(' ');
      return `<section class="gpi-data-consent" aria-label="${ctx.tr('Project data-source confirmation', '确认当前项目数据源')}">
        <span class="gpi-data-consent-icon">${ctx.icon('shield', 16)}</span>
        <div class="gpi-data-consent-body">
          <strong>${ctx.tr('How should EasyICU prepare data for this study?', '这项研究要如何准备数据？')}</strong>
          <p>${ctx.esc(label || ctx.tr('Validated project source', '已验证的项目数据源'))}</p>
          <div class="gpi-data-consent-actions">
            <button class="btn primary" type="button" data-gpi-data-source-action="use_study_required_data">${ctx.tr('Prepare only study-required data (recommended)', '只准备本研究需要的数据（推荐）')}</button>
            <button class="btn" type="button" data-gpi-data-source-action="begin_full_data_selection">${ctx.tr('Extract all supported data', '提取全部支持数据')}</button>
            <button class="btn" type="button" data-gpi-data-source-action="reuse_project_source">${ctx.tr('Reuse the previous complete package', '使用之前的完整数据包')}</button>
          </div>
          <small>${ctx.tr('The recommended option plans first and materializes only named concepts. Full extraction opens the local extraction owner. Reuse starts no new extraction.', '推荐项会先生成计划，再只物化计划点名的概念；全量提取会打开本地提取功能；复用不会启动新的提取。')}</small>
        </div>
      </section>`;
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

  window.EU_GUIDED_PI_DATA_CONSENT = {
    authorization,
    requiresConfirmation,
    selectionInProgress,
    matchesSourceSelection,
    renderPast,
    render,
    actionFromEvent,
  };
})();
