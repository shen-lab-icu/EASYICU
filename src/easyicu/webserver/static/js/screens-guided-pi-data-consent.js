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

  function render(session, ctx) {
    const current = authorization(session);
    if (current.status === 'pending' && current.reason === 'project_source_confirmation_required') {
      const source = current.source || {};
      const label = [source.label, source.reference_release ? `v${source.reference_release}` : '']
        .filter(Boolean).join(' ');
      return `<section class="gpi-data-consent" aria-label="${ctx.tr('Project data-source confirmation', '确认当前项目数据源')}">
        <span class="gpi-data-consent-icon">${ctx.icon('shield', 16)}</span>
        <div class="gpi-data-consent-body">
          <strong>${ctx.tr('Use the validated data source already bound to this project?', '使用当前项目已验证的数据源吗？')}</strong>
          <p>${ctx.esc(label || ctx.tr('Validated project source', '已验证的项目数据源'))}</p>
          <div class="gpi-data-consent-actions">
            <button class="btn primary" type="button" data-gpi-data-source-action="reuse_project_source">${ctx.tr('Use current project source', '使用当前项目数据源')}</button>
            <button class="btn" type="button" data-gpi-data-source-action="begin_local_selection">${ctx.tr('Choose another local source', '选择其他本地数据源')}</button>
          </div>
          <small>${ctx.tr('This confirms only the path-free source identity for this conversation; it does not start extraction or analysis.', '这只确认本次对话使用的数据源身份，不会启动提取或分析。')}</small>
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
    render,
    actionFromEvent,
  };
})();
