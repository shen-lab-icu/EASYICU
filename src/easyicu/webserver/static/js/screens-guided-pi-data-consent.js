/* Owner: Guided Pi data-consent gate widget. */
/* Conversation-level data-source confirmation for Guided Copilot.
   StudyContext remains the scientific owner; this module only renders and
   decodes the per-session consent gate. */
(function () {
  'use strict';

  // Confirmations the researcher made on a validated source EasyICU offered:
  // the project's own, or a registered export of the database they named.
  const CHOSEN_SOURCE_MODES = new Set(['reuse_project_source', 'select_registered_source']);

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
    if (current.status !== 'confirmed' || !CHOSEN_SOURCE_MODES.has(current.confirmation_mode)) {
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
    // In the context strip the body floats over the conversation, so it is a
    // popover menu: an outside press or Escape closes it.
    return `<details class="gpi-data-consent" data-popover-menu aria-label="${ctx.tr('Confirmed study data source', '本研究已确认的数据源')}">
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
    if (current.status !== 'confirmed' || !CHOSEN_SOURCE_MODES.has(current.confirmation_mode)) return '';
    const selected = selectedScopeAction(current);
    const decision = current.confirmation_mode === 'select_registered_source'
      ? ctx.tr('An export EasyICU already held · only study-required data', '使用 EasyICU 已有的数据 · 只准备研究所需数据')
      : selected === 'begin_full_data_selection'
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

  // One registered export as the researcher compares it: name, date, size.
  function exportFacts(row, ctx) {
    const stays = Number(row.stays);
    const modules = Number(row.module_count);
    return [
      String(row.label || '').trim(),
      row.generated_date ? ctx.tr(`generated ${row.generated_date}`, `${row.generated_date} 生成`) : '',
      Number.isFinite(stays) && stays > 0
        ? ctx.tr(`${stays.toLocaleString('en-US')} ICU stays`, `${stays.toLocaleString('en-US')} 例 ICU 住院`) : '',
      Number.isFinite(modules) && modules > 0 ? ctx.tr(`${modules} modules`, `${modules} 个模块`) : '',
    ].filter(Boolean).join(' · ');
  }

  function useExportButton(database, row, ctx, label, className) {
    return `<button class="${className}" type="button" data-gpi-data-source-action="use_registered_source" data-gpi-data-source-database="${ctx.esc(database.key)}" data-gpi-data-source-id="${ctx.esc(row.source_id)}">${label}</button>`;
  }

  function folderButton(database, ctx, primary) {
    return `<button class="btn${primary ? ' primary' : ''}" type="button" data-gpi-data-source-action="begin_local_selection" data-gpi-data-source-database="${ctx.esc(database.key)}">${ctx.tr('Choose a local folder', '选择本地文件夹')}</button>`;
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
    const named = current.status === 'pending' && ctx.namedDemo && ctx.namedDemo.id ? ctx.namedDemo : null;
    if (named) {
      // The question already names this official demo: offer it directly.
      const verb = named.pending ? ctx.tr('Preparing…', '正在准备…')
        : named.prepared ? ctx.tr('Use it for this conversation', '用于本次会话')
          : ctx.tr('Prepare it and use it here', '准备并用于本次会话');
      return `<section class="gpi-data-consent" aria-label="${ctx.tr('Bind data source', '绑定数据源')}"><div class="gpi-data-consent-body">
        <strong>${ctx.tr('Your question names this data', '你的问题指定了这份数据')}</strong>
        <p>${ctx.esc(named.label)}${named.prepared ? ctx.esc(ctx.tr(' · already prepared on this computer', ' · 本机已准备')) : ''}</p>
        <div class="gpi-data-consent-actions">
          <button class="btn primary" type="button" data-gpi-named-demo="${ctx.esc(named.id)}" ${named.pending ? 'disabled' : ''}>${verb}</button>
          <button class="btn" type="button" data-gpi-data-source-action="begin_local_selection">${ctx.tr('Choose another source', '选择其他数据源')}</button>
        </div>
        <small>${ctx.tr('Official demo data only; EasyICU registers it and confirms it as this conversation’s source, then proposes the plan. Confirming a source does not approve analysis.', '仅官方 Demo 数据；EasyICU 注册后将其确认为本次会话的数据源，再拟定研究计划。确认数据源不等于批准分析。')}</small>
      </div></section>`;
    }
    const database = current.status === 'pending' && ctx.namedDatabase && ctx.namedDatabase.key ? ctx.namedDatabase : null;
    const offer = database && database.offer ? database.offer : null;
    if (offer && offer.state === 'loading') {
      // Until the host answers, offer only the folder, and not as the primary
      // action, so no button changes under the researcher's pointer.
      return `<section class="gpi-data-consent" aria-label="${ctx.tr('Bind data source', '绑定数据源')}" aria-busy="true"><div class="gpi-data-consent-body">
        <strong>${ctx.tr('Your question names ', '你的问题指定了 ')}${ctx.esc(database.label)}</strong>
        <p>${ctx.tr(`Looking for ${database.label} data EasyICU already holds…`, `正在查找 EasyICU 中已有的 ${database.label} 数据…`)}</p>
        <div class="gpi-data-consent-actions">${folderButton(database, ctx, false)}</div>
      </div></section>`;
    }
    const recommended = offer && offer.state === 'ready' && offer.recommended && offer.recommended.source_id
      ? offer.recommended : null;
    const tied = offer && offer.state === 'ready' && !recommended
      ? (offer.choices || []).filter(row => row && row.source_id).slice(0, 4) : [];
    if (recommended) {
      return `<section class="gpi-data-consent" aria-label="${ctx.tr('Bind data source', '绑定数据源')}"><div class="gpi-data-consent-body">
        <strong>${ctx.tr('Your question names ', '你的问题指定了 ')}${ctx.esc(database.label)}</strong>
        <p>${ctx.tr(`EasyICU already holds validated ${database.label} data:`, `EasyICU 中已有一份验证过的 ${database.label} 数据：`)}</p>
        <p class="gpi-data-consent-facts">${ctx.esc(exportFacts(recommended, ctx))}</p>
        <div class="gpi-data-consent-actions">
          ${useExportButton(database, recommended, ctx, ctx.tr('Use this data', '使用这份数据'), 'btn primary')}
          ${folderButton(database, ctx, false)}
        </div>
        <small>${ctx.tr('EasyICU then proposes the research plan and prepares only the data it needs. Choosing a source does not approve analysis.', '选定后，EasyICU 会拟定研究计划，只准备计划所需的数据。选择数据源不等于批准分析。')}</small>
      </div></section>`;
    }
    if (tied.length) {
      return `<section class="gpi-data-consent" aria-label="${ctx.tr('Bind data source', '绑定数据源')}"><div class="gpi-data-consent-body">
        <strong>${ctx.tr('Your question names ', '你的问题指定了 ')}${ctx.esc(database.label)}</strong>
        <p>${ctx.tr(`EasyICU holds ${tied.length} validated ${database.label} exports and cannot tell which one this study should use.`, `EasyICU 中有 ${tied.length} 份验证过的 ${database.label} 数据，无法替你判断这项研究该用哪一份。`)}</p>
        <ul class="gpi-data-consent-choices">${tied.map(row => `<li>
          <span>${ctx.esc(exportFacts(row, ctx))}</span>
          ${useExportButton(database, row, ctx, ctx.tr('Use this one', '使用这份'), 'btn sm')}
        </li>`).join('')}</ul>
        <div class="gpi-data-consent-actions">${folderButton(database, ctx, false)}</div>
        <small>${ctx.tr('Choosing a source does not approve analysis.', '选择数据源不等于批准分析。')}</small>
      </div></section>`;
    }
    if (database) {
      // The question already names this database: open its source selection
      // directly instead of asking which database to use.
      return `<section class="gpi-data-consent" aria-label="${ctx.tr('Bind data source', '绑定数据源')}"><div class="gpi-data-consent-body">
        <strong>${ctx.tr('Your question names ', '你的问题指定了 ')}${ctx.esc(database.label)}</strong>
        <p>${ctx.tr(`Choose the ${database.label} data for this conversation. EasyICU then proposes the research plan; no analysis starts yet.`, `选择本次会话要用的 ${database.label} 数据，EasyICU 随后拟定研究计划，此时不会开始分析。`)}</p>
        <div class="gpi-data-consent-actions">
          <button class="btn primary" type="button" data-gpi-data-source-action="begin_local_selection" data-gpi-data-source-database="${ctx.esc(database.key)}">${ctx.tr(`Choose ${database.label} data`, `选择 ${database.label} 数据`)}</button>
          <button class="btn" type="button" data-gpi-data-source-action="begin_local_selection">${ctx.tr('Choose another source', '选择其他数据源')}</button>
        </div>
      </div></section>`;
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
          <button class="btn" type="button" data-gpi-data-source-action="cancel_local_selection">${ctx.tr('Leave local selection', '放弃本地选择')}</button>
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

  // The database a card action is scoped to, when the card names one.
  function databaseFromEvent(event) {
    const target = event && event.target && event.target.closest
      ? event.target.closest('[data-gpi-data-source-action]')
      : null;
    return target ? String(target.dataset.gpiDataSourceDatabase || '') : '';
  }

  // The registered export a "use this data" action names, when it names one.
  function sourceIdFromEvent(event) {
    const target = event && event.target && event.target.closest
      ? event.target.closest('[data-gpi-data-source-action]')
      : null;
    return target ? String(target.dataset.gpiDataSourceId || '') : '';
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
    databaseFromEvent,
    sourceIdFromEvent,
  });
})();
