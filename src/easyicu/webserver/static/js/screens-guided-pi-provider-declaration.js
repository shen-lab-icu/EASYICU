/* Owner: Guided Pi credentialed-data declaration control. */
/* The local user's statement of how the connected API model handles data.
   The host owns the record, its binding to one exact connection, and what
   the statement permits (pi_copilot/provider_config.py): a credentialed
   source is used only with on_device or zero_retention_declared. This owner
   shows the current record and submits a change from the settings page. */
(function () {
  'use strict';

  const VALUES = ['unspecified', 'on_device', 'zero_retention_declared'];

  function optionLabel(value, tr) {
    if (value === 'on_device') return tr('The model runs on this computer', '模型在本机运行');
    if (value === 'zero_retention_declared') return tr('The service keeps no data', '服务不保留数据');
    return tr('Not declared', '未声明');
  }

  // The radio the page shows: an unsaved choice first, then the record.
  // A lapsed record belonged to another connection, so nothing is chosen.
  function shownValue(state) {
    const record = state.declaration || {};
    if (VALUES.includes(state.declarationChoice)) return state.declarationChoice;
    if (record.lapsed === true || record.status_code === 'pi_provider_declaration_lapsed') return 'unspecified';
    return VALUES.includes(record.value) ? record.value : 'unspecified';
  }

  function savedValue(state) {
    const record = state.declaration || {};
    if (record.lapsed === true || record.status_code === 'pi_provider_declaration_lapsed') return '';
    return VALUES.includes(record.value) ? record.value : 'unspecified';
  }

  function statusNote(options) {
    const { state, config, tr, esc } = options;
    const record = state.declaration || {};
    const code = String(record.status_code || '');
    const note = (kind, text) => `<div class="gpi-config-note ${kind}"><span class="gpi-dot"></span>${esc(text)}</div>`;
    if (code === 'pi_provider_declaration_lapsed') {
      return note('warn', tr('The connection changed, so the earlier declaration no longer applies. Choose again for this connection.', '连接已更改，之前的声明不再适用。请为当前连接重新选择。'));
    }
    if (code === 'pi_provider_declaration_administrator_configuration' || record.change_authority === 'administrator_configuration') {
      return note('', tr('The administrator set this declaration in the configuration, so it cannot be changed here.', '这项声明由管理员在配置中设定，无法在此修改。'));
    }
    if (code === 'pi_provider_declaration_record_unavailable') {
      return note('warn', tr('EasyICU could not read the declaration record, so this model is treated as undeclared. Check that the EasyICU data folder is readable, then try again.', 'EasyICU 无法读取声明记录，因此按未声明处理这个模型。请检查 EasyICU 数据文件夹是否可读，然后重试。'));
    }
    // A recorded declaration changes the connection fingerprint, so the host
    // requires a new verification before the declaration applies.
    if (code === 'pi_provider_declaration_recorded' && savedValue(state) !== 'unspecified'
      && config && config.connection_verified === false) {
      return note('warn', tr('Verify the connection again to apply this declaration.', '请重新验证连接，这项声明才会生效。'));
    }
    return '';
  }

  function renderSection(options) {
    const { state, tr, esc } = options;
    const record = state.declaration;
    if (options.staticPreview || !record) return '';
    if (record.status_code === 'pi_provider_declaration_connection_required') return '';
    const locked = record.may_change !== true;
    const busy = state.declarationBusy === true;
    const shown = shownValue(state);
    const unchanged = shown === savedValue(state);
    const radios = VALUES.map(value => `
          <label class="gpi-declaration-option"><input type="radio" name="declaration" value="${value}" ${value === shown ? 'checked' : ''} ${locked || busy ? 'disabled' : ''}><span>${esc(optionLabel(value, tr))}</span></label>`).join('');
    return `
      <form class="gpi-provider-section gpi-declaration" data-gpi-declaration-form aria-labelledby="gpi-declaration-title">
        <div class="gpi-provider-heading"><h3 id="gpi-declaration-title">${tr('Credentialed data', '需授权的数据')}</h3></div>
        <p class="gpi-declaration-notice">${tr(
          'For credentialed databases such as MIMIC-IV and eICU, EasyICU uses a model only if it runs on this computer or its service keeps no data, as the PhysioNet data use agreement requires. Choose an option only after confirming it in the service’s terms.',
          '对于 MIMIC-IV、eICU 等需授权的数据库，只有当模型在本机运行、或其服务不保留数据时，EasyICU 才会使用它，这是 PhysioNet 数据使用协议的要求。请先在服务条款中确认，再作选择。',
        )}</p>
        <fieldset class="gpi-declaration-options" aria-labelledby="gpi-declaration-title" ${locked ? 'disabled' : ''}>${radios}
        </fieldset>
        ${statusNote(options)}
        ${state.declarationError ? `<div class="gpi-error inline">${esc(state.declarationError)}</div>` : ''}
        ${locked ? '' : `<button class="btn gpi-declaration-save" type="submit" ${busy || unchanged ? 'disabled' : ''}>${busy ? tr('Saving…', '正在保存…') : tr('Save', '保存')}</button>`}
      </form>`;
  }

  function create(options) {
    const { state, api, tr, render, errorText, reloadStatus } = options;

    async function load(renderAfter) {
      if (!api().loadPiCopilotProviderDeclaration) return;
      try {
        state.declaration = await api().loadPiCopilotProviderDeclaration();
        state.declarationError = '';
      } catch (error) {
        state.declaration = null;
        if (renderAfter) state.declarationError = errorText(error);
      }
      state.declarationChoice = '';
      if (renderAfter) render();
    }

    function handleChange(event) {
      const form = event.target.closest && event.target.closest('[data-gpi-declaration-form]');
      if (!form) return false;
      if (event.target.matches('[name="declaration"]')) {
        state.declarationChoice = String(event.target.value || '');
        state.declarationError = '';
        render();
      }
      return true;
    }

    async function save(form) {
      const record = state.declaration;
      if (state.declarationBusy || !record || record.may_change !== true || !api().savePiCopilotProviderDeclaration) return;
      const value = String(new FormData(form).get('declaration') || '');
      if (!VALUES.includes(value)) return;
      state.declarationBusy = true; state.declarationError = ''; render();
      try {
        state.declaration = await api().savePiCopilotProviderDeclaration({
          connection: record.connection,
          value,
          notice_key: record.notice_key,
          notice_version: record.notice_version,
          channel: 'settings_ui',
        });
        state.declarationChoice = '';
        if (reloadStatus) await reloadStatus();
      } catch (error) {
        // A stale form reloads the record it should have shown.
        if (error && /connection_changed|notice_mismatch/.test(String(error.code || ''))) await load(false);
        state.declarationError = errorText(error);
      } finally {
        state.declarationBusy = false; render();
      }
    }

    function handleSubmit(event) {
      const form = event.target.closest && event.target.closest('[data-gpi-declaration-form]');
      if (!form) return false;
      event.preventDefault(); save(form);
      return true;
    }

    return Object.freeze({ load, handleChange, handleSubmit });
  }

  window.EasyICU.guidedPi.declare('providerDeclaration', { renderSection, create });
})();
