/* Owner: Idea Mining Zotero source widget. */
(function () {
  'use strict';

  function esc(v) {
    return String(v == null ? '' : v).replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
  }

  function create(deps) {
    const d = deps || {};
    const state = {
      searching: false,
      importing: false,
      query: '',
      result: null,
      selected: null,
      pasteText: '',
      pasteImport: null,
    };
    const t = d.t || ((en) => en);
    const icon = d.icon || (() => '');
    const fieldValue = d.fieldValue || (() => '');
    const collectPayload = d.collectPayload || (() => ({}));
    const applySuggestedPayload = d.applySuggestedPayload || (() => {});
    const repaint = d.repaint || (() => {});
    const setError = d.setError || (() => {});
    const setSourceResolved = d.setSourceResolved || (() => {});
    const setSourceType = d.setSourceType || (() => {});
    const ensureTopicFromTitle = d.ensureTopicFromTitle || (() => {});

    function connectorStatus() {
      const policy = window.EU_CAPABILITIES || {};
      const caps = policy.capabilities || {};
      return caps.zotero_connector || {};
    }

    function connectorEnabled() {
      const policy = window.EU_CAPABILITIES || {};
      const policySettings = policy.settings || {};
      if (Object.prototype.hasOwnProperty.call(policySettings, 'connector_zotero_enabled')) {
        return policySettings.connector_zotero_enabled === true;
      }
      const settings = window.EU_SETTINGS || {};
      return settings.connector_zotero_enabled === true;
    }

    function statusLine() {
      const status = connectorStatus();
      if (status.available) return t('Zotero Desktop is connected. Paste import also works.', 'Zotero Desktop 已连接；也可以直接粘贴导入。');
      if (!connectorEnabled()) return t('Auto-connect is off. Paste DOI, BibTeX, RIS, or title/abstract below to continue.', '自动连接未开启。可在下方粘贴 DOI、BibTeX、RIS 或标题摘要继续。');
      return t('Auto-connect is not reachable right now. Paste import still works locally.', '当前无法自动连接；仍可用下方粘贴导入在本地继续。');
    }

    function applySource(data, fallbackItem) {
      if (data && data.blocked) {
        // Blocked payloads carry no source_adapter, so storing them as the
        // resolved source made the ideas note render a green "Source ready".
        setSourceResolved(null);
        setError(t('Auto-connect is blocked. Paste DOI, BibTeX, RIS, or title/abstract below to continue.', '自动连接被阻断。请在下方粘贴 DOI、BibTeX、RIS 或标题摘要继续。'));
        return;
      }
      setSourceResolved(data);
      state.selected = (data && data.item) || fallbackItem || null;
      applySuggestedPayload(data && data.suggested_payload);
      setSourceType('zotero');
      ensureTopicFromTitle();
      setError(null);
    }

    function runSearch() {
      if (state.searching || state.importing) return;
      collectPayload(document);
      state.query = (document.querySelector('#ideaZoteroQuery') || {}).value || fieldValue('title') || fieldValue('topic');
      state.query = String(state.query || '').trim();
      if (!state.query) {
        setError(t('Enter a title, author, DOI, or keyword first.', '请先输入标题、作者、DOI 或关键词。'));
        repaint();
        return;
      }
      if (!(window.EU_API && window.EU_API.searchZotero)) {
        setError(t('Zotero auto-connect is unavailable. Paste the source below instead.', 'Zotero 自动连接不可用。请改用下方粘贴导入。'));
        repaint();
        return;
      }
      setSourceType('zotero');
      state.searching = true;
      state.result = null;
      setError(null);
      repaint();
      window.EU_API.searchZotero({ query: state.query, limit: 8 })
        .then(data => {
          state.result = data;
          setSourceResolved(null);
        })
        .catch(e => { setError(e.message || String(e)); })
        .finally(() => { state.searching = false; repaint(); });
    }

    function useItem(index) {
      const rows = state.result && Array.isArray(state.result.items) ? state.result.items : [];
      const item = rows[Number(index || 0)];
      if (!item) return;
      if (!(window.EU_API && window.EU_API.zoteroSource)) {
        setError(t('Zotero source adapter is unavailable. Paste the source below instead.', 'Zotero 来源适配器不可用。请改用下方粘贴导入。'));
        repaint();
        return;
      }
      collectPayload(document);
      setSourceType('zotero');
      state.searching = true;
      setError(null);
      repaint();
      window.EU_API.zoteroSource({ item })
        .then(data => applySource(data, item))
        .catch(e => { setError(e.message || String(e)); })
        .finally(() => { state.searching = false; repaint(); });
    }

    function importPaste() {
      if (state.searching || state.importing) return;
      collectPayload(document);
      state.pasteText = (document.querySelector('#ideaZoteroPaste') || {}).value || state.pasteText || '';
      state.pasteText = String(state.pasteText || '').trim();
      if (!state.pasteText) {
        setError(t('Paste a DOI, BibTeX, RIS entry, or title/abstract first.', '请先粘贴 DOI、BibTeX、RIS 条目或标题摘要。'));
        repaint();
        return;
      }
      if (!(window.EU_API && window.EU_API.importZoteroSource)) {
        setError(t('Pasted-source import is unavailable.', '粘贴文献导入不可用。'));
        repaint();
        return;
      }
      setSourceType('zotero');
      state.importing = true;
      setError(null);
      repaint();
      window.EU_API.importZoteroSource({ text: state.pasteText })
        .then(data => {
          state.pasteImport = data;
          applySource(data, data && data.item);
        })
        .catch(e => { setError(e.message || String(e)); })
        .finally(() => { state.importing = false; repaint(); });
    }

    function render() {
      const connectorOn = connectorEnabled();
      const status = connectorStatus();
      const rows = state.result && Array.isArray(state.result.items) ? state.result.items.slice(0, 8) : [];
      const blocked = state.result && state.result.blocked;
      const tone = connectorOn && status.available ? 'ok' : 'warn';
      const pasteReady = !!(state.pasteImport && state.pasteImport.item);
      return `
        <div class="ideas-zotero-source">
          <div class="ideas-zotero-search">
            <label class="field ideas-field"><span>${t('Auto-connect search', '自动连接检索')}</span><input id="ideaZoteroQuery" placeholder="${t('paper title, author, DOI, or keyword', '标题、作者、DOI 或关键词')}" value="${esc(state.query || fieldValue('title') || fieldValue('topic'))}" /></label>
            <button class="btn ${connectorOn ? 'primary' : ''}" type="button" data-idea-zotero-search ${state.searching || !connectorOn ? 'aria-disabled="true"' : ''}>${state.searching ? '<span class="spin"></span>' : icon('search', 13)} ${state.searching ? t('Searching', '检索中') : t('Search library', '检索文献库')}</button>
            <button class="btn sm" type="button" data-idea-open-settings>${t('Settings', '设置')}</button>
          </div>
          <div class="ideas-zotero-status ${tone}">
            <div>${icon(tone === 'ok' ? 'check' : 'shield', 13)}</div>
            <div><b>${connectorOn ? t('Zotero Desktop link', 'Zotero 桌面连接') : t('Paste import available', '可直接粘贴导入')}</b><span>${esc(statusLine())}</span></div>
          </div>
          ${blocked ? `<div class="note warn mt-10"><div class="ico">${icon('shield', 14)}</div><div class="body"><div class="t">${t('Auto-connect unavailable', '自动连接不可用')}</div><div class="d">${t('Paste DOI, BibTeX, RIS, or title/abstract below; no Zotero setup is required.', '请在下方粘贴 DOI、BibTeX、RIS 或标题摘要；不需要配置 Zotero。')}</div></div></div>` : ''}
          ${rows.length ? `<div class="ideas-zotero-results mt-10">
            ${rows.map((row, i) => `<article class="ideas-zotero-row">
              <div class="ideas-zotero-paper">
                <b>${esc(row.title || t('Untitled Zotero item', '未命名 Zotero 条目'))}</b>
                <span>${esc([row.first_author, row.year, row.journal, row.doi ? 'DOI ' + row.doi : ''].filter(Boolean).join(' · '))}</span>
              </div>
              <button class="btn sm" type="button" data-idea-use-zotero="${i}">${t('Use', '使用')}</button>
            </article>`).join('')}
          </div>` : ''}
          <div class="ideas-zotero-paste">
            <label class="field ideas-field"><span>${t('Paste source instead', '直接粘贴文献')}</span><textarea id="ideaZoteroPaste" rows="5" placeholder="${t('Paste DOI, BibTeX, RIS, or title + abstract', '粘贴 DOI、BibTeX、RIS，或标题 + 摘要')}">${esc(state.pasteText)}</textarea></label>
            <div class="ideas-zotero-paste-actions">
              <span>${t('No Zotero setup required. Parsed locally as metadata only.', '不需要配置 Zotero。本地解析，仅保留元数据。')}</span>
              <button class="btn primary" type="button" data-idea-zotero-import ${state.importing ? 'aria-disabled="true"' : ''}>${state.importing ? '<span class="spin"></span>' : icon('check', 13)} ${state.importing ? t('Importing', '导入中') : t('Use pasted source', '使用粘贴文献')}</button>
            </div>
          </div>
          ${state.selected && !pasteReady ? `<div class="ideas-zotero-selected mt-10">${icon('check', 13)} <span>${t('Zotero item selected', '已选择 Zotero 文献')}: ${esc(state.selected.title || state.selected.key || '')}</span></div>` : ''}
          ${pasteReady ? `<div class="ideas-zotero-selected mt-10">${icon('check', 13)} <span>${t('Literature source ready', '文献来源已就绪')}: ${esc((state.pasteImport.item || {}).title || '')}</span></div>` : ''}
        </div>`;
    }

    function wire(root) {
      const search = root.querySelector('[data-idea-zotero-search]');
      if (search) search.addEventListener('click', () => {
        if (search.getAttribute('aria-disabled') === 'true') return;
        runSearch();
      });
      const query = root.querySelector('#ideaZoteroQuery');
      if (query) query.addEventListener('input', () => { state.query = query.value; });
      const paste = root.querySelector('#ideaZoteroPaste');
      if (paste) paste.addEventListener('input', () => { state.pasteText = paste.value; });
      const importBtn = root.querySelector('[data-idea-zotero-import]');
      if (importBtn) importBtn.addEventListener('click', () => {
        if (importBtn.getAttribute('aria-disabled') === 'true') return;
        importPaste();
      });
      root.querySelectorAll('[data-idea-use-zotero]').forEach(btn => btn.addEventListener('click', () => {
        useItem(btn.dataset.ideaUseZotero || 0);
      }));
    }

    function reset() {
      state.searching = false;
      state.importing = false;
      state.query = '';
      state.result = null;
      state.selected = null;
      state.pasteText = '';
      state.pasteImport = null;
    }

    return { render, wire, reset, state };
  }

  window.EU_IDEA_ZOTERO = { create };
})();
