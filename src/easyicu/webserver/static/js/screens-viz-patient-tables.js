/* Patient Review table owner: one-module lazy pages with stale-response guards. */
(function () {
  'use strict';

  const host = (window.EU_PATIENT_REVIEW = window.EU_PATIENT_REVIEW || {});
  const MAX_CACHE_ENTRIES = 12;
  const previewCache = new Map();
  const state = {
    sourceKey: '',
    module: null,
    page: 1,
    pageSize: 24,
    loading: false,
    error: '',
    requestSeq: 0,
  };

  function reset() {
    state.sourceKey = '';
    state.module = null;
    state.page = 1;
    state.pageSize = 24;
    state.loading = false;
    state.error = '';
    state.requestSeq += 1;
    previewCache.clear();
  }

  function previews(drill) {
    const tables = drill && drill.data_tables || {};
    return Array.isArray(tables.table_previews) ? tables.table_previews : [];
  }

  function cacheKey(module, page, pageSize) {
    return [state.sourceKey, module, page, pageSize].join('|');
  }

  function rememberPreview(preview) {
    if (!state.sourceKey || !preview || !preview.module) return;
    const page = Number(preview.page || preview.pagination && preview.pagination.page || 1);
    const pageSize = Number(preview.page_size || preview.pagination && preview.pagination.page_size || 24);
    const key = cacheKey(preview.module, page, pageSize);
    if (previewCache.has(key)) previewCache.delete(key);
    previewCache.set(key, preview);
    while (previewCache.size > MAX_CACHE_ENTRIES) {
      previewCache.delete(previewCache.keys().next().value);
    }
  }

  function applyPreview(drill, preview, fallbackPage, fallbackPageSize) {
    if (!preview || !drill || !drill.data_tables) return false;
    drill.data_tables.table_previews = [preview];
    if (drill.data_tables.module_picker) {
      drill.data_tables.module_picker.default_module = preview.module;
    }
    state.module = preview.module;
    state.page = Number(preview.page || preview.pagination && preview.pagination.page || fallbackPage || 1);
    state.pageSize = Number(preview.page_size || preview.pagination && preview.pagination.page_size || fallbackPageSize || 24);
    return true;
  }

  function prime(drill) {
    const rawSourceKey = String(drill && drill.source && drill.source.path_hash || '');
    const sourceKey = drill && drill.demo
      ? `demo:${String(drill.source && (drill.source.database || drill.source.source_id) || rawSourceKey || 'seeded')}`
      : rawSourceKey;
    if (state.sourceKey && sourceKey && state.sourceKey !== sourceKey) reset();
    state.sourceKey = sourceKey;
    const tables = drill && drill.data_tables || {};
    const availableModules = (tables.modules || []).map(row => row && row.module).filter(Boolean);
    if (!state.module || !availableModules.includes(state.module)) {
      state.module = tables.module_picker && tables.module_picker.default_module || availableModules[0] || null;
    }
    const preview = previews(drill).find(row => row && row.module === state.module) || previews(drill)[0];
    if (preview) {
      state.module = preview.module || state.module;
      state.page = Number(preview.page || preview.pagination && preview.pagination.page || 1);
      state.pageSize = Number(preview.page_size || preview.pagination && preview.pagination.page_size || 24);
      rememberPreview(preview);
    }
  }

  function snapshot(drill) {
    prime(drill);
    return Object.assign({}, state);
  }

  function activePreview(drill) {
    prime(drill);
    return previews(drill).find(row => row && row.module === state.module) || previews(drill)[0] || null;
  }

  function statusHtml(helpers) {
    const h = Object.assign({ t: en => en, esc: value => String(value == null ? '' : value), icon: () => '' }, helpers || {});
    if (state.loading) {
      return `<div class="note info mt-10" data-patient-table-loading role="status" aria-live="polite"><div class="ico"><span class="spin"></span></div><div class="body"><div class="t">${h.t('Loading one bounded module page…', '正在加载一个有界模块页…')}</div><div class="d">${h.t('The cohort review and other modules stay in place.', '队列审阅和其他模块保持不变。')}</div></div></div>`;
    }
    if (state.error) {
      return `<div class="note warn mt-10" data-patient-table-error role="alert"><div class="ico">${h.icon('alert', 14)}</div><div class="body"><div class="t">${h.t('Module page was not replaced', '模块页未被替换')}</div><div class="d">${h.esc(state.error)}</div></div></div>`;
    }
    return '';
  }

  function moduleStatus(module, activePreview, helpers) {
    const h = Object.assign({ t: en => en }, helpers || {});
    if (activePreview && module && activePreview.module === module.module) {
      return h.t('loaded', '已加载');
    }
    if (module && module.review_status === 'inventory_only') {
      return h.t('available · load', '可用 · 按需加载');
    }
    return h.t('reviewed', '已审阅');
  }

  function sourceBody(config) {
    const path = config && typeof config.sourcePath === 'function' ? config.sourcePath() : '';
    return path ? { source_path: path } : {};
  }

  function repaint(config) {
    if (config && typeof config.repaint === 'function') config.repaint();
  }

  function restoreFocus(target) {
    if (!target || typeof document === 'undefined') return;
    let element = null;
    if (target.kind === 'module') {
      element = Array.from(document.querySelectorAll('[data-pt-table-module]'))
        .find(button => button.dataset.ptTableModule === target.module) || null;
    } else if (target.kind === 'page') {
      element = document.querySelector(`[data-pt-page-${target.action}]`);
    } else if (target.kind === 'page-size') {
      element = document.querySelector('[data-pt-page-size]');
    }
    if (!element || element.disabled) {
      element = document.querySelector('[data-pt-table-module][aria-pressed="true"]');
    }
    if (element && typeof element.focus === 'function') {
      element.focus({ preventScroll: true });
    }
  }

  function load(config, next) {
    const drill = config && config.drill && config.drill();
    if (!drill || drill.demo) {
      if (next.module) state.module = next.module;
      state.page = next.page || 1;
      state.pageSize = next.pageSize || state.pageSize;
      repaint(config);
      restoreFocus(next.focus);
      return;
    }
    prime(drill);
    const module = next.module || state.module;
    if (!module) return;
    const page = Math.max(1, Number(next.page || state.page || 1));
    const pageSize = [24, 50, 100].includes(Number(next.pageSize)) ? Number(next.pageSize) : state.pageSize;
    const cached = previewCache.get(cacheKey(module, page, pageSize));
    if (cached) {
      applyPreview(drill, cached, page, pageSize);
      state.loading = false;
      state.error = '';
      repaint(config);
      restoreFocus(next.focus);
      return;
    }
    const api = window.EU_API;
    if (!api || typeof api.loadPatientReviewTablePreview !== 'function') {
      state.error = 'Patient module table API is unavailable.';
      repaint(config);
      restoreFocus(next.focus);
      return;
    }
    const seq = ++state.requestSeq;
    state.loading = true;
    state.error = '';
    repaint(config);
    api.loadPatientReviewTablePreview(Object.assign(sourceBody(config), {
      table_module: module,
      table_page: page,
      table_page_size: pageSize,
    })).then(payload => {
      if (seq !== state.requestSeq) return;
      const preview = payload && payload.module_preview;
      if (applyPreview(drill, preview, page, pageSize)) {
        rememberPreview(preview);
      }
      state.loading = false;
      repaint(config);
      restoreFocus(next.focus);
    }).catch(error => {
      if (seq !== state.requestSeq) return;
      state.loading = false;
      state.error = String(error && error.message || error);
      repaint(config);
      restoreFocus(next.focus);
    });
  }

  function bind(root, config) {
    if (!root) return;
    root.querySelectorAll('[data-pt-table-module]').forEach(button => button.addEventListener('click', event => {
      event.preventDefault();
      const module = button.dataset.ptTableModule;
      if (state.loading || !module || (module === state.module && !state.error)) return;
      load(config, {
        module,
        page: 1,
        pageSize: state.pageSize,
        focus: { kind: 'module', module },
      });
    }));
    const previous = root.querySelector('[data-pt-page-prev]');
    if (previous) previous.addEventListener('click', event => {
      event.preventDefault();
      if (!state.loading && !previous.disabled) load(config, {
        page: Math.max(1, state.page - 1),
        focus: { kind: 'page', action: 'prev' },
      });
    });
    const next = root.querySelector('[data-pt-page-next]');
    if (next) next.addEventListener('click', event => {
      event.preventDefault();
      if (!state.loading && !next.disabled) load(config, {
        page: state.page + 1,
        focus: { kind: 'page', action: 'next' },
      });
    });
    const pageSize = root.querySelector('[data-pt-page-size]');
    if (pageSize) pageSize.addEventListener('change', () => {
      if (state.loading) return;
      const parsed = Number(pageSize.value);
      load(config, {
        page: 1,
        pageSize: [24, 50, 100].includes(parsed) ? parsed : 24,
        focus: { kind: 'page-size' },
      });
    });
  }

  host.tables = {
    activePreview,
    bind,
    load,
    moduleStatus,
    prime,
    reset,
    snapshot,
    statusHtml,
  };
})();
