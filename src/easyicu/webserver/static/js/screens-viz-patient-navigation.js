/* Patient Review entity-navigation owner: bounded pages and verified detail loads. */
(function () {
  'use strict';

  const host = (window.EU_PATIENT_REVIEW = window.EU_PATIENT_REVIEW || {});
  const state = {
    sourceKey: '',
    page: null,
    loading: false,
    error: '',
    requestSeq: 0,
  };

  function reset() {
    state.sourceKey = '';
    state.page = null;
    state.loading = false;
    state.error = '';
    state.requestSeq += 1;
  }

  function prime(drill) {
    if (drill && drill.demo) {
      if (state.sourceKey || state.page) reset();
      return;
    }
    const sourceKey = String(drill && drill.source && drill.source.path_hash || '');
    if (state.sourceKey && sourceKey && state.sourceKey !== sourceKey) reset();
    state.sourceKey = sourceKey;
    if (drill && drill.entity_navigation) state.page = drill.entity_navigation;
  }

  function helpersOf(value) {
    return Object.assign({
      t: en => en,
      esc: value => String(value == null ? '' : value),
      fmtInt: value => Number(value || 0).toLocaleString(),
      icon: () => '',
    }, value || {});
  }

  function render({ drill, selected, opts, helpers }) {
    prime(drill);
    const h = helpersOf(helpers);
    const fallback = Array.isArray(drill && drill.entities) ? drill.entities : [];
    const page = state.page || {};
    const options = Array.isArray(page.options) && page.options.length
      ? page.options
      : fallback;
    if (!options.length) return '';
    const selectedRef = selected && selected.ref;
    const selectedOrdinal = Number(selected && selected.ordinal || page.selected_ordinal || 0);
    const total = Number(page.total_entities || drill && drill.summary && drill.summary.entities || options.length);
    const title = opts && opts.title || h.t('Case navigator', '病例导航');
    const detail = opts && opts.detail || h.t(
      'Switch one pseudonymous entity without rebuilding cohort quality or table summaries.',
      '切换一个伪匿名实体时，不会重新计算队列质量或表格摘要。',
    );
    const pageNumber = Number(page.page || 1);
    const pageCount = Number(page.page_count || 1);
    const hasPaging = !drill.demo && pageCount > 1;
    const status = state.loading
      ? h.t('Loading entity…', '正在加载实体…')
      : (state.error || '');
    return `
      <div class="pt-entity-nav mt-16" data-patient-entity-navigator>
        <div>
          <div class="eyebrow">${h.esc(title)}</div>
          <div class="pt-entity-current">${h.esc(h.t('Selected', '当前'))}: ${h.esc(selected && (selected.label || selected.ref) || '—')}${selectedOrdinal && total ? ` · ${h.fmtInt(selectedOrdinal)} / ${h.fmtInt(total)}` : ''}</div>
          <div class="pt-entity-detail">${h.esc(detail)}</div>
          ${status ? `<div class="pt-entity-status ${state.error ? 'bad' : ''}" role="status" aria-live="polite">${h.esc(status)}</div>` : ''}
        </div>
        <div class="pt-entity-browser">
          ${hasPaging ? `<div class="pt-entity-pager" role="group" aria-label="${h.esc(h.t('Entity page controls', '实体分页控件'))}">
            <button type="button" class="btn sm" data-patient-entity-page-prev ${page.has_previous && !state.loading ? '' : 'disabled'}>${h.icon('arrow-left', 12)} ${h.t('Previous group', '上一组')}</button>
            <span class="mono">${h.fmtInt(page.row_start)}-${h.fmtInt(page.row_end)} / ${h.fmtInt(total)} · ${h.t('page', '第')} ${h.fmtInt(pageNumber)} / ${h.fmtInt(pageCount)}</span>
            <button type="button" class="btn sm" data-patient-entity-page-next ${page.has_next && !state.loading ? '' : 'disabled'}>${h.t('Next group', '下一组')} ${h.icon('arrow', 12)}</button>
            <button type="button" class="btn sm" data-patient-entity-page-random ${state.loading ? 'disabled' : ''}>${h.icon('refresh', 12)} ${h.t('Random group', '随机一组')}</button>
          </div>` : ''}
          <div class="pt-entity-chiprow" role="group" aria-label="${h.esc(h.t('Entities on this page', '本页实体'))}">
            ${options.map(item => `<button type="button" class="chip ${item.ref === selectedRef ? 'solid' : ''}" data-patient-entity="${h.esc(item.ref)}" data-patient-ordinal="${h.esc(item.ordinal || '')}" aria-pressed="${item.ref === selectedRef ? 'true' : 'false'}" ${state.loading ? 'disabled' : ''}>${h.esc(item.label || item.ref)}</button>`).join('')}
          </div>
        </div>
      </div>`;
  }

  function sourceBody(config) {
    const path = config && typeof config.sourcePath === 'function'
      ? config.sourcePath()
      : '';
    return path ? { source_path: path } : {};
  }

  function repaint(config) {
    if (config && typeof config.repaint === 'function') config.repaint();
  }

  function restoreFocus(target, selectedRef) {
    if (typeof document === 'undefined') return;
    let element = null;
    if (target && target.kind === 'entity') {
      element = Array.from(document.querySelectorAll('[data-patient-entity]'))
        .find(button => button.dataset.patientEntity === target.ref) || null;
    } else if (target && target.kind === 'page') {
      element = document.querySelector(`[data-patient-entity-page-${target.action}]`);
    }
    if (!element || element.disabled) {
      element = Array.from(document.querySelectorAll('[data-patient-entity]'))
        .find(button => button.dataset.patientEntity === selectedRef && !button.disabled)
        || document.querySelector('[data-patient-entity]:not([disabled])');
    }
    if (element && typeof element.focus === 'function') {
      element.focus({ preventScroll: true });
    }
  }

  function loadPage(config, patch) {
    const drill = config && config.drill && config.drill();
    const page = state.page || drill && drill.entity_navigation || {};
    const body = Object.assign(sourceBody(config), {
      entity_page: patch && patch.page || page.page || 1,
      entity_page_size: page.page_size || 12,
      selected_ref: drill && drill.selected && drill.selected.ref,
      selected_ordinal: drill && drill.selected && drill.selected.ordinal,
    });
    if (patch && patch.random) body.random_page = true;
    const api = window.EU_API;
    if (!api || typeof api.loadPatientReviewEntities !== 'function') {
      state.error = 'Patient entity navigation API is unavailable.';
      repaint(config);
      restoreFocus(
        patch && patch.focus,
        drill && drill.selected && drill.selected.ref,
      );
      return;
    }
    const seq = ++state.requestSeq;
    state.loading = true;
    state.error = '';
    repaint(config);
    api.loadPatientReviewEntities(body).then(payload => {
      if (seq !== state.requestSeq) return;
      state.page = payload && payload.navigation || null;
      if (drill && state.page) drill.entity_navigation = state.page;
      state.loading = false;
      repaint(config);
      restoreFocus(
        patch && patch.focus,
        drill && drill.selected && drill.selected.ref,
      );
    }).catch(error => {
      if (seq !== state.requestSeq) return;
      state.loading = false;
      state.error = String(error && error.message || error);
      repaint(config);
      restoreFocus(
        patch && patch.focus,
        drill && drill.selected && drill.selected.ref,
      );
    });
  }

  function loadEntity(config, ref, ordinal) {
    const drill = config && config.drill && config.drill();
    const api = window.EU_API;
    if (!api || typeof api.loadPatientReviewEntity !== 'function') {
      state.error = 'Patient entity detail API is unavailable.';
      repaint(config);
      restoreFocus({ kind: 'entity', ref }, drill && drill.selected && drill.selected.ref);
      return;
    }
    const seq = ++state.requestSeq;
    state.loading = true;
    state.error = '';
    repaint(config);
    api.loadPatientReviewEntity(Object.assign(sourceBody(config), {
      entity_ref: ref,
      entity_ordinal: Number(ordinal),
    })).then(payload => {
      if (seq !== state.requestSeq) return;
      if (drill && payload) {
        ['selected', 'entities', 'time_lanes', 'trajectory_review', 'patient_overview'].forEach(key => {
          if (payload[key] != null) drill[key] = payload[key];
        });
        if (state.page) {
          state.page.selected_ref = payload.selected && payload.selected.ref;
          state.page.selected_ordinal = payload.selected && payload.selected.ordinal;
          (state.page.options || []).forEach(item => {
            item.selected = item.ref === state.page.selected_ref;
          });
        }
      }
      state.loading = false;
      repaint(config);
      restoreFocus({ kind: 'entity', ref }, ref);
    }).catch(error => {
      if (seq !== state.requestSeq) return;
      state.loading = false;
      state.error = String(error && error.message || error);
      repaint(config);
      restoreFocus({ kind: 'entity', ref }, drill && drill.selected && drill.selected.ref);
    });
  }

  function bind(root, config) {
    if (!root) return;
    root.querySelectorAll('[data-patient-entity]').forEach(button => button.addEventListener('click', event => {
      event.preventDefault();
      const ref = button.dataset.patientEntity;
      if (!ref || state.loading) return;
      const drill = config && config.drill && config.drill();
      if (drill && drill.demo && typeof config.selectDemo === 'function') {
        config.selectDemo(ref);
        return;
      }
      loadEntity(config, ref, button.dataset.patientOrdinal);
    }));
    const previous = root.querySelector('[data-patient-entity-page-prev]');
    if (previous) previous.addEventListener('click', event => {
      event.preventDefault();
      if (!previous.disabled) loadPage(config, {
        page: Math.max(1, Number((state.page || {}).page || 1) - 1),
        focus: { kind: 'page', action: 'prev' },
      });
    });
    const next = root.querySelector('[data-patient-entity-page-next]');
    if (next) next.addEventListener('click', event => {
      event.preventDefault();
      if (!next.disabled) loadPage(config, {
        page: Number((state.page || {}).page || 1) + 1,
        focus: { kind: 'page', action: 'next' },
      });
    });
    const random = root.querySelector('[data-patient-entity-page-random]');
    if (random) random.addEventListener('click', event => {
      event.preventDefault();
      if (!random.disabled) loadPage(config, {
        random: true,
        focus: { kind: 'page', action: 'random' },
      });
    });
  }

  host.navigation = { bind, loadEntity, loadPage, prime, render, reset };
})();
