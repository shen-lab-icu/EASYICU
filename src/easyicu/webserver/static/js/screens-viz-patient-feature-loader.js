/* Patient Review single-feature lazy-load owner. */
(function () {
  'use strict';

  const host = (window.EU_PATIENT_REVIEW = window.EU_PATIENT_REVIEW || {});
  const MAX_CACHE_ENTRIES = 320;
  const MAX_PARALLEL_LOADS = 4;
  const cache = new Map();
  const state = {
    sourceKey: '',
    loadingKeys: new Set(),
    errorByFeature: new Map(),
    requestSeq: 0,
  };

  function reset() {
    state.sourceKey = '';
    state.loadingKeys.clear();
    state.errorByFeature.clear();
    state.requestSeq += 1;
    cache.clear();
  }

  function prime(drill) {
    if (drill && drill.demo) {
      if (state.sourceKey || cache.size) reset();
      return;
    }
    const sourceKey = String(drill && drill.source && drill.source.path_hash || '');
    if (state.sourceKey && sourceKey && state.sourceKey !== sourceKey) reset();
    state.sourceKey = sourceKey;
  }

  function selectedRef(drill) {
    return String(drill && drill.selected && drill.selected.ref || '');
  }

  function cacheKey(drill, feature) {
    return [state.sourceKey, selectedRef(drill), String(feature || '')].join('|');
  }

  function remember(key, payload) {
    if (!key || !payload) return;
    if (cache.has(key)) cache.delete(key);
    cache.set(key, payload);
    while (cache.size > MAX_CACHE_ENTRIES) {
      cache.delete(cache.keys().next().value);
    }
  }

  function stateFor(feature, drill) {
    prime(drill);
    const key = cacheKey(drill, feature);
    const payload = cache.get(key);
    return {
      loading: state.loadingKeys.has(key),
      error: state.errorByFeature.get(key) || '',
      loaded: Boolean(payload),
      status: payload && payload.status,
      payload,
    };
  }

  function augmentLanes(lanes, drill) {
    prime(drill);
    if (!drill || drill.demo) return Array.isArray(lanes) ? lanes : [];
    const prefix = [state.sourceKey, selectedRef(drill), ''].join('|');
    const lazyByModule = new Map();
    cache.forEach((payload, key) => {
      if (!key.startsWith(prefix) || !payload || !payload.signal) return;
      const module = String(payload.feature && payload.feature.module || '');
      if (!module) return;
      const signal = Object.assign({}, payload.signal, {
        module,
        lazy_loaded: true,
      });
      const rows = lazyByModule.get(module) || [];
      rows.push(signal);
      lazyByModule.set(module, rows);
    });
    const byModule = new Map(
      (Array.isArray(lanes) ? lanes : []).map(lane => [String(lane && lane.lane || ''), lane]),
    );
    const moduleIds = new Set([...byModule.keys(), ...lazyByModule.keys()]);
    return Array.from(moduleIds).filter(Boolean).map(module => {
      const lane = byModule.get(module) || {
        lane: module,
        label: module,
        signals: [],
        signal_count: 0,
        status: 'unavailable',
      };
      const lazy = lazyByModule.get(module) || [];
      const lazyKeys = new Set(lazy.map(signal => String(signal.feature || signal.key || '')));
      const existing = (lane.signals || []).filter(signal => !lazyKeys.has(String(signal && (signal.feature || signal.key) || '')));
      const signals = lazy.concat(existing);
      return Object.assign({}, lane, {
        signals,
        signal_count: signals.length,
        status: signals.length ? 'ready' : lane.status,
      });
    });
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

  function restoreFocus(feature) {
    if (typeof document === 'undefined' || !feature) return;
    const element = Array.from(document.querySelectorAll('[data-patient-feature-load]'))
      .find(button => button.dataset.patientFeatureLoad === feature && !button.disabled);
    if (element && typeof element.focus === 'function') {
      element.focus({ preventScroll: true });
    }
  }

  function restoreModuleFocus(module) {
    if (typeof document === 'undefined' || !module) return;
    const element = Array.from(document.querySelectorAll('[data-patient-module-load]'))
      .find(button => button.dataset.patientModuleLoad === module && !button.disabled);
    if (element && typeof element.focus === 'function') {
      element.focus({ preventScroll: true });
    }
  }

  function requestFeature(config, drill, feature, seq) {
    const api = window.EU_API;
    const key = cacheKey(drill, feature);
    return api.loadPatientReviewFeature(Object.assign(sourceBody(config), {
      entity_ref: drill.selected && drill.selected.ref,
      entity_ordinal: Number(drill.selected && drill.selected.ordinal),
      feature,
    })).then(payload => {
      if (seq !== state.requestSeq) return;
      remember(key, payload);
    }).catch(error => {
      if (seq !== state.requestSeq) return;
      state.errorByFeature.set(key, String(error && error.message || error));
    }).finally(() => {
      if (seq === state.requestSeq) state.loadingKeys.delete(key);
    });
  }

  function loadMany(config, features, module = '') {
    const drill = config && config.drill && config.drill();
    if (!drill || drill.demo) return Promise.resolve(false);
    prime(drill);
    const requested = Array.from(new Set(
      (Array.isArray(features) ? features : [features])
        .map(feature => String(feature || '').trim())
        .filter(Boolean),
    ));
    const pending = requested.filter(feature => {
      const key = cacheKey(drill, feature);
      return !cache.has(key) && !state.loadingKeys.has(key);
    });
    if (!pending.length) return Promise.resolve(true);
    const api = window.EU_API;
    if (!api || typeof api.loadPatientReviewFeature !== 'function') {
      pending.forEach(feature => {
        state.errorByFeature.set(
          cacheKey(drill, feature),
          'Patient feature API is unavailable.',
        );
      });
      repaint(config);
      return Promise.resolve(false);
    }
    const seq = state.requestSeq;
    pending.forEach(feature => {
      const key = cacheKey(drill, feature);
      state.loadingKeys.add(key);
      state.errorByFeature.delete(key);
    });
    repaint(config);

    let nextIndex = 0;
    const worker = async () => {
      while (nextIndex < pending.length) {
        const feature = pending[nextIndex];
        nextIndex += 1;
        await requestFeature(config, drill, feature, seq);
      }
    };
    const workers = Array.from(
      { length: Math.min(MAX_PARALLEL_LOADS, pending.length) },
      () => worker(),
    );
    return Promise.all(workers).then(() => {
      if (seq !== state.requestSeq) return false;
      repaint(config);
      if (module) restoreModuleFocus(module);
      return true;
    });
  }

  function load(config, feature) {
    const promise = loadMany(config, [feature]);
    promise.finally(() => restoreFocus(feature));
    return promise;
  }

  function bind(root, config) {
    if (!root) return;
    root.querySelectorAll('[data-patient-feature-load]').forEach(button => {
      button.addEventListener('click', event => {
        event.preventDefault();
        const feature = button.dataset.patientFeatureLoad;
        const drill = config && config.drill && config.drill();
        const current = stateFor(feature, drill);
        if (!feature || current.loading || current.loaded) return;
        load(config, feature);
      });
    });
    root.querySelectorAll('[data-patient-module-load]').forEach(button => {
      button.addEventListener('click', event => {
        event.preventDefault();
        const module = button.dataset.patientModuleLoad;
        const card = button.closest('[data-patient-series-module]');
        const features = card
          ? Array.from(card.querySelectorAll('[data-patient-feature-load]'))
            .map(featureButton => featureButton.dataset.patientFeatureLoad)
            .filter(Boolean)
          : [];
        if (!features.length) return;
        loadMany(config, features, module);
      });
    });
    root.querySelectorAll('[data-patient-inventory-toggle]').forEach(button => {
      button.addEventListener('click', event => {
        event.preventDefault();
        const open = button.dataset.patientInventoryToggle === 'open';
        root.querySelectorAll('details.pt-feature-inventory')
          .forEach(details => { details.open = open; });
      });
    });
  }

  host.features = {
    augmentLanes,
    bind,
    load,
    loadMany,
    prime,
    reset,
    stateFor,
  };
})();
