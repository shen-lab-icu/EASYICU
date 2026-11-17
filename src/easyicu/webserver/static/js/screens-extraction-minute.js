/* Owner: explicit minute-resolution choices and results in Data Extraction. */
(function () {
  function create({ t, escHtml, hosted, repaint, backgroundRepaint }) {
    let state = null;

    function render({ path, database, ready }) {
      if (!state || state.path !== path || state.database !== database || state.ready !== ready) {
        state = { path, database, ready, group: null, failed: false, pending: false, selected: new Set() };
      }
      if (path && database && ready && !state.group && !state.failed && !state.pending && window.EU_API?.getJSON) {
        load(state);
      }
    }

    function reset(keepSource) {
      if (state?.ready && keepSource && state.path === keepSource.path && state.database === keepSource.database) return;
      state = null;
    }

    function load(current) {
      current.failed = false;
      current.pending = true;
      window.EU_API.getJSON('/api/extraction/module-options?database=' + encodeURIComponent(current.database))
        .catch(() => null)
        .then(payload => {
          // A new connection to the same folder has its own state object.
          if (current !== state) return;
          const options = payload?.minute_resolution;
          current.group = Array.isArray(options?.modules) ? options : null;
          current.failed = !current.group;
          current.pending = false;
          backgroundRepaint();
        });
    }

    function choices(wanted) {
      const { group, selected } = state || {};
      if (hosted() || !group || group.unavailable_reason) return [];
      return group.modules.filter(row => selected.has(row.id) && (!wanted || wanted.has(row.id)));
    }

    function markup() {
      const { database, group, failed, pending, selected } = state;
      const hostedReason = hosted() || group?.unavailable_reason === 'subhour_unavailable_when_hosted';
      let note = '';
      if (hostedReason) {
        note = t('Patient-level minute exports are unavailable on hosted instances.', '托管实例不提供患者级分钟导出。');
      } else if (failed) {
        note = t('Could not load coverage.', '无法加载覆盖范围。') +
          ` <button type="button" class="linkbtn" data-ex-minute-retry>${t('Retry', '重试')}</button>`;
      } else if (pending) {
        note = t('Checking database coverage…', '正在查询数据库覆盖范围…');
      } else if (!group) {
        note = t('Database coverage is unavailable.', '数据库覆盖范围暂不可用。');
      } else if (selected.size) {
        note = t('Minute exports can be larger and take longer.', '分钟级导出可能占用更多空间、耗时更长。');
      }
      const source = window.EU_DATABASE_NAMES?.name(database) || database;
      return `
        <section class="ex-minute-group" aria-labelledby="exMinuteHeading">
          <h3 id="exMinuteHeading">${t('Minute resolution', '分钟级分辨率')}</h3>
          ${note ? `<p class="ex-minute-note">${note}</p>` : ''}
          <div class="modgrid">
            ${(group?.modules || []).map(row => {
              const label = escHtml(t(...row.label));
              if (hostedReason) return `<div class="ex-minute-choice">${label}</div>`;
              if (row.unavailable_reason) return `<div class="ex-minute-choice">${label}
                <span class="ex-minute-note">${t('No covered signals', '无覆盖信号')}</span></div>`;
              const on = selected.has(row.id);
              return `<div>
                <label class="ex-minute-choice"><input type="checkbox" data-ex-minute-module="${escHtml(row.id)}" ${on ? 'checked' : ''}> ${label}</label>
                ${on ? `<p class="ex-minute-note"><b>${escHtml(source)}</b> · ${escHtml(row.signals.join(', '))}</p>` : ''}
              </div>`;
            }).join('')}
          </div>
        </section>`;
    }

    function bind(root, markStale) {
      const current = state;
      root.querySelector('[data-ex-minute-retry]')?.addEventListener('click', () => {
        if (current !== state || !current?.failed || !window.EU_API?.getJSON) return;
        load(current);
        repaint();
      });
      root.querySelectorAll('[data-ex-minute-module]').forEach(input => {
        input.addEventListener('change', () => {
          if (current !== state || !current?.group) return;
          const id = input.dataset.exMinuteModule;
          if (input.checked) current.selected.add(id);
          else current.selected.delete(id);
          markStale();
          repaint();
          const inputs = root.querySelectorAll('[data-ex-minute-module]');
          Array.from(inputs).find(row => row.dataset.exMinuteModule === id)?.focus();
        });
      });
    }

    function resultDetail(file) {
      const resolution = file.time_resolution;
      if (!resolution) return '';
      const signals = file.concept_ids || Object.keys(resolution.signals || {});
      const grid = resolution.interval_minutes;
      const gridLabel = t(`${grid}-minute grid`, `${grid} 分钟网格`);
      return escHtml([gridLabel, signals.join(', ')].filter(Boolean).join(' · '));
    }

    return {
      render, markup, bind, resultDetail, reset,
      clear: () => state?.selected.clear(),
      keys: () => choices().map(row => row.id),
      count: () => choices().reduce((count, row) => count + row.signals.length, 0),
      payload: wanted => Object.fromEntries(choices(wanted).map(row => [row.id, row.signals])),
    };
  }

  window.EU_EXTRACTION_MINUTE = { create };
})();
