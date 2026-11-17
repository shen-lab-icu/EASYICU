/* Owner: explicit minute-resolution choices and results in Data Extraction. */
(function () {
  function create({ t, escHtml, hosted, repaint, backgroundRepaint }) {
    const selected = new Set();
    let database = '';
    let group = null;
    let failed = false;
    let request = 0;

    function load(key) {
      database = key;
      group = null;
      failed = false;
      selected.clear();
      const current = ++request;
      window.EU_API.getJSON('/api/extraction/module-options?database=' + encodeURIComponent(key))
        .then(payload => {
          if (current !== request) return;
          group = payload.minute_resolution;
          backgroundRepaint();
        })
        .catch(() => {
          if (current !== request) return;
          failed = true;
          backgroundRepaint();
        });
    }

    function choices(wanted) {
      if (hosted() || !group || group.unavailable_reason) return [];
      return group.modules.filter(row => selected.has(row.id) && (!wanted || wanted.has(row.id)));
    }

    function render(key) {
      if (key && key !== database && window.EU_API?.getJSON) load(key);
      const hostedReason = hosted() || group?.unavailable_reason === 'subhour_unavailable_when_hosted';
      let note = '';
      if (hostedReason) {
        note = t('Patient-level minute exports are unavailable on hosted instances.', '托管实例不提供患者级分钟导出。');
      } else if (failed) {
        note = t('Could not load coverage.', '无法加载覆盖范围。') +
          ` <button type="button" class="linkbtn" data-ex-minute-retry>${t('Retry', '重试')}</button>`;
      } else if (!key || !window.EU_API?.getJSON) {
        note = t('Database coverage is unavailable.', '数据库覆盖范围暂不可用。');
      } else if (!group) {
        note = t('Checking database coverage…', '正在查询数据库覆盖范围…');
      } else if (selected.size) {
        note = t('Minute exports can be larger and take longer.', '分钟级导出可能占用更多空间、耗时更长。');
      }
      const source = window.EU_DATABASE_NAMES?.name(key) || key;
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
      root.querySelector('[data-ex-minute-retry]')?.addEventListener('click', () => { load(database); repaint(); });
      root.querySelectorAll('[data-ex-minute-module]').forEach(input => {
        input.addEventListener('change', () => {
          const id = input.dataset.exMinuteModule;
          if (input.checked) selected.add(id);
          else selected.delete(id);
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
      render, bind, resultDetail,
      clear: () => selected.clear(),
      keys: () => choices().map(row => row.id),
      count: () => choices().reduce((count, row) => count + row.signals.length, 0),
      payload: wanted => Object.fromEntries(choices(wanted).map(row => [row.id, row.signals])),
    };
  }

  window.EU_EXTRACTION_MINUTE = { create };
})();
