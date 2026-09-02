/* Cross-DB source choice owner: official demo pair, registered exports, or raw ICU roots. */
(function () {
  'use strict';

  const { esc } = window.EU_HTML;

  function text(en, zh) {
    return window.EU_LANG === 'zh' ? zh : en;
  }

  function registeredPaths() {
    const host = window.EU_CROSSDB_SOURCE_HOST;
    if (!host || typeof host.registeredPaths !== 'function') return [];
    const paths = host.registeredPaths();
    return Array.from(new Set((Array.isArray(paths) ? paths : []).map(path => String(path || '').trim()).filter(Boolean)));
  }

  function render(options) {
    const registryHtml = String(options && options.registryHtml || '');
    const count = registeredPaths().length;
    const ready = count >= 2;
    const sourceLabel = count === 1
      ? text('1 selected export', '已选择 1 个导出')
      : text(`${count} selected exports`, `已选择 ${count} 个导出`);
    return `
      <section data-crossdb-source-choice>
        ${registryHtml}
        <div class="gate-strip mt-14">
          <span class="pill ${ready ? 'ok' : 'warn'}">${sourceLabel}</span>
          <span class="panel-sub">${ready
            ? text('Ready for an aggregate-only consistency check.', '已可运行仅聚合的一致性检查。')
            : text('Add and select at least two EasyICU exports.', '请添加并选择至少两个 EasyICU 导出。')}</span>
          <div class="grow"></div>
          <button class="btn primary" type="button" data-crossdb-run-registered ${ready ? '' : 'aria-disabled="true"'}>${text('Start consistency check', '开始一致性检查')}</button>
        </div>
      </section>`;
  }

  function officialPaths() {
    const host = window.EU_CROSSDB_SOURCE_HOST;
    if (!host || typeof host.officialPaths !== 'function') return [];
    return Array.from(new Set(host.officialPaths().map(path => String(path || '').trim()).filter(Boolean)));
  }

  function officialSourceStatus(source, pairReady) {
    const status = source && source.status || {};
    const active = pairReady || Boolean(status.active);
    const labels = {
      not_downloaded: text('Not installed', '尚未安装'),
      downloaded: text('Downloaded', '已下载'),
      converted: text('Converted', '已转换'),
      prepared: text('Ready', '已就绪'),
    };
    return {
      active,
      label: active ? text('Ready', '已就绪') : (labels[status.state] || text('Not installed', '尚未安装')),
      tone: active ? 'ok' : (status.state === 'not_downloaded' ? 'neutral' : 'warn'),
    };
  }

  function officialScope(source) {
    const scope = source && source.scope || {};
    if (scope.patients) return text(`${scope.patients} patients`, `${scope.patients} 名患者`);
    if (scope.icu_stays) return text(`${scope.icu_stays} ICU stays`, `${scope.icu_stays} 次 ICU 住院`);
    return text('bounded public cohort', '有界公开队列');
  }

  function compactOfficialPair(sourceOwner, pairReady) {
    const snapshot = sourceOwner && typeof sourceOwner.snapshot === 'function' ? sourceOwner.snapshot() : null;
    const sources = snapshot && snapshot.catalog && Array.isArray(snapshot.catalog.sources)
      ? snapshot.catalog.sources
      : [];
    if (!sources.length || snapshot.error || snapshot.job) {
      return sourceOwner && typeof sourceOwner.render === 'function'
        ? sourceOwner.render({ t: text, esc }, { scope: 'crossdb', showFallback: false })
        : `<div class="note warn"><div class="body"><div class="d">${text('Official demo-source controls are unavailable.', '官方演示数据源控件暂不可用。')}</div></div></div>`;
    }
    return `<div class="crossdb-demo-pair" data-official-demo-sources data-demo-source-scope="crossdb">
      ${sources.slice(0, 2).map(source => {
        const status = officialSourceStatus(source, pairReady);
        const provenance = source.provenance || {};
        const license = provenance.license || {};
        const download = source.download || {};
        return `<article class="crossdb-demo-source ${status.active ? 'ready' : ''}">
          <div class="crossdb-demo-source-main">
            <span class="crossdb-demo-source-icon">DB</span>
            <div>
              <small>${esc(provenance.provider || 'PhysioNet')} · ${esc(source.version || '')}</small>
              <b>${esc(source.title || source.id)}</b>
              <span>${esc(officialScope(source))} · ${esc(download.size_label || '')} · ${esc(source.database || '')}</span>
            </div>
          </div>
          <div class="crossdb-demo-source-action">
            <span class="pill ${status.tone}">${esc(status.label)}</span>
            ${status.active ? '' : `<button class="btn sm" type="button" data-demo-source-prepare="${esc(source.id)}">${text('Prepare', '准备数据')}</button>`}
          </div>
          <details>
            <summary>${text('License & provenance', '许可与溯源')}</summary>
            <p>${esc(license.name || 'ODbL 1.0')} · ${text('deidentified real records', '去标识化真实记录')}</p>
          </details>
        </article>`;
      }).join('')}
    </div>`;
  }

  function renderDemo(options) {
    const sourceOwner = window.EU_OFFICIAL_DEMO_SOURCES || window.EU_PATIENT_DEMO_SOURCES;
    const syntheticHtml = String(options && options.syntheticHtml || '');
    const count = officialPaths().length;
    const ready = count >= 2;
    const officialHtml = compactOfficialPair(sourceOwner, ready);
    return `
      <section data-crossdb-demo-source-choice>
        <div class="card pad">
          <div class="panel-head">
            <div>
              <div class="eyebrow">${text('Official demo pair', '官方 Demo 组合')}</div>
              <div class="panel-title" style="font-size:17px;">${text('MIMIC-IV Demo + eICU Demo', 'MIMIC-IV Demo + eICU Demo')}</div>
              <div class="panel-sub mt-4">${text('Two real deidentified public demo exports, processed through the normal EasyICU mapping pipeline.', '两个真实去标识化公开 Demo 导出，均经过正常 EasyICU 映射流程。')}</div>
            </div>
            <span class="pill ${ready ? 'ok' : 'warn'}"><span class="dot"></span>${count} / 2 ${text('ready', '已就绪')}</span>
          </div>
          <div class="mt-16">
            ${officialHtml}
          </div>
          <div class="gate-strip mt-16">
            <span class="panel-sub">${ready
              ? text('Both deidentified exports are registered locally.', '两个去标识化导出均已在本地注册。')
              : text('Prepare both official demos before running.', '运行前请先准备好两个官方 Demo。')}</span>
            <div class="grow"></div>
            <button class="btn primary" type="button" data-crossdb-run-official ${ready ? '' : 'aria-disabled="true"'}>${text('Start consistency check', '开始一致性检查')}</button>
          </div>
        </div>
        <details class="card crossdb-offline-fallback mt-16" data-crossdb-synthetic-fallback>
          <summary class="crossdb-source-summary">
            <div>
              <div class="eyebrow">${text('Offline fallback', '离线兜底')}</div>
              <div class="panel-title mt-4">${text('Seeded synthetic multi-database preview', '种子合成多数据库预览')}</div>
              <div class="panel-sub mt-4">${text('UI rehearsal only. These six simulated frames are not official database records or scientific results.', '仅用于界面演练。这六组模拟特征帧不是官方数据库记录，也不是科学结果。')}</div>
            </div>
            <span class="pill warn">${text('Synthetic', '合成')}</span>
          </summary>
          <div class="crossdb-source-detail">${syntheticHtml}</div>
        </details>
      </section>`;
  }

  function renderLoading() {
    const context = window.EU_DATA_MODE_CONTEXT;
    const officialPair = context && context.kind === 'official_demo_pair';
    const count = officialPair ? officialPaths().length : registeredPaths().length;
    const title = officialPair
      ? text('Loading official demo summaries…', '正在加载官方 Demo 摘要…')
      : text('Loading registered export summaries…', '正在加载已注册导出的摘要…');
    const meta = officialPair
      ? text(`${count} official demos · aggregate-only`, `${count} 个官方 Demo · 仅聚合`)
      : text(`${count} local exports · aggregate-only`, `${count} 个本地导出 · 仅聚合`);
    return `<div class="card pad" data-crossdb-registered-loading role="status" aria-live="polite">
      <div class="load-strip">
        <span class="spin accent"></span>
        <div class="grow">
          <div style="font-weight:600;font-size:12.75px;">${title}</div>
          <div class="mono" style="font-size:11px;color:var(--ink-4);margin-top:2px;">${meta}</div>
        </div>
        <button class="btn sm" type="button" data-crossdb-cancel>${text('Cancel', '取消')}</button>
      </div>
      <div class="indet mt-12"></div>
      <div class="panel-sub mt-8">${text('EasyICU is reading bounded summaries from the selected exports. Raw ICU folders are not scanned.', 'EasyICU 正在读取所选导出的有界摘要；不会扫描原始 ICU 文件夹。')}</div>
    </div>`;
  }

  function wire(root) {
    if (!root || typeof root.querySelectorAll !== 'function') return;
    root.querySelectorAll('[data-crossdb-run-registered]').forEach(button => {
      if (button.dataset.crossdbRegisteredBound === '1') return;
      button.dataset.crossdbRegisteredBound = '1';
      button.addEventListener('click', event => {
        event.preventDefault();
        event.stopPropagation();
        if (button.getAttribute('aria-disabled') === 'true' || registeredPaths().length < 2) return;
        const host = window.EU_CROSSDB_SOURCE_HOST;
        if (host && typeof host.runRegistered === 'function') host.runRegistered();
      });
    });
    const sourceOwner = window.EU_OFFICIAL_DEMO_SOURCES || window.EU_PATIENT_DEMO_SOURCES;
    const host = window.EU_CROSSDB_SOURCE_HOST;
    const demoChoice = sourceOwner && typeof root.querySelector === 'function'
      ? root.querySelector('[data-crossdb-demo-source-choice]')
      : null;
    if (sourceOwner && demoChoice) {
      sourceOwner.ensureLoaded(() => {
        if (host && typeof host.repaint === 'function') host.repaint();
      });
      sourceOwner.bind(root, {
        refresh: () => {
          if (host && typeof host.repaint === 'function') host.repaint();
        },
        openPrepared: sourceId => {
          if (host && typeof host.openOfficial === 'function') host.openOfficial(sourceId);
        },
      });
      root.querySelectorAll('[data-crossdb-run-official]').forEach(button => {
        if (button.dataset.crossdbOfficialBound === '1') return;
        button.dataset.crossdbOfficialBound = '1';
        button.addEventListener('click', event => {
          event.preventDefault();
          event.stopPropagation();
          if (button.getAttribute('aria-disabled') === 'true' || officialPaths().length < 2) return;
          if (host && typeof host.runOfficial === 'function') host.runOfficial();
        });
      });
    }
  }

  window.EU_CROSSDB_SOURCE_CHOICE = { render, renderDemo, renderLoading, wire };
})();
