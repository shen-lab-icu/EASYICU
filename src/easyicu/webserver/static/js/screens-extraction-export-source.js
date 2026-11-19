/* Owner: Data Extraction — an existing export is not source data.
   A module export's cohort and features are fixed, so it cannot be
   extracted again. When the folder scan names the source folder the export
   was made from (dataio.scan_path: source_data_path), the extraction page
   offers that folder for a new cohort; otherwise a run stops with a plain
   explanation before any job starts. */
(function () {
  'use strict';

  const t = (en, zh) => window.t(en, zh);
  const html = value => window.EU_HTML.esc(value);

  function sourceCta(facts, hosted) {
    if (!facts || facts.source !== 'module' || !facts.sourceDataPath || hosted) return '';
    return `<button class="btn primary" data-ex-source-data>${window.icon('db', 14)} ${html(t('Define a new cohort from its source data', '从它的源数据定义新队列'))}</button>`;
  }

  function refusal() {
    return {
      ok: false,
      reason: 'export_not_source',
      message: t(
        'This folder is an existing EasyICU export, not source data, so it cannot be extracted again. Go back and choose the database source folder (for an official demo: "Define a new cohort from its source data"), or review this export in Patient review or Cohort statistics.',
        '这个文件夹是已有的 EasyICU 导出，不是源数据，不能再次抽取。请返回并选择数据库的源数据文件夹（官方演示数据可点“从它的源数据定义新队列”），或在患者审阅、队列统计中查看这个导出。',
      ),
    };
  }

  function bind(root, sourcePath, open) {
    const button = root.querySelector('[data-ex-source-data]');
    if (!button) return;
    button.addEventListener('click', () => {
      const path = sourcePath();
      if (path) open(path);
    });
  }

  window.EU_EXTRACTION_EXPORT_SOURCE = { sourceCta, refusal, bind };
})();
