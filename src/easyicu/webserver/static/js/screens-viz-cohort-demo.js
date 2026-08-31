/* Seeded Cohort Statistics preview payloads and disclosure notes. */
(function () {
  let t;
  let icon;
  let esc;
  let fmtInt;
  let cohortText;
  let demoRowsForModule;

  function cohortDemoPanelNote(kind) {
    const detail = kind === 'sofa'
      ? t('This SOFA-2 reclassification panel uses fixed seeded values to preview the chart and table layout. It is not computed from a local export and must not be used for manuscript claims.', '这个 SOFA-2 重分层面板使用固定 seeded 数值来预览图表和表格布局；它不是从本地导出计算的，不能用于稿件结论。')
      : t('This coverage audit uses fixed EasyICU catalog-shaped demo values to preview the module coverage workflow. It is not computed from a local export and must not be used for manuscript claims.', '这个覆盖审计使用固定的 EasyICU catalog-shaped 演示值来预览模块覆盖工作流；它不是从本地导出计算的，不能用于稿件结论。');
    return `<div class="note warn mt-12" data-demo-cohort-panel="${esc(kind)}"><div class="ico">${icon('shield', 14)}</div><div class="body"><div class="t">${cohortText('Seeded demo only')}</div><div class="d">${detail}</div></div></div>`;
  }

  function cohortDemoCoverageReview() {
    const modules = demoCatalogModules();
    const coverage = modules.map((module, index) => {
      const featureCount = (module.features || []).length;
      const penalty = (index % 5) * 4 + Math.max(0, featureCount - 12) * 0.25;
      const coveragePct = Math.max(58, Math.min(100, 98 - penalty));
      const rows = demoRowsForModule(module.module, featureCount, coveragePct);
      const qualityStatus = coveragePct >= 85 ? 'ok' : (coveragePct >= 70 ? 'warn' : 'neutral');
      return {
        module: module.label,
        module_key: module.module,
        rows,
        column_count: featureCount,
        covered_entities: Math.max(1, Math.round(10 * coveragePct / 100)),
        coverage_pct: Number(coveragePct.toFixed(1)),
        quality_status: qualityStatus,
      };
    });
    const coverageValues = coverage.map(row => row.coverage_pct).sort((a, b) => a - b);
    const median = coverageValues.length ? coverageValues[Math.floor(coverageValues.length / 2)] : null;
    return {
      demo: true,
      coverage,
      quality: {
        modules_ok: coverage.filter(row => row.quality_status === 'ok').length,
        watchlist_count: coverage.filter(row => row.quality_status === 'warn').length,
        median_coverage_pct: median,
        modules_neutral: coverage.filter(row => row.quality_status === 'neutral').length,
        modules_unknown: 0,
      },
    };
  }

  function cohortDemoSofaExactMatrix(pairs) {
    const labels = Array.from({ length: 25 }, (_, score) => String(score));
    const total = pairs.length || 1;
    return labels.map(sourceLabel => {
      const sourceScore = Number(sourceLabel);
      const cells = labels.map(targetLabel => {
        const targetScore = Number(targetLabel);
        const count = pairs.filter(([sofa1, sofa2]) => sofa1 === sourceScore && sofa2 === targetScore).length;
        return { label: targetLabel, count, pct: Number((count / total * 100).toFixed(1)) };
      });
      return { label: sourceLabel, count: cells.reduce((acc, cell) => acc + cell.count, 0), cells };
    });
  }

  function cohortDemoSofaReview() {
    const demoPairs = [[2, 2], [4, 7], [8, 5], [7, 7], [6, 9], [10, 13], [12, 11], [3, 3], [5, 5], [9, 9]];
    return {
      demo: true,
      summary: {
        cohort_size: 10,
        sofa2: {
          count: 10,
          median: 7,
          mean: 7.4,
          min: 1,
          max: 16,
          bins: [
            { label: '0-5', count: 4, pct: 40.0 },
            { label: '6-9', count: 3, pct: 30.0 },
            { label: '10-13', count: 2, pct: 20.0 },
            { label: '14+', count: 1, pct: 10.0 },
          ],
        },
      },
      sofa_reclassification: {
        status: 'ready',
        paired_count: 10,
        coverage_pct: 100.0,
        direction_counts: {
          up: { count: 3, pct: 30.0 },
          down: { count: 2, pct: 20.0 },
          same: { count: 5, pct: 50.0 },
        },
        delta_summary: { median: 1.0 },
        severity_bins: ['0-5', '6-9', '10-13', '14+'],
        transition_matrix: [
          { label: '0-5', cells: [{ count: 3, pct: 30.0 }, { count: 1, pct: 10.0 }, { count: 0, pct: 0.0 }, { count: 0, pct: 0.0 }] },
          { label: '6-9', cells: [{ count: 1, pct: 10.0 }, { count: 2, pct: 20.0 }, { count: 1, pct: 10.0 }, { count: 0, pct: 0.0 }] },
          { label: '10-13', cells: [{ count: 0, pct: 0.0 }, { count: 0, pct: 0.0 }, { count: 1, pct: 10.0 }, { count: 1, pct: 10.0 }] },
          { label: '14+', cells: [{ count: 0, pct: 0.0 }, { count: 0, pct: 0.0 }, { count: 0, pct: 0.0 }, { count: 0, pct: 0.0 }] },
        ],
        exact_score_bins: Array.from({ length: 25 }, (_, score) => String(score)),
        exact_score_matrix: cohortDemoSofaExactMatrix(demoPairs),
        score_scale: { min: 0, max: 24, unit: 'SOFA points', aggregation: 'nearest_integer_clamped_0_24' },
      },
    };
  }
  function init(config) {
    t = config.t;
    icon = config.icon;
    esc = config.esc;
    fmtInt = config.fmtInt;
    cohortText = config.cohortText;
    demoRowsForModule = config.demoRowsForModule;
  }

  window.EU_VIZ_COHORT_DEMO = {
    init,
    panelNote: cohortDemoPanelNote,
    coverageReview: cohortDemoCoverageReview,
    sofaReview: cohortDemoSofaReview,
  };
})();

