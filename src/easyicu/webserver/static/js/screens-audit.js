/* Screens: Coverage Audit + SOFA Reclassification.
   Coverage audit mirrors data_coverage_audit_page.py (module × subgroup
   coverage matrix + eligibility flow). SOFA reclassification mirrors
   sofa_reclassification.py (worst-ICU / first-24h / time-aligned modes,
   Up/Same/Down groups, organ deltas, severity confusion matrix).
   Demo numbers are seeded + deterministic. */
(function () {
  const S = (window.SCREENS = window.SCREENS || {});
  const C = () => window.EU_CATALOG;
  function L(en, zh) { return t(en, zh); }

  /* ============================================================
     COVERAGE AUDIT
     ============================================================ */
  // per-module base coverage (demo) and the 5 clinical subgroups
  const COV_MODULES = [
    ['vitals', 'Vital Signs', '生命体征', 99],
    ['laboratory', 'Laboratory', '实验室', 93],
    ['input_output', 'Input / Output', '出入量', 84],
    ['medications', 'Medications', '药物', 79],
    ['resp_support', 'Respiratory Support', '呼吸支持', 71],
    ['severity', 'Severity Scores', '严重程度评分', 96],
    ['demographics', 'Demographics', '人口统计', 100],
    ['outcomes', 'Outcomes', '结局', 100],
  ];
  const COV_SUBGROUPS = [
    ['overall', 'Overall', '总体', 248, 0],
    ['survived', 'Survived', '存活', 198, -1],
    ['deceased', 'Deceased', '死亡', 50, 3],
    ['sofa_low', 'SOFA ≤ 6', 'SOFA ≤ 6', 121, -2],
    ['sofa_high', 'SOFA > 6', 'SOFA > 6', 127, 2],
  ];
  function covCell(base, delta) { return Math.max(0, Math.min(100, base + delta)); }
  function covColor(v) {
    // matches the real warm→teal scale: low = warm cream, high = teal
    if (v >= 95) return 'oklch(72% 0.07 200)';
    if (v >= 88) return 'oklch(82% 0.055 200)';
    if (v >= 80) return 'oklch(90% 0.04 200)';
    if (v >= 70) return 'oklch(94% 0.03 90)';
    return 'oklch(93% 0.06 70)';
  }
  function covTextColor(v) { return v >= 88 ? '#0c2a30' : 'var(--ink-2)'; }

  function coverageMatrix() {
    const mods = COV_MODULES, subs = COV_SUBGROUPS;
    const header = `<div class="cov-cell cov-corner"></div>` +
      subs.map(s => `<div class="cov-cell cov-colh"><span>${L(s[1], s[2])}</span><span class="cov-sn mono">n=${s[3]}</span></div>`).join('');
    const rows = mods.map(m => {
      const cells = subs.map(s => {
        const v = covCell(m[3], s[4]);
        return `<div class="cov-cell cov-v" style="background:${covColor(v)};color:${covTextColor(v)}" title="${L(m[1], m[2])} · ${L(s[1], s[2])}: ${v.toFixed(0)}%">${v.toFixed(0)}</div>`;
      }).join('');
      return `<div class="cov-rowh">${L(m[1], m[2])}</div>${cells}`;
    }).join('');
    return `<div class="cov-grid" style="grid-template-columns:150px repeat(${subs.length}, 1fr);">${header}${rows}</div>`;
  }

  // eligibility waterfall (demo Step-2 flow)
  const ELIG_FLOW = [
    ['Candidate ICU stays', '候选 ICU 住院', 412, 0, 'before Step 2 filters', '第 2 步筛选前'],
    ['Age ≥ 18', '年龄 ≥ 18', 401, 11, 'Step 2 cohort filter', '第 2 步队列筛选'],
    ['First ICU stay', '首次 ICU 住院', 332, 69, 'Step 2 cohort filter', '第 2 步队列筛选'],
    ['ICU LOS ≥ 24h', 'ICU 时长 ≥ 24h', 271, 61, 'Step 2 cohort filter', '第 2 步队列筛选'],
    ['Sepsis-3 positive', 'Sepsis-3 阳性', 248, 23, 'Step 2 cohort filter', '第 2 步队列筛选'],
    ['Final extracted cohort', '最终提取队列', 248, 0, '60.2% retained', '保留 60.2%'],
  ];

  /* Coverage Audit — now rendered as a tab inside Cohort Statistics
     (window.EUAudit.panel()), not a standalone nav destination. */
  window.EUAudit = {
    panel() {
      const cards = [
        [L('Patients', '患者数'), '248'],
        [L('Modules', '模块数'), '8'],
        [L('Clinical concepts', '临床概念'), String(C().totalConcepts)],
        [L('Median coverage', '覆盖度中位数'), '92%'],
        [L('Watchlist', '覆盖度关注'), '2'],
      ];
      return `
      <div class="page-head" style="display:none">
        <div class="eyebrow">${t('Data coverage & eligibility audit', '数据覆盖度与纳排审计')}</div>
        <h1 style="margin-top:6px;">${t('Coverage Audit', '覆盖度审计')}</h1>
        <p class="lead">${t('Module-level data coverage across clinically meaningful subgroups, plus the Step 2 extraction flow. This is the check the evidence gate reads before any modelling.', '按临床相关亚组展示模块级数据覆盖度,以及第 2 步抽取的纳排流程。这是证据闸在建模前会读取的检查。')}</p>
      </div>

      <div class="audit-cards">
        ${cards.map(([k, v]) => `<div class="audit-card"><div class="ac-k">${k}</div><div class="ac-v mono">${v}</div></div>`).join('')}
      </div>

      <div class="audit-cols">
        <div class="card pad">
          <div class="cov-title"><span class="cov-letter">B</span>${t('Data coverage by module and subgroup (%)', '按模块和亚组的数据覆盖度 (%)')}</div>
          ${coverageMatrix()}
          <div class="cov-legend">
            <span>${t('Lower', '较低')}</span>
            <span class="cl-bar"></span>
            <span>${t('Higher', '较高')}</span>
          </div>
        </div>
        <div class="card pad">
          <div class="cov-title">${t('Eligibility flow', '纳排流程')}</div>
          <div class="elig-flow">
            ${ELIG_FLOW.map((s, i) => `
              <div class="elig-step ${i === ELIG_FLOW.length - 1 ? 'final' : ''}">
                <div class="es-bar" style="width:${Math.round(s[2] / ELIG_FLOW[0][2] * 100)}%;"></div>
                <div class="es-body">
                  <div class="es-label">${L(s[0], s[1])}</div>
                  <div class="es-meta"><span class="es-count mono">${s[2].toLocaleString()}</span>${s[3] ? `<span class="es-excl">−${s[3]}</span>` : `<span class="es-note">${L(s[4], s[5])}</span>`}</div>
                </div>
              </div>`).join('')}
          </div>
        </div>
      </div>

      <div class="audit-note">${icon('shield', 13)} <span>${t('Missingness denominators: d=LOS uses stay-specific ICU time; d=72h uses a fallback window; d=demo uses the simulated horizon; d=static means one observation per stay.', '缺失率分母:d=LOS 按患者 ICU 住院时长;d=72h 为兜底时间窗;d=demo 为演示时间窗;d=static 表示每位患者单次观测。')}</span></div>`;
    },
  };

  /* ============================================================
     SOFA RECLASSIFICATION
     ============================================================ */
  const RC_MODES = [
    ['worst_icu', 'Worst ICU score', 'ICU期间最高分', 'Patient-level maximum SOFA-1 and SOFA-2 across the whole ICU stay.', '按患者汇总 ICU 全程 SOFA-1 与 SOFA-2 的最高值。'],
    ['first24', 'First 24h paired worst', '首24小时配对最高', 'Patient-level maximum from time-aligned points in the first 24 ICU hours.', '仅用入 ICU 后 0–24h 同时间点配对的分数,按患者取最高。'],
    ['aligned', 'Time-aligned points', '同时间点配对', 'Row-level comparison at the same stay and charttime; denominator is paired time points.', '相同 stay 与 charttime 上逐点比较;分母为配对时间点。'],
  ];
  let rcMode = 'worst_icu';

  // seeded demo metrics per mode
  const RC_DATA = {
    worst_icu: {
      denom: 248, denomLabel: ['Patients', '患者数'], denomHint: ['paired SOFA', '双 SOFA 记录'],
      up: 18.5, down: 9.7, same: 71.8, median: '0.0', range: 'range −4 to +6',
      groups: [['Up-classified', '上调分层', 46, 18.5, 31.0, 6.2], ['Same', '不变', 178, 71.8, 18.5, 4.1], ['Down-classified', '下调分层', 24, 9.7, 12.5, 3.4]],
    },
    first24: {
      denom: 241, denomLabel: ['Patients', '患者数'], denomHint: ['first-24h paired', '前 24h 配对'],
      up: 15.8, down: 8.3, same: 75.9, median: '0.0', range: 'range −3 to +5',
      groups: [['Up-classified', '上调分层', 38, 15.8, 28.9, 5.8], ['Same', '不变', 183, 75.9, 17.5, 4.0], ['Down-classified', '下调分层', 20, 8.3, 15.0, 3.6]],
    },
    aligned: {
      denom: 5972, denomLabel: ['Paired points', '配对时间点'], denomHint: ['time-aligned rows', '同时间点记录'],
      up: 12.4, down: 7.1, same: 80.5, median: '0.0', range: 'range −3 to +4',
      groups: [['Up-classified', '上调分层', 740, 12.4, 26.1, 0], ['Same', '不变', 4808, 80.5, 19.0, 0], ['Down-classified', '下调分层', 424, 7.1, 14.6, 0]],
    },
  };

  // organ-level delta (SOFA-2 − SOFA-1), demo — SOFA-2 relaxes liver/coag/renal thresholds
  const RC_ORGANS = [
    ['Liver', '肝脏', 0.18, 0.34, 78, 12],
    ['Coagulation', '凝血', 0.12, 0.28, 61, 18],
    ['Renal', '肾脏', 0.09, 0.31, 54, 27],
    ['Respiratory', '呼吸', 0.03, 0.19, 38, 29],
    ['Neurological', '神经', 0.07, 0.15, 33, 14],
    ['Cardiovascular', '循环', -0.06, 0.22, 31, 44],
  ];

  // 6×6 severity confusion matrix (rows SOFA-1, cols SOFA-2), seeded; slight up-shift
  const RC_BINS = ['0-2', '3-5', '6-8', '9-11', '12-15', '≥16'];
  const RC_MATRIX = [
    [38, 11, 1, 0, 0, 0],
    [7, 52, 14, 1, 0, 0],
    [0, 6, 41, 9, 1, 0],
    [0, 0, 5, 22, 4, 0],
    [0, 0, 0, 3, 9, 1],
    [0, 0, 0, 0, 1, 2],
  ];
  function rcMatrixMax() { let m = 0; RC_MATRIX.forEach(r => r.forEach(v => { if (v > m) m = v; })); return m; }

  /* SOFA Reclassification — now rendered as a tab inside Cohort Statistics
     (window.EUSofa.panel()/.bind()), not a standalone nav destination. */
  window.EUSofa = {
    panel() {
      const d = RC_DATA[rcMode];
      const discordant = (d.up + d.down).toFixed(1);
      const kpis = [
        [d.denom.toLocaleString(), L(d.denomLabel[0], d.denomLabel[1]), L(d.denomHint[0], d.denomHint[1]), 'n'],
        [discordant + '%', L('Discordant', '重新分层'), 'SOFA-2 ≠ SOFA-1', 'swap'],
        [d.up.toFixed(1) + '%', L('Up-classified', '上调分层'), L('higher SOFA-2', 'SOFA-2 更高'), 'up'],
        [d.down.toFixed(1) + '%', L('Down-classified', '下调分层'), L('lower SOFA-2', 'SOFA-2 更低'), 'down'],
        [d.median, L('Median Δ', 'Δ 中位数'), L(d.range, d.range.replace('range', '范围')), 'delta'],
      ];
      const kpiIco = { n: 'cohort', swap: 'refresh', up: 'arrow', down: 'arrow', delta: 'layers' };
      const maxPct = Math.max(...d.groups.map(g => g[3]));
      const mmax = rcMatrixMax();
      return `
      <div class="page-head" style="display:none">
        <div class="eyebrow">${t('SOFA-1 vs SOFA-2 reclassification', 'SOFA-1 与 SOFA-2 重分类')}</div>
        <h1 style="margin-top:6px;">${t('SOFA Reclassification', 'SOFA 重分类')}</h1>
        <p class="lead">${t('How the same cohort scores under SOFA-1 vs the 2025 SOFA-2 standard — who moves up, who moves down, and which organ systems drive the change.', '同一队列在 SOFA-1 与 2025 版 SOFA-2 标准下的评分差异 —— 谁上调、谁下调,以及哪些器官系统在驱动变化。')}</p>
      </div>

      <div class="rc-modes">
        <span class="rc-modes-lbl">${t('Analysis mode', '分析口径')}</span>
        <div class="rc-seg">
          ${RC_MODES.map(m => `<button class="${rcMode === m[0] ? 'active' : ''}" data-rc-mode="${m[0]}">${L(m[1], m[2])}</button>`).join('')}
        </div>
        <span class="rc-mode-desc">${L(RC_MODES.find(m => m[0] === rcMode)[3], RC_MODES.find(m => m[0] === rcMode)[4])}</span>
      </div>

      <div class="rc-kpis">
        ${kpis.map(([v, label, hint, kind]) => `
          <div class="rc-kpi rc-${kind}">
            <div class="rk-top"><span class="rk-ico">${icon(kpiIco[kind], 13)}</span><span class="rk-label">${label}</span></div>
            <div class="rk-val mono">${v}</div>
            <div class="rk-hint">${hint}</div>
          </div>`).join('')}
      </div>

      <div class="rc-cols">
        <div class="card pad">
          <div class="rc-sec-t">${t('Reclassification groups', '重分类分组')}</div>
          <div class="rc-groups">
            ${d.groups.map(g => `
              <div class="rc-grow">
                <div class="rg-head"><span class="rg-name">${L(g[0], g[1])}</span><span class="rg-pct mono">${g[3].toFixed(1)}%</span></div>
                <div class="rg-bar"><div class="rg-fill ${g[0] === 'Up-classified' ? 'up' : g[0] === 'Down-classified' ? 'down' : 'same'}" style="width:${(g[3] / maxPct * 100).toFixed(0)}%;"></div></div>
                <div class="rg-meta"><span>${g[2].toLocaleString()} ${L('patients', '例')}</span><span>${L('mortality', '死亡率')} <b>${g[4].toFixed(1)}%</b></span>${g[5] ? `<span>${L('LOS', '住院')} <b>${g[5].toFixed(1)}d</b></span>` : ''}</div>
              </div>`).join('')}
          </div>
          <div class="rc-sec-t mt-20">${t('Organ-level change (SOFA-2 − SOFA-1)', '器官级变化 (SOFA-2 − SOFA-1)')}</div>
          <div class="rc-organs">
            <div class="ro-row ro-head"><span class="ro-name">${t('Organ', '器官')}</span><span>${t('Mean Δ', '平均 Δ')}</span><span>${t('|Δ|', '|Δ|')}</span><span>${t('Up', '上调')}</span><span>${t('Down', '下调')}</span></div>
            ${RC_ORGANS.map(o => `
              <div class="ro-row">
                <span class="ro-name">${L(o[0], o[1])}</span>
                <span class="mono ${o[2] > 0 ? 'pos' : o[2] < 0 ? 'neg' : ''}">${o[2] > 0 ? '+' : ''}${o[2].toFixed(2)}</span>
                <span class="mono">${o[3].toFixed(2)}</span>
                <span class="mono ro-up">${o[4]}</span>
                <span class="mono ro-down">${o[5]}</span>
              </div>`).join('')}
          </div>
        </div>

        <div class="card pad">
          <div class="rc-sec-t">${t('Severity reclassification matrix', '严重度重分类矩阵')}</div>
          <div class="rc-matrix-wrap">
            <div class="rc-axis-y">${t('SOFA-1 severity', 'SOFA-1 严重度')}</div>
            <div>
              <div class="rc-matrix" style="grid-template-columns:auto repeat(${RC_BINS.length}, 1fr);">
                <div class="rm-corner mono">1 \\ 2</div>
                ${RC_BINS.map(b => `<div class="rm-colh mono">${b}</div>`).join('')}
                ${RC_MATRIX.map((row, ri) => `
                  <div class="rm-rowh mono">${RC_BINS[ri]}</div>
                  ${row.map((v, ci) => {
                    const intensity = v === 0 ? 0 : v / mmax;
                    const diag = ri === ci;
                    const bg = v === 0 ? 'var(--surface-2)' : `oklch(${(96 - intensity * 28).toFixed(1)}% ${(0.02 + intensity * 0.09).toFixed(3)} ${diag ? 255 : ci > ri ? 25 : 150})`;
                    return `<div class="rm-cell ${diag ? 'diag' : ci > ri ? 'up' : 'down'}" style="background:${bg};color:${intensity > 0.55 ? '#fff' : 'var(--ink-3)'}" title="SOFA-1 ${RC_BINS[ri]} → SOFA-2 ${RC_BINS[ci]}: ${v}">${v || ''}</div>`;
                  }).join('')}
                `).join('')}
              </div>
              <div class="rc-axis-x">${t('SOFA-2 severity', 'SOFA-2 严重度')}</div>
            </div>
          </div>
          <div class="rc-matrix-legend">
            <span class="rml up">${t('above diagonal · up-classified', '对角线上方 · 上调')}</span>
            <span class="rml diag">${t('diagonal · unchanged', '对角线 · 不变')}</span>
            <span class="rml down">${t('below · down-classified', '下方 · 下调')}</span>
          </div>
        </div>
      </div>`;
    },
    bind(root) {
      root.querySelectorAll('[data-rc-mode]').forEach(b => b.addEventListener('click', () => { rcMode = b.dataset.rcMode; window.__euRender(); }));
    },
  };
})();
