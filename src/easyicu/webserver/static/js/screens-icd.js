/* Screen: ICD Cohort Filter — disease-cohort definition by ICD code.
   Mirrors src/easyicu/webapp/icd_preview.py + sidebar.py demo estimator:
   include/exclude token matching, matched %, top matching codes, and the
   net cohort after filters. Supported on MIMIC-IV / MIMIC-III / eICU. */
(function () {
  const S = (window.SCREENS = window.SCREENS || {});
  function L(en, zh) { return t(en, zh); }

  // databases that expose an ICD diagnosis table (icd_preview.DB_META_PREVIEW + _supports_icd_filter)
  const ICD_DBS = [
    ['miiv', 'MIMIC-IV', 'ICD-9 / ICD-10 · diagnoses_icd'],
    ['mimic', 'MIMIC-III', 'ICD-9 · diagnoses_icd'],
    ['eicu', 'eICU-CRD', 'ICD-9 + text · diagnosis'],
  ];
  let icdDb = 'miiv';
  let icdInclude = 'A41, R65';   // sepsis / severe sepsis — matches the demo study
  let icdExclude = '';
  const ICD_TOTAL = 412;         // candidate ICU stays (matches the eligibility flow)

  // deterministic demo match fraction per ICD prefix (sidebar._demo_icd_token_fraction)
  function tokenFraction(token) {
    const n = String(token || '').toUpperCase().replace(/\./g, '').trim();
    if (!n) return 0;
    if (/^(A40|A41|A42)/.test(n)) return 0.40;
    if (/^(R57|785)/.test(n)) return 0.30;
    if (/^(J12|J13|J14|J15|J18)/.test(n)) return 0.24;
    if (/^(I50|428)/.test(n)) return 0.22;
    if (/^(N17|N18)/.test(n)) return 0.18;
    if (/^(C|D0)/.test(n)) return 0.12;
    return 0.14;
  }
  function splitTokens(q) { return String(q || '').split(/[,\s]+/).map(s => s.trim()).filter(Boolean); }
  function estimateMatch(total, tokens) {
    if (!total || !tokens.length) return 0;
    let miss = 1.0;
    tokens.slice(0, 12).forEach(tok => { miss *= 1 - tokenFraction(tok); });
    const frac = Math.min(0.88, Math.max(0, 1 - miss));
    return Math.min(total, Math.max(1, Math.round(total * frac)));
  }

  // demo top-codes per category (code, description) — realistic ICD references
  const CODE_BANK = {
    sepsis: [['A419', 'Sepsis, unspecified organism'], ['A4189', 'Other specified sepsis'], ['A412', 'Sepsis due to unspecified staphylococcus'], ['R6520', 'Severe sepsis without septic shock'], ['R6521', 'Severe sepsis with septic shock']],
    shock: [['R570', 'Cardiogenic shock'], ['R571', 'Hypovolemic shock'], ['R579', 'Shock, unspecified'], ['R6521', 'Severe sepsis with septic shock']],
    pneumonia: [['J189', 'Pneumonia, unspecified organism'], ['J159', 'Unspecified bacterial pneumonia'], ['J690', 'Aspiration pneumonia'], ['J156', 'Pneumonia due to Gram-negative bacteria']],
    hf: [['I509', 'Heart failure, unspecified'], ['I5021', 'Acute systolic heart failure'], ['I5033', 'Acute on chronic diastolic HF'], ['I5023', 'Acute on chronic systolic HF']],
    renal: [['N179', 'Acute kidney failure, unspecified'], ['N170', 'Acute kidney failure with tubular necrosis'], ['N189', 'Chronic kidney disease, unspecified'], ['N186', 'End stage renal disease']],
    cancer: [['C349', 'Malignant neoplasm of bronchus/lung'], ['C189', 'Malignant neoplasm of colon'], ['D0', 'Carcinoma in situ']],
    other: [['—', 'Matching diagnosis codes']],
  };
  function tokenCategory(token) {
    const n = String(token || '').toUpperCase().replace(/\./g, '').trim();
    if (/^(A40|A41|A42)/.test(n)) return 'sepsis';
    if (/^(R57|785)/.test(n)) return 'shock';
    if (/^(J12|J13|J14|J15|J18)/.test(n)) return 'pneumonia';
    if (/^(I50|428)/.test(n)) return 'hf';
    if (/^(N17|N18)/.test(n)) return 'renal';
    if (/^(C|D0)/.test(n)) return 'cancer';
    if (/^R65/.test(n)) return 'sepsis';
    return 'other';
  }
  function topCodes(tokens, matched) {
    const seen = new Set();
    const out = [];
    tokens.forEach(tok => { (CODE_BANK[tokenCategory(tok)] || CODE_BANK.other).forEach(([c, d]) => { if (!seen.has(c)) { seen.add(c); out.push([c, d]); } }); });
    const rows = out.slice(0, 6);
    // distribute the matched count across the top codes (descending, deterministic)
    const weights = rows.map((_, i) => Math.pow(0.62, i));
    const wsum = weights.reduce((a, b) => a + b, 0);
    return rows.map((r, i) => [r[0], r[1], Math.max(1, Math.round(matched * weights[i] / wsum))]);
  }

  function dirCard(label, query, kind) {
    const tokens = splitTokens(query);
    if (!tokens.length) return `<div class="icd-card empty"><div class="icd-card-h">${label}</div><div class="icd-empty-line">${L('No tokens — add ICD code prefixes or terms above.', '暂无条件 —— 在上方输入 ICD 编码前缀或关键词。')}</div></div>`;
    const matched = estimateMatch(ICD_TOTAL, tokens);
    const pct = (matched / ICD_TOTAL * 100);
    const codes = topCodes(tokens, matched);
    return `
      <div class="icd-card ${kind}">
        <div class="icd-card-h"><span class="icd-dir ${kind}">${label}</span><span class="icd-match mono">${matched.toLocaleString()} / ${ICD_TOTAL.toLocaleString()} · ${pct.toFixed(1)}%</span></div>
        <div class="icd-tokens">${tokens.map(tk => `<span class="icd-tok">${tk}</span>`).join('')}</div>
        <div class="icd-codes-h">${L('Top matching ICD codes', '匹配频率最高的 ICD 编码')}</div>
        <div class="icd-codes">
          ${codes.map(([c, d, n]) => `<div class="icd-code-row"><span class="mono icc-code">${c}</span><span class="icc-desc">${d}</span><span class="mono icc-n">${n}</span></div>`).join('')}
        </div>
      </div>`;
  }

  /* ICD Cohort Filter — folded into the extraction cohort filter as a compact
     block — the single source for ICD cohort filtering. The legacy standalone
     screen was removed; the router redirects #icd → extraction. */
  window.EUIcd = {
    block() {
      const incTokens = splitTokens(icdInclude);
      const excTokens = splitTokens(icdExclude);
      const incMatch = estimateMatch(ICD_TOTAL, incTokens);
      const excMatch = estimateMatch(ICD_TOTAL, excTokens);
      const base = incTokens.length ? incMatch : ICD_TOTAL;
      const net = Math.max(base - Math.min(excMatch, base), 0);
      const netPct = ICD_TOTAL ? net / ICD_TOTAL * 100 : 0;
      return `
      <div class="icd-embed">
        <div class="row" style="justify-content:space-between;align-items:flex-start;gap:10px;flex-wrap:wrap;">
          <div>
            <div class="row gap-6" style="font-size:12.5px;font-weight:600;color:var(--ink-2);">${icon('list', 14)} ${L('Disease cohort (ICD)', '疾病队列（ICD）')}</div>
            <div style="font-size:11px;color:var(--ink-4);margin-top:3px;max-width:520px;">${L('Carve a disease cohort by diagnosis code, before the inclusion criteria above. Include narrows · exclude removes.', '在上面的纳入标准之前，按诊断编码划出疾病队列。包含用于缩小 · 排除用于剔除。')}</div>
          </div>
          <span class="mono" style="font-size:10.5px;color:var(--ink-4);white-space:nowrap;">MIMIC-IV · MIMIC-III · eICU</span>
        </div>
        <div class="icd-inputs" style="margin-top:12px;">
          <div class="icd-field">
            <label>${icon('plus', 13)} ${L('Include diagnoses', '包含诊断')}</label>
            <input id="icdIncInput" type="text" value="${icdInclude.replace(/"/g, '&quot;')}" placeholder="${L('e.g. A41, R65, sepsis', '如 A41、R65、sepsis')}" autocomplete="off" spellcheck="false" />
            <span class="icd-field-hint">${L('Comma- or space-separated ICD prefixes or terms', '逗号或空格分隔的 ICD 前缀或关键词')}</span>
          </div>
          <div class="icd-field">
            <label>${icon('close', 13)} ${L('Exclude diagnoses', '排除诊断')}</label>
            <input id="icdExcInput" type="text" value="${icdExclude.replace(/"/g, '&quot;')}" placeholder="${L('e.g. C (cancers), N18', '如 C（肿瘤）、N18')}" autocomplete="off" spellcheck="false" />
            <span class="icd-field-hint">${L('Patients matching these are removed from the cohort', '匹配这些条件的患者会被剔除')}</span>
          </div>
        </div>
        <div class="icd-net" style="margin-top:12px;">
          <div class="icd-net-ico">${icon('cohort', 18)}</div>
          <div class="icd-net-body">
            <div class="icd-net-t">${L('Final cohort after ICD filters', 'ICD 筛选后的最终队列')}</div>
            <div class="icd-net-sub">${incTokens.length ? L('include', '包含') + ' ' + incMatch.toLocaleString() : L('all candidates', '全部候选')}${excTokens.length ? ' − ' + L('exclude', '排除') + ' ' + Math.min(excMatch, base).toLocaleString() : ''}</div>
          </div>
          <div class="icd-net-val"><span class="mono inv">${net.toLocaleString()}</span><span class="icd-net-tot mono">/ ${ICD_TOTAL.toLocaleString()} · ${netPct.toFixed(1)}%</span></div>
        </div>
        <div class="icd-net-bar"><div class="inb-fill" style="width:${netPct.toFixed(1)}%;"></div></div>
        <div class="icd-cards" style="margin-top:12px;">
          ${dirCard(L('Include', '包含'), icdInclude, 'inc')}
          ${dirCard(L('Exclude', '排除'), icdExclude, 'exc')}
        </div>
      </div>`;
    },
    bind(root) {
      const inc = root.querySelector('#icdIncInput');
      const exc = root.querySelector('#icdExcInput');
      let timer = null;
      function live(which, el) {
        el.addEventListener('input', () => {
          if (which === 'inc') icdInclude = el.value; else icdExclude = el.value;
          clearTimeout(timer);
          timer = setTimeout(() => {
            window.__euRender();
            const again = document.querySelector(which === 'inc' ? '#icdIncInput' : '#icdExcInput');
            if (again) { again.focus(); again.setSelectionRange(again.value.length, again.value.length); }
          }, 280);
        });
      }
      if (inc) live('inc', inc);
      if (exc) live('exc', exc);
    },
  };
})();
