/* ICD cohort criteria owned by Data Extraction.
   This widget records include/exclude rules only. Cohort sizes and code
   frequencies must come from the exact selected source during extraction;
   the browser never estimates them from seeded demo constants. */
(function () {
  function L(en, zh) { return t(en, zh); }

  let icdInclude = '';
  let icdExclude = '';

  function splitTokens(query) {
    return String(query || '')
      .split(/[,\s]+/)
      .map(token => token.trim())
      .filter(Boolean);
  }

  function sourceLabel(context) {
    const value = context && String(context.databaseLabel || '').trim();
    if (value) return value;
    const database = context && String(context.database || '').trim().toLowerCase();
    if (database === 'miiv') return 'MIMIC-IV';
    if (database === 'mimic') return 'MIMIC-III';
    if (database === 'eicu') return 'eICU-CRD';
    return L('Current selected source', '当前所选数据源');
  }

  function tokenSummary(label, tokens, kind) {
    if (!tokens.length) return '';
    return `<div class="icd-rule-row ${kind}" style="display:flex;align-items:center;gap:8px;margin-top:8px;">
      <span class="icd-rule-label" style="font-size:11px;font-weight:600;color:var(--ink-3);">${label}</span>
      <span class="icd-tokens">${tokens.map(token => `<span class="icd-tok">${escHtml(token)}</span>`).join('')}</span>
    </div>`;
  }

  window.EUIcd = {
    apply(value) {
      const source = value && typeof value === 'object' ? value : {};
      const include = Array.isArray(source.include_diagnoses)
        ? source.include_diagnoses : splitTokens(source.icd_include);
      const exclude = Array.isArray(source.exclude_diagnoses)
        ? source.exclude_diagnoses : splitTokens(source.icd_exclude);
      icdInclude = include.map(token => String(token).trim()).filter(Boolean).join(', ');
      icdExclude = exclude.map(token => String(token).trim()).filter(Boolean).join(', ');
      return this.contract();
    },
    contract() {
      return {
        icd_include: splitTokens(icdInclude).join(', '),
        icd_exclude: splitTokens(icdExclude).join(', '),
        include_diagnoses: splitTokens(icdInclude),
        exclude_diagnoses: splitTokens(icdExclude),
      };
    },
    block(context) {
      const includeTokens = splitTokens(icdInclude);
      const excludeTokens = splitTokens(icdExclude);
      const label = sourceLabel(context);
      const safeLabel = escHtml(label);
      const real = !!(context && context.real);
      const realMessage = escHtml(L(
        `The rules will be evaluated against ${label} when extraction starts. No estimated patient count or synthetic code frequency is shown.`,
        `开始抽取时会在 ${label} 中计算这些条件。这里不展示估算人数或合成的编码频数。`
      ));
      return `
      <div class="icd-embed">
        <div class="row" style="justify-content:space-between;align-items:flex-start;gap:10px;flex-wrap:wrap;">
          <div>
            <div class="row gap-6" style="font-size:12.5px;font-weight:600;color:var(--ink-2);">${icon('list', 14)} ${L('Disease cohort (ICD)', '疾病队列（ICD）')}</div>
            <div style="font-size:11px;color:var(--ink-4);margin-top:3px;max-width:520px;">${L('Filter the current source by diagnosis-code prefixes or terms. Include narrows the cohort; exclude removes matching stays.', '在当前数据源中按诊断编码前缀或关键词筛选。包含用于缩小队列；排除用于剔除匹配住院。')}</div>
          </div>
          <span class="pill mono" data-icd-source>${safeLabel}</span>
        </div>
        <div class="icd-inputs" style="margin-top:12px;">
          <div class="icd-field">
            <label>${icon('plus', 13)} ${L('Include diagnoses', '包含诊断')}</label>
            <input id="icdIncInput" type="text" value="${escHtml(icdInclude)}" placeholder="${L('e.g. J18, N17, I50', '如 J18、N17、I50')}" autocomplete="off" spellcheck="false" />
            <span class="icd-field-hint">${L('Comma- or space-separated ICD prefixes or terms', '逗号或空格分隔的 ICD 前缀或关键词')}</span>
          </div>
          <div class="icd-field">
            <label>${icon('close', 13)} ${L('Exclude diagnoses', '排除诊断')}</label>
            <input id="icdExcInput" type="text" value="${escHtml(icdExclude)}" placeholder="${L('Optional', '可选')}" autocomplete="off" spellcheck="false" />
            <span class="icd-field-hint">${L('Matching stays are removed from the cohort', '匹配条件的住院会从队列中剔除')}</span>
          </div>
        </div>
        ${(includeTokens.length || excludeTokens.length) ? `<div class="icd-rule-summary" aria-label="${L('Selected ICD criteria', '已选 ICD 条件')}">
          ${tokenSummary(L('Include', '包含'), includeTokens, 'inc')}
          ${tokenSummary(L('Exclude', '排除'), excludeTokens, 'exc')}
        </div>` : ''}
        <div class="note info" style="padding:10px 12px;margin-top:12px;">
          <div class="ico">${icon('shield', 14)}</div>
          <div class="body"><div class="t" style="font-size:12px;">${real ? L('Bound to the current local source', '已绑定当前本地数据源') : L('Filter definition only', '仅保存筛选定义')}</div><div class="d" style="font-size:11px;margin:0;">${real
            ? realMessage
            : L('Demo mode stores these rules but does not claim a real cohort size. Use a local source to calculate the matched cohort.', '演示模式仅保存这些条件，不声称真实队列人数。请连接本地数据源后计算匹配队列。')}</div></div>
        </div>
      </div>`;
    },
    bind(root) {
      const inc = root.querySelector('#icdIncInput');
      const exc = root.querySelector('#icdExcInput');
      let timer = null;
      function live(which, input) {
        input.addEventListener('input', () => {
          if (which === 'inc') icdInclude = input.value;
          else icdExclude = input.value;
          clearTimeout(timer);
          timer = setTimeout(() => {
            window.__euRender();
            const again = document.querySelector(which === 'inc' ? '#icdIncInput' : '#icdExcInput');
            if (again) {
              again.focus();
              again.setSelectionRange(again.value.length, again.value.length);
            }
          }, 280);
        });
      }
      if (inc) live('inc', inc);
      if (exc) live('exc', exc);
    },
  };
})();
