/* Owner: product label projection widget. */
/* User-visible product labels are projected here so internal runtime names do
   not leak differently across Copilot and Copilot conversation. Internal API,
   storage, and diagnostics identifiers remain unchanged.

   D-P2-1 default-title contract: when the stored title is empty or one of the
   legacy defaults below, every consumer projects `fallback || 研究项目 /
   Research project`. Call sites must pass the stored question (or id) as the
   fallback and must NOT invent new per-screen defaults — extend the sets here
   instead so the rail, the known-folder picker, the removal dialog and the
   run-files owner stay consistent. Consumers call this defensively
   (`window.EU_PRODUCT_LABELS?.projectTitle?.(...) ?? String(...).slice(0,200)`)
   so a bundle served without this owner still renders bounded raw text. */
(function () {
  const COPILOT_DEFAULT_TITLES = new Set(['Pi Copilot', 'EasyICU Copilot']);
  const PROJECT_DEFAULT_TITLES = new Set([
    ...COPILOT_DEFAULT_TITLES,
    'Untitled ICU study',
    'Untitled guided study',
  ]);

  function copilotTitle(value, fallback) {
    const title = String(value || '').trim();
    const alternative = String(fallback || '').trim();
    if (!title || COPILOT_DEFAULT_TITLES.has(title)) {
      return alternative || 'EasyICU Copilot';
    }
    return title;
  }

  function projectTitle(value, fallback) {
    const title = String(value || '').trim();
    const alternative = String(fallback || '').trim();
    if (!title || PROJECT_DEFAULT_TITLES.has(title)) {
      return alternative || (window.EU_LANG === 'zh' ? '研究项目' : 'Research project');
    }
    return title;
  }

  // The short plan card's only vocabulary owner. Clinical terms reuse the
  // catalog; an unknown column remains unresolved instead of becoming UI text.
  function clinicalPlan(summary, tr) {
    const zh = tr('en', 'zh') === 'zh';
    const rows = Array.isArray(summary && summary.lines) ? summary.lines : [];
    const pick = pair => tr(pair[0], pair[1]);
    const titles = {
      population: ['Population', '研究人群'], comparison: ['Comparison', '比较'],
      outcome: ['Outcome', '结局'], adjustment: ['Adjustment', '调整'], data_checks: ['Data checks', '数据检查'],
    };
    const englishTerm = value => value.replace(/[A-Za-z][A-Za-z0-9]*/g, word =>
      /^(?:Charlson|Glasgow|Richmond|Elixhauser|Walraven)$/.test(word) || !/^[A-Z][a-z]+$/.test(word) ? word : word.toLowerCase());
    const spacedZh = value => value.replace(/([\u3400-\u9fff])([A-Za-z0-9<>≥≤=≠])/g, '$1 $2')
      .replace(/([A-Za-z0-9])([\u3400-\u9fff])/g, '$1 $2');
    const clean = value => {
      const text = String(value || '').trim();
      if (!text || text.length > 64 || /[_\n]|<[/A-Za-z!]|\b(?:column|schema|identifier)\b|\b(?:patient|subject|stay)\s+(?:id|key)\b/i.test(text)) return '';
      const words = text.replace(/mmol\/L|mg\/dL|mmHg|ICU|SOFA|APACHE|BMI|HbA1c/gi, '');
      if (zh && /[A-Za-z]/.test(words)) return '';
      if (!zh && /[\u3400-\u9fff]/.test(text)) return '';
      return text;
    };
    const name = term => {
      const id = String(term && term.concept || '');
      if (/(?:^|_)(?:id|key|identifier)$/.test(id)) return '';
      const dict = window.EU_CATALOG && window.EU_CATALOG.dict || {};
      const pair = Object.prototype.hasOwnProperty.call(dict, id) ? dict[id] : null;
      const label = clean(term && term.label) || (Array.isArray(pair) ? clean(String(pick(pair) || '').replace(/\s*[（(][^）)]*[）)]\s*$/, '')) : '');
      return zh ? spacedZh(label) : englishTerm(label);
    };
    const roster = terms => {
      const all = Array.isArray(terms) ? terms : [];
      const names = all.map(name);
      const known = names.filter(Boolean);
      const short = known.slice(0, known.length > 4 ? 3 : 4).join(tr(', ', '、'));
      return { text: known.length > 4 ? tr(`${short} and ${known.length - 3} more`, `${short}等 ${known.length} 项`) : short,
        unresolved: names.some(value => !value) };
    };
    const lines = Object.entries(titles).map(([key, title]) => {
      const row = rows.find(item => item && item.key === key) || { unresolved: true };
      const terms = roster(row.variables);
      let uncertain = row.unresolved || (row.mode !== 'prediction' || key !== 'adjustment') && terms.unresolved;
      if (key === 'comparison' && row.mode === 'descriptive') title = ['Groups', '分组'];
      if (key === 'comparison' && row.mode === 'prediction') title = ['Predictors', '预测变量'];
      if (key === 'adjustment' && row.mode === 'prediction') title = ['Evaluation', '评估'];
      let text = '';
      if (key === 'population') {
        const criteria = Array.isArray(row.criteria) ? row.criteria : [];
        const named = criteria.map(item => {
          const label = name(item);
          const op = { '>=': '≥', '<=': '≤', '==': '=', '!=': '≠', '>': '>', '<': '<' }[item.op];
          if (!label || !op || !Number.isFinite(item.value)) return '';
          const condition = item.concept === 'age' && !zh ? `aged ${op}${item.value}`
            : `${label} ${op}${item.value}${item.concept === 'age' ? ' 岁' : ''}`;
          return `${item.exclude ? tr('excluding ', '排除') : ''}${condition}`;
        });
        uncertain ||= named.some(value => !value);
        const conditions = named.filter(Boolean).slice(0, 4).join(tr(', ', '、'));
        text = conditions ? tr(`ICU patients ${conditions}`, `${conditions}的 ICU 患者`)
          : row.all_rows ? tr('All patients in the supplied cohort', '所提供队列的全部患者')
            : tr('Eligibility to be confirmed', '入组条件待确认');
        if (criteria.length > 4) text = tr(`Patients meeting ${criteria.length} declared criteria`, `符合 ${criteria.length} 项预设条件的患者`);
      } else if (key === 'comparison') {
        const groups = (row.groups || []).map(group => {
          // Mathematical '<' is a threshold, not markup; all rendering escapes it.
          const value = String(zh ? group.zh : group.en || '');
          return clean(value) ? value : '';
        });
        uncertain ||= row.mode === 'association' && groups.some(value => !value);
        const sharedUnit = groups[0] && groups[0].match(/\s+([A-Za-zµμ][A-Za-z0-9µμ/²³^%·-]*)$/);
        const englishGroups = [...groups];
        if (sharedUnit && groups[1] && groups[1].endsWith(sharedUnit[0])) englishGroups[0] = groups[0].slice(0, -sharedUnit[0].length);
        text = row.mode === 'prediction'
          ? terms.text || tr('Predictors to be confirmed', '预测变量待确认')
          : !terms.text ? tr('Groups to be confirmed', '分组待确认')
            : row.mode === 'descriptive' ? tr(`By ${terms.text}`, `按${terms.text}分层`)
              : groups.length === 2 && groups.every(Boolean) ? tr(`${terms.text} ${englishGroups.join(' vs ')}`, `${terms.text} ${groups.join(' 与 ')} 两组`)
                : row.continuous ? tr(`${terms.text} levels`, `${terms.text}水平`)
                  : tr(`${terms.text}; groups to be confirmed`, `${terms.text}；分组待确认`);
      } else if (key === 'outcome') {
        text = terms.text || tr('Outcome to be confirmed', '结局待确认');
      } else if (key === 'adjustment') {
        text = row.mode === 'descriptive' ? tr('None; stratified description only', '不做调整，仅分层描述')
          : row.mode === 'unadjusted' ? tr('None', '不做调整')
            : row.mode === 'prediction' ? tr('Predictions against observed outcomes', '预测与实际结局对照')
              + (row.evaluation === 'discrimination_calibration' ? tr(' (discrimination and calibration)', '（区分度与校准）') : '')
              : terms.text || tr('Adjustment set to be confirmed', '调整变量待确认');
      } else {
        const missing = {
          complete_case: ['Complete-case analysis', '完整病例分析'],
          explicit_missing_category: ['Declared missing categories retained', '保留已声明的未测量类别'],
          training_imputation: ['Imputation using training-set information only', '仅用训练集信息填补缺失'],
        }[row.missing];
        text = `${missing ? pick(missing) : tr('Missing-data handling to be confirmed', '缺失处理待确认')}${tr('; availability checked before the run', '；运行前核对缺失与可用性')}`;
      }
      // Summarize a long roster by its declared count, never slice a name,
      // threshold, unit or unresolved clause halfway through.
      if (text.length > (zh ? 40 : 96)) {
        const count = (row.variables || row.criteria || []).length;
        const compact = {
          population: tr(`Patients meeting ${count} declared criteria`, `符合 ${count} 项预设条件的患者`),
          comparison: row.mode === 'descriptive' ? tr('By the declared stratifier', '按预设变量分层')
            : row.mode === 'prediction' ? tr(`${count} declared predictors`, `${count} 项预设预测变量`)
              : tr('Declared exposure groups or levels', '预设的暴露分组或水平'),
          outcome: tr('Declared primary outcome', '已声明的主要结局'),
          adjustment: row.mode === 'prediction' ? text : tr(`${count} declared variables`, `${count} 项预设调整变量`),
        };
        text = compact[key] || text;
      }
      if (uncertain && !/待确认|to be confirmed/.test(text)) text += tr('; details to be confirmed', '，细节待确认');
      text = zh ? spacedZh(text) : text.replace(/^[a-z]/, letter => letter.toUpperCase());
      return { key, label: pick(title), text };
    });
    const timing = summary && summary.timing;
    let sentence = '';
    if (timing) {
      const origin = timing.anchor === 'icu_admit' ? tr('ICU admission', '入 ICU')
        : timing.anchor === 'hospital_admit' ? tr('hospital admission', '入院') : '';
      if (!timing.unresolved && origin && Number.isFinite(timing.hours) && timing.hours >= 0) {
        sentence = timing.hours === 0 ? tr(`Follow-up starts at ${origin}`, `从${origin}开始随访`)
          : tr(`Follow-up starts ${timing.hours} hours after ${origin}`, `从${origin} 后第 ${timing.hours} 小时开始随访`);
        if (timing.risk_set) sentence += tr(' and includes only patients still in hospital then', '，只纳入此时仍在院的患者');
        if (zh) sentence = spacedZh(sentence);
        sentence += tr('.', '。');
      } else sentence = tr('Follow-up start remains unresolved.', '随访起点待确认。');
    }
    return { lines, timing: sentence };
  }

  function stopAction(kind, tr) {
    const labels = {
      retry: ['Retry from the failed step', '从失败步骤重试'],
      edit_plan: ['Regenerate the plan', '重新生成计划'],
      plan_edit: ['Open plan changes', '打开计划修改'],
      reexport_data: ['Open data preparation', '打开数据准备'],
      open_provider_settings: ['Open model settings', '打开模型设置'],
      receipt: ['Open failure receipt', '打开失败回执'],
      review: ['Open recovery options', '打开恢复操作'],
    };
    const pair = labels[kind];
    return pair ? tr(pair[0], pair[1]) : '';
  }

  window.EU_PRODUCT_LABELS = Object.freeze({ copilotTitle, projectTitle, clinicalPlan, stopAction });
})();
