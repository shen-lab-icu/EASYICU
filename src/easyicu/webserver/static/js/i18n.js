/* Lightweight bilingual helper (EN / 中文).
   t('English', '中文') → returns the active language string.
   setLang('en'|'zh') persists + re-renders the whole shell. */
(function () {
  try { window.EU_LANG = localStorage.getItem('easyicu_lang') || 'en'; } catch (e) { window.EU_LANG = 'en'; }
  function validLang(l) { return l === 'en' || l === 'zh'; }
  function validMode(m) { return m === 'demo' || m === 'real'; }
  function applyLangDom(l) {
    document.documentElement.setAttribute('lang', l === 'zh' ? 'zh' : 'en');
    document.body && document.body.classList.toggle('lang-zh', l === 'zh');
  }
  function applyDisplayDom(settings) {
    settings = settings || {};
    const density = settings.density === 'compact' ? 'compact' : 'comfortable';
    if (document.body) {
      document.body.setAttribute('data-density', density);
      document.body.setAttribute('data-reduce-motion', settings.reduce_motion ? 'true' : 'false');
    }
  }
  window.applySettingsState = function (settings, opts) {
    settings = settings || {};
    opts = opts || {};
    const lang = validLang(settings.language) ? settings.language : (validLang(window.EU_LANG) ? window.EU_LANG : 'en');
    window.EU_LANG = lang;
    if (opts.syncStorage) {
      try { localStorage.setItem('easyicu_lang', lang); } catch (e) {}
    }
    const mode = validMode(settings.data_mode) ? settings.data_mode : (validMode(window.EU_DATA) ? window.EU_DATA : 'demo');
    window.EU_DATA = mode;
    if (opts.syncStorage) {
      try { localStorage.setItem('easyicu_home_data', mode); } catch (e) {}
    }
    applyLangDom(lang);
    applyDisplayDom(settings);
  };
  window.t = function (en, zh) {
    return window.EU_LANG === 'zh' ? (zh == null ? en : zh) : en;
  };
  window.setLang = function (l) {
    if (!validLang(l)) return;
    window.EU_LANG = l;
    try { localStorage.setItem('easyicu_lang', l); } catch (e) {}
    if (window.EU_SETTINGS) window.EU_SETTINGS.language = l;
    if (window.EU_API && window.EU_API.saveSetting) {
      window.EU_API.saveSetting('language', l).catch(function (err) {
        console.warn('[EasyICU] language setting sync failed', err);
      });
    }
    applyLangDom(l);
    if (window.__euRender) window.__euRender();
  };
  // apply on boot
  document.addEventListener('DOMContentLoaded', function () {
    applyLangDom(window.EU_LANG);
    applyDisplayDom(window.EU_SETTINGS || {});
  });

  /* ---- lightweight styled confirm modal (replaces native confirm) ---- */
  window.euConfirm = function (opts) {
    opts = opts || {};
    const ic = (n, s) => (window.icon ? window.icon(n, s || 18) : '');
    const ov = document.createElement('div');
    ov.className = 'eu-modal-ov';
    ov.innerHTML =
      '<div class="eu-modal" role="dialog" aria-modal="true">' +
        '<div class="em-ico ' + (opts.tone || '') + '">' + ic(opts.icon || 'refresh') + '</div>' +
        '<div class="em-t">' + (opts.title || '') + '</div>' +
        '<div class="em-d">' + (opts.body || '') + '</div>' +
        '<div class="em-actions">' +
          '<button class="btn sm" data-em-cancel>' + (opts.cancel || window.t('Cancel', '取消')) + '</button>' +
          '<button class="btn primary sm" data-em-ok>' + (opts.ok || window.t('Continue', '继续')) + '</button>' +
        '</div>' +
      '</div>';
    document.body.appendChild(ov);
    const close = () => ov.remove();
    ov.addEventListener('click', (e) => { if (e.target === ov) { close(); if (opts.oncancel) opts.oncancel(); } });
    ov.querySelector('[data-em-cancel]').addEventListener('click', () => { close(); if (opts.oncancel) opts.oncancel(); });
    ov.querySelector('[data-em-ok]').addEventListener('click', () => { close(); if (opts.onok) opts.onok(); });
    setTimeout(() => { const b = ov.querySelector('[data-em-ok]'); if (b) b.focus(); }, 30);
  };

  /* ---- global workspace mode: Demo / Real (cross-cutting state) ---- */
  try { window.EU_DATA = localStorage.getItem('easyicu_home_data') || 'demo'; } catch (e) { window.EU_DATA = 'demo'; }
  // becomes true once the user has produced downstream work worth protecting
  window.EU_HASWORK = false;
  window.setDataMode = function (m, opts) {
    if (m !== 'demo' && m !== 'real') return;
    if (window.EU_DATA === m) return;
    const apply = () => {
      window.EU_STALE = true;  // changing the source invalidates downstream
      window.EU_DATA = m;
      window.EU_VIZ_WORKSPACE = null;
      window.EU_CROSSDB_WORKSPACE = null;
      window.EU_PATIENT_DRILLDOWN = null;
      window.EU_COHORT_REVIEW = null;
      try { localStorage.setItem('easyicu_home_data', m); } catch (e) {}
      if (window.EU_SETTINGS) window.EU_SETTINGS.data_mode = m;
      if (window.EU_API && window.EU_API.saveSetting) {
        window.EU_API.saveSetting('data_mode', m).catch(function (err) {
          console.warn('[EasyICU] data mode setting sync failed', err);
        });
      }
      if (window.__euVizResetForDataMode) window.__euVizResetForDataMode();
      if (window.__euExtractReset) window.__euExtractReset();  // prev extraction belonged to the old source
      if (window.__euRender) window.__euRender();
    };
    if (window.EU_HASWORK && !(opts && opts.force) && window.euConfirm) {
      const to = m === 'real' ? window.t('Real', '真实') : window.t('Demo', '演示');
      window.euConfirm({
        icon: 'refresh', tone: 'warn',
        title: window.t('Switch data source?', '切换数据源?'),
        body: window.t(
          'Switching to <b>' + to + '</b> data marks your current cohort, extraction and review as <b>out&#8209;of&#8209;date</b> — you\u2019ll re-run them against the new source. Nothing already computed is deleted.',
          '切换到<b>' + to + '</b>数据会把当前的队列、抽取与审阅标记为<b>过期</b> —— 需基于新数据源重新运行。已计算的结果不会被删除。'),
        ok: window.t('Switch \u00b7 mark stale', '切换 · 标记过期'),
        onok: apply,
      });
    } else {
      apply();
    }
  };

  /* ---- downstream staleness: upstream edits invalidate review / analyze ---- */
  window.EU_STALE = false;
  window.setStale = function (b) {
    window.EU_STALE = !!b;
    if (window.__euRender) window.__euRender();
  };

  /* ---- clinical-term glossary (inline hover definitions) ---- */
  window.GLOSS = {
    'SOFA': ['Sequential Organ Failure Assessment — a 0–24 organ-dysfunction severity score.', 'SOFA(序贯器官衰竭评估):0–24 的器官功能障碍严重度评分。'],
    'Sepsis-3': ['2016 consensus: suspected infection + organ dysfunction (SOFA rise ≥ 2).', 'Sepsis-3(2016 共识):疑似感染 + 器官功能障碍(SOFA 升高 ≥ 2)。'],
    'cohort': ['The set of ICU stays your study includes — it defines your denominator.', '队列:研究纳入的 ICU 住院集合,决定你的分母。'],
    'denominator': ['The population every rate and p-value is computed against.', '分母:所有比率与 p 值据以计算的总体。'],
    'concept': ['A standardized clinical variable (e.g. lactate), mapped consistently across databases.', '概念:跨库一致映射的标准化临床变量(如乳酸)。'],
    'KDIGO': ['Kidney Disease: Improving Global Outcomes — the staging criteria for AKI.', 'KDIGO:急性肾损伤(AKI)的分期标准。'],
    'AKI': ['Acute kidney injury — a sudden drop in kidney function, staged by KDIGO.', 'AKI(急性肾损伤):肾功能急剧下降,按 KDIGO 分期。'],
    'RRT': ['Renal replacement therapy — dialysis or related kidney support.', 'RRT(肾脏替代治疗):透析等肾脏支持。'],
    'LOS': ['Length of stay — time the patient is admitted in the ICU.', 'LOS(住院时长):患者在 ICU 的停留时间。'],
  };
  window.gloss = function (term, shown) {
    const g = window.GLOSS[term];
    const label = shown || term;
    if (!g) return label;
    const tip = (window.EU_LANG === 'zh' ? g[1] : g[0]).replace(/"/g, '&quot;');
    return '<span class="term" tabindex="0" data-tip="' + tip + '">' + label + '</span>';
  };

  /* ---- study depth / goal: where the study STOPS (mirrors copilot DEPTH axis) ---- */
  try { window.EU_GOAL = localStorage.getItem('easyicu_goal') || 'full'; } catch (e) { window.EU_GOAL = 'full'; }
  window.goalPhase = function () { return { extract: 1, review: 2, full: 3 }[window.EU_GOAL] || 3; };
  window.goalLabel = function () {
    return {
      extract: window.t('Extract only', '仅抽取'),
      review: window.t('Extract + review', '抽取 + 审阅'),
      full: window.t('Full study', '完整研究'),
    }[window.EU_GOAL] || window.t('Full study', '完整研究');
  };
  window.cycleGoal = function () {
    const order = ['extract', 'review', 'full'];
    window.EU_GOAL = order[(order.indexOf(window.EU_GOAL) + 1) % order.length];
    try { localStorage.setItem('easyicu_goal', window.EU_GOAL); } catch (e) {}
    if (window.__euRender) window.__euRender();
  };
})();
