/* Screens: Get Started (tutorial) + AI Assistant.
   Get Started orients new users to the workflow; Assistant is the
   evidence-bound sidebar AI. Assistant messages give workflow guidance and
   plan structure only — never invented scientific results. */
(function () {
  const S = (window.SCREENS = window.SCREENS || {});

  /* ---------------- GET STARTED ---------------- */
  S.tutorial = {
    section: 'tutorial', nav: 'tutorial',
    crumbs: ['Home', 'Get Started'],
    actionHtml: `<button class="btn primary" data-nav="extraction">${icon('play', 13)} Start demo</button>`,
    rail() {
      return `
      <div class="rail-sep"></div>
      <div class="rail-block">
        <div class="rail-head"><span class="t">The workflow</span></div>
        <div class="col gap-6" style="font-size:12.5px;color:var(--ink-3);">
          ${[['Extract', 'extract', 'extraction'], ['Review', 'patient', 'patient'], ['Benchmark', 'benchmark', 'crossdb'], ['Agent', 'agent', 'agent']].map(([t, ic, nav]) =>
            `<a class="nav-item" data-nav="${nav}" style="height:30px;"><span class="ico">${icon(ic, 15)}</span>${t}</a>`).join('')}
        </div>
        <div class="eyebrow mt-16" style="margin-bottom:8px;">Shortcuts</div>
        <div class="col" style="font-size:12px;">
          <div class="shortcut-row"><span style="color:var(--ink-3);">Open Page guide</span><span class="keys"><span class="kbd">⌘</span><span class="kbd">K</span></span></div>
          <div class="shortcut-row"><span style="color:var(--ink-3);">Switch section</span><span class="keys"><span class="kbd">1</span>–<span class="kbd">5</span></span></div>
          <div class="shortcut-row"><span style="color:var(--ink-3);">Toggle language</span><span class="keys"><span class="kbd">L</span></span></div>
        </div>
      </div>`;
    },
    render() {
      return `
      <div class="page-head" style="margin-bottom:18px;">
        <div class="eyebrow">Get started · 快速上手</div>
        <h1 style="margin-top:6px;">A quiet, reviewable path from data to draft</h1>
        <p class="lead">EasyICU runs entirely on your machine. Start with reproducible demo data to learn the flow, then point it at local ICU exports when you’re ready.</p>
      </div>

      <div class="guide-hero">
        <div class="row" style="justify-content:space-between;align-items:flex-start;gap:20px;flex-wrap:wrap;">
          <div style="max-width:60ch;">
            <div class="row gap-10" style="align-items:center;">
              <div class="mode-mark accent" style="width:34px;height:34px;">${icon('flask', 17)}</div>
              <div style="font-size:17px;font-weight:600;letter-spacing:-0.01em;">New here? Take the 2-minute demo tour</div>
            </div>
            <p style="font-size:12.75px;color:var(--ink-3);line-height:1.6;margin:12px 0 0;">No tokens, no setup, no patient data. The demo generates 10 mock ICU stays so every screen, table, and review gate is fully explorable before you connect anything real.</p>
          </div>
          <div class="col gap-8" style="flex:none;">
            <button class="btn primary lg" data-nav="extraction">${icon('play', 15)} Start demo</button>
          </div>
        </div>
      </div>

      <div class="sec-stack"><div class="lbl">The four stages</div><h2>How a study moves through EasyICU</h2></div>
      <div class="card pad">
        <div class="guide-steps">
          ${[
            ['1', 'Frame the question', 'Pick a data mode — Demo generates reproducible mock data, Real connects a local export folder (nothing is ever uploaded) — then pin down what you’re asking: an outcome, a time window, and a comparator.', [['Configure source', 'extraction'], ['Talk it through', 'guided']], true],
            ['2', 'Extract & gate the data', 'One click runs the recommended extraction — cohort, coverage-audited feature modules, and a reproducible export. Need more control? Open Customize to set every detail. Nothing runs on incomplete data.', [['Open extraction', 'extraction']], false],
            ['3', 'Review & explore', 'Inspect patient-level tables and time series, step up to cohort contrasts and SOFA reclassification, then benchmark one cohort definition across two or more ICU databases.', [['Patient Review', 'patient'], ['Cohort Statistics', 'cohort'], ['Cross-DB Benchmark', 'crossdb']], false],
            ['4', 'Analyze & draft', 'Plan, run, and review an auditable pipeline with the Research Agent. The manuscript draft stays locked until every evidence check passes and you confirm.', [['Open Research Agent', 'agent'], ['Start Guided study', 'guided']], false],
          ].map(([n, t, d, actions, acc]) => `
            <div class="guide-step ${acc ? 'accent' : ''}">
              <div class="gs-num">${n}</div>
              <div>
                <div class="gs-t">${t}</div>
                <div class="gs-d">${d}</div>
                <div class="gs-actions">
                  ${actions.map(([lab, nav], i) => `<button class="btn ${i === 0 ? 'primary' : ''} sm" data-nav="${nav}">${lab} ${icon('arrow', 13)}</button>`).join('')}
                </div>
              </div>
            </div>`).join('')}
        </div>
      </div>

      <div class="sec-stack"><div class="lbl">Good to know</div><h2>Common questions</h2></div>
      <div class="faq">
        ${[
          ['Is any patient data uploaded?', 'No. EasyICU is local-first and the guarantee is enforced — extraction, review, and analysis all run on your machine. The only thing that can ever leave (and only if you explicitly enable it) is the Research Agent’s plan text, never patient rows.'],
          ['What exactly is Demo Mode?', 'A reproducible synthetic dataset — by default 10 ICU stays over 24 hours across 19 feature modules. It produces no scientific findings; every number is a seeded placeholder so you can learn the interface safely.'],
          ['Why is the manuscript draft locked?', 'Drafting is a deliberate second stage. The agent only writes after each evidence check passes — denominators resolved, coverage above threshold, tables reproducing from the manifest — and after a reviewer signs off. Claims stay traceable to logged artifacts.'],
          ['Which databases are supported?', 'Six standardized ICU sources: MIMIC-IV, eICU-CRD, AmsterdamUMCdb, HiRID, MIMIC-III, and SICdb. EasyICU detects known export layouts when you select a local folder.'],
          ['Do I need API tokens?', 'Not for Demo Mode or any extraction, review, or benchmark work — those never call a model. Tokens only apply if you connect an external model endpoint for the Research Agent.'],
        ].map(([q, a], i) => `
          <div class="faq-item ${i === 0 ? 'open' : ''}">
            <button class="faq-q">${q}<span class="fq-chev">${icon('chevron', 14)}</span></button>
            <div class="faq-a">${a}</div>
          </div>`).join('')}
      </div>`;
    },
    afterRender(root) {
      root.querySelectorAll('.faq-q').forEach(q => {
        q.addEventListener('click', () => q.closest('.faq-item').classList.toggle('open'));
      });
    },
  };

})();
