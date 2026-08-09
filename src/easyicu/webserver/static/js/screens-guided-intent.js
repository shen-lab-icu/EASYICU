/* EasyICU — Guided Copilot: typed study-intent owner.

   Owns ONE thing: turning the user's own sentence into a typed study-contract
   proposal (via POST /api/copilot/study-intent) and rendering it, including the
   slots the backend could NOT read.

   Why this is a separate owner file: `screens-guided.js` is already well past
   the per-module budget in CLAUDE.md, and this is a distinct seam (contract
   reading + contract card) rather than more conversation script. It shares no
   mutable state with the guided IIFE — the guided screen calls in, gets an
   immutable result object back, and owns its own rendering decision.

   Contract with callers:
     EU_STUDY_INTENT.extract(question, opts) -> Promise<contract|null>
     EU_STUDY_INTENT.cardHtml(contract, opts) -> string
     EU_STUDY_INTENT.isComplete(contract) -> boolean
   `null` means "could not read" — never a fabricated contract. */
(function () {
  'use strict';

  window.EU_STUDY_INTENT = window.EU_STUDY_INTENT || {};
  const NS = window.EU_STUDY_INTENT;

  const SLOT_LABELS = {
    population: ['Population', '人群'],
    exposure: ['Exposure', '暴露'],
    outcome: ['Outcome', '结局'],
    outcome_type: ['Outcome type', '结局类型'],
    time_window_hours: ['Time window', '时间窗'],
    comparator: ['Comparator', '对照'],
    analysis_family: ['Analysis', '分析族'],
  };
  const SLOT_ORDER = ['population', 'exposure', 'outcome', 'outcome_type', 'time_window_hours', 'comparator', 'analysis_family'];

  function tx(en, zh) { return window.t ? window.t(en, zh) : en; }
  function esc(v) {
    return String(v == null ? '' : v).replace(/[&<>"']/g, c => (
      { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]
    ));
  }
  function label(slot) {
    const pair = SLOT_LABELS[slot] || [slot, slot];
    return tx(pair[0], pair[1]);
  }
  function displayValue(slot, value) {
    if (value == null || value === '') return '';
    if (slot === 'time_window_hours') return tx(`first ${value}h`, `前 ${value} 小时`);
    return String(value);
  }

  /* Ask the backend to read the question. Resolves to null on any failure —
     callers must treat that as "unknown", never as an empty contract. */
  NS.extract = async function extract(question, opts) {
    const text = String(question == null ? '' : question).trim();
    if (!text) return null;
    const o = opts || {};
    try {
      const res = await fetch('/api/copilot/study-intent', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', Accept: 'application/json' },
        body: JSON.stringify({
          question: text,
          llm_provider: o.llm_provider || 'offline',
          external_llm_opt_in: !!o.external_llm_opt_in,
          language: window.EU_LANG || 'en',
        }),
      });
      if (!res.ok) return null;
      const data = await res.json();
      return data && data.ok ? data : null;
    } catch (e) {
      return null;
    }
  };

  NS.isComplete = function isComplete(contract) {
    return !!(contract && contract.complete);
  };

  NS.unreadSlots = function unreadSlots(contract) {
    return (contract && Array.isArray(contract.unread)) ? contract.unread.slice() : [];
  };

  /* One card: what we read, from which words, and what we did NOT read.
     The unread block is the point of the whole module — it is the difference
     between asking the user and silently defaulting. */
  NS.cardHtml = function cardHtml(contract, opts) {
    if (!contract) return '';
    const o = opts || {};
    const slots = contract.slots || {};
    const read = SLOT_ORDER.filter(s => slots[s] && slots[s].value != null && slots[s].value !== '');
    const unread = NS.unreadSlots(contract);

    const readRows = read.map(s => {
      const ev = slots[s].evidence;
      return `<div class="setup-row">
        <span class="k">${esc(label(s))}</span>
        <span class="vv">${esc(displayValue(s, slots[s].value))}${ev ? ` <span class="mono" style="color:var(--ink-4);font-size:10.5px;">← “${esc(ev)}”</span>` : ''}</span>
      </div>`;
    }).join('');

    const unreadRows = unread.length
      ? `<div class="note warn mt-12" style="padding:9px 11px;">
           <div class="ico">${o.icon ? o.icon('alert', 13) : ''}</div>
           <div class="body"><div class="d" style="font-size:11px;margin:0;">
             ${esc(tx('Not stated in your question — I will not fill these in for you:', '你的问题里没有说明,我不会替你填这些:'))}
             <strong>${unread.map(s => esc(label(s))).join(' · ')}</strong>.
             ${esc(tx('Add them in a sentence, or set them in the next steps.', '可以再补一句话,或在后续步骤里设置。'))}
           </div></div>
         </div>`
      : '';

    const sourceNote = contract.source === 'llm'
      ? tx('read by the language model, validated against the typed contract', '由语言模型读取,已按类型化合同校验')
      : tx('read locally — no model call', '本地读取 · 未调用模型');

    return `
      <div class="eyebrow" style="margin:0 0 6px;">${esc(tx('What I read from your question', '我从你的问题里读到的'))} <span class="mono" style="color:var(--ink-4);">${contract.read_count}/${contract.slot_count}</span></div>
      <div class="col gap-6" style="font-size:12.25px;">${readRows || `<div class="setup-row"><span class="k">—</span><span class="vv">${esc(tx('nothing readable yet', '暂时没读到可用信息'))}</span></div>`}</div>
      ${unreadRows}
      <div class="m-cite" style="margin-top:11px;">${esc(sourceNote)}</div>`;
  };
})();
