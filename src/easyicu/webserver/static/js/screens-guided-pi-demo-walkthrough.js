/* Guided Pi workflow-demo walkthrough.
   Owner: replaying the registered demo run one beat at a time. The demo
   owner supplies the run's six chapters; this owner reveals them in order
   with each activity row's recorded duration, stops at every traceability
   card (screens-guided-pi-demo-trace.js), waits for the reader's own plan
   confirmation, and keeps the reader's place. It never calls a model and
   never touches a project: the whole replay is the read-only demo record. */
(function () {
  'use strict';

  const { esc } = window.EU_HTML;
  const TRACE_URL = 'assets/demo/sofa2-run-trace.json?v=20261026-fig1';
  // Replay pacing (ms). A row whose recorded time is long holds a little
  // longer so the difference is felt; the row itself shows the real time.
  const PACE = { message: 900, user: 600, step: 650, longStep: 1500, card: 700, chapter: 500 };
  const LONG_STEP_MS = 20000;

  const walk = {
    trace: null, status: 'idle', chapters: [], chapter: 0, beat: 0, steps: 0,
    playing: false, waiting: '', confirmed: false, skipped: false,
    selection: { target: 'cell:0:rate' }, asked: -1, timer: 0, token: 0, deps: null,
  };
  // The HTML each revealed beat was last drawn with, so a refresh redraws
  // only beats whose content changed.
  const drawn = new Map();
  // Set when the reader asks to go on, so the log follows the new content
  // even if they had scrolled back to an earlier card.
  let followNext = false;

  function tr(en, zh) { return window.EU_LANG === 'zh' ? zh : en; }
  function reducedMotion() {
    return typeof window.matchMedia === 'function' && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  }
  function modules() { return window.EasyICU.guidedPi; }
  function demo() { return modules().optional('demo'); }
  function root() { return document.querySelector('[data-gpi-walk-root]'); }
  function clearTimer() { if (walk.timer) clearTimeout(walk.timer); walk.timer = 0; }

  function load() {
    if (walk.status === 'ready' || walk.status === 'loading') return;
    walk.status = 'loading';
    const token = walk.token;
    const done = trace => {
      if (token !== walk.token) return;
      walk.trace = trace;
      walk.status = trace ? 'ready' : 'failed';
      walk.chapters = demo().chapters(trace);
      walk.language = window.EU_LANG;
      refresh();
      schedule(PACE.chapter);
    };
    if (typeof fetch !== 'function') { done(null); return; }
    fetch(TRACE_URL, { credentials: 'same-origin' })
      .then(response => (response.ok ? response.json() : null))
      .then(trace => done(trace && trace.schema_version === 'easyicu.demo-walkthrough-trace/1' ? trace : null))
      .catch(() => done(null));
  }
  function start() {
    clearTimer();
    walk.token += 1;
    Object.assign(walk, {
      chapter: 0, beat: 0, steps: 0, playing: true, waiting: '', confirmed: false,
      skipped: false, selection: { target: 'cell:0:rate' }, asked: -1,
    });
    drawn.clear();
    if (walk.status === 'ready' || walk.status === 'failed') {
      walk.chapters = demo().chapters(walk.trace);
      walk.language = window.EU_LANG;
      schedule(PACE.chapter);
    } else {
      walk.status = 'idle';
      load();
    }
  }
  function stop() { clearTimer(); walk.token += 1; walk.playing = false; }

  // ---- position --------------------------------------------------------
  function current() {
    const chapter = walk.chapters[walk.chapter];
    return chapter ? chapter.beats[walk.beat] : null;
  }
  function finished() { return walk.chapters.length > 0 && walk.chapter >= walk.chapters.length; }
  function advanceBeat() {
    walk.steps = 0;
    walk.beat += 1;
    const chapter = walk.chapters[walk.chapter];
    if (chapter && walk.beat >= chapter.beats.length) { walk.chapter += 1; walk.beat = 0; return true; }
    return false;
  }
  // One replay step; returns the delay before the next, or null to wait.
  function stepOnce() {
    const beat = current();
    if (!beat) { walk.playing = false; return null; }
    if (beat.role === 'activity') {
      const total = beat.steps.length;
      if (walk.steps < total && !reducedMotion()) {
        walk.steps += 1;
        const row = beat.steps[walk.steps - 1];
        const long = row.durationKnown !== false && row.endedAt - row.startedAt > LONG_STEP_MS;
        return walk.steps < total ? (long ? PACE.longStep : PACE.step) : PACE.step;
      }
      return advanceBeat() ? PACE.chapter : PACE.message;
    }
    if (beat.card && beat.stop && walk.waiting !== beat.card) {
      walk.waiting = beat.card;
      walk.playing = false;
      // The last stop ends the replay; nothing waits behind it.
      if (beat.stop === 'end') { walk.waiting = ''; advanceBeat(); }
      return null;
    }
    walk.waiting = '';
    const changedChapter = advanceBeat();
    if (finished()) { walk.playing = false; return null; }
    const next = current();
    if (changedChapter) return PACE.chapter;
    return next && next.role === 'user' ? PACE.user : next && next.card ? PACE.card : PACE.message;
  }
  function schedule(delay) {
    clearTimer();
    if (!walk.playing || delay == null) return;
    const token = walk.token;
    walk.timer = setTimeout(() => {
      if (token !== walk.token) return;
      if (!root()) { stop(); return; }
      const next = stepOnce();
      refresh();
      schedule(next);
    }, reducedMotion() ? Math.min(delay, 250) : delay);
  }
  // Reveal everything; the confirmation a reader skips is recorded as skipped.
  function skipToEnd() {
    clearTimer();
    if (!walk.confirmed) walk.skipped = true;
    walk.confirmed = true;
    walk.chapter = walk.chapters.length;
    walk.beat = 0;
    walk.steps = 0;
    walk.waiting = '';
    walk.playing = false;
    refresh();
  }

  // ---- rendering -------------------------------------------------------
  function cards() {
    const deps = walk.deps;
    return modules().require('demoTrace').create({ tr, icon: deps.iconHtml, button: deps.button });
  }
  function resource(name) {
    const owner = demo();
    const rows = owner && typeof owner.standardResources === 'function' ? owner.standardResources() : [];
    return rows.find(row => row.artifact === name) || null;
  }
  function runningActivity(beat, shown) {
    const durationText = walk.deps.durationText;
    const rows = beat.steps.slice(0, shown).map((row, index) => {
      const live = index === shown - 1;
      const time = row.durationKnown === false || !durationText ? '' : durationText(row.startedAt, row.endedAt);
      return `<li class="${live ? 'is-live' : 'is-done'}">${live ? '<span class="gpi-running-spinner" aria-hidden="true"></span>' : walk.deps.iconHtml('check', 13)}<span>${esc(row.label)}</span>${time ? `<em>${esc(time)}</em>` : ''}</li>`;
    }).join('');
    return `<div class="gpi-walk-running" role="status" aria-live="polite"><div class="gpi-walk-running-head"><span class="gpi-running-spinner" aria-hidden="true"></span><strong>${esc(beat.displayTitle || tr('Working', '处理中'))}</strong><span>${esc(tr('replay · recorded times', '回放 · 真实耗时'))}</span></div><ol>${rows}</ol></div>`;
  }
  function cardHtml(beat) {
    const trace = walk.trace;
    const owner = demo();
    if (beat.card === 'confirm') {
      if (walk.confirmed) return '';
      return cards().confirmCard(trace, walk.deps.durationText || (() => ''));
    }
    if (beat.card === 'followups') {
      return `${cards().followUpsCard(owner.followUps(), walk.asked)}${owner.governanceHtml({ tr, esc, button: walk.deps.button })}`;
    }
    // Without the trace record only the conversation replays; the result
    // keeps the demo owner's own figure and table.
    if (!trace) return beat.card === 'result' ? owner.resultHtml({ tr, esc, button: walk.deps.button, figureOnly: true }) : '';
    if (beat.card === 'dictionary') return cards().dictionaryCard(trace);
    if (beat.card === 'literature') return cards().literatureCard(trace, { resource });
    if (beat.card === 'result') return cards().resultCard(trace, walk.selection, owner);
    if (beat.card === 'boundary') return cards().boundaryCard(trace, owner);
    return '';
  }
  function beatHtml(beat, partialSteps) {
    if (beat.card) return cardHtml(beat);
    if (beat.role === 'activity' && partialSteps != null) return runningActivity(beat, partialSteps);
    if (beat.role === 'demo_action' && walk.skipped) {
      return walk.deps.messageHtml({ ...beat, label: tr('Plan confirmation skipped in this demo', '演示中跳过了确认'), note: tr('In your own project the analysis waits for this control.', '在你自己的项目里，分析会一直等你按下确认。') }, { interactive: false });
    }
    return walk.deps.messageHtml(beat, { interactive: false });
  }
  // The revealed beats, keyed so a refresh replaces only what changed and a
  // reader's open <details> elsewhere stay open.
  function revealed() {
    const out = [];
    walk.chapters.forEach((chapter, c) => {
      chapter.beats.forEach((beat, b) => {
        const before = c < walk.chapter || (c === walk.chapter && b < walk.beat);
        const here = c === walk.chapter && b === walk.beat;
        if (before) {
          const html = beatHtml(beat);
          if (html) out.push({ key: `${c}.${b}`, html });
        }
        else if (here && beat.role === 'activity' && walk.steps > 0) out.push({ key: `${c}.${b}`, html: beatHtml(beat, walk.steps) });
        else if (here && beat.card && walk.waiting === beat.card) out.push({ key: `${c}.${b}`, html: beatHtml(beat) });
      });
    });
    return out;
  }
  function logHtml() {
    if (walk.status === 'loading' || walk.status === 'idle') return '';
    return revealed().map(row => {
      drawn.set(row.key, row.html);
      return `<div class="gpi-walk-beat" data-gpi-walk-beat="${row.key}">${row.html}</div>`;
    }).join('');
  }
  function chapterLabel() {
    const total = walk.chapters.length;
    if (!total) return '';
    if (finished()) return tr('Replay complete', '回放结束');
    return `${walk.chapter + 1} / ${total} · ${walk.chapters[walk.chapter].label}`;
  }
  function controlsHtml() {
    const end = finished();
    const confirmWait = walk.waiting === 'confirm';
    const play = end
      ? `<button class="btn sm" type="button" data-gpi-walk="restart">${esc(tr('Replay again', '重新播放'))}</button>`
      : confirmWait
        ? `<span class="gpi-walk-hint">${esc(tr('Confirm the plan above to continue', '在上方确认方案后继续'))}</span>`
        : `<button class="btn sm" type="button" data-gpi-walk="${walk.playing ? 'pause' : 'play'}">${esc(walk.playing ? tr('Pause', '暂停') : tr('Continue', '继续'))}</button>`;
    const skip = end ? '' : `<button class="gpi-link" type="button" data-gpi-walk="skip">${esc(tr('Skip to the result', '跳到结果'))}</button>`;
    return `<div class="gpi-walk-controls" data-gpi-walk-controls>
      <span class="gpi-walk-chapter">${esc(chapterLabel())}</span>
      <div class="gpi-walk-actions">${skip}${play}${end ? `<button class="btn primary" type="button" data-gpi-demo-exit>${esc(tr('Start my own research', '开始我自己的研究'))}</button>` : ''}</div>
    </div>`;
  }
  // The demo's research-workflow strip follows the replay: a stage is done
  // once the chapters that complete it have played.
  const STAGE_CHAPTER = { question: 0, setup: 1, extraction: 1, idea: 2, plan: 3, analysis: 4, interpretation: 5, manuscript: 5 };
  function workflow() {
    const base = demo().workflow();
    const playedThrough = finished() ? walk.chapters.length : walk.chapter;
    let done = 0;
    const stages = base.stages.map(stage => {
      const chapter = STAGE_CHAPTER[stage.id];
      const reached = chapter != null && chapter < playedThrough;
      if (reached && stage.status === 'complete') done += 1;
      const status = walk.chapters[walk.chapter] && walk.chapters[walk.chapter].id === 'confirm' ? 'review_required' : 'running';
      // A stage not yet replayed carries no recorded outcome.
      return reached ? stage : { id: stage.id, status: chapter === walk.chapter ? status : 'pending', reason_code: '' };
    });
    const currentStage = stages.find(stage => stage.status === 'running' || stage.status === 'review_required');
    return { ...base, stages, completed_required_stages: done, current_stage: currentStage ? currentStage.id : base.current_stage };
  }
  function panelHtml(deps) {
    walk.deps = deps;
    if (walk.status === 'idle' && !walk.chapters.length) load();
    // The chapters carry reader text; a language switch rebuilds them in
    // place. Their shape does not depend on the language, so the place holds.
    if (walk.chapters.length && walk.language !== window.EU_LANG) {
      walk.chapters = demo().chapters(walk.trace);
      drawn.clear();
    }
    walk.language = window.EU_LANG;
    return `<div class="gpi-log gpi-walk-log" data-gpi-log data-gpi-walk-log>${logHtml()}</div>
      <footer class="gpi-demo-footer gpi-walk-footer">${controlsHtml()}</footer>`;
  }

  function nearBottom(log) { return log.scrollHeight - log.scrollTop - log.clientHeight < 96; }
  function refresh() {
    const host = root();
    if (!host || !walk.deps) return;
    const log = host.querySelector('[data-gpi-walk-log]');
    if (log) {
      const follow = followNext || nearBottom(log);
      followNext = false;
      const existing = new Map([...log.querySelectorAll(':scope > [data-gpi-walk-beat]')].map(node => [node.dataset.gpiWalkBeat, node]));
      const wanted = revealed();
      const keep = new Set(wanted.map(row => row.key));
      existing.forEach((node, key) => { if (!keep.has(key)) { node.remove(); drawn.delete(key); } });
      wanted.forEach(row => {
        const node = existing.get(row.key);
        if (node && drawn.get(row.key) === row.html) return;
        const next = document.createElement('div');
        next.className = 'gpi-walk-beat';
        next.dataset.gpiWalkBeat = row.key;
        drawn.set(row.key, row.html);
        next.innerHTML = row.html;
        if (node) node.replaceWith(next); else log.appendChild(next);
      });
      if (follow) log.scrollTop = log.scrollHeight;
    }
    const controls = host.querySelector('[data-gpi-walk-controls]');
    if (controls) controls.outerHTML = controlsHtml();
    const strip = host.querySelector('[data-gpi-walk-workflow]');
    const stage = finished() ? -1 : walk.chapter;
    if (strip && walk.deps.workflowHtml && strip.dataset.gpiWalkStage !== String(stage)) {
      strip.dataset.gpiWalkStage = String(stage);
      strip.innerHTML = walk.deps.workflowHtml(workflow());
      // The right rail's to-do list follows the same chapters.
      if (typeof walk.deps.syncAside === 'function') walk.deps.syncAside();
    }
  }
  // A card that changes in place re-renders alone.
  function refreshCard(kind) {
    const host = root();
    const node = host && host.querySelector(`[data-gpi-trace-card="${kind}"]`);
    const beatNode = node && node.closest('[data-gpi-walk-beat]');
    if (!beatNode) { refresh(); return; }
    const [c, b] = beatNode.dataset.gpiWalkBeat.split('.').map(Number);
    const beat = walk.chapters[c] && walk.chapters[c].beats[b];
    if (!beat) return;
    const html = beatHtml(beat);
    drawn.set(beatNode.dataset.gpiWalkBeat, html);
    beatNode.innerHTML = html;
  }

  function handleClick(event) {
    const control = event.target.closest('[data-gpi-walk]');
    if (!control || !root() || !root().contains(control)) return false;
    event.preventDefault();
    const action = control.dataset.gpiWalk;
    if (action === 'pause') { walk.playing = false; clearTimer(); refresh(); return true; }
    if (action === 'play') {
      walk.playing = true;
      followNext = true;
      const next = stepOnce();
      refresh();
      schedule(next == null ? null : next);
      return true;
    }
    if (action === 'skip') { skipToEnd(); return true; }
    if (action === 'restart') { start(); refresh(); return true; }
    if (action === 'confirm') {
      walk.confirmed = true;
      followNext = true;
      walk.waiting = '';
      walk.playing = true;
      advanceBeat();
      refresh();
      schedule(PACE.message);
      return true;
    }
    if (action === 'select') {
      walk.selection = { target: control.dataset.gpiWalkTarget || 'cell:0:rate' };
      refreshCard('result');
      return true;
    }
    if (action === 'followup') {
      const index = Number(control.dataset.gpiWalkIndex);
      walk.asked = walk.asked === index ? -1 : index;
      refreshCard('followups');
      return true;
    }
    if (action === 'dictionary') {
      const card = root().querySelector('[data-gpi-trace-card="dictionary"]');
      if (card) {
        card.querySelectorAll('.gpi-dict-term > details').forEach((node, index) => { if (index < 2) node.open = true; });
        card.scrollIntoView({ behavior: reducedMotion() ? 'auto' : 'smooth', block: 'start' });
        card.classList.remove('is-flash');
        void card.offsetWidth;
        card.classList.add('is-flash');
      }
      return true;
    }
    return false;
  }

  window.EasyICU.guidedPi.declare('demoWalkthrough', {
    start, stop, panelHtml, workflow, handleClick,
    state: () => ({ status: walk.status, chapter: walk.chapter, beat: walk.beat, steps: walk.steps, playing: walk.playing, waiting: walk.waiting, confirmed: walk.confirmed, skipped: walk.skipped, finished: finished() }),
  });
})();
