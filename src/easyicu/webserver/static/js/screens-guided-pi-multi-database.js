/* Owner: multi-database batch cards in the Guided Copilot conversation.
   A batch runs one reviewed study on several registered databases
   (webserver/research_multi_*.py). This owner reads each batch's scoped status,
   shows the combined review card with every database's disposition, sends the
   displayed card digest on the user's own click, and shows each database's
   sealed primary estimate side by side. It never pools estimates and never
   decides a substitution. */
(function () {
  'use strict';

  const POLL_MS = 5000;
  const SCALES = {
    odds_ratio: ['Odds ratio', '比值比（OR）'], risk_ratio: ['Risk ratio', '风险比（RR）'], hazard_ratio: ['Hazard ratio', '风险比（HR）'],
    risk_difference: ['Risk difference', '风险差'], mean_difference: ['Mean difference', '均值差'], beta: ['Coefficient', '回归系数'],
  };
  const STATES = {
    pending: ['Waiting', '等待运行'], running: ['Running', '运行中'], completed: ['Done', '已完成'],
    failed: ['Did not complete', '未完成'], cancelled: ['Cancelled', '已取消'], not_run: ['Not run', '不运行'],
  };
  const REASONS = {
    multi_database_cancelled: ['Cancelled.', '已取消。'],
    multi_database_source_unavailable: ['No registered data package for this database.', '这个数据库没有已登记的数据包。'],
    multi_database_source_ambiguous: ['More than one data package could serve; choose one in Data.', '有多个数据包可用，请先在数据页选定一个。'],
    multi_database_opt_in_required: ['This database needs a qualifying model connection first.', '这个数据库需要先连接符合要求的模型。'],
    multi_database_design_unassessed: ['The study design has not been assessed for this database.', '研究设计在这个数据库上还没有评估。'],
    multi_database_design_not_recompilable: ['The study cannot be rebuilt on this database.', '这项研究无法在这个数据库上重建。'],
    multi_database_primary_estimate_unavailable: ['The run finished without a primary estimate to compare.', '运行完成，但没有可比较的主要估计。'],
    multi_database_execution_failed: ['The run did not complete.', '运行没有完成。'],
    multi_database_execution_interrupted: ['The run was interrupted.', '运行被中断。'],
    multi_database_result_changed: ['The result no longer matches its record, so it is not shown.', '结果与记录不一致，因此不显示。'],
    multi_database_estimates_incompatible: ['The estimates measure different things, so they are not compared.', '各库估计的含义不同，因此不作比较。'],
  };

  function create(deps) {
    const { tr, esc, api, session, projectId, disabled, repaint, errorText } = deps;
    const statuses = new Map();
    let timer = null;
    const pair = (map, key) => { const row = map[String(key || '')]; return row ? tr(row[0], row[1]) : ''; };
    const adjustments = () => window.EasyICU.guidedPi.optional('sourceAdjustments');
    function databaseName(key) { return (window.EU_DATABASE_NAMES && window.EU_DATABASE_NAMES.name(String(key || ''))) || String(key || ''); }
    function reasonText(code) {
      const key = String(code || '');
      if (REASONS[key]) return tr(REASONS[key][0], REASONS[key][1]);
      const owner = adjustments();
      const missing = owner && typeof owner.missingText === 'function' ? owner.missingText({ code: key }) : '';
      return missing ? tr(`It ${missing}.`, `${missing}。`) : '';
    }
    function number(value) {
      return Number(value).toLocaleString(window.EU_LANG === 'zh' ? 'zh-CN' : 'en-US', { maximumFractionDigits: 2, minimumFractionDigits: 2 });
    }
    function conceptName(concept) {
      const row = (window.EU_CATALOG && window.EU_CATALOG.dict || {})[concept];
      return row ? tr(row[0], row[1]) : String(concept || '');
    }

    function key(sessionRow) { return `${projectId()}|${sessionRow && sessionRow.session_id}`; }
    function references(sessionRow) {
      const list = sessionRow && sessionRow.conversation_replay && sessionRow.conversation_replay.multi_database_batches;
      return Array.isArray(list) ? list.filter(row => row && row.batch_id) : [];
    }
    async function load(batchId, owner) {
      const current = session();
      if (!current || key(current) !== owner) return;
      const entry = statuses.get(batchId) || {};
      if (entry.loading) return;
      statuses.set(batchId, { ...entry, loading: true });
      try {
        const payload = await api().getPiMultiDatabase(batchId, projectId(), current.session_id);
        if (key(session()) !== owner) return;
        statuses.set(batchId, { payload, loading: false, at: Date.now(), owner });
      } catch (error) {
        statuses.set(batchId, { ...entry, loading: false, error: errorText(error), at: Date.now(), owner });
      }
      repaint();
    }
    const settled = payload => payload && (payload.status === 'completed' || (payload.status === 'review' && payload.card));
    function refresh(sessionRow) {
      const owner = key(sessionRow);
      let pending = false;
      references(sessionRow).forEach(row => {
        const entry = statuses.get(row.batch_id);
        if (!entry || entry.owner !== owner) { void load(row.batch_id, owner); pending = true; return; }
        if (!settled(entry.payload)) {
          pending = true;
          if (!entry.loading && Date.now() - (entry.at || 0) >= POLL_MS) void load(row.batch_id, owner);
        }
      });
      if (pending && !timer) timer = setTimeout(() => { timer = null; if (session()) repaint(); }, POLL_MS);
    }

    function timeMs(value) { const at = Date.parse(String(value || '')); return Number.isFinite(at) ? at : 0; }
    function rows(sessionRow) {
      refresh(sessionRow);
      const turns = Array.isArray(sessionRow && sessionRow.conversation_replay && sessionRow.conversation_replay.turns)
        ? sessionRow.conversation_replay.turns : [];
      return references(sessionRow).map(reference => {
        const turn = reference.message_job_id && turns.find(row => row && row.job_id === reference.message_job_id);
        const at = turn ? timeMs(turn.ended_at || turn.started_at) + 1 : timeMs(reference.created_at);
        return { role: 'multi_database', id: 'multi-database-' + reference.batch_id, reference, timelineAt: at };
      });
    }
    function place(timeline, cards) {
      const list = Array.isArray(timeline) ? timeline.slice() : [];
      cards.slice().sort((left, right) => left.timelineAt - right.timelineAt).forEach(card => {
        const index = card.timelineAt ? list.findIndex(row => {
          const at = Number(row && (row.timelineAt || row.startedAt) || 0);
          return at && at > card.timelineAt;
        }) : -1;
        if (index >= 0) list.splice(index, 0, card); else list.push(card);
      });
      return list;
    }

    function estimateText(primary) {
      if (!primary) return '';
      const interval = `${number(primary.ci_low)}–${number(primary.ci_high)}`;
      return `${esc(pair(SCALES, primary.unit) || primary.unit)} ${number(primary.estimate)}${tr(` (${interval})`, `（${interval}）`)} · n = ${Number(primary.population_n).toLocaleString()}`;
    }
    function reviewTarget(target) {
      const name = esc(databaseName(target.database));
      if (target.disposition === 'not_run') {
        const reasons = [...new Set((target.reason_codes || []).map(reasonText).filter(Boolean))];
        return `<li class="is-not-run"><strong>${name}</strong><span>${esc(pair(STATES, 'not_run'))}</span>${reasons.length ? `<p>${esc(reasons.join(' '))}</p>` : ''}</li>`;
      }
      const owner = adjustments();
      const items = owner && typeof owner.items === 'function'
        ? owner.items({ status: 'substitution', database: target.database, substitutions: target.substitutions || [] }) : [];
      const binned = (target.definition_codes || []).filter(code => String(code).startsWith('binned_covariate:'))
        .map(code => conceptName(String(code).slice('binned_covariate:'.length)));
      const notes = binned.length ? `<p>${esc(tr(`Recorded in bands: ${binned.join(', ')}.`, `按区间记录：${binned.join('、')}。`))}</p>` : '';
      const label = target.disposition === 'adjusted' ? tr('With changed definitions', '口径有调整') : tr('Exact', '精确对应');
      return `<li class="${target.disposition === 'adjusted' ? 'is-adjusted' : ''}"><strong>${name}</strong><span>${esc(label)}</span>${items.length ? `<ul class="gpi-multi-swaps">${items.join('')}</ul>` : ''}${notes}</li>`;
    }
    function progressTarget(card, row, batchId, canAct) {
      const name = esc(databaseName(row.database));
      const state = String(row.state || '');
      const result = row.primary ? `<b>${estimateText(row.primary)}</b>` : '';
      const reason = row.reason_code && state !== 'completed' ? reasonText(row.reason_code)
        : row.primary_state === 'unavailable' ? reasonText(row.reason_code) : '';
      const cancel = canAct && (state === 'pending' || state === 'running')
        ? `<button class="btn sm" type="button" data-gpi-multi-cancel="${esc(batchId)}" data-gpi-multi-database="${esc(row.database)}">${esc(tr('Cancel', '取消'))}</button>` : '';
      return `<li class="is-${esc(state)}"><strong>${name}</strong><span>${esc(pair(STATES, state) || state)}</span>${result}${reason ? `<p>${esc(reason)}</p>` : ''}${cancel}</li>`;
    }
    function comparisonHtml(answer) {
      const comparison = answer && answer.heterogeneity;
      if (!comparison) return '';
      if (comparison.reason_code) {
        const text = comparison.reason_code === 'multi_database_comparison_insufficient' ? '' : reasonText(comparison.reason_code);
        return text ? `<p class="gpi-multi-note">${esc(text)}</p>` : '';
      }
      const agree = comparison.description_code === 'directions_agree';
      return `<p class="gpi-multi-note">${esc(tr(
        `Across ${comparison.n_databases} databases the estimates range from ${number(comparison.minimum)} to ${number(comparison.maximum)}; they ${agree ? 'all lie on the same side' : 'lie on different sides'} of ${number(comparison.null_value)}.`,
        `${comparison.n_databases} 个数据库的估计在 ${number(comparison.minimum)} 至 ${number(comparison.maximum)} 之间，${agree ? '都在' : '分布在'} ${number(comparison.null_value)} 的${agree ? '同一侧' : '两侧'}。`))}</p>`;
    }

    function render(row) {
      const batchId = String(row && row.reference && row.reference.batch_id || '');
      const entry = statuses.get(batchId) || {};
      const payload = entry.payload || null;
      const canAct = !disabled();
      const head = `<header class="gpi-multi-head"><strong>${esc(tr('Same study on other databases', '在其他数据库上运行同一研究'))}</strong></header>`;
      let body;
      if (!payload) {
        body = `<p class="gpi-multi-note">${esc(entry.error || tr('Loading…', '正在读取…'))}</p>`;
      } else if (!payload.card) {
        body = `<p class="gpi-multi-note">${esc(tr('Preparing a review for each database…', '正在为每个数据库准备审阅…'))}</p>`;
      } else if (payload.status === 'review') {
        const targets = payload.card.targets || [];
        const runnable = targets.filter(target => target.disposition !== 'not_run').length;
        body = `<ul class="gpi-multi-targets">${targets.map(reviewTarget).join('')}</ul>
          <p class="gpi-multi-note">${esc(tr('Each database runs in turn and reports its own result; results are not pooled.', '各数据库依次运行，结果分别报告，不合并。'))}</p>
          ${runnable ? `<div class="gpi-multi-actions"><button class="btn primary sm" type="button" data-gpi-multi-approve="${esc(batchId)}" data-gpi-multi-card="${esc(payload.card_sha256)}"${canAct ? '' : ' disabled'}>${esc(tr(`Approve and run ${runnable} database${runnable > 1 ? 's' : ''}`, `批准并运行 ${runnable} 个数据库`))}</button></div>` : ''}`;
      } else {
        const answer = payload.answer || {};
        const reference = answer.reference && answer.reference.primary
          ? `<li class="is-reference"><strong>${esc(databaseName(answer.reference.primary.database))}</strong><span>${esc(tr('Original study', '原研究'))}</span><b>${estimateText(answer.reference.primary)}</b></li>` : '';
        body = `<ul class="gpi-multi-targets">${reference}${(answer.targets || []).map(item => progressTarget(payload.card, item, batchId, canAct)).join('')}</ul>${comparisonHtml(answer)}`;
      }
      return `<article class="gpi-message assistant gpi-multi-database" data-gpi-multi-batch="${esc(batchId)}"><div class="gpi-message-body">${head}${body}</div></article>`;
    }

    async function act(batchId, request) {
      const current = session();
      if (!current || disabled()) return;
      const owner = key(current);
      try {
        await request(current);
      } catch (error) {
        statuses.set(batchId, { ...(statuses.get(batchId) || {}), error: errorText(error) });
      }
      statuses.set(batchId, { ...(statuses.get(batchId) || {}), at: 0 });
      await load(batchId, owner);
    }
    function handleClick(event) {
      const approve = event.target.closest('[data-gpi-multi-approve]');
      if (approve) {
        if (!approve.disabled) {
          approve.disabled = true;
          const batchId = approve.dataset.gpiMultiApprove;
          void act(batchId, current => api().approvePiMultiDatabase({ project_id: projectId(), session_id: current.session_id,
            batch_id: batchId, card_sha256: approve.dataset.gpiMultiCard }));
        }
        return true;
      }
      const cancel = event.target.closest('[data-gpi-multi-cancel]');
      if (cancel) {
        if (!cancel.disabled) {
          cancel.disabled = true;
          const batchId = cancel.dataset.gpiMultiCancel;
          void act(batchId, current => api().cancelPiMultiDatabase({ project_id: projectId(), session_id: current.session_id,
            batch_id: batchId, database: cancel.dataset.gpiMultiDatabase }));
        }
        return true;
      }
      return false;
    }
    return { rows, place, render, handleClick };
  }

  window.EasyICU.guidedPi.declare('multiDatabase', { create });
})();
