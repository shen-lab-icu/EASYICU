/* Owner: quick-answer cards in the Guided Copilot conversation.
   Renders the host's quick-answer attachments (webserver/pi_copilot/
   quick_answers.py): the digest-bound card, the evidence-bound answer and the
   full-study handoff. Qualification, numbers and approval stay with the host;
   a click hands the displayed digest to the plan-action dispatcher. Every
   number shown is an answer row's value. */
(function () {
  'use strict';

  const STATISTICS = {
    count: ['Count', '人数'], proportion: ['Proportion', '比例'], mean: ['Mean', '均值'],
    median: ['Median (IQR)', '中位数（四分位数间距）'], iqr: ['Median (IQR)', '中位数（四分位数间距）'],
  };
  const SUMMARIES = {
    max: ['highest', '最高'], min: ['lowest', '最低'], first: ['first', '首次'],
    last: ['last', '末次'], mean: ['mean', '平均'], value: ['', ''],
  };
  const UNITS = { icu_stay: ['ICU stays', 'ICU 入住'], patient: ['patients', '患者'], hospitalization: ['hospitalizations', '住院'] };
  const TESTS = {
    welch_t_or_anova: ['Welch t test / ANOVA', 'Welch t 检验 / 方差分析'],
    mann_whitney_or_kruskal: ['Mann–Whitney / Kruskal–Wallis test', 'Mann–Whitney / Kruskal–Wallis 检验'],
    chi_square: ['Chi-squared test', '卡方检验'],
    fisher_exact: ["Fisher's exact test", 'Fisher 精确检验'],
  };
  // Why a question was not answered here, per stable host code.
  const REASONS = {
    quick_requires_adjusted_study: ['It needs adjustment for other factors, which a full study does.', '需要调整其他因素，这要在完整研究里做。'],
    quick_requires_survival_study: ['It is about time to an event, which a full study analyses.', '涉及到事件发生的时间，这要在完整研究里分析。'],
    quick_requires_multistep_study: ['It needs several analysis steps, which a full study plans.', '需要多步分析，这要在完整研究里规划。'],
    quick_substitution_requires_full_study: ['This database answers it only with changed definitions, which a full study reviews.', '这个数据库只能改口径回答，口径调整要在完整研究里审阅。'],
    quick_named_level_undistinguished: ['The data record whether it happened, not which kind. Ask without naming the kind.', '数据只记录是否发生，不区分类型。请不指定类型重新提问。'],
    quick_repeated_units_ask_first_stay: ['The same patient has several ICU stays. Ask about each patient\'s first ICU stay, or make it a full study.', '同一患者有多次 ICU 入住。可改问每位患者的首次 ICU 入住，或改为完整研究。'],
    quick_repeated_units_require_full_study: ['The same patient has several ICU stays; this comparison needs the full study.', '同一患者有多次 ICU 入住，这个比较需要在完整研究里处理。'],
    quick_independent_units_unverified: ['The groups cannot be confirmed to hold separate patients.', '无法确认各组是不同的患者。'],
    quick_small_cells_suppressed: ['A group is too small to show under the privacy rule.', '有的组人数太少，按隐私规则不显示结果。'],
    quick_insufficient_observations: ['Too few recorded values to compute it.', '有效记录太少，无法计算。'],
    quick_population_empty: ['No ICU stays meet these criteria.', '没有符合这些条件的 ICU 入住。'],
    quick_capability_refused: ['This database does not record what the question needs.', '这个数据库没有记录问题所需的数据。'],
    quick_capability_unassessed: ['What the question needs has not been assessed for this database yet.', '问题所需的数据在这个数据库上还没有评估。'],
    quick_qualifying_provider_required: ['This database needs a model connection that qualifies for credentialed data.', '这个数据库需要符合受控数据要求的模型连接。'],
    quick_identifier_forbidden: ['Questions about an individual patient are not answered here.', '不回答针对单个患者的问题。'],
    quick_two_groups_required: ['A comparison needs at least two groups.', '比较至少需要两组。'],
    quick_grouping_scope_exceeded: ['The grouping is wider than a quick answer covers.', '分组超出了快速回答的范围。'],
    quick_answer_too_large: ['The answer would be too large for a quick answer.', '结果太多，超出了快速回答的范围。'],
    quick_prepared_source_too_large: ['The data package is too large for a quick answer.', '数据包太大，超出了快速回答的范围。'],
    quick_time_budget_exceeded: ['It took longer than a quick answer allows.', '计算超出了快速回答的时间。'],
    quick_execution_failed: ['The calculation did not complete.', '计算没有完成。'],
    quick_evidence_binding_failed: ['The result could not be tied to its evidence, so it is not shown.', '结果无法与证据对应，因此不显示。'],
  };
  const AMBIGUOUS = ['quick_reading_ambiguous', 'quick_reading_invalid', 'quick_measure_ambiguous', 'quick_measure_unresolved',
    'quick_window_unresolved', 'quick_level_unresolved', 'quick_levels_unresolved', 'quick_grouping_ambiguous', 'quick_comparison_ambiguous'];
  const NOT_PREPARED = ['quick_population_not_prepared', 'quick_prepared_columns_missing', 'quick_unit_not_prepared',
    'quick_prepared_source_required', 'quick_bound_source_unavailable', 'quick_unit_identity_missing', 'quick_unit_identity_invalid'];
  const CHANGED = ['quick_card_stale', 'quick_plan_changed', 'quick_population_changed', 'quick_source_changed', 'quick_levels_changed',
    'quick_source_identity_mismatch', 'quick_source_metadata_mismatch', 'quick_approval_mismatch'];

  function create(deps) {
    const { tr, esc, act, disabled } = deps;
    const pair = (map, key) => { const row = map[String(key || '')]; return row ? tr(row[0], row[1]) : ''; };

    function reasonText(code) {
      const key = String(code || '');
      if (REASONS[key]) return tr(REASONS[key][0], REASONS[key][1]);
      if (AMBIGUOUS.includes(key)) return tr('The question leaves the measure, time window or groups open.', '问题里的指标、时间窗或分组不够明确。');
      if (NOT_PREPARED.includes(key)) return tr('The current data package does not hold what the question needs.', '当前数据包里没有问题所需的数据。');
      if (CHANGED.includes(key)) return tr('The data or the question changed after this card was made. Ask again.', '卡片生成后数据或问题有变化，请重新提问。');
      return tr('This question is not suited to a quick answer.', '这个问题不适合快速回答。');
    }
    function conceptName(concept) {
      const row = (window.EU_CATALOG && window.EU_CATALOG.dict || {})[concept];
      return row ? tr(row[0], row[1]) : String(concept || '');
    }
    function databaseName(source) {
      const key = String(source && source.database || '');
      return (window.EU_DATABASE_NAMES && window.EU_DATABASE_NAMES.name(key)) || key;
    }
    function hours(value) { return Number.isInteger(value) ? String(value) : String(Number(value).toFixed(1)); }
    function measureText(reading) {
      const measure = reading && reading.measure || {};
      const span = measure.window;
      const summary = pair(SUMMARIES, measure.summary);
      const name = conceptName(measure.concept);
      const level = measure.level == null || measure.level === '' ? '' : ` = ${measure.level}`;
      const statistic = pair(STATISTICS, measure.statistic);
      return tr(
        `${statistic} of ${summary ? summary + ' ' : ''}${name}${level}${span ? `, ICU hours ${hours(span.start_hours)}–${hours(span.end_hours)}` : ''}`,
        `${span ? `入 ICU ${hours(span.start_hours)}–${hours(span.end_hours)} 小时` : ''}${name}${summary ? summary + '值' : ''}${level}的${statistic}`,
      );
    }
    function populationText(reading) {
      const criteria = Array.isArray(reading && reading.population && reading.population.criteria) ? reading.population.criteria : [];
      const quotes = criteria.map(row => String(row && row.quote || '').trim()).filter(Boolean);
      const unit = pair(UNITS, reading && reading.analysis_unit) || pair(UNITS, 'icu_stay');
      return quotes.length ? `${quotes.join(tr('; ', '；'))}${tr(` (${unit})`, `（按 ${unit}）`)}` : tr(`All ${unit}`, `全部 ${unit}`);
    }
    function substitutionsHtml(capability) {
      const owner = window.EasyICU.guidedPi.optional('sourceAdjustments');
      if (!owner || !capability || capability.status !== 'substitution' || typeof owner.items !== 'function') return '';
      const items = owner.items(capability);
      return items.length ? `<section class="gpi-source-adjustments"><h4>${esc(tr(`How ${databaseName(capability)} changes this question`, `${databaseName(capability)} 上的口径调整`))}</h4><ul>${items.join('')}</ul></section>` : '';
    }
    function number(value) {
      return typeof value === 'number' ? value.toLocaleString(window.EU_LANG === 'zh' ? 'zh-CN' : 'en-US', { maximumFractionDigits: 3 }) : String(value);
    }
    // One line per group, each value read from its own answer row.
    function answerTable(answer) {
      const rows = Array.isArray(answer && answer.rows) ? answer.rows : [];
      const groups = [];
      const byGroup = new Map();
      let comparison = null;
      rows.forEach(row => {
        if (String(row.statistic || '').startsWith('p_value:')) { comparison = row; return; }
        const group = String(row.group || 'Overall');
        if (!byGroup.has(group)) { byGroup.set(group, {}); groups.push(group); }
        const cells = byGroup.get(group);
        const key = row.level ? `${row.statistic}:${row.level}` : row.statistic;
        cells[key] = row;
        if (row.statistic === 'numerator' || row.statistic === 'percentage' || row.statistic === 'denominator') cells[row.statistic] = row;
      });
      const unitOf = row => row.unit && row.unit !== '%' ? ' ' + esc(row.unit) : row.unit === '%' ? '%' : '';
      const withUnit = row => row ? `${number(row.value)}${unitOf(row)}` : '';
      const lines = groups.map(group => {
        const cells = byGroup.get(group);
        let result = '';
        if (cells.median && cells.q1 && cells.q3) {
          const quartiles = `${number(cells.q1.value)}–${number(cells.q3.value)}`;
          result = `${number(cells.median.value)}${tr(` (IQR ${quartiles})`, `（${quartiles}）`)}${unitOf(cells.median)}`;
        } else if (cells.median) result = withUnit(cells.median);
        else if (cells.mean) result = withUnit(cells.mean);
        else if (cells.q1 && cells.q3) result = `${number(cells.q1.value)}–${withUnit(cells.q3)}`;
        else if (cells.numerator && cells.denominator) {
          result = `${number(cells.numerator.value)} / ${number(cells.denominator.value)}${cells.percentage ? tr(` (${withUnit(cells.percentage)})`, `（${withUnit(cells.percentage)}）`) : ''}`;
        }
        const observed = cells.nonmissing_n && cells.cohort_n
          ? tr(`${number(cells.nonmissing_n.value)} of ${number(cells.cohort_n.value)} with a value`, `${number(cells.cohort_n.value)} 例中 ${number(cells.nonmissing_n.value)} 例有值`)
          : cells.cohort_n ? tr(`${number(cells.cohort_n.value)} in the group`, `共 ${number(cells.cohort_n.value)} 例`) : '';
        const label = group === 'Overall' ? tr('All', '全部') : esc(group);
        return `<tr><th scope="row">${label}</th><td>${result}</td><td>${observed}</td></tr>`;
      });
      const test = comparison ? String(comparison.statistic).slice('p_value:'.length) : '';
      const testLine = comparison
        ? `<p class="gpi-quick-test">${esc(tr('Between groups', '组间比较'))}${tr(': ', '：')}P = ${number(comparison.value)}${pair(TESTS, test) ? tr(` (${esc(pair(TESTS, test))})`, `（${esc(pair(TESTS, test))}）`) : ''}</p>` : '';
      return lines.length ? `<table class="gpi-quick-table"><tbody>${lines.join('')}</tbody></table>${testLine}` : '';
    }

    function render(row) {
      const attachment = row && row.attachment || {};
      const id = String(attachment.quick_answer_id || '');
      const proposal = attachment.proposal || {};
      const answer = attachment.answer || null;
      const card = proposal.card || null;
      const reading = (card && card.plan && card.plan.effective_reading) || (proposal.expand && (proposal.expand.effective_reading || proposal.expand.reading)) || {};
      const source = (answer && answer.source) || (card && card.source) || (proposal.expand && proposal.expand.source) || {};
      const off = disabled() ? ' disabled' : '';
      const expandButton = primary => `<button class="btn ${primary ? 'primary ' : ''}sm" type="button" data-gpi-quick-expand="${esc(id)}"${off}>${esc(tr('Make it a full study', '改为完整研究'))}</button>`;
      const head = `<header class="gpi-quick-head"><strong>${esc(tr('Quick answer', '快速回答'))}</strong><span>${esc(databaseName(source))}</span></header>`;
      const question = reading.question ? `<p class="gpi-quick-question">${esc(reading.question)}</p>` : '';
      const scope = `<p class="gpi-quick-scope">${esc(tr('A descriptive summary: no adjustment and no causal reading; missing values are counted, not filled.', '描述性汇总：不调整其他因素，不作因果解释；缺失值只计数，不填补。'))}</p>`;
      let body = '';
      if (answer && answer.status === 'ok') {
        body = `<p class="gpi-quick-measure">${esc(measureText(reading))}</p>${answerTable(answer)}${scope}<div class="gpi-quick-actions">${expandButton(false)}</div>`;
      } else if (answer) {
        body = `<p class="gpi-quick-reason">${esc(reasonText(answer.reason))}</p><div class="gpi-quick-actions">${expandButton(true)}</div>`;
      } else if (proposal.status === 'approval_required' && card) {
        const substitution = card.capability && card.capability.status === 'substitution';
        body = `<dl class="gpi-quick-plan"><dt>${esc(tr('Population', '人群'))}</dt><dd>${esc(populationText(reading))}</dd><dt>${esc(tr('Calculation', '计算'))}</dt><dd>${esc(measureText(reading))}</dd>${Array.isArray(reading.grouping) && reading.grouping.length
          ? `<dt>${esc(tr('Groups', '分组'))}</dt><dd>${esc(reading.grouping.map(conceptName).join(tr(', ', '、')))}</dd>` : ''}</dl>${substitutionsHtml(card.capability)}${scope}<div class="gpi-quick-actions">${expandButton(false)}<button class="btn primary sm" type="button" data-gpi-quick-approve="${esc(id)}" data-gpi-quick-card="${esc(card.sha256)}" data-gpi-quick-accept="${substitution ? 'true' : 'false'}"${off}>${esc(substitution ? tr('Accept the changes and calculate', '同意口径调整并计算') : tr('Calculate', '计算'))}</button></div>`;
      } else {
        body = `<p class="gpi-quick-reason">${esc(reasonText(proposal.reason))}</p><div class="gpi-quick-actions">${expandButton(true)}</div>`;
      }
      return `<article class="gpi-message assistant gpi-quick-answer" data-gpi-quick-answer="${esc(id)}"><div class="gpi-message-body">${head}${question}${body}</div></article>`;
    }

    function timeMs(value) { const at = Date.parse(String(value || '')); return Number.isFinite(at) ? at : 0; }
    // Each attachment answers the turn that produced it.
    function rows(session) {
      const list = Array.isArray(session && session.quick_answers) ? session.quick_answers : [];
      const turns = Array.isArray(session && session.conversation_replay && session.conversation_replay.turns)
        ? session.conversation_replay.turns : [];
      return list.filter(item => item && item.quick_answer_id).map(item => {
        const turn = turns.find(row => row && row.job_id && row.job_id === item.message_job_id);
        const at = turn ? timeMs(turn.ended_at || turn.started_at) : 0;
        return { role: 'quick_answer', id: 'quick-answer-' + item.quick_answer_id, attachment: item, timelineAt: at ? at + 1 : 0 };
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
    function handleClick(event) {
      const approve = event.target.closest('[data-gpi-quick-approve]');
      if (approve) {
        if (!approve.disabled && !disabled()) {
          void act({ code: 'approve_quick_answer', quick_answer_id: approve.dataset.gpiQuickApprove,
            card_sha256: approve.dataset.gpiQuickCard, accept_substitutions: approve.dataset.gpiQuickAccept === 'true' });
        }
        return true;
      }
      const expand = event.target.closest('[data-gpi-quick-expand]');
      if (expand) {
        if (!expand.disabled && !disabled()) void act({ code: 'expand_into_full_study', quick_answer_id: expand.dataset.gpiQuickExpand });
        return true;
      }
      return false;
    }
    return { rows, place, render, handleClick, reasonText };
  }

  // A card the host qualified waits for the researcher's click instead of an
  // automatic full plan: the card the turn just made, and a card on the
  // study's own question until it becomes a full study. A stopped or
  // full-study proposal holds nothing (plan-actions.js).
  function holdsAutomaticPlan(session, workflow) {
    const list = Array.isArray(session && session.quick_answers) ? session.quick_answers : [];
    const lastJob = String(session && session.last_message_job_id || '');
    const configuration = workflow && workflow.study_setup_receipt && workflow.study_setup_receipt.configuration;
    const question = String(configuration && configuration.question || '').trim();
    return list.some(item => {
      const proposal = item && item.proposal || {};
      if (proposal.status !== 'approval_required' || !proposal.card || item.study_context_id) return false;
      const reading = proposal.expand && proposal.expand.reading || {};
      return (Boolean(lastJob) && item.message_job_id === lastJob)
        || (Boolean(question) && String(reading.question || '').trim() === question);
    });
  }

  window.EasyICU.guidedPi.declare('quickAnswers', { create, holdsAutomaticPlan });
})();
