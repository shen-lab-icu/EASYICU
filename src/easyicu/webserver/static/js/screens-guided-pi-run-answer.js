/* Owner: Guided Pi run-answer widget. */
/* The completed run's direct answer, read the way a finished task reads in
   the reference workspace: what was found, in the run's own registered
   numbers, the main figure beside it, the one limit that bounds it, and next
   questions built from this plan's own variables.
   It selects values the run already registered -- the result tables through
   the `resultSummary` owner, evidence-bound claims, the figure gallery -- and
   never computes a new estimate or raises the run's analysis-only authority. */
(function () {
  'use strict';

  const REQUIRED = ['result_tables.json', 'agent_plan.json'];
  const OPTIONAL = ['run_context.json', 'figure_gallery.json', 'manuscript_provenance.json'];
  const RASTER_IMAGE = /^data:image\/(?:png|jpeg|webp);base64,[A-Za-z0-9+/=]+$/;
  // Plan inputs that describe measurement bookkeeping or row identity, not a
  // baseline characteristic a follow-up question could adjust for.
  const NON_COVARIATE = /^artifact:|^table:|(?:^|_)(?:id|n|measured|time)$/;
  const AGGREGATE_SUFFIX = /_(?:first|last|min|max|mean|median)$/;
  const CJK = /[\u3400-\u9fff]/;
  // Reader names for the result-table kinds a file note mentions, in the
  // order a reader meets them; audits are counted rather than named.
  const TABLE_KINDS = [
    [/exposure_outcome_distribution/, ['group proportions and outcomes', '分组比例与结局']],
    [/table_one/, ['baseline characteristics', '基线特征']],
    [/estimates|odds_ratio|absolute_risk|cox|logistic/, ['model estimates', '模型估计']],
    [/robustness|sensitivity|rcs_/, ['sensitivity analyses', '敏感性分析']],
    [/(?:cohort|population)_(?:analysis_)?flow(?!_source)/, ['cohort flow', '队列流程']],
  ];
  const AUDIT_TABLE = /audit|missing|denominator/;
  // Analysis types (the planner's registry keys) whose answer is a model's
  // predictive performance: the host's typed block for the primary model
  // (result_tables.json `prediction_performance`), never a table preview.
  const PREDICTION_TYPES = new Set(['prediction_model', 'dynamic_prediction']);
  const PREDICTION_PERFORMANCE_SCHEMA = 'easyicu.web-prediction-performance/1';
  // The host run gate's two states. A block with any other scope shows no
  // answer, since the card has no wording for it.
  const PREDICTION_AUTHORITY_SCOPES = new Set(['analysis_only', 'blocked']);
  // The methods that re-estimate on each patient's first ICU stay, as the
  // binary-association sensitivity executor runs them.
  const FIRST_STAY_METHODS = new Set(['first_stay_association', 'one_stay_per_patient_association']);
  // A step's method head: the executor reads it before " with ", the planner
  // before "(".
  const methodHead = method => String(method || '').trim().toLowerCase()
    .split(' with ', 1)[0].split('(', 1)[0].trim();

  function create(deps) {
    const { tr, esc, api, projectId, resourceButton } = deps;
    let current = { key: '', projection: null, loading: false, failed: false };

    function runRefs(latestRun) {
      const refs = Array.isArray(latestRun && latestRun.artifact_refs) ? latestRun.artifact_refs : [];
      const byName = new Map();
      refs.forEach(ref => {
        if (ref && ref.run_id === latestRun.run_id && /^[a-f0-9]{64}$/.test(String(ref.sha256 || ''))) {
          byName.set(String(ref.artifact || ''), ref);
        }
      });
      return byName;
    }

    function keyFor(latestRun) {
      if (!latestRun || !latestRun.run_id || typeof projectId !== 'function') return '';
      const refs = runRefs(latestRun);
      if (REQUIRED.some(name => !refs.has(name))) return '';
      return [projectId(), latestRun.run_id]
        .concat(REQUIRED.concat(OPTIONAL).map(name => (refs.get(name) || {}).sha256 || '-')).join(':');
    }

    function finite(value) {
      if (value == null || (typeof value === 'string' && !value.trim())) return null;
      const number = Number(value);
      return Number.isFinite(number) ? number : null;
    }
    function count(value) {
      const number = finite(value);
      return number == null ? '' : Math.round(number).toLocaleString('en-US');
    }
    function percent(value) {
      const number = finite(value);
      return number == null ? '' : `${number.toFixed(2)}%`;
    }
    // A display label without its parenthetical definition: the definition
    // stays in the plan and the report; a sentence needs the name.
    function shortLabel(value) {
      return String(value || '').replace(/\s*[（(][^）)]*[）)]\s*$/, '').trim();
    }

    // The name a Chinese reader sees for one plan column. The plan registers
    // its display labels for the English manuscript; a Chinese label stays as
    // registered. Otherwise a column that is itself a catalog concept (an
    // exact id, not an aggregate such as `_max`) takes the catalog's governed
    // Chinese name. Any other column keeps its registered label, so nothing
    // is translated or guessed.
    function readerLabel(labels, column) {
      const registered = typeof labels[column] === 'string' ? labels[column] : '';
      if (window.EU_LANG !== 'zh' || CJK.test(registered)) return registered;
      const dict = window.EU_CATALOG && window.EU_CATALOG.dict ? window.EU_CATALOG.dict : {};
      const row = Object.prototype.hasOwnProperty.call(dict, column) ? dict[column] : null;
      const name = Array.isArray(row) && typeof row[1] === 'string' ? row[1].trim() : '';
      return CJK.test(name) ? name : registered;
    }

    function conceptName(variable) {
      const dict = window.EU_CATALOG && window.EU_CATALOG.dict ? window.EU_CATALOG.dict : {};
      const base = String(variable || '').replace(AGGREGATE_SUFFIX, '');
      const row = dict[variable] || dict[base];
      return Array.isArray(row) ? tr(String(row[0] || base), String(row[1] || row[0] || base)) : '';
    }

    function primarySpec(plan) {
      const steps = Array.isArray(plan && plan.steps) ? plan.steps : [];
      const specs = steps.filter(step => step && step.planned_analysis_role === 'primary'
        && step.exposure_outcome_distribution_spec).map(step => step.exposure_outcome_distribution_spec);
      return specs.length === 1 ? specs[0] : null;
    }

    function baselineCovariates(plan, exposure) {
      const steps = Array.isArray(plan && plan.steps) ? plan.steps : [];
      const baseline = steps.find(step => step && step.table_one_spec && Array.isArray(step.inputs));
      if (!baseline) return [];
      // Every form of the grouping concept (its _min, _n, _first_time...) is
      // the exposure itself, not a covariate.
      const grouping = [exposure, baseline.table_one_spec.group_by]
        .map(value => String(value || '').replace(AGGREGATE_SUFFIX, '')).filter(Boolean);
      const seen = new Set();
      return baseline.inputs.map(String)
        .filter(name => name && !NON_COVARIATE.test(name)
          && !grouping.some(base => name === base || name.startsWith(`${base}_`)))
        .map(conceptName)
        .filter(name => name && !seen.has(name) && seen.add(name))
        .slice(0, 5);
    }

    function mainFigure(gallery) {
      const figures = Array.isArray(gallery && gallery.figures) ? gallery.figures : [];
      const usable = figures.filter(figure => figure && RASTER_IMAGE.test(String(figure.data_url || '')));
      const main = usable.find(figure => figure.placement === 'main' || figure.tier === 'primary_publication') || usable[0] || null;
      return {
        figure: main ? { dataUrl: String(main.data_url), caption: String(main.caption || '').slice(0, 400) } : null,
        total: figures.length,
        main: figures.filter(figure => figure && figure.placement === 'main').length,
      };
    }

    // The registered primary estimate: the typed estimate the result summary
    // marks primary, else the evidence-bound primary_or claim. Any other
    // estimate (a covariate's, a sensitivity analysis') is not this run's
    // answer, so without a primary one the answer names none.
    function primaryEstimate(summary, claims) {
      const typed = Array.isArray(summary && summary.estimates)
        ? summary.estimates.find(row => row && row.primary === true) : null;
      if (typed && typed.display && typed.display.value) {
        return {
          measure: String(typed.measure || ''), label: String(typed.label || typed.contrast || ''),
          value: String(typed.display.value), low: String(typed.display.low || ''), high: String(typed.display.high || ''),
        };
      }
      const find = patterns => claims.find(row => patterns.some(pattern => pattern.test(String(row && row.source_field || '')))) || null;
      const effect = find([/^primary_or$/]);
      if (!effect || effect.display_value == null) return null;
      const low = find([/^primary_or_ci\[0\]$/, /^primary_or_ci_low$/]);
      const high = find([/^primary_or_ci\[1\]$/, /^primary_or_ci_high$/]);
      return {
        measure: 'OR', label: '', value: String(effect.display_value),
        low: low && low.display_value != null ? String(low.display_value) : '',
        high: high && high.display_value != null ? String(high.display_value) : '',
      };
    }

    // The primary model's performance as the host bound it to its evidence;
    // a block that does not validate shows nothing rather than a guess.
    function predictionPerformance(block) {
      if (!block || typeof block !== 'object' || block.schema_version !== PREDICTION_PERFORMANCE_SCHEMA
        || block.product !== 'table:model_performance' || !String(block.evidence_id || '').trim()
        || !PREDICTION_AUTHORITY_SCOPES.has(block.authority_scope)) return null;
      const unit = value => {
        const number = typeof value === 'number' ? finite(value) : null;
        return number != null && number >= 0 && number <= 1 ? number : null;
      };
      const whole = value => (Number.isInteger(value) && value >= 0 ? value : null);
      const auroc = unit(block.auroc);
      const low = unit(block.auroc_ci_low);
      const high = unit(block.auroc_ci_high);
      const brier = unit(block.brier_score);
      if (auroc == null || low == null || high == null || brier == null || low > auroc || auroc > high) return null;
      // The validation set's patients and events cannot outnumber its rows.
      const records = whole(block.validation_n);
      const patients = whole(block.validation_subject_n);
      const events = whole(block.validation_event_n);
      if (records == null || events == null || !patients || patients > records || events > records) return null;
      return {
        evidenceId: String(block.evidence_id),
        auroc, low, high, brier, records, patients, events,
        slope: block.calibration_status === 'estimated' && typeof block.calibration_slope === 'number'
          ? finite(block.calibration_slope) : null,
        intervalMethod: String(block.auroc_ci_method || ''),
        analysisOnly: block.authority_scope === 'analysis_only',
        paperAuthorized: block.paper_authorization_allowed === true,
      };
    }

    function project(rows) {
      const payload = name => (rows[name] && rows[name].payload && typeof rows[name].payload === 'object')
        ? rows[name].payload : {};
      const plan = payload('agent_plan.json');
      const context = payload('run_context.json');
      const provenance = payload('manuscript_provenance.json');
      const summarizer = window.EasyICU.guidedPi.optional('resultSummary');
      const summary = summarizer && typeof summarizer.summarize === 'function'
        ? summarizer.summarize(payload('result_tables.json'), plan, context) : {};
      // A grouped exposure is named by its grouping record where the plan
      // registers no label (screens-agent-exposure-levels.js).
      const levels = window.AGENT_EXPOSURE_LEVELS ? window.AGENT_EXPOSURE_LEVELS.reader(context, tr) : null;
      const claims = (Array.isArray(provenance.claims) ? provenance.claims : [])
        .concat(Array.isArray(summary.claims) ? summary.claims : []);
      const claim = field => claims.find(row => row && row.source_field === field) || null;
      const spec = primarySpec(plan);
      const labels = plan.display_labels && typeof plan.display_labels === 'object' ? plan.display_labels : {};
      const exposureLabel = spec
        ? shortLabel(readerLabel(labels, spec.exposure)) || (levels ? levels.variableName(spec.exposure) : '') : '';
      const outcomeLabel = shortLabel((spec && readerLabel(labels, spec.outcome)) || summary.outcomeLabel || '');
      const groups = (Array.isArray(summary.exposureLevels) ? summary.exposureLevels : [])
        .filter(row => row && row.label && finite(row.n) != null);
      const figures = mainFigure(payload('figure_gallery.json'));
      const tables = Array.isArray(payload('result_tables.json').tables) ? payload('result_tables.json').tables : [];
      const total = claim('n_total');
      const overallRisk = claim('overall_outcome.risk_pct');
      const source = context.source && typeof context.source === 'object'
        ? String(context.source.label || context.source.database || '') : '';
      return {
        source,
        question: String(context.question || plan.research_question || ''),
        analysisType: String(plan.analysis_type || ''),
        exposureLabel,
        outcomeLabel,
        groups,
        total: total && total.display_value != null ? String(total.display_value) : '',
        overallRisk: overallRisk && overallRisk.display_value != null ? String(overallRisk.display_value) : '',
        estimate: primaryEstimate(summary, claims),
        prediction: PREDICTION_TYPES.has(String(plan.analysis_type || ''))
          ? predictionPerformance(payload('result_tables.json').prediction_performance) : null,
        covariates: baselineCovariates(plan, spec && spec.exposure),
        steps: (Array.isArray(plan.steps) ? plan.steps : [])
          .filter(step => step && step.method !== 'visualization' && String(step.intent || '').trim())
          .map(step => (levels ? levels.inText(String(step.intent).trim()) : String(step.intent).trim())).slice(0, 8),
        groupsUnreadable: summary.groupsUnreadable === true,
        figure: figures.figure,
        figureTotal: figures.total,
        figureMain: figures.main,
        tableTotal: tables.length,
        tableKinds: TABLE_KINDS.filter(([pattern]) => tables.some(table => {
          const name = String(table && table.name || '').toLowerCase();
          return pattern.test(name) && !/source_data/.test(name);
        })).map(([, names]) => tr(names[0], names[1])),
        auditTables: tables.filter(table => AUDIT_TABLE.test(String(table && table.name || '').toLowerCase())).length,
        // An executed step bound to a declared first-stay sensitivity. A
        // design candidate the plan rejected, or a field that only names
        // first stays, is not one. The host's own first-stay restriction is
        // in no run file this card reads.
        firstStay: (Array.isArray(plan.steps) ? plan.steps : []).some(step => step
          && FIRST_STAY_METHODS.has(methodHead(step.method))
          && Array.isArray(step.sensitivity_spec_ids) && step.sensitivity_spec_ids.length > 0),
      };
    }

    async function load(latestRun) {
      const key = keyFor(latestRun);
      if (!key || current.key === key) return false;
      current = { key, projection: null, loading: true, failed: false };
      const expectedProjectId = projectId();
      const refs = runRefs(latestRun);
      try {
        const client = typeof api === 'function' ? api() : null;
        if (!client || typeof client.loadPiCopilotResearchArtifact !== 'function') throw new Error('run_answer_api_unavailable');
        const loaded = await Promise.all(REQUIRED.concat(OPTIONAL).map(async name => {
          const ref = refs.get(name);
          if (!ref) return [name, null];
          try {
            const response = await client.loadPiCopilotResearchArtifact(expectedProjectId, latestRun.run_id, name, ref.sha256);
            return [name, response && response.ok !== false ? response : null];
          } catch (error) {
            if (REQUIRED.includes(name)) throw error;
            return [name, null];
          }
        }));
        if (current.key !== key || projectId() !== expectedProjectId) return false;
        const rows = Object.fromEntries(loaded);
        if (REQUIRED.some(name => !rows[name])) throw new Error('run_answer_artifact_missing');
        current = { key, projection: project(rows), loading: false, failed: false };
      } catch (_error) {
        if (current.key !== key || projectId() !== expectedProjectId) return false;
        current = { key, projection: null, loading: false, failed: true };
      }
      return true;
    }

    function projectionFor(latestRun) {
      const key = keyFor(latestRun);
      return key && current.key === key ? current.projection : null;
    }

    function answerSentences(view) {
      const sentences = [];
      // A prediction study answers only with its primary model's performance.
      if (PREDICTION_TYPES.has(view.analysisType)) {
        const model = view.prediction;
        if (!model) return sentences;
        // How the host drew the interval, in the reader vocabulary's words;
        // a method it has no name for is not named.
        const methodName = window.AGENT_READER_VOCAB
          ? window.AGENT_READER_VOCAB.value('auroc_ci_method', model.intervalMethod) : '';
        const method = methodName ? tr(`, ${methodName}`, `，${methodName}`) : '';
        const slope = model.slope != null
          ? tr(`, calibration slope ${model.slope.toFixed(2)}`, `，校准斜率 ${model.slope.toFixed(2)}`) : '';
        // The sample size and event count go with the performance (TRIPOD);
        // the patients are named when some have more than one ICU stay.
        const sample = model.patients < model.records
          ? tr(`validation set: ${count(model.records)} ICU stays from ${count(model.patients)} patients, ${count(model.events)} outcome events`,
            `验证集 ${count(model.records)} 个 ICU 入住记录，来自 ${count(model.patients)} 名患者，${count(model.events)} 个结局事件`)
          : tr(`validation set: ${count(model.records)} ICU stays, ${count(model.events)} outcome events`,
            `验证集 ${count(model.records)} 个 ICU 入住记录，${count(model.events)} 个结局事件`);
        sentences.push(tr(
          `Internal validation in the same database (${sample}): AUROC ${model.auroc.toFixed(3)} (95% CI ${model.low.toFixed(3)}–${model.high.toFixed(3)}${method}), Brier score ${model.brier.toFixed(3)}${slope}.`,
          `同库内部验证（${sample}）：AUROC ${model.auroc.toFixed(3)}（95% CI ${model.low.toFixed(3)}–${model.high.toFixed(3)}${method}），Brier ${model.brier.toFixed(3)}${slope}。`,
        ));
        return sentences;
      }
      const groups = view.groups;
      const source = view.source || tr('this data source', '本数据源');
      if (groups.length === 2) {
        const reference = groups[0];
        const focal = groups[1];
        if (view.total && focal.n != null && focal.sharePct != null) {
          sentences.push(tr(
            `Of ${view.total} ICU stays in ${source}, ${count(focal.n)} (${percent(focal.sharePct)}) were in the group “${focal.label}”.`,
            `在 ${source} 的 ${view.total} 个 ICU 入住记录中，「${focal.label}」有 ${count(focal.n)} 个（${percent(focal.sharePct)}）。`,
          ));
        }
        if (focal.outcomeRatePct != null && reference.outcomeRatePct != null) {
          const part = row => tr(
            `${percent(row.outcomeRatePct)} (${count(row.events)}/${count(row.denominator)}) in “${row.label}”`,
            `「${row.label}」组 ${percent(row.outcomeRatePct)}（${count(row.events)}/${count(row.denominator)}）`,
          );
          const outcome = view.outcomeLabel || tr('Outcome', '结局');
          const overall = view.overallRisk ? tr(`; overall ${view.overallRisk}`, `；总体 ${view.overallRisk}`) : '';
          sentences.push(tr(
            `${outcome}: ${part(focal)} vs ${part(reference)}${overall}.`,
            `${outcome}：${part(focal)}，${part(reference)}${overall}。`,
          ));
        }
      } else if (groups.length > 2) {
        const rows = groups.slice(0, 6).map(row => row.outcomeRatePct != null
          ? tr(`“${row.label}” ${percent(row.outcomeRatePct)} (${count(row.events)}/${count(row.denominator)})`,
            `「${row.label}」${percent(row.outcomeRatePct)}（${count(row.events)}/${count(row.denominator)}）`)
          : tr(`“${row.label}” n = ${count(row.n)}`, `「${row.label}」${count(row.n)} 个`));
        const outcome = view.outcomeLabel || tr('the outcome', '结局');
        sentences.push(tr(
          `${outcome} by ${view.exposureLabel || 'group'} in ${source}: ${rows.join('; ')}.`,
          `${source} 中按${view.exposureLabel || '分组'}的${outcome}：${rows.join('；')}。`,
        ));
      }
      if (view.groupsUnreadable && (groups.length || view.estimate) && window.AGENT_EXPOSURE_LEVELS) {
        sentences.push(window.AGENT_EXPOSURE_LEVELS.unreadableText(tr));
      }
      if (view.estimate) {
        const interval = view.estimate.low && view.estimate.high
          ? tr(` (95% CI ${view.estimate.low}–${view.estimate.high})`, `（95% CI ${view.estimate.low}–${view.estimate.high}）`) : '';
        const contrast = view.estimate.label ? tr(` for ${view.estimate.label}`, `（${view.estimate.label}）`) : '';
        sentences.push(tr(
          `Primary estimate${contrast}: ${view.estimate.measure} ${view.estimate.value}${interval}.`,
          `主要估计${contrast}：${view.estimate.measure} ${view.estimate.value}${interval}。`,
        ));
      }
      return sentences;
    }

    // The bound on the answer follows the study: a prediction study's answer
    // is its model's performance, in the result tables; otherwise a
    // registered estimate is an association and group counts a description.
    function caveat(view) {
      if (view.prediction) {
        const scope = view.prediction.analysisOnly
          ? tr('for analysis only', '仅供分析')
          : tr('this run did not pass its automated checks, so the numbers are for review only', '本次运行未通过自动校验，数字仅供核对');
        const paper = view.prediction.paperAuthorized ? '' : tr(', without paper authorization', '，未获论文授权');
        return tr(
          `Model performance from internal validation in the same database; ${scope}${paper}. It needs external validation before use in another database, and it does not show cause and effect. Each row is an analysis record (ICU stay), not necessarily an independent patient.`,
          `这是同一数据库内部验证的模型表现，${scope}${paper}；用于其他数据库前需要外部验证，也不说明因果。统计单位是分析记录（ICU 入住），不一定对应独立患者。`,
        );
      }
      return view.estimate
        ? tr('An observational association estimate; it does not establish cause and effect. Each row is an analysis record (ICU stay), not necessarily an independent patient.',
          '这是观察性关联估计，不能据此推断因果；统计单位是分析记录（ICU 入住），不一定对应独立患者。')
        : tr('A descriptive comparison without adjustment; it does not show cause and effect. Each row is an analysis record (ICU stay), not necessarily an independent patient.',
          '这是未调整的描述性比较，不能说明因果；统计单位是分析记录（ICU 入住），不一定对应独立患者。');
    }

    function figureButton(latestRun, view) {
      if (!view.figure || typeof resourceButton !== 'function') return '';
      const ref = runRefs(latestRun).get('figure_gallery.json');
      if (!ref) return '';
      const label = view.figureTotal > 1
        ? tr(`Open all ${view.figureTotal} figures`, `打开全部 ${view.figureTotal} 张图`)
        : tr('Open the figure', '打开图件');
      return `<figure class="gpi-run-answer-figure">${resourceButton(
        { ...ref, label },
        label,
        { className: 'gpi-run-answer-figure-open', contentHtml: `<img src="${esc(view.figure.dataUrl)}" alt="${esc(view.figure.caption || tr('Main result figure', '主要结果图'))}" loading="lazy" decoding="async">` },
      )}<figcaption>${esc(tr('Main figure', '主图'))} · ${resourceButton({ ...ref, label }, label)}</figcaption></figure>`;
    }

    function render(latestRun) {
      const key = keyFor(latestRun);
      if (!key || current.key !== key) return '';
      if (current.loading) {
        return `<div class="gpi-run-answer is-loading" data-gpi-run-answer="${esc(latestRun.run_id)}" aria-busy="true"><p>${esc(tr('Reading this run’s registered results…', '正在读取本次运行登记的结果…'))}</p></div>`;
      }
      const view = current.projection;
      const sentences = view ? answerSentences(view) : [];
      if (!view || !sentences.length) return '';
      return `<div class="gpi-run-answer" data-gpi-run-answer="${esc(latestRun.run_id)}">
        <div class="gpi-run-answer-text">${sentences.map(text => `<p>${esc(text)}</p>`).join('')}<p class="gpi-run-answer-caveat">${esc(caveat(view))}</p></div>
        ${figureButton(latestRun, view)}
      </div>`;
    }

    // Next questions a researcher would ask of this result, phrased from the
    // plan's own labels and variables. Null until the run's answer is read,
    // so the card can fall back to its artifact-based suggestions.
    function followUps(latestRun) {
      const view = projectionFor(latestRun);
      // A prediction study reports no group or estimate facts to build
      // them from, so its card keeps the artifact-based suggestions.
      if (!view || PREDICTION_TYPES.has(view.analysisType) || !view.groups.length && !view.estimate) return null;
      const exposure = view.exposureLabel || tr('the exposure', '暴露');
      const outcome = view.outcomeLabel || tr('the outcome', '结局');
      const list = [];
      if (view.estimate) {
        list.push(tr(
          `Explain what the primary estimate (${view.estimate.measure} ${view.estimate.value}) means clinically, including its uncertainty.`,
          `解释主要估计（${view.estimate.measure} ${view.estimate.value}）的临床含义和不确定性。`,
        ));
      } else if (view.covariates.length) {
        list.push(tr(
          `After adjusting for ${view.covariates.join(', ')}, is ${exposure} still associated with ${outcome}?`,
          `在调整${view.covariates.join('、')}后，${exposure}与${outcome}的关联还成立吗？`,
        ));
      }
      if (view.covariates.some(name => /age|年龄/i.test(name))) {
        list.push(tr(
          `Stratified by age, how does ${outcome} differ across the ${exposure} groups?`,
          `按年龄分层后，${exposure}各组的${outcome}有什么差别？`,
        ));
      }
      // A plan that already re-estimates on first stays has its answer.
      if (!view.firstStay) {
        list.push(tr(
          `Keep only each patient’s first ICU stay and recompute these results.`,
          '只保留每位患者的首次 ICU 入住，重新计算这些结果。',
        ));
      }
      list.push(tr(
        `Replicate this analysis in another database and compare the results.`,
        '在另一个数据库中复现这项分析，比较结果是否一致。',
      ));
      return list.slice(0, 4);
    }

    // What each result file actually holds for this run, for the card's file
    // table. Empty when the run's answer has not been read.
    function fileNote(latestRun, artifact) {
      const view = projectionFor(latestRun);
      if (!view) return '';
      if (artifact === 'result_tables.json' && view.tableTotal) {
        const kinds = view.tableKinds.join(tr(', ', '、'));
        const audits = view.auditTables
          ? tr(`${kinds ? '; ' : ''}${view.auditTables} data-quality audits`, `${kinds ? '，另有' : ''} ${view.auditTables} 张数据质量审计表`) : '';
        return kinds || audits
          ? tr(`${view.tableTotal} tables: ${kinds}${audits}`, `${view.tableTotal} 张表：${kinds}${audits}`)
          : tr(`${view.tableTotal} tables`, `${view.tableTotal} 张表`);
      }
      if (artifact === 'figure_gallery.json' && view.figureTotal) {
        const supporting = Math.max(0, view.figureTotal - view.figureMain);
        return view.figureMain && supporting
          ? tr(`${view.figureMain} main and ${supporting} supplementary figures`, `${view.figureMain} 张主图、${supporting} 张补充图`)
          : tr(`${view.figureTotal} figures`, `${view.figureTotal} 张图`);
      }
      if (artifact === 'full_analysis_report.json' && view.steps.length) {
        return tr(`Question, ${view.steps.length} analysis steps, results and limits on one page`,
          `一页看完问题、${view.steps.length} 个分析步骤、结果与局限`);
      }
      return '';
    }

    // The executed plan, in the plan's own words, for the review's Approach tab.
    function approachSteps(latestRun) {
      const view = projectionFor(latestRun);
      return view ? view.steps.slice() : [];
    }

    function materials(latestRun) {
      const view = projectionFor(latestRun);
      if (!view) return '';
      return [view.source, view.total ? tr(`${view.total} ICU stays`, `${view.total} 个 ICU 入住记录`) : '']
        .filter(Boolean).join(' · ');
    }

    function limitation(latestRun) {
      const view = projectionFor(latestRun);
      if (!view) return '';
      // A prediction study's limit is stated only beside the model
      // performance it bounds.
      if (PREDICTION_TYPES.has(view.analysisType)) return view.prediction ? caveat(view) : '';
      return view.groups.length || view.estimate ? caveat(view) : '';
    }

    return Object.freeze({ load, render, followUps, fileNote, approachSteps, materials, limitation });
  }

  window.EasyICU.guidedPi.declare('runAnswer', { create });
})();
