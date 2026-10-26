/* Owner: Guided Pi result-summary widget. */
/* Registered result-table summary adapter for Guided Copilot reports.
   It selects already-produced aggregate values and never derives a new
   scientific estimate or raises the run's claim authority. */
(function () {
  'use strict';

  function rows(payload) {
    const tables = payload && Array.isArray(payload.tables) ? payload.tables : [];
    const seen = new Set();
    return tables.filter(table => table && !table.is_source_copy
      && !/source_data|publication_figure_source_|copied beside a promoted publication figure/i.test(`${table.name || ''} ${table.label || ''}`)
      && (!table.source_status || table.source_status !== 'stale')).filter(table => {
      // Repeat transport entries are the same evidence only when both their
      // verified identity and projected contents agree. Equal numbers alone
      // never merge independent producers, models or conflicting snapshots.
      const source = tableSource(table);
      if (!source) return true;
      const key = JSON.stringify([source, table.produced_by_step, table.headers, table.rows,
        table.row_model_ids, table.preview_truncated, table.group_by, table.descriptive_summary]);
      if (seen.has(key)) return false;
      seen.add(key);
      return true;
    }).flatMap(table => {
      const compact = table.descriptive_summary;
      if (tableSource(table) && compact && compact.schema_version === 'easyicu.web-descriptive-summary/1'
        && Array.isArray(compact.continuous) && Array.isArray(compact.categorical)
        && compact.continuous.length + compact.categorical.length <= 64
        && compact.continuous.concat(compact.categorical)
          .every(record => record && typeof record === 'object' && !Array.isArray(record))) {
        // This block covers the full verified aggregate, independently of
        // the display preview. Keep its original evidence/table identity.
        const complete = { ...table, preview_truncated: false };
        return compact.continuous.concat(compact.categorical)
          .map((record, rowIndex) => ({ record, headers: Object.keys(record), table: complete, rowIndex }));
      }
      const headers = Array.isArray(table && table.headers) ? table.headers.map(String) : [];
      if (!headers.length) return [];
      return (Array.isArray(table.rows) ? table.rows : []).map((values, rowIndex) => {
        const record = {};
        headers.forEach((header, index) => { record[header] = Array.isArray(values) ? values[index] : null; });
        if (!headers.includes('model_id') && Array.isArray(table.row_model_ids)) record.model_id = table.row_model_ids[rowIndex];
        return { record, headers, table, rowIndex };
      });
    });
  }

  function finite(value) {
    if (value == null || (typeof value === 'string' && value.trim() === '')) return null;
    const number = Number(value);
    return Number.isFinite(number) ? number : null;
  }

  function integerDisplay(value) {
    const number = finite(value);
    return number == null ? '' : Math.round(number).toLocaleString('en-US');
  }

  function percentDisplay(value) {
    const number = finite(value);
    return number == null ? '' : `${number.toFixed(2)}%`;
  }

  // A grouped exposure's levels are codes; the run's grouping record names
  // them in the page's language (screens-agent-exposure-levels.js), before
  // any label the plan registers for its English figures and manuscript.
  // Any other variable's levels go by the plan's labels.
  function groupLevels(context) {
    const owner = window.AGENT_EXPOSURE_LEVELS;
    return owner ? owner.reader(context, window.t || (en => en)) : null;
  }

  // Distribution and estimate sentences use one registered name; an
  // unlabelled code keeps its variable so it is never presented as a name.
  function levelLabel(variable, value, labels, levels) {
    const text = String(value == null ? '' : value);
    const numeric = /^-?(?:0|[1-9]\d*)(?:\.\d+)?$/.test(text);
    const keys = [`${variable}=${JSON.stringify(value)}`, `${variable}=${text}`,
      ...(numeric ? [`${variable}=${Number(text)}`, `${variable}=${JSON.stringify(String(Number(text)))}`] : [])];
    const registered = keys.map(key => labels[key]).find(label => typeof label === 'string' && label.trim());
    return (levels && levels.levelName(variable, value)) || registered
      || `${variable}=${numeric ? Number(text) : text}`;
  }

  function claim(sourceField, value, displayValue, source) {
    if (value == null || displayValue === '') return null;
    const table = source && source.table;
    return {
      source_field: sourceField,
      canonical_value: value,
      display_value: displayValue,
      // A CSV file is the verified destination. No scalar JSON mapping was
      // recorded, so a fabricated pointer must not enter exact-value checks.
      source_json_pointer: '',
      evidence: tableSource(table) || {},
    };
  }

  function tableSource(table) {
    if (!table || (table.source_status && table.source_status !== 'current')) return null;
    const evidenceId = String(table.evidence_id || '');
    const digest = String(table.sha256 || '').toLowerCase();
    return /^[A-Za-z0-9_.-]{1,160}$/.test(evidenceId) && /^[a-f0-9]{64}$/.test(digest)
      ? { evidence_id: evidenceId, sha256: digest, kind: 'table' } : null;
  }

  // The adjusted-association producer's typed estimate table.  Only its own
  // columns identify a row as an effect estimate; a table that merely has an
  // "estimate" column is not one.
  const ESTIMATE_COLUMNS = ['fit_status', 'estimate', 'ci_low', 'ci_high', 'effect_scale', 'is_primary_contrast'];
  const FITTED = new Set(['fitted', 'ok', 'converged', 'success']);
  const MEASURES = {
    odds_ratio: 'OR', hazard_ratio: 'HR', risk_ratio: 'RR', relative_risk: 'RR', risk_difference: 'RD',
  };

  function isTrue(value) {
    return ['true', '1', 'yes'].includes(String(value == null ? '' : value).trim().toLowerCase());
  }

  function ratioDisplay(value, measure) {
    const number = finite(value);
    if (number == null) return '';
    return measure === 'RD' ? number.toFixed(3) : number.toFixed(2);
  }

  // Only the host's complete, current secondary-product binding supplies
  // stratum estimates. A raw preview, filename or primary-model row cannot.
  function stratifiedResult(payload, plan) {
    const tables = (Array.isArray(payload?.tables) ? payload.tables : [])
      .filter(table => table && table.stratified_summary && !table.is_source_copy
        && table.source_status === 'current' && tableSource(table));
    if (tables.length !== 1) return null;
    const table = tables[0], block = table.stratified_summary;
    const owners = (plan?.steps || []).filter(step => step.step_id === table.produced_by_step);
    if (owners.length !== 1) return null;
    const step = owners[0];
    if (step.planned_analysis_role !== 'secondary'
      || !['association.effect_modification', 'time_to_event.subgroup_hr'].includes(step.scientific_action_id)
      || !(step.expected_outputs || []).includes(block.product)
      || !(step.inputs || []).includes(block.subgroup_column)
      || block.schema_version !== 'easyicu.web-stratified-summary/1'
      || !block.contrast || !Array.isArray(block.strata) || block.strata.length < 2 || block.strata.length > 6) return null;
    const measure = MEASURES[block.effect_scale], p = finite(block.interaction_p_value);
    if (!measure || p == null || p < 0 || p > 1) return null;
    const seen = new Set();
    const strata = block.strata.map(row => {
      if (!row || !row.level || seen.has(row.level)) return null;
      seen.add(row.level);
      const values = [row.estimate, row.ci_low, row.ci_high].map(finite);
      if (values.some(value => value == null) || values[1] > values[0] || values[0] > values[2]
        || (measure !== 'RD' && values[1] <= 0)) return null;
      return { level: String(row.level), value: ratioDisplay(values[0], measure),
        low: ratioDisplay(values[1], measure), high: ratioDisplay(values[2], measure) };
    });
    const displayP = window.EU_GUIDED_CONTRACTS?.fmtP;
    return strata.every(Boolean) ? { variable: block.subgroup_column, contrast: block.contrast,
      measure, strata, p: displayP ? displayP(p) : p < 0.001 ? '<0.001'
        : p.toFixed(3).replace(/0+$/, '').replace(/\.$/, ''), source: tableSource(table) } : null;
  }

  function primaryRecords(records, plan, requireProducer = false) {
    const steps = new Map((Array.isArray(plan && plan.steps) ? plan.steps : [])
      .filter(step => step && step.step_id).map(step => [step.step_id, step.planned_analysis_role]));
    return records.filter(item => {
      const role = steps.get(item.table.produced_by_step);
      return role === 'primary' || (!requireProducer && role == null);
    });
  }

  // Only a primary producer's registered aggregate table can answer a
  // continuous descriptive question. No patient rows, unit conversion,
  // inferred denominator, or combination of independent tables is involved.
  function descriptivePartitions(candidates) {
    const overall = item => item.record.row_role === 'overall'
      || String(item.record.group || '').trim().toLowerCase() === 'overall';
    const grouped = new Set(candidates.filter(item => !overall(item)).map(item => item.record.variable));
    return candidates.filter(item => !overall(item) || !grouped.has(item.record.variable));
  }

  function descriptiveDisplay(item, variable) {
    const display = item.table.descriptive_summary?.variable_display?.[variable];
    return display && typeof display === 'object' && !Array.isArray(display) ? display : {};
  }

  function continuousGroups(records, plan) {
    const candidates = primaryRecords(records, plan, true).filter(item =>
      tableSource(item.table) && ['variable', 'group'].every(key => item.headers.includes(key))
      && ['n_nonmissing', 'nonmissing_n', 'n'].some(key => item.headers.includes(key))
      && ['median', 'mean'].some(key => item.headers.includes(key)));
    if (new Set(candidates.map(item => item.table)).size !== 1
      || candidates.some(item => item.table.preview_truncated)) return [];
    const selected = descriptivePartitions(candidates);
    const seen = new Set();
    const groups = selected.map(item => {
      const row = item.record;
      const variable = String(row.variable || '').trim();
      const group = String(row.group == null ? '' : row.group).trim();
      const key = JSON.stringify([variable, group]);
      const countKey = ['n_nonmissing', 'nonmissing_n', 'n'].find(name => item.headers.includes(name));
      const n = finite(row[countKey]);
      if (!variable || (!group && row.row_role !== 'overall') || seen.has(key) || !Number.isSafeInteger(n) || n < 0) return null;
      seen.add(key);
      const spec = (plan.steps || []).find(step => step.step_id === item.table.produced_by_step)?.table_one_spec;
      const declared = (spec && spec.variables || []).find(entry => entry.name === variable);
      if (declared && declared.variable_kind !== 'continuous') return null;
      const kinds = ['median', 'mean'];
      const statistics = [];
      for (const kind of kinds) {
        const columns = kind === 'median' ? ['median', 'q25', 'q75'] : ['mean', 'sd'];
        if (!item.headers.includes(columns[0])) continue;
        const values = columns.map(name => finite(row[name]));
        if (values.every(value => value == null)) continue;
        const spread = columns.slice(1).some(name => values[columns.indexOf(name)] != null
          || ![null, undefined, '', 'NA'].includes(row[name]));
        if (n === 0 || values[0] == null || (spread && (values.some(value => value == null)
          || (kind === 'median' ? values[1] > values[0] || values[0] > values[2] : values[1] < 0)))) return null;
        statistics.push({ kind, values: (spread ? columns : columns.slice(0, 1))
          .map(name => String(row[name]).trim()) });
      }
      const groupVariable = String(row.group_variable || item.table.group_by || '').trim();
      return { variable, group: row.row_role === 'overall' || group.toLowerCase() === 'overall' ? '' : group,
        groupVariable, display: descriptiveDisplay(item, variable), groupDisplay: descriptiveDisplay(item, groupVariable),
        n, statistics, unit: String(row.unit || '').trim(), source: tableSource(item.table) };
    });
    return groups.every(Boolean) && groups.some(row => row.statistics.length) ? groups : [];
  }

  function categoricalGroups(records, plan) {
    const candidates = primaryRecords(records, plan, true).filter(item => tableSource(item.table)
      && ['variable', 'group', 'level', 'numerator', 'denominator', 'percentage']
        .every(key => item.headers.includes(key)));
    if (new Set(candidates.map(item => item.table)).size !== 1
      || candidates.some(item => item.table.preview_truncated)) return [];
    const seen = new Set();
    const groups = descriptivePartitions(candidates).map(item => {
      const row = item.record;
      const variable = String(row.variable || '').trim();
      const group = String(row.group == null ? '' : row.group).trim();
      const level = String(row.level == null ? '' : row.level).trim();
      const numerator = finite(row.numerator);
      const denominator = finite(row.denominator);
      const percentage = finite(row.percentage);
      const key = JSON.stringify([variable, group, level]);
      if (!variable || (!group && row.row_role !== 'overall') || !level || seen.has(key)
        || !Number.isSafeInteger(numerator) || !Number.isSafeInteger(denominator)
        || numerator < 0 || denominator <= 0 || numerator > denominator
        || percentage == null || percentage < 0 || percentage > 100) return null;
      seen.add(key);
      const groupVariable = String(row.group_variable || item.table.group_by || '').trim();
      return { variable, group: row.row_role === 'overall' || group.toLowerCase() === 'overall' ? '' : group,
        groupVariable, display: descriptiveDisplay(item, variable), groupDisplay: descriptiveDisplay(item, groupVariable),
        level, numerator, denominator,
        percentage: String(row.percentage).trim(), source: tableSource(item.table) };
    });
    return groups.every(Boolean) ? groups : [];
  }

  // One registered primary model's contrasts, primary contrast first.  Any
  // doubt -- no primary table, two of them, an unknown effect scale -- returns
  // nothing rather than a guessed headline.
  function primaryModels(records) {
    const byTable = new Map();
    records.filter(item => ESTIMATE_COLUMNS.every(header => item.headers.includes(header))).forEach(item => {
      if (!byTable.has(item.table)) byTable.set(item.table, []);
      byTable.get(item.table).push(item);
    });
    const primaryRows = items => items.filter(item => !item.headers.includes('analysis_role')
      || String(item.record.analysis_role || '').trim().toLowerCase() === 'primary');
    return [...byTable.values()]
      .map(primaryRows)
      .flatMap(items => {
        const models = new Map();
        items.forEach(item => {
          const id = String(item.record.model_id || '');
          if (!models.has(id)) models.set(id, []);
          models.get(id).push(item);
        });
        return [...models.values()];
      })
      .filter(items => items.some(item => isTrue(item.record.is_primary_contrast)));
  }

  function estimates(candidates, plan, levels) {
    if (candidates.length !== 1 || candidates[0].filter(item => isTrue(item.record.is_primary_contrast)).length !== 1) return [];
    const fitted = candidates[0].filter(item => FITTED.has(String(item.record.fit_status || '').trim().toLowerCase())
      && finite(item.record.estimate) != null);
    const scales = new Set(fitted.map(item => String(item.record.effect_scale || '').trim().toLowerCase()));
    const measure = scales.size === 1 ? MEASURES[[...scales][0]] : null;
    if (!measure || !fitted.some(item => isTrue(item.record.is_primary_contrast))) return [];
    const labels = plan && plan.display_labels || {};
    return fitted
      .sort((a, b) => Number(isTrue(b.record.is_primary_contrast)) - Number(isTrue(a.record.is_primary_contrast)))
      .map(item => {
        const record = item.record;
        const exposure = String(record.exposure || '');
        const level = String(record.exposure_level == null ? '' : record.exposure_level).trim();
        const reference = String(record.reference_level == null ? '' : record.reference_level).trim();
        const contrast = String(record.contrast || '').trim() || (level && reference ? `${level} vs ${reference}` : level);
        const label = (levels && levels.contrastName(exposure, contrast)) || (level && reference
          ? `${levelLabel(exposure, level, labels, levels)} vs ${levelLabel(exposure, reference, labels, levels)}`
          : contrast || (typeof labels[exposure] === 'string' ? labels[exposure] : (levels && levels.variableName(exposure)) || exposure));
        const value = finite(record.estimate);
        const low = finite(record.ci_low);
        const high = finite(record.ci_high);
        return {
          label, contrast, measure, value, low, high, n: finite(record.n),
          evidenceId: String(item.table.evidence_id || ''),
          source: tableSource(item.table),
          primary: isTrue(record.is_primary_contrast),
          display: { value: ratioDisplay(value, measure), low: ratioDisplay(low, measure), high: ratioDisplay(high, measure) },
        };
      });
  }

  // A continuous exposure's curve contrasts: the adjusted odds ratio at one
  // exposure value against a reference value, as the spline producer writes
  // them.  Which row is the run's answer is bound by its evidence claim, not
  // chosen here, so every row is offered with the table it came from.
  const CURVE_CONTRAST_COLUMNS = ['exposure', 'exposure_value', 'reference_exposure_value', 'adjusted_odds_ratio', 'ci_low', 'ci_high'];

  function curveContrasts(records) {
    return records.filter(item => CURVE_CONTRAST_COLUMNS.every(header => item.headers.includes(header)))
      .map(item => ({
        evidenceId: String(item.table && item.table.evidence_id || ''),
        producerStep: String(item.table && item.table.produced_by_step || ''),
        n: finite(item.record.exposure_density_population_n),
        exposure: String(item.record.exposure || ''),
        value: finite(item.record.exposure_value),
        reference: finite(item.record.reference_exposure_value),
        ratio: finite(item.record.adjusted_odds_ratio),
        low: finite(item.record.ci_low),
        high: finite(item.record.ci_high),
      }))
      .filter(row => row.evidenceId && row.exposure && [row.value, row.reference, row.ratio, row.low, row.high]
        .every(number => number != null));
  }

  function summarize(payload, plan, context) {
    const records = rows(payload);
    const primary = primaryRecords(records, plan);
    const models = primaryModels(primary);
    const levels = groupLevels(context);
    const candidates = records.filter(item => [
      'row_role', 'n_rows', 'exposure_denominator', 'exposure_pct',
      'outcome_events', 'outcome_denominator', 'outcome_rate_pct',
    ].every(header => item.headers.includes(header)));
    // Never combine independent primary/sensitivity tables into one cohort.
    const distribution = new Set(candidates.map(item => item.table)).size === 1 ? candidates : [];
    const specs = (Array.isArray(plan && plan.steps) ? plan.steps : [])
      .filter(step => step && step.planned_analysis_role === 'primary' && step.exposure_outcome_distribution_spec)
      .map(step => step.exposure_outcome_distribution_spec);
    // Model-led plans register the same exposure on their primary model,
    // without a separate primary distribution step (absolute risk is context).
    const requirements = (plan?.steps || []).filter(step => step?.planned_analysis_role === 'primary')
      .flatMap(step => step.model_requirements || [])
      .filter(row => row?.analysis_role === 'primary' && row.exposure_source);
    const model = specs.length === 0 && requirements.length === 1 ? requirements[0] : null;
    const spec = specs.length === 1 ? specs[0] : model ? {
      exposure: model.exposure_source, outcome: model.outcome, exposure_levels: model.exposure_levels,
    } : null;
    const labels = plan && plan.display_labels || {};
    const overall = distribution.find(item => String(item.record.row_role || '') === 'overall') || null;
    const exposureLevels = distribution
      .filter(item => String(item.record.row_role || '') === 'exposure_level')
      .map(item => {
        const raw = item.record.exposure_level;
        const variable = spec?.exposure || item.record.exposure || item.table.group_by || 'exposure';
        return {
          level: String(raw == null ? '' : raw),
          label: levelLabel(variable, raw, labels, levels),
          n: finite(item.record.n_rows),
          sharePct: finite(item.record.exposure_pct),
          events: finite(item.record.outcome_events),
          denominator: finite(item.record.outcome_denominator),
          missingN: finite(item.record.outcome_missing_n),
          outcomeRatePct: finite(item.record.outcome_rate_pct),
        };
      });
    // A population-flow ledger is a typed contract.  Audit summaries may also
    // expose generic `stage`/`n` columns, but those counts describe audited
    // concepts rather than patients and must never drive clinical denominators.
    const modelFlow = records.filter(item => [
      'stage', 'n', 'excluded_from_previous', 'population_rule',
    ].every(header => item.headers.includes(header)));
    const cohortRows = records.filter(item => item.headers.includes('n_before') && item.headers.includes('n_remaining'));
    const source = modelFlow.find(item => /source|universe/.test(String(item.record.stage || '').toLowerCase()))
      || cohortRows[0] || overall;
    const complete = [...modelFlow].reverse().find(item => /complete.*(case|model)|(case|model).*complete/.test(
      String(item.record.stage || '').toLowerCase(),
    )) || null;
    const beforeComplete = complete ? modelFlow.slice(0, modelFlow.indexOf(complete)) : modelFlow;
    const eligible = (beforeComplete.length ? beforeComplete[beforeComplete.length - 1] : null)
      || (cohortRows.length ? cohortRows[cohortRows.length - 1] : overall);
    const sourceN = source ? finite(source.record.n != null
      ? source.record.n : (source.record.n_before != null ? source.record.n_before : source.record.n_rows)) : null;
    const eligibleN = eligible ? finite(eligible.record.n != null
      ? eligible.record.n : (eligible.record.n_remaining != null ? eligible.record.n_remaining : eligible.record.n_rows)) : null;
    const completeN = complete ? finite(complete.record.n) : null;
    const eventN = overall ? finite(overall.record.outcome_events) : null;
    const exposureN = overall ? finite(overall.record.exposure_denominator) : null;
    const outcomeN = overall ? finite(overall.record.outcome_denominator) : null;
    const missingOutcomeN = overall ? finite(overall.record.outcome_missing_n) : null;
    const riskPct = overall ? finite(overall.record.outcome_rate_pct) : null;
    return {
      claims: [
        claim('cohort.n_stays', sourceN, integerDisplay(sourceN), source),
        claim('n_total', eligibleN, integerDisplay(eligibleN), eligible),
        claim('n_complete_case', completeN, integerDisplay(completeN), complete || eligible),
        claim('exposure_distribution.n_total', exposureN, integerDisplay(exposureN), overall),
        claim('n_events', eventN, integerDisplay(eventN), overall),
        claim('overall_outcome.event_n', eventN, integerDisplay(eventN), overall),
        claim('overall_outcome.n', outcomeN, integerDisplay(outcomeN), overall),
        claim('overall_outcome.missing_n', missingOutcomeN, integerDisplay(missingOutcomeN), overall),
        claim('overall_outcome.risk_pct', riskPct, percentDisplay(riskPct), overall),
        ...exposureLevels.map((row, index) => claim(
          `exposures[0].groups[${index}].outcome_risk_pct`,
          row.outcomeRatePct,
          percentDisplay(row.outcomeRatePct),
          distribution.filter(item => String(item.record.row_role || '') === 'exposure_level')[index],
        )),
      ].filter(Boolean),
      exposureSpec: spec,
      exposureLevels,
      continuousGroups: continuousGroups(records, plan),
      categoricalGroups: categoricalGroups(records, plan),
      answerSources: [...new Set(primaryRecords(records, plan, true).concat(models.flat()).map(item => item.table))]
        .map(tableSource).filter(Boolean)
        .filter((source, index, all) => all.findIndex(other => other.evidence_id === source.evidence_id
          && other.sha256 === source.sha256) === index),
      estimates: estimates(models, plan, levels),
      stratified: stratifiedResult(payload, plan),
      primaryModelAmbiguous: models.length > 1
        || models.some(items => items.filter(item => isTrue(item.record.is_primary_contrast)).length > 1),
      measurementMissingness: records.filter(item => item.table.evidence_id
        && ['concept', 'n_total', 'value_missing_n'].every(header => item.headers.includes(header)))
        .map(item => ({ concept: String(item.record.concept || ''), total: finite(item.record.n_total),
          missing: finite(item.record.value_missing_n), evidenceId: String(item.table.evidence_id) }))
        .filter(row => row.concept && Number.isSafeInteger(row.total) && row.total >= 0
          && Number.isSafeInteger(row.missing) && row.missing >= 0 && row.missing <= row.total),
      curveContrasts: curveContrasts(records),
      // The groups are shown by their codes because their record cannot be read.
      groupsUnreadable: !!(levels && levels.unreadable),
      outcomeLabel: spec && typeof labels[spec.outcome] === 'string' ? labels[spec.outcome] : '',
    };
  }

  window.EasyICU.guidedPi.declare('resultSummary', { summarize, levelLabel });
})();
