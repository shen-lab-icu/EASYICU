/* Registered result-table summary adapter for Guided Copilot reports.
   It selects already-produced aggregate values and never derives a new
   scientific estimate or raises the run's claim authority. */
(function () {
  'use strict';

  function rows(payload) {
    const tables = payload && Array.isArray(payload.tables) ? payload.tables : [];
    return tables.flatMap(table => {
      const headers = Array.isArray(table && table.headers) ? table.headers.map(String) : [];
      if (!headers.length) return [];
      return (Array.isArray(table.rows) ? table.rows : []).map((values, rowIndex) => {
        const record = {};
        headers.forEach((header, index) => { record[header] = Array.isArray(values) ? values[index] : null; });
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

  function claim(sourceField, value, displayValue, source) {
    if (value == null || displayValue === '') return null;
    const table = source && source.table;
    return {
      source_field: sourceField,
      canonical_value: value,
      display_value: displayValue,
      source_json_pointer: table && table.name
        ? `/tables/${String(table.name)}/rows/${Number(source.rowIndex || 0)}`
        : '',
      evidence: table && table.evidence_id ? { evidence_id: String(table.evidence_id) } : {},
    };
  }

  function summarize(payload, plan) {
    const records = rows(payload);
    const candidates = records.filter(item => [
      'row_role', 'n_rows', 'exposure_denominator', 'exposure_pct',
      'outcome_events', 'outcome_denominator', 'outcome_rate_pct',
    ].every(header => item.headers.includes(header)));
    // Never combine independent primary/sensitivity tables into one cohort.
    const distribution = new Set(candidates.map(item => item.table)).size === 1 ? candidates : [];
    const specs = (Array.isArray(plan && plan.steps) ? plan.steps : [])
      .filter(step => step && step.planned_analysis_role === 'primary' && step.exposure_outcome_distribution_spec)
      .map(step => step.exposure_outcome_distribution_spec);
    const spec = specs.length === 1 ? specs[0] : null;
    const labels = plan && plan.display_labels || {};
    const overall = distribution.find(item => String(item.record.row_role || '') === 'overall') || null;
    const exposureLevels = distribution
      .filter(item => String(item.record.row_role || '') === 'exposure_level')
      .map(item => {
        const raw = item.record.exposure_level;
        const matches = (Array.isArray(spec && spec.exposure_levels) ? spec.exposure_levels : [])
          .filter(level => String(level) === String(raw)
            || (typeof level === 'number' && typeof raw === 'string'
              && /^-?(?:0|[1-9]\d*)(?:\.\d+)?$/.test(raw) && Number(raw) === level));
        const key = matches.length === 1 ? `${spec.exposure}=${JSON.stringify(matches[0])}` : '';
        return {
        level: String(item.record.exposure_level == null ? '' : item.record.exposure_level),
        label: key && typeof labels[key] === 'string' ? labels[key] : String(raw == null ? '' : raw),
        n: finite(item.record.n_rows),
        sharePct: finite(item.record.exposure_pct),
        events: finite(item.record.outcome_events),
        denominator: finite(item.record.outcome_denominator),
        outcomeRatePct: finite(item.record.outcome_rate_pct),
      }; });
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
    const riskPct = overall ? finite(overall.record.outcome_rate_pct) : null;
    return {
      claims: [
        claim('cohort.n_stays', sourceN, integerDisplay(sourceN), source),
        claim('n_total', eligibleN, integerDisplay(eligibleN), eligible),
        claim('n_complete_case', completeN, integerDisplay(completeN), complete || eligible),
        claim('exposure_distribution.n_total', eligibleN, integerDisplay(eligibleN), overall || eligible),
        claim('n_events', eventN, integerDisplay(eventN), overall),
        claim('overall_outcome.event_n', eventN, integerDisplay(eventN), overall),
        claim('overall_outcome.risk_pct', riskPct, percentDisplay(riskPct), overall),
        ...exposureLevels.map((row, index) => claim(
          `exposures[0].groups[${index}].outcome_risk_pct`,
          row.outcomeRatePct,
          percentDisplay(row.outcomeRatePct),
          distribution.filter(item => String(item.record.row_role || '') === 'exposure_level')[index],
        )),
      ].filter(Boolean),
      exposureLevels,
      outcomeLabel: spec && typeof labels[spec.outcome] === 'string' ? labels[spec.outcome] : '',
    };
  }

  window.EasyICU.guidedPi.declare('resultSummary', { summarize });
})();
