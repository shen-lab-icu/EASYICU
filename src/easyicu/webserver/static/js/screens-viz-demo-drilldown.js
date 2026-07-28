/* Clinically constrained demo Patient Review drilldown owner. */
(function () {
  const {
    catalogModuleLabel, catalogFeatureMeta,
    DEMO_ENTITY_COUNT, DEMO_CHART_HOURS,
    demoCatalogModules, demoIsTimeIndexed, demoFeatureModule,
    demoReviewStatus, demoQualityStatus, demoRateTone,
    demoTableValue, demoCharttimeAt, demoSignal, demoTimeLanes,
    demoHasClinicalModel, demoScenarioName, demoCategorySection, demoQualityPanelRows,
  } = window.VIZ_DEMO;

  function demoTablePreviewRowContext(rowIndex, timeIndexed) {
    const idx = Math.max(0, Number(rowIndex) || 0);
    if (!timeIndexed) {
      return {
        entityIndex: idx,
        timeIndex: 0,
        entityRef: `demo_ent_${idx + 1}`,
        charttime: null,
        valueSeed: idx,
      };
    }
    const timepointsPerEntity = 12;
    const entityIndex = Math.floor(idx / timepointsPerEntity);
    const timeIndex = idx % timepointsPerEntity;
    return {
      entityIndex,
      timeIndex,
      entityRef: `demo_ent_${entityIndex + 1}`,
      charttime: demoCharttimeAt(timeIndex),
      valueSeed: entityIndex * 13 + timeIndex,
    };
  }

  function buildPatientDrilldown(selectedRef) {
    const modules = demoCatalogModules();
    const totalFeatures = modules.reduce((acc, row) => acc + row.features.length, 0);
    const moduleProfiles = modules.map(row => {
      const modeledFeatures = row.features.filter(demoHasClinicalModel);
      const timeIndexed = demoIsTimeIndexed(row.module);
      const typicalRows = timeIndexed && modeledFeatures.length
        ? Math.round(DEMO_ENTITY_COUNT * modeledFeatures.reduce(
          (sum, feature) => sum + demoSignal(feature, 0).values.length,
          0,
        ) / modeledFeatures.length)
        : (modeledFeatures.length ? DEMO_ENTITY_COUNT : 0);
      const coverage = modeledFeatures.length ? 100 : 0;
      return {
        module: row.module,
        label: row.label,
        rows: typicalRows,
        feature_count: row.features.length,
        observed_features: modeledFeatures.length,
        entities: modeledFeatures.length ? DEMO_ENTITY_COUNT : 0,
        coverage_pct: coverage,
        time_indexed: timeIndexed,
        dynamic_features: timeIndexed ? modeledFeatures.length : 0,
        static_features: timeIndexed ? 0 : modeledFeatures.length,
        preview_features: modeledFeatures.slice(0, 6),
      };
    });
    const observedFeatures = moduleProfiles.reduce((acc, row) => acc + row.observed_features, 0);
    const totalRows = moduleProfiles.reduce((acc, row) => acc + (Number(row.rows) || 0), 0);
    const tableModules = moduleProfiles.map(row => ({
      module: row.module,
      label: row.label,
      review_features: row.feature_count,
      observed_features: row.observed_features,
      rows: row.rows,
      entities: row.entities,
      coverage_pct: row.coverage_pct,
      share_pct: totalFeatures ? Number((row.feature_count / totalFeatures * 100).toFixed(1)) : null,
      shape: row.time_indexed ? 'time_indexed' : 'static',
      dynamic_features: row.dynamic_features,
      static_features: row.static_features,
      preview_features: row.preview_features.map(feature => {
        const meta = catalogFeatureMeta(feature);
        return { feature, name: meta.name || feature, unit: meta.unit || '', group: row.label };
      }),
      status: demoReviewStatus(row.coverage_pct, row.feature_count),
    }));
    const selectedIndex = Math.max(0, Math.min(4, Number(String(selectedRef || 'demo_ent_1').replace(/\D+/g, '')) - 1 || 0));
    const entitySummary = idx => {
      const sofaValues = demoSignal('sofa2', idx).values.map(Number).filter(Number.isFinite);
      const sofa2Max = sofaValues.length ? Math.max(...sofaValues) : null;
      const outcome = demoTableValue('death', idx, DEMO_CHART_HOURS.length - 1) ? 'Deceased' : 'Survived';
      return {
        ref: `demo_ent_${idx + 1}`,
        label: `Synthetic entity ${idx + 1}`,
        ordinal: idx + 1,
        outcome,
        severity: `SOFA-2 max ${sofa2Max == null ? '—' : sofa2Max}`,
        scenario: demoScenarioName(idx),
        sofa2Max,
      };
    };
    const entities = [0, 1, 2, 3, 4].map(entitySummary);
    const timeLanes = demoTimeLanes(selectedIndex);
    const signalIndex = {};
    timeLanes.forEach(lane => (lane.signals || []).forEach(sig => { if (!signalIndex[sig.feature]) signalIndex[sig.feature] = sig; }));
    const selected = {
      ref: entities[selectedIndex].ref,
      label: entities[selectedIndex].label,
      ordinal: selectedIndex + 1,
      demographics: {
        age: demoTableValue('age', selectedIndex, 0),
        sex: demoTableValue('sex', selectedIndex, 0),
      },
      scores: {
        sofa2_max: entities[selectedIndex].sofa2Max,
        sepsis3_sofa2: demoTableValue('sep3_sofa2', selectedIndex, 0),
      },
      outcomes: {
        status: entities[selectedIndex].outcome,
        icu_los_days: demoTableValue('los_icu', selectedIndex, DEMO_CHART_HOURS.length - 1),
      },
      signals: (timeLanes.find(l => l.lane === 'vitals') || {}).signals || [],
    };
    const qualityRows = [];
    modules.forEach(moduleRow => {
      moduleRow.features.filter(demoHasClinicalModel).forEach(feature => {
        const timeIndexed = demoIsTimeIndexed(moduleRow.module);
        const records = timeIndexed
          ? Array.from({ length: DEMO_ENTITY_COUNT }, (_, idx) => demoSignal(feature, idx).values.length)
            .reduce((sum, count) => sum + count, 0)
          : DEMO_ENTITY_COUNT;
        const coverage = 100;
        const missing = 0;
        const outlier = 0;
        const duplicate = 0;
        const meta = catalogFeatureMeta(feature);
        qualityRows.push({
          feature,
          name: meta.name || feature,
          module: moduleRow.module,
          records,
          entities: DEMO_ENTITY_COUNT,
          coverage_pct: coverage,
          missing_pct: missing,
          out_of_physio_pct: outlier,
          duplicate_time_pct: duplicate,
          density_per_entity: Number((records / DEMO_ENTITY_COUNT).toFixed(3)),
          time_indexed: timeIndexed,
          status: demoQualityStatus(missing, outlier, duplicate),
        });
      });
    });
    const weight = qualityRows.reduce((acc, row) => acc + Math.max(row.records, 1), 0) || 1;
    const weighted = key => Number((qualityRows.reduce((acc, row) => acc + (Number(row[key]) || 0) * Math.max(row.records, 1), 0) / weight).toFixed(1));
    const qualitySummary = {
      concept_count: qualityRows.length,
      total_records: qualityRows.reduce((acc, row) => acc + row.records, 0),
      weighted_missing_pct: weighted('missing_pct'),
      weighted_out_of_physio_pct: weighted('out_of_physio_pct'),
      weighted_duplicate_time_pct: weighted('duplicate_time_pct'),
      denominator_entities: DEMO_ENTITY_COUNT,
    };
    const topIssues = qualityRows.slice().sort((a, b) =>
      (b.missing_pct - a.missing_pct) || (b.out_of_physio_pct - a.out_of_physio_pct) || (b.records - a.records)
    ).slice(0, 5);
    const quality = moduleProfiles.map(row => ({
      module: row.module,
      rows: row.rows,
      column_count: row.feature_count,
      covered_entities: row.entities,
      coverage_pct: row.coverage_pct,
      quality_status: row.coverage_pct >= 80 ? 'ok' : (row.coverage_pct >= 50 ? 'warn' : 'bad'),
    }));
    const readyLanes = timeLanes.filter(row => row.status === 'ready' && (row.signals || []).length);
    const loadedSignals = readyLanes.reduce((acc, row) => acc + (row.signals || []).length, 0);
    const comparisonFeatures = qualityRows.filter(row => row.time_indexed).sort((a, b) => b.records - a.records).slice(0, 8)
      .map(row => ({ feature: row.feature, name: row.name, module: row.module, records: row.records, entities: row.entities, coverage_pct: row.coverage_pct, density_per_entity: row.density_per_entity }));
    const compareFeature = (comparisonFeatures[0] && comparisonFeatures[0].feature) || 'hr';
    const compareMeta = catalogFeatureMeta(compareFeature);
    const compareModule = demoFeatureModule(compareFeature);
    const comparisonTraces = entities.map((entity, idx) => {
      const signal = demoSignal(compareFeature, idx);
      const values = (signal.values || []).map(Number).filter(Number.isFinite);
      return {
        ref: entity.ref,
        label: entity.label,
        values,
        times: (signal.times || []).slice(0, values.length),
        time_axis: signal.time_axis,
        point_count: signal.point_count,
        bounded: true,
        max_points: signal.max_points,
      };
    }).filter(trace => trace.values.length >= 2);
    const sections = [
      demoCategorySection('vitals', 'Vital Signs Snapshot', ['hr', 'map', 'sbp', 'dbp', 'resp', 'temp', 'spo2'], signalIndex),
      demoCategorySection('labs', 'Key Laboratory Snapshot', ['lact', 'crea', 'plt', 'wbc', 'hgb', 'bili', 'glu'], signalIndex),
      demoCategorySection('scores', 'Scores and sepsis flags', ['sofa', 'sofa2', 'qsofa', 'sirs', 'gcs'], signalIndex),
      demoCategorySection('support', 'Support and therapies', ['mech_vent', 'vent_ind', 'rrt', 'norepi_rate', 'epi_rate', 'peep'], signalIndex),
    ];
    const cohortEntities = Array.from({ length: DEMO_ENTITY_COUNT }, (_, idx) => entitySummary(idx));
    const cohortAges = cohortEntities.map((_, idx) => Number(demoTableValue('age', idx, 0))).filter(Number.isFinite);
    const cohortFemale = cohortEntities.filter((_, idx) => demoTableValue('sex', idx, 0) === 'F').length;
    const cohortDeaths = cohortEntities.filter(row => row.outcome === 'Deceased').length;
    const cohortLos = cohortEntities.map((_, idx) => Number(demoTableValue('los_icu', idx, DEMO_CHART_HOURS.length - 1))).filter(Number.isFinite).sort((a, b) => a - b);
    const cohortSofa = cohortEntities.map(row => Number(row.sofa2Max)).filter(Number.isFinite).sort((a, b) => a - b);
    const cohortSepsis = cohortEntities.filter((_, idx) => demoTableValue('sep3_sofa2', idx, 0)).length;
    const midpoint = values => values.length
      ? (values.length % 2 ? values[Math.floor(values.length / 2)] : (values[values.length / 2 - 1] + values[values.length / 2]) / 2)
      : null;
    const summary = {
      entities: DEMO_ENTITY_COUNT,
      modules: modules.length,
      file_count: modules.length,
      total_rows: totalRows,
      review_entities: DEMO_ENTITY_COUNT,
      review_entity_cap: DEMO_ENTITY_COUNT,
      review_scope: 'clinically_constrained_synthetic_icu_cohort',
      static_aggregate_scope: 'synthetic_rule_based_v1',
      dynamic_aggregate_scope: 'synthetic_irregular_48h_window',
      mean_age: Number((cohortAges.reduce((sum, value) => sum + value, 0) / cohortAges.length).toFixed(1)),
      female_pct: Number((cohortFemale / DEMO_ENTITY_COUNT * 100).toFixed(1)),
      mortality: Number((cohortDeaths / DEMO_ENTITY_COUNT * 100).toFixed(1)),
      median_los_icu: Number(midpoint(cohortLos).toFixed(1)),
      median_sofa2: Number(midpoint(cohortSofa).toFixed(1)),
      sepsis_pct: Number((cohortSepsis / DEMO_ENTITY_COUNT * 100).toFixed(1)),
    };
    const tablePreviews = tableModules.slice(0, 32).map((row, moduleIdx) => {
      const timeIndexed = row.shape === 'time_indexed';
      const features = (row.preview_features || []).map(f => f.feature || f.name).filter(Boolean).slice(0, timeIndexed ? 6 : 8);
      const displayColumns = ['entity'].concat(timeIndexed ? ['charttime'] : []).concat(features);
      const previewLimit = timeIndexed ? 24 : 8;
      const previewRows = Array.from({ length: previewLimit }, (_, idx) => {
        const context = demoTablePreviewRowContext(idx, timeIndexed);
        const out = { entity: context.entityRef };
        if (timeIndexed) out.charttime = context.charttime;
        features.forEach(feature => {
          out[feature] = demoTableValue(feature, context.entityIndex, context.timeIndex);
        });
        return out;
      });
      return {
        module: row.module,
        label: row.label,
        file: `${row.module}.demo`,
        rows_total: row.rows,
        columns_total: Math.max(row.review_features + (timeIndexed ? 2 : 1), displayColumns.length),
        display_columns: displayColumns,
        hidden_columns: Math.max(0, row.review_features - features.length),
        row_cap: previewRows.length,
        column_cap: displayColumns.length,
        pseudonymous_entity_column: true,
        status: 'ready',
        rows: previewRows,
        row_count: previewRows.length,
        truncated_rows: row.rows > previewRows.length,
        truncated_columns: row.review_features > features.length,
        payload_scope: 'synthetic_pseudonymous_module_table_preview',
      };
    });
    return {
      ok: true,
      mode: 'demo',
      demo: true,
      source: { label: 'Demo · clinically constrained synthetic ICU cohort', database: 'demo', path_hash: 'synthetic-rule-v1' },
      provenance: {
        computed_from: ['window.EU_CATALOG.groups', 'window.EU_CATALOG.groupConcepts', 'fully_synthetic_rule_based_v1'],
        payload_scope: 'clinically_constrained_synthetic_demo_no_real_patient_rows',
        signals: 'deterministic_correlated_synthetic_icu_trajectories',
        real_patient_rows_used: false,
      },
      demo_generation: {
        version: 'fully_synthetic_rule_based_v1',
        seed_policy: 'deterministic_entity_and_feature_hash',
        scenarios: ['shock_recovery', 'respiratory_recovery', 'late_deterioration', 'renal_dominant', 'stable_recovery'],
        constraints: ['irregular_feature_specific_cadence', 'correlated_hemodynamics', 'derived_scores', 'stepwise_interventions'],
        intended_use: 'interaction_and_layout_review_only',
        prohibited_use: 'clinical_inference_or_manuscript_results',
      },
      privacy: {
        raw_rows_returned: false,
        direct_identifiers_returned: false,
        max_entity_options: 5,
        max_points_per_signal: 12,
        payload_tables_are_aggregated: true,
      },
      summary,
      eligibility_flow: {
        title: 'Eligibility flow (ICU stays)',
        title_i18n: { en: 'Eligibility flow (ICU stays)', zh: '入组筛选流程（ICU 住院）' },
        has_stepwise_report: true,
        payload_scope: 'demo_cohort_attrition_metadata_only',
        privacy: { patient_rows_returned: false, direct_identifiers_returned: false },
        steps: [
          {
            id: 'source_total',
            label: 'All ICU stays',
            label_i18n: { en: 'All ICU stays', zh: '全部 ICU 住院' },
            count: 72,
            denominator: 72,
            pct_of_initial: 100,
            excluded: null,
            excluded_pct_of_previous: null,
            note: 'catalog-shaped demo source pool',
            note_i18n: { en: 'catalog-shaped demo source pool', zh: '目录形演示来源池' },
            basis: 'seeded_demo',
          },
          {
            id: 'adult_stay_filter',
            label: 'Adult first ICU stay',
            label_i18n: { en: 'Adult first ICU stay', zh: '成人首次 ICU 住院' },
            count: 56,
            denominator: 72,
            pct_of_initial: 77.8,
            excluded: 16,
            excluded_pct_of_previous: 22.2,
            note: 'age >= 18 · first stay',
            note_i18n: { en: 'age >= 18 · first stay', zh: '年龄 ≥ 18 · 首次 ICU' },
            basis: 'seeded_demo',
          },
          {
            id: 'target_clinical_cohort',
            label: 'Sepsis-3 cohort',
            label_i18n: { en: 'Sepsis-3 cohort', zh: 'Sepsis-3 脓毒症队列' },
            count: DEMO_ENTITY_COUNT,
            denominator: 72,
            pct_of_initial: Number((DEMO_ENTITY_COUNT / 72 * 100).toFixed(1)),
            excluded: 8,
            excluded_pct_of_previous: 14.3,
            note: 'suspected infection + SOFA signal',
            note_i18n: { en: 'suspected infection + SOFA signal', zh: '疑似感染 + SOFA 信号' },
            basis: 'seeded_demo',
          },
          {
            id: 'final_cohort',
            label: 'Final review cohort',
            label_i18n: { en: 'Final review cohort', zh: '最终审阅队列' },
            count: DEMO_ENTITY_COUNT,
            denominator: 72,
            pct_of_initial: Number((DEMO_ENTITY_COUNT / 72 * 100).toFixed(1)),
            excluded: 0,
            excluded_pct_of_previous: 0,
            note: 'UI preview only',
            note_i18n: { en: 'UI preview only', zh: '仅用于界面预览' },
            basis: 'seeded_demo',
            final: true,
          },
        ],
      },
      module_profiles: moduleProfiles,
      entities,
      selected,
      time_lanes: timeLanes,
      quality,
      quality_metrics: {
        summary: qualitySummary,
        features: qualityRows.slice(0, 80),
        top_issues: topIssues,
        payload_scope: 'catalog_seeded_quality_metrics_no_row_payload',
      },
      data_tables: {
        loaded_summary: {
          entities: DEMO_ENTITY_COUNT,
          review_features: totalFeatures,
          observed_features: observedFeatures,
          module_count: modules.length,
          source_count: 1,
        },
        module_picker: {
          default_module: tableModules[0] && tableModules[0].module,
          module_count: tableModules.length,
          selection_mode: 'module_then_feature',
        },
        detail_gate: {
          title: 'Clinically constrained synthetic demo; no source rows',
          default_open: false,
          reason: 'Modeled values share one deterministic synthetic stay model; unmodeled catalog concepts remain unavailable instead of receiving invented values.',
          available_detail_modes: ['module_glance', 'single_feature_metadata'],
        },
        modules: tableModules,
        table_previews: tablePreviews,
        payload_scope: 'easyicu_synthetic_demo_without_real_row_payload',
      },
      trajectory_review: {
        contract: [
          { index: '01', label: 'Entity scope', detail: `${entities.length} synthetic entity options exposed`, status: 'ready' },
          { index: '02', label: 'Loaded signals', detail: `${loadedSignals} clinically modeled signals`, status: 'ready' },
          { index: '03', label: 'Feature matrices', detail: `${readyLanes.length} matrix groups available`, status: 'ready' },
          { index: '04', label: 'Review mode', detail: 'clinical lanes / single entity / same-feature comparison', status: 'ready' },
        ],
        modes: [
          { id: 'feature_matrix', label: 'Feature Matrix', status: 'ready', description: 'Bounded time-window by feature matrices for grouped EasyICU catalog signals.' },
          { id: 'single_entity', label: 'Single Patient', status: 'ready', description: 'Selected synthetic entity trends and latest values.' },
          { id: 'multi_entity_comparison', label: 'Multi-Patient Comparison', status: 'ready', description: 'One selected feature compared across bounded pseudonymous entities.' },
        ],
        lanes: timeLanes,
        single_entity: { selected_ref: selected.ref, selected_label: selected.label, signals: selected.signals.slice(0, 12) },
        multi_entity_comparison: {
          selection_cap: 5,
          normalization_available: true,
          feature: compareFeature,
          label: compareMeta.name || compareFeature,
          unit: compareMeta.unit || '',
          time_axis: comparisonTraces[0] ? comparisonTraces[0].time_axis : {},
          module: compareModule,
          module_label: catalogModuleLabel(compareModule),
          traces: comparisonTraces,
          compared_entities: comparisonTraces.length,
          features: comparisonFeatures,
          payload_scope: 'synthetic_pseudonymous_multi_entity_same_feature_traces',
        },
        payload_scope: 'synthetic_demo_feature_matrix_semantics_bounded',
      },
      patient_overview: {
        navigator: {
          current: selected.label,
          ordinal: selected.ordinal,
          options: entities.map(item => ({ ref: item.ref, label: item.label, outcome: item.outcome, severity: item.severity })),
          actions: ['first', 'previous', 'next', 'last', 'random'],
        },
        dashboard: {
          mode: 'Dashboard',
          summary_cards: [
            { label: 'Age / sex', value: `${selected.demographics.age} / ${selected.demographics.sex}`, tone: 'neutral' },
            { label: 'SOFA-2 max', value: String(selected.scores.sofa2_max), tone: selected.scores.sofa2_max < 10 ? 'warn' : 'bad' },
            { label: 'Sepsis-3', value: selected.scores.sepsis3_sofa2 ? 'Positive' : 'Negative', tone: selected.scores.sepsis3_sofa2 ? 'warn' : 'ok' },
            { label: 'Outcome', value: selected.outcomes.status, tone: selected.outcomes.status === 'Deceased' ? 'bad' : 'ok' },
            { label: 'ICU LOS', value: `${selected.outcomes.icu_los_days} d`, tone: 'neutral' },
          ],
          trend_panels: sections.filter(s => s.available_count).slice(0, 3),
          sofa_comparator: signalIndex.sofa && signalIndex.sofa2
            ? { status: 'ready', features: [{ feature: 'sofa', label: 'SOFA-1', current: signalIndex.sofa.current, values: signalIndex.sofa.values }, { feature: 'sofa2', label: 'SOFA-2', current: signalIndex.sofa2.current, values: signalIndex.sofa2.values }] }
            : { status: 'unavailable', reason: 'SOFA-1 and SOFA-2 signals are both required.' },
        },
        category_view: { mode: 'Category View', sections },
        data_table: {
          mode: 'Data Table',
          available_features: totalFeatures,
          row_preview: 'blocked',
          reason: 'Demo preserves the Patient Overview contract without returning source rows.',
        },
        payload_scope: 'catalog_demo_patient_overview_semantics_pseudonymous',
      },
      quality_review: {
        summary_cards: [
          { label: 'QC concepts', value: qualitySummary.concept_count, tone: 'ok' },
          { label: 'Synthetic observations', value: qualitySummary.total_records, tone: 'accent' },
          { label: 'Weighted missing', value: qualitySummary.weighted_missing_pct, unit: '%', tone: demoRateTone(qualitySummary.weighted_missing_pct, 5, 20) },
          { label: 'Out-of-physio', value: qualitySummary.weighted_out_of_physio_pct, unit: '%', tone: demoRateTone(qualitySummary.weighted_out_of_physio_pct, 1, 5) },
          { label: 'Duplicate TS', value: qualitySummary.weighted_duplicate_time_pct, unit: '%', tone: demoRateTone(qualitySummary.weighted_duplicate_time_pct, 0.5, 2) },
        ],
        contract: [
          { index: '01', label: 'Modeled concept scope', detail: `${qualitySummary.concept_count} concepts · ${DEMO_ENTITY_COUNT} synthetic entities · ${qualitySummary.total_records} generated observations`, status: 'ready' },
          { index: '02', label: 'Missingness gate', detail: `${qualitySummary.weighted_missing_pct}% weighted missing`, status: demoRateTone(qualitySummary.weighted_missing_pct, 5, 20) },
          { index: '03', label: 'Physiologic range', detail: `${qualitySummary.weighted_out_of_physio_pct}% out-of-range values`, status: demoRateTone(qualitySummary.weighted_out_of_physio_pct, 1, 5) },
          { index: '04', label: 'Temporal integrity', detail: `${qualitySummary.weighted_duplicate_time_pct}% duplicate time rows`, status: demoRateTone(qualitySummary.weighted_duplicate_time_pct, 0.5, 2) },
        ],
        panels: [
          { id: 'missingness', label: 'Missingness', rows: demoQualityPanelRows(qualityRows.slice().sort((a, b) => b.missing_pct - a.missing_pct), 'missing_pct') },
          { id: 'outliers', label: 'Out-of-Physio', rows: demoQualityPanelRows(qualityRows.slice().sort((a, b) => b.out_of_physio_pct - a.out_of_physio_pct), 'out_of_physio_pct') },
          { id: 'temporal', label: 'Temporal Integrity', rows: demoQualityPanelRows(qualityRows.slice().sort((a, b) => b.duplicate_time_pct - a.duplicate_time_pct), 'duplicate_time_pct') },
        ],
        top_issues: topIssues,
        module_coverage: quality,
        payload_scope: 'catalog_demo_quality_semantics_aggregate_only',
      },
      blocked_features: [
        { id: 'demo_not_manuscript_result', status: 'blocked', reason: 'Synthetic demo values exercise the UI only; load a real export for analysis.' },
        { id: 'raw_identifier_table', status: 'blocked', reason: 'Patient Review returns aggregates and pseudonymous synthetic entities only.' },
      ],
    };
  }

  window.VIZ_DEMO_DRILLDOWN = { buildPatientDrilldown };
})();
