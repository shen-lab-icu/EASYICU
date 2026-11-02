/* Read-only prepared-data, deferred-input scope and plan-population rendering. */
(function () {
  'use strict';
  function create(host) {
    const tr = host.tr;
    const esc = host.esc;
    function preparedDataStatus() {
      const authorization = (host.session() && host.session().data_source_authorization) || {};
      const scope = String(authorization.extraction_scope || '');
      const source = authorization.source || {};
      const sourceLabel = String(source.label || source.reference_release || source.database || '').trim();
      const titles = {
        reuse_prepared_full: tr('Prepared source data and planned variables are available', '已复用现有数据源，计划变量可用'),
        all_supported: tr('Source data and planned variables are available', '数据源与计划变量可用'),
        study_required: tr('Study data source and planned variables are available', '本研究数据源与计划变量可用'),
      };
      return {
        title: titles[scope] || tr('The source data and planned variables are available', '数据源与计划变量可用'),
        sourceLabel,
        detail: tr(
          'Approval reuses this source. The final analytic cohort, preprocessing, and models are still pending and run only after approval.',
          '批准后复用该数据源；最终分析队列、数据预处理和模型仍待执行，不会重复提取已有数据。',
        ),
      };
    }

    function validDeferredInputScope(scope) {
      if (!scope) return false;
      const digest = value => typeof value === 'string' && /^[a-f0-9]{64}$/.test(value);
      return scope.schema_version === 'easyicu.first-observed-review-scope/1'
        && scope.verified === true && digest(scope.recipe_sha256) && digest(scope.plan_sha256)
        && scope.pending_materialization === true && scope.actual_cohort_exists === false
        && scope.analysis_authorized === false
        && ['question', 'source_label', 'population_summary', 'interval_summary'].every(key => typeof scope[key] === 'string' && scope[key])
        && Array.isArray(scope.snapshot_files) && scope.snapshot_files.length > 0
        && scope.snapshot_files.every(file => file && typeof file.file === 'string'
          && (digest(file.sha256) || (file.sha256 == null && file.hash_status === 'computed_after_authorization')))
        && Array.isArray(scope.decoded_files) && scope.decoded_files.length > 0
        && scope.decoded_files.every(file => file && typeof file.file === 'string'
          && Array.isArray(file.columns) && file.columns.length > 0
          && file.columns.every(column => typeof column === 'string' && column));
    }

    function deferredInputScopeHtml(scope) {
      if (!scope) return '';
      if (!validDeferredInputScope(scope)) return `<div class="gpi-confirmation-data-status" role="status"><strong>${esc(tr('Data-read scope could not be verified', '无法核验本计划的数据读取范围'))}</strong><small>${esc(tr('Refresh the reviewed plan before approving. Analysis has not been authorized.', '请刷新并核验完整计划后再批准；科研分析尚未获授权。'))}</small></div>`;
      const budget = scope.budget || {};
      const number = key => Number.isFinite(budget[key]) && budget[key] >= 0 ? String(budget[key]) : tr('unavailable', '未核实');
      const files = scope.snapshot_files.map(file => `<li>${esc(file.file)} · ${esc(file.sha256 || tr('SHA-256 will be computed after approval', '批准后计算 SHA-256'))}</li>`).join('');
      const columns = scope.decoded_files.map(file => `<li>${esc(file.file)}: ${esc(file.columns.join(', '))}</li>`).join('');
      return `<div class="gpi-confirmation-data-status" data-gpi-deferred-input-scope>
        <strong>${esc(tr('Data preparation and analysis covered by this plan', '本次完整计划包含的数据准备与分析'))}</strong>
        <p>${esc(scope.question)}</p><p>${esc(tr('Source', '来源'))}: ${esc(scope.source_label)}</p>
        <p>${esc(scope.population_summary)}</p><p>${esc(scope.interval_summary)}</p>
        <p>${esc(tr('The private first-ICU coordinate is prepared. The analytic cohort has not been materialized and outcome analysis has not started. Approval authorizes the declared preparation followed by this same analysis plan.', '私有首次 ICU 坐标已准备；分析队列尚未物化，结局分析尚未开始。本次批准将授权下列数据准备，并按同一完整计划继续分析。'))}</p>
        <p>${esc(tr('Files are hashed in full. Decoding may read every listed column, including outcomes; this scope is broader than the earlier six-field preparation.', '将对文件完整字节计算摘要；解码可能读取下列全部字段，包括结局字段。此范围超过先前的六字段准备。'))}</p>
        <p>${esc(tr('Cumulative budget', '累计预算'))}: ${esc(number('used_calls'))}/${esc(number('max_calls'))} ${esc(tr('calls', '次调用'))} · ${esc(number('used_tokens'))}/${esc(number('max_tokens'))} tokens · $${esc(number('used_cost_usd'))}/$${esc(number('max_cost_usd'))} · ${esc(number('max_wall_time_seconds'))} s ${esc(tr('time limit; previous usage remains counted', '时限；先前用量继续计入'))}</p>
        <details><summary>${esc(tr('Exact files, decoded fields and plan identity', '完整文件、解码字段与计划身份'))}</summary>
          <strong>${esc(tr('Whole-file hashes', '完整文件摘要'))}</strong><ul>${files}</ul>
          <strong>${esc(tr('Maximum decoded fields', '最大解码字段范围'))}</strong><ul>${columns}</ul>
          ${/^[a-f0-9]{64}$/.test(scope.source_manifest_sha256 || '') ? `<p>${esc(tr('Source manifest', '来源清单'))}: ${esc(scope.source_manifest_sha256)}</p>` : ''}
          <p>${esc(tr('Plan', '计划'))}: ${esc(scope.plan_sha256)}</p><p>${esc(tr('Preparation recipe', '准备方案'))}: ${esc(scope.recipe_sha256)}</p>
        </details>
      </div>`;
    }

    function reviewedPopulationSpecHtml(projection) {
      if (!projection || projection.schema_version !== 'easyicu.reviewed_population_spec_projection/1' || !Array.isArray(projection.criteria)) return '';
      const missingLabels = {
        exclude: tr('exclude patients with unknown values', '排除无法判定的患者'),
        retain: tr('retain patients with unknown values', '保留无法判定的患者'),
        stop: tr('stop if any value is unknown', '存在无法判定的值时停止'),
      };
      const disposition = {
        applied_by_plan: tr('applied by the plan', '计划中已施加'),
        applied_by_source: tr('applied by the source', '来源已施加'),
        requires_extraction: tr('needs data preparation', '需要数据准备'),
        not_applied: tr('not applied', '尚未施加'),
      };
      const shownSourceQuotes = new Set();
      const rows = projection.criteria.map(item => {
        const c = item && item.criterion || {};
        const fields = [];
        const concept = Array.isArray(c.concepts_all_of) ? c.concepts_all_of.join(', ') : c.concept;
        if (concept) fields.push(`${tr('Concept', '概念')}: ${concept}`);
        if (c.kind === 'diagnosis_codes' && Array.isArray(c.codes)) fields.push(`${tr('Diagnosis code prefixes', '诊断编码前缀')} (${String(c.system || '').toUpperCase()}): ${c.codes.join(', ')}`);
        if (c.op && typeof c.value === 'number') fields.push(`${tr('Threshold', '阈值')}: ${c.op} ${c.value} ${c.unit || tr('(unit not stated)', '（单位未注明）')}`);
        if (c.summary) fields.push(`${tr('Summary', '汇总方式')}: ${c.summary}`);
        if (Object.prototype.hasOwnProperty.call(c, 'window')) {
          const w = c.window;
          fields.push(w ? `${tr('Window', '时间窗')}: [${w.start_hours}, ${w.end_hours}) ${tr('hours from ICU admission', '小时，以 ICU 入住为起点')}` : tr('Window: whole ICU stay', '时间窗：整个 ICU 入住期间'));
        }
        if (['measurement', 'condition_present', 'event_absent'].includes(c.kind)) fields.push(`${tr('Unknown values', '无法判定时')}: ${missingLabels[c.missing_policy] || tr('not specified', '未注明')}`);
        if (c.kind === 'event_absent') fields.push(tr('Event must be absent', '要求事件未发生'));
        const role = c.role === 'exclude' ? tr('Exclude', '排除') : tr('Include', '纳入');
        const ageBounds = [];
        if (typeof c.min_years === 'number') ageBounds.push(`≥ ${c.min_years}`);
        if (typeof c.max_years === 'number') ageBounds.push(`≤ ${c.max_years}`);
        const title = c.kind === 'age_years' && ageBounds.length
          ? `${tr('Age', '年龄')} ${ageBounds.join(' · ')} ${tr('years', '岁')}`
          : c.kind === 'first_observed_icu_single_hospital_adult'
            ? tr('Adult, one visible hospitalization and first verifiable ICU stay', '成年患者、仅一次可见住院且首次 ICU 可核实')
            : String(c.quote || '');
        const quote = String(c.quote || '');
        const sourceQuote = title !== quote && quote && !shownSourceQuotes.has(quote)
          ? `<small>${esc(tr('Source wording', '来源原文'))}: ${esc(quote)}</small>` : '';
        if (title !== quote && quote) shownSourceQuotes.add(quote);
        return `<li><strong>${esc(role)} · ${esc(title)}</strong>${sourceQuote}<small>${esc(fields.join(' · '))}</small><small>${esc(disposition[item.disposition] || tr('status unverified', '状态未核验'))}</small></li>`;
      }).join('');
      if (!rows) return '';
      return `<details class="gpi-confirmation-data-status" open><summary>${esc(tr('Population conditions in the reviewed plan', '待审计划中的人群条件'))}</summary><small>${esc(tr('These are the plan’s proposed conditions and decisions; review them before approval.', '这些是计划提出的条件和处理方式，请在批准前逐条审阅。'))}</small><ul>${rows}</ul></details>`;
    }

    function populationPreparationHtml(preparation) {
      if (!preparation) return '';
      const reasons = {
        population_preparation_all_stays_conflict: tr('Conflicts with your choice to retain all stays.', '与已选择的保留全部入住冲突。'),
        population_preparation_patient_hospitalization_scope_mismatch: tr('The question asks for each patient’s first ICU stay; the first ICU stay per hospitalization is different. Revise the plan and verify patient ordering before preparation.', '原问题要求每位患者首次 ICU 入住；每次住院首次 ICU 不是同一人群。请先修订计划并核实患者入住顺序。'),
        population_preparation_eicu_demo_first_observed_unavailable: tr('The requested eICU Demo subset needs a verified complete-source selection and exclusion record; this preparation is not available yet.', '所需的 eICU Demo 子人群需要完整来源核验与排除记录；该数据准备方式尚不可用。'),
        population_preparation_bound_unrepresentable: tr('Extraction cannot preserve these exact bounds.', '提取无法保留这组精确界限。'),
        population_preparation_empty_age_range: tr('Conflicts with the current age range.', '与当前年龄范围冲突。'),
        population_determined_after_time_zero: tr('This condition is determined after time zero.', '该条件在研究起点之后才能确定。'),
        population_preparation_kind_unsupported: tr('Extraction settings cannot express this complete condition.', '当前提取设置不能完整表达该条件。'),
        population_inclusion_not_applied: tr('This population criterion remains unapplied.', '这条人群条件仍无法施加。'),
        population_kind_not_typed: tr('No supported condition kind expresses it.', '目前没有可准确表达该条件的类型。'),
        population_concept_unavailable: tr('The required concept is unavailable.', '所需临床概念不可用。'),
        population_identifier_column: tr('An identifier cannot define this clinical condition.', '标识列不能判定该临床条件。'),
        population_column_unresolved: tr('The required data column cannot be resolved.', '无法定位所需数据列。'),
        population_condition_column_not_status: tr('This column does not establish whether the condition occurred.', '该列不能判定条件是否发生。'),
        population_unit_mismatch: tr('The units do not match.', '单位不匹配。'),
        population_unit_unrecorded: tr('The required unit is not recorded.', '所需单位没有记录。'),
        population_window_unreadable: tr('The required time window is unavailable.', '无法读取所需时间窗。'),
        population_whole_stay_in_finite_window: tr('A limited window cannot establish whole-stay status.', '有限时间窗不能判定整次住院的状态。'),
        population_event_time_not_hours: tr('The event time cannot be read on the required hour scale.', '事件时间不能按所需小时尺度读取。'),
        population_threshold_outside_domain: tr('The threshold cannot distinguish values in this column.', '该阈值无法区分该列的实际取值。'),
        population_concept_without_values: tr('The concept has no measured values here.', '该概念在此处没有实际测量值。'),
      };
      const rows = (preparation.criteria || []).map(row => {
        let source = row.proposed_by_system ? tr('System proposal', '系统建议') : tr('Stated by the study', '研究原文');
        if (row.proposed_by_system && row.quote_found_in_question) {
          source += ` · ${tr('Words from the question, system interpretation', '题目用语，系统解读')}`;
        }
        if (row.attribution === 'typed_study_setting' && row.typed_study_setting) {
          const fields = Object.entries(row.typed_study_setting).map(([field, value]) => `${field} = ${value}`).join(', ');
          source += ` · ${tr('Study setting', '研究设置')}: ${fields}`;
        }
        const state = reasons[row.reason] || (row.disposition === 'requires_extraction'
          ? tr('Needs extraction with these settings.', '需按这些设置提取。')
          : row.disposition === 'not_applied' ? tr('Not applied.', '未施加。')
            : tr('Already applied by the source or plan.', '已由数据来源或计划施加。'));
        return `<li>${esc(row.quote)} <small>${esc(source)} · ${esc(state)}${row.accepted ? ` · ${esc(tr('Omission accepted', '已接受此条局限'))}` : ''}</small></li>`;
      }).join('');
      const labels = {
        age_min: tr('Minimum age', '最低年龄'), age_max: tr('Maximum age', '最高年龄'),
        min_icu_los_hours: tr('Minimum ICU stay (hours)', '最短 ICU 时长（小时）'),
        exclude_readmissions: tr('Keep each patient’s first ICU stay only', '仅保留每位患者的首次 ICU 入住'),
        first_icu_stay_per_hospitalization: tr('Keep the first ICU stay within each hospitalization', '仅保留每次住院的首次 ICU 入住'),
      };
      const value = item => typeof item === 'boolean' ? (item ? tr('Yes', '是') : tr('No', '否')) : String(item);
      const changes = (preparation.changes || []).map(row =>
        `<li>${esc(labels[row.field] || row.field)}：${esc(value(row.before))} → ${esc(value(row.after))}</li>`).join('');
      // The source cannot order one patient's stays across hospitalizations
      // (workflow.patient_first_stay_substitutes): say why the scope is the
      // hospitalization, not the patient.
      const substitutes = (host.workflow() || {}).patient_first_stay_substitutes;
      const perHospitalization = (preparation.changes || []).some(row => row.field === 'first_icu_stay_per_hospitalization' && row.after === true);
      const scopeNote = perHospitalization && Array.isArray(substitutes) && substitutes.length ? `<p>${esc(tr(
        'This database cannot tell which ICU stay was a patient’s first across separate hospitalizations, so the first ICU stay within each hospitalization is used; one patient can contribute several hospitalizations.',
        '这个数据库无法确定同一患者跨不同住院的首次 ICU 入住，因此按每次住院的首次 ICU 入住；同一患者可能贡献多次住院。',
      ))}</p>` : '';
      return `<div class="gpi-population-preparation"><ul>${rows}</ul>${changes ? `<strong>${esc(tr('Settings to apply', '将应用的设置'))}</strong><ul>${changes}</ul>${scopeNote}` : ''}</div>`;
    }

    function populationAcceptanceHtml(proposal) {
      if (!proposal) return '';
      const applied = (proposal.applied_criteria || []).map(quote => `<li>${esc(quote)}</li>`).join('');
      const pending = (proposal.criteria || []).find(row => !row.accepted);
      return `<div class="gpi-population-acceptance"><strong>${esc(tr('Population selected by the plan', '计划实际筛选的人群'))}</strong>
        ${applied ? `<ul>${applied}</ul>` : `<p>${esc(tr('All input records in the bound data package; none of the omitted conditions below filter those records.', '绑定数据包的全部输入记录；下列未施加条件均不参与筛选。'))}</p>`}
        <small>${esc(tr('Step-specific time-zero, outcome and missing-data rules still follow the complete plan.', '各步骤的研究起点、结局与缺失处理仍以完整计划为准。'))}</small>
        ${populationPreparationHtml(proposal)}
        ${pending ? `<p><strong>${esc(tr('Current decision', '当前待确认'))}：</strong>${esc(pending.quote)}</p>` : `<p>${esc(tr('All listed omissions are recorded. Review the complete plan before approving execution.', '所列局限已逐条记录，请继续审阅完整计划后再决定是否批准执行。'))}</p>`}</div>`;
    }

    return Object.freeze({
      preparedDataStatus, validDeferredInputScope, deferredInputScopeHtml,
      reviewedPopulationSpecHtml, populationPreparationHtml, populationAcceptanceHtml,
    });
  }
  window.EasyICU.guidedPi.declare('confirmationData', { create });
})();
