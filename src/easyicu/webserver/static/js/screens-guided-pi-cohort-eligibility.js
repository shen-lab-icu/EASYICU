/* Guided Copilot primary-cohort confirmation owner.
   It renders only server-issued option coordinates and returns the selected
   coordinate to the shell. It never derives consent from conversation text. */
(function () {
  'use strict';

  function create(host) {
    const tr = host.tr;
    const esc = host.esc;

    function selection() {
      const session = host.session() || {};
      const value = session.cohort_eligibility_selection;
      return value && value.present ? value : null;
    }

    function workflow() {
      return typeof host.workflow === 'function' ? (host.workflow() || {}) : {};
    }

    function analysisDesign() {
      const receipt = workflow().study_setup_receipt || {};
      const configuration = receipt.configuration || {};
      return configuration.analysis_design || {};
    }

    function contractSummary(contract) {
      const scope = contract || {};
      const population = scope.population || {};
      const admission = scope.admission_eligibility || {};
      const diagnosis = scope.diagnosis_eligibility || {};
      const phenotype = scope.phenotype_window || {};
      const sampling = scope.sampling || {};
      const repeated = admission.repeated_admission_policy === 'first_icu_admission_only'
        ? tr('first ICU admission only', '仅首次 ICU 入住')
        : tr('all ICU admissions', '全部 ICU 入住');
      const age = `${Number(admission.minimum_age_years || 0)}–${Number(admission.maximum_age_years || 100)}`;
      const details = [
        `${tr('Population', '人群')}: ${String(population.definition || 'all_bound_icu_stays')}`,
        `${tr('Age', '年龄')}: ${age}`,
        `${tr('Admissions', '入住')}: ${repeated}`,
        `${tr('Minimum ICU stay', '最短 ICU 停留')}: ${Number(admission.minimum_icu_duration_hours || 0)} h`,
      ];
      if (diagnosis.enabled) {
        details.push(`${tr('Diagnosis include/exclude', '诊断纳入/排除')}: ${(diagnosis.include || []).length}/${(diagnosis.exclude || []).length}`);
      }
      if (phenotype.observation_window_hours) {
        details.push(`${tr('Phenotype window', '表型时间窗')}: ${Number(phenotype.observation_window_hours)} h`);
      }
      if (sampling.status === 'capped') {
        details.push(`${tr('Sampling cap', '抽样上限')}: ${Number(sampling.max_patients)}`);
      }
      return details.map(value => `<li>${esc(value)}</li>`).join('');
    }

    function render() {
      const value = selection();
      const options = value && Array.isArray(value.options) ? value.options : [];
      const actionCode = String(workflow().next_action_code || '');
      const planNeedsThisDecision = [
        'cohort_eligibility_confirmation_required',
        'plan_scientific_changes_required',
        'failed_pipeline_requires_fresh_plan',
        'planner_checkpoint_resume_available',
      ].includes(actionCode);
      if (!value || value.stated || value.blocker_code !== 'cohort_eligibility_confirmation_required'
        || !options.length || !planNeedsThisDecision || host.busy() || host.sessionIsStale()) return '';

      const admission = value.primary_cohort_contract
        && value.primary_cohort_contract.admission_eligibility || {};
      const minimumAge = Number(admission.minimum_age_years || 0);
      const minimumDuration = Number(admission.minimum_icu_duration_hours || 0);
      const preferredIds = minimumAge === 18 && minimumDuration === 0
        ? ['adults_all_admissions', 'adults_first_admission']
        : minimumAge === 0 && minimumDuration === 0
          ? ['no_eligibility_filter', 'first_admission_only']
          : [];
      const byId = new Map(options.map(option => [String(option.id || ''), option]));
      const focused = preferredIds.map(id => byId.get(id)).filter(Boolean);
      const visibleOptions = focused.length === 2 ? focused : options.slice(0, 3);
      const currentDigest = String(value.primary_cohort_contract_sha256 || '');
      const design = analysisDesign();
      const repeatedAdmissionsRecommended = String(admission.repeated_admission_policy || '') === 'all_icu_admissions'
        && String(design.analysis_unit || '') === 'icu_stay'
        && String(design.variance_estimator || '') === 'cluster_robust'
        && String(design.cluster_unit || '') === 'patient';
      const rationale = repeatedAdmissionsRecommended
        ? tr(
          'The analysis unit is an ICU stay and repeated stays are already handled with patient-clustered robust variance, so keeping every eligible stay is recommended.',
          '当前以 ICU stay 为分析单位，并已设置患者层聚类稳健方差，因此推荐保留全部符合条件的 ICU 入住。',
        )
        : tr(
          'The recommended option preserves the cohort definition already configured for this study.',
          '推荐项会保留当前研究已经配置好的队列定义。',
        );
      return `<section class="gpi-cohort-eligibility" aria-label="${esc(tr('Repeated ICU stay choice', '重复 ICU 入住选择'))}">
        <div class="gpi-cohort-eligibility-head">
          <strong>${esc(tr('How should repeat ICU stays be handled?', '同一患者多次住 ICU，怎么处理？'))}</strong>
          <small>${esc(tr('Choose once, then EasyICU will generate the plan.', '选择一次，EasyICU 随后会继续生成计划。'))}</small>
        </div>
        <p class="gpi-cohort-rationale"><strong>${esc(tr('Recommendation', '推荐'))}</strong><span>${esc(rationale)}</span></p>
        <div class="gpi-cohort-options">${visibleOptions.map(option => {
          const label = option.label || {};
          const detail = option.detail || {};
          const localizedLabel = window.EU_LANG === 'zh' ? label.zh : label.en;
          const localizedDetail = window.EU_LANG === 'zh' ? detail.zh : detail.en;
          const recommended = String(option.primary_cohort_contract_sha256 || '') === currentDigest;
          return `<button class="btn sm${recommended ? ' primary is-recommended' : ''}" type="button"
            data-gpi-cohort-option="${esc(option.id)}"
            data-gpi-cohort-revision="${esc(option.expected_revision)}"
            data-gpi-cohort-scope="${esc(option.primary_cohort_contract_sha256)}"
            data-gpi-cohort-event="${esc(option.selection_event_id)}">
            <strong>${recommended ? `${esc(tr('Recommended', '推荐'))} · ` : ''}${esc(localizedLabel || option.id)}</strong>
            <small>${esc(localizedDetail || '')}</small>
          </button>`;
        }).join('')}</div>
        <details class="gpi-cohort-evidence"><summary>${esc(tr('Why this recommendation?', '查看推荐依据'))}</summary><ul class="gpi-cohort-contract">${contractSummary(value.primary_cohort_contract)}</ul></details>
      </section>`;
    }

    function actionFromEvent(event) {
      const button = event.target && event.target.closest
        ? event.target.closest('[data-gpi-cohort-option]') : null;
      if (!button) return null;
      return {
        option_id: String(button.dataset.gpiCohortOption || ''),
        expected_revision: Number(button.dataset.gpiCohortRevision),
        primary_cohort_contract_sha256: String(button.dataset.gpiCohortScope || ''),
        selection_event_id: String(button.dataset.gpiCohortEvent || ''),
      };
    }

    function repeatedStayDecisionHtml(copies) {
      const value = selection();
      const options = value && Array.isArray(value.options) ? value.options : [];
      const admission = value && value.primary_cohort_contract
        && value.primary_cohort_contract.admission_eligibility || {};
      const minimumAge = Number(admission.minimum_age_years || 0);
      const minimumDuration = Number(admission.minimum_icu_duration_hours || 0);
      let optionIds = [];
      if (minimumAge === 0 && minimumDuration === 0) {
        optionIds = ['first_admission_only', 'no_eligibility_filter'];
      } else if (minimumAge === 18 && minimumDuration === 0) {
        optionIds = ['adults_first_admission', 'adults_all_admissions'];
      }
      const byId = new Map(options.map(option => [String(option.id || ''), option]));
      const selected = optionIds.map(id => byId.get(id)).filter(Boolean);
      if (selected.length !== 2) return '';
      const presentation = Array.isArray(copies) ? copies : [];
      return selected.map((option, index) => {
        const copy = presentation[index] || {};
        return `<button class="gpi-decision-option" type="button"
          data-gpi-cohort-option="${esc(option.id)}"
          data-gpi-cohort-revision="${esc(option.expected_revision)}"
          data-gpi-cohort-scope="${esc(option.primary_cohort_contract_sha256)}"
          data-gpi-cohort-event="${esc(option.selection_event_id)}">
          <strong>${esc(copy.label || option.id)}</strong>
          <span>${esc(copy.effect || '')}</span>
          <small>${esc(copy.requirement || '')}</small>
        </button>`;
      }).join('');
    }

    return { render, actionFromEvent, repeatedStayDecisionHtml };
  }

  window.EU_GUIDED_PI_COHORT_ELIGIBILITY = { create };
})();
