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
      if (!value || value.stated || !options.length || host.busy() || host.sessionIsStale()) return '';
      return `<section class="gpi-cohort-eligibility" aria-label="${esc(tr('Primary cohort confirmation', '确认主队列'))}">
        <div class="gpi-cohort-eligibility-head">
          <strong>${esc(tr('Confirm the exact primary-cohort contract', '确认精确的主队列合同'))}</strong>
          <small>${esc(tr('Only a button below can create eligibility authority. Conversation text may propose changes but cannot approve them.', '只有下方按钮能生成资格权限；对话文字可以提出修改，但不能代替批准。'))}</small>
        </div>
        <ul class="gpi-cohort-contract">${contractSummary(value.primary_cohort_contract)}</ul>
        <div class="gpi-cohort-options">${options.map(option => {
          const label = option.label || {};
          const detail = option.detail || {};
          const localizedLabel = window.EU_LANG === 'zh' ? label.zh : label.en;
          const localizedDetail = window.EU_LANG === 'zh' ? detail.zh : detail.en;
          return `<button class="btn sm" type="button"
            data-gpi-cohort-option="${esc(option.id)}"
            data-gpi-cohort-revision="${esc(option.expected_revision)}"
            data-gpi-cohort-scope="${esc(option.primary_cohort_contract_sha256)}"
            data-gpi-cohort-event="${esc(option.selection_event_id)}">
            <strong>${esc(localizedLabel || option.id)}</strong>
            <small>${esc(localizedDetail || '')}</small>
            <ul class="gpi-cohort-contract gpi-cohort-option-contract">
              ${contractSummary(option.primary_cohort_contract)}
            </ul>
          </button>`;
        }).join('')}</div>
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

    return { render, actionFromEvent };
  }

  window.EU_GUIDED_PI_COHORT_ELIGIBILITY = { create };
})();
