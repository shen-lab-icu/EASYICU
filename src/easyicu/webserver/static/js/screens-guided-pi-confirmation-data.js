/* Read-only prepared-data and deferred-input scope rendering. */
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

    return Object.freeze({ preparedDataStatus, validDeferredInputScope, deferredInputScopeHtml });
  }
  window.EasyICU.guidedPi.declare('confirmationData', { create });
})();
