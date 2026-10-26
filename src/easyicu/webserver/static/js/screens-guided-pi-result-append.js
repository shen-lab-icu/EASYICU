/* Owner: result additions to one study; the server owns executable rosters. */
(function () {
  'use strict';
  // Display copy for the host's result follow-ups (research_followups.py).
  // Only additions to this study are offered here: another database waits for
  // the multi-database owner and a new study has no launch route yet.
  const FOLLOWUP_KINDS = {
    prespecified_sensitivity: ['Run the pre-specified sensitivity analysis', '补做方案预设的敏感性分析'],
    functional_form_check: ['Check how continuous covariates enter the model', '检查连续协变量的函数形式'],
    named_subgroup: ['Analyse the subgroup the design names', '分析设计中写明的亚组'],
    dose_response: ['Check dose–response for the exposure', '看暴露的剂量反应关系'],
    complementary_survival: ['Look at time to the same outcome', '用生存分析看同一结局的发生时间'],
    descriptive_context: ['Describe the population behind the result', '描述主要结果背后的人群'],
    complementary_association: ['Add the association analysis', '补充关联分析'],
    article_extension: ['Extend to a full article', '扩展成完整论文'],
  };
  const FOLLOWUP_REASONS = {
    prespecified_not_run: ['Pre-specified in the plan but not yet run.', '方案里预设了，但还没做。'],
    sealed_linear_covariate: ['The plan entered a continuous covariate as a straight line.', '方案把连续协变量按线性纳入。'],
    design_named_subgroup_not_run: ['The design names this subgroup; it has not been analysed.', '设计写明了这个亚组，但还没分析。'],
    sealed_continuous_exposure_source: ['The exposure is continuous or ordered.', '暴露是连续或有序变量。'],
    paired_event_time_available: ['The outcome has a recorded event time.', '结局有记录的发生时间。'],
    primary_result_denominator_context: ['It helps read the main result’s denominator.', '有助于解读主要结果的分母。'],
    first_result_completed: ['The first result is complete.', '首轮结果已完成。'],
  };
  function create({ tr, esc, api, projectId, host, session, canPreview, onError, errorText }) {
    let current = null;
    let busy = false;
    // Host suggestions that this study's offered additions can run, one per kind.
    function suggested(rows) {
      const suggestions = current && current.followups && Array.isArray(current.followups.suggestions) ? current.followups.suggestions : [];
      const seen = new Set();
      return suggestions.filter(row => {
        const kind = row && row.target_action === 'append' && row.delta ? String(row.delta.kind || '') : '';
        if (!kind || seen.has(kind) || !FOLLOWUP_KINDS[row.kind] || !rows.some(option => option.kind === kind && option.available)) return false;
        seen.add(kind);
        return true;
      }).slice(0, 4);
    }
    function suggestionHtml(row) {
      const reason = (Array.isArray(row.reason_codes) ? row.reason_codes : []).map(code => FOLLOWUP_REASONS[code]).find(Boolean);
      return `<li><button class="btn sm" type="button" data-gpi-result-append="${esc(row.delta.kind)}"${busy || !canPreview() ? ' disabled' : ''}>${esc(tr(...FOLLOWUP_KINDS[row.kind]))}</button>${reason ? `<span>${esc(tr(...reason))}</span>` : ''}</li>`;
    }
    function render(run) {
      const valid = current && current.project === projectId() && current.run_id === run.run_id;
      const offered = valid ? current.options : [];
      const picks = valid ? suggested(offered) : [];
      const rows = offered.filter(row => !picks.some(pick => pick.delta.kind === row.kind));
      const notice = valid && current.source_run_sha256
        ? tr('Additions use the same data version as this result.', '追加分析沿用这次结果的数据版本。') : '';
      return `<div class="gpi-result-append" data-append-run="${esc(run.run_id)}">${offered.length ? `<strong>${esc(tr('Continue this study', '继续这项研究'))}</strong><p>${esc(notice)}</p>` : ''}${picks.length ? `<ul class="gpi-result-followups">${picks.map(suggestionHtml).join('')}</ul>` : ''}${rows.map(row => {
        const label = esc(tr(row.label_en, row.label_zh));
        return row.available
          ? `<button class="btn sm" type="button" data-gpi-result-append="${esc(row.kind)}"${busy || !canPreview() ? ' disabled' : ''}>${label}</button>`
          : `<p>${label}：${esc(tr(row.status_en, row.status_zh))} <button class="btn sm" type="button" data-gpi-append-owner="${esc(row.owner)}">${esc(tr('Open analysis settings', '进入分析设置'))}</button></p>`;
      }).join('')}</div>`;
    }
    async function load(run) {
      const expectedProject = projectId();
      if (!expectedProject || !run.run_id || !api().loadPiCopilotAppendOptions) return;
      if (current && current.project === expectedProject && current.run_id === run.run_id) return;
      const requested = { project: expectedProject, run_id: run.run_id, options: [] };
      current = requested;
      try {
        const result = await api().loadPiCopilotAppendOptions(expectedProject, run.run_id);
        if (current !== requested || projectId() !== expectedProject || result.run_id !== run.run_id) return;
        current = { ...requested, ...result, options: Array.isArray(result.options) ? result.options : [] };
      } catch (_error) {
        if (current !== requested) return;
        current.options = [];
      }
      const node = host && host()?.querySelector('.gpi-result-append');
      if (node && node.dataset.appendRun === run.run_id) node.outerHTML = render(run);
    }
    async function propose(kind) {
      if (busy || !canPreview() || !session() || !current || current.project !== projectId()
        || !current.options.some(row => row.kind === kind && row.available) || !current.source_run_sha256) return false;
      const expectedProject = projectId();
      busy = true;
      try {
        await api().proposePiCopilotAppend(session().session_id, {
          project_id: expectedProject, run_id: current.run_id, kind, expected_source_sha256: current.source_run_sha256,
        });
        return expectedProject === projectId();
      } catch (error) {
        if (expectedProject === projectId()) onError(`${tr('Could not prepare the addition', '无法准备追加分析')}: ${errorText(error)}`);
        return false;
      } finally { busy = false; }
    }
    return Object.freeze({ load, render, propose });
  }
  window.EasyICU.guidedPi.declare('resultAppend', { create });
})();
