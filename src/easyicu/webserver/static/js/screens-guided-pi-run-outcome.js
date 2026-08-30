/* Guided Copilot durable completed-run card owner.
   A validated analysis remains visible after refresh even when publication
   review stays closed. It renders only host-projected artifact references. */
(function () {
  'use strict';

  function create(deps) {
    const {
      tr, esc, iconHtml, resourceButton, api, projectId, canPreview,
      preview, workflowContext, errorText, onError,
    } = deps;
    const labels = {
      'result_tables.json': ['View result tables', '查看结果表'],
      'figure_gallery.json': ['View analysis figures', '查看分析图表'],
      'manuscript_provenance.json': ['Preview evidence-bound article', '预览证据绑定文章'],
      'manuscript_scaffold.pdf': ['View LaTeX manuscript PDF', '查看 LaTeX 论文'],
      'scientific_readiness.json': ['View scientific review', '查看科学审阅'],
    };

    function render(latestRun, workflow) {
      if (!latestRun || latestRun.present !== true || latestRun.analysis_results_available !== true) return '';
      const stages = Array.isArray(workflow && workflow.stages) ? workflow.stages : [];
      const analysis = stages.find(stage => stage && stage.id === 'analysis');
      if (!analysis || analysis.status !== 'complete') return '';
      const resources = Array.isArray(latestRun.artifact_refs) ? latestRun.artifact_refs : [];
      const actions = Object.keys(labels).map(name => {
        const resource = resources.find(row => row && row.artifact === name);
        const label = tr(labels[name][0], labels[name][1]);
        return resource ? resourceButton({ ...resource, label }, label) : '';
      }).filter(Boolean);
      actions.splice(2, 0, `<button class="btn sm" type="button" data-gpi-run-outcome-data>${iconHtml('chart', 13)} ${esc(tr('Preview analysis data', '预览分析数据'))}</button>`);
      if (!actions.length) return '';
      return `<section class="gpi-run-outcome" role="status" aria-label="${esc(tr('Completed analysis results', '已完成的分析结果'))}">
        <div class="gpi-run-outcome-icon" aria-hidden="true">${iconHtml('check', 17)}</div>
        <div class="gpi-run-outcome-copy">
          <strong>${esc(tr('Analysis complete — results are ready', '分析已完成，可以查看结果'))}</strong>
          <p>${esc(tr(
            'The approved analysis and numeric checks completed. Publication-level review is still open, but it does not hide these analysis-only results.',
            '已批准的分析与数值核验均已完成。投稿级审阅尚未闭合，但不会再隐藏这批分析结果。',
          ))}</p>
          <div class="gpi-run-outcome-actions">${actions.join('')}</div>
          <small>${esc(tr('Analysis-only: suitable for review, not yet authorized for publication claims.', '当前为分析级结果：可以审阅，但尚未获准作为投稿结论。'))}</small>
        </div>
      </section>`;
    }

    async function openData(button) {
      if (!canPreview()) return;
      const client = api();
      if (!client.preparePiCopilotDataWorkbenchSnapshot || !preview()) {
        onError(tr('The analysis Data Workbench is temporarily unavailable. Refresh this project and try again.', '分析数据工作台暂时不可用，请刷新当前项目后重试。'));
        return;
      }
      const original = button ? button.innerHTML : '';
      if (button) {
        button.disabled = true;
        button.textContent = tr('Preparing Data Workbench…', '正在准备数据工作台…');
      }
      try {
        const payload = await client.preparePiCopilotDataWorkbenchSnapshot(projectId());
        const resource = payload && payload.resource;
        if (!resource) throw new Error(tr('EasyICU did not return a Data Workbench snapshot.', 'EasyICU 未返回数据工作台快照。'));
        resource.label = tr('Analysis data preview', '分析数据预览');
        preview().open(resource, projectId(), workflowContext());
      } catch (error) {
        onError(errorText(error));
      } finally {
        if (button && button.isConnected) {
          button.disabled = false;
          button.innerHTML = original;
        }
      }
    }

    return Object.freeze({ render, openData });
  }

  window.EU_GUIDED_PI_RUN_OUTCOME = { create };
})();
