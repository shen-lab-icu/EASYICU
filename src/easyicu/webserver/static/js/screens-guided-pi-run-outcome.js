/* Guided Copilot durable completed-run card owner.
   A validated analysis remains visible after refresh even when publication
   review stays closed. It renders only host-projected artifact references. */
(function () {
  'use strict';

  function create(deps) {
    const {
      tr, esc, iconHtml, resourceButton, api, projectId, canPreview,
      preview, workflowContext, errorText, recordHostAction, onError,
    } = deps;
    const labels = {
      'result_tables.json': ['View result tables', '查看结果表'],
      'figure_gallery.json': ['View analysis figures', '查看分析图表'],
      'manuscript_provenance.json': ['Preview evidence-bound article', '预览证据绑定文章'],
      'scientific_readiness.json': ['View scientific review', '查看科学审阅'],
    };

    function render(latestRun, workflow) {
      if (!latestRun || latestRun.present !== true || latestRun.analysis_results_available !== true) return '';
      const stages = Array.isArray(workflow && workflow.stages) ? workflow.stages : [];
      const analysis = stages.find(stage => stage && stage.id === 'analysis');
      if (!analysis || !['complete', 'review_required'].includes(String(analysis.status || ''))) return '';
      const validated = latestRun.analysis_validated === true;
      const numericVerified = latestRun.numeric_verified === true;
      const manuscriptReady = latestRun.manuscript_ready === true;
      const resources = Array.isArray(latestRun.artifact_refs) ? latestRun.artifact_refs : [];
      const ledger = resources.find(row => row && row.artifact === 'evidence_ledger.json');
      const detailActions = latestRun.run_id && ledger ? [
        resourceButton({
          kind: 'research_report', run_id: latestRun.run_id, artifact: 'full_analysis_report.json',
          label: tr('View complete analysis report', '查看完整分析报告'), media_type: 'application/json', sha256: ledger.sha256,
        }, tr('View complete analysis report', '查看完整分析报告')),
        manuscriptReady ? resourceButton({
          kind: 'research_report', run_id: latestRun.run_id, artifact: 'article_report.json',
          label: tr('View article report with figures', '查看含图文章报告'), media_type: 'application/json', sha256: ledger.sha256,
        }, tr('View article report with figures', '查看含图文章报告')) : '',
        resourceButton({
          kind: 'research_report', run_id: latestRun.run_id, artifact: 'technical_report.json',
          label: tr('View technical analysis report', '查看技术分析报告'), media_type: 'application/json', sha256: ledger.sha256,
        }, tr('View technical analysis report', '查看技术分析报告')),
      ] : [];
      if (manuscriptReady) {
        const manuscriptPdf = resources.find(row => row && row.artifact === 'manuscript_scaffold.pdf');
        if (manuscriptPdf) {
          detailActions.push(resourceButton(
            { ...manuscriptPdf, label: tr('View LaTeX manuscript PDF', '查看 LaTeX 论文') },
            tr('View LaTeX manuscript PDF', '查看 LaTeX 论文'),
          ));
        }
      }
      if (ledger) {
        detailActions.push(resourceButton(
          { ...ledger, label: tr('View evidence ledger', '查看证据台账') },
          tr('View evidence ledger', '查看证据台账'),
        ));
      }
      const primaryActions = [
        `<button class="btn sm primary" type="button" data-gpi-run-outcome-data>${iconHtml('chart', 13)} ${esc(tr('Open data visualization', '打开数据可视化'))}</button>`,
        ...Object.keys(labels).map(name => {
          if (!manuscriptReady && name === 'manuscript_provenance.json') return '';
          const resource = resources.find(row => row && row.artifact === name);
          const label = tr(labels[name][0], labels[name][1]);
          return resource ? resourceButton({ ...resource, label }, label) : '';
        }).filter(Boolean),
      ];
      const retryAvailable = Boolean(
        workflow && workflow.analysis_validation_retry_available === true
      );
      if (retryAvailable && (!validated || !manuscriptReady)) {
        const retryLabel = validated
          ? tr('Restore manuscript and evidence checks', '恢复稿件与证据校验')
          : tr('Repair and revalidate', '修复并重新校验');
        primaryActions.unshift(`<button class="btn sm primary" type="button" data-gpi-run-outcome-retry>${iconHtml('refresh', 13)} ${esc(retryLabel)}</button>`);
      }
      if (!primaryActions.length && !detailActions.length) return '';
      return `<section class="gpi-run-outcome" role="status" aria-label="${esc(tr('Completed analysis results', '已完成的分析结果'))}">
        <div class="gpi-run-outcome-icon" aria-hidden="true">${iconHtml(validated ? 'check' : 'shield', 17)}</div>
        <div class="gpi-run-outcome-copy">
          <strong>${esc(validated
            ? tr('Analysis complete — results are ready for review', '分析已完成，可以审阅结果')
            : tr('Results generated — one validation item needs review', '结果已经生成，仍有一项校验需要处理'))}</strong>
          <p>${esc(tr(
            validated
              ? numericVerified
                ? 'The approved analysis and numeric checks completed. Publication-level review is still open, but it does not hide these analysis-only results.'
                : 'The approved analysis and automated validation completed. Manuscript-level numeric provenance and publication review remain open, so these results are for analysis review only.'
              : 'Execution, evidence binding, and numeric checks completed. The results remain reviewable while the outstanding validation item is repaired.',
            validated
              ? numericVerified
                ? '已批准的分析与数值核验均已完成。投稿级审阅尚未闭合，但不会再隐藏这批分析结果。'
                : '已批准的分析与自动校验已经完成。稿件级数字溯源和投稿审阅尚未闭合，因此当前结果仅供分析审阅。'
              : '执行、证据绑定和数值核验已经完成。剩余校验项修复期间，结果仍可查看。',
          ))}</p>
          <div class="gpi-run-outcome-actions" aria-label="${esc(tr('Recommended review order', '建议审阅顺序'))}">${primaryActions.join('')}</div>
          ${detailActions.length ? `<details class="gpi-run-outcome-more"><summary>${esc(tr(`Complete artifacts and audit (${detailActions.length})`, `完整产物与审计（${detailActions.length}）`))}</summary><div class="gpi-run-outcome-actions">${detailActions.join('')}</div></details>` : ''}
          <small>${esc(tr('Analysis-only: suitable for review, not yet authorized for publication claims.', '当前为分析级结果：可以审阅，但尚未获准作为投稿结论。'))}</small>
        </div>
      </section>`;
    }

    async function openData(button) {
      if (!canPreview()) return;
      const client = api();
      if (!client.preparePiCopilotDataWorkbenchSnapshot || !preview()) {
        onError(tr('The source-data snapshot is temporarily unavailable. Refresh this project and try again.', '源数据快照暂时不可用，请刷新当前项目后重试。'));
        return;
      }
      const original = button ? button.innerHTML : '';
      if (button) {
        button.disabled = true;
        button.textContent = tr('Preparing source snapshot…', '正在准备源数据快照…');
      }
      try {
        const payload = await client.preparePiCopilotDataWorkbenchSnapshot(projectId());
        const resource = payload && payload.resource;
        if (!resource) throw new Error(tr('EasyICU did not return a Data Workbench snapshot.', 'EasyICU 未返回数据工作台快照。'));
        resource.label = tr('EasyICU data visualization', 'EasyICU 数据可视化');
        preview().open(resource, projectId(), workflowContext());
        const context = workflowContext();
        await recordHostAction(
          'review_prepared_data',
          `${String((context && context.currentRunId) || projectId())}:${String(resource.snapshot_sha256 || '')}`,
        );
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

  window.EasyICU.guidedPi.declare('runOutcome', { create });
})();
