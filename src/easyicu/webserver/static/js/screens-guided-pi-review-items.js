/* Owner: the itemised scientific review of one finished run.
   The host's scientific review projection (webserver/scientific_review.py)
   owns every item: the reviewed approach, the data and materials, the number
   check and the limitations, each with the record it came from. This file
   names each item's code in plain words, shows its recorded value compactly
   and offers the existing preview of its source. It computes nothing. The
   run outcome card (screens-guided-pi-run-outcome.js) places the panels. */
(function () {
  'use strict';

  const LABELS = {
    SR_APPROACH_POPULATION: ['Population', '研究人群'],
    SR_APPROACH_OUTCOME: ['Outcome', '结局'],
    SR_APPROACH_TIME_ZERO: ['Time zero', '时间零点'],
    SR_APPROACH_ESTIMAND: ['Estimand', '估计目标'],
    SR_APPROACH_OBSERVATION_WINDOW: ['Observation window', '观察窗口'],
    SR_APPROACH_EXPOSURE: ['Exposure', '暴露'],
    SR_APPROACH_GROUPING: ['Grouping', '分组'],
    SR_APPROACH_MODEL: ['Model', '模型'],
    SR_APPROACH_DIAGNOSTICS: ['Diagnostics', '诊断检查'],
    SR_APPROACH_STEP: ['Analysis step', '分析步骤'],
    SR_DATA_DATABASE: ['Database', '数据库'],
    SR_DATA_VERSION: ['Database version', '数据库版本'],
    SR_DATA_EXPORT_ID: ['Export', '导出编号'],
    SR_COHORT: ['Cohort', '队列'],
    SR_COHORT_FLOW: ['Inclusion and exclusion', '纳入与排除'],
    SR_CONCEPT_DEFINITION: ['Variable definition', '变量定义'],
    SR_CONCEPT_USED: ['Clinical concept used', '使用的临床概念'],
    SR_SOURCE_SUBSTITUTION: ['Substituted definition', '替代定义'],
    SR_METHOD: ['Method', '方法'],
    SR_RECORDED_PACKAGE_VERSIONS: ['Package versions', '软件包版本'],
    SR_RECORDED_PYTHON_VERSION: ['Python version', 'Python 版本'],
    SR_PACKAGE_VERSIONS_NOT_RECORDED: ['Package versions were not recorded', '未记录软件包版本'],
    SR_RECORDED_DATA_PROPERTY_NOTES: ['Data properties', '数据特性'],
    SR_RECORDED_LIMITATIONS: ['Recorded limitation', '已记录的局限'],
    SR_RECORDED_DIAGNOSTICS: ['Diagnostic result', '诊断结果'],
    SR_RECORDED_CLAIM_CEILING: ['How far the result can be used', '结论可用范围'],
    SR_CLAIM_CEILING: ['How far the result can be used', '结论可用范围'],
    SR_METADATA_INCOMPLETE: ['The run record is incomplete', '运行记录不完整'],
    SR_GATE: ['A check did not pass', '有检查未通过'],
    SR_FINDING: ['Review finding', '审阅意见'],
    SR_NUMBER_BOUND: ['Bound to evidence', '有证据'],
    SR_NUMBER_UNBOUND: ['No evidence record', '没有证据记录'],
    SR_NUMBER_SUPPRESSED: ['Hidden small count', '已隐藏的小样本数'],
  };
  const SECTIONS = { approach: 'approach', materials: 'materials', limitations: 'limitations', numbers: 'number_check' };

  function create(deps) {
    const { tr, esc, api, projectId, resourceButton } = deps;
    let current = { key: '', review: null, loading: false, error: '' };

    function label(item) {
      const pair = LABELS[item && item.code];
      return pair ? tr(...pair) : String((item && item.text) || '').slice(0, 120);
    }
    // Recorded values are bounded JSON from the host; show scalars as text and
    // structures as short key: value pairs, never as markup.
    function valueText(value, depth = 0) {
      if (value === null || value === undefined || value === '') return '';
      if (typeof value === 'boolean') return value ? tr('yes', '是') : tr('no', '否');
      if (typeof value !== 'object') return String(value).slice(0, 300);
      if (depth > 1) return JSON.stringify(value).slice(0, 160);
      if (Array.isArray(value)) return value.slice(0, 8).map(row => valueText(row, depth + 1)).filter(Boolean).join(tr('; ', '；'));
      return Object.entries(value).slice(0, 8).map(([key, row]) => {
        const text = valueText(row, depth + 1);
        return text ? `${key}: ${text}` : '';
      }).filter(Boolean).join(tr('; ', '；'));
    }
    function sourceButton(latestRun, item) {
      const source = item && item.source;
      if (!source || !resourceButton || !/^[a-f0-9]{64}$/.test(String(source.artifact_sha256 || ''))) return '';
      const ref = { kind: 'research_artifact', run_id: latestRun.run_id, artifact: source.artifact,
        sha256: source.artifact_sha256, label: source.artifact };
      if (source.evidence_id && /^[a-f0-9]{64}$/.test(String(source.sha256 || ''))) {
        Object.assign(ref, { evidence_id: source.evidence_id, evidence_sha256: source.sha256, evidence_kind: 'artifact',
          evidence_label: tr('Source', '来源'), evidence_pointer: source.evidence_json_pointer || source.json_pointer || '' });
      }
      return resourceButton(ref, tr('View source', '查看来源'));
    }
    function itemHtml(latestRun, item) {
      const missing = item.status === 'not_recorded';
      const text = missing ? tr('Not recorded', '未记录') : valueText(item.value);
      const defect = item.code === 'SR_NUMBER_UNBOUND' || item.code === 'SR_PACKAGE_VERSIONS_NOT_RECORDED'
        || item.code === 'SR_METADATA_INCOMPLETE' || missing;
      return `<li class="gpi-review-item${defect ? ' is-defect' : ''}" data-review-item="${esc(String(item.code || ''))}"><div><strong>${esc(label(item))}</strong>${text ? `<p>${esc(text)}${item.value_truncated ? '…' : ''}</p>` : ''}</div>${sourceButton(latestRun, item)}</li>`;
    }
    function more(section) {
      return section && section.next_offset !== null && section.next_offset !== undefined
        ? `<li class="gpi-review-more"><p>${esc(tr(`Showing ${section.items.length} of ${section.total}; the rest are in the run’s records.`, `显示 ${section.items.length} / ${section.total} 条，其余见运行记录。`))}</p></li>` : '';
    }

    function numbersHtml(latestRun, section) {
      const counts = section.counts || {};
      const surface = { answer: tr('answer', '回答'), manuscript: tr('draft', '稿件') };
      const summary = tr(
        `${counts.bound || 0} numbers have an evidence record, ${counts.unbound || 0} do not, ${counts.suppressed || 0} are hidden small counts.`,
        `${counts.bound || 0} 个数字有证据记录，${counts.unbound || 0} 个没有，${counts.suppressed || 0} 个是按小样本规则隐藏的数。`,
      );
      const coverage = section.coverage && section.coverage.manuscript === 'unavailable'
        ? tr(' No draft yet, so only the answer was checked.', '尚无稿件，只核对了回答中的数字。') : '';
      // Numbers without evidence first: each is a defect to resolve, never hidden.
      const rows = section.items.slice().sort((a, b) => (a.code === 'SR_NUMBER_UNBOUND' ? 0 : 1) - (b.code === 'SR_NUMBER_UNBOUND' ? 0 : 1));
      return `<li class="gpi-review-item${counts.unbound ? ' is-defect' : ''}"><div><strong>${esc(tr('Number check', '数字核对'))}</strong><p>${esc(summary + coverage)}</p></div></li>`
        + rows.map(item => {
          const where = surface[item.surface] ? ` · ${surface[item.surface]}` : '';
          return `<li class="gpi-review-item${item.code === 'SR_NUMBER_UNBOUND' ? ' is-defect' : ''}" data-review-item="${esc(String(item.code || ''))}"><div><strong>${esc(valueText(item.value) || '—')}</strong><p>${esc(label(item) + where)}</p></div>${sourceButton(latestRun, item)}</li>`;
        }).join('') + more(section);
    }

    function keyFor(latestRun) {
      return latestRun && latestRun.run_id && typeof projectId === 'function' ? `${projectId()}:${latestRun.run_id}` : '';
    }
    async function load(latestRun) {
      const key = keyFor(latestRun);
      const client = typeof api === 'function' ? api() : null;
      if (!key || current.key === key || !client || typeof client.loadPiRunScientificReview !== 'function') return false;
      current = { key, review: null, loading: true, error: '' };
      try {
        const payload = await client.loadPiRunScientificReview(projectId(), latestRun.run_id);
        if (current.key !== key) return false;
        const review = payload && payload.review;
        if (!review || review.run_id !== latestRun.run_id) throw new Error('scientific_review_payload_invalid');
        current = { key, review, loading: false, error: '' };
      } catch (failure) {
        if (current.key !== key) return false;
        current = { key, review: null, loading: false, error: String((failure && failure.code) || 'unavailable') };
      }
      return true;
    }

    // name: approach | materials | numbers | limitations. Empty until loaded,
    // so the card keeps its own summary rows meanwhile.
    // exclude(item) drops items the card already shows from their own record.
    function panelHtml(latestRun, name, exclude) {
      const review = current.key === keyFor(latestRun) ? current.review : null;
      const section = review && review[SECTIONS[name]];
      if (!section || !Array.isArray(section.items)) return '';
      if (name === 'numbers') return numbersHtml(latestRun, section);
      const kept = typeof exclude === 'function' ? section.items.filter(item => !exclude(item)) : section.items;
      if (!kept.length) return '';
      const rows = name === 'approach' ? kept.slice().sort((a, b) => (a.order || 0) - (b.order || 0)) : kept;
      return rows.map(item => itemHtml(latestRun, item)).join('') + more(section);
    }
    function analysisOnly(latestRun) {
      const review = current.key === keyFor(latestRun) ? current.review : null;
      return Boolean(review && review.analysis_only === true && review.publication_authorized !== true);
    }

    return { load, panelHtml, analysisOnly, valueText };
  }

  window.EasyICU.guidedPi.declare('reviewItems', { create, LABELS });
})();
