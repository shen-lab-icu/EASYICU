/* Guided Pi conversation-resource transport owner.
   Keeps resource identity and DOM projection out of the already-large screen
   shell. It carries coordinates only; governed payloads load in the preview. */
(function () {
  'use strict';

  function create(deps) {
    const { esc } = deps;

    function name(resource) {
      return resource && String(resource.label || resource.artifact || resource.file || '').trim();
    }
    function key(resource) {
      if (!resource) return '';
      if (resource.kind === 'literature_source') return `literature:${resource.pmid || resource.doi || resource.url || ''}`;
      if (resource.kind === 'demo_artifact') return `demo:${resource.artifact || ''}`;
      if (resource.kind === 'demo_document') return `demo-document:${resource.artifact || ''}`;
      if (resource.kind === 'idea_plan') return `idea-plan:${resource.run_id || ''}`;
      if (resource.kind === 'data_package_review') return `data-package:${resource.study_context_id || ''}:${resource.study_revision || 0}:${resource.review_sha256 || ''}`;
      if (resource.kind === 'data_workbench_snapshot') return `data-workbench:${resource.view || ''}:${resource.snapshot_sha256 || ''}`;
      if (resource.kind === 'native_workspace') return `native-workspace:${resource.route || ''}:${resource.study_context_id || ''}:${resource.job_id || resource.source_id || resource.state || ''}:${resource.entry_mode || ''}`;
      return resource.kind === 'research_artifact' || resource.kind === 'research_report' || resource.kind === 'research_document' || resource.kind === 'system_validation_document'
        ? `research:${resource.run_id || ''}:${resource.artifact || ''}`
        : `${resource.kind || 'file'}:${resource.file || ''}`;
    }
    function label(resource) {
      if (!resource) return '';
      if (resource.conversation_label) return String(resource.conversation_label);
      if (resource.kind === 'idea_plan') {
        return window.EU_LANG === 'zh' ? 'Idea Mining 方案预览' : 'Idea Mining plan preview';
      }
      const demo = window.EasyICU.guidedPi.optional('demo');
      if (resource.kind === 'demo_artifact' && demo && typeof demo.artifactLabel === 'function') {
        return demo.artifactLabel(resource.artifact || resource.label || '');
      }
      if (resource.kind === 'research_artifact' && window.AGENT_RENDER && typeof window.AGENT_RENDER.artifactTitle === 'function') {
        return window.AGENT_RENDER.artifactTitle(resource.artifact || resource.label || '');
      }
      return name(resource);
    }
    /* Which artifacts a researcher opens first, and one message's resource
       list ranked by that order. Ranking is resource identity work -- it uses
       the same ``key`` that de-duplicates them -- so it belongs here rather
       than inline in the screen shell. */
    const PREFERRED_ARTIFACTS = [
      'idea_plan.json',
      'full_analysis_report.json', 'article_report.json',
      'technical_report.json',
      'system_validation_report.html', 'system_validation_report.pdf',
      'system_validation_report.json', 'result_tables.json', 'figure_gallery.json',
      'manuscript_scaffold.pdf', 'manuscript_draft.json', 'agent_plan.json',
      'literature_evidence.json', 'evidence_ledger.json', 'quality_gate.json',
    ];
    function rank(resource) {
      const position = PREFERRED_ARTIFACTS.indexOf(String((resource && resource.artifact) || ''));
      return position < 0 ? PREFERRED_ARTIFACTS.length : position;
    }
    function forMessage(row, limit) {
      const rows = Array.isArray(row && row.resources) ? row.resources : [];
      return rows
        .filter(resource => key(resource))
        .filter((resource, index, all) => all.findIndex(item => key(item) === key(resource)) === index)
        .sort((left, right) => rank(left) - rank(right))
        .slice(0, Number.isFinite(limit) ? limit : 8);
    }

    function groupForMessage(row, limit) {
      const rows = forMessage(row, limit);
      return {
        primary: rows.filter(resource => resource.kind !== 'literature_source'),
        topicLiterature: rows.filter(resource => resource.kind === 'literature_source' && resource.authority_class !== 'literature_method'),
        methodLiterature: rows.filter(resource => resource.kind === 'literature_source' && resource.authority_class === 'literature_method'),
      };
    }

    function kind(resource) {
      const supported = new Set([
        'demo_document', 'demo_artifact', 'data_package_review',
        'data_workbench_snapshot', 'native_workspace', 'system_validation_document',
        'research_document', 'research_report', 'research_artifact', 'idea_plan', 'literature_source', 'webpage',
      ]);
      return supported.has(resource && resource.kind) ? resource.kind : 'file';
    }
    function button(resource, overrideLabel) {
      if (!resource) return '';
      return `<button class="gpi-resource-link" type="button"
        data-gpi-resource-kind="${esc(kind(resource))}"
        data-gpi-resource-file="${esc(resource.file || '')}"
        data-gpi-resource-run="${esc(resource.run_id || '')}"
        data-gpi-resource-artifact="${esc(resource.artifact || '')}"
        data-gpi-resource-label="${esc(resource.kind === 'demo_artifact' ? (resource.title || label(resource)) : label(resource))}"
        data-gpi-resource-media="${esc(resource.media_type || 'text/plain')}"
        data-gpi-resource-url="${esc(resource.url || '')}"
        data-gpi-resource-title="${esc(resource.title || resource.label || '')}"
        data-gpi-resource-year="${esc(resource.year || '')}"
        data-gpi-resource-venue="${esc(resource.venue || '')}"
        data-gpi-resource-relevance="${esc(resource.relevance || '')}"
        data-gpi-resource-retrieval-fit="${esc(resource.retrieval_fit || '')}"
        data-gpi-resource-retrieval-rationale="${esc(resource.retrieval_rationale || '')}"
        data-gpi-resource-doi="${esc(resource.doi || '')}"
        data-gpi-resource-pmid="${esc(resource.pmid || '')}"
        data-gpi-resource-study="${esc(resource.study_context_id || '')}"
        data-gpi-resource-revision="${esc(resource.study_revision == null ? '' : resource.study_revision)}"
        data-gpi-resource-route="${esc(resource.route || '')}"
        data-gpi-resource-state="${esc(resource.state || '')}"
        data-gpi-resource-job="${esc(resource.job_id || '')}"
        data-gpi-resource-source="${esc(resource.source_id || '')}"
        data-gpi-resource-database="${esc(resource.expected_database || '')}"
        data-gpi-resource-entry-mode="${esc(resource.entry_mode || '')}"
        data-gpi-resource-view="${esc(resource.view || '')}"
        data-gpi-resource-authority="${esc(resource.authority_class || '')}"
        data-gpi-resource-digest="${esc(resource.snapshot_sha256 || resource.review_sha256 || resource.checked_sha256 || resource.sha256 || '')}">${esc(overrideLabel || label(resource))}</button>`;
    }

    function renderForMessage(row, limit) {
      // The current candidate-plan card immediately below this receipt owns
      // its plan, evidence, and review links. Repeating the same artifacts in
      // the receipt gives the researcher two competing review surfaces.
      if (['auto_generate_plan', 'auto_revise_plan', 'generate_plan']
        .includes(String(row && row.hostActionCode || ''))) return '';
      const grouped = groupForMessage(row, limit);
      const hasLiterature = grouped.topicLiterature.length || grouped.methodLiterature.length;
      const hasDataWorkbench = ['prepare_analysis_data', 'review_prepared_data']
        .includes(String(row && row.hostActionCode || ''));
      if (!grouped.primary.length && !hasLiterature && !hasDataWorkbench) return '';
      const zh = window.EU_LANG === 'zh';
      const list = resources => `<div class="gpi-resource-list">${resources.map(resource => button(resource)).join('')}</div>`;
      const numberedLiteratureList = resources => `<ol class="gpi-resource-list gpi-literature-resource-list">${resources.map(resource => `<li>${button(resource)}</li>`).join('')}</ol>`;
      const sections = [];
      if (hasDataWorkbench) {
        sections.push(`<div class="gpi-resource-section"><span class="gpi-resource-section-title">${esc(zh ? '审阅准备数据' : 'Review prepared data')}</span><div class="gpi-resource-list"><button class="gpi-resource-link" type="button" data-gpi-run-outcome-data>${esc(zh ? '打开数据可视化' : 'Open data visualization')}</button></div></div>`);
      }
      if (grouped.primary.length) {
        const onlyPlan = grouped.primary.every(resource => resource.kind === 'idea_plan');
        sections.push(`<div class="gpi-resource-section"><span class="gpi-resource-section-title">${esc(zh && onlyPlan ? '查看方案' : (zh ? '打开证据和产物' : (onlyPlan ? 'View plan' : 'Open evidence and artifacts')))}</span>${list(grouped.primary)}</div>`);
      }
      if (hasLiterature) {
        sections.push(`<div class="gpi-resource-section"><span class="gpi-resource-section-title">${esc(zh ? '本题文献检索' : 'Topic literature search')}</span>${grouped.topicLiterature.length
          ? numberedLiteratureList(grouped.topicLiterature)
          : `<p class="gpi-resource-empty">${esc(zh ? '本轮没有可展示的本题检索候选；这不代表没有相关文献。' : 'No topic-search candidates are available in this turn; this does not mean that no relevant literature exists.')}</p>`}</div>`);
      }
      if (grouped.methodLiterature.length) {
        sections.push(`<details class="gpi-method-literature"><summary>${esc(zh ? `方法学参考（${grouped.methodLiterature.length}）` : `Method references (${grouped.methodLiterature.length})`)}<span>${esc(zh ? '通用设计参考，不是本题检索结果' : 'General design references, not topic-search results')}</span></summary>${list(grouped.methodLiterature)}</details>`);
      }
      return `<div class="gpi-message-resources" aria-label="${esc(zh ? '本轮方案与文献' : 'Plan and literature for this turn')}">${sections.join('')}</div>`;
    }

    function fromButton(element) {
      if (!element) return null;
      return {
        file: element.dataset.gpiResourceFile,
        kind: element.dataset.gpiResourceKind,
        run_id: element.dataset.gpiResourceRun,
        artifact: element.dataset.gpiResourceArtifact,
        label: element.dataset.gpiResourceLabel,
        media_type: element.dataset.gpiResourceMedia,
        url: element.dataset.gpiResourceUrl,
        title: element.dataset.gpiResourceTitle,
        year: element.dataset.gpiResourceYear,
        venue: element.dataset.gpiResourceVenue,
        relevance: element.dataset.gpiResourceRelevance,
        retrieval_fit: element.dataset.gpiResourceRetrievalFit,
        retrieval_rationale: element.dataset.gpiResourceRetrievalRationale,
        doi: element.dataset.gpiResourceDoi,
        pmid: element.dataset.gpiResourcePmid,
        study_context_id: element.dataset.gpiResourceStudy,
        study_revision: element.dataset.gpiResourceRevision,
        route: element.dataset.gpiResourceRoute,
        state: element.dataset.gpiResourceState,
        job_id: element.dataset.gpiResourceJob,
        source_id: element.dataset.gpiResourceSource,
        expected_database: element.dataset.gpiResourceDatabase,
        entry_mode: element.dataset.gpiResourceEntryMode,
        view: element.dataset.gpiResourceView,
        authority_class: element.dataset.gpiResourceAuthority,
        snapshot_sha256: element.dataset.gpiResourceDigest,
        review_sha256: element.dataset.gpiResourceDigest,
        checked_sha256: element.dataset.gpiResourceDigest,
        sha256: element.dataset.gpiResourceDigest,
      };
    }

    return { name, key, label, button, forMessage, groupForMessage, renderForMessage, fromButton };
  }

  window.EasyICU.guidedPi.declare('resources', { create });
})();
