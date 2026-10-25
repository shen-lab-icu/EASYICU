"""Focused reader-facing literature projection contracts."""

import json
import shutil

import pytest

from .test_pi_copilot_static import _ESCAPE_OWNER, _read
from easyicu.webserver.pi_copilot import projections
from easyicu.webserver.literature_projection import project_run_literature

from tests.support.node import run_node


_MODULE_OWNER = _read("js/screens-guided-pi-modules.js")


def _render_literature(payload, *, language="zh", meta=None):
    source = _read("js/screens-guided-pi-literature.js")
    script = f"""
      global.window = {{ EU_LANG: {json.dumps(language)} }};
      eval({_ESCAPE_OWNER!r}); eval({_MODULE_OWNER!r}); eval({source!r});
      process.stdout.write(window.EasyICU.guidedPi.require('literature')
        .renderArtifact({json.dumps(payload)}, {json.dumps(meta or {})}));
    """
    return run_node("node", script, check=True).stdout


@pytest.mark.parametrize("projected", [False, True])
def test_curated_guidance_is_not_a_source_abstract(projected):
    rows = [
        {"key": "guide", "title": "Reporting checklist",
         "relevance": "Methodology: Describe exclusions and missing values."},
        {"key": "paper", "title": "Synthetic marker study",
         "relevance": "Study-design excerpt: Marker alpha was measured at admission."},
        {"key": "note", "title": "Background reference",
         "relevance": "A planning note, with no retained source passage.",
         "screening": {"disposition": "exclude", "design_excerpt_available": False}},
    ]
    payload = {"citations": rows}
    if projected:
        payload = project_run_literature(run_id="run_notes", bundle=payload, plan={})
    html = _render_literature(payload)
    cards = html.split('<article class="gpi-lit-card ')[1:]
    by_title = {row["title"]: next(card.split("</article>")[0] for card in cards
                                  if row["title"] in card) for row in rows}
    assert "系统参考说明（非原文摘要）" in by_title["Reporting checklist"]
    assert "查看保留的来源片段" not in by_title["Reporting checklist"]
    assert "查看保留的来源片段" in by_title["Synthetic marker study"]
    assert "查看保留的来源片段" not in by_title["Background reference"]
    assert "查看系统保留的摘要片段" not in html


@pytest.mark.parametrize("language", ["zh", "en"])
def test_paper_use_comes_from_binding_and_keeps_actual_application(language):
    payload = {
        "citations": [{"key": "guide", "title": "A neutral checklist"}],
        "step_citation_map": [{
            "intent": "Fit an adjusted model in the eligible population",
            "citation_bindings": [{
                "key": "guide", "title": "A neutral checklist",
                "design_elements": ["reporting"],
                "application": "Report each exclusion and its recorded reason <exactly>.",
                "divergence": None,
            }],
        }],
    }
    html = _render_literature(payload, language=language)
    card = html.split('<article class="gpi-lit-card ')[1].split("</article>")[0]
    assert "Report each exclusion and its recorded reason &lt;exactly&gt;." in card
    assert "Fit an adjusted model" not in card
    assert "确定研究人群与纳排标准" not in card
    assert "确定混杂因素与统计模型" not in card
    assert ("规范研究报告" if language == "zh" else "Transparent reporting") in card
    assert ("差异尚未核对" if language == "zh" else "Differences not yet appraised") in card


@pytest.mark.parametrize("disposition,bound", [("include", False), ("exclude", False), ("exclude", True)])
def test_retained_paper_shows_recorded_screening_reason(disposition, bound):
    payload = {
        "direct_comparator_count": int(disposition == "include"),
        "direct_comparator_keys": ["paper"] if disposition == "include" else [],
        "citations": [{"key": "paper", "title": "Marker alpha study",
                       "screening": {"disposition": disposition,
                                     "rationale": "Recorded decision: <scope differs>.",
                                     "population_match": True, "exposure_match": False}}],
        "step_citation_map": [{"intent": "Model choice", "citation_bindings": [{
            "key": "paper", "design_elements": ["exposure"],
            "application": "Compare the marker definition only.",
            "divergence": "Admission measurement differs from the planned window.",
        }]}] if bound else [],
    }
    html = _render_literature(payload)
    assert "Recorded decision: &lt;scope differs&gt;." in html
    assert ("候选纳入理由" if disposition == "include" else "未作为直接依据的原因") in html
    if bound:
        assert "Admission measurement differs from the planned window." in html
    else:
        assert "尚未记录具体应用" in html


def test_search_summary_distinguishes_search_receipt_from_preset_references():
    html = _render_literature({"search": {
        "search_conducted": True, "searched_at": "2026-09-03T04:05:06+00:00",
        "curated_seed_count": 3, "prisma": {"identified": 2, "screened": 1},
    }})
    assert "2026-09-03T04:05:06+00:00" in html
    assert "本次检索" in html and "<strong>2</strong>" in html
    assert "系统预置参考" in html and "<strong>3</strong>" in html
    unknown = _render_literature({"search": {"search_conducted": True}})
    assert "检索时间未记录" in unknown
    assert "<strong>未记录</strong>" in unknown


def test_citation_membership_is_not_reviewed_support_or_novelty():
    payload = {
        "direct_comparator_count": 1, "direct_comparator_keys": ["paper"],
        "citations": [{"key": "paper", "title": "Marker alpha study"}],
        "step_citation_map": [{"intent": "Adjusted model", "citation_bindings": [{
            "key": "method", "title": "Method reference", "design_elements": ["adjustment"],
        }]}],
        "novelty_positioning": {"status": "review_required", "review_disposition": "not_available",
                                "reviewer_owner": None, "reviewed_at": None},
    }
    html = _render_literature(payload)
    assert "已绑定文献，待核对" in html
    assert "已有科学依据" not in html
    assert "gpi-lit-coverage supported" not in html
    assert "创新性比较：尚未核验" in html
    assert "data-gpi-literature-review" not in html
    linked = _render_literature(payload, meta={"reviewAvailable": True})
    assert "data-gpi-literature-review" in linked and "查看科学审阅" in linked


@pytest.mark.parametrize("artifact", ["scientific_readiness.json", "scientific_plan_review.json"])
def test_literature_review_entry_uses_only_same_run_digest_bound_catalog(artifact):
    preview = _read("js/screens-guided-pi-preview.js")
    literature = _read("js/screens-guided-pi-literature.js")
    script = f"""
      const assert = require('node:assert/strict');
      global.window = {{ EU_LANG: 'zh' }};
      global.document = {{ getElementById: () => null }};
      eval({_ESCAPE_OWNER!r}); eval({_MODULE_OWNER!r}); eval({literature!r});
      window.EasyICU.guidedPi.declare('resources', {{ create: () => ({{key: r => JSON.stringify(r)}}) }});
      eval({preview!r});
      let handler, content = '', renders = 0; const opened = [];
      const host = {{ get innerHTML() {{ return content; }},
        set innerHTML(value) {{ content = value; renders++; }},
        addEventListener: (_t, fn) => {{ handler = fn; }},
        querySelector: () => null, querySelectorAll: () => [], replaceChildren: () => {{}} }};
      const owner = window.EasyICU.guidedPi.require('preview');
      const review = {{kind: 'research_artifact', run_id: 'run_current',
        artifact: {json.dumps(artifact)}, sha256: 'b'.repeat(64), label: '科学审阅'}};
      const loads = [];
      window.EU_API = {{ loadPiCopilotResearchArtifact: async (...args) => {{
        loads.push(args); return {{payload: {{citations: []}}}};
      }} }};
      owner.mount(host);
      owner.setStudyResources([review], 'project_current', r => opened.push(r));
      owner.open({{kind: 'research_artifact', run_id: 'run_current',
        artifact: 'literature_evidence.json', sha256: 'a'.repeat(64)}}, 'project_current');
      const click = () => handler({{target: {{ closest: s => s === '[data-gpi-literature-review]' ? {{}} : null }} }});
      setImmediate(() => {{
        assert.match(host.innerHTML, /data-gpi-literature-review/);
        click(); assert.equal(opened.length, 1); assert.equal(opened[0].sha256, review.sha256);
        const stableRenders = renders;
        owner.setStudyResources([review], 'project_current', r => opened.push(r));
        assert.equal(renders, stableRenders, 'Polling must preserve expanded literature details');
        for (const [r, project] of [
          [{{...review, run_id: 'run_old'}}, 'project_current'],
          [{{...review, sha256: ''}}, 'project_current'],
          [{{...review, artifact: 'unavailable.json'}}, 'project_current'],
          [review, 'project_other'],
        ]) {{
          owner.setStudyResources([r], project, r => opened.push(r));
          assert.doesNotMatch(host.innerHTML, /data-gpi-literature-review/);
          click(); assert.equal(opened.length, 1);
        }}
        assert.equal(loads.length, 1, 'No speculative review fetch');
        owner.setStudyResources([review], 'project_current');
        click();
        assert.deepEqual(loads[1], ['project_current', 'run_current', review.artifact, review.sha256]);
        process.stdout.write('ok');
      }});
    """
    assert run_node("node", script, check=True).stdout == "ok"


@pytest.mark.parametrize("renderer", ["renderSource", "renderArtifact"])
@pytest.mark.parametrize("title", [
    "Robustness of sepsis-3 criteria in critically ill patients",
    "Prediction of 30-day mortality for ICU patients with Sepsis-3",
    "Prognostic accuracy of the SOFA score in suspected infection",
    "Assessment of STROBE reporting quality in cohort studies",
])
def test_chinese_reader_preserves_the_exact_article_title(renderer, title):
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is not installed")
    row = {"key": "retrieved_study", "title": title}
    payload = row if renderer == "renderSource" else {"citations": [row]}
    source = _read("js/screens-guided-pi-literature.js")
    script = f"""
      global.window = {{ EU_LANG: 'zh' }};
      eval({_ESCAPE_OWNER!r}); eval({_MODULE_OWNER!r}); eval({source!r});
      console.log(window.EasyICU.guidedPi.require('literature').{renderer}({json.dumps(payload)}));
    """
    html = run_node(node, script, check=True).stdout
    assert f"<h4>{title}</h4>" in html
    assert "脓毒症 Sepsis-3 共识定义" not in html
    assert "SOFA 器官功能评分定义" not in html


def test_run_projection_retains_bounded_source_notices_without_mutating_evidence():
    notices = ["Author correction: 10.1038/example.", "x" * 900]
    bundle = {"citations": [{"key": "source", "title": "Source title", "bibliographic_notices": notices}]}
    before = json.dumps(bundle)
    projected = project_run_literature(run_id="run_test", bundle=bundle, plan={})
    assert projected["citations"][0]["bibliographic_notices"] == [notices[0], "x" * 600]
    assert json.dumps(bundle) == before


@pytest.mark.parametrize("notices", ["not a list", {"private": "not a notice"}, [None, {}, 123]])
def test_run_projection_does_not_coerce_malformed_notices(notices):
    projected = project_run_literature(
        run_id="run_test", plan={},
        bundle={"citations": [{"key": "source", "title": "Source title", "bibliographic_notices": notices}]},
    )
    assert projected["citations"][0]["bibliographic_notices"] == []


@pytest.mark.parametrize("renderer", ["renderSource", "renderArtifact"])
def test_reader_displays_escaped_bibliographic_notices(renderer):
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is not installed")
    row = {"key": "source", "title": "Source title", "bibliographic_notices": ["Correction: <script>alert(1)</script>"]}
    payload = row if renderer == "renderSource" else {"citations": [row]}
    source = _read("js/screens-guided-pi-literature.js")
    script = f"""
      global.window = {{ EU_LANG: 'zh' }};
      eval({_ESCAPE_OWNER!r}); eval({_MODULE_OWNER!r}); eval({source!r});
      console.log(window.EasyICU.guidedPi.require('literature').{renderer}({json.dumps(payload)}));
    """
    html = run_node(node, script, check=True).stdout
    assert "来源勘误与声明" in html
    assert "Correction: &lt;script&gt;alert(1)&lt;/script&gt;" in html
    assert "<script>" not in html


def test_literature_reader_separates_direct_evidence_from_system_references() -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is not installed")
    source = _read("js/screens-guided-pi-literature.js")
    script = f"""
      global.window = {{ EU_LANG: 'zh' }};
      eval({_ESCAPE_OWNER!r});
      eval({_MODULE_OWNER!r});
      eval({source!r});
      const html = window.EasyICU.guidedPi.require('literature').renderArtifact({{
        direct_comparator_count: 0,
        direct_comparator_keys: [],
        search: {{
          search_conducted: true,
          note: 'Retrieval ran; PRISMA counts describe the records returned.',
          prisma: {{identified: 8, screened: 8, included: 0}},
          queries: {{pubmed: ['lactate AND in-hospital mortality AND ICU']}},
        }},
        evidence_boundary: 'internal EvidenceStore boundary',
        citations: [
          {{ key: 'strobe_2007', title: 'STROBE statement', year: '2007',
             source_url: 'https://pubmed.ncbi.nlm.nih.gov/17938396/' }},
          {{ key: 'singer_sepsis3_2016', title: 'Sepsis-3 consensus', year: '2016' }},
          {{ key: 'retrieved_lar', title: 'Lactate-to-albumin ratio study', year: '2023',
             screening: {{disposition: 'exclude', population_match: true,
               exposure_match: false, outcome_match: true,
               design_excerpt_available: true, publication_type_eligible: true}} }},
        ],
        step_citation_map: [
          {{ step_id: 'descriptive_quality_summary', planned_analysis_role: 'primary',
             intent: '说明研究对象、变量定义和描述性结果。',
             citation_bindings: [{{ key: 'strobe_2007', title: 'STROBE statement',
               year: '2007', application: '规范报告研究对象与变量定义。',
               design_elements: ['reporting'],
               source_url: 'https://pubmed.ncbi.nlm.nih.gov/17938396/' }}] }},
          {{ step_id: '04_publication_figure_fallback', planned_analysis_role: 'auxiliary',
             intent: 'internal fallback', citation_bindings: [] }},
        ],
      }});
      console.log(html);
    """
    completed = run_node(node, script, check=True)
    html = completed.stdout
    assert "没有找到能直接支持这个问题的研究" in html
    assert "共检索到 8 篇候选、完成 8 篇筛选" in html
    assert "本次检索暂无文章通过筛选" in html
    assert "变量与协变量依据" in html
    assert "统计方法依据" in html
    assert "报告规范" in html
    assert "仅有报告规范不能决定研究因素时间窗" in html
    assert "这些文献只规范透明报告，不证明当前研究因素与结局存在关联" in html
    assert "1 个决定" not in html
    assert "系统参考库里的其他资料" in html
    assert "为什么显示" not in html
    assert "PRISMA" not in html
    assert "EvidenceStore" not in html
    assert "descriptive_quality_summary" not in html
    assert "04_publication_figure_fallback" not in html
    assert "auxiliary" not in html


def test_literature_source_preview_preserves_retrieval_fit_without_claiming_acceptance() -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is not installed")
    source = _read("js/screens-guided-pi-literature.js")
    script = f"""
      global.window = {{ EU_LANG: 'zh' }};
      eval({_ESCAPE_OWNER!r});
      eval({_MODULE_OWNER!r});
      eval({source!r});
      console.log(window.EasyICU.guidedPi.require('literature').renderSource({{
        title: 'ICU hypotension treatment by staffing level',
        url: 'https://pubmed.ncbi.nlm.nih.gov/26975737/',
        pmid: '26975737',
        retrieval_fit: 'direct_retrieval_fit',
        retrieval_rationale: 'Direct retrieval fit; full screening remains pending.',
      }}));
    """
    completed = run_node(node, script, check=True)
    html = completed.stdout
    assert "直接检索匹配 · 待筛选" in html
    assert "检索到但未采用" not in html
    assert "系统参考资料" not in html


def test_literature_source_preview_explains_article_type_and_full_text_boundary() -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is not installed")
    source = _read("js/screens-guided-pi-literature.js")
    script = f"""
      global.window = {{ EU_LANG: 'zh' }};
      eval({_ESCAPE_OWNER!r});
      eval({_MODULE_OWNER!r});
      eval({source!r});
      console.log(window.EasyICU.guidedPi.require('literature').renderSource({{
        title: 'Delayed awakening after sedation interruption',
        url: 'https://pubmed.ncbi.nlm.nih.gov/12345/',
        pmid: '12345',
        retrieval_fit: 'adjacent_retrieval_fit',
        article_kind: 'systematic_review',
        publication_types: ['Systematic Review'],
        abstract_excerpt: 'Delayed awakening has multiple competing explanations.',
        source_review_status: 'reviewed',
        full_text: {{status: 'reviewed', url: 'https://pmc.ncbi.nlm.nih.gov/articles/PMC12345/', evidence_spans: [
          {{section: 'results', label: 'Results', excerpt: 'Recovery definitions varied across studies.'}}
        ]}},
      }}));
    """
    completed = run_node(node, script, check=True)
    html = completed.stdout
    assert "系统综述 / Meta 分析" in html
    assert "为什么收录这篇文献" in html
    assert "它能支持什么" in html
    assert "它不能支持什么" in html
    assert "Delayed awakening has multiple competing explanations" in html
    assert "正文补充" in html
    assert "Recovery definitions varied across studies" in html
    assert "不能完整回答当前 Idea" in html


def test_replay_projection_keeps_bounded_literature_retrieval_fit() -> None:
    resource = projections._project_replay_resource(
        {
            "kind": "literature_source",
            "url": "https://pubmed.ncbi.nlm.nih.gov/26975737/",
            "title": "ICU hypotension treatment by staffing level",
            "pmid": "26975737",
            "authority_class": "literature_retrieval_candidate",
            "retrieval_fit": "direct_retrieval_fit",
            "retrieval_rationale": "Direct retrieval fit; full screening pending.",
            "relevance": "Direct retrieval fit; full screening pending.",
        }
    )
    assert resource is not None
    assert resource["retrieval_fit"] == "direct_retrieval_fit"
    assert "full screening pending" in resource["retrieval_rationale"]


def test_preview_sanitizer_keeps_only_bounded_literature_retrieval_fit() -> None:
    preview = _read("js/screens-guided-pi-preview.js")

    assert "const retrievalFit = ['direct_retrieval_fit', 'adjacent_retrieval_fit', 'unclassified']" in preview
    assert "retrieval_fit: retrievalFit" in preview
    assert "String(value.retrieval_rationale || '').slice(0, 600)" in preview
