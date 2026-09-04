"""Focused reader-facing literature projection contracts."""

import shutil
import subprocess

import pytest

from .test_pi_copilot_static import _ESCAPE_OWNER, _read
from easyicu.webserver.pi_copilot import projections


_MODULE_OWNER = _read("js/screens-guided-pi-modules.js")


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
    completed = subprocess.run(
        [node, "--eval", script], check=True, capture_output=True, text=True
    )
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
    completed = subprocess.run(
        [node, "--eval", script], check=True, capture_output=True, text=True
    )
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
    completed = subprocess.run(
        [node, "--eval", script], check=True, capture_output=True, text=True
    )
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
