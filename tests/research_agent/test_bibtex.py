"""BibTeX export + LaTeX integration (T3.4).

Pin three contracts:

1. ``render_bibtex`` produces a syntactically-plausible ``.bib`` for
   any LiteratureBundle, with sanitised keys and best-effort entry
   typing.
2. ``scaffold_to_latex`` emits ``\\bibliography{…}`` +
   ``\\bibliographystyle{…}`` (or a ``thebibliography`` block when
   ``inline_bibliography=True``) so a single ``pdflatex`` cycle can
   resolve every reference.
3. The end-to-end pipeline writes both ``manuscript_scaffold.tex``
   and ``manuscript_scaffold.bib`` whenever a literature bundle is
   produced — and the ``.tex`` references the ``.bib`` by basename.
"""

from __future__ import annotations

import re
from pathlib import Path


# ---------------------------------------------------------------------------
# Pure-rendering tests (no pipeline)
# ---------------------------------------------------------------------------


def test_sanitise_bibtex_key_strips_unsafe_chars(ra):
    from easyicu.research_agent.reporting.bibtex import sanitise_bibtex_key
    assert sanitise_bibtex_key("vincent_sofa_1996") == "vincent_sofa_1996"
    assert sanitise_bibtex_key("Vincent JL — 1996") == "vincent_jl___1996" or \
           sanitise_bibtex_key("Vincent JL — 1996").startswith("vincent_jl")
    # Hyphen is replaced (biber dislikes inside \cite{}).
    assert "-" not in sanitise_bibtex_key("foo-bar")
    # Empty / whitespace → "ref".
    assert sanitise_bibtex_key("") == "ref"
    assert sanitise_bibtex_key("   ") == "ref"


def test_render_bibtex_basic(ra):
    from easyicu.research_agent.literature import CitationRecord, LiteratureBundle
    from easyicu.research_agent.reporting.bibtex import render_bibtex
    bundle = LiteratureBundle(
        research_question="x",
        citations=[
            CitationRecord(
                key="vincent_sofa_1996",
                title="The SOFA score to describe organ dysfunction/failure.",
                year="1996", venue="Intensive Care Medicine",
                pmid="8844239",
                doi="10.1007/BF01709751",
                relevance="Foundational SOFA reference.",
            ),
            CitationRecord(
                key="ricu_2023", title="ricu: R's interface to intensive care data.",
                year="2023", venue="Software",
                url="https://github.com/eth-mds/ricu",
            ),
        ],
    )
    bib = render_bibtex(bundle)
    # Header forms
    assert bib.startswith("@article{vincent_sofa_1996,")
    assert "@software{ricu_2023," in bib
    # Title is wrapped in extra braces to preserve case.
    assert "{{The SOFA score" in bib

    # Whitespace-tolerant field matcher: verify each "name … = {value}"
    # appears regardless of how the renderer pads the column.
    def _has_field(text: str, name: str, value: str) -> bool:
        return re.search(rf"\b{name}\s*=\s*\{{{re.escape(value)}\}}", text) is not None

    assert _has_field(bib, "year", "1996")
    assert _has_field(bib, "journal", "Intensive Care Medicine")
    assert _has_field(bib, "doi", "10.1007/BF01709751")
    assert _has_field(bib, "note", "PMID: 8844239")
    # Software entry uses howpublished, not journal.
    assert _has_field(bib, "howpublished", "Software")
    # The annotation field carries the agent's relevance note.
    assert "annote" in bib


def test_render_bibtex_disambiguates_duplicate_keys(ra):
    from easyicu.research_agent.literature import CitationRecord, LiteratureBundle
    from easyicu.research_agent.reporting.bibtex import render_bibtex
    bundle = LiteratureBundle(
        research_question="x",
        citations=[
            CitationRecord(key="dup", title="A", year="2020"),
            CitationRecord(key="dup", title="B", year="2021"),
            CitationRecord(key="dup", title="C", year="2022"),
        ],
    )
    bib = render_bibtex(bundle)
    keys = re.findall(r"@\w+\{([^,]+),", bib)
    assert keys == ["dup", "dup_2", "dup_3"], keys


def test_render_bibtex_empty(ra):
    from easyicu.research_agent.reporting.bibtex import render_bibtex
    assert render_bibtex(None) == ""
    from easyicu.research_agent.literature import LiteratureBundle
    assert render_bibtex(LiteratureBundle(research_question="x", citations=[])) == ""


def test_render_bibtex_escapes_field_specials(ra):
    from easyicu.research_agent.literature import CitationRecord, LiteratureBundle
    from easyicu.research_agent.reporting.bibtex import render_bibtex
    bundle = LiteratureBundle(
        research_question="x",
        citations=[
            CitationRecord(
                key="tricky_2024", title="100% certain & sound",
                year="2024", venue="Journal of $tuff",
            ),
        ],
    )
    bib = render_bibtex(bundle)
    assert r"\&" in bib
    assert r"\%" in bib
    assert r"\$" in bib


def test_thebibliography_block(ra):
    from easyicu.research_agent.literature import CitationRecord, LiteratureBundle
    from easyicu.research_agent.reporting.bibtex import render_thebibliography_block
    bundle = LiteratureBundle(
        research_question="x",
        citations=[
            CitationRecord(
                key="vincent_sofa_1996", title="SOFA score.",
                year="1996", venue="Intensive Care Medicine", pmid="8844239",
            ),
        ],
    )
    block = render_thebibliography_block(bundle)
    assert block.startswith(r"\begin{thebibliography}")
    assert r"\bibitem{vincent_sofa_1996}" in block
    assert "PMID:8844239" in block


# ---------------------------------------------------------------------------
# scaffold_to_latex integration
# ---------------------------------------------------------------------------


def test_scaffold_to_latex_emits_bibliography_directives(ra):
    from easyicu.research_agent.literature import CitationRecord, LiteratureBundle
    from easyicu.research_agent.reporting.latex import scaffold_to_latex

    bundle = LiteratureBundle(
        research_question="x",
        citations=[
            CitationRecord(key="vincent_sofa_1996", title="SOFA.", year="1996",
                           venue="Intensive Care Medicine", pmid="8844239"),
        ],
    )
    md = (
        "# Evidence-bound ICU study\n\n"
        "**Keywords:** ICU, evidence\n\n"
        "## Methods\n\n"
        "Cohort definition follows prior work [@vincent_sofa_1996].\n"
    )
    tex = scaffold_to_latex(markdown=md, title="T", authors=["A"], bibliography=bundle)
    assert r"\bibliographystyle{plain}" in tex
    assert r"\bibliography{manuscript_scaffold}" in tex
    assert r"\cite{vincent_sofa_1996}" in tex
    assert r"\nocite{" not in tex
    assert r"\title{Evidence-bound ICU study}" in tex
    assert "Keywords" in tex
    # The old itemize-style references list must NOT be present.
    assert r"\section*{References}" not in tex or r"\bibliography{" in tex


def test_scaffold_to_latex_inline_bibliography_fallback(ra):
    from easyicu.research_agent.literature import CitationRecord, LiteratureBundle
    from easyicu.research_agent.reporting.latex import scaffold_to_latex

    bundle = LiteratureBundle(
        research_question="x",
        citations=[
            CitationRecord(key="ricu_2023", title="ricu.", year="2023",
                           venue="Software"),
        ],
    )
    tex = scaffold_to_latex(
        markdown="# Methods\n\nbody\n",
        bibliography=bundle, inline_bibliography=True,
    )
    assert r"\begin{thebibliography}" in tex
    assert r"\bibitem{ricu_2023}" in tex
    # No external bibtex pointer when inline.
    assert r"\bibliography{" not in tex


def test_scaffold_to_latex_no_bibliography(ra):
    """Without a bundle, no References section / nocite is emitted."""
    from easyicu.research_agent.reporting.latex import scaffold_to_latex
    tex = scaffold_to_latex(markdown="# Methods\n\nbody\n", bibliography=None)
    assert r"\bibliography{" not in tex
    assert r"\nocite{" not in tex
    assert r"\begin{thebibliography}" not in tex


def test_latex_template_preamble_supports_venues(ra):
    from easyicu.research_agent.reporting.latex import latex_template_preamble, scaffold_to_latex

    assert r"\documentclass{nature}" in latex_template_preamble("nature")
    assert "sn-jnl" in latex_template_preamble("npj")
    assert "elsarticle" in latex_template_preamble("lancet")
    tex = scaffold_to_latex(
        markdown="# Title\n\n## Methods\n\nBody",
        venue_template="npj",
    )
    assert "sn-jnl" in tex


# ---------------------------------------------------------------------------
# End-to-end: pipeline writes .tex AND .bib
# ---------------------------------------------------------------------------


def test_unknown_markdown_literature_key_fails_closed(ra):
    from easyicu.research_agent.literature import CitationRecord, LiteratureBundle
    from easyicu.research_agent.reporting.latex import scaffold_to_latex
    import pytest

    bundle = LiteratureBundle(
        research_question="x",
        citations=[CitationRecord(key="known", title="Known.", year="2024")],
    )
    with pytest.raises(ValueError, match="absent from the run-bound bibliography"):
        scaffold_to_latex(
            markdown="# Methods\n\nClaim [@invented].\n",
            bibliography=bundle,
        )


def test_pipeline_writes_bib_alongside_tex(ra, synthetic_cohort, tmp_path: Path):
    pipeline = ra.ResearchAgentPipeline(workdir=tmp_path, llm=ra.MockLLMClient())
    result = pipeline.run(
        question="Is admission SOFA-2 associated with ICU mortality?",
        cohort=synthetic_cohort,
        cohort_name="bibtex_test",
        database="synthetic",
        target_outcome="death",
    )
    run_dir = Path(result.workdir)
    tex = run_dir / "manuscript_scaffold.tex"
    bib = run_dir / "manuscript_scaffold.bib"
    assert tex.exists(), "manuscript_scaffold.tex must be written"
    assert bib.exists(), "manuscript_scaffold.bib must be written when literature is present"

    tex_text = tex.read_text(encoding="utf-8")
    bib_text = bib.read_text(encoding="utf-8")
    assert r"\bibliography{manuscript_scaffold}" in tex_text
    assert bib_text.startswith("@") and "}\n" in bib_text
    # Every exact manuscript citation must resolve to the run-bound .bib.
    cite_keys = re.findall(r"\\cite\{([^}]+)\}", tex_text)
    bib_keys = set(re.findall(r"@\w+\{([^,]+),", bib_text))
    missing = [k for k in cite_keys if k not in bib_keys]
    assert not missing, f"\\cite keys absent from .bib: {missing}"
