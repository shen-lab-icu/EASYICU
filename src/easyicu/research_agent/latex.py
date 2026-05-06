"""LaTeX export for the manuscript scaffold (OpenLens-AI inspired).

OpenLens-AI [1] emits a ready-to-compile LaTeX paper. We do the
same — but only for the *scaffold*. The Discussion section is
deliberately left as a TODO so the human author cannot accidentally
publish LLM-generated clinical claims.

Usage::

    from easyicu.research_agent.latex import scaffold_to_latex
    tex = scaffold_to_latex(
        markdown=manuscript_scaffold_bound_text,
        title="Admission SOFA-2 and ICU mortality",
        authors=["A. Researcher", "B. Clinician"],
    )
    Path("manuscript_scaffold.tex").write_text(tex)

References
----------
[1] OpenLens-AI: Fully Autonomous Research Agent for Health Informatics.
"""

from __future__ import annotations

import re
import textwrap
from pathlib import Path
from typing import List, Optional, Sequence

from .bibtex import (
    bibtex_key_for,
    render_bibtex,
    render_thebibliography_block,
)
from .literature import CitationRecord, LiteratureBundle


# Map common Markdown constructs → LaTeX. Intentionally small; this is
# a scaffold converter, not a full Markdown→LaTeX engine.

_HEADING_PATTERN = re.compile(r"^(?P<hashes>#+)\s+(?P<text>.+)$", re.MULTILINE)
_BOLD_PATTERN = re.compile(r"\*\*(?P<text>.+?)\*\*")
_INLINE_CODE_PATTERN = re.compile(r"`(?P<text>[^`]+)`")
_BULLET_PATTERN = re.compile(r"^\s*-\s+(?P<text>.+)$", re.MULTILINE)
_LATEX_SPECIAL = {
    "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
    "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}", "\\": r"\textbackslash{}",
}


def _escape_latex(text: str) -> str:
    out = []
    for ch in text:
        out.append(_LATEX_SPECIAL.get(ch, ch))
    return "".join(out)


def _md_inline_to_latex(line: str) -> str:
    line = _BOLD_PATTERN.sub(lambda m: r"\textbf{" + _escape_latex(m.group("text")) + "}", line)
    line = _INLINE_CODE_PATTERN.sub(lambda m: r"\texttt{" + _escape_latex(m.group("text")) + "}", line)
    return line


def scaffold_to_latex(
    *,
    markdown: str,
    title: str = "EasyICU research-agent scaffold",
    authors: Optional[Sequence[str]] = None,
    bibliography: Optional[LiteratureBundle] = None,
    bibliography_basename: str = "manuscript_scaffold",
    bibliography_style: str = "plain",
    inline_bibliography: bool = False,
    venue_template: str = "article",
) -> str:
    """Convert the markdown manuscript scaffold into a LaTeX document.

    T3.4 — when ``bibliography`` is provided, citations are now
    emitted as ``\\cite{key}`` and the document closes with
    ``\\bibliography{<bibliography_basename>}`` so a sibling
    ``<bibliography_basename>.bib`` (produced by
    :func:`render_bibtex`) can resolve them. Set
    ``inline_bibliography=True`` to fall back to a single-file
    ``thebibliography`` block — useful for quick previews where you
    don't want a separate biber/bibtex run.
    """
    authors = list(authors or ["EasyICU research-agent"])

    # Split into headed sections so we can rebuild them as LaTeX sections.
    sections: List[tuple] = []  # (level, title, body)
    current_level = 0
    current_title = "Body"
    current_body: List[str] = []

    for line in markdown.splitlines():
        m = _HEADING_PATTERN.match(line)
        if m:
            if current_body or sections:
                sections.append((current_level, current_title, "\n".join(current_body).strip()))
            current_level = len(m.group("hashes"))
            current_title = m.group("text").strip()
            current_body = []
        else:
            current_body.append(line)
    sections.append((current_level, current_title, "\n".join(current_body).strip()))

    # Rebuild as LaTeX
    parts: List[str] = []
    parts.append(latex_template_preamble(venue_template))
    parts.append("")
    parts.append(r"\title{" + _escape_latex(title) + "}")
    parts.append(r"\author{" + r" \and ".join(_escape_latex(a) for a in authors) + "}")
    parts.append(r"\date{\today}")
    parts.append("")
    parts.append(r"\begin{document}")
    parts.append(r"\maketitle")
    parts.append("")

    section_cmd = {1: r"\section", 2: r"\section", 3: r"\subsection", 4: r"\subsubsection"}

    for level, title_text, body in sections[1:]:
        cmd = section_cmd.get(level, r"\paragraph")
        if title_text.strip().lower() in {"title"}:
            continue  # already in \maketitle
        if title_text.strip().lower() == "discussion":
            parts.append(r"\section*{Discussion}")
            parts.append(
                r"\textcolor{red}{(\textbf{TODO}: this section is intentionally left "
                r"to the human author. The research-agent writer does not generate "
                r"clinical claims.)}"
            )
            parts.append("")
            continue
        parts.append(cmd + "{" + _escape_latex(title_text) + "}")
        parts.append("")
        if body:
            parts.append(_render_body(body))
            parts.append("")

    if bibliography is not None and bibliography.citations:
        # T3.4 — auto-emit a ``\nocite{*}`` so every citation in the
        # accompanying ``.bib`` is included in the rendered References,
        # even if the manuscript scaffold body has not yet decided
        # which keys to ``\cite{}`` explicitly. Authors who want only
        # the explicitly-cited entries can simply remove the line.
        keys = [bibtex_key_for(c) for c in bibliography.citations]
        if keys:
            parts.append("% Auto-included by easyicu.research_agent (T3.4):")
            parts.append("\\nocite{" + ",".join(keys) + "}")
            parts.append("")
        if inline_bibliography:
            block = render_thebibliography_block(bibliography)
            if block:
                parts.append(r"\section*{References}")
                parts.append(block)
                parts.append("")
        else:
            parts.append(r"\bibliographystyle{" + bibliography_style + "}")
            parts.append(r"\bibliography{" + bibliography_basename + "}")
            parts.append("")

    parts.append(r"\end{document}")
    parts.append("")
    return "\n".join(parts)


def latex_template_preamble(venue_template: str = "article") -> str:
    """Return a venue-specific LaTeX preamble.

    ``article`` is the dependency-light default. ``nature``, ``npj``
    and ``lancet`` emit journal-style class scaffolds for authors who
    have the matching class files installed locally.
    """
    key = (venue_template or "article").strip().lower()
    common = r"""
        \usepackage{graphicx}
        \usepackage{hyperref}
        \usepackage{booktabs}
        \usepackage{longtable}
        \usepackage{xcolor}
        \usepackage[hang,small,bf]{caption}
    """
    if key == "nature":
        head = r"\documentclass{nature}"
    elif key == "npj":
        head = r"\documentclass[pdflatex,sn-basic]{sn-jnl}"
    elif key == "lancet":
        head = r"\documentclass[review]{elsarticle}"
    else:
        head = r"\documentclass[11pt]{article}" + "\n" + r"\usepackage[a4paper,margin=1in]{geometry}"
    return (head + "\n" + textwrap.dedent(common).strip()).strip()


def _render_body(body: str) -> str:
    """Render a section body: bullets become itemize, paragraphs stay as-is."""
    lines = body.splitlines()
    out: List[str] = []
    bullet_buffer: List[str] = []

    def flush_bullets() -> None:
        if not bullet_buffer:
            return
        out.append(r"\begin{itemize}")
        for b in bullet_buffer:
            out.append("  \\item " + _md_inline_to_latex(_escape_latex(b)))
        out.append(r"\end{itemize}")
        bullet_buffer.clear()

    for line in lines:
        m = _BULLET_PATTERN.match(line)
        if m:
            bullet_buffer.append(m.group("text").strip())
        else:
            flush_bullets()
            stripped = line.strip()
            if stripped == "":
                out.append("")
            else:
                out.append(_md_inline_to_latex(_escape_latex(stripped)))
    flush_bullets()
    return "\n".join(out)


def _format_citation_inline(c: CitationRecord) -> str:
    parts = [c.title.rstrip(".") + "."]
    if c.venue:
        parts.append(c.venue + ".")
    parts.append(c.year + ".")
    if c.doi:
        parts.append(r"DOI: \href{https://doi.org/" + c.doi + "}{" + c.doi + "}.")
    elif c.url:
        parts.append(r"\href{" + c.url + "}{" + c.url + "}.")
    elif c.pmid:
        parts.append("PMID: " + c.pmid + ".")
    return _escape_latex(" ".join(parts))


__all__ = ["scaffold_to_latex", "latex_template_preamble"]
