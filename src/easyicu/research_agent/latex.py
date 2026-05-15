"""LaTeX export for the manuscript scaffold.

This module emits a ready-to-compile LaTeX paper scaffold. The
Discussion section is
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
"""

from __future__ import annotations

import re
import textwrap
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

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
# Markdown links: ``[label](url "title")``. Evidence links may point to
# relative paths and include a sha256 title; we render them as LaTeX
# footnotes so the PDF carries a readable provenance marker instead of a
# dead relative hyperlink.
_MD_LINK_PATTERN = re.compile(
    r"\[(?P<label>[^\]]+)\]\((?P<url>[^)\s]+)(?:\s+\"[^\"]*\")?\)"
)
# HTML comments — ``<!-- ... -->`` — produced by the binder when
# evidence ids are unresolved. They must be stripped, otherwise
# pdflatex tries to render ``<!`` as text.
_HTML_COMMENT_PATTERN = re.compile(r"<!--.*?-->", re.S)
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


def _escape_url(url: str) -> str:
    """Escape a URL for use inside ``\\href{...}``.

    hyperref tolerates most ASCII; the only characters we must
    backslash-escape are ``%`` and ``#`` (which break LaTeX's
    tokenizer) and ``\\`` itself. Spaces are encoded as ``%20`` so the
    URL doesn't break across LaTeX line wraps.
    """
    return (
        url.replace("\\", r"\textbackslash{}")
        .replace("%", r"\%")
        .replace("#", r"\#")
        .replace(" ", r"%20")
    )


def _md_inline_to_latex(line: str) -> str:
    """Convert markdown inline syntax to LaTeX.

    The function expects raw markdown — do NOT pre-escape special
    characters. We extract markdown links / inline code / bold *first*
    (while their syntactic markers ``[]``, ``()``, `` ` `` are still
    intact), stash the rendered LaTeX in placeholders, escape the
    remaining plain text, then put the rendered LaTeX back.
    """
    placeholders: list = []

    def _link(match: "re.Match[str]") -> str:
        url = _escape_url(match.group("url"))
        label = _escape_latex(match.group("label"))
        # Evidence links (relative paths to evidence/ directory) are
        # rendered as superscript footnote-style markers rather than
        # clickable \href — reviewers reading a PDF can't click a
        # relative path anyway. External URLs (http/https) keep \href.
        raw_url = match.group("url")
        if raw_url.startswith(("http://", "https://")):
            rendered = r"\href{" + url + "}{" + label + "}"
        else:
            # Evidence citation: render as a footnote-style marker with
            # the label preserved, so PDF reviewers can still see the
            # bound evidence id even though relative paths are not clickable.
            rendered = (
                r"\textsuperscript{[" + label + "]}"
                r"\footnote{\texttt{" + _escape_latex(raw_url) + "}}"
            )
        placeholders.append(rendered)
        return f"\x00{len(placeholders) - 1}\x00"

    def _bold(match: "re.Match[str]") -> str:
        placeholders.append(r"\textbf{" + _escape_latex(match.group("text")) + "}")
        return f"\x00{len(placeholders) - 1}\x00"

    def _code(match: "re.Match[str]") -> str:
        placeholders.append(r"\texttt{" + _escape_latex(match.group("text")) + "}")
        return f"\x00{len(placeholders) - 1}\x00"

    line = _MD_LINK_PATTERN.sub(_link, line)
    line = _BOLD_PATTERN.sub(_bold, line)
    line = _INLINE_CODE_PATTERN.sub(_code, line)
    # Now escape the remaining plain text *between* placeholders.
    line = _escape_latex(line)

    def _restore(match: "re.Match[str]") -> str:
        return placeholders[int(match.group(1))]

    # Restore placeholders from the literal NUL markers inserted above.
    line = re.sub("\x00(\\d+)\x00", _restore, line)
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
    figure_paths: Optional[Sequence[Tuple[str, str]]] = None,
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

    # Strip HTML comments (``<!-- ... -->``) — produced by the binder
    # when evidence ids are unresolved. pdflatex would otherwise emit
    # raw ``<!`` text into the rendered PDF.
    markdown = _HTML_COMMENT_PATTERN.sub("", markdown)

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
            parts.append(r"\section{Discussion}")
            parts.append("")
            parts.append(r"\footnotetext{Evidence citations are rendered as footnote-style markers so the bound manuscript remains readable in PDF form.}")
            if body:
                parts.append(_render_body(body))
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

    # Auto-embed registered figures as a Figures appendix.
    if figure_paths:
        parts.append(r"\clearpage")
        parts.append(r"\section*{Figures}")
        parts.append("")
        for idx, (fig_id, fig_rel_path) in enumerate(figure_paths, start=1):
            parts.append(r"\begin{figure}[htbp]")
            parts.append(r"\centering")
            parts.append(
                r"\includegraphics[width=\textwidth]{"
                + _escape_latex(fig_rel_path)
                + "}"
            )
            parts.append(
                r"\caption{" + _escape_latex(fig_id.replace("_", " ").title()) + "}"
            )
            parts.append(r"\label{fig:" + fig_id + "}")
            parts.append(r"\end{figure}")
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
            out.append("  \\item " + _md_inline_to_latex(b))
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
                out.append(_md_inline_to_latex(stripped))
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
