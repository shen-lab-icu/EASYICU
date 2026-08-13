"""LaTeX export for the manuscript scaffold.

This module emits a ready-to-compile LaTeX paper scaffold. The
Discussion section is
deliberately left as a TODO so the human author cannot accidentally
publish LLM-generated clinical claims.

Usage::

    from easyicu.research_agent.reporting.latex import scaffold_to_latex
    tex = scaffold_to_latex(
        markdown=manuscript_scaffold_bound_text,
        title="Admission SOFA-2 and ICU mortality",
        authors=["A. Researcher", "B. Clinician"],
    )
    Path("manuscript_scaffold.tex").write_text(tex)
"""

from __future__ import annotations

import re
from pathlib import PurePosixPath
import textwrap
from typing import List, Optional, Sequence, Tuple

from .bibtex import (
    render_thebibliography_block,
    sanitise_bibtex_key,
)
from ..literature import CitationRecord, LiteratureBundle


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
_NUMERIC_FOOTNOTE_MARKER_PATTERN = re.compile(r"\[\^(?P<id>[A-Za-z0-9_-]+)\]")
_NUMERIC_FOOTNOTE_DEFINITION_PATTERN = re.compile(
    r"^\[\^(?P<id>[A-Za-z0-9_-]+)\]:\s*(?P<text>.+)$"
)
_LITERATURE_CITATION_PATTERN = re.compile(
    r"\[@(?P<key>[A-Za-z0-9_.:-]+)\]"
)
_STANDARD_SECTION_HEADINGS = frozenset(
    {
        "abstract",
        "introduction",
        "methods",
        "results",
        "discussion",
        "limitations",
        "conclusion",
        "conclusions",
        "data availability",
        "data and code availability",
        "funding",
        "conflict of interest",
        "conflicts of interest",
        "references",
    }
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
            # Internal evidence links are manuscript audit metadata, not
            # literature citations.  Keep the stable label readable without
            # dumping a filesystem-shaped path into every page footnote.
            rendered = r"\textsuperscript{\texttt{" + label + "}}"
        placeholders.append(rendered)
        return f"\x00{len(placeholders) - 1}\x00"

    def _bold(match: "re.Match[str]") -> str:
        placeholders.append(r"\textbf{" + _escape_latex(match.group("text")) + "}")
        return f"\x00{len(placeholders) - 1}\x00"

    def _code(match: "re.Match[str]") -> str:
        placeholders.append(r"\texttt{" + _escape_latex(match.group("text")) + "}")
        return f"\x00{len(placeholders) - 1}\x00"

    def _numeric_marker(match: "re.Match[str]") -> str:
        placeholders.append(
            r"\textsuperscript{\texttt{"
            + _escape_latex(match.group("id"))
            + "}}"
        )
        return f"\x00{len(placeholders) - 1}\x00"

    def _literature_citation(match: "re.Match[str]") -> str:
        placeholders.append(
            r"\cite{" + sanitise_bibtex_key(match.group("key")) + "}"
        )
        return f"\x00{len(placeholders) - 1}\x00"

    line = _LITERATURE_CITATION_PATTERN.sub(_literature_citation, line)
    line = _NUMERIC_FOOTNOTE_MARKER_PATTERN.sub(_numeric_marker, line)
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
    draft_watermark: bool = False,
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

    if bibliography is not None:
        allowed_citation_keys = {record.key for record in bibliography.citations}
        requested_citation_keys = set(_LITERATURE_CITATION_PATTERN.findall(markdown))
        unknown_citation_keys = sorted(
            requested_citation_keys - allowed_citation_keys
        )
        if unknown_citation_keys:
            raise ValueError(
                "manuscript cites keys absent from the run-bound bibliography: "
                + ", ".join(unknown_citation_keys)
            )

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

    # Writer output uses the first H1 as the article title and commonly places
    # the keyword line in that H1 body.  The old converter skipped the entire
    # first section unconditionally, which silently discarded both pieces and
    # rendered a generic API fallback title instead.  Only consume a leading H1
    # as document metadata; all other first sections remain ordinary content.
    leading_title_body = ""
    body_sections = list(sections)
    if body_sections and body_sections[0][0] == 1:
        heading_title = str(body_sections[0][1]).strip()
        heading_key = heading_title.casefold()
        if heading_title and heading_key not in {
            "body",
            *_STANDARD_SECTION_HEADINGS,
        }:
            if heading_key != "title":
                title = heading_title
            leading_title_body = str(body_sections[0][2] or "").strip()
            body_sections = body_sections[1:]

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
    if draft_watermark:
        parts.extend(
            [
                r"\begin{center}",
                r"\fcolorbox{red}{red!5}{\parbox{0.92\linewidth}{\centering\bfseries\color{red}DRAFT -- NOT FOR SUBMISSION\\Human scientific and authorship review required.}}",
                r"\end{center}",
                "",
            ]
        )

    section_cmd = {1: r"\section", 2: r"\section", 3: r"\subsection", 4: r"\subsubsection"}

    if leading_title_body:
        parts.append(_render_body(leading_title_body))
        parts.append("")

    for level, title_text, body in body_sections:
        cmd = section_cmd.get(level, r"\paragraph")
        if title_text.strip().lower() in {"title"}:
            continue  # already in \maketitle
        if title_text.strip().lower() == "discussion":
            parts.append(r"\section{Discussion}")
            parts.append("")
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
            normalized_figure_path = str(fig_rel_path).replace("\\", "/")
            figure_parts = PurePosixPath(normalized_figure_path).parts
            if (
                not normalized_figure_path
                or normalized_figure_path.startswith("/")
                or ".." in figure_parts
                or figure_parts[:2] == ("evidence", "evidence")
            ):
                raise ValueError(
                    "figure_paths must contain one safe run-relative evidence path"
                )
            parts.append(r"\begin{figure}[htbp]")
            parts.append(r"\centering")
            parts.append(
                r"\includegraphics[width=\textwidth]{"
                + _escape_latex(normalized_figure_path)
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
        definition = _NUMERIC_FOOTNOTE_DEFINITION_PATTERN.match(line.strip())
        if definition:
            flush_bullets()
            out.append(
                r"\par\noindent\textsuperscript{\texttt{"
                + _escape_latex(definition.group("id"))
                + r"}}\texttt{"
                + _escape_latex(definition.group("text"))
                + "}"
            )
            continue
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
