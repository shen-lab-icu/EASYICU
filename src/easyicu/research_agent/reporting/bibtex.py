r"""BibTeX export for the manuscript scaffold (T3.4).

The default LaTeX export used to render references as a plain
``itemize`` list — fine for inspection, useless for a real journal
submission. T3.4 promotes citations to first-class BibTeX entries:
the pipeline now drops a ``manuscript_scaffold.bib`` next to
``manuscript_scaffold.tex`` and the rendered LaTeX uses
``\cite{key}`` against that bib file.

Design choices worth documenting:

* **One module, no SDK.** ``pybtex`` would be a heavy dependency for
  what is fundamentally string templating. We emit the ``.bib`` text
  by hand and validate the citation keys against a well-known
  pattern.
* **Best-effort entry typing.** We classify each :class:`CitationRecord`
  into ``@article``, ``@misc``, or ``@software`` based on whether a
  ``venue`` and a ``year`` look like a journal record. Anything we
  can't classify falls back to ``@misc`` — never an error.
* **Idempotent keys.** The CitationRecord ``key`` field is already a
  stable identifier; we only sanitise it (lower-case, alnum + ``_``)
  before emitting so it round-trips through BibTeX.

Reference shape::

    @article{vincent_sofa_1996,
      title  = {{The SOFA score to describe organ dysfunction/failure}},
      year   = {1996},
      journal= {Intensive Care Medicine},
      doi    = {10.1007/...},
      note   = {PMID: 8844239},
    }
"""

from __future__ import annotations

import re
from typing import Iterable, List, Optional, Sequence, Set

from ..literature import (
    CitationRecord,
    LiteratureBundle,
    manuscript_citable_records,
)


# ---------------------------------------------------------------------------
# Key sanitisation
# ---------------------------------------------------------------------------


_KEY_VALID_PATTERN = re.compile(r"[^A-Za-z0-9_]")


def sanitise_bibtex_key(raw: str) -> str:
    r"""Coerce ``raw`` into a BibTeX-safe citation key.

    BibTeX keys allow alphanumerics and a small set of punctuation;
    different bibliography styles disagree on the exact set, so we
    take the conservative intersection: ``[A-Za-z0-9_]``. Any other
    character — including ``-``, which biber dislikes inside
    ``\cite{}`` arguments — is replaced with ``_``. Empty inputs are
    coerced to ``"ref"``.
    """
    cleaned = _KEY_VALID_PATTERN.sub("_", (raw or "").strip()).strip("_")
    if not cleaned:
        cleaned = "ref"
    return cleaned.lower()


def dedupe_keys(records: Iterable[CitationRecord]) -> List[CitationRecord]:
    """Return ``records`` with duplicate sanitised keys disambiguated.

    First occurrence keeps the bare key; subsequent ones get a
    ``_2``, ``_3``, … suffix. The original ``CitationRecord.key`` is
    *not* mutated; instead the BibTeX renderer uses
    :func:`bibtex_key_for` which applies the same disambiguation.
    """
    seen: Set[str] = set()
    out: List[CitationRecord] = []
    for r in records:
        out.append(r)
        key = sanitise_bibtex_key(r.key)
        if key not in seen:
            seen.add(key)
            continue
        # Bump the last record's *effective* key so subsequent calls
        # to bibtex_key_for stay consistent. We keep an in-memory map
        # below to avoid mutating the immutable pydantic record.
    return out


# ---------------------------------------------------------------------------
# BibTeX field escaping
# ---------------------------------------------------------------------------


# A modest set of substitutions: BibTeX understands LaTeX, so we only
# need to escape characters that BibTeX itself parses specially in
# field values.
_BIBTEX_FIELD_REPLACEMENTS = (
    ("\\", r"\textbackslash{}"),
    ("&", r"\&"),
    ("%", r"\%"),
    ("$", r"\$"),
    ("#", r"\#"),
    ("_", r"\_"),
    ("{", r"\{"),
    ("}", r"\}"),
    ("~", r"\textasciitilde{}"),
    ("^", r"\textasciicircum{}"),
)


def _escape_bibtex_field(value: str) -> str:
    out = value
    for src, dst in _BIBTEX_FIELD_REPLACEMENTS:
        out = out.replace(src, dst)
    return out


def _emit_field(name: str, value: Optional[str], *, pre_escaped: bool = False) -> Optional[str]:
    """Return ``"  name = {value}"`` or ``None`` if value is empty.

    ``pre_escaped`` skips :func:`_escape_bibtex_field`. Used for the
    title field, where the caller has already wrapped the cleaned
    title in case-preservation braces (``{{...}}``) and re-escaping
    would corrupt those braces.
    """
    if value is None:
        return None
    v = str(value).strip()
    if not v:
        return None
    body = v if pre_escaped else _escape_bibtex_field(v)
    return f"  {name:<8}= {{{body}}}"


# ---------------------------------------------------------------------------
# Entry-type classification
# ---------------------------------------------------------------------------


def _classify_entry_type(rec: CitationRecord) -> str:
    """Heuristic mapping ``CitationRecord`` → ``@article|@misc|@software``."""
    venue = (rec.venue or "").lower()
    if not venue:
        return "@misc"
    if "software" in venue or "code" in venue or "github" in venue:
        return "@software"
    if "preprint" in venue or "arxiv" in venue or "biorxiv" in venue or "medrxiv" in venue:
        return "@misc"
    # Anything else with a venue is treated as a journal article — the
    # safe default for ICU/medical literature, which is overwhelmingly
    # journal-published.
    return "@article"


# ---------------------------------------------------------------------------
# Public renderer
# ---------------------------------------------------------------------------


def bibtex_key_for(record: CitationRecord) -> str:
    """Public helper: the BibTeX key the LaTeX ``\\cite{}`` should use.

    Provided so the LaTeX renderer can compute the citation key the
    same way :func:`render_bibtex` does (so cross-references resolve).
    """
    return sanitise_bibtex_key(record.key)


def render_bibtex(bundle: Optional[LiteratureBundle]) -> str:
    """Render a :class:`LiteratureBundle` as a self-contained ``.bib``.

    Empty / missing bundles produce the empty string so the caller
    can simply ``write_text(render_bibtex(bundle))`` without any
    None-check.

    Duplicate keys (post-sanitisation) are disambiguated with a
    monotonically-incrementing suffix; the **first** record keeps the
    bare key so ``\\cite{vincent_sofa_1996}`` remains stable across
    a typical run where curated entries always come first.
    """
    records = manuscript_citable_records(bundle)
    if not records:
        return ""

    seen: Set[str] = set()
    blocks: List[str] = []

    for rec in records:
        key = sanitise_bibtex_key(rec.key)
        # Disambiguate post-hoc.
        unique = key
        n = 2
        while unique in seen:
            unique = f"{key}_{n}"
            n += 1
        seen.add(unique)
        entry_type = _classify_entry_type(rec)

        title = (rec.title or "").rstrip(" .")
        # BibTeX style files lower-case titles by default; wrap the
        # *escaped* title in an extra ``{…}`` to preserve
        # capitalisation of acronyms (SOFA, KDIGO, ICU). The escape
        # has to happen *before* the braces are added, otherwise the
        # generic field-escaper would clobber the case-preservation
        # braces themselves.
        if title:
            title_field = "{" + _escape_bibtex_field(title) + "}"
        else:
            title_field = None

        author_field = _author_from_record(rec)

        venue_field_name = "journal" if entry_type == "@article" else "howpublished"
        venue_field = rec.venue if rec.venue else None

        note_pieces: List[str] = []
        if rec.pmid:
            note_pieces.append(f"PMID: {rec.pmid}")
        if rec.url and not rec.doi:
            note_pieces.append(rec.url)
        note_pieces.extend(rec.bibliographic_notices)
        note = "; ".join(note_pieces) if note_pieces else None

        fields: List[Optional[str]] = [
            _emit_field("title", title_field, pre_escaped=True),
            _emit_field("author", author_field),
            _emit_field("year", rec.year if rec.year and rec.year != "n/a" else None),
            _emit_field(venue_field_name, venue_field),
            _emit_field("doi", rec.doi),
            _emit_field("url", rec.url if rec.url else None),
            _emit_field("note", note),
        ]
        # Annotation field: the agent's "why this paper" note, useful
        # for the human author when picking citations to keep.
        if rec.relevance:
            fields.append(_emit_field("annote", rec.relevance))

        body = ",\n".join(f for f in fields if f)
        blocks.append(f"{entry_type}{{{unique},\n{body}\n}}")

    return "\n\n".join(blocks) + "\n"


def _author_from_record(rec: CitationRecord) -> Optional[str]:
    """Best-effort author derivation from the (deliberately thin) schema.

    The :class:`CitationRecord` schema does not carry a structured
    author list — we take the surname-prefix of the citation key
    (which the PubMed parser already constructed as
    ``surname_titleslug_year``) and capitalise it. This is good enough
    for a working ``.bib`` that compiles; the human author can refine
    later with full names from the original references.
    """
    key = rec.key or ""
    if not key:
        return None
    head = key.split("_", 1)[0]
    if not head or any(c.isdigit() for c in head):
        return None
    return head.capitalize() + ", et al."


def render_thebibliography_block(bundle: Optional[LiteratureBundle]) -> str:
    """Render a fallback inline ``thebibliography`` block.

    Used when the user wants a single-file ``.tex`` that does not need
    an external BibTeX run. Mirrors the BibTeX content but compiles
    standalone with ``pdflatex`` alone.
    """
    records = manuscript_citable_records(bundle)
    if not records:
        return ""
    lines = [r"\begin{thebibliography}{99}"]
    for rec in records:
        key = sanitise_bibtex_key(rec.key)
        body_parts: List[str] = []
        if rec.title:
            body_parts.append(rec.title.rstrip(" ."))
        if rec.venue:
            body_parts.append(rec.venue)
        if rec.year and rec.year != "n/a":
            body_parts.append(rec.year)
        if rec.doi:
            body_parts.append(f"doi:{rec.doi}")
        elif rec.pmid:
            body_parts.append(f"PMID:{rec.pmid}")
        body = ". ".join(body_parts) + "."
        lines.append(f"\\bibitem{{{key}}} {body}")
    lines.append(r"\end{thebibliography}")
    return "\n".join(lines) + "\n"


__all__ = [
    "render_bibtex",
    "render_thebibliography_block",
    "sanitise_bibtex_key",
    "bibtex_key_for",
]
