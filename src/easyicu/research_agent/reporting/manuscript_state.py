"""Whether a run produced a manuscript, as a value rather than a sentence.

The write phase signalled "no manuscript" to the readiness gate by writing an
English paragraph into markdown, and readiness recovered the signal by looking
for ``"Manuscript scaffold not generated"`` in ``manuscript_text[:300]`` and by
matching phrases like ``writer failed``. Ten sites wrote the sentence, six read
it back, and the manifest-comment regex was defined verbatim in two modules.

The write phase already knows the answer precisely — it holds a
``CritiqueReport(status="blocked")``, or it paused deliberately after the
analysis phase — and threw that away at the seam, leaving readiness to
re-derive a coarser version of it from prose. Prose is the wrong carrier: it
is written for a human reader, it gets reworded, it is translated, and a
reword silently changes what a gate concludes.

So the state travels as a value. When it has to survive a process boundary it
is written into the document as a machine-readable marker comment beside the
human paragraph, and read back from there. The paragraph stays, unchanged in
purpose: it is what the author sees. It is no longer what the gate parses.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal, Optional

#: The heading every "no manuscript" document starts with. Author-facing.
NOT_GENERATED_HEADING = "# Manuscript scaffold not generated"

#: How far into a document the heading must appear to count. Historically the
#: readers used 300 or 600 characters interchangeably; one number now.
NOT_GENERATED_HEAD_CHARS = 600

#: Unresolved manifest annotations left inline by the binder. Defined once:
#: ``readiness`` and ``write_phase`` each carried a verbatim copy.
MANIFEST_COMMENT_RE = re.compile(
    r"<!--\s*(?P<level>warning|error)\s*:\s*see manifest\s*-->",
    flags=re.IGNORECASE,
)

_STATE_MARKER_RE = re.compile(
    r"<!--\s*easyicu:manuscript-state\s+kind=(?P<kind>[a-z_]+)"
    r"(?:\s+reason_code=(?P<reason>[A-Za-z0-9_.:-]+))?\s*-->"
)

ManuscriptStateKind = Literal["produced", "blocked", "paused", "probe"]

__all__ = [
    "MANIFEST_COMMENT_RE",
    "NOT_GENERATED_HEAD_CHARS",
    "NOT_GENERATED_HEADING",
    "ManuscriptState",
    "ManuscriptStateKind",
    "is_not_generated",
    "read_manuscript_state",
    "render_not_generated",
]


@dataclass(frozen=True)
class ManuscriptState:
    """Why this run does or does not have a manuscript.

    ``blocked`` is a refusal: a gate did not pass and drafting was not
    attempted, or the writer produced nothing bindable. ``paused`` is a
    deliberate stop the operator asked for — the run is not broken and can be
    resumed. Readiness treats both as "no manuscript", but an operator reading
    a run needs to know which one happened, and only ``blocked`` is a defect.
    """

    kind: ManuscriptStateKind
    reason_code: str = ""
    detail: str = ""

    def __post_init__(self) -> None:
        if self.kind not in ("produced", "blocked", "paused", "probe"):
            raise ValueError(
                "manuscript state must be produced, blocked, paused, or probe"
            )
        if self.kind in ("blocked", "paused") and not self.reason_code:
            raise ValueError(f"{self.kind} manuscript state requires a reason_code")
        if self.kind == "produced" and self.reason_code:
            raise ValueError("a produced manuscript has no reason_code")

    @property
    def generated(self) -> bool:
        return self.kind == "produced"

    @classmethod
    def produced(cls) -> "ManuscriptState":
        return cls(kind="produced")

    @classmethod
    def blocked(cls, reason_code: str, detail: str = "") -> "ManuscriptState":
        return cls(kind="blocked", reason_code=reason_code, detail=detail)

    @classmethod
    def paused(cls, reason_code: str, detail: str = "") -> "ManuscriptState":
        return cls(kind="paused", reason_code=reason_code, detail=detail)

    def marker(self) -> str:
        """The machine-readable half of the document."""
        reason = f" reason_code={self.reason_code}" if self.reason_code else ""
        return f"<!-- easyicu:manuscript-state kind={self.kind}{reason} -->"


def render_not_generated(state: ManuscriptState, prose: str) -> str:
    """Render the author-facing document for a run with no manuscript.

    ``prose`` is for the person reading the run and may be reworded freely.
    The marker beside it is what a gate reads, so rewording the paragraph can
    no longer change what the gate concludes.
    """
    if state.generated:
        raise ValueError("render_not_generated requires a blocked or paused state")
    body = str(prose).strip()
    return f"{NOT_GENERATED_HEADING}\n\n{state.marker()}\n\n{body}\n"


def read_manuscript_state(manuscript_text: str) -> Optional[ManuscriptState]:
    """Recover the state a document was written with, if it carries one.

    Returns ``None`` for a document with no "not generated" heading — that is
    a real draft, or a document this module did not write. A document written
    before the marker existed still reports ``blocked`` from its heading, with
    no reason code; that is exactly as much as it ever carried.
    """
    text = str(manuscript_text or "")
    head = text.strip()[:NOT_GENERATED_HEAD_CHARS]
    if NOT_GENERATED_HEADING not in head:
        return None
    match = _STATE_MARKER_RE.search(head)
    if match is None:
        return ManuscriptState(kind="blocked", reason_code="unspecified")
    kind = match.group("kind")
    reason = match.group("reason") or ""
    try:
        return ManuscriptState(
            kind=kind,  # type: ignore[arg-type]
            reason_code=reason or "unspecified",
        )
    except ValueError:
        # An unreadable marker is still an unmistakable "not generated"
        # heading; degrade to the heading's meaning rather than to a draft.
        return ManuscriptState(kind="blocked", reason_code="unspecified")


def is_not_generated(manuscript_text: str) -> bool:
    """True when this document says no manuscript was produced."""
    return read_manuscript_state(manuscript_text) is not None
