"""Typed failures for concept extraction.

Why this exists: a concept loader that cannot read its source has two very
different things it might mean, and a bare empty frame says both at once.

* "I read the source and the answer is genuinely nothing" — no patient in
  this cohort died, no dose was recorded, the window is empty. A legitimate
  result.
* "I could not read the source" — the file is missing, permissions changed, a
  column was renamed upstream, the query engine raised. Not a result at all.

Collapsing the second into the first is how a permission error becomes a
reported mortality of zero. For outcome, exposure and cohort concepts that is
not a degraded answer, it is a wrong one that looks entirely normal
downstream. These exceptions exist so the second case fails loudly instead.
"""

from __future__ import annotations

from typing import Optional


__all__ = [
    "ConceptError",
    "ConceptExtractionUnavailable",
    "ConceptTableReadError",
]


class ConceptError(RuntimeError):
    """Base class for concept-layer failures."""


class ConceptExtractionUnavailable(ConceptError):
    """A concept could not be determined, so no value may be reported for it.

    Raised instead of returning an empty result whenever the emptiness would
    be an artefact of the failure rather than a statement about the patients.
    """

    def __init__(
        self,
        *,
        concept_id: str,
        database: str,
        stage: str,
        detail: str,
        cause: Optional[BaseException] = None,
    ) -> None:
        self.concept_id = str(concept_id)
        self.database = str(database)
        self.stage = str(stage)
        self.detail = str(detail)
        super().__init__(
            f"concept {self.concept_id!r} on {self.database!r} could not be "
            f"determined at stage {self.stage!r}: {self.detail}. Refusing to "
            "return an empty result, which downstream code cannot tell apart "
            "from a genuine absence of events."
        )
        if cause is not None:
            self.__cause__ = cause


class ConceptTableReadError(ConceptError):
    """A source table could be neither read nor proven absent.

    Raised when a concept callback needs a mapping/auxiliary table (e.g.
    ``icustays``/``admissions``/``patients``) and the read fails with an error
    other than table absence (``FileNotFoundError``/``KeyError``). Carries the
    source identity (database + table + stage) so the failure cannot be
    mistaken for "no mapping needed".
    """

    def __init__(
        self,
        *,
        database: str,
        table: str,
        stage: str,
        detail: str,
        cause: Optional[BaseException] = None,
    ) -> None:
        self.database = str(database)
        self.table = str(table)
        self.stage = str(stage)
        self.detail = str(detail)
        super().__init__(
            f"table {self.table!r} on {self.database!r} could not be read "
            f"at stage {self.stage!r}: {self.detail}. Refusing to fall back "
            "to None, which downstream code cannot tell apart from a "
            "genuinely absent table."
        )
        if cause is not None:
            self.__cause__ = cause
