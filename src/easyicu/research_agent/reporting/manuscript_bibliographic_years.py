"""Recognize explicit publication years against the citable run-bound bundle.

This does not infer years from citation keys or exempt year-shaped results.
Callers retain responsibility for admitting the bundle from verified inputs.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..literature import LiteratureBundle


_NAMED_DEFINITION = re.compile(
    r"\bthe[ \t]+(?:source-bound[ \t]+)?(?P<year>(?:19|20)\d{2})[ \t]+"
    r"(?P<name>[A-Za-z][A-Za-z0-9-]*(?:[ \t]+[A-Za-z][A-Za-z0-9-]*){0,5})[ \t]+"
    r"(?:definition|guidelines?|framework|criteria|consensus statement)"
    r"[ \t]*[.,;:]?[ \t]*"
    r"(?P<citation>\[@[A-Za-z0-9_.:-]+(?:;[ \t]*@[A-Za-z0-9_.:-]+)*\])",
    re.I,
)


def _words(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.casefold()))


def bibliographic_year_spans(
    manuscript: str, literature: LiteratureBundle | None,
) -> list[tuple[int, int]]:
    """Admit only locally cited named-definition publication metadata.

    A title match, exact metadata year, unique key and persistent source ID
    are all required. The citation cannot be borrowed from another sentence
    or paragraph. This validates the year, not the clinical claim itself.
    """
    if literature is None:
        return []
    from ..literature import manuscript_citable_records

    records = manuscript_citable_records(literature)
    spans = []
    for match in _NAMED_DEFINITION.finditer(manuscript):
        keys = set(re.findall(r"@([A-Za-z0-9_.:-]+)", match["citation"]))
        for key in keys:
            matches = [record for record in records if record.key == key]
            if len(matches) != 1:
                continue
            record = matches[0]
            if (
                record.year.strip() == match["year"]
                and (record.doi or record.pmid)
                and re.search(r"\b" + re.escape(_words(match["name"])) + r"\b", _words(record.title))
            ):
                spans.append(match.span("year"))
                break
    return spans
