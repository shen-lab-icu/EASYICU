"""A kind of a concept the question names that the analysis data cannot tell apart.

Owner of one decision: whether a question names a level of a categorical
concept -- invasive or non-invasive mechanical ventilation, a ventilator mode
-- that no analysis quantity the host prepares distinguishes, and if so the
launch stops (:data:`NAMED_LEVEL_UNDISTINGUISHED`) before any model is called.

Why
---
The concept dictionary declares some categorical concepts' levels:
``mech_vent`` is ``invasive`` or ``noninvasive``; ``vent_mode`` and
``vent_breath_seq`` name ventilator settings.  For a concept recorded over
time, the host reads no level: the wide materializer summarizes a categorical
value as its recorded presence in a window (``cohort.materializer``), the
ventilation windows ``vent_ind`` are derived from every level, and the
question reader binds 「有创机械通气」 to ``vent_ind`` on the words 「机械通气」.  A
question about one level was therefore planned on any level, invasive or not,
and nothing said so.  Until an owner produces a level-specific quantity, the
honest answer is to stop and ask for the question the data can answer.

What is read
------------
:data:`_LEVEL_WORDS` is a closed host vocabulary: each entry names the
concept, the level the dictionary declares for it, and the words (English and
Chinese) that name that level.  A test holds every entry to the dictionary.
The words are read, not interpreted: a level named to exclude it, or beside
the other level, is still a level the data cannot separate, so the researcher
rewords the question (for example 「机械通气」, which every level answers).

A categorical concept the dictionary records once per stay (``target`` is
``id_tbl``: admission type, sex) keeps its own value as an analysis column, so
its levels are distinguished and nothing here stops them.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, Optional, Tuple

__all__ = [
    "NAMED_LEVEL_UNDISTINGUISHED",
    "NamedLevel",
    "named_levels",
    "stop_undistinguished_named_levels",
    "undistinguished_named_levels",
]

#: Why a launch stops: the question names a level no analysis quantity separates.
NAMED_LEVEL_UNDISTINGUISHED = "question_named_level_undistinguished"

#: (concept, declared level, the words that name it).  An entry for a concept's
#: narrower spelling comes first ("non-invasive ventilation" before "invasive
#: ventilation"), and a span one entry reads is not read again by another.  An
#: acronym is bounded by Latin letters and digits only, so it is read beside
#: Chinese text with no space (「SIMV模式」).
_LEVEL_WORDS: Tuple[Tuple[str, str, str], ...] = (
    (
        "mech_vent",
        "noninvasive",
        r"\bnon[\s-]?invasive(?:ly)?[\s-]+"
        r"(?:(?:mechanical(?:ly)?|positive[\s-]+pressure)[\s-]+)?ventilat\w*"
        r"|(?<![A-Za-z0-9])(?:NIV|NIPPV|Bi-?PAP)(?![A-Za-z0-9])"
        r"|无创(?:正压|机械|辅助)?(?:通气|呼吸机|呼吸支持)",
    ),
    (
        "mech_vent",
        "invasive",
        r"(?<![\w-])(?<!non\s)invasive(?:ly)?[\s-]+(?:mechanical(?:ly)?[\s-]+)?ventilat\w*"
        r"|有创(?:正压|机械)?(?:通气|呼吸机|呼吸支持)",
    ),
    (
        "vent_mode",
        "pressure",
        r"\bpressure[\s-]+control(?:led)?[\s-]+(?:mechanical[\s-]+)?(?:ventilation|mode)\b"
        r"|压力控制(?:通气|模式)",
    ),
    (
        "vent_mode",
        "volume",
        r"\bvolume[\s-]+control(?:led)?[\s-]+(?:mechanical[\s-]+)?(?:ventilation|mode)\b"
        r"|容量控制(?:通气|模式)",
    ),
    ("vent_breath_seq", "simv", r"(?<![A-Za-z0-9])SIMV(?![A-Za-z0-9])|同步间歇指令通气"),
)
_COMPILED = tuple(
    (concept, level, re.compile(words, re.IGNORECASE))
    for concept, level, words in _LEVEL_WORDS
)


@dataclass(frozen=True)
class NamedLevel:
    """One level of a categorical concept, in the words that name it."""

    concept: str
    level: str
    evidence: str

    def record(self) -> Dict[str, str]:
        return {"concept": self.concept, "level": self.level, "evidence": self.evidence}


def named_levels(question: str) -> Tuple[NamedLevel, ...]:
    """Each level of a categorical concept ``question`` names, in text order.

    One record per concept and level, with the first words that name it.
    """

    text = str(question or "")
    taken: list[Tuple[int, int]] = []
    found: Dict[Tuple[str, str], Tuple[int, NamedLevel]] = {}
    for concept, level, pattern in _COMPILED:
        for match in pattern.finditer(text):
            start, end = match.span()
            if any(start < stop and begin < end for begin, stop in taken):
                continue
            taken.append((start, end))
            key = (concept, level)
            if key not in found or start < found[key][0]:
                found[key] = (start, NamedLevel(concept, level, match.group(0)))
    return tuple(item for _start, item in sorted(found.values(), key=lambda pair: pair[0]))


@lru_cache(maxsize=None)
def _declaration(concept: str) -> Optional[Tuple[str, str, Tuple[str, ...]]]:
    """The dictionary's class, target and declared levels of ``concept``."""

    from easyicu.concept.loader import load_dictionary

    definition = load_dictionary().get(concept)
    if definition is None:
        return None
    levels = getattr(definition, "levels", None) or ()
    return (
        str(getattr(definition, "class_name", "") or ""),
        str(getattr(definition, "target", "") or ""),
        tuple(str(level) for level in levels),
    )


def _analysis_data_keep_levels(concept: str) -> bool:
    """Whether an analysis column holds the concept's own level.

    A categorical concept recorded once per stay is materialized as its value;
    one recorded over time is summarized as its presence in a window, which no
    level survives.  An undeclared concept is never vouched for.
    """

    declaration = _declaration(concept)
    if declaration is None:
        return False
    class_name, target, _levels = declaration
    return class_name == "fct_cncpt" and target == "id_tbl"


def undistinguished_named_levels(question: str) -> Tuple[NamedLevel, ...]:
    """The levels ``question`` names that no analysis quantity distinguishes."""

    return tuple(
        item for item in named_levels(question)
        if not _analysis_data_keep_levels(item.concept)
    )


def _message(found: Tuple[NamedLevel, ...]) -> str:
    named = "; ".join(
        f"{item.evidence!r} is the {item.level} kind of {item.concept} "
        f"({', '.join((_declaration(item.concept) or ('', '', ()))[2])})"
        for item in found
    )
    return (
        f"The question names one kind of a recorded concept: {named}. The analysis "
        "data record only whether it was present in a time window, not which kind, "
        "so a plan would answer a broader question. Planning stopped before any "
        "model was called, and no analysis was run. Ask the question without naming "
        "the kind (for example \"mechanical ventilation\"), whose answer covers every kind."
    )


def stop_undistinguished_named_levels(question: str) -> None:
    """Stop a launch whose question names a level the analysis data cannot separate."""

    from easyicu.webserver.research_pipeline_run_errors import ResearchPipelineRunError

    found = undistinguished_named_levels(question)
    if not found:
        return
    details: Dict[str, Any] = {
        "concept": found[0].concept,
        "named_levels": [item.record() for item in found],
    }
    raise ResearchPipelineRunError(
        "question_named_level_undistinguished",
        _message(found),
        details=details,
    )
