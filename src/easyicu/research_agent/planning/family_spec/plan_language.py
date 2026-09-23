"""Language of the researcher-facing plan recommendation (``reviewable_plan``).

The Progressive Planner writes the selected design's six recommendation items
"in the research question's language".  The host family templates follow the
same contract with a deterministic test instead of a model judgement, so a
question asked in Chinese is answered with a Chinese recommendation in the
conversation that reviews it.

Only those six items follow the question.  Estimands, step objectives and the
Planner's reader labels keep their own owners' language: they feed the
executors, the reviewer and the manuscript, which have language contracts of
their own.
"""

from __future__ import annotations

import re
from typing import Literal, Sequence

PlanLanguage = Literal["en", "zh"]

# CJK Unified Ideographs, Extension A and the compatibility block.
_HAN = re.compile(r"[㐀-䶿一-鿿豈-﫿]")


def plan_language(research_question: str) -> PlanLanguage:
    """``zh`` when the question is written with Han characters, else ``en``."""

    return "zh" if _HAN.search(str(research_question or "")) else "en"


def listing(items: Sequence[str], language: PlanLanguage) -> str:
    """Join reader labels with the list separator of the plan language."""

    return ("、" if language == "zh" else ", ").join(str(item) for item in items)


def sentence(text: str) -> str:
    """Capitalize an English item that may begin with a joined phrase.

    The six items carry no label prefix: the design contract fixes their order
    (``REVIEWABLE_PLAN_ITEM_ORDER``) and every reader shows its own label per
    position, so a prefix would only repeat it in one language.
    """

    return text[:1].upper() + text[1:]


__all__ = ["PlanLanguage", "listing", "plan_language", "sentence"]
