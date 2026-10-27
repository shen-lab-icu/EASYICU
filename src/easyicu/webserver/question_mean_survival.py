"""A mean survival time the question asks, which no study plan answers.

Owner of one decision: whether a question asks how long patients lived on
average -- a restricted mean survival time (RMST): 「28 天内平均存活多少天？与
未通气的患者相比，调整年龄和性别后差多少？」, "restricted mean survival time at
28 days" -- and if so the launch stops (:data:`MEAN_SURVIVAL_UNPLANNED`) before
any model is called.

Why
---
The words name a horizon (28 days), and the readers bind it to the 28-day
mortality endpoint (``mort_28d``, with its follow-up the time-to-event endpoint
censored at day 28).  But the estimand is the mean time alive up to the
horizon, a difference in days.  A plan on that endpoint reports the share who
died by day 28 (an odds ratio or risk difference) or a hazard ratio: another
answer, and nothing would say so.  The platform plans a restricted mean
survival contrast only as a prespecified sensitivity of a survival study
(``sensitivity_authority``'s ``restricted_mean_survival``), never as a study's
main result.  Until an owner plans it, the honest answer is to stop and ask
for an estimand a plan answers.

A mean survival the question asks only as a sensitivity analysis ("with RMST
as a sensitivity analysis", 「并以 28 天限制性平均生存时间差作为敏感性分析」,
"a sensitivity analysis using RMST") is not stopped: that is the sensitivity
a survival plan can state.  The words must bind to the mean survival itself.
"Estimate the 28-day RMST difference … and use a Cox model as a sensitivity
analysis" asks the RMST as the main result, and so does a clause that calls
it primary (「以 … 平均生存时间差为主要结局」).
"""

from __future__ import annotations

import re
from typing import Any, Dict, Tuple

from easyicu.outcome_availability import StatedMeanSurvival, stated_mean_survival_mentions

__all__ = [
    "MEAN_SURVIVAL_UNPLANNED",
    "asked_mean_survival",
    "stop_unplanned_mean_survival",
]

#: Why a launch stops: the question asks a mean survival time no plan answers.
MEAN_SURVIVAL_UNPLANNED = "question_mean_survival_unplanned"

# The clause a mean survival stands in.
_CLAUSE = re.compile(r"[^,，;；。？?！!\n]+")
# The clause calls a result its primary one.
_PRIMARY = re.compile(r"\bprimary\b|\bmain\b|主要|主分析|首要", re.IGNORECASE)
# The words that follow a mean survival and make it the sensitivity: its own
# contrast word at most (差, difference), then "as (a) sensitivity",
# 「作为敏感性分析」.
_AS_SENSITIVITY_AFTER = re.compile(
    r"^\s*(?:差值?|之差|的差异|的差别|(?:differences?|contrasts?)\b)?\s*"
    r"(?:\b(?:as|for)\s+(?:an?\s+|the\s+)?(?:\w+\s+)?sensitivity\b"
    r"|(?:作为|做|作|为)\s*敏感性(?:分析|检验)?|敏感性分析)",
    re.IGNORECASE,
)
# The words before a mean survival that make it the sensitivity: "a
# sensitivity analysis with/using/of (the) RMST", 「敏感性分析采用 RMST」.
_SENSITIVITY_BEFORE = re.compile(
    r"(?:\bsensitivity\s+analys\w*\s+(?:with|using|by|of|on)\s+(?:the\s+|a\s+|an\s+)?(?:\d+[\s-]*days?[\s-]+)?"
    r"|敏感性分析\s*(?:中|时)?\s*(?:采用|使用|用|报告|计算|估计|以)?\s*(?:\d+\s*天\s*)?)$",
    re.IGNORECASE,
)


def _asked_as_sensitivity(text: str, mention: StatedMeanSurvival) -> bool:
    """Whether the words that make a result a sensitivity analysis bind to ``mention``."""

    clause = next(
        (match.group(0) for match in _CLAUSE.finditer(text)
         if match.start() <= mention.start < match.end()),
        "",
    )
    if _PRIMARY.search(clause):
        return False
    return bool(
        _AS_SENSITIVITY_AFTER.match(text[mention.end:mention.end + 40])
        or _SENSITIVITY_BEFORE.search(text[max(0, mention.start - 60):mention.start])
    )


def asked_mean_survival(question: str) -> Tuple[StatedMeanSurvival, ...]:
    """Where ``question`` asks a mean survival time as its own result, in text order."""

    text = str(question or "")
    return tuple(
        mention for mention in stated_mean_survival_mentions(text)
        if not _asked_as_sensitivity(text, mention)
    )


def _message(found: Tuple[StatedMeanSurvival, ...]) -> str:
    named = ", ".join(repr(item.evidence) for item in found)
    return (
        f"The question asks how long patients lived on average, a restricted mean "
        f"survival time: {named}. A plan would answer it with the share who died by "
        "the horizon or with a hazard ratio, which is another estimand, and a "
        "restricted mean survival time is not yet planned as a study's main result. "
        "Planning stopped before any model was called, and no analysis was run. Ask "
        "about death by the horizon (for example 28-day mortality) or for a survival "
        "analysis (a hazard ratio)."
    )


def stop_unplanned_mean_survival(question: str) -> None:
    """Stop a launch whose question asks a mean survival time no plan answers."""

    from easyicu.webserver.research_pipeline_run_errors import ResearchPipelineRunError

    found = asked_mean_survival(question)
    if not found:
        return
    details: Dict[str, Any] = {
        "mean_survival": [
            {"evidence": item.evidence, "start": item.start, "end": item.end} for item in found
        ],
    }
    raise ResearchPipelineRunError(MEAN_SURVIVAL_UNPLANNED, _message(found), details=details)
