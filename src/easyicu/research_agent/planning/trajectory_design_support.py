"""Refuse an explicitly requested longitudinal family the trajectory owner lacks.

A longitudinal panel is not itself a request for a mixed-effects or joint
model. Only an affirmative method declaration reaches this refusal.
"""

import re

_PATTERN = re.compile(
    r"mixed[- ]effects?|mixed[- ]model|random[- ]slopes?|joint[- ](?:longitudinal[- ]survival[- ])?model|混合效应|随机斜率|联合(?:纵向生存)?模型",
    re.I,
)
_NEGATED = re.compile(
    r"(?:without|not (?:use|using)|do not (?:use|fit)|instead of|不使用|不用|无需|而非)\s*(?:an?\s+)?$",
    re.I,
)
ALTERNATIVES = "Supported alternatives are fixed-window trajectory phenotyping on a preset ICU time grid, or per-stay phenotyping of recorded summaries."


def unsupported_trajectory_design(
    question: str, preferred_methods: str = ""
) -> str | None:
    text = str(question or "") + " " + str(preferred_methods or "")
    if not re.search(r"trajector|轨迹", text, re.I):
        return None
    for match in _PATTERN.finditer(text):
        if not _NEGATED.search(text[max(0, match.start() - 30) : match.start()]):
            return f"The requested {match.group()} trajectory design has no signed execution owner. {ALTERNATIVES}"
    return None
