"""Read explicit uncertainty requirements from the study and its amendment.

This is a requirement detector, not a method selector or source authority.
The host-bound amendment is parsed as data; historical plans in that contract
are never treated as words newly requested by the researcher.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import re
import unicodedata
from typing import Any, Mapping


_CONTRACT_HEADER = re.compile(r"(?m)^Host-bound request to revise\b")
_MARKER = re.compile(r'\n\s*(\{\s*"schema_version"\s*:\s*"easyicu\.plan-change-request/1")')
_INTERVAL = re.compile(
    r"置信区间|信赖区间|confidence\s+intervals?|"
    r"(?<![a-z])(?:\d{1,2}(?:\.\d+)?\s*%\s*)ci(?:s)?\b",
    re.IGNORECASE,
)
_NEGATION = re.compile(
    r"(?:不(?:再)?(?:需要|要求|提供|计算|报告|包含|展示|给出)|无需|不要|无须|不做)\s*$|"
    r"(?:\bno|\bwithout|\b(?:do\s+not|don't)\s+(?:need|report|calculate|provide|include)|"
    r"\bdoes\s+not\s+(?:need|require)|\bnot\s+requiring|\bneed\s+not\s+report)"
    r"\s*(?:an?\s+|the\s+)?$",
    re.IGNORECASE,
)
_PERCENT_BEFORE = re.compile(r"(?<![\d.])(\d{1,2}(?:\.\d+)?)\s*%\s*$")
_JOINED_PERCENT_BEFORE = re.compile(
    r"(?<![\d.])(\d{1,2}(?:\.\d+)?)\s*%\s*(?:、|和|及|与|and)\s*"
    r"(\d{1,2}(?:\.\d+)?)\s*%\s*$", re.IGNORECASE,
)
_PERCENT_JOINER_BEFORE = re.compile(
    r"(?<![\d.])(\d{1,2}(?:\.\d+)?)\s*%\s*(?:、|和|及|与|and)\s*$",
    re.IGNORECASE,
)
_ADDITIVE_LEVEL = re.compile(
    r"另外|额外|同时|也|新增|(?<!不)再|\b(?:add|also|additionally|in\s+addition)\b",
    re.IGNORECASE,
)
_PERCENT_AFTER = re.compile(
    r"^\s*(?:(?:of|at|to|改为|设为|为)\s*)?[(（]?\s*(\d{1,2}(?:\.\d+)?)\s*%",
    re.IGNORECASE,
)
_NEGATIVE_SUFFIX = re.compile(
    r"^\s*(?:不(?:需要|要求|计算|报告|提供)|无需|不是必须|"
    r"(?:is|are)\s+not\s+(?:needed|required|requested)|not\s+(?:needed|required))",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class UncertaintyRequirement:
    requested: bool
    confidence_levels: tuple[float, ...]


def amendment_user_messages(contract: str) -> tuple[str, ...]:
    """Extract only path-free user messages from the exact host-bound envelope.

    A malformed candidate is refused, never replaced by planner-authored text.
    Other contract forms, including plan repair prompts, carry no amendment.
    """
    if not contract:
        return ()
    decoder = json.JSONDecoder()
    messages: list[str] = []
    for header in _CONTRACT_HEADER.finditer(contract):
        match = _MARKER.search(contract, header.end(), min(len(contract), header.end() + 2_000))
        if match is None:
            raise ValueError("host_bound_plan_revision_contract_unreadable")
        try:
            data, _ = decoder.raw_decode(contract[match.start(1):])
        except json.JSONDecodeError as exc:
            raise ValueError("host_bound_plan_revision_contract_unreadable") from exc
        if not isinstance(data, dict) or data.get("schema_version") != "easyicu.plan-change-request/1":
            raise ValueError("host_bound_plan_revision_contract_unreadable")
        message = data.get("user_message")
        if not isinstance(message, str) or not 0 < len(message) <= 12_000:
            raise ValueError("host_bound_plan_revision_contract_unreadable")
        messages.append(message)
    return tuple(messages)


def requested_uncertainty(*texts: str) -> UncertaintyRequirement:
    """Find explicit interval requests, leaving generic uncertainty unspecified.

    A negation is checked directly before each mention, rather than suppressing
    unrelated positive requests in the same sentence. Requested confidence
    levels are captured only adjacent to the confidence-interval wording.
    """
    requested = False
    levels: set[float] = set()
    for text in texts:
        normalized = unicodedata.normalize("NFKC", str(text or ""))
        # Separate user messages can amend a previously stated level, while
        # two levels named in the same message are both requirements.
        stated_levels: set[float] = set()
        saw_level = False
        for match in _INTERVAL.finditer(normalized):
            before = normalized[max(0, match.start() - 35):match.start()]
            after = normalized[match.end():match.end() + 45]
            # Ignore an adjacent level when testing the negation itself.
            joined = _JOINED_PERCENT_BEFORE.search(before)
            embedded = re.match(r"(\d{1,2}(?:\.\d+)?)\s*%", match.group())
            prefix = _PERCENT_JOINER_BEFORE.search(before) if embedded else None
            if joined or prefix:
                negation_prefix = before[:(joined or prefix).start()]
            else:
                negation_prefix = re.sub(r"\d{1,2}(?:\.\d+)?\s*%\s*$", "", before)
            value = embedded or _PERCENT_BEFORE.search(before) or _PERCENT_AFTER.search(after)
            if joined:
                values = joined.groups()
            elif prefix and value:
                values = (prefix.group(1), value.group(1))
            else:
                values = (value.group(1),) if value else ()
            if _NEGATION.search(negation_prefix) or _NEGATIVE_SUFFIX.match(after):
                # Removing one explicitly named level does not withdraw a
                # different level the researcher also requested.
                if values and levels:
                    levels.difference_update(float(value) / 100.0 for value in values)
                    requested = bool(levels)
                    stated_levels = set(levels)
                else:
                    requested = False
                    levels.clear()
                    stated_levels.clear()
                saw_level = False
                continue
            requested = True
            for raw_level in values:
                level = float(raw_level) / 100.0
                if 0 < level < 1:
                    if not saw_level:
                        if _ADDITIVE_LEVEL.search(before):
                            stated_levels.update(levels)
                        else:
                            levels.clear()
                            stated_levels.clear()
                        saw_level = True
                    stated_levels.add(level)
                    levels = set(stated_levels)
    return UncertaintyRequirement(requested=requested, confidence_levels=tuple(sorted(levels)))


def primary_distribution_gaps(
    plan: Mapping[str, Any], requirement: UncertaintyRequirement,
) -> tuple[tuple[str, ...], tuple[str, ...]] | None:
    """Compare a requested CI against the exact primary distribution contract.

    ``None`` means no applicable interval requirement. An empty first tuple
    means no typed distribution exists, which cannot establish CI coverage.
    This makes no finding on method appropriateness or source authority.
    """
    from .analysis_types import canonical_analysis_family

    if not requirement.requested or canonical_analysis_family(plan.get("analysis_type")) != "descriptive_epidemiology":
        return None
    steps = [s for s in plan.get("steps", ())
             if isinstance(s, Mapping) and isinstance(s.get("exposure_outcome_distribution_spec"), Mapping)]
    primary = [s for s in steps if s.get("planned_analysis_role") == "primary"]
    steps = primary or steps
    missing = []
    for step in steps:
        spec = step["exposure_outcome_distribution_spec"]
        level = spec.get("confidence_level")
        if (spec.get("interval_method") in {None, "none_counts_only"}
                or isinstance(level, bool) or not isinstance(level, (int, float))
                or not 0 < level < 1
                or any(abs(float(level) - requested) >= 1e-9 for requested in requirement.confidence_levels)):
            missing.append(str(step.get("step_id") or ""))
    return tuple(str(step.get("step_id") or "") for step in steps), tuple(missing)
