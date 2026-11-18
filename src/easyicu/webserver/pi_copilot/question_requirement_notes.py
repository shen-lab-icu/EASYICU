"""The question-requirement records as the plan review card reads them.

The planning owner keeps two records of what the question asks of a plan
(``planning.question_requirements``): the planning record, judged on the plan
the Planner compiled, and the judgment of the plan a review request offers,
which is what an approval rests on.  The card reads the judgment of the plan
under review.  Only when there is none, or it judged another plan, does the
card read the planning record, and then it says the judgments may not hold
for the plan.  It lists only what a reviewer must look at, each with its
source: a requirement no step answers or the plan cannot carry out, a claim
the plan made that the host could not verify, and, on a route that lists no
requirements, each concept the question names.  Requirements the host verified
as answered are counted, so the list reads as complete.  A record that exists
but cannot be read is reported as unavailable, never dropped.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from easyicu.research_agent.planning.question_requirements import (
    MAX_REQUIREMENT_CONCEPTS,
    QUESTION_REQUIREMENTS_FILENAME,
    QUESTION_REQUIREMENTS_REVIEW_FILENAME,
)

UNREADABLE_REASON = "question_requirements_record_unreadable"
_QUOTE_LIMIT = 240
_NOTE_LIMIT = 300
_SHOWN_DISPOSITIONS = ("not_covered", "capability_gap", "attested", "definition_only")
_RECORD_FILES = (QUESTION_REQUIREMENTS_REVIEW_FILENAME, QUESTION_REQUIREMENTS_FILENAME)


def _text(value: Any, limit: int) -> str:
    return " ".join(str(value or "").split())[:limit]


def _source(row: Mapping[str, Any]) -> str:
    """Who stands behind an item: the host's check, or the plan's own claim.

    The planning owner says which (``JudgedRequirement.verified_by_host``); a
    row that does not say so is the plan's claim.
    """

    return "host_verified" if row.get("verified_by_host") is True else "plan_declared"


def _item(row: Mapping[str, Any]) -> Dict[str, Any]:
    item: Dict[str, Any] = {
        "id": _text(row.get("id"), 8),
        "kind": _text(row.get("kind"), 40),
        "quote": _text(row.get("quote"), _QUOTE_LIMIT),
        "disposition": str(row.get("disposition")),
        "source": _source(row),
    }
    note = _text(row.get("note"), _NOTE_LIMIT)
    if note:
        item["note"] = note
    gap = row.get("gap")
    if isinstance(gap, Mapping):
        item["gap_requirement"] = _text(gap.get("requirement"), 80)
        item["gap_verification"] = _text(row.get("gap_verification"), 40)
    choice = row.get("exposure_choice")
    if isinstance(choice, Mapping):
        item["exposure_choice"] = {
            "primary_exposure": _text(choice.get("primary_exposure"), 128),
            "additional_exposures": [
                _text(value, 128)
                for value in list(choice.get("additional_exposures") or ())[:MAX_REQUIREMENT_CONCEPTS - 1]
            ],
        }
    return item


def _record_to_read(
    records: Mapping[str, Any],
) -> Tuple[Optional[str], Optional[Mapping[str, Any]], bool]:
    """The record the card reads, and whether it judged the plan under review."""

    def readable(name: str) -> Optional[Mapping[str, Any]]:
        record = records.get(name)
        return record if isinstance(record, Mapping) and record else None

    review = readable(QUESTION_REQUIREMENTS_REVIEW_FILENAME)
    planning = readable(QUESTION_REQUIREMENTS_FILENAME)
    if review is not None and review.get("judged_on_plan_under_review") is True:
        return QUESTION_REQUIREMENTS_REVIEW_FILENAME, review, True
    if planning is not None:
        return QUESTION_REQUIREMENTS_FILENAME, planning, False
    if review is not None:
        return QUESTION_REQUIREMENTS_REVIEW_FILENAME, review, False
    return None, None, False


def project_question_requirement_notes(
    records: Optional[Mapping[str, Any]], *, recorded: Iterable[str] = ()
) -> Optional[Dict[str, Any]]:
    """The card's view of one plan's question-requirement records.

    ``records`` maps each record file to its content, as the run projects them
    (``agent_pipeline_runs._load_question_requirements``); ``recorded`` names
    the record files the run wrote, so a record that cannot be read is
    reported, not hidden.  ``None`` when the run wrote neither.
    """

    name, record, judged_on_plan_under_review = _record_to_read(
        records if isinstance(records, Mapping) else {}
    )
    if record is None:
        written = recorded if isinstance(recorded, (list, tuple, set, frozenset)) else ()
        if not {str(value) for value in written} & set(_RECORD_FILES):
            return None
        return {
            "status": "unavailable",
            "reason_code": UNREADABLE_REASON,
            "items": [],
            "covered_count": 0,
            "unstated": [],
        }
    judged = [row for row in record.get("judged") or () if isinstance(row, Mapping)]
    items: List[Dict[str, Any]] = []
    for row in judged:
        if row.get("disposition") not in _SHOWN_DISPOSITIONS:
            continue
        clauses = row.get("clauses")
        if row.get("disposition") == "not_covered" and isinstance(clauses, list) and clauses:
            items.extend(
                _item({**row, "quote": clause.get("quote")}) for clause in clauses
                if isinstance(clause, Mapping) and clause.get("disposition") == "not_covered"
            )
        else:
            items.append(_item(row))
    # The approval stops first, in the record's order, then the plan's claims.
    items.sort(key=lambda item: _SHOWN_DISPOSITIONS.index(item["disposition"]))
    unstated = [
        {
            "concepts": [
                _text(value, 128) for value in list(row.get("concepts") or ())[:4]
            ],
            "evidence": _text(row.get("evidence"), _QUOTE_LIMIT),
            "read": bool(row.get("reading_step_ids"))
            or bool(row.get("cohort_criterion")),
        }
        for row in record.get("unstated") or ()
        if isinstance(row, Mapping)
    ]
    return {
        "status": "shown",
        "reason_code": None,
        "record": name,
        "judged_on_plan_under_review": judged_on_plan_under_review,
        "route": _text(record.get("route"), 40),
        "items": items[:6],
        "covered_count": sum(
            1 for row in judged if row.get("disposition") == "covered"
        ),
        "unstated": unstated[:8],
    }


def exposure_arrangement_offer(workflow_code: str, review: Mapping[str, Any]) -> dict[str, Any] | None:
    """The current plan's unfulfilled exposure decision, with an explicit default."""
    if workflow_code != "question_requirement_not_covered":
        return None
    notes = review.get("question_requirements") or {}
    if (
        not review.get("run_id")
        or notes.get("judged_on_plan_under_review") is not True
        or notes.get("unstated")
    ):
        return None
    uncovered = [item for item in notes.get("items") or () if item.get("disposition") == "not_covered"]
    if len(uncovered) != 1:
        return None
    (item,) = uncovered
    choice = item.get("exposure_choice")
    if item.get("source") != "host_verified" or not isinstance(choice, Mapping):
        return None
    primary = choice.get("primary_exposure")
    others = choice.get("additional_exposures")
    if not isinstance(primary, str) or not isinstance(others, list) or not all(
        isinstance(concept, str) for concept in others
    ):
        return None
    if not others or len(set([primary, *others])) != len(others) + 1 or any(
        not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,127}", concept) for concept in [primary, *others]
    ):
        return None
    additional = ", ".join(others)
    arrangements = {
        "primary_and_secondary": (
            f"Keep {primary} as the primary exposure and analyse {additional} separately as secondary exposures.",
            f"保留 {primary} 为主要暴露，分别对 {additional} 进行次要暴露分析。",
        ),
        "analyse_both": (
            f"Analyse {primary} and {additional} as parallel exposure analyses and report separate results.",
            f"对 {primary} 和 {additional} 分别进行平行的暴露分析并报告各自结果。",
        ),
        "only_primary": (
            f"Change my exposure scope to analyse {primary} only. Drop the exposure analyses of {additional}.",
            f"将我的暴露分析范围缩小到 {primary}，将 {additional} 移出本研究的暴露范围。",
        ),
    }
    return {
        "primary": primary,
        "additional": others,
        "default_option": "analyse_both" if re.search(
            r"co[- ]?primary|equally primary|共同主要|同等主要", str(item.get("quote") or ""), re.I
        ) else "primary_and_secondary",
        "options": {
            key: {
                "en": f"I approve this exposure arrangement. Please revise the research plan for review. {en}",
                "zh": f"我确认这项暴露安排。{zh} 请修订研究计划供我审阅。",
            }
            for key, (en, zh) in arrangements.items()
        },
    }


__all__ = ["UNREADABLE_REASON", "exposure_arrangement_offer", "project_question_requirement_notes"]
