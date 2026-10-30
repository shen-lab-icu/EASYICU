"""Recognize a user's explicit request to revise the current candidate plan."""

from __future__ import annotations

import re
from typing import Any

from .turn_authority import infer_explicit_turn_actions

_REVISION_REQUEST = re.compile(
    r"(?:修订|修正|修改|调整|完善)[^。.!！?？;；,，]{0,24}(?:计划|方案)"
    r"|\b(?:revise|amend|modify|update)\s+(?:(?:the|this|current|complete|candidate|research|analysis)\s+)*plan\b",
    re.IGNORECASE,
)
_PLACEHOLDER_PREFIXES = (
    "Please revise the complete research plan using the following review comments, preserve the original question and data source, and explain the changes. Do not start analysis:",
    "请根据以下审阅意见修订整份研究计划，保留原始问题与数据来源，并说明修改依据。不要开始分析：",
)


def _has_amendment(message: str) -> bool:
    text = str(message or "").strip()
    for prefix in _PLACEHOLDER_PREFIXES:
        if text.lower().startswith(prefix.lower()):
            return bool(text[len(prefix):].strip(" \t\n:：,，.。!！"))
    # A bare "revise the plan" does not contain the amendment to bind.
    remainder = _REVISION_REQUEST.sub("", text)
    remainder = re.sub(
        r"(?:请|现在|谢谢|供审阅|仅供审阅|只供审阅|不执行分析|不提取数据|不要开始分析|"
        r"please|thank you|for review|do not (?:start|run) analysis)", "", remainder, flags=re.I,
    )
    return bool(re.search(r"[\w\u4e00-\u9fff]", remainder))


def requested_candidate_run(
    message: str, *, message_origin: str | None, projection: Any
) -> str:
    """Return only the authoritative, reviewable run for a typed amendment.

    This is not plan approval. The normal one-turn provider grant and replan
    owner remain responsible for disclosure, freshness and job admission.
    """
    if message_origin != "typed" or "provider_run" not in infer_explicit_turn_actions(message):
        return ""
    if not _REVISION_REQUEST.search(message) or not _has_amendment(message):
        return ""
    workflow = projection.workflow
    review = workflow.plan_review_summary or {}
    run_id = str(review.get("run_id") or "")
    latest = projection.latest_run or {}
    if not run_id or latest.get("run_id") != run_id or (projection.active_job or {}).get("present"):
        return ""
    candidates = [row for row in projection.runs if row.get("run_id") == run_id]
    if len(candidates) != 1:
        return ""
    row = candidates[0]
    if not row.get("authoritative") or row.get("run_status") != "human_review_pending":
        return ""
    if "agent_plan.json" not in (row.get("artifact_names") or []):
        return ""
    return run_id
