"""Bounded outbound observation context for replanning calls."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Sequence

from ..research_context.outbound import project_outbound_records

REPLANNER_STEP_SUMMARY_CHAR_BUDGET = 3000
REPLANNER_TOTAL_RECORDS_CHAR_BUDGET = 24000
REPLANNER_PROBE_CHAR_BUDGET = 6000
_FINDING_KEYS = ("validator", "severity", "message")
_MAX_FINDINGS_PER_LIST = 8
_FINDING_MESSAGE_CHARS = 240
_RECORD_KEEP_KEYS = (
    "step_id",
    "intent",
    "status",
    "semantics_family",
    "returncode",
    "timed_out",
    "deterministic_code_fallback",
    "concept_audit_error_count",
    "concept_repair_attempts",
    "code_repair_attempts",
    "isolation_degraded",
    "dependency_step_id",
)
_RECORD_FINDING_KEYS = ("usage_findings", "visual_findings")


def clip_json(value: Any, *, char_budget: int) -> str:
    """Serialize JSON and clip its outbound prompt projection deterministically."""

    text = json.dumps(value, ensure_ascii=False, default=str)
    if len(text) <= char_budget:
        return text
    head = text[: max(0, char_budget)]
    return f"{head}…[truncated {len(text) - len(head)} chars for replanner context budget]"


def _compact_findings(raw: Any) -> List[Dict[str, Any]]:
    if not isinstance(raw, list):
        return []
    out: List[Dict[str, Any]] = []
    for item in raw[:_MAX_FINDINGS_PER_LIST]:
        if not isinstance(item, dict):
            continue
        compact: Dict[str, Any] = {}
        for key in _FINDING_KEYS:
            if key not in item:
                continue
            value = item[key]
            if key == "message" and isinstance(value, str):
                value = value[:_FINDING_MESSAGE_CHARS]
            compact[key] = value
        if compact:
            out.append(compact)
    return out


def slim_record_for_replanner(record: Dict[str, Any]) -> Dict[str, Any]:
    """Project one completed-step record to the fields used for replanning."""

    slim = {key: record[key] for key in _RECORD_KEEP_KEYS if key in record}
    summary = record.get("step_summary")
    if summary is not None:
        summary_text = json.dumps(summary, ensure_ascii=False, default=str)
        slim["step_summary"] = (
            clip_json(summary, char_budget=REPLANNER_STEP_SUMMARY_CHAR_BUDGET)
            if len(summary_text) > REPLANNER_STEP_SUMMARY_CHAR_BUDGET
            else summary
        )
    for key in _RECORD_FINDING_KEYS:
        compact = _compact_findings(record.get(key))
        if compact:
            slim[key] = compact
    return slim


def slim_completed_records_for_prompt(
    records: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Apply the outbound projector and collapse oldest records to the cap."""

    slimmed = project_outbound_records(records)
    if len(json.dumps(slimmed, ensure_ascii=False, default=str)) <= (
        REPLANNER_TOTAL_RECORDS_CHAR_BUDGET
    ):
        return slimmed
    for index, record in enumerate(slimmed):
        if len(json.dumps(slimmed, ensure_ascii=False, default=str)) <= (
            REPLANNER_TOTAL_RECORDS_CHAR_BUDGET
        ):
            break
        slimmed[index] = {
            "step_id": record.get("step_id"),
            "status": record.get("status"),
            "collapsed": "older step elided to fit replanner context budget",
        }
    return slimmed


__all__ = [
    "REPLANNER_PROBE_CHAR_BUDGET",
    "clip_json",
    "slim_completed_records_for_prompt",
    "slim_record_for_replanner",
]
