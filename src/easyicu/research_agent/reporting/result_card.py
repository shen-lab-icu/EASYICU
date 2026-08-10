"""Evidence-bound result interpretation card for host projections.

The card never derives a new scientific number or causal explanation.  It
organises already-produced Research Agent claims, gate receipts, limitations,
and artifact references for human interpretation in the Web Copilot.
"""

from __future__ import annotations

import re
from typing import Any, List, Literal, Mapping, Optional

from pydantic import BaseModel, ConfigDict, Field


def _text(value: Any, limit: int = 1200) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()[:limit]


def _evidence_ids(value: Any) -> List[str]:
    values = value if isinstance(value, list) else [value] if value else []
    return [text for text in (_text(item, 160) for item in values[:40]) if text]


class InterpretationClaim(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    text: str
    evidence_ids: List[str] = Field(default_factory=list, max_length=40)
    status: str = "evidence_bound_draft"


class ResultInterpretationCard(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["easyicu.result-interpretation-card/1"] = (
        "easyicu.result-interpretation-card/1"
    )
    run_id: str
    status: Literal["blocked", "analysis_only", "review_ready"]
    claim_ceiling: Literal["unsupported", "analysis_only", "reportable"]
    gate_status: str
    readiness_status: str
    summary: str
    claims: List[InterpretationClaim] = Field(default_factory=list, max_length=40)
    limitations: List[str] = Field(default_factory=list, max_length=40)
    artifact_names: List[str] = Field(default_factory=list, max_length=80)
    human_review_required: bool = True
    generated_numbers: Literal[False] = False
    source: Literal["research_agent_artifacts_only"] = "research_agent_artifacts_only"


def _claim_rows(manuscript: Mapping[str, Any]) -> List[InterpretationClaim]:
    rows: List[InterpretationClaim] = []
    for raw in list(manuscript.get("claims") or [])[:40]:
        if isinstance(raw, str):
            claim_text = _text(raw)
            evidence_ids: List[str] = []
            status = "evidence_bound_draft"
        elif isinstance(raw, Mapping):
            claim_text = _text(
                raw.get("text")
                or raw.get("claim")
                or raw.get("sentence")
                or raw.get("statement")
            )
            evidence_ids = _evidence_ids(
                raw.get("evidence_ids") or raw.get("evidence_id")
            )
            status = _text(raw.get("status") or "evidence_bound_draft", 120)
        else:
            continue
        if claim_text:
            rows.append(
                InterpretationClaim(
                    text=claim_text,
                    evidence_ids=evidence_ids,
                    status=status or "evidence_bound_draft",
                )
            )
    if rows:
        return rows
    for raw in list(manuscript.get("sentences") or [])[:40]:
        if isinstance(raw, str) and _text(raw):
            rows.append(InterpretationClaim(text=_text(raw)))
        elif isinstance(raw, Mapping):
            sentence = _text(raw.get("text") or raw.get("sentence"))
            if sentence:
                rows.append(
                    InterpretationClaim(
                        text=sentence,
                        evidence_ids=_evidence_ids(raw.get("evidence_ids")),
                        status=_text(raw.get("status") or "evidence_bound_draft", 120),
                    )
                )
    return rows


def build_result_interpretation_card(
    *,
    run_id: Any,
    review: Mapping[str, Any],
    manuscript: Optional[Mapping[str, Any]],
) -> ResultInterpretationCard:
    """Build a non-generative interpretation surface from governed artifacts."""

    gate = review.get("gate") if isinstance(review.get("gate"), Mapping) else {}
    readiness = (
        review.get("readiness") if isinstance(review.get("readiness"), Mapping) else {}
    )
    gate_status = _text(gate.get("status") or "unknown", 120)
    readiness_status = _text(readiness.get("status") or "unknown", 120)
    reportable = bool(readiness.get("reportable"))
    analysis_only = gate_status == "analysis_only" and not reportable
    claim_ceiling: Literal["unsupported", "analysis_only", "reportable"] = (
        "reportable"
        if reportable
        else "analysis_only"
        if analysis_only
        else "unsupported"
    )
    status: Literal["blocked", "analysis_only", "review_ready"] = (
        "review_ready"
        if reportable
        else "analysis_only"
        if analysis_only
        else "blocked"
    )
    checks = gate.get("checks") if isinstance(gate.get("checks"), list) else []
    limitations: List[str] = []
    for check in checks[:80]:
        if not isinstance(check, Mapping) or check.get("passed") is not False:
            continue
        message = _text(
            check.get("reason")
            or check.get("message")
            or check.get("name")
            or check.get("check")
        )
        if message and message not in limitations:
            limitations.append(message)
    gate_reason = _text(gate.get("reason"))
    if gate_reason and gate_reason not in limitations:
        limitations.insert(0, gate_reason)
    artifact_names = sorted(
        {
            _text(row.get("name"), 160)
            for row in list(review.get("artifacts") or [])[:80]
            if isinstance(row, Mapping) and _text(row.get("name"), 160)
        }
    )
    summary = (
        "Research Agent evidence and manuscript claims are ready for human interpretation review."
        if status == "review_ready"
        else "Research Agent outputs are analysis-only and require human review before reporting."
        if status == "analysis_only"
        else "The current EasyICU gate does not support scientific result interpretation."
    )
    return ResultInterpretationCard(
        run_id=_text(run_id, 160) or "unknown_run",
        status=status,
        claim_ceiling=claim_ceiling,
        gate_status=gate_status,
        readiness_status=readiness_status,
        summary=summary,
        claims=_claim_rows(manuscript or {}),
        limitations=limitations[:40],
        artifact_names=artifact_names,
    )


__all__ = [
    "InterpretationClaim",
    "ResultInterpretationCard",
    "build_result_interpretation_card",
]
