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


class InterpretationTable(BaseModel):
    """Small aggregate-only slice copied from a governed result table.

    ``entries`` deliberately remains a header/cell representation instead of
    turning the values into a new scientific summary.  The Research Agent owns
    the numbers; this card only makes a bounded subset visible to the
    conversational shell.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    label: str
    evidence_id: str
    columns: List[str] = Field(default_factory=list, max_length=16)
    entries: List[List[str]] = Field(default_factory=list, max_length=12)
    preview_truncated: bool = False


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
    claims: List[InterpretationClaim] = Field(default_factory=list, max_length=12)
    result_tables: List[InterpretationTable] = Field(
        default_factory=list, max_length=6
    )
    limitations: List[str] = Field(default_factory=list, max_length=40)
    artifact_names: List[str] = Field(default_factory=list, max_length=80)
    human_review_required: bool = True
    generated_numbers: Literal[False] = False
    source: Literal["research_agent_artifacts_only"] = "research_agent_artifacts_only"


def _claim_rows(manuscript: Mapping[str, Any]) -> List[InterpretationClaim]:
    rows: List[InterpretationClaim] = []
    for raw in list(manuscript.get("claims") or [])[:12]:
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
    for raw in list(manuscript.get("sentences") or [])[:12]:
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


_DISTRIBUTION_COLUMNS = frozenset(
    {
        "n_rows",
        "exposure_denominator",
        "exposure_pct",
        "outcome_events",
        "outcome_denominator",
        "outcome_rate_pct",
    }
)
_EFFECT_COLUMNS = frozenset({"estimate", "ci_low", "ci_high", "effect_scale"})
_ROBUSTNESS_COLUMNS = frozenset(
    {"axis", "total_specs", "converged_specs", "range_low", "range_high"}
)


def _interpretation_tables(
    payload: Optional[Mapping[str, Any]],
) -> List[InterpretationTable]:
    """Select typed, aggregate result surfaces without deriving new values."""

    if not isinstance(payload, Mapping):
        return []
    selected: List[InterpretationTable] = []
    for raw in list(payload.get("tables") or [])[:40]:
        if not isinstance(raw, Mapping):
            continue
        columns = [_text(item, 120) for item in list(raw.get("headers") or [])[:16]]
        column_set = set(columns)
        if not (
            _DISTRIBUTION_COLUMNS.issubset(column_set)
            or _EFFECT_COLUMNS.issubset(column_set)
            or _ROBUSTNESS_COLUMNS.issubset(column_set)
        ):
            continue
        entries: List[List[str]] = []
        for entry in list(raw.get("rows") or [])[:12]:
            if not isinstance(entry, (list, tuple)):
                continue
            entries.append(
                [_text(value, 160) for value in list(entry)[: len(columns)]]
            )
        if not columns or not entries:
            continue
        selected.append(
            InterpretationTable(
                name=_text(raw.get("name"), 160) or "aggregate_result",
                label=_text(raw.get("label"), 300) or "Aggregate result",
                evidence_id=_text(raw.get("evidence_id"), 160),
                columns=columns,
                entries=entries,
                preview_truncated=bool(raw.get("preview_truncated")),
            )
        )
        if len(selected) >= 6:
            break
    return selected


def _scientific_limitations(
    scientific_readiness: Optional[Mapping[str, Any]],
) -> List[str]:
    if not isinstance(scientific_readiness, Mapping):
        return []
    values: List[str] = []
    for finding in list(scientific_readiness.get("findings") or [])[:40]:
        if not isinstance(finding, Mapping):
            continue
        code = _text(finding.get("code"), 160)
        message = _text(finding.get("message"), 600)
        value = f"{code}: {message}" if code and message else message or code
        if value and value not in values:
            values.append(value)
    return values


def build_result_interpretation_card(
    *,
    run_id: Any,
    review: Mapping[str, Any],
    manuscript: Optional[Mapping[str, Any]],
    result_tables: Optional[Mapping[str, Any]] = None,
    scientific_readiness: Optional[Mapping[str, Any]] = None,
) -> ResultInterpretationCard:
    """Build a non-generative interpretation surface from governed artifacts."""

    gate = review.get("gate") if isinstance(review.get("gate"), Mapping) else {}
    readiness = (
        review.get("readiness") if isinstance(review.get("readiness"), Mapping) else {}
    )
    gate_status = _text(gate.get("status") or "unknown", 120)
    readiness_status = _text(readiness.get("status") or "unknown", 120)
    reportable = bool(readiness.get("reportable"))
    checks = gate.get("checks") if isinstance(gate.get("checks"), list) else []
    scientific_facts = (
        scientific_readiness.get("facts")
        if isinstance(scientific_readiness, Mapping)
        and isinstance(scientific_readiness.get("facts"), Mapping)
        else {}
    )
    analysis_facts = (
        scientific_facts.get("analysis")
        if isinstance(scientific_facts.get("analysis"), Mapping)
        else {}
    )
    interpretation_tables = _interpretation_tables(result_tables)
    validated_analysis_only = bool(
        isinstance(scientific_readiness, Mapping)
        and scientific_readiness.get("status") == "analysis_only"
        and scientific_readiness.get("claim_ceiling") == "analysis_only"
        and analysis_facts.get("analysis_validated") is True
        and interpretation_tables
    )
    analysis_only = bool(
        not reportable
        and (gate_status == "analysis_only" or validated_analysis_only)
    )
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
    for limitation in _scientific_limitations(scientific_readiness):
        if limitation not in limitations:
            limitations.append(limitation)
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
        result_tables=interpretation_tables,
        limitations=limitations[:40],
        artifact_names=artifact_names,
    )


__all__ = [
    "InterpretationClaim",
    "InterpretationTable",
    "ResultInterpretationCard",
    "build_result_interpretation_card",
]
