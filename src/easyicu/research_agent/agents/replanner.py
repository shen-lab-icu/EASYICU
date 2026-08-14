"""ReplannerAgent and its prompt context-budget guards."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Mapping, Optional, Sequence

from ..planning import scientific_action_catalog as _scientific_actions
from ..trajectory.plan_contract import (
    trajectory_planner_contract_guide,
)
from ..providers.protocol import LLMMessage
from ..research_context.prompt_scope import (
    planner_variable_catalog,
    scoped_planner_context,
)
from ..research_context.outbound import project_outbound_probe
from ..schema import (
    AnalysisPlan,
    ResearchContext,
)
from . import plan_payload as _payload

from ._support import _REPLANNER_GUIDE, _SYSTEM_GUIDE, _format_context
from .planner import PlannerAgent, PlannerPromptBudgetError, _PLANNER_PROMPT_BYTE_LIMIT, _PRINCIPLES_GUIDE

# ---------------------------------------------------------------------------
# Replanner prompt context-budget guards
# ---------------------------------------------------------------------------
#
# The replanner prompt embeds every completed step's record (incl. its full
# ``step_summary.json``) plus the probe summary. Both are written by step code
# and are NOT byte-capped at the source — a single step that dumps a wide
# interaction matrix or per-subgroup table into its summary (the exact failure
# class noted in CLAUDE.md: pilot-1's 295-leaf interaction dump) would otherwise
# inflate the prompt without bound, multiplied by up to ``max_total_steps`` (12).
# Small-context local engines (glm / qwen / deepseek, see ``providers/llm.py``) overflow
# well before a frontier model would.
#
# These guards slim ONLY the prompt projection. The full records keep flowing to
# the in-run validators (``_step_contract_findings`` et al.) and to disk/evidence
# untouched, so auditability and replay are unaffected.
_REPLANNER_STEP_SUMMARY_CHAR_BUDGET = 3000
_REPLANNER_TOTAL_RECORDS_CHAR_BUDGET = 24000
_REPLANNER_PROBE_CHAR_BUDGET = 6000
_REPLANNER_FINDING_KEYS = ("validator", "severity", "message")
_REPLANNER_MAX_FINDINGS_PER_LIST = 8
_REPLANNER_FINDING_MESSAGE_CHARS = 240
# Top-level record keys the replanner actually reasons over (status, intent,
# observed artefacts, validation findings). Inputs the replanner already has via
# CURRENT PLAN (e.g. analysis_request / visualization_request) are dropped.
_REPLANNER_RECORD_KEEP_KEYS = (
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
_REPLANNER_RECORD_FINDING_KEYS = ("usage_findings", "visual_findings")


def _clip_json(value: Any, *, char_budget: int) -> str:
    """Serialize ``value`` to JSON, clipping to ``char_budget`` deterministically."""
    text = json.dumps(value, ensure_ascii=False, default=str)
    if len(text) <= char_budget:
        return text
    head = text[: max(0, char_budget)]
    return (
        f"{head}…[truncated {len(text) - len(head)} chars for replanner context budget]"
    )


def _compact_findings(raw: Any) -> List[Dict[str, Any]]:
    """Project a findings list down to validator / severity / clipped message."""
    if not isinstance(raw, list):
        return []
    out: List[Dict[str, Any]] = []
    for item in raw[:_REPLANNER_MAX_FINDINGS_PER_LIST]:
        if not isinstance(item, dict):
            continue
        compact: Dict[str, Any] = {}
        for key in _REPLANNER_FINDING_KEYS:
            if key not in item:
                continue
            val = item[key]
            if key == "message" and isinstance(val, str):
                val = val[:_REPLANNER_FINDING_MESSAGE_CHARS]
            compact[key] = val
        if compact:
            out.append(compact)
    return out


def _slim_record_for_replanner(record: Dict[str, Any]) -> Dict[str, Any]:
    """Project one completed-step record to the compact view the replanner needs."""
    slim: Dict[str, Any] = {}
    for key in _REPLANNER_RECORD_KEEP_KEYS:
        if key in record:
            slim[key] = record[key]
    summary = record.get("step_summary")
    if summary is not None:
        summary_text = json.dumps(summary, ensure_ascii=False, default=str)
        if len(summary_text) > _REPLANNER_STEP_SUMMARY_CHAR_BUDGET:
            slim["step_summary"] = _clip_json(
                summary, char_budget=_REPLANNER_STEP_SUMMARY_CHAR_BUDGET
            )
        else:
            slim["step_summary"] = summary
    for key in _REPLANNER_RECORD_FINDING_KEYS:
        compact = _compact_findings(record.get(key))
        if compact:
            slim[key] = compact
    return slim


def _slim_completed_records_for_prompt(
    records: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Slim every record, then enforce a global budget by collapsing oldest first.

    Records are slimmed independently, then — if the serialized blob still
    exceeds :data:`_REPLANNER_TOTAL_RECORDS_CHAR_BUDGET` — the oldest records are
    collapsed to an identity stub (newest steps carry the freshest signal the
    replanner needs), keeping the projection deterministic and order-stable.
    """
    from ..research_context.outbound import project_outbound_records

    slimmed = project_outbound_records(records)
    if len(json.dumps(slimmed, ensure_ascii=False, default=str)) <= (
        _REPLANNER_TOTAL_RECORDS_CHAR_BUDGET
    ):
        return slimmed
    for idx in range(len(slimmed)):
        blob = json.dumps(slimmed, ensure_ascii=False, default=str)
        if len(blob) <= _REPLANNER_TOTAL_RECORDS_CHAR_BUDGET:
            break
        rec = slimmed[idx]
        slimmed[idx] = {
            "step_id": rec.get("step_id"),
            "status": rec.get("status"),
            "collapsed": "older step elided to fit replanner context budget",
        }
    return slimmed


class ReplannerAgent(PlannerAgent):
    """Revise an existing plan after probe outputs or executed steps."""

    def run(
        self,
        *,
        context: ResearchContext,
        current_plan: AnalysisPlan,
        probe_summary: Optional[Dict[str, Any]] = None,
        completed_step_records: Optional[Sequence[Dict[str, Any]]] = None,
        directive: Optional[str] = None,
        allowed_literature_citation_keys: Optional[Sequence[str]] = None,
        direct_comparator_literature_keys: Optional[Sequence[str]] = None,
    ) -> AnalysisPlan:
        completed = _slim_completed_records_for_prompt(
            list(completed_step_records or [])
        )
        # A ``directive`` is a high-priority, runtime-issued instruction (e.g. a
        # self-inflicted-block override on a task-viable cohort). It is surfaced
        # first so the replanner cannot bury it under the routine revise prose.
        directive_block = (
            f"PRIORITY RUNTIME DIRECTIVE (override prior plan revisions):\n{directive}\n\n"
            if directive
            else ""
        )
        replanner_context = scoped_planner_context(context)
        current_citation_keys = tuple(
            key
            for step in current_plan.steps
            for key in step.literature_citation_keys
        )
        allowed_citation_keys = _payload.normalize_literature_citation_keys(
            allowed_literature_citation_keys
            if allowed_literature_citation_keys is not None
            else current_citation_keys
        )
        direct_comparator_keys = _payload.normalize_literature_citation_keys(
            direct_comparator_literature_keys
        )
        literature_authority = _payload.bind_literature_citation_authority(
            "",
            allowed_citation_keys,
            direct_comparator_keys=direct_comparator_keys,
        )
        messages = [
            LLMMessage(
                role="system",
                content=_SYSTEM_GUIDE + _PRINCIPLES_GUIDE + "\n\n" + _REPLANNER_GUIDE,
            ),
            LLMMessage(
                role="user",
                content=(
                    directive_block
                    + "Revise the ICU-AWARE RESEARCH PLAN as JSON matching the "
                    "AnalysisPlan schema. Keep completed steps unchanged and "
                    "revise only the remaining steps when the probe summary or "
                    "completed step outputs justify it.\n\n"
                    + _scientific_actions.planner_scientific_action_guide(current_plan.analysis_type, detail="names_only") + "\n\n"
                    + trajectory_planner_contract_guide(
                        context=context,
                        analysis_type=current_plan.analysis_type,
                    )
                    + "\n\n"
                    f"CURRENT PLAN:\n{current_plan.model_dump_json(indent=2)}\n\n"
                    + literature_authority
                    + ("\n\n" if literature_authority else "")
                    + f"PROBE SUMMARY:\n{_clip_json(project_outbound_probe(probe_summary or {}), char_budget=_REPLANNER_PROBE_CHAR_BUDGET)}\n\n"
                    f"COMPLETED STEP RECORDS:\n{json.dumps(completed, ensure_ascii=False, default=str)}\n\n"
                    "RESEARCH CONTEXT:\n"
                    + _format_context(
                        replanner_context,
                        include_materialized_input_facts=True,
                        compact_method_constraints=True,
                    )
                    + "\n\n"
                    + planner_variable_catalog(context, replanner_context)
                ),
            ),
        ]
        replanner_bytes = sum(
            len(str(message.content or "").encode("utf-8")) for message in messages
        )
        if replanner_bytes > _PLANNER_PROMPT_BYTE_LIMIT:
            raise PlannerPromptBudgetError(
                "Replanner prompt transport budget exceeded: "
                f"{replanner_bytes} > {_PLANNER_PROMPT_BYTE_LIMIT} bytes. "
                "No plan, completed-step evidence, or scientific coordinate was "
                "truncated; reduce the scoped discovery catalog."
            )
        from ..providers.structured_retry import call_llm_with_structured_retry

        def parse_revised(raw: str) -> AnalysisPlan:
            decision_authority: dict[str, Mapping[str, Any]] = {}
            for decision in current_plan.know_how_decisions:
                card = decision_authority.setdefault(
                    decision.card_id,
                    {
                        "version": decision.card_version,
                        "file_sha256": decision.card_sha256,
                        "claims": {},
                    },
                )
                card["claims"][decision.claim_id] = tuple(decision.citation_ids)
            candidate = self._parse(
                raw,
                context,
                allowed_know_how_decisions=decision_authority,
                allowed_literature_citation_keys=allowed_citation_keys,
                direct_comparator_literature_keys=direct_comparator_keys,
                require_scientific_actions=True,
            )
            if candidate.know_how_decisions != current_plan.know_how_decisions:
                raise ValueError(
                    "Replanner must preserve know_how_decisions exactly; it may not "
                    "change claim dispositions, citations, versions, or card SHA."
                )
            return candidate

        revised = call_llm_with_structured_retry(
            self.llm,
            messages,
            parser=parse_revised,
            role="replanner",
            max_retries=2,
            max_tokens=4096,
            temperature=0.1,
            format_reminder=(
                "The JSON must be a single AnalysisPlan object with keys: "
                "research_question, steps, rationale, and the exact CURRENT PLAN "
                "know_how_decisions when present. Every step must include "
                "planned_analysis_role. Keep completed step_ids "
                "from the CURRENT PLAN unchanged; only revise the remaining steps."
                + _payload.literature_citation_retry_suffix(
                    allowed_citation_keys,
                    direct_comparator_keys=direct_comparator_keys,
                )
                + _payload.planner_science_retry_guide()
            ),
        )
        if revised.revision <= current_plan.revision:
            revised = revised.model_copy(update={"revision": current_plan.revision + 1})
        return revised
