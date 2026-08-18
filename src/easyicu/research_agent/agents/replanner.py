"""ReplannerAgent and its prompt context-budget guards."""

from __future__ import annotations

import json
from typing import Any, Dict, Mapping, Optional, Sequence

from ..planning import scientific_action_catalog as _scientific_actions
from ..planning.analysis_types import canonical_analysis_family
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

from .replanner_context import (
    REPLANNER_PROBE_CHAR_BUDGET as _REPLANNER_PROBE_CHAR_BUDGET,
    clip_json as _clip_json,
    slim_completed_records_for_prompt as _slim_completed_records_for_prompt,
)
from .runtime_suffix_replanner import run_runtime_suffix_replan


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
        suffix_only: bool = False,
    ) -> AnalysisPlan:
        if suffix_only:
            return run_runtime_suffix_replan(
                self,
                context=context,
                current_plan=current_plan,
                probe_summary=probe_summary,
                completed_step_records=completed_step_records,
                directive=directive,
                allowed_literature_citation_keys=allowed_literature_citation_keys,
                direct_comparator_literature_keys=direct_comparator_literature_keys,
            )
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
                    # A replan must not invent a scientific family the locked
                    # plan never declared.  When the (possibly legacy) plan has
                    # no canonical analysis_type, omit the action guide rather
                    # than guessing one: the plan's own steps remain the only
                    # authority for what may be revised.
                    + (
                        _scientific_actions.planner_scientific_action_guide(
                            current_plan.analysis_type, detail="names_only"
                        )
                        + "\n\n"
                        if canonical_analysis_family(current_plan.analysis_type)
                        is not None
                        else ""
                    )
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
