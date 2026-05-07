"""Agent definitions: planner, coder, analyzer, writer.

Each agent is a small object with one method (``run``) and one job.
They are stateless except for the LLM client and the prompt
templates. Coordination — the loop, the validators, the evidence
store — lives in :mod:`pipeline`. That separation is important so
that:

* a paper reviewer can read each agent in isolation,
* the loop can be replayed with a different LLM or with a mock,
* future work can swap in LangGraph / AutoGen without touching the
  agents.

Prompt design rule: every prompt is grounded in
:class:`ResearchContext`. The agents never see raw row-level data
through the prompt — only the structured context. The LLM cannot
hallucinate variable names, time windows or aggregation rules
because they are pinned in the system message.
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .analysis_types import infer_analysis_type, planner_analysis_type_guide
from .icu_rules import VariableKind, default_time_windows
from .llm import LLMClient, LLMMessage
from .prompts import PROMPT_PACK_VERSION, load_prompt_pack
from .schema import (
    AggregationRule,
    AgentRuntimeState,
    AnalysisPlan,
    ClinicalSemanticsResolution,
    AnalysisStep,
    ConceptRef,
    ConceptDescriptor,
    CritiqueReport,
    DataExtractionRequest,
    DataExtractionResult,
    EvidenceRef,
    ManuscriptDraftPacket,
    ReflectionMemoryEntry,
    ResearchContext,
    StatisticalAnalysisRequest,
    StatisticalAnalysisResult,
    VariableRole,
    VisualizationRequest,
    VisualizationResult,
)
from .temporal_semantics import ConceptValidationLayer, ICUEpisodeResolver, TemporalAlignmentEngine


def _dump_raw(text: str, tag: str) -> Optional[Path]:
    """Best-effort save of an LLM response that failed to parse (T1.3).

    Creates ``research_output/llm_debug/<tag>_<timestamp>.txt`` with the
    full raw response. Silent on any IO failure so the debug aid never
    masks the underlying parse error.
    """
    try:
        log_dir = Path(
            os.environ.get("EASYICU_LLM_DEBUG_DIR")
            or "./research_output/llm_debug"
        )
        log_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%dT%H%M%S_%f")
        path = log_dir / f"{tag}_{ts}.txt"
        path.write_text(text or "", encoding="utf-8")
        return path
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Shared prompt fragments
# ---------------------------------------------------------------------------


_PROMPT_PACK = load_prompt_pack()
_SYSTEM_GUIDE = _PROMPT_PACK["system"]
_CODER_GUIDE = _PROMPT_PACK["coder"]
_REPLANNER_GUIDE = _PROMPT_PACK["replanner"]
_WRITER_GUIDE = _PROMPT_PACK["writer"]


def _format_variable(v: ConceptDescriptor) -> str:
    miss = ""
    if v.missingness is not None:
        miss = (
            f" missing={v.missingness.fraction_missing:.1%} "
            f"(severity={v.missingness.missingness_severity})"
        )
    pit = f" pitfalls={v.pitfalls!r}" if v.pitfalls else ""
    rng = f" range={v.valid_range}" if v.valid_range else ""
    unit = f" unit={v.unit}" if v.unit else ""
    return (
        f"- {v.name} | role={v.role.value} dtype={v.dtype}{unit}{rng}"
        f" agg_default={v.aggregation_default.value if v.aggregation_default else 'any'}"
        f"{miss}{pit}"
    )


def _format_context(ctx: ResearchContext) -> str:
    lines = [
        f"Research question: {ctx.research_question}",
        f"Cohort: {ctx.cohort.cohort_name} ({ctx.cohort.database})"
        f" — {ctx.cohort.n_stays:,} stays / {ctx.cohort.n_patients:,} patients",
    ]
    if ctx.cohort.inclusion_criteria:
        lines.append("Inclusion: " + "; ".join(ctx.cohort.inclusion_criteria))
    if ctx.cohort.exclusion_criteria:
        lines.append("Exclusion: " + "; ".join(ctx.cohort.exclusion_criteria))
    if ctx.target_outcome:
        lines.append(f"Target outcome: {ctx.target_outcome}")
    lines.append("Time windows:")
    for w in ctx.time_windows:
        lines.append(f"  - {w.name}: {w.start_hours}-{w.end_hours}h from {w.anchor}")
    lines.append("Variables:")
    for v in ctx.variables:
        lines.append(_format_variable(v))
    if ctx.cross_database_validation:
        lines.append("Cross-database replication planned: " + ", ".join(ctx.cross_database_validation))
    if ctx.user_preferences is not None:
        prefs = ctx.user_preferences
        lines.append("User preferences:")
        if prefs.inferred_analysis_family:
            lines.append(f"  - inferred_analysis_family: {prefs.inferred_analysis_family}")
        if prefs.starter_template_key:
            lines.append(f"  - starter_template_key: {prefs.starter_template_key}")
        if prefs.preferred_methods:
            lines.append(f"  - preferred_methods: {prefs.preferred_methods}")
        if prefs.evaluation_focus:
            lines.append(f"  - evaluation_focus: {prefs.evaluation_focus}")
        if prefs.subgroup_sensitivity:
            lines.append(f"  - subgroup_sensitivity: {prefs.subgroup_sensitivity}")
        if prefs.timing_and_design:
            lines.append(f"  - timing_and_design: {prefs.timing_and_design}")
        if prefs.data_constraints:
            lines.append(f"  - data_constraints: {prefs.data_constraints}")
        if prefs.must_have_outputs:
            lines.append(f"  - must_have_outputs: {prefs.must_have_outputs}")
        if prefs.covariates:
            lines.append("  - covariates: " + ", ".join(prefs.covariates))
        if prefs.extra_notes:
            lines.append(f"  - extra_notes: {prefs.extra_notes}")
    if ctx.notes:
        lines.append("User/run notes: " + ctx.notes)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------


class PlannerAgent:
    """Produces an :class:`AnalysisPlan` from the research context.

    The planner is the only agent that emits structured JSON. All
    downstream agents receive the parsed plan, so a hallucinated step
    cannot leak past the parser.
    """

    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm
        self.last_dropped_plan_keys: Dict[str, List[str]] = {
            "top_level": [],
            "steps": [],
        }

    def run(self, context: ResearchContext) -> AnalysisPlan:
        messages = [
            LLMMessage(role="system", content=_SYSTEM_GUIDE),
            LLMMessage(
                role="user",
                content=(
                    "Produce an ICU-AWARE RESEARCH PLAN as JSON matching the "
                    "AnalysisPlan schema. First infer the EHR analysis type, "
                    "then choose only the steps justified by that family and "
                    "the available context. The plan must not assume that "
                    "every task needs Table 1, outcome incidence, missingness, "
                    "or a primary association model. If cross-database "
                    "replication is requested, include a cross-database step, "
                    "but mark it as a feasibility / protocol step unless the "
                    "ResearchContext explicitly provides external cohort files. "
                    "Use score-specific QC steps only when a relevant score is "
                    "actually central to the question. Do not put invented "
                    "prefixed variables such as eicu:age in `inputs`. Honor "
                    "explicit user preferences and requested outputs when they "
                    "are compatible with the cohort and analysis family.\n\n"
                    + planner_analysis_type_guide()
                    + "\n\n"
                    "OUTPUT FORMAT — VERY IMPORTANT:\n"
                    "Return *only* a single JSON object matching the "
                    "AnalysisPlan schema. No prose, no markdown headings, no "
                    "trailing commentary. A ```json … ``` fence is acceptable; "
                    "anything outside that fence will be discarded.\n\n"
                    "Required JSON shape (truncated example):\n"
                    "{\n"
                    '  "research_question": "<copy from context>",\n'
                    '  "steps": [\n'
                    "    {\n"
                    '      "step_id": "01_table_one",\n'
                    '      "intent": "<one sentence>",\n'
                    '      "inputs": ["<variable names from context>"],\n'
                    '      "expected_outputs": ["table:table_one"],\n'
                    '      "method": "descriptive",\n'
                    '      "icu_rule_refs": ["aggregation_rule_for"]\n'
                    "    }\n"
                    "  ],\n"
                    '  "rationale": "<one paragraph>"\n'
                    "}\n\n"
                    "RESEARCH CONTEXT:\n" + _format_context(context)
                ),
            ),
        ]
        raw = self.llm.complete(messages, max_tokens=4096, temperature=0.2)
        return self._parse(raw, context)

    def _parse(self, raw: str, context: ResearchContext) -> AnalysisPlan:
        text = raw.strip()
        # Strip a fenced block anywhere in the response (already
        # tolerant of the leading-prose case).
        if "```" in text:
            text = _strip_code_fence(text)
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            # Last-ditch: try to recover a JSON block from inside the response.
            match = _first_json_block(text)
            if match is None:
                # T1.3 — be loud about exactly what came back. Dump the
                # whole raw response so the user can hand it to a human
                # debugger or back to Claude for prompt iteration.
                _dump_raw(raw, "planner_unparseable")
                head = (raw or "").strip().replace("\n", " ⏎ ")[:600]
                raise ValueError(
                    f"Planner LLM did not return parseable JSON "
                    f"(len={len(raw or '')}). "
                    f"First 600 chars: {head!r}. "
                    "Full raw response written to "
                    "research_output/llm_debug/planner_unparseable_*.txt; "
                    "set EASYICU_LLM_DEBUG=1 to also capture every LLM call."
                )
            data = json.loads(match)
        if "research_question" not in data:
            data["research_question"] = context.research_question
        data, dropped = _normalise_plan_payload(data)
        self.last_dropped_plan_keys = dropped
        return AnalysisPlan.model_validate(data)


class ReplannerAgent(PlannerAgent):
    """Revise an existing plan after probe outputs or executed steps."""

    def run(
        self,
        *,
        context: ResearchContext,
        current_plan: AnalysisPlan,
        probe_summary: Optional[Dict[str, Any]] = None,
        completed_step_records: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> AnalysisPlan:
        completed = list(completed_step_records or [])
        messages = [
            LLMMessage(role="system", content=_SYSTEM_GUIDE + "\n\n" + _REPLANNER_GUIDE),
            LLMMessage(
                role="user",
                content=(
                    "Revise the ICU-AWARE RESEARCH PLAN as JSON matching the "
                    "AnalysisPlan schema. Keep completed steps unchanged and "
                    "revise only the remaining steps when the probe summary or "
                    "completed step outputs justify it.\n\n"
                    f"CURRENT PLAN:\n{current_plan.model_dump_json(indent=2)}\n\n"
                    f"PROBE SUMMARY:\n{json.dumps(probe_summary or {}, ensure_ascii=False, default=str)}\n\n"
                    f"COMPLETED STEP RECORDS:\n{json.dumps(completed, ensure_ascii=False, default=str)}\n\n"
                    "RESEARCH CONTEXT:\n" + _format_context(context)
                ),
            ),
        ]
        raw = self.llm.complete(messages, max_tokens=4096, temperature=0.1)
        revised = self._parse(raw, context)
        if revised.revision <= current_plan.revision:
            revised = revised.model_copy(update={"revision": current_plan.revision + 1})
        return revised


# ---------------------------------------------------------------------------
# ICU-native worker agents and runtime supervisor
# ---------------------------------------------------------------------------


class ClinicalSemanticsAgent:
    """Resolve ICU-specific semantics into deterministic typed state.

    Inspired by HealthFlow's meta/evaluator loop and OpenLens' shared-state
    handoffs, this agent stays deterministic by default: it interprets the
    already-constrained :class:`ResearchContext` rather than free-reading raw
    tables.
    """

    def __init__(self) -> None:
        self._alignment = TemporalAlignmentEngine()
        self._concept_validation = ConceptValidationLayer()

    def run(self, *, context: ResearchContext) -> ClinicalSemanticsResolution:
        family = infer_analysis_type(context).key
        windows, constraints = self._alignment.infer(
            research_question=context.research_question,
            timing_and_design=(context.user_preferences.timing_and_design if context.user_preferences else None),
            explicit_windows=context.time_windows,
        )
        concept_refs: List[ConceptRef] = []
        caveats: List[str] = []
        for variable in context.variables:
            concept_refs.append(
                ConceptRef(
                    name=variable.name,
                    role=variable.role,
                    source_concept=variable.source_concept,
                    analysis_window=variable.analysis_window,
                )
            )
            payload = self._concept_validation.validate_descriptor_payload(
                source_info={
                    "source_tables": variable.source_tables,
                    "item_ids": variable.item_ids,
                    "unit_normalization": variable.unit_normalization,
                    "temporal_resolution": variable.temporal_resolution,
                    "clinical_caveats": variable.clinical_caveats,
                    "missingness_semantics": variable.missingness_semantics,
                },
                column_name=variable.name,
            )
            if payload.get("clinical_caveats"):
                caveats.extend(str(x) for x in payload["clinical_caveats"])
        ambiguity_notes: List[str] = []
        if not constraints and context.user_preferences and context.user_preferences.timing_and_design:
            ambiguity_notes.append(
                "Timing/design preferences were provided but no deterministic temporal constraint could be parsed."
            )
        safety_guardrails = sorted(
            {
                caveat
                for variable in context.variables
                for caveat in ([*(variable.pitfalls or []), *(variable.forbidden_transformations or [])])
                if caveat
            }
        )
        provenance_notes = [
            "Clinical semantics derived from typed ResearchContext, not raw SQL access.",
            f"Inferred analysis family: {family}.",
        ]
        return ClinicalSemanticsResolution(
            analysis_family=family,
            target_outcome=context.target_outcome,
            temporal_constraints=constraints or context.temporal_constraints,
            recommended_time_windows=windows or list(context.time_windows),
            target_concepts=concept_refs,
            ambiguity_notes=ambiguity_notes,
            safety_guardrails=safety_guardrails,
            provenance_notes=provenance_notes,
        )


class DataExtractionAgent:
    """Create a constrained extraction request/result handoff.

    This agent does *not* query raw SQL. It packages the cohort and concept
    provenance already resolved by EASYICU's data layer into a typed request
    that downstream agents can consume safely.
    """

    def __init__(self) -> None:
        self._resolver = ICUEpisodeResolver()

    def build_request(self, *, context: ResearchContext, semantics: ClinicalSemanticsResolution) -> DataExtractionRequest:
        return DataExtractionRequest(
            cohort_name=context.cohort.cohort_name,
            database=context.cohort.database,
            concept_refs=semantics.target_concepts,
            time_windows=semantics.recommended_time_windows,
            temporal_constraints=semantics.temporal_constraints,
            cohort_provenance=context.cohort.provenance,
            notes=semantics.provenance_notes,
        )

    def materialize(
        self,
        *,
        context: ResearchContext,
        request: DataExtractionRequest,
    ) -> DataExtractionResult:
        provenance = dict(context.cohort.provenance or {})
        provenance["extraction_request"] = request.model_dump(mode="json")
        provenance["episode_resolution"] = self._resolver.resolve(
            df=_empty_df_placeholder(),
            database=context.cohort.database,
            id_columns=context.cohort.id_columns,
            time_columns=context.cohort.time_columns,
            outcome_columns=context.cohort.outcome_columns,
            target_outcome=context.target_outcome,
            cohort_path=context.cohort_parquet,
        ).provenance
        return DataExtractionResult(
            cohort_path=context.cohort_parquet or "",
            n_rows=context.cohort.n_stays,
            concept_refs=request.concept_refs,
            provenance=provenance,
            evidence_refs=[],
        )


class StatisticalAnalysisAgent:
    """Typed analysis-step planner for execution and benchmark accounting."""

    def build_request(
        self,
        *,
        context: ResearchContext,
        semantics: ClinicalSemanticsResolution,
        step: AnalysisStep,
        evidence_refs: Sequence[EvidenceRef],
    ) -> StatisticalAnalysisRequest:
        prefs = context.user_preferences
        return StatisticalAnalysisRequest(
            step=step,
            analysis_family=semantics.analysis_family,
            target_outcome=context.target_outcome,
            covariates=list((prefs.covariates if prefs else []) or []),
            evaluation_focus=(prefs.evaluation_focus if prefs else None),
            must_have_outputs=(prefs.must_have_outputs if prefs else None),
            evidence_refs=list(evidence_refs),
            notes=semantics.safety_guardrails,
        )

    def summarize_result(
        self,
        *,
        step: AnalysisStep,
        step_summary: Dict[str, Any],
        evidence_refs: Sequence[EvidenceRef],
        validator_messages: Sequence[str],
        analysis_family: str,
    ) -> StatisticalAnalysisResult:
        estimate = _coerce_primary_estimate(step_summary)
        return StatisticalAnalysisResult(
            step_id=step.step_id,
            method_family=analysis_family,
            primary_estimate=estimate[0],
            estimate_label=estimate[1],
            estimate_interval=estimate[2],
            summary_metrics=dict(step_summary or {}),
            evidence_refs=list(evidence_refs),
            validator_messages=list(validator_messages),
        )


class VisualizationAgent:
    """Typed publication-figure handoff derived from registered evidence."""

    def build_request(
        self,
        *,
        context: ResearchContext,
        semantics: ClinicalSemanticsResolution,
        step: AnalysisStep,
        evidence_refs: Sequence[EvidenceRef],
    ) -> VisualizationRequest:
        prefs = context.user_preferences
        return VisualizationRequest(
            step=step,
            analysis_family=semantics.analysis_family,
            evidence_refs=list(evidence_refs),
            must_have_outputs=(prefs.must_have_outputs if prefs else None),
            notes=[
                "All figure claims must remain evidence-bound.",
                *semantics.safety_guardrails,
            ],
        )

    def summarize_result(
        self,
        *,
        step: AnalysisStep,
        evidence_refs: Sequence[EvidenceRef],
        qa_messages: Sequence[str],
    ) -> VisualizationResult:
        titles = [ref.description or ref.evidence_id for ref in evidence_refs if ref.kind == "figure"]
        return VisualizationResult(
            step_id=step.step_id,
            figure_titles=titles,
            evidence_refs=list(evidence_refs),
            qa_messages=list(qa_messages),
        )


class ManuscriptAgent:
    """Draft-only manuscript agent that stays human-supervised for discussion."""

    def __init__(self, llm: LLMClient, *, language: str = "en") -> None:
        self.llm = llm
        self.language = language

    def build_packet(
        self,
        *,
        context: ResearchContext,
        semantics: ClinicalSemanticsResolution,
        evidence_refs: Sequence[EvidenceRef],
        findings: Sequence[str],
        caveats: Sequence[str],
    ) -> ManuscriptDraftPacket:
        return ManuscriptDraftPacket(
            title=context.research_question,
            abstract_focus=context.target_outcome,
            analysis_family=semantics.analysis_family,
            evidence_refs=list(evidence_refs),
            findings=list(findings),
            caveats=list(caveats),
        )

    def run(
        self,
        *,
        context: ResearchContext,
        evidence_ids: Sequence[str],
    ) -> str:
        return WriterAgent(self.llm, language=self.language).run(
            context=context,
            evidence_ids=evidence_ids,
        )


class CriticAgent:
    """Structured evaluator for execute→critique→revise loops.

    The implementation is intentionally conservative: deterministic findings and
    missing-evidence checks take precedence over free-form LLM critique so the
    runtime stays ICU-safe even when no evaluator model is configured.
    """

    def __init__(self, llm: Optional[LLMClient] = None) -> None:
        self.llm = llm

    def review_step(
        self,
        *,
        step: AnalysisStep,
        step_summary: Dict[str, Any],
        evidence_refs: Sequence[EvidenceRef],
        findings: Sequence[str],
    ) -> CritiqueReport:
        concerns = [msg for msg in findings if msg]
        status: str = "pass"
        if not evidence_refs:
            status = "blocked"
            concerns.append("No evidence refs were registered for this step.")
        elif concerns:
            status = "needs_revision"
        return CritiqueReport(
            status=status,  # type: ignore[arg-type]
            reviewer="CriticAgent",
            concerns=concerns,
            unsupported_claims=[],
            missing_evidence_refs=[] if evidence_refs else [step.step_id],
            suggested_repairs=_suggest_repairs_for(step_summary, findings),
            related_evidence_refs=list(evidence_refs),
        )

    def review_manuscript(
        self,
        *,
        scaffold: str,
        available_evidence_ids: Sequence[str],
    ) -> CritiqueReport:
        missing = sorted(set(re.findall(r"\[evidence missing: ([^\]]+)\]", scaffold)))
        concerns: List[str] = []
        if missing:
            concerns.append("Manuscript contains unresolved evidence placeholders.")
        unsupported = _sentences_missing_evidence_tokens(scaffold)
        if unsupported:
            concerns.append("Some result-like sentences were filtered or remain unsupported.")
        status: str = "pass"
        if missing:
            status = "blocked"
        elif unsupported:
            status = "needs_revision"
        return CritiqueReport(
            status=status,  # type: ignore[arg-type]
            reviewer="CriticAgent",
            concerns=concerns,
            unsupported_claims=unsupported,
            missing_evidence_refs=missing,
            suggested_repairs=[
                "Ensure every quantitative result sentence cites a valid {evidence:<id>} placeholder.",
                "Regenerate unsupported narrative from registered evidence artifacts only.",
            ],
            related_evidence_refs=[
                EvidenceRef(evidence_id=eid)
                for eid in available_evidence_ids
                if eid in set(available_evidence_ids)
            ],
        )


class RuntimeSupervisor:
    """Coordinator for typed shared state and gated worker execution.

    This stays lighter than LangGraph today, but follows the same shared-state /
    supervisor-worker idea seen in LangGraph supervisor patterns and OpenLens:
    each worker reads and writes a typed state object instead of passing long
    natural-language transcripts directly.
    """

    def __init__(
        self,
        *,
        clinical_semantics: Optional[ClinicalSemanticsAgent] = None,
        data_extraction: Optional[DataExtractionAgent] = None,
        statistical_analysis: Optional[StatisticalAnalysisAgent] = None,
        visualization: Optional[VisualizationAgent] = None,
        critic: Optional[CriticAgent] = None,
    ) -> None:
        self.clinical_semantics = clinical_semantics or ClinicalSemanticsAgent()
        self.data_extraction = data_extraction or DataExtractionAgent()
        self.statistical_analysis = statistical_analysis or StatisticalAnalysisAgent()
        self.visualization = visualization or VisualizationAgent()
        self.critic = critic or CriticAgent()

    def bootstrap_state(self, *, run_id: str, context: ResearchContext) -> AgentRuntimeState:
        semantics = self.clinical_semantics.run(context=context)
        extraction_request = self.data_extraction.build_request(context=context, semantics=semantics)
        extraction_result = self.data_extraction.materialize(context=context, request=extraction_request)
        reflections = _initial_reflection_memory(context=context, semantics=semantics)
        return AgentRuntimeState(
            run_id=run_id,
            analysis_family=semantics.analysis_family,
            semantics=semantics,
            extraction_request=extraction_request,
            extraction_result=extraction_result,
            reflection_memory=reflections,
        )

    def prepare_step_state(
        self,
        *,
        state: AgentRuntimeState,
        context: ResearchContext,
        step: AnalysisStep,
        evidence_refs: Sequence[EvidenceRef],
    ) -> AgentRuntimeState:
        analysis_request = self.statistical_analysis.build_request(
            context=context,
            semantics=state.semantics or self.clinical_semantics.run(context=context),
            step=step,
            evidence_refs=evidence_refs,
        )
        visualization_request = self.visualization.build_request(
            context=context,
            semantics=state.semantics or self.clinical_semantics.run(context=context),
            step=step,
            evidence_refs=evidence_refs,
        )
        return state.model_copy(
            update={
                "current_step": step,
                "analysis_request": analysis_request,
                "visualization_request": visualization_request,
                "evidence_refs": list(evidence_refs),
            }
        )

    def critique_step(
        self,
        *,
        state: AgentRuntimeState,
        step_summary: Dict[str, Any],
        evidence_refs: Sequence[EvidenceRef],
        findings: Sequence[str],
    ) -> AgentRuntimeState:
        critique = self.critic.review_step(
            step=state.current_step or AnalysisStep(step_id="unknown", intent="unknown"),
            step_summary=step_summary,
            evidence_refs=evidence_refs,
            findings=findings,
        )
        analysis_result = self.statistical_analysis.summarize_result(
            step=state.current_step or AnalysisStep(step_id="unknown", intent="unknown"),
            step_summary=step_summary,
            evidence_refs=evidence_refs,
            validator_messages=findings,
            analysis_family=state.analysis_family or "unknown",
        )
        visualization_result = self.visualization.summarize_result(
            step=state.current_step or AnalysisStep(step_id="unknown", intent="unknown"),
            evidence_refs=evidence_refs,
            qa_messages=[msg for msg in findings if "visual" in msg.lower()],
        )
        new_memory = list(state.reflection_memory)
        if critique.status == "pass":
            new_memory.append(
                ReflectionMemoryEntry(
                    category="successful_workflow",
                    summary=f"{analysis_result.step_id} executed with evidence-bound outputs.",
                    analysis_family=state.analysis_family,
                    recommendation="Reuse this step family and validator bundle on similar ICU tasks.",
                )
            )
        elif critique.status in {"needs_revision", "blocked"}:
            new_memory.append(
                ReflectionMemoryEntry(
                    category="failed_pattern",
                    summary=f"{analysis_result.step_id} triggered critique status={critique.status}.",
                    analysis_family=state.analysis_family,
                    recommendation="Inspect validator findings before advancing to manuscript generation.",
                    metadata={"concerns": critique.concerns},
                )
            )
        return state.model_copy(
            update={
                "analysis_result": analysis_result,
                "visualization_result": visualization_result,
                "critique": critique,
                "reflection_memory": new_memory,
                "evidence_refs": list(evidence_refs),
            }
        )


# ---------------------------------------------------------------------------
# Coder
# ---------------------------------------------------------------------------


class CoderAgent:
    """Generates a self-contained Python analysis script for one step."""

    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    def run(self, *, context: ResearchContext, step: AnalysisStep) -> str:
        messages = [
            LLMMessage(role="system", content=_SYSTEM_GUIDE + _CODER_GUIDE),
            LLMMessage(
                role="user",
                content=(
                    f"Write the Python CODE for STEP {step.step_id}.\n"
                    f"Step intent: {step.intent}\n"
                    f"Step inputs: {step.inputs}\n"
                    f"Expected outputs: {step.expected_outputs}\n"
                    f"Method: {step.method or '(unspecified — choose conservatively)'}\n\n"
                    "OUTPUT FORMAT — VERY IMPORTANT:\n"
                    "Return *only* a complete, runnable Python script. A "
                    "```python … ``` fence is acceptable; any text outside "
                    "the fence will be discarded. Do NOT include the cohort "
                    "data inline; read it from `os.environ['COHORT_PARQUET']`. "
                    "Do NOT print or describe what the script does — write "
                    "the script itself. Respect explicit user preferences "
                    "recorded in the ResearchContext, especially requested "
                    "outputs, evaluation metrics, timing rules, and design "
                    "constraints.\n\n"
                    "RESEARCH CONTEXT:\n" + _format_context(context)
                ),
            ),
        ]
        raw = self.llm.complete(messages, max_tokens=4096, temperature=0.1)
        return _strip_code_fence(raw.strip())

    def repair(
        self,
        *,
        context: ResearchContext,
        step: AnalysisStep,
        code: str,
        run_log: str,
        attempt: int = 1,
    ) -> str:
        """Ask the coder model for a minimal executable repair.

        Real hosted/free-tier models often produce scripts that are
        logically fine but brittle around pandas/matplotlib edge cases.
        The pipeline keeps the first failure as evidence, then gives
        the coder the traceback once and asks for a complete replacement
        script.
        """
        messages = [
            LLMMessage(role="system", content=_SYSTEM_GUIDE + _CODER_GUIDE),
            LLMMessage(
                role="user",
                content=(
                    f"REPAIR THE PYTHON CODE FOR STEP {step.step_id}.\n"
                    f"Repair attempt: {attempt}\n"
                    f"Step intent: {step.intent}\n"
                    f"Step inputs: {step.inputs}\n"
                    f"Expected outputs: {step.expected_outputs}\n"
                    f"Method: {step.method or '(unspecified)'}\n\n"
                    "The previous script failed at execution time. Return "
                    "only a complete replacement Python script that follows "
	                    "the original code contract and writes the same expected "
	                    "artefacts when possible. Make the smallest robust fix; "
	                    "do not add prose, markdown, or an explanation. Keep "
                        "honoring explicit user preferences recorded in the "
                        "ResearchContext.\n\n"
	                    "PREVIOUS SCRIPT:\n```python\n"
                    + code[-12000:]
                    + "\n```\n\n"
                    "RUN LOG / TRACEBACK:\n```\n"
                    + run_log[-8000:]
                    + "\n```\n\n"
                    "RESEARCH CONTEXT:\n" + _format_context(context)
                ),
            ),
        ]
        raw = self.llm.complete(messages, max_tokens=4096, temperature=0.05)
        return _strip_code_fence(raw.strip())


# ---------------------------------------------------------------------------
# Analyzer (interpretation)
# ---------------------------------------------------------------------------


class AnalyzerAgent:
    """Turns step outputs into a short, evidence-grounded interpretation."""

    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    def run(
        self,
        *,
        context: ResearchContext,
        step: AnalysisStep,
        step_summary: Dict[str, Any],
        evidence_ids: Sequence[str],
    ) -> str:
        messages = [
            LLMMessage(role="system", content=_SYSTEM_GUIDE),
            LLMMessage(
                role="user",
                content=(
                    f"INTERPRET the results of step {step.step_id}.\n"
                    f"Step intent: {step.intent}\n"
                    f"Numeric summary (machine-readable): {json.dumps(step_summary, default=str)}\n"
                    f"Evidence ids you may cite verbatim: {list(evidence_ids)}\n\n"
                    "Constraints:\n"
                    "- Cite at least one evidence_id for every numeric claim, "
                    "in the form {{evidence:<id>}}.\n"
                    "- Do not introduce numbers that are not in the summary.\n"
                    "- 4 sentences max. No clinical recommendations.\n\n"
                    "RESEARCH CONTEXT:\n" + _format_context(context)
                ),
            ),
        ]
        return self.llm.complete(messages, max_tokens=512, temperature=0.2).strip()


# ---------------------------------------------------------------------------
# Writer (manuscript scaffolder)
# ---------------------------------------------------------------------------


class WriterAgent:
    """Produces a manuscript scaffold whose every claim cites an evidence id.

    The writer does NOT generate Discussion or clinical claims; that is
    a policy decision encoded in the prompt and enforced by the
    pipeline (Discussion section is left blank with a note for the
    human author).
    """

    def __init__(self, llm: LLMClient, *, language: str = "en") -> None:
        self.llm = llm
        lang = (language or "en").lower()
        self.language = "zh" if lang.startswith(("zh", "cn", "chinese")) else "en"

    def run(
        self,
        *,
        context: ResearchContext,
        evidence_ids: Sequence[str],
    ) -> str:
        messages = [
            LLMMessage(role="system", content=_SYSTEM_GUIDE + _WRITER_GUIDE),
            LLMMessage(
                role="user",
                content=(
                    "Write a MANUSCRIPT scaffold (markdown) with sections "
                    "Title, Abstract (one paragraph), Methods, Results. "
                    "Leave Discussion as a one-line stub: "
                    "'(left to the human author)'.\n\n"
                    + _writer_language_instruction(self.language)
                    + "\n\n"
                    "CITATION RULE — VERY IMPORTANT:\n"
                    "`{evidence:<id>}` is a *citation*, not a value. It "
                    "binds to a markdown link in the rendered manuscript. "
                    "Treat it like an inline footnote.\n"
                    "  • DO write the actual numbers in prose, then cite. "
                    "    e.g. `The cohort comprised 51,838 stays "
                    "{evidence:table_one}.`\n"
                    "  • DO write `(see {evidence:primary_association})` "
                    "    after a sentence describing a finding.\n"
                    "  • DO NOT use a placeholder *as the noun*. "
                    "    e.g. NEVER write `a cohort of {evidence:table_one} "
                    "patients` — the binder has no number to substitute "
                    "and the manuscript becomes unreadable. Pull the "
                    "number from the registered tables/statistics first, "
                    "write it inline, then cite.\n"
                    "  • If a number is unknown, say so explicitly "
                    "    (e.g. `the median age was [TBD] years "
                    "{evidence:table_one}`) — never paper over it with "
                    "a placeholder noun.\n\n"
                    "PLACEHOLDER FORMAT:\n"
                    "Exact form `{evidence:<id>}`, no spaces inside braces. "
                    "Use only ids from the list below; anything else "
                    "renders as `[evidence missing: …]`. Prefer short "
                    "semantic aliases (`table_one`, `outcome_rate`, "
                    "`sofa_strata`, `primary_association`) when available.\n\n"
                    "OUTPUT FORMAT:\n"
                    "Return *only* the markdown manuscript. No commentary "
                    "before or after. A leading ```markdown … ``` fence is "
                    "acceptable and will be stripped.\n\n"
                    f"Available evidence ids and aliases: {list(evidence_ids)}\n\n"
                    "RESEARCH CONTEXT:\n" + _format_context(context)
                ),
            ),
        ]
        raw = self.llm.complete(messages, max_tokens=2048, temperature=0.2).strip()
        # Free-tier models often wrap markdown in ```markdown … ```; the
        # binder needs raw markdown to find {evidence:*} placeholders.
        return _strip_code_fence(raw)


def _writer_language_instruction(language: str) -> str:
    if language == "zh":
        return (
            "OUTPUT LANGUAGE: zh / Simplified Chinese. Keep section headings "
            "as markdown headings. Preserve every `{evidence:<id>}` placeholder "
            "exactly as ASCII; do not translate evidence ids, filenames, variable "
            "names, or code-like tokens."
        )
    return (
        "OUTPUT LANGUAGE: en / English. Preserve every `{evidence:<id>}` "
        "placeholder exactly as ASCII."
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _strip_code_fence(text: str) -> str:
    """Extract the content of the first ```...``` fenced block, if any.

    Free-tier LLMs frequently wrap their output with explanatory prose:

        Here's the analysis plan you asked for:
        ```json
        { ... }
        ```
        Let me know if you need anything else!

    A naïve "starts with ``` ?" check misses that. We instead find the
    first triple-backtick fence anywhere in the response and return
    only the contents of the first balanced fence. If no fence is
    found, the original text is returned unchanged so the JSON / code
    parsers downstream can still try.
    """
    if "```" not in text:
        return text
    # Match ```optional-language\n<body>\n``` (DOTALL, non-greedy)
    m = re.search(r"```[^\n`]*\n(.*?)\n```", text, flags=re.DOTALL)
    if m is None:
        # Stripped of the language tag but no closing fence — fall back to
        # everything after the first fence.
        idx = text.find("```")
        rest = text[idx + 3:]
        # drop a leading language tag (json, python, etc.) on the same line
        nl = rest.find("\n")
        if nl >= 0 and rest[:nl].strip().isalnum():
            rest = rest[nl + 1:]
        # if there's still a trailing fence, cut at it
        end = rest.find("```")
        if end >= 0:
            rest = rest[:end]
        return rest.strip() + "\n"
    return m.group(1).strip() + "\n"


def _first_json_block(text: str) -> Optional[str]:
    """Find the first balanced ``{...}`` block, ignoring braces inside strings.

    Robust against free-tier LLM output that sprinkles braces across
    inline prose / comments / code blocks. Walks the text once,
    tracking string state and escape sequences so brace counts inside
    `"…{…}…"` don't fool us.
    """
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_str = False
    escape = False
    for i in range(start, len(text)):
        c = text[i]
        if in_str:
            if escape:
                escape = False
            elif c == "\\":
                escape = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def _normalise_plan_payload(
    data: Dict[str, Any]
) -> Tuple[Dict[str, Any], Dict[str, List[str]]]:
    """Drop hosted-model extras before validating the strict schema.

    Returns both the normalized payload and a structured summary of the
    keys that were discarded so the pipeline can surface them in the
    manifest instead of silently suppressing them.
    """
    allowed_plan = {"research_question", "steps", "rationale", "revision"}
    allowed_step = {
        "step_id",
        "intent",
        "inputs",
        "expected_outputs",
        "method",
        "icu_rule_refs",
    }
    dropped: Dict[str, List[str]] = {"top_level": [], "steps": []}
    out = {}
    for key, value in data.items():
        if key in allowed_plan:
            out[key] = value
        else:
            dropped["top_level"].append(str(key))
    steps = []
    for idx, raw_step in enumerate(out.get("steps", []) or []):
        if isinstance(raw_step, dict):
            step_payload = {}
            for key, value in raw_step.items():
                if key in allowed_step:
                    step_payload[key] = value
                else:
                    step_id = raw_step.get("step_id") or f"step[{idx}]"
                    dropped["steps"].append(f"{step_id}:{key}")
            steps.append(step_payload)
    out["steps"] = steps
    return out, dropped


def _coerce_primary_estimate(
    step_summary: Dict[str, Any]
) -> Tuple[Optional[float], Optional[str], Optional[List[float]]]:
    candidates = [
        ("primary_or", "odds_ratio"),
        ("primary_hr", "hazard_ratio"),
        ("auroc", "auroc"),
        ("brier_score", "brier_score"),
        ("calibration_slope", "calibration_slope"),
        ("silhouette", "silhouette"),
    ]
    for key, label in candidates:
        value = step_summary.get(key)
        if isinstance(value, (int, float)):
            interval = step_summary.get(f"{key}_ci")
            if isinstance(interval, list) and len(interval) == 2:
                try:
                    return float(value), label, [float(interval[0]), float(interval[1])]
                except Exception:
                    pass
            return float(value), label, None
    model_results = step_summary.get("model_results")
    if isinstance(model_results, dict):
        for label, payload in model_results.items():
            if isinstance(payload, dict):
                estimate = payload.get("estimate") or payload.get("value") or payload.get("or")
                if isinstance(estimate, (int, float)):
                    interval = payload.get("ci") or payload.get("interval")
                    if isinstance(interval, list) and len(interval) == 2:
                        try:
                            return float(estimate), str(label), [float(interval[0]), float(interval[1])]
                        except Exception:
                            pass
                    return float(estimate), str(label), None
    return None, None, None


def _suggest_repairs_for(step_summary: Dict[str, Any], findings: Sequence[str]) -> List[str]:
    repairs: List[str] = []
    text = " ".join(findings).lower()
    if "calibration" in text:
        repairs.append("Add or surface calibration diagnostics before accepting the result.")
    if "leakage" in text:
        repairs.append("Revisit train/test split and feature timing to eliminate data leakage.")
    if "competing risk" in text:
        repairs.append("Use a competing-risks aware analysis plan rather than a simple binary endpoint.")
    if "evidence" in text:
        repairs.append("Register missing artifacts and bind them through evidence_id before drafting results.")
    if not repairs and step_summary:
        repairs.append("Review the step summary and regenerate the step with explicit guardrails.")
    return repairs


def _sentences_missing_evidence_tokens(scaffold: str) -> List[str]:
    unsupported: List[str] = []
    text = re.sub(r"```.*?```", " ", scaffold, flags=re.S)
    for raw_sentence in re.split(r"(?<=[.!?。！？])\s+", text):
        sentence = raw_sentence.strip()
        if not sentence:
            continue
        if "{evidence:" in sentence:
            continue
        if "[evidence missing:" in sentence:
            unsupported.append(sentence)
            continue
        has_number = bool(re.search(r"\d", sentence))
        has_claimy_word = bool(
            re.search(
                r"\b(cohort|stays|patients|mortality|death|auroc|auc|hazard|odds|risk|cluster|survival|ci|p=|calibration|brier)\b",
                sentence,
                flags=re.I,
            )
        )
        if has_number and has_claimy_word:
            unsupported.append(sentence)
    return unsupported


def _initial_reflection_memory(
    *, context: ResearchContext, semantics: ClinicalSemanticsResolution
) -> List[ReflectionMemoryEntry]:
    entries = [
        ReflectionMemoryEntry(
            category="reusable_template",
            summary=(
                f"Analysis family {semantics.analysis_family} selected for question: "
                f"{context.research_question}"
            ),
            analysis_family=semantics.analysis_family,
            recommendation="Prefer typed shared state and ICU semantic guardrails over free-form handoffs.",
        )
    ]
    for note in semantics.safety_guardrails[:5]:
        entries.append(
            ReflectionMemoryEntry(
                category="reusable_template",
                summary=f"ICU guardrail: {note}",
                analysis_family=semantics.analysis_family,
                recommendation="Carry this guardrail into planning, coding, and critique prompts.",
            )
        )
    return entries


def _empty_df_placeholder():
    import pandas as pd

    return pd.DataFrame()


__all__ = [
    "PlannerAgent",
    "ReplannerAgent",
    "ClinicalSemanticsAgent",
    "DataExtractionAgent",
    "StatisticalAnalysisAgent",
    "VisualizationAgent",
    "ManuscriptAgent",
    "CriticAgent",
    "RuntimeSupervisor",
    "CoderAgent",
    "AnalyzerAgent",
    "WriterAgent",
    "PROMPT_PACK_VERSION",
]
