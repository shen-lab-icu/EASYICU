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

Contract
--------
The 11 agent classes below expose typed entry points
(``run``, ``build_request`` + ``materialize``, ``review_step``,
``_call_section``) tailored to each agent's role. :mod:`pipeline` invokes
them by name with concrete typed arguments; mypy / IDEs can statically
check each call site. Add new agents in the same style — there is no
generic dispatch Protocol because nothing in the codebase consumes one.
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .analysis_types import (
    canonical_analysis_family,
    infer_analysis_type,
    locked_analysis_type_guide,
    planner_analysis_type_guide,
)
from .trajectory_contract import trajectory_phenotyping_code_contract
from .trajectory_plan_contract import (
    trajectory_planner_contract_guide,
    trajectory_role_code_contract,
)
from .method_capabilities import coder_method_capability_block
from .cohort_schema import ALLOWED_CTAS_AGGREGATIONS, known_concept_ids
from .icu_rules import (
    GENERAL_ICU_ANALYSIS_PRINCIPLES,
    VariableKind,
    default_time_windows,
)
from .llm import LLMClient, LLMMessage
from .plan_utils import effect_output_authorized
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
from .temporal_semantics import (
    ConceptValidationLayer,
    ICUEpisodeResolver,
    TemporalAlignmentEngine,
)


def _dump_raw(text: str, tag: str) -> Optional[Path]:
    """Best-effort save of an LLM response that failed to parse (T1.3).

    Creates ``research_output/llm_debug/<tag>_<timestamp>.txt`` with the
    full raw response. Silent on any IO failure so the debug aid never
    masks the underlying parse error.
    """
    try:
        log_dir = Path(
            os.environ.get("EASYICU_LLM_DEBUG_DIR") or "./research_output/llm_debug"
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

PLANNER_MAX_RETRIES = 4

_ICU_RULE_TO_CTAS_HINT = {
    AggregationRule.FIRST_VALUE.value: (
        "first",
        'ICU rule label "first_value" maps to CTAS aggregation "first".',
    ),
    AggregationRule.MAX_LAST.value: (
        "max",
        'ICU rule label "max_or_last" maps to CTAS aggregation "max" or '
        '"last"; prefer "max" for acuity scores.',
    ),
    AggregationRule.MEAN_MEDIAN.value: (
        "median",
        'ICU rule label "mean_or_median" maps to CTAS aggregation "mean" or '
        '"median"; prefer "median" for robustness checks.',
    ),
    AggregationRule.MEDIAN_ONLY.value: (
        "median",
        'ICU rule label "median_only" maps to CTAS aggregation "median".',
    ),
    AggregationRule.SUM.value: (
        "sum",
        'ICU rule label "sum" maps to CTAS aggregation "sum".',
    ),
    AggregationRule.ANY.value: (
        "any",
        'ICU rule label "any" maps to CTAS aggregation "any".',
    ),
    AggregationRule.NONE.value: (
        "first",
        'ICU rule label "none" is not a CTAS aggregation; use an explicit '
        'window and choose "first" only when filtering a point-in-time value.',
    ),
}


def _ctas_aggregation_hint(rule: Optional[AggregationRule]) -> str:
    """Return a CTAS-compatible aggregation hint for variable context."""

    if rule is None:
        return "any"
    ctas_value, note = _ICU_RULE_TO_CTAS_HINT.get(
        rule.value,
        ("any", f'Unrecognised ICU rule label "{rule.value}"; choose a CTAS enum.'),
    )
    return f"{ctas_value} (icu_rule={rule.value}; {note})"


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
    obs = _format_observed_domain(v.observed_domain)
    trajectory = ""
    if v.fixed_window_trajectory is not None:
        metadata = v.fixed_window_trajectory
        trajectory = (
            f" trajectory_family={metadata.family}"
            f" time_bin=[{metadata.window_start_hours:g},{metadata.window_end_hours:g})h"
            f" source_scale={metadata.source_scale}"
            f" representation={metadata.representation_kind}"
            f" anchor={metadata.anchor or 'unspecified_agent_must_declare'}"
        )
    return (
        f"- {v.name} | role={v.role.value} dtype={v.dtype}{unit}{rng}{obs}"
        f" agg_default={_ctas_aggregation_hint(v.aggregation_default)}"
        f"{trajectory}{miss}{pit}"
    )


def _format_observed_domain(domain: Optional[Dict[str, Any]]) -> str:
    """Render the cohort-observed value domain as a compact, fact-only hint.

    Surfaces ``is_binary`` / ``is_constant`` and the observed [min, max] so the
    planner interprets a column by its real values, not its name (a
    ``<score>_max`` column observed binary {0,1} must not be thresholded as a
    0-24 scale). States facts only — never prescribes a derivation.
    """
    if not domain:
        return ""
    if domain.get("is_constant"):
        return " observed=CONSTANT(single value; no variation to model)"
    if domain.get("is_binary"):
        return (
            " observed={0,1} BINARY(already 2-level; a numeric cutoff >1 is degenerate)"
        )
    levels = domain.get("levels")
    if levels:
        shown = ",".join(levels[:6])
        more = "…" if len(levels) > 6 else ""
        return f" observed_levels={{{shown}{more}}}(categorical; encode as-is)"
    lo, hi = domain.get("min"), domain.get("max")
    n_unique = domain.get("n_unique")
    if lo is not None and hi is not None:
        return f" observed=[{lo:g},{hi:g}] n_unique={n_unique}"
    if n_unique is not None:
        return f" observed_n_unique={n_unique}"
    return ""


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
    if ctx.primary_exposure:
        lines.append(
            f"Primary exposure/predictor: {ctx.primary_exposure} "
            "(authoritative; related representations are secondary unless the "
            "study contract explicitly replaces it)"
        )
    lines.append("Time windows:")
    for w in ctx.time_windows:
        lines.append(f"  - {w.name}: {w.start_hours}-{w.end_hours}h from {w.anchor}")
    lines.append("Variables:")
    for v in ctx.variables:
        lines.append(_format_variable(v))
    if ctx.cross_database_validation:
        lines.append(
            "Cross-database replication planned: "
            + ", ".join(ctx.cross_database_validation)
        )
    if ctx.user_preferences is not None:
        prefs = ctx.user_preferences
        lines.append("User preferences:")
        if prefs.inferred_analysis_family:
            lines.append(
                f"  - inferred_analysis_family: {prefs.inferred_analysis_family}"
            )
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
    # Variable-type method-compatibility self-review checklist (Patch B):
    # derived from ctx.variables via the generic compatibility matrix in
    # method_compatibility.py. Appended once so every agent role
    # (planner / coder / analyzer / writer) sees the same up-front
    # constraints and the matrix is the single source of truth.
    from .method_compatibility import render_variable_constraints

    constraints = render_variable_constraints(ctx)
    if constraints:
        lines.append("")
        lines.append(constraints)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------


def _format_concept_id_allowlist() -> str:
    """Render legal EasyICU concept ids for CTAS planner prompts."""

    concept_ids = sorted(known_concept_ids())
    if not concept_ids:
        return (
            "ALLOWED concept_ids — no concept dictionary entries were loaded. "
            "Do not invent concept_id values; ask for a configured concept "
            "dictionary before emitting a CohortDefinition."
        )
    lines = [
        "ALLOWED concept_ids — the ONLY values acceptable in any "
        "CohortDefinition or RobustnessSpec.cohort_override.concept_id field. "
        'Synthesizing new names (e.g. "score_at_admission", '
        '"concept_peak_window", "condition_onset_window") is forbidden — these '
        "are operationalizations, not concepts. To operationalize a concept "
        "over a time window, use the 5-tuple form: "
        'concept_id="<one from the list below>" + time_window + aggregation.',
        "",
    ]
    lines.extend(f"- `{concept_id}`" for concept_id in concept_ids)
    return "\n".join(lines)


def _format_ctas_schema_constraints() -> str:
    """Render CTAS enum and cross-field constraints for planner prompts."""

    enum_values = ", ".join(f'"{value}"' for value in ALLOWED_CTAS_AGGREGATIONS)
    return (
        "CTAS SCHEMA CONSTRAINTS — you MUST satisfy ALL of these in every "
        "ConceptPredicate, including those inside "
        "robustness_specs[*].cohort_override:\n\n"
        "1. aggregation MUST be exactly one of these 10 values (case-sensitive, "
        f"no synonyms): {enum_values}. Do NOT use ICU rule labels or synonyms "
        'such as "first_value", "max_or_last", "mean_or_median", '
        '"mean_median", "median_only", "latest", "most_recent", '
        '"earliest", "average", or "total" as aggregation values; these '
        "are not CTAS enum values and will be rejected.\n\n"
        "2. time_window.end_offset_hours MUST be strictly greater than "
        "time_window.start_offset_hours. Zero-width windows (end == start) "
        "are invalid. If you want a single instant, use a small window like "
        "[0h, 1h] instead.\n\n"
        '3. aggregation "any" / "all" can only be paired with op in '
        '{"==", "!=", "missing", "not_missing"}; they yield '
        "booleans, not numeric thresholds.\n\n"
        "4. If the ResearchContext shows an ICU rule label in an "
        "`agg_default=... (icu_rule=...)` annotation, translate it before "
        'writing CTAS JSON: first_value -> "first"; max_or_last -> '
        '"max" or "last" (prefer "max" for acuity scores); '
        'mean_or_median / mean_median -> "mean" or "median" (prefer '
        '"median" for robustness checks); median_only -> "median".'
    )


def _build_planner_user_prompt(context: ResearchContext) -> str:
    """Build the planner user prompt with runtime concept-id grounding."""

    return (
        "Produce an ICU-AWARE RESEARCH PLAN as JSON matching the "
        "AnalysisPlan schema. First infer the EHR analysis type, "
        "then choose only the steps justified by that family and "
        "the available context. The plan must not assume that "
        "every task needs Table 1, outcome incidence, missingness, "
        "or a primary association model. That said, a baseline "
        "characteristics table (Table 1) IS a reporting standard for "
        "observational/association and prediction-model families "
        "(STROBE item 14 / TRIPOD): for those families include a "
        "baseline characteristics step (e.g. expected_outputs "
        "['table:table_one']) describing the analytic cohort before "
        "the primary analysis. Omit it only when the family genuinely "
        "does not call for one (e.g. a pure feasibility/protocol task, "
        "or a clustering task whose per-cluster characteristics table "
        "already carries the descriptive reporting). If cross-database "
        "replication is requested, include a cross-database step, "
        "but mark it as a feasibility / protocol step unless the "
        "ResearchContext explicitly provides external cohort files. "
        "Use score-specific QC steps only when a relevant score is "
        "actually central to the question. Do not put invented "
        "prefixed variables such as eicu:age in `inputs`. Honor "
        "explicit user preferences and requested outputs when they "
        "are compatible with the cohort and analysis family.\n"
        "Choose step boundaries that make the analysis reviewable. A later step "
        "may consume an earlier standardized artifact only when that dependency "
        "is explicit in `inputs` and the producer declares it in "
        "`expected_outputs`; never rely on hidden in-memory state. Do not force "
        "prediction, clustering, or any other family into a hard-coded mega-step "
        "or split it solely to fit a shared pipeline template.\n\n"
        "The typed `model_requirements` roster currently covers only a complex "
        "binary/continuous adjusted-association step whose method is exactly "
        "`adjusted_association_models` and whose expected outputs include "
        "`table:adjusted_association_estimates`. For that supported contract, "
        "record each pre-specified estimand/model in the roster instead of "
        "leaving the scientific decision only in prose. Each "
        "entry has `requirement_id`, `outcome`, `outcome_type` (binary or "
        "continuous), `method_family`, `exposure_source`, `analysis_role` "
        "(primary, secondary, or sensitivity), `analysis_set` (source_aware or "
        "complete_case), and `required_for_step_success`. You decide this roster; "
        "the execution layer only verifies it. `method_family` must be a binary "
        "logistic family or a continuous linear/quantile family matching "
        "`outcome_type`. Primary and secondary entries must be required for step "
        "success; only a sensitivity entry may be optional. Leave the array empty "
        "for survival, prediction, "
        "mixed-effects, clustering, and every other analysis family; those use "
        "their own family-specific planning and validation contracts.\n\n"
        + _format_concept_id_allowlist()
        + "\n\n"
        + _format_ctas_schema_constraints()
        + "\n\n"
        "Every cohort/exposure/outcome concept used to define the "
        "analysis population must be represented as a typed cohort "
        "definition: concept_id, time_window, aggregation, operator, "
        'and value. You may write `cohort: {"from_named": "..."}` '
        "only when the caller has explicitly registered that named "
        "pattern for this case; otherwise supply the full five-tuple "
        "predicate. Free-text cohort strings are invalid.\n"
        "If your plan includes any cohort-definition, eligibility, or "
        "attrition step, `cohort.inclusion` MUST contain at least one "
        "structured predicate (plus any `exclusion`). An empty cohort there is "
        "rejected: 纳排 expressed only in step prose cannot be enforced or "
        "audited, and the analysis would silently run on the full universe.\n\n"
        "Pre-specify robustness variants before execution. Add a "
        "`robustness_specs` array with at least 3 cohort-axis, "
        "2 missingness-axis, and 2 outcome-axis alternatives. "
        "These are advisory execution specifications: do not use "
        "them to change the primary analysis, and do not describe "
        "their results as surprising or unexpected.\n\n"
        + locked_analysis_type_guide(infer_analysis_type(context))
        + "\n\n"
        + planner_analysis_type_guide()
        + "\n\n"
        + trajectory_planner_contract_guide(
            context=context,
            analysis_type=infer_analysis_type(context).key,
        )
        + "\n\n"
        "OUTPUT FORMAT — VERY IMPORTANT:\n"
        "Return *only* a single JSON object matching the "
        "AnalysisPlan schema. No prose, no markdown headings, no "
        "trailing commentary. A ```json … ``` fence is acceptable; "
        "anything outside that fence will be discarded.\n\n"
        "Required JSON shape (truncated example):\n"
        "The example values are illustrative only; do not prefer SOFA or "
        "any example concept unless the ResearchContext supports it.\n"
        "{\n"
        '  "research_question": "<copy from context>",\n'
        '  "cohort": {\n'
        '    "name": "primary",\n'
        '    "inclusion": [\n'
        "      {\n"
        '        "concept_id": "<easyicu concept id>",\n'
        '        "time_window": {"anchor": "icu_admit", "start_offset_hours": 0, "end_offset_hours": 24},\n'
        '        "aggregation": "max",\n'
        '        "op": ">=",\n'
        '        "value": 1\n'
        "      }\n"
        "    ],\n"
        '    "exclusion": []\n'
        "  },\n"
        '  "steps": [\n'
        "    {\n"
        '      "step_id": "01_table_one",\n'
        '      "intent": "<one sentence>",\n'
        '      "inputs": ["<variable names from context>"],\n'
        '      "expected_outputs": ["table:table_one"],\n'
        '      "method": "descriptive",\n'
        '      "icu_rule_refs": ["aggregation_rule_for"],\n'
        '      "model_requirements": [],\n'
        '      "trajectory_stability_spec": null\n'
        "    }\n"
        "  ],\n"
        '  "robustness_specs": [\n'
        "    {\n"
        '      "spec_id": "alt_cohort_max_during_stay",\n'
        '      "axis": "cohort",\n'
        '      "description": "Use stay-level max SOFA instead of admission-window max.",\n'
        '      "cohort_override": {\n'
        '        "name": "alt_max_stay",\n'
        '        "inclusion": [\n'
        "          {\n"
        '            "concept_id": "sofa",\n'
        '            "time_window": {"anchor": "icu_admit", "start_offset_hours": 0, "end_offset_hours": 168},\n'
        '            "aggregation": "max",\n'
        '            "op": ">=",\n'
        '            "value": 2\n'
        "          }\n"
        "        ],\n"
        '        "exclusion": []\n'
        "      },\n"
        '      "missing_override": null,\n'
        '      "outcome_override": null\n'
        "    },\n"
        "    {\n"
        '      "spec_id": "alt_missing_complete_case",\n'
        '      "axis": "missing",\n'
        '      "description": "Use complete-case handling for required variables.",\n'
        '      "cohort_override": null,\n'
        '      "missing_override": {"strategy": "complete_case"},\n'
        '      "outcome_override": null\n'
        "    }\n"
        "  ],\n"
        '  "rationale": "<one paragraph>"\n'
        "}\n\n"
        "RESEARCH CONTEXT:\n" + _format_context(context)
    )


def _render_methodological_principles() -> str:
    """Render the cross-cutting ICU principles for injection into a prompt.

    Faithful to the impartiality contract on :class:`MethodologicalPrinciple`:
    ``error`` principles are objective mistakes the plan must avoid, while
    ``caution`` principles are defensible analytical choices the planner must
    surface and justify but never have imposed on it. Case-neutral by
    construction — the principle layer hard-codes no benchmark task, variable,
    score or database.
    """
    errors = [p for p in GENERAL_ICU_ANALYSIS_PRINCIPLES if p.kind == "error"]
    cautions = [p for p in GENERAL_ICU_ANALYSIS_PRINCIPLES if p.kind == "caution"]
    lines = [
        "\n\nCROSS-CUTTING ICU METHODOLOGY (case-neutral; apply when planning):",
        "Objective errors to avoid — wrong under any study design:",
    ]
    lines.extend(f"- [{p.phase}] {p.principle}" for p in errors)
    lines.append(
        "Defensible choices — state and justify in the plan; do not let them "
        "pass silently, but the analyst, not these rules, decides:"
    )
    lines.extend(f"- [{p.phase}] {p.principle}" for p in cautions)
    return "\n".join(lines)


# Rendered once: the principle layer is static. Injected into the planner
# system message so the (previously unused) principles actually steer the plan.
_PRINCIPLES_GUIDE = _render_methodological_principles()


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
            LLMMessage(role="system", content=_SYSTEM_GUIDE + _PRINCIPLES_GUIDE),
            LLMMessage(role="user", content=_build_planner_user_prompt(context)),
        ]
        from .structured_retry import call_llm_with_structured_retry

        return call_llm_with_structured_retry(
            self.llm,
            messages,
            parser=lambda raw: self._parse(raw, context),
            role="planner",
            max_retries=PLANNER_MAX_RETRIES,
            max_tokens=4096,
            temperature=0.2,
            format_reminder=(
                "The JSON must be a single object with keys: "
                "research_question (string), cohort (object or null), "
                "steps (array of objects "
                "each with step_id, intent, inputs, expected_outputs, "
                "method, icu_rule_refs, optional model_requirements, and optional "
                "trajectory_stability_spec), "
                "rationale (string). "
                "All string values must be plain ASCII or UTF-8 quoted strings; "
                "do not use special Unicode whitespace inside values."
            ),
        )

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
        plan = AnalysisPlan.model_validate(data)
        # Family inference is a planner hint, not execution authority. Preserve
        # a valid agent-selected family (and its rationale); only fill the field
        # when the agent omitted it. A non-empty declaration is nevertheless a
        # closed execution contract: typos/novel labels must trigger the
        # structured retry loop instead of bypassing family-specific checks.
        declared_family = str(plan.analysis_type or "").strip()
        if not declared_family:
            plan.analysis_type = infer_analysis_type(context).key
        else:
            canonical_family = canonical_analysis_family(declared_family)
            if canonical_family is None:
                raise ValueError(
                    "Unknown analysis_type declaration "
                    f"{declared_family!r}; choose a key from the analysis-type "
                    "catalog instead of inventing or misspelling a family"
                )
            plan.analysis_type = canonical_family
        return plan


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
# Small-context local engines (glm / qwen / deepseek, see ``llm.py``) overflow
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
    slimmed = [_slim_record_for_replanner(r) for r in records]
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
                    + trajectory_planner_contract_guide(
                        context=context,
                        analysis_type=current_plan.analysis_type,
                    )
                    + "\n\n"
                    f"CURRENT PLAN:\n{current_plan.model_dump_json(indent=2)}\n\n"
                    f"PROBE SUMMARY:\n{_clip_json(probe_summary or {}, char_budget=_REPLANNER_PROBE_CHAR_BUDGET)}\n\n"
                    f"COMPLETED STEP RECORDS:\n{json.dumps(completed, ensure_ascii=False, default=str)}\n\n"
                    "RESEARCH CONTEXT:\n" + _format_context(context)
                ),
            ),
        ]
        from .structured_retry import call_llm_with_structured_retry

        revised = call_llm_with_structured_retry(
            self.llm,
            messages,
            parser=lambda raw: self._parse(raw, context),
            role="replanner",
            max_retries=2,
            max_tokens=4096,
            temperature=0.1,
            format_reminder=(
                "The JSON must be a single AnalysisPlan object with keys: "
                "research_question, steps, rationale. Keep completed step_ids "
                "from the CURRENT PLAN unchanged; only revise the remaining steps."
            ),
        )
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
            timing_and_design=(
                context.user_preferences.timing_and_design
                if context.user_preferences
                else None
            ),
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
        if (
            not constraints
            and context.user_preferences
            and context.user_preferences.timing_and_design
        ):
            ambiguity_notes.append(
                "Timing/design preferences were provided but no deterministic temporal constraint could be parsed."
            )
        safety_guardrails = sorted(
            {
                caveat
                for variable in context.variables
                for caveat in (
                    [
                        *(variable.pitfalls or []),
                        *(variable.forbidden_transformations or []),
                    ]
                )
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

    def build_request(
        self, *, context: ResearchContext, semantics: ClinicalSemanticsResolution
    ) -> DataExtractionRequest:
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
        titles = [
            ref.description or ref.evidence_id
            for ref in evidence_refs
            if ref.kind == "figure"
        ]
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
        evidence_digest: Optional[str] = None,
    ) -> str:
        return WriterAgent(self.llm, language=self.language).run(
            context=context,
            evidence_ids=evidence_ids,
            evidence_digest=evidence_digest,
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
            suggested_repairs=(
                []
                if status == "pass"
                else _suggest_repairs_for(step_summary, concerns)
            ),
            related_evidence_refs=list(evidence_refs),
        )

    def review_manuscript(
        self,
        *,
        scaffold: str,
        available_evidence_ids: Sequence[str],
    ) -> CritiqueReport:
        missing = sorted(
            set(
                re.findall(
                    r"(?:\[evidence missing:\s*([^\]]+)\]|<!--\s*evidence missing:\s*([^>]+)-->)",
                    scaffold,
                    flags=re.I,
                )
            )
        )
        missing = sorted(
            {
                (first or second).strip()
                for first, second in missing
                if (first or second).strip()
            }
        )
        concerns: List[str] = []
        if missing:
            concerns.append("Manuscript contains unresolved evidence placeholders.")
        unsupported = _sentences_missing_evidence_tokens(scaffold)
        if unsupported:
            concerns.append(
                "Some result-like sentences were filtered or remain unsupported."
            )
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
            suggested_repairs=(
                []
                if status == "pass"
                else [
                    "Ensure every quantitative result sentence cites a valid {evidence:<id>} placeholder.",
                    "Regenerate unsupported narrative from registered evidence artifacts only.",
                ]
            ),
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

    def bootstrap_state(
        self, *, run_id: str, context: ResearchContext
    ) -> AgentRuntimeState:
        semantics = self.clinical_semantics.run(context=context)
        extraction_request = self.data_extraction.build_request(
            context=context, semantics=semantics
        )
        extraction_result = self.data_extraction.materialize(
            context=context, request=extraction_request
        )
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
        visualization_request = None
        if any(
            str(item or "").strip().lower().startswith("figure:")
            for item in step.expected_outputs
        ):
            visualization_request = self.visualization.build_request(
                context=context,
                semantics=state.semantics
                or self.clinical_semantics.run(context=context),
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
            step=state.current_step
            or AnalysisStep(step_id="unknown", intent="unknown"),
            step_summary=step_summary,
            evidence_refs=evidence_refs,
            findings=findings,
        )
        analysis_result = self.statistical_analysis.summarize_result(
            step=state.current_step
            or AnalysisStep(step_id="unknown", intent="unknown"),
            step_summary=step_summary,
            evidence_refs=evidence_refs,
            validator_messages=findings,
            analysis_family=state.analysis_family or "unknown",
        )
        visualization_result = self.visualization.summarize_result(
            step=state.current_step
            or AnalysisStep(step_id="unknown", intent="unknown"),
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


#: Pre-execution forbidden-pattern check is capped at this many repair
#: rounds so a stubborn LLM can't burn token budget indefinitely. If the
#: code still violates the matrix after this many repairs, return the
#: last attempt unchanged — the post-hoc validator in
#: ``audits/patterns.py`` will halt the run with the same warning and
#: the violation chain is captured in the audit trail.
_MAX_PRE_EXEC_COMPATIBILITY_REPAIRS = 2


# Output-token budget for analysis-script generation and repair. A full
# robustness step (multiple model fits + an AIC/CI summary block) can exceed
# 4096 output tokens with a verbose model and truncate analysis.py mid-expression.
# 8192 roughly doubles the headroom. If truncation recurs at this cap, add a
# finish_reason=="length" continuation rather than raising it blindly.
_CODER_MAX_TOKENS = 8192


def _declared_output_scope_contract(step: AnalysisStep) -> str:
    """Keep code generation inside the plan's typed product boundary.

    Figure outputs are split into rendering-only steps before execution.  A
    science step that redraws them anyway duplicates work and creates a second,
    undeclared evidence owner.  Required runtime metadata and source-data
    companions remain allowed; only undeclared scientific products are barred.
    """

    outputs = [str(item or "").strip() for item in step.expected_outputs]
    has_figure = any(item.lower().startswith("figure:") for item in outputs)
    effect_authorized = effect_output_authorized(step)
    lines = [
        "DECLARED OUTPUT SCOPE (binding):",
        "- Create only the scientific products named in Expected outputs, plus "
        "required step_summary.json and necessary source-data or diagnostic companions.",
        f"- effect_output_authorized: {str(effect_authorized).lower()}.",
        "- The planner-owned Method and Expected outputs are binding. Typed model "
        "requirements are part of the method contract. The inferred analysis family "
        "is context only and cannot authorize additional scientific products.",
    ]
    if effect_authorized:
        lines.append(
            "- Effect authorization does not widen scope: emit effect estimates or "
            "contrasts only when their exact scientific product is named in Expected outputs."
        )
    else:
        lines.extend(
            [
                "- effect_output_authorized=false: do not add reference-group contrasts, "
                "risk ratios (RR), odds ratios (OR), hazard ratios (HR), risk differences "
                "(RD), model coefficients, or interactions to declared table columns, "
                "nested step_summary fields, or output registries; likewise do not add "
                "p-values for any such undeclared effect contrast or interaction.",
                "- Descriptive counts, denominators, rates, absolute summaries, and "
                "uncertainty intervals for those same descriptive summaries remain allowed "
                "when they are inside the declared product scope.",
            ]
        )
    if has_figure:
        lines.append(
            "- Figure rendering is allowed only for the explicitly declared figure products."
        )
    else:
        lines.append(
            "- This step declares no figure product. Do not render, save, or register "
            "figures; leave presentation to a separately declared figure step."
        )
    return "\n".join(lines) + "\n"


def _typed_input_scope_contract(step: AnalysisStep) -> str:
    """Bind planned upstream products to their run-authoritative files."""

    supported_kinds = {
        "artifact",
        "dataset",
        "figure",
        "log",
        "manifest",
        "model",
        "statistic",
        "table",
    }
    typed_inputs = []
    for item in step.inputs or []:
        kind, separator, product = str(item or "").strip().partition(":")
        if separator and kind.strip().lower() in supported_kinds and product.strip():
            typed_inputs.append(str(item))
    if not typed_inputs:
        return ""
    return (
        "TYPED INPUT BINDING (binding):\n"
        "- This step has typed upstream inputs. At instrumented execution, read "
        "the JSON manifest at os.environ['EASYICU_RESOLVED_INPUTS_JSON'].\n"
        "- Look up each typed input by its exact kind:name key in manifest['inputs']. "
        "Read its exact file as Path(os.environ['EASYICU_RUN_DIR']) / "
        "binding['relative_path']; the manifest also supplies evidence_id and sha256.\n"
        "- Do not glob EASYICU_EVIDENCE_DIR, choose a file by mtime or basename, "
        "follow a legacy alias, or reconstruct a declared upstream product from "
        "COHORT_PARQUET. COHORT_PARQUET remains the source only for untyped raw "
        "variables or steps with no typed upstream input.\n"
        "- In step_summary.json, record one input_bindings row per typed input "
        "that the script attempts to consume. Copy its exact input_key, "
        "evidence_id, and sha256; report loaded as a boolean and, for each loaded "
        "tabular input, its row_count. Do not duplicate a contradictory loaded "
        "or row-count claim elsewhere in the summary.\n"
        "- If a block claims status='checked' for a subset reconciliation between "
        "two typed tables, name the exact reference_artifact and subset_artifact "
        "input keys, non-empty key_columns, every shared non-key column under "
        "value_columns_checked, and value_mismatch_n=0 only after actually "
        "comparing them. The host repeats that key-and-value comparison. If it "
        "was not performed, do not call the reconciliation checked.\n"
        f"- Exact typed inputs for this step: {typed_inputs}\n"
    )


class CoderAgent:
    """Generates a self-contained Python analysis script for one step.

    Patch C (2026-05-25) added a post-codegen, pre-execution
    forbidden-pattern check: ``run`` scans the freshly written script
    for matrix violations (e.g. ``MiniBatchKMeans`` over an ordinal
    SOFA component) and, when a violation is detected, automatically
    invokes ``repair`` with a structured error message. This makes
    the agent layer the *first* line of defence; the existing
    post-hoc validator in ``audits/patterns.py`` remains the second.
    """

    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm
        self.last_compatibility_violations: List[Dict[str, object]] = []
        self.last_compatibility_repair_attempts: int = 0

    def run(self, *, context: ResearchContext, step: AnalysisStep) -> str:
        from .method_compatibility import (
            detect_forbidden_pattern_usage,
            format_violation_message,
        )

        _family = infer_analysis_type(context)
        messages = [
            LLMMessage(role="system", content=_SYSTEM_GUIDE + _CODER_GUIDE),
            LLMMessage(
                role="user",
                content=(
                    f"Write the Python CODE for STEP {step.step_id}.\n"
                    f"Analysis-family context: {_family.key} ({_family.name}). "
                    "Use this only to reject method-incompatible substitutions. "
                    "Execute the planner-owned Method and Expected outputs below; "
                    "the family label does not authorize another method, estimand, "
                    "figure, or scientific product.\n"
                    f"Step intent: {step.intent}\n"
                    f"Step inputs: {step.inputs}\n"
                    f"Expected outputs: {step.expected_outputs}\n"
                    "Model requirements: "
                    f"{json.dumps([item.model_dump(mode='json') for item in step.model_requirements], ensure_ascii=False)}\n"
                    f"Method: {step.method or '(unspecified — choose conservatively)'}\n\n"
                    + _declared_output_scope_contract(step)
                    + _typed_input_scope_contract(step)
                    + coder_method_capability_block()
                    + trajectory_phenotyping_code_contract(
                        context=context,
                        step=step,
                    )
                    + trajectory_role_code_contract(context=context, step=step)
                    + "\n\n"
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
        raw = self.llm.complete(messages, max_tokens=_CODER_MAX_TOKENS, temperature=0.1)
        code = _strip_code_fence(raw.strip())

        # Patch C: post-codegen pre-execution compatibility enforcement.
        # Loops up to _MAX_PRE_EXEC_COMPATIBILITY_REPAIRS times; each
        # iteration that still violates the matrix invokes the existing
        # ``repair`` pathway with the violation message as the
        # ``run_log`` field, so the coder LLM gets a structured error
        # in the same shape it already understands. After the budget
        # is exhausted the bad code is returned unchanged so the
        # post-hoc validator can record the issue in the audit trail.
        self.last_compatibility_repair_attempts = 0
        for attempt in range(1, _MAX_PRE_EXEC_COMPATIBILITY_REPAIRS + 1):
            violations = detect_forbidden_pattern_usage(code, context, step)
            self.last_compatibility_violations = violations
            if not violations:
                break
            err = format_violation_message(violations)
            self.last_compatibility_repair_attempts = attempt
            code = self.repair(
                context=context,
                step=step,
                code=code,
                run_log=err,
                attempt=attempt,
            )
        return code

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
        family = infer_analysis_type(context)
        repair_specialization = _repair_specialization(run_log)
        messages = [
            LLMMessage(role="system", content=_SYSTEM_GUIDE + _CODER_GUIDE),
            LLMMessage(
                role="user",
                content=(
                    f"REPAIR THE PYTHON CODE FOR STEP {step.step_id}.\n"
                    f"Analysis-family context: {family.key} ({family.name}). "
                    "Use this only for method-compatibility checks. Preserve the "
                    "planner-owned Method and Expected outputs; the family label "
                    "cannot add or replace a scientific product.\n"
                    f"Repair attempt: {attempt}\n"
                    f"Step intent: {step.intent}\n"
                    f"Step inputs: {step.inputs}\n"
                    f"Expected outputs: {step.expected_outputs}\n"
                    "Model requirements: "
                    f"{json.dumps([item.model_dump(mode='json') for item in step.model_requirements], ensure_ascii=False)}\n"
                    f"Method: {step.method or '(unspecified)'}\n\n"
                    + _declared_output_scope_contract(step)
                    + _typed_input_scope_contract(step)
                    + trajectory_phenotyping_code_contract(
                        context=context,
                        step=step,
                    )
                    + trajectory_role_code_contract(context=context, step=step)
                    + "\n\n"
                    "The previous script failed at execution time. Return "
                    "only a complete replacement Python script that follows "
                    "the original code contract and writes the same expected "
                    "artefacts when possible. Make the smallest robust fix; "
                    "do not add prose, markdown, or an explanation. Keep "
                    "honoring explicit user preferences recorded in the "
                    "ResearchContext.\n\n"
                    "REPAIR CHECKLIST:\n"
                    + repair_specialization
                    + "- IMPORTS: "
                    + coder_method_capability_block()
                    + "\n"
                    "  **There is NO `easyicu.research_agent.rcs` / "
                    "`.metrics` / `.utils` etc.** — the analysis script runs in a "
                    "sandbox with a closed import contract. A method-specific "
                    "code contract above may explicitly name documented "
                    "`easyicu.research_agent.methods.*` modules; only those exact "
                    "modules and symbols are allowed for that method. All other "
                    "`easyicu.*` imports and undocumented project-local modules "
                    "remain forbidden. If you need restricted cubic splines and "
                    "no documented helper was named, use `patsy` or `numpy`; if "
                    "you need calibration, import from `sklearn.calibration`.\n"
                    "- If this is an association or prediction step, keep every named "
                    "primary predictor/exposure in the fitted design matrix.\n"
                    "- If you read `result.params[name]`, `result.conf_int().loc[name]`, "
                    "or `result.pvalues[name]`, ensure `name` is present in `X.columns` "
                    "before fitting.\n"
                    "- If categorical variables are dummy-encoded, rebuild `x_cols` after "
                    "encoding and include the primary predictor plus dummy columns.\n"
                    "- Do not numeric-coerce string categorical variables such as `sex` "
                    "before dummy encoding; this converts all categories to NaN. After "
                    "dummy encoding, drop missing rows using the rebuilt `[outcome] + x_cols` "
                    "list, not the old covariate list containing removed categorical columns.\n"
                    "- If this is a prediction step, use a scikit-learn Pipeline or an "
                    "equivalent leak-free split/CV workflow and write AUROC plus Brier or "
                    "calibration metrics to step_summary.json. Keep categorical columns "
                    "as objects until a ColumnTransformer categorical branch handles them; "
                    "do not coerce mixed numeric + categorical feature frames all at once. "
                    "If you use `calibration_curve`, import it from `sklearn.calibration`, "
                    "not `sklearn.metrics`. "
                    "Never use `('onehot', 'passthrough')` for a categorical branch feeding "
                    "a numeric estimator; import OneHotEncoder and use "
                    "`OneHotEncoder(handle_unknown='ignore', sparse_output=False)`.\n"
                    "- If this is a clustering step, impute/scale numeric features before "
                    "clustering and write cluster_count/n_clusters plus silhouette_score "
                    "when at least two clusters are present.\n"
                    "- If this is a table-one or descriptive step, rebuild the table as "
                    "flat row dictionaries with explicit continuous/categorical helper "
                    "functions. Do not infer table shape with `next(iter(...))`, and do "
                    "not assume binary category keys are int 1 or string '1'.\n"
                    "- Define `out_dir = os.environ['STEP_OUT_DIR']` before any model "
                    "fitting `try/except`; exception paths must still be able to write "
                    "step_summary.json and any diagnostic tables.\n"
                    "- Do not assign a local result variable the same name as a helper "
                    "function called in that scope (for example, never write "
                    "`audit = audit(...)`); Python treats that name as local and the "
                    "call can fail with UnboundLocalError. Use a distinct result name.\n"
                    "- Optional plotting or publication figure helpers must not make a "
                    "completed metric step fail. Write step_summary.json after computing "
                    "metrics, wrap figure-helper calls, and fall back to plain PNG/SVG if "
                    "a helper signature is wrong.\n"
                    "- If fitting cannot be completed, write a step_summary.json with null "
                    "numeric fields and a precise error, then exit normally. Do not `raise` "
                    "again after writing the failure summary.\n\n"
                    "PREVIOUS SCRIPT:\n```python\n" + code[-12000:] + "\n```\n\n"
                    "RUN LOG / TRACEBACK:\n```\n" + run_log[-8000:] + "\n```\n\n"
                    "RESEARCH CONTEXT:\n" + _format_context(context)
                ),
            ),
        ]
        raw = self.llm.complete(
            messages, max_tokens=_CODER_MAX_TOKENS, temperature=0.05
        )
        repaired = _strip_code_fence(raw.strip())
        if not _looks_like_python_script(repaired):
            raise ValueError(
                "Coder repair returned non-script output; refusing to replace "
                "the previous analysis script."
            )
        return repaired


def _repair_specialization(run_log: str) -> str:
    """Add a binding repair contract for a diagnosed method-suite failure.

    The trigger is category-level validator evidence, never a benchmark item,
    concept, variable, or figure name.  Scientific column selection remains in
    the failed Agent script; the helper only validates that declared choice.
    """

    normalized = re.sub(r"[^a-z0-9]+", " ", str(run_log).lower()).strip()
    sparse_event_signals = (
        "binary event reconciliation",
        "binary event presence",
        "sparse event triad",
        "count flag representative triad",
        "representative value",
        "reconcile binary event presence",
    )
    if not any(signal in normalized for signal in sparse_event_signals):
        return ""
    return (
        "- DIAGNOSED SPARSE-EVENT REPAIR (binding): import and call "
        "`easyicu.research_agent.methods.source_status."
        "reconcile_binary_event_presence` with the count, measured-flag, and "
        "representative columns already selected by the Agent. Use its "
        "`values`, `audit`, and `status_table` directly. Do not replace those "
        "columns, rebuild custom masks, silently numeric-coerce original "
        "representative values, or change the exposure, cohort, outcome, or "
        "model. The helper is the documented project-local import authorized "
        "for this diagnosed contract.\n"
    )


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
                    "in the form {evidence:<id>}.\n"
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
    """Produces a full manuscript by writing each section in a separate
    LLM call, then concatenating. This avoids the "lazy middle"
    problem where small models truncate Introduction / Discussion.

    Each section call gets:
    - the full research context (variables, cohort, question),
    - the machine evidence digest (numbers to cite),
    - the list of available evidence ids,
    - a section-specific instruction with word-count target.

    The downstream causal-audit + critic loop reject drafts that use
    causal language for associations or cite non-existent evidence ids.
    """

    def __init__(self, llm: LLMClient, *, language: str = "en") -> None:
        self.llm = llm
        lang = (language or "en").lower()
        self.language = "zh" if lang.startswith(("zh", "cn", "chinese")) else "en"

    def _call_section(
        self,
        *,
        section_name: str,
        instruction: str,
        context: ResearchContext,
        evidence_ids: Sequence[str],
        evidence_digest: Optional[str],
        max_tokens: int = 2048,
    ) -> str:
        lang_inst = _writer_language_instruction(self.language)
        evidence_list = (
            ", ".join(str(eid) for eid in evidence_ids) if evidence_ids else "(none)"
        )
        messages = [
            LLMMessage(role="system", content=_SYSTEM_GUIDE + _WRITER_GUIDE),
            LLMMessage(
                role="user",
                content=(
                    f"Write ONLY the **{section_name}** section of an ICU research "
                    "manuscript in markdown. Do NOT write any other section.\n\n"
                    f"{instruction}\n\n"
                    f"{lang_inst}\n\n"
                    "CITATION RULE:\n"
                    "- `{evidence:<id>}` is an inline citation (like a footnote number).\n"
                    "- Write the actual number in prose, then cite: "
                    "`mortality was 12% {evidence:outcome_rate}`.\n"
                    "- Use exactly single braces: `{evidence:<id>}`, not "
                    "`{{evidence:<id>}}`.\n"
                    "- Every body-text sentence that reports, interprets, compares, "
                    "or explains cohort composition, exposure prevalence, outcome "
                    "frequency, model estimates, sensitivity/robustness, missingness, "
                    "data quality, mechanisms, strengths, or limitations must include "
                    "at least one evidence citation.\n"
                    "- Keywords, Data/code availability, Funding, and Conflicts of "
                    "interest are manuscript metadata and do not need evidence citations.\n"
                    "- NEVER use a placeholder as a noun. If a number is unavailable, omit the sentence.\n"
                    f"- Only use ids from this list: {evidence_list}\n\n"
                    "OUTPUT DISCIPLINE:\n"
                    "- Output ONLY finished, publishable manuscript prose. Do NOT include "
                    "your reasoning, planning, working notes, or meta-commentary about the "
                    "evidence digest — e.g. no lines such as 'First, extract ...', 'Let me "
                    "...', 'Actually, X says ...', a trailing 'no number?', or bullet lists "
                    "that restate the digest.\n"
                    "- If you cannot support a sentence from the listed evidence, omit it "
                    "silently; do not narrate the gap.\n\n"
                    "LANGUAGE POLICY:\n"
                    "- Use ONLY associational phrasing. Forbidden: 'caused by', 'causal', "
                    "'attributable to', 'effect of', 'due to', 'leads to', 'drives'.\n"
                    "- Allowed: 'was associated with', 'correlated with', 'observed alongside', "
                    "'consistent with', 'may reflect'.\n\n"
                    "SECTION-SPECIFIC LENGTH TARGET:\n"
                    f"- {section_name}: follow the requested length and paragraph structure exactly.\n\n"
                    "MACHINE EVIDENCE DIGEST:\n"
                    + (evidence_digest or "(none)")
                    + "\n\nRESEARCH CONTEXT:\n"
                    + _format_context(context)
                ),
            ),
        ]
        raw = self.llm.complete(
            messages, max_tokens=max_tokens, temperature=0.3
        ).strip()
        return _strip_code_fence(raw)

    def run(
        self,
        *,
        context: ResearchContext,
        evidence_ids: Sequence[str],
        evidence_digest: Optional[str] = None,
    ) -> str:
        common = dict(
            context=context,
            evidence_ids=evidence_ids,
            evidence_digest=evidence_digest,
        )

        # The eight manuscript sections are independent: each _call_section is
        # built only from `common`, and no section's text feeds another's prompt.
        # They are the single largest LLM-latency contributor in a run (8
        # sequential LLM calls). Issue them concurrently and reassemble in the
        # fixed manuscript order below — output stays order-deterministic; only
        # wall-clock changes (8 sequential calls -> ~1). _call_section mutates no
        # shared state and self.llm.complete is safe under concurrent requests.
        from concurrent.futures import ThreadPoolExecutor

        _ex = ThreadPoolExecutor(max_workers=8)

        _f_title = _ex.submit(
            self._call_section,
            section_name="Title and Keywords",
            instruction=(
                "Write:\n"
                "1. `# <title>` — 12-20 words, include study design + cohort + primary finding direction.\n"
                "2. On the next line: `**Keywords:** keyword1, keyword2, ...` (5-7 keywords).\n"
                "Nothing else."
            ),
            max_tokens=256,
            **common,
        )

        _f_abstract = _ex.submit(
            self._call_section,
            section_name="Abstract",
            instruction=(
                "Write `## Abstract` with four labelled paragraphs:\n"
                "- **Background:** 2-3 sentences (clinical importance, knowledge gap).\n"
                "- **Methods:** 3-4 sentences (cohort, design, primary analysis, ICU-aware aggregation).\n"
                "- **Results:** 4-5 sentences (N, outcome incidence, primary effect size with 95% CI and p, one supporting finding).\n"
                "- **Conclusions:** 1-2 sentences (associational phrasing only, call for validation).\n"
                "Target: 200-300 words total."
            ),
            max_tokens=1024,
            **common,
        )

        _f_introduction = _ex.submit(
            self._call_section,
            section_name="Introduction",
            instruction=(
                "Write `## Introduction` with 4-5 paragraphs (900-1200 words total):\n"
                "- Para 1: Clinical importance of the ICU question and why it matters now.\n"
                "- Para 2: Prior evidence on the key predictor / score / exposure. Use evidence ids from the digest when possible and cite literature if available.\n"
                "- Para 3: What prior studies did well, and where they still leave uncertainty.\n"
                "- Para 4: The specific gap in the literature that this study addresses.\n"
                "- Para 5: One sentence on the objective, one sentence on the hypothesis, and one sentence on the expected contribution.\n"
                "Requirements: write full prose (no bullets), avoid generic filler, and include at least one evidence citation or literature citation in each paragraph when evidence is available. Do not collapse the introduction into two sentences."
            ),
            max_tokens=4096,
            **common,
        )

        _f_methods = _ex.submit(
            self._call_section,
            section_name="Methods",
            instruction=(
                "Write `## Methods` with sub-sections:\n"
                "### Study design and cohort\n"
                "  Database, setting (ICU type), inclusion/exclusion criteria, time period.\n"
                "### Variables\n"
                "  Primary predictor, outcome, covariates. For each, state the ICU-aware "
                "aggregation rule (from the research context: ordinal → max-in-window, "
                "labs → median, etc.).\n"
                "### Statistical analysis\n"
                "  Model family (logistic regression / Cox / clustering), adjustment set, "
                "sensitivity analyses (multiple-testing correction, subgroup analysis, "
                "ICU-rule-specific strata or missingness-pattern audits raised by the "
                "research context).\n"
                "### Software and reproducibility\n"
                "  One sentence: 'Analyses were conducted through the EasyICU research-agent "
                "pipeline; the full reproducibility envelope (prompt/response SHA-256 hashes, "
                "per-step scripts, and dependency lockfile) is released as supplementary material.'\n"
                "Target: 400-600 words."
            ),
            max_tokens=2048,
            **common,
        )

        _f_results = _ex.submit(
            self._call_section,
            section_name="Results",
            instruction=(
                "Write `## Results` with sub-sections:\n"
                "### Cohort characteristics\n"
                "  N, key demographics, cite {evidence:table_one} if available.\n"
                "### Primary outcome\n"
                "  Incidence, cite {evidence:outcome_rate}.\n"
                "### Primary association\n"
                "  Effect size, 95% CI, p-value, cite {evidence:primary_association} or "
                "{evidence:model_performance}.\n"
                "### Sensitivity and subgroup analyses\n"
                "  Multiple-testing result, subgroup heterogeneity, E-value if available.\n"
                "### ICU-specific quality control\n"
                "  Report any ICU-rule-specific finding raised by the research-context "
                "validators (e.g. a stratum where a derived score collapsed to a "
                "degenerate value, a missingness pattern that violated an aggregation "
                "rule). Cite the corresponding registered evidence id. Omit this "
                "subsection if no such finding was produced.\n"
                "Target: 400-600 words. Every numeric claim MUST have an {evidence:id} citation."
            ),
            max_tokens=2048,
            **common,
        )

        _f_discussion = _ex.submit(
            self._call_section,
            section_name="Discussion",
            instruction=(
                "Write `## Discussion` with 5 paragraphs (900-1300 words total):\n"
                "- Para 1: Restate the main finding and interpret it cautiously in the context of the results.\n"
                "- Para 2: Compare with prior literature and explain where this study agrees or diverges.\n"
                "- Para 3: Discuss plausible mechanisms using only associational language ('may reflect', 'could be consistent with', 'one possible explanation').\n"
                "- Para 4: Clinical implications, limits to generalisability, and why the result should not be over-interpreted.\n"
                "- Para 5: Strengths of the pipeline, evidence traceability, ICU-aware rules, and reproducibility.\n"
                "Requirements: full prose only, no bullets, no recommendations or causal claims, and at least one evidence citation or literature citation in each paragraph when available. Do not collapse the discussion into a two-sentence stub."
            ),
            max_tokens=4096,
            **common,
        )

        _f_limitations = _ex.submit(
            self._call_section,
            section_name="Limitations",
            instruction=(
                "Write `## Limitations` — one paragraph, 150-250 words. Include at least:\n"
                "1. Observational design → no causal inference, residual confounding.\n"
                "2. Single synthetic/database cohort → limited external generalisability.\n"
                "3. One ICU-specific limitation drawn from the registered evidence, "
                "expressed in concept-neutral terms — e.g. component-level missingness "
                "in a derived-score input, ordinal-score aggregation choice, time-window "
                "definition. Do not invent an ICU limitation that is not supported by a "
                "registered audit finding.\n"
                "4. LLM-in-the-loop limitation (generated code was audited but not manually "
                "reviewed line-by-line)."
            ),
            max_tokens=1024,
            **common,
        )

        _f_conclusion = _ex.submit(
            self._call_section,
            section_name="Conclusion, Data availability, Funding, COI",
            instruction=(
                "Write these sections exactly:\n"
                "## Conclusion\n"
                "1-2 sentences. Associational phrasing. Each conclusion sentence must cite "
                "at least one registered evidence id. End with a call for prospective / "
                "external validation only if it can be tied to sensitivity, limitation, "
                "or validation evidence.\n\n"
                "## Data and code availability\n"
                "'The cohort, generated scripts, SHA-256 evidence store, reproducibility "
                "envelope, STROBE checklist, and supplementary tables are released alongside "
                "this manuscript.'\n\n"
                "## Funding\n"
                "'Funding information was not available to the analysis agent and "
                "should be completed by the authors before journal submission.'\n\n"
                "## Conflicts of interest\n"
                "'The authors declare no conflicts of interest.'"
            ),
            max_tokens=512,
            **common,
        )

        # Collect in fixed manuscript order. .result() re-raises any section's
        # exception (fail-closed, as in the sequential version); the pool is
        # always shut down.
        try:
            title = _f_title.result()
            abstract = _f_abstract.result()
            introduction = _f_introduction.result()
            methods = _f_methods.result()
            results = _f_results.result()
            discussion = _f_discussion.result()
            limitations = _f_limitations.result()
            conclusion = _f_conclusion.result()
        finally:
            _ex.shutdown(wait=True)

        # Concatenate all sections.
        parts = [
            title,
            abstract,
            introduction,
            methods,
            results,
            discussion,
            limitations,
            conclusion,
        ]
        manuscript = "\n\n".join(p.strip() for p in parts if p.strip())
        return manuscript


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
        rest = text[idx + 3 :]
        # drop a leading language tag (json, python, etc.) on the same line
        nl = rest.find("\n")
        if nl >= 0 and rest[:nl].strip().isalnum():
            rest = rest[nl + 1 :]
        # if there's still a trailing fence, cut at it
        end = rest.find("```")
        if end >= 0:
            rest = rest[:end]
        return rest.strip() + "\n"
    return m.group(1).strip() + "\n"


def _looks_like_python_script(text: str) -> bool:
    stripped = (text or "").strip()
    if not stripped or stripped in {"{}", "[]", "null", "None"}:
        return False
    script_markers = (
        "\nimport ",
        "import ",
        "\nfrom ",
        "from ",
        "\ndef ",
        "def ",
        "os.environ",
        "pd.",
        "json.",
        ".to_csv",
        "write_text",
        "STEP_OUT_DIR",
        "COHORT_PARQUET",
    )
    return any(marker in stripped for marker in script_markers)


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
    data: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, List[str]]]:
    """Drop hosted-model extras before validating the strict schema.

    Returns both the normalized payload and a structured summary of the
    keys that were discarded so the pipeline can surface them in the
    manifest instead of silently suppressing them.
    """
    allowed_plan = {
        "research_question",
        "analysis_type",
        "cohort",
        "steps",
        "robustness_specs",
        "rationale",
        "revision",
    }
    allowed_step = {
        "step_id",
        "intent",
        "inputs",
        "expected_outputs",
        "method",
        "icu_rule_refs",
        "model_requirements",
        "trajectory_stability_spec",
    }
    allowed_model_requirement = {
        "requirement_id",
        "outcome",
        "outcome_type",
        "method_family",
        "exposure_source",
        "analysis_role",
        "analysis_set",
        "required_for_step_success",
    }
    allowed_robustness_spec = {
        "spec_id",
        "axis",
        "description",
        "cohort_override",
        "missing_override",
        "outcome_override",
    }
    dropped: Dict[str, List[str]] = {
        "top_level": [],
        "steps": [],
        "model_requirements": [],
        "robustness_specs": [],
    }
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
            requirements = []
            for req_idx, raw_requirement in enumerate(
                step_payload.get("model_requirements", []) or []
            ):
                if not isinstance(raw_requirement, dict):
                    requirements.append(raw_requirement)
                    continue
                requirement_payload = {}
                requirement_id = (
                    raw_requirement.get("requirement_id")
                    or f"step[{idx}].model_requirements[{req_idx}]"
                )
                for key, value in raw_requirement.items():
                    if key in allowed_model_requirement:
                        requirement_payload[key] = value
                    else:
                        dropped["model_requirements"].append(
                            f"{requirement_id}:{key}"
                        )
                requirements.append(requirement_payload)
            if "model_requirements" in step_payload:
                step_payload["model_requirements"] = requirements
            steps.append(step_payload)
    out["steps"] = steps
    specs = []
    for idx, raw_spec in enumerate(out.get("robustness_specs", []) or []):
        if not isinstance(raw_spec, dict):
            specs.append(raw_spec)
            continue
        spec_payload = {}
        spec_id = raw_spec.get("spec_id") or f"robustness_specs[{idx}]"
        for key, value in raw_spec.items():
            if key in allowed_robustness_spec:
                spec_payload[key] = value
            else:
                dropped["robustness_specs"].append(f"{spec_id}:{key}")
        specs.append(spec_payload)
    if "robustness_specs" in out:
        out["robustness_specs"] = specs
    return out, dropped


def _coerce_primary_estimate(
    step_summary: Dict[str, Any],
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
                # Explicit presence check, not a truthiness `or` chain: a
                # legitimate zero-valued estimate (e.g. a log-odds of 0.0) is
                # falsy and would fall through to the missing keys and yield
                # None, dropping a real estimate from primary_estimate.
                estimate = next(
                    (
                        payload[k]
                        for k in ("estimate", "value", "or")
                        if k in payload and payload[k] is not None
                    ),
                    None,
                )
                if isinstance(estimate, (int, float)) and not isinstance(
                    estimate, bool
                ):
                    interval = next(
                        (
                            payload[k]
                            for k in ("ci", "interval")
                            if k in payload and payload[k] is not None
                        ),
                        None,
                    )
                    if isinstance(interval, list) and len(interval) == 2:
                        try:
                            return (
                                float(estimate),
                                str(label),
                                [float(interval[0]), float(interval[1])],
                            )
                        except Exception:
                            pass
                    return float(estimate), str(label), None
    return None, None, None


def _suggest_repairs_for(
    step_summary: Dict[str, Any], findings: Sequence[str]
) -> List[str]:
    repairs: List[str] = []
    text = " ".join(findings).lower()
    if "calibration" in text:
        repairs.append(
            "Add or surface calibration diagnostics before accepting the result."
        )
    if "leakage" in text:
        repairs.append(
            "Revisit train/test split and feature timing to eliminate data leakage."
        )
    if "competing risk" in text:
        repairs.append(
            "Use a competing-risks aware analysis plan rather than a simple binary endpoint."
        )
    if "evidence" in text:
        repairs.append(
            "Register missing artifacts and bind them through evidence_id before drafting results."
        )
    if not repairs and step_summary:
        repairs.append(
            "Review the step summary and regenerate the step with explicit guardrails."
        )
    return repairs


def _sentences_missing_evidence_tokens(scaffold: str) -> List[str]:
    unsupported: List[str] = []
    text = re.sub(r"```.*?```", " ", scaffold, flags=re.S)
    cleaned_lines: List[str] = []
    section_label_re = re.compile(
        r"^\*\*(?:background|methods?|results?|conclusions?|discussion|limitations?)\s*:\*\*\s*",
        flags=re.I,
    )
    metadata_line_re = re.compile(
        r"^\s*(?:#{1,6}\s*)?(?:\*\*)?"
        r"(?:keywords?|key words|data\s+(?:and\s+code\s+)?availability|"
        r"code\s+availability|funding|conflicts?\s+of\s+interest|"
        r"acknowledg(?:e)?ments?|ethics\s+approval)"
        r"\s*(?:\*\*)?\s*[:：]?",
        flags=re.I,
    )
    in_metadata_section = False
    for raw_line in text.splitlines():
        stripped = raw_line.strip()
        if not stripped:
            cleaned_lines.append(" ")
            continue
        if re.match(r"^#{1,6}\s+", stripped):
            in_metadata_section = bool(metadata_line_re.match(stripped))
            continue
        if in_metadata_section or metadata_line_re.match(stripped):
            continue
        # Skip footnote/provenance DEFINITION lines (``[^claim_1]: value=...;
        # step=...; evidence=<name>``). These are auto-appended by the numeric
        # binder as machine provenance, not author-written result sentences:
        # they carry numbers + claimy words (auroc/brier/death) but reference
        # evidence via a plaintext ``evidence=<step>`` token (no ``](evidence/)``
        # link) when a claim binds to a step-level virtual evidence, so the
        # support check can mis-flag the whole footnote block as unsupported
        # prose. The block proves the claims are bound and is not prose to audit.
        if re.match(r"^\[\^[^\]]+\]:", stripped):
            continue
        match = section_label_re.match(stripped)
        if match:
            stripped = stripped[match.end() :].strip()
            if not stripped:
                continue
        cleaned_lines.append(stripped)
    text = " ".join(cleaned_lines)
    for raw_sentence in re.split(r"(?<=[.!?。！？])\s+", text):
        sentence = raw_sentence.strip()
        if not sentence:
            continue
        if "{evidence:" in sentence or re.search(
            r"\]\(\s*evidence/[^)]+\)", sentence, flags=re.I
        ):
            continue
        if re.search(
            r"(?:\[evidence missing:\s*[^\]]+\]|<!--\s*evidence missing:\s*[^>]+-->)",
            sentence,
            flags=re.I,
        ):
            unsupported.append(sentence)
            continue
        has_number = bool(re.search(r"\d", sentence))
        has_claimy_word = bool(
            re.search(
                r"\b(cohort|stays|patients|mortality|death|auroc|auc|hazard|odds|risk|cluster|survival|ci|p=|calibration|brier|discrimination|performance|robust(?:ness)?|overfitting|miscalibration|missingness|generalisability|generalizability)\b",
                sentence,
                flags=re.I,
            )
        )
        has_unquantified_result_claim = bool(
            re.search(
                r"\b(performance|robust(?:ness)?|consistent|overfitting|miscalibration|missingness|generalisability|generalizability)\b",
                sentence,
                flags=re.I,
            )
        )
        if (has_number and has_claimy_word) or has_unquantified_result_claim:
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
