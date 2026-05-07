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

from .analysis_types import planner_analysis_type_guide
from .icu_rules import VariableKind, default_time_windows
from .llm import LLMClient, LLMMessage
from .prompts import PROMPT_PACK_VERSION, load_prompt_pack
from .schema import (
    AggregationRule,
    AnalysisPlan,
    AnalysisStep,
    ConceptDescriptor,
    ResearchContext,
    VariableRole,
)


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
                    "prefixed variables such as eicu:age in `inputs`.\n\n"
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
                    "the script itself.\n\n"
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
	                    "do not add prose, markdown, or an explanation.\n\n"
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


__all__ = [
    "PlannerAgent",
    "ReplannerAgent",
    "CoderAgent",
    "AnalyzerAgent",
    "WriterAgent",
    "PROMPT_PACK_VERSION",
]
