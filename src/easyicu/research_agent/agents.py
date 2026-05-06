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
import textwrap
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .icu_rules import VariableKind, default_time_windows
from .llm import LLMClient, LLMMessage
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


_SYSTEM_GUIDE = textwrap.dedent(
    """
    You are an analysis agent inside the EasyICU research-agent layer.

    Hard rules (the validators enforce them; violating them wastes tokens):
    1. Never average an ordinal column. SOFA components and totals,
       GCS, KDIGO stages and any column with role "ordinal_score" or
       "composite_score" are aggregated by max-within-window or
       reported as median(IQR), never as mean(SD).
    2. Never silently impute missing values. If a column has high
       missingness, it should be reported alongside the analysis or
       handled explicitly (e.g. "complete-case", "missing-indicator
       variables").
    3. Never invent variables, concepts, or time windows. You may
       only refer to variables that appear in the ResearchContext you
       are given; if you need something else, return a question rather
       than fabricating.
    4. Right-skewed laboratory variables (creatinine, lactate,
       bilirubin, ...) are reported as median (IQR), not mean (SD).
    5. ICU mortality, hospital mortality and 28-day mortality are NOT
       interchangeable. Use the column the ResearchContext designates
       as ``target_outcome``.
    6. Cross-database validation is executable only when an external
       cohort file is explicitly available to the script. If the context
       names eICU, HiRID, or another database but the script only receives
       `COHORT_PARQUET`, write a replication protocol / harmonisation
       checklist instead of fabricating prefixed columns or fitting a
       model on unavailable data.
    """
).strip()


def _format_variable(v: ConceptDescriptor) -> str:
    miss = ""
    if v.missingness is not None:
        miss = (
            f" missing={v.missingness.fraction_missing:.1%} "
            f"({v.missingness.missingness_kind})"
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

    def run(self, context: ResearchContext) -> AnalysisPlan:
        messages = [
            LLMMessage(role="system", content=_SYSTEM_GUIDE),
            LLMMessage(
                role="user",
                content=(
                    "Produce an ICU-AWARE RESEARCH PLAN as JSON matching the "
                    "AnalysisPlan schema. The plan must include: a Table 1 "
                    "step, an outcome-incidence step, a missingness audit "
                    "step, a primary-association step, and — if any "
                    "composite ordinal score is in scope — a stratum-level "
                    "audit step (specifically checking score==0 against "
                    "score==1 to detect component-missingness artefacts). "
                    "If cross-database replication is requested, include a "
                    "cross-database step, but mark it as a feasibility / "
                    "protocol step unless the ResearchContext explicitly "
                    "provides external cohort files. Do not put invented "
                    "prefixed variables such as eicu:age in `inputs`.\n\n"
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
        data = _normalise_plan_payload(data)
        return AnalysisPlan.model_validate(data)


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
	                    "TRACEBACK-SPECIFIC REQUIREMENTS:\n"
	                    "- If statsmodels reports `Pandas data cast to numpy dtype of object`, "
	                    "`exog contains inf or nans`, or any Logit/OLS dtype/missing-data "
	                    "error, rebuild the modelling dataframe from scratch. After dummy "
	                    "encoding categorical predictors, coerce the entire design matrix "
	                    "with `X = X.apply(pd.to_numeric, errors=\"coerce\").astype(float)`, "
	                    "coerce `y = pd.to_numeric(y, errors=\"coerce\").astype(float)`, "
	                    "replace +/-inf with NaN, align y/X on the finite complete-case "
	                    "index, and only then call `sm.add_constant(..., has_constant=\"add\")` "
	                    "and `sm.Logit(y, X)`. For `pd.get_dummies`, pass `dtype=float` so "
	                    "dummy columns are not boolean/object. Assert or explicitly check "
	                    "`np.isfinite(X.to_numpy()).all()` and `np.isfinite(y.to_numpy()).all()` "
	                    "before fitting.\n\n"
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


_CODER_GUIDE = textwrap.dedent(
    '''

    Code contract:
    - Read the cohort from os.environ["COHORT_PARQUET"].
    - Write any artefact (CSV, PNG, JSON) to os.environ["STEP_OUT_DIR"].
    - Use matplotlib's "Agg" backend; do not call plt.show().
    - Save a machine-readable summary to step_summary.json containing
      every numeric statistic that downstream agents may quote.

    JSON SERIALISATION — MANDATORY (free-tier models trip this):
    - Every dict you write to disk must contain only Python primitives
      (int, float, str, bool, list, dict, None). numpy scalars and
      pandas missing values are NOT JSON-serialisable.
    - Define this helper near the top of EVERY script that writes JSON,
      and ALWAYS pass it as `default=to_jsonable`:

        def to_jsonable(x):
            import math
            import numpy as np
            import pandas as pd
            if isinstance(x, (np.integer,)):
                return int(x)
            if isinstance(x, (np.floating,)):
                v = float(x)
                return v if math.isfinite(v) else None
            if isinstance(x, (np.bool_,)):
                return bool(x)
            if isinstance(x, np.ndarray):
                return x.tolist()
            try:
                if pd.isna(x):
                    return None
            except (TypeError, ValueError):
                pass
            return str(x)

      Then call ALWAYS: `json.dump(payload, f, indent=2,
      default=to_jsonable, ensure_ascii=False)`.

    CSV CELL VALUES — MANDATORY:
    - Every CSV cell must be a SCALAR (one number or one short string).
      Do NOT put tuples, lists, numpy reprs, or "(median, q25, q75)"
      strings into a single cell. If you want median + IQR per group,
      emit THREE columns: `<var>_median`, `<var>_q25`, `<var>_q75`.
      The downstream binder treats each cell as a citable value;
      tuples-as-strings break every downstream consumer.
    - For categorical summaries write one row per category, not a
      single cell with `(n, pct)` tuples.

    STATISTICS APIs:
    - For binomial confidence intervals use
      `statsmodels.stats.proportion.proportion_confint` or compute the
      normal/Wilson interval directly. Do not use the non-existent
      `scipy.stats.proportion.proportion_confint`, and do not import
      deprecated `scipy.stats.binom_test`.
    - For statsmodels regressions, build a numeric design matrix:
      after `pd.get_dummies(...)`, call
      `X = X.apply(pd.to_numeric, errors="coerce").astype(float)`,
      drop rows with missing y/X, and add a constant after coercion.
      The safe pattern is:
      `model_df = df[[y_col] + x_cols].copy()`;
      `model_df = model_df.apply(pd.to_numeric, errors="coerce")`;
      `model_df = model_df.replace([np.inf, -np.inf], np.nan).dropna()`;
      `y = model_df[y_col].astype(float)`;
      `X = sm.add_constant(model_df[x_cols].astype(float), has_constant="add")`.
      Never call `sm.Logit(y, X)` until `np.isfinite(X.to_numpy()).all()`
      and `np.isfinite(y.to_numpy()).all()` are true. If lactate is
      missing-indicator imputed, still drop missing non-lactate covariates
      such as MAP after creating the lactate indicator/imputed column.

    PYTHON HYGIENE:
    - Python collection constructors must be syntactically valid:
      write `set(["a", "b"])` or `{"a", "b"}`, never `set("a", "b")`.
    - Only import from the standard library, pandas, numpy, scipy,
      matplotlib, and statsmodels. No network access.

    ROBUSTNESS:
    - If the step is cross-database validation but only `COHORT_PARQUET`
      is available, do not fit models for unavailable databases. Write
      a CSV/JSON protocol listing target databases, required variables,
      harmonisation notes, and reproducibility checks. Wrap any value
      that came from the cohort (n, mortality rate, sha256) in
      `int(...)` / `float(...)` before putting it in the protocol dict.
    - If a column the prompt asks about is absent from the cohort,
      gracefully skip it and record the omission in step_summary.json
      under a `"skipped"` key — do not crash the step.
    - Do not let optional plotting break a completed table/statistic
      step. If a plot is secondary, wrap only the plotting block in a
      narrow `try/except Exception as exc` and record the skipped plot
      in step_summary.json.
    - Never assume every category or stratum exists. Before `.iloc[0]`
      on filtered rows, check `.empty`; before indexing category
      columns, check membership in `df.columns`.
    - Matplotlib error bars must be non-negative. Build yerr values
      with `np.maximum(0, upper - estimate)` and
      `np.maximum(0, estimate - lower)`, and replace missing bounds
      with zero-width intervals.
    - Do not delete or modify the cohort parquet.
    - Print step_summary.json contents to stdout at the end.
    '''
).strip()


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


_WRITER_GUIDE = textwrap.dedent(
    """

    Writer contract:
    - Discussion is OFF-LIMITS. End the manuscript with a one-line stub
      saying Discussion is left to the human author.
    - Every numeric claim in Methods or Results must be either spelled
      out in words ("the cohort comprised X stays") with X immediately
      followed by an {evidence:<id>} placeholder, OR phrased as
      "see {evidence:<id>}".
    - Sentences without an evidence id will be removed by the post-processor.
    - Do not invent confidence intervals, p-values, or hazard ratios.
    """
).strip()


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


def _normalise_plan_payload(data: Dict[str, Any]) -> Dict[str, Any]:
    """Drop hosted-model extras before validating the strict schema."""
    allowed_plan = {"research_question", "steps", "rationale", "revision"}
    allowed_step = {
        "step_id",
        "intent",
        "inputs",
        "expected_outputs",
        "method",
        "icu_rule_refs",
    }
    out = {k: v for k, v in data.items() if k in allowed_plan}
    steps = []
    for raw_step in out.get("steps", []) or []:
        if isinstance(raw_step, dict):
            steps.append({k: v for k, v in raw_step.items() if k in allowed_step})
    out["steps"] = steps
    return out


__all__ = [
    "PlannerAgent",
    "CoderAgent",
    "AnalyzerAgent",
    "WriterAgent",
]
