"""LLM client abstraction for the research agent layer.

Two design rules:

1. **The pipeline must run end-to-end without an LLM.** The
   :class:`MockLLMClient` returns deterministic, ICU-aware canned
   responses derived directly from the :class:`ResearchContext`. It
   is what the unit tests and the offline demo use, and it is also a
   useful baseline to compare a real LLM against.

2. **No SDK is imported until used.** ``OpenAIClient`` lazy-imports
   ``openai``; if it is not installed the user gets a clear
   ImportError only when they actually try to invoke the model. This
   keeps ``import easyicu.research_agent`` cheap.

Adding another provider (Anthropic, Ollama, vLLM, ...) is a matter
of writing one class with a ``complete(messages, **kwargs) -> str``
method. The pipeline never imports a specific provider.
"""

from __future__ import annotations

import base64
import json
import mimetypes
import os
import re
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Sequence

from .analysis_types import infer_analysis_type
from .skills import build_dynamic_core_plan_steps
from .schema import (
    AnalysisPlan,
    AnalysisStep,
    ConceptDescriptor,
    ResearchContext,
    VariableRole,
)


@dataclass
class LLMMessage:
    role: str  # "system" | "user" | "assistant"
    content: str


class LLMClient(Protocol):
    """Minimal interface every provider must satisfy."""

    name: str

    def complete(self, messages: Sequence[LLMMessage], *, max_tokens: int = 2048,
                 temperature: float = 0.2) -> str: ...


# ---------------------------------------------------------------------------
# Mock client: ICU-aware canned responses, used for tests / offline demo
# ---------------------------------------------------------------------------


class MockLLMClient:
    """Deterministic, context-aware stub that exercises the full pipeline.

    The mock client doesn't pretend to be an LLM — it inspects the
    last user message and the embedded :class:`ResearchContext` (when
    one is provided through the special ``__context__`` attribute) to
    return responses that:

    * make the pipeline progress (so tests cover full code paths);
    * follow ICU rules (no mean-of-ordinal; max-aggregate SOFA;
      treat los_icu with median);
    * are fully deterministic (no randomness).

    For agents that want a real LLM, point :class:`ResearchAgentPipeline`
    at :class:`OpenAIClient` instead.
    """

    name = "mock"

    def __init__(self, context: Optional[ResearchContext] = None) -> None:
        self.context = context
        # Populated by :meth:`complete` so a wrapping ``MeteredClient``
        # picks up deterministic token counts in tests / offline demo
        # without falling back to the chars/4 heuristic.
        self.last_usage: Optional[Dict[str, int]] = None

    def complete(self, messages: Sequence[LLMMessage], *, max_tokens: int = 2048,
                 temperature: float = 0.2) -> str:
        last_user = next(
            (m.content for m in reversed(messages) if m.role == "user"),
            "",
        )
        ctx = self.context
        if ctx is None:
            response = _mock_generic_response(last_user)
        else:
            # Match on unique anchor phrases each agent injects, in order of
            # specificity. Order matters: the coder prompt may include step
            # intents that mention the word 'plan' (e.g. 'cross-database
            # replication plan'), so plan matching must come last.
            upper = last_user.upper()
            if (
                "WRITE THE PYTHON CODE FOR STEP" in upper
                or "WRITE THE PYTHON CODE" in upper
                or "REPAIR THE PYTHON CODE FOR STEP" in upper
                or "REPAIR THE PYTHON CODE" in upper
            ):
                response = _mock_code_for_step(ctx, last_user)
            elif "INTERPRET THE RESULTS OF STEP" in upper or "INTERPRET THE RESULTS" in upper:
                response = _mock_interpretation(ctx, last_user)
            elif (
                "WRITE A MANUSCRIPT SCAFFOLD" in upper
                or "MANUSCRIPT SCAFFOLD" in upper
                or "WRITE METHODS" in upper
            ):
                language = "zh" if (
                    "OUTPUT LANGUAGE: ZH" in upper
                    or "SIMPLIFIED CHINESE" in upper
                ) else "en"
                response = _mock_manuscript_scaffold(ctx, language=language)
            elif (
                "REVISE THE ICU-AWARE RESEARCH PLAN" in upper
                or "REVISE THE RESEARCH PLAN" in upper
                or "COMPLETED STEP RECORDS" in upper and "CURRENT PLAN" in upper
            ):
                response = _mock_replan_json(ctx, last_user)
            elif (
                "ICU-AWARE RESEARCH PLAN" in upper
                or "RESEARCH PLAN AS JSON" in upper
                or "ANALYSISPLAN SCHEMA" in upper
            ):
                response = _mock_plan_json(ctx)
            elif "LITERATURE" in upper and ("REVIEW" in upper or "CITATION" in upper):
                response = _mock_literature(ctx)
            else:
                response = _mock_generic_response(last_user)

        # Deterministic synthetic usage so cost-tracking tests don't have
        # to rely on the chars/4 fallback. We round to the same chars/4
        # rule the meter would use, but mark the record as authoritative
        # because the count is reproducible across mock runs.
        prompt_chars = sum(len(m.content or "") for m in messages)
        completion_chars = len(response or "")
        self.last_usage = {
            "prompt_tokens": max(1, prompt_chars // 4),
            "completion_tokens": max(1, completion_chars // 4),
            "total_tokens": max(1, (prompt_chars + completion_chars) // 4),
        }
        return response


def _mock_generic_response(prompt: str) -> str:
    return (
        "MOCK RESPONSE — no live LLM configured. The research-agent "
        "pipeline is running with the deterministic mock client; pass "
        "an OpenAIClient (or another LLMClient) to ResearchAgentPipeline "
        "to enable real planning and code generation."
    )


def _mock_literature(ctx: ResearchContext) -> str:
    """Return a small, hand-curated literature scaffold as JSON.

    The mock client cannot reach PubMed; instead it emits a short
    list of canonical references for each common ICU question, so
    the LiteratureAgent can run end-to-end offline. Real-LLM users
    pass a populated client and skip this branch.
    """
    sofa_in_scope = any(v.name.lower() in {"sofa", "sofa2"} for v in ctx.variables)
    aki_in_scope = any(v.name.lower() in {"creat", "kdigo", "aki"} for v in ctx.variables)
    seps_in_scope = any(v.name.lower() in {"sep3", "sepsis", "lact"} for v in ctx.variables)
    citations: List[Dict[str, str]] = []
    if sofa_in_scope:
        citations.append({
            "key": "vincent_sofa_1996",
            "title": "The SOFA (Sepsis-related Organ Failure Assessment) score to describe organ dysfunction/failure.",
            "year": "1996",
            "venue": "Intensive Care Medicine",
            "relevance": "Defines the SOFA score and its 0-4 ordinal components used here.",
        })
    if seps_in_scope:
        citations.append({
            "key": "singer_sepsis3_2016",
            "title": "The Third International Consensus Definitions for Sepsis and Septic Shock (Sepsis-3).",
            "year": "2016",
            "venue": "JAMA",
            "relevance": "Sepsis-3 reframes sepsis around SOFA-defined organ dysfunction; underpins outcome interpretation.",
        })
    if aki_in_scope:
        citations.append({
            "key": "kdigo_aki_2012",
            "title": "KDIGO Clinical Practice Guideline for Acute Kidney Injury.",
            "year": "2012",
            "venue": "Kidney International Supplements",
            "relevance": "Defines KDIGO AKI staging used by the EasyICU AKI module.",
        })
    citations.append({
        "key": "easyicu_2026",
        "title": "EasyICU: a Python toolkit for ICU dataset standardisation, inspired by ricu.",
        "year": "2026",
        "venue": "Software (this work)",
        "relevance": "Source of the cohort and concept dictionary used in the analysis.",
    })
    return json.dumps({"citations": citations}, indent=2, ensure_ascii=False)


def _mock_plan_json(ctx: ResearchContext) -> str:
    """Compose a minimal but valid AnalysisPlan as JSON.

    The mock plan keeps the outer research loop deterministic while
    selecting inner analysis steps dynamically from the question and
    context. This mirrors the default ClinicalSkill behaviour: keep
    the governance structure stable, but avoid forcing the same
    descriptive checks for every research question.
    """
    outcome = ctx.target_outcome or _pick_outcome(ctx)
    primary_pred = _pick_primary_predictor(ctx, outcome=outcome)
    analysis_type = infer_analysis_type(
        ctx,
        primary_predictor=primary_pred,
        target_outcome=outcome,
    )
    steps = build_dynamic_core_plan_steps(
        ctx,
        primary_predictor=primary_pred,
        target_outcome=outcome,
        scope_label="current ICU research question",
        rationale_note="Use the predictor's ICU-aware aggregation default and the first_24h anchor when applicable.",
        analysis_type_key=analysis_type.key,
    )

    if ctx.cross_database_validation:
        steps.append(
            AnalysisStep(
                step_id="06_cross_database_plan",
                intent=(
                    "Document a replication plan for: "
                    + ", ".join(ctx.cross_database_validation)
                    + ". Run the same pipeline with the same research_context schema; "
                    "compare cohort sizes, missingness profiles and primary-association "
                    "effect estimates."
                ),
                inputs=[],
                expected_outputs=["log:cross_database_plan"],
                method="replication_protocol",
            )
        )

    plan = AnalysisPlan(
        research_question=ctx.research_question,
        steps=steps,
        rationale=(
            f"Mock plan generated from ResearchContext for analysis type "
            f"'{analysis_type.key}'. The outer loop stays stable, while inner "
            "analysis steps are selected from the task family, variable roles "
            "and missingness metadata instead of being forced as a one-size-fits-all checklist."
        ),
    )
    return plan.model_dump_json(indent=2)


def _mock_replan_json(ctx: ResearchContext, prompt: str) -> str:
    """Deterministic replan: preserve completed steps, adjust remaining plan conservatively."""
    plan = AnalysisPlan.model_validate_json(_mock_plan_json(ctx))
    try:
        current_match = re.search(r"CURRENT PLAN:\n(\{.*?\})\n\nPROBE SUMMARY:", prompt, flags=re.DOTALL)
        if current_match:
            current = AnalysisPlan.model_validate_json(current_match.group(1))
            plan = current
    except Exception:
        pass
    outcome = ctx.target_outcome or _pick_outcome(ctx) or "outcome"
    analysis_type = infer_analysis_type(
        ctx,
        primary_predictor=_pick_primary_predictor(ctx, outcome=outcome),
        target_outcome=outcome,
    )
    probe_text = prompt.lower()
    allows_score_audit = analysis_type.key in {
        "association_study",
        "descriptive_epidemiology",
        "cross_database_replication",
    }
    if (
        allows_score_audit
        and "sofa_zero_anomaly" in probe_text
        and not any("sofa_zero" in s.step_id for s in plan.steps)
    ):
        score = _pick_primary_predictor(ctx, outcome=outcome) or "sofa2"
        plan.steps.append(
            AnalysisStep(
                step_id="05_sofa_zero_audit",
                intent=f"Audit whether {score}==0 behaves anomalously relative to adjacent strata.",
                inputs=[score, outcome],
                expected_outputs=["table:sofa_strata", "statistic:stratum_audit"],
                method="stratum_audit",
                icu_rule_refs=["aggregation_rule_for"],
            )
        )
    return plan.model_copy(update={"revision": plan.revision + 1}).model_dump_json(indent=2)


def _pick_outcome(ctx: ResearchContext) -> Optional[str]:
    for v in ctx.variables:
        if v.role == VariableRole.OUTCOME and v.name.lower() in {"death", "death_icu", "death_hosp", "mortality"}:
            return v.name
    for v in ctx.variables:
        if v.role == VariableRole.OUTCOME:
            return v.name
    return None


def _normalise_for_question_match(text: str) -> str:
    """Normalise user-facing text so ``SOFA-2`` matches a ``sofa2`` column."""
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def _tokens_for_question_match(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


_QUESTION_ALIASES: Dict[str, tuple[str, ...]] = {
    "kdigo_stage": ("kdigo", "akistage", "kdigoakistage", "peakfirst24hkdigo"),
    "vaso": ("vasopressor", "vasopressorexposure", "anyvasopressor", "pressor"),
    "map": ("meanarterialpressure", "arterialpressure"),
    "gcs": ("glasgowcomascale", "worstglasgowcoma", "comascale"),
    "lact": ("lactate",),
}


def _question_mentions_variable(ctx: ResearchContext, variable_name: str) -> bool:
    question = _normalise_for_question_match(ctx.research_question)
    tokens = _tokens_for_question_match(ctx.research_question)
    name = _normalise_for_question_match(variable_name)
    if name and (name in tokens or (len(name) >= 4 and name in question)):
        return True
    aliases = _QUESTION_ALIASES.get(variable_name.lower(), ())
    return any(alias in question for alias in aliases)


def _score_preference_key(ctx: ResearchContext, name: str) -> tuple[int, int, str]:
    lower = name.lower()
    mentioned_rank = 0 if _question_mentions_variable(ctx, name) else 1
    sofa_rank = 0 if lower == "sofa2" else 1 if lower == "sofa" else 2
    return (mentioned_rank, sofa_rank, lower)


def _pick_sofa_score(ctx: ResearchContext) -> Optional[str]:
    """Choose the SOFA-family variable, preferring the question and SOFA-2."""
    candidates = [
        v.name for v in ctx.variables
        if v.name.lower() in {"sofa", "sofa2"}
        and v.role in {VariableRole.COMPOSITE_SCORE, VariableRole.ORDINAL_SCORE}
    ]
    if not candidates:
        return None
    return sorted(candidates, key=lambda name: _score_preference_key(ctx, name))[0]


def _pick_primary_predictor(ctx: ResearchContext, outcome: Optional[str]) -> Optional[str]:
    """Heuristic: prefer question-mentioned variables, then scores/vitals/labs."""
    pref_order = [
        VariableRole.COMPOSITE_SCORE,
        VariableRole.ORDINAL_SCORE,
        VariableRole.VITAL,
        VariableRole.LAB,
        VariableRole.INTERVENTION,
        VariableRole.DEMOGRAPHIC,
    ]
    eligible_roles = set(pref_order)
    mentioned = [
        v.name for v in ctx.variables
        if v.name != outcome
        and v.role in eligible_roles
        and _question_mentions_variable(ctx, v.name)
    ]
    if mentioned:
        return sorted(mentioned, key=lambda name: _score_preference_key(ctx, name))[0]

    sofa_score = _pick_sofa_score(ctx)
    if sofa_score and sofa_score != outcome:
        return sofa_score

    by_role: Dict[VariableRole, List[str]] = {r: [] for r in pref_order}
    for v in ctx.variables:
        if v.name == outcome:
            continue
        if v.role in by_role:
            by_role[v.role].append(v.name)
    for r in pref_order:
        if by_role[r]:
            return by_role[r][0]
    return None


def _mock_code_for_step(ctx: ResearchContext, prompt: str) -> str:
    """Return a minimal, ICU-aware analysis script for the requested step.

    The mock writes safe code: it never averages an ordinal score, it
    reports median (IQR) for labs, and it produces a CSV per requested
    table and a PNG per requested figure.

    When ``step_id`` matches ``*_primary_association``, the mock emits a
    purpose-built logistic-regression script (T1.6) so the pipeline
    actually produces an odds ratio rather than re-running the
    descriptive boilerplate.
    """
    step_id = _extract_step_id(prompt) or "step"
    outcome = ctx.target_outcome or _pick_outcome(ctx) or "death"
    # Mirror the planner's role-aware detection so the naive arm of T1.4
    # never produces a sofa_strata.csv it lacks the rule for.
    sofa_var = _pick_sofa_score(ctx)

    if "primary_association" in step_id:
        primary_pred = _pick_primary_predictor(ctx, outcome=outcome) or sofa_var or "age"
        return _mock_code_primary_association(
            ctx=ctx,
            step_id=step_id,
            outcome=outcome,
            predictor=primary_pred,
        )

    # Inline script as a triple-quoted heredoc — note: keep this tight; the
    # runner persists it byte-for-byte and hashes it as evidence.
    code = textwrap.dedent(
        f'''
        # AUTO-GENERATED by easyicu.research_agent.MockLLMClient
        # step_id: {step_id}
        # research_question: {ctx.research_question!r}
        # rules: ordinal scores -> max; labs -> median(IQR); never mean an ordinal column.
        from __future__ import annotations
        import json
        import os
        from pathlib import Path
        import pandas as pd
        import numpy as np
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        cohort_path = os.environ["COHORT_PARQUET"]
        out_dir = Path(os.environ["STEP_OUT_DIR"])
        out_dir.mkdir(parents=True, exist_ok=True)

        df = pd.read_parquet(cohort_path)
        step_kind = {step_id!r}.lower()
        do_table_one = "table_one" in step_kind
        do_outcome_incidence = "outcome_incidence" in step_kind
        do_missingness = "missingness" in step_kind
        do_sofa_audit = any(token in step_kind for token in ("sofa_zero_audit", "stratum_audit", "composite"))
        do_protocol_only = any(token in step_kind for token in ("protocol", "plan"))

        outcome_col = {outcome!r} if {outcome!r} in df.columns else None
        sofa_col = {sofa_var!r} if {sofa_var!r} else None
        if sofa_col and sofa_col not in df.columns:
            sofa_col = None
        if not any((do_table_one, do_outcome_incidence, do_missingness, do_sofa_audit)):
            if do_protocol_only:
                do_table_one = False
                do_outcome_incidence = False
                do_missingness = False
                do_sofa_audit = False
            else:
                do_table_one = True
                do_outcome_incidence = True
                do_missingness = True
                do_sofa_audit = sofa_col is not None and outcome_col is not None

        summary = {{}}

        if do_protocol_only:
            protocol_lines = [
                f"# Protocol note for {{step_kind}}",
                f"- Research question: {ctx.research_question}",
                "- Available variables: " + ", ".join(df.columns.astype(str).tolist()),
                "- This is a task-family planning/protocol step rather than a finished effect estimate.",
            ]
            (out_dir / "protocol_notes.md").write_text("\\n".join(protocol_lines), encoding="utf-8")
            summary["protocol_notes_path"] = "protocol_notes.md"

        # ---- Table 1: cohort summary, ICU-aware ----
        if do_table_one:
            rows = []
            for col in df.columns:
                s = df[col]
                n = int(len(s))
                n_miss = int(s.isna().sum())
                row = {{
                    "variable": col,
                    "n": n,
                    "n_missing": n_miss,
                    "frac_missing": (n_miss / n) if n else 0.0,
                }}
                if pd.api.types.is_numeric_dtype(s):
                    if sofa_col is not None and col == sofa_col:
                        # ordinal: report mode + range, never mean
                        s_int = s.dropna().astype("Int64")
                        if len(s_int) > 0:
                            mode_val = int(s_int.mode().iloc[0])
                        else:
                            mode_val = None
                        row["mode"] = mode_val
                        row["min"] = (None if s.dropna().empty else float(s.min()))
                        row["max"] = (None if s.dropna().empty else float(s.max()))
                    else:
                        s_clean = s.dropna()
                        if len(s_clean) > 0:
                            row["median"] = float(s_clean.median())
                            row["q25"] = float(s_clean.quantile(0.25))
                            row["q75"] = float(s_clean.quantile(0.75))
                elif s.dtype == bool or set(s.dropna().unique()) <= {{0, 1}}:
                    pos = int(s.fillna(0).astype(int).sum())
                    row["n_positive"] = pos
                    row["pct_positive"] = (pos / n) if n else 0.0
                rows.append(row)
            table_one = pd.DataFrame(rows)
            table_one.to_csv(out_dir / "table_one.csv", index=False)
            summary["table_one_path"] = "table_one.csv"

        # ---- Outcome incidence ----
        if do_outcome_incidence and outcome_col is not None:
            inc = float(df[outcome_col].dropna().astype(int).mean())
            summary["outcome_col"] = outcome_col
            summary["outcome_rate"] = inc
            pd.DataFrame([{{
                "outcome": outcome_col,
                "n_total": int(df[outcome_col].notna().sum()),
                "n_events": int(df[outcome_col].dropna().astype(int).sum()),
                "outcome_rate": inc,
            }}]).to_csv(out_dir / "outcome_incidence.csv", index=False)
            summary["outcome_incidence_path"] = "outcome_incidence.csv"

        # ---- Missingness audit ----
        if do_missingness:
            miss = pd.DataFrame({{
                "variable": df.columns,
                "n_missing": [int(df[c].isna().sum()) for c in df.columns],
                "n_total": [int(len(df))] * len(df.columns),
                "frac_missing": [
                    (int(df[c].isna().sum()) / max(len(df), 1)) for c in df.columns
                ],
            }})
            miss.to_csv(out_dir / "missingness.csv", index=False)
            summary["missingness_path"] = "missingness.csv"
            try:
                miss_plot = miss.sort_values("frac_missing", ascending=False).head(12)
                fig, ax = plt.subplots(figsize=(5.2, max(2.8, 0.35 * len(miss_plot))))
                ax.barh(miss_plot["variable"], miss_plot["frac_missing"], color="#7aa6d1")
                ax.invert_yaxis()
                ax.set_xlabel("Fraction missing")
                ax.set_ylabel("Variable")
                ax.set_title("Missingness audit")
                fig.tight_layout()
                fig.savefig(out_dir / "missingness_heatmap.png", dpi=160)
                plt.close(fig)
                summary["missingness_figure_path"] = "missingness_heatmap.png"
            except Exception:
                pass

        # ---- SOFA stratum audit (the key sofa==0 / sofa2==0 check) ----
        if do_sofa_audit and sofa_col is not None and outcome_col is not None:
            sub = df[[sofa_col, outcome_col]].dropna()
            sub[sofa_col] = sub[sofa_col].astype(int)
            grp = sub.groupby(sofa_col)[outcome_col].agg(["count", "mean"]).reset_index()
            grp.columns = [sofa_col, "n", "outcome_rate"]
            grp.to_csv(out_dir / "sofa_strata.csv", index=False)
            summary["sofa_strata_path"] = "sofa_strata.csv"

            # Anomaly flag: if sofa==0 outcome rate exceeds sofa==1 rate, flag.
            try:
                rate_at_zero = float(grp.loc[grp[sofa_col] == 0, "outcome_rate"].iloc[0])
                rate_at_one = float(grp.loc[grp[sofa_col] == 1, "outcome_rate"].iloc[0])
                summary["sofa_zero_anomaly"] = rate_at_zero > rate_at_one
                summary["sofa_zero_rate"] = rate_at_zero
                summary["sofa_one_rate"] = rate_at_one
            except (IndexError, KeyError):
                summary["sofa_zero_anomaly"] = False

            fig, ax = plt.subplots(figsize=(5, 3.2))
            ax.plot(grp[sofa_col], grp["outcome_rate"], marker="o", color="#1f77b4")
            ax.set_xlabel(sofa_col)
            ax.set_ylabel(f"{{outcome_col}} rate")
            ax.set_title(f"Outcome rate by {{sofa_col}} stratum")
            fig.tight_layout()
            fig.savefig(out_dir / "sofa_strata.png", dpi=160)
            plt.close(fig)
            summary["sofa_figure_path"] = "sofa_strata.png"

        # ---- Persist machine-readable summary ----
        with open(out_dir / "step_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
        print(json.dumps(summary, indent=2, ensure_ascii=False, default=str))
        '''
    ).strip() + "\n"
    return code


def _mock_code_primary_association(
    *, ctx: ResearchContext, step_id: str, outcome: str, predictor: str,
) -> str:
    """Logistic-regression script for the ``*_primary_association`` step (T1.6).

    The script fits ``outcome ~ predictor + age + sex`` using statsmodels
    when available, falls back to a numpy / scipy MLE otherwise, and
    persists coefficients, 95 % CI and odds ratios to
    ``primary_association.csv``. ``step_summary.json`` records the OR
    for ``predictor`` so the StatisticalValidator can re-derive it.

    The script is deterministic and self-contained — exactly what the
    runner expects. Aggregations remain ICU-aware: ``predictor`` is
    used as-is (the planner already chose the right column), ``age`` is
    treated as continuous, ``sex`` is one-hot encoded as a binary
    indicator (``sex_M``).
    """
    # The script itself checks at runtime which adjustment columns are
    # available; we don't need to bake the answer in here.
    code = textwrap.dedent(
        f'''
        # AUTO-GENERATED by easyicu.research_agent.MockLLMClient
        # step_id: {step_id}
        # research_question: {ctx.research_question!r}
        # method: logistic regression of {outcome} on {predictor} (+age, +sex if present)
        from __future__ import annotations
        import json
        import os
        from pathlib import Path

        import numpy as np
        import pandas as pd
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        cohort_path = os.environ["COHORT_PARQUET"]
        out_dir = Path(os.environ["STEP_OUT_DIR"])
        out_dir.mkdir(parents=True, exist_ok=True)

        def _erf_approx(x):
            # Abramowitz & Stegun 7.1.26 (max error 1.5e-7) — avoids a
            # scipy.stats import so this script works in barebones envs.
            sign = np.sign(x)
            x = np.abs(x)
            a1 = 0.254829592; a2 = -0.284496736; a3 = 1.421413741
            a4 = -1.453152027; a5 = 1.061405429; p = 0.3275911
            t = 1.0 / (1.0 + p * x)
            y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * np.exp(-x * x)
            return sign * y


        df = pd.read_parquet(cohort_path)
        outcome_col = {outcome!r}
        predictor_col = {predictor!r}
        cols_needed = [c for c in [predictor_col, outcome_col, "age", "sex"] if c in df.columns]
        sub = df[cols_needed].dropna().copy()

        # one-hot the binary sex column (M -> 1, F -> 0); leave any other
        # categorical sex coding alone with a warning written to summary.
        sex_used = False
        if "sex" in sub.columns:
            uniq = set(map(str, sub["sex"].dropna().unique()))
            if uniq <= {{"M", "F"}}:
                sub["sex_M"] = (sub["sex"].astype(str) == "M").astype(int)
                sex_used = True
            sub = sub.drop(columns=["sex"])

        terms = [predictor_col]
        if "age" in sub.columns:
            terms.append("age")
        if sex_used:
            terms.append("sex_M")

        y = sub[outcome_col].astype(int).to_numpy()
        X = np.column_stack([np.ones(len(sub))] + [sub[t].astype(float).to_numpy() for t in terms])
        names = ["intercept"] + terms

        # ---- Fit ----
        coefs = None
        cov = None
        backend = "manual"
        try:
            import statsmodels.api as sm  # type: ignore
            res = sm.Logit(y, X).fit(disp=0, method="newton", maxiter=200)
            coefs = np.asarray(res.params, dtype=float)
            cov = np.asarray(res.cov_params(), dtype=float)
            backend = "statsmodels"
        except Exception:
            try:
                from scipy import optimize  # type: ignore

                def _neg_ll(beta):
                    z = X @ beta
                    # log-sum-exp stable log(1+exp(z))
                    log_ll = np.where(z >= 0, np.log1p(np.exp(-z)), -z + np.log1p(np.exp(z)))
                    return float(np.sum((1 - y) * z + log_ll))

                def _grad(beta):
                    p = 1.0 / (1.0 + np.exp(-(X @ beta)))
                    return X.T @ (p - y)

                beta0 = np.zeros(X.shape[1])
                opt = optimize.minimize(_neg_ll, beta0, jac=_grad, method="BFGS")
                coefs = opt.x
                p = 1.0 / (1.0 + np.exp(-(X @ coefs)))
                W = p * (1 - p)
                # Fisher information; pseudo-inverse for numerical safety.
                fisher = (X.T * W) @ X
                cov = np.linalg.pinv(fisher)
                backend = "scipy_bfgs"
            except Exception as exc:
                # Last-ditch: skip but still write a parseable artefact so the
                # pipeline doesn't crash; downstream validator will flag it.
                pd.DataFrame([{{"variable": "(skipped)", "reason": str(exc)}}]).to_csv(
                    out_dir / "primary_association.csv", index=False)
                with open(out_dir / "step_summary.json", "w", encoding="utf-8") as f:
                    json.dump({{
                        "method": "logistic_regression",
                        "predictor": predictor_col,
                        "outcome": outcome_col,
                        "primary_or": None,
                        "skipped": True,
                        "reason": str(exc),
                    }}, f, indent=2, ensure_ascii=False)
                print("(primary_association skipped):", exc)
                raise SystemExit(0)

        se = np.sqrt(np.maximum(np.diag(cov), 0.0))
        z = coefs / np.where(se > 0, se, np.nan)
        # Two-sided normal-approx p (avoids dependency on scipy.stats here):
        #   p = 2 * (1 - Φ(|z|)) = 1 - erf(|z| / sqrt(2))
        p_val = np.where(np.isnan(z), np.nan,
                         1.0 - _erf_approx(np.abs(z) / np.sqrt(2.0)))
        ci_lo = coefs - 1.959963984540054 * se
        ci_hi = coefs + 1.959963984540054 * se
        rows = []
        for nm, b, s, lo, hi, pv in zip(names, coefs, se, ci_lo, ci_hi, p_val):
            rows.append({{
                "variable": nm,
                "coef": float(b),
                "std_err": float(s),
                "ci_lower": float(lo),
                "ci_upper": float(hi),
                "odds_ratio": float(np.exp(b)),
                "or_lower": float(np.exp(lo)),
                "or_upper": float(np.exp(hi)),
                "p_value": float(pv) if not np.isnan(pv) else None,
            }})
        coef_df = pd.DataFrame(rows)
        coef_df.to_csv(out_dir / "primary_association.csv", index=False)

        primary_or = float(np.exp(coefs[names.index(predictor_col)]))
        primary_or_lo = float(np.exp(ci_lo[names.index(predictor_col)]))
        primary_or_hi = float(np.exp(ci_hi[names.index(predictor_col)]))

        # Forest plot (skipped for the intercept; non-binary sex coding excluded).
        try:
            plot_rows = coef_df[coef_df["variable"] != "intercept"].reset_index(drop=True)
            fig, ax = plt.subplots(figsize=(5, 0.6 + 0.5 * len(plot_rows)))
            ys = np.arange(len(plot_rows))
            ax.errorbar(
                plot_rows["odds_ratio"], ys,
                xerr=[plot_rows["odds_ratio"] - plot_rows["or_lower"],
                      plot_rows["or_upper"] - plot_rows["odds_ratio"]],
                fmt="o", color="#1f77b4",
            )
            ax.axvline(1.0, linestyle="--", color="grey", linewidth=0.8)
            ax.set_yticks(ys)
            ax.set_yticklabels(plot_rows["variable"])
            ax.set_xlabel(f"Odds ratio for {{outcome_col}}")
            ax.set_title(f"Adjusted association ({{backend}})")
            fig.tight_layout()
            fig.savefig(out_dir / "primary_association_curve.png", dpi=160)
            plt.close(fig)
        except Exception:
            # Plot is decorative; never fail the step over it.
            pass

        # ---- Outcome incidence (cheap and the validator cross-checks it) ----
        outcome_rate = float(df[outcome_col].dropna().astype(int).mean()) if outcome_col in df.columns else None

        summary = {{
            "method": "logistic_regression",
            "backend": backend,
            "predictor": predictor_col,
            "outcome": outcome_col,
            "n_used": int(len(sub)),
            "outcome_rate": outcome_rate,
            "primary_or": primary_or,
            "primary_or_ci": [primary_or_lo, primary_or_hi],
            "primary_association_path": "primary_association.csv",
        }}
        with open(out_dir / "step_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
        print(json.dumps(summary, indent=2, ensure_ascii=False, default=str))
        '''
    ).strip() + "\n"
    return code


def _extract_step_id(prompt: str) -> Optional[str]:
    m = re.search(r"step_id\s*[:=]\s*([\w-]+)", prompt)
    if m:
        return m.group(1)
    m = re.search(r"\b(step\s+)?(\d{2,3}_[\w-]+)\b", prompt)
    return m.group(2) if m else None


def _mock_interpretation(ctx: ResearchContext, prompt: str) -> str:
    """Brief, evidence-grounded interpretation paragraph."""
    sofa_var = _pick_sofa_score(ctx)
    outcome = ctx.target_outcome or _pick_outcome(ctx) or "the primary outcome"
    parts: List[str] = []
    parts.append(
        f"The cohort of {ctx.cohort.n_stays:,} ICU stays from {ctx.cohort.database} "
        f"was analysed against the question: '{ctx.research_question}'."
    )
    if sofa_var:
        parts.append(
            f"As planned, we audited {sofa_var}-stratum incidence of {outcome}. "
            f"If the {sofa_var}==0 stratum has an elevated {outcome} rate compared "
            f"with {sofa_var}==1, the most parsimonious explanation is component-"
            f"level missingness rather than truly absent organ dysfunction; the "
            f"missingness audit table should be consulted before drawing clinical "
            "conclusions."
        )
    parts.append(
        "All numerical claims in this paragraph are bound to entries in the evidence "
        "store; reviewers can verify each value against its generating script and run log."
    )
    return " ".join(parts)


def _mock_manuscript_scaffold(ctx: ResearchContext, *, language: str = "en") -> str:
    """Return a minimal manuscript scaffold in markdown.

    The scaffold is deliberately *thin*: title, methods, results
    skeleton, all referencing evidence ids that the writer will inject
    from the evidence store. Discussion and clinical claims are left
    blank — that is policy, not laziness.
    """
    sofa_var = _pick_sofa_score(ctx)
    outcome = ctx.target_outcome or _pick_outcome(ctx) or "the primary outcome"
    cross_db = ", ".join(ctx.cross_database_validation) if ctx.cross_database_validation else "(none planned)"
    if language == "zh":
        return textwrap.dedent(f"""
        # 手稿脚手架

        > 由 easyicu.research_agent 生成。以下每个数值性主张都必须带有
        > `{{evidence:<id>}}` 证据占位符；未绑定证据的句子会被后处理拦截。

        ## 标题
        {ctx.cohort.database} ICU 患者中 {sofa_var or "主要预测变量"} 与 {outcome} 的关系：
        一项可追溯的 agent 辅助分析。

        ## 方法
        队列由 {{evidence:table_one}} 描述，包含来自 {ctx.cohort.database} 的
        {ctx.cohort.n_stays:,} 次 ICU 住院。纳入标准：
        {", ".join(ctx.cohort.inclusion_criteria) or "见 cohort_config.json"}。
        排除标准：{", ".join(ctx.cohort.exclusion_criteria) or "见 cohort_config.json"}。

        变量处理遵循 EasyICU 概念字典和 {{evidence:research_context}} 中的
        ICU-aware 聚合规则：有序评分在窗口内取最大值；右偏实验室指标以中位数
        (IQR) 描述；时间窗分析使用 {{evidence:research_context}} 中定义的
        {", ".join(w.name for w in ctx.time_windows)} 窗口。

        跨数据库复现计划：{cross_db}。

        ## 结果
        结局发生率：{{evidence:outcome_rate}}。
        缺失情况：{{evidence:missingness}}。
        主要关联：{{evidence:primary_association}}。
        {sofa_var + " 分层审计：{evidence:sofa_strata}。" if sofa_var else ""}

        ## 讨论
        *(留给人类作者；writer agent 不在没有人工确认的情况下生成临床主张或建议。)*
        """).strip() + "\n"
    return textwrap.dedent(f"""
    # Manuscript scaffold

    > Generated by easyicu.research_agent. Every numeric claim below is an
    > `{{evidence_id}}` placeholder filled in from the evidence store.
    > Sentences without an evidence id are blocked by the writer.

    ## Title
    Association between {sofa_var or "the primary predictor"} and {outcome}
    in {ctx.cohort.database} ICU patients: a traceable agent-assisted analysis.

    ## Methods
    Cohort: {{evidence:table_one}} describes the {ctx.cohort.n_stays:,} ICU stays from
    {ctx.cohort.database} included in this study. Inclusion criteria:
    {", ".join(ctx.cohort.inclusion_criteria) or "see cohort_config.json"}.
    Exclusion criteria: {", ".join(ctx.cohort.exclusion_criteria) or "see cohort_config.json"}.

    Variable handling followed the EasyICU concept dictionary and the ICU-aware
    aggregation rules in {{evidence:research_context}}: ordinal scores were
    aggregated by maximum within window; right-skewed laboratory measurements
    were summarised as median (IQR); time-window analyses used the
    {", ".join(w.name for w in ctx.time_windows)} windows defined in
    {{evidence:research_context}}.

    Cross-database replication: {cross_db}.

    ## Results
    Outcome incidence: {{evidence:outcome_rate}}.
    Missingness profile: {{evidence:missingness}}.
    Primary association: {{evidence:primary_association}}.
    {sofa_var + "-stratum audit: {evidence:sofa_strata}." if sofa_var else ""}

    ## Discussion
    *(left to the human author; the writer agent declines to generate clinical
    claims and recommendations without explicit human sign-off.)*
    """).strip() + "\n"


# ---------------------------------------------------------------------------
# OpenAI client (optional — only imported on first use)
# ---------------------------------------------------------------------------


class OpenAIClient:
    """Thin wrapper around ``openai>=1.0`` chat completions.

    Usage::

        from easyicu.research_agent import OpenAIClient

        # OpenAI proper
        llm = OpenAIClient(model="gpt-4o-mini")

        # OpenRouter (free tier) — anything OpenAI-compatible works the
        # same way; the ``base_url`` is the only knob that differs.
        llm = OpenAIClient(
            model="google/gemini-2.0-flash-exp:free",
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ["OPENROUTER_API_KEY"],
            extra_headers={"HTTP-Referer": "https://github.com/shen-lab-icu/easyicu",
                           "X-Title": "EasyICU research-agent"},
        )

        pipeline = ResearchAgentPipeline(llm=llm, ...)

    The class deliberately does not bundle prompt templates or
    streaming logic — that lives in :mod:`agents` so any provider can
    be swapped in.
    """

    name = "openai"

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        request_timeout: float = 120.0,
        extra_headers: Optional[Dict[str, str]] = None,
        extra_body: Optional[Dict[str, Any]] = None,
    ) -> None:
        try:
            from openai import OpenAI  # type: ignore
        except Exception as exc:  # pragma: no cover - exercised only when SDK missing
            raise ImportError(
                "OpenAIClient requires the 'openai' package. Install with `pip install openai`."
            ) from exc

        kwargs: Dict[str, Any] = {}
        # Accept either OPENAI_API_KEY (vanilla) or OPENROUTER_API_KEY so
        # users don't have to alias the variable themselves.
        env_key = (
            api_key
            or os.environ.get("OPENAI_API_KEY")
            or os.environ.get("OPENROUTER_API_KEY")
        )
        if env_key:
            kwargs["api_key"] = env_key
        if base_url or os.environ.get("OPENAI_BASE_URL"):
            kwargs["base_url"] = base_url or os.environ.get("OPENAI_BASE_URL")
        # OpenRouter recommends — and some providers require — a
        # ``HTTP-Referer`` / ``X-Title`` header for analytics. Pass them
        # to the SDK as default headers when supplied.
        if extra_headers:
            kwargs["default_headers"] = dict(extra_headers)
        self._client = OpenAI(**kwargs)
        self._model = model
        self._timeout = request_timeout
        self._extra_body = dict(extra_body or {})

    def complete(self, messages: Sequence[LLMMessage], *, max_tokens: int = 2048,
                 temperature: float = 0.2) -> str:
        chat_messages = [{"role": m.role, "content": m.content} for m in messages]
        create_kwargs: Dict[str, Any] = {
            "model": self._model,
            "messages": chat_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "timeout": self._timeout,
        }
        if self._extra_body:
            create_kwargs["extra_body"] = self._extra_body
        resp = self._client.chat.completions.create(**create_kwargs)  # type: ignore[arg-type]
        # T3.2 cost tracking: stash the SDK's reported usage so a wrapping
        # ``MeteredClient`` can pull authoritative token counts instead of
        # falling back to the chars/4 heuristic. Defensive: not every
        # provider populates ``usage`` on every response.
        try:
            usage = getattr(resp, "usage", None)
            if usage is not None:
                self.last_usage = {
                    "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
                    "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
                    "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
                }
            else:
                self.last_usage = None
        except Exception:
            self.last_usage = None

        # T1.3 — robust content extraction. Reasoning-tuned models
        # (GLM-4.5, DeepSeek-R1, o1-style) often leave ``content``
        # empty and put the answer in ``reasoning`` / ``reasoning_content``.
        # OpenRouter typically surfaces this under ``reasoning``;
        # z.ai's native API uses ``reasoning_content``. Fall through
        # the common attributes and finally scan the message dump.
        choice = resp.choices[0]
        self.last_finish_reason = getattr(choice, "finish_reason", None)
        msg = choice.message
        content = (getattr(msg, "content", None) or "").strip()
        if not content:
            for attr in ("reasoning_content", "reasoning"):
                val = getattr(msg, attr, None)
                if isinstance(val, str) and val.strip():
                    content = val.strip()
                    break
        if not content:
            # Last-resort: walk the SDK's model_dump() and pick the
            # longest non-trivial string field. Catches providers that
            # use ``thinking`` or other vendor-specific keys.
            try:
                dump = msg.model_dump() if hasattr(msg, "model_dump") else dict(msg)  # type: ignore[arg-type]
                best = ""
                for k, v in (dump or {}).items():
                    if k in {"role", "refusal", "annotations"}:
                        continue
                    if isinstance(v, str) and len(v.strip()) > len(best):
                        best = v.strip()
                if best:
                    content = best
            except Exception:
                pass

        # Optional debug dump — ``EASYICU_LLM_DEBUG=1 …`` writes one
        # JSON file per call so the user can inspect what the model
        # actually returned (finish_reason, raw message, prompt).
        if os.environ.get("EASYICU_LLM_DEBUG"):
            try:
                from datetime import datetime
                from pathlib import Path
                log_dir = Path(
                    os.environ.get("EASYICU_LLM_DEBUG_DIR")
                    or "./research_output/llm_debug"
                )
                log_dir.mkdir(parents=True, exist_ok=True)
                ts = datetime.now().strftime("%Y%m%dT%H%M%S_%f")
                payload = {
                    "model": self._model,
                    "finish_reason": getattr(choice, "finish_reason", None),
                    "prompt_messages": chat_messages,
                    "raw_message": (
                        msg.model_dump() if hasattr(msg, "model_dump") else str(msg)
                    ),
                    "extracted_content_head": content[:1200],
                    "extracted_content_chars": len(content),
                }
                (log_dir / f"{ts}.json").write_text(
                    json.dumps(payload, indent=2, ensure_ascii=False, default=str),
                    encoding="utf-8",
                )
            except Exception:
                pass

        return content

    def complete_with_images(
        self,
        *,
        prompt: str,
        image_paths: Sequence[Path],
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> str:
        """Run a multimodal chat-completions request against image files.

        This method is intentionally optional: normal agents continue
        to use ``complete(...)``. ``VLMVisualQAAdapter`` checks for the
        method with ``hasattr`` and falls back to text-only review when
        a provider does not support image inputs.
        """
        content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
        for path in image_paths:
            p = Path(path)
            mime = mimetypes.guess_type(str(p))[0] or "application/octet-stream"
            data = base64.b64encode(p.read_bytes()).decode("ascii")
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:{mime};base64,{data}"},
            })
        create_kwargs: Dict[str, Any] = {
            "model": self._model,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "timeout": self._timeout,
        }
        if self._extra_body:
            create_kwargs["extra_body"] = self._extra_body
        resp = self._client.chat.completions.create(**create_kwargs)  # type: ignore[arg-type]
        try:
            usage = getattr(resp, "usage", None)
            if usage is not None:
                self.last_usage = {
                    "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
                    "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
                    "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
                }
            else:
                self.last_usage = None
        except Exception:
            self.last_usage = None
        choice = resp.choices[0]
        self.last_finish_reason = getattr(choice, "finish_reason", None)
        msg = choice.message
        return (getattr(msg, "content", None) or "").strip()


# ---------------------------------------------------------------------------
# Per-agent LLM router (T2.3 — different tool, different brain)
# ---------------------------------------------------------------------------


_ROUTER_ROLES = ("planner", "coder", "analyzer", "writer", "literature")


class LLMRouter:
    """Per-role LLM client mapping.

    The four research-agent agents have very different needs:

    * **Planner** must emit valid JSON matching the AnalysisPlan schema.
      A frontier model usually wins here; a small model often emits
      malformed JSON that even our hardened parser cannot recover.
    * **Coder** writes the largest *output* (code + plot calls). A
      mid-tier model that is fast and cheap is the sweet spot.
    * **Analyzer** runs the shortest prompt of all (one paragraph
      input, four sentences out). The cheapest available model is
      usually fine.
    * **Writer** is brief (≈ 600 tokens) but needs to follow the
      ``{evidence:<id>}`` format precisely. A mid-tier model is
      typically enough.
    * **Literature** is optional; the offline curated registry is the
      default, but the agent can be wired through this router too.

    Running everything on the same model wastes money and rate limit.
    The :class:`LLMRouter` lets the pipeline use a different
    :class:`LLMClient` per role::

        router = LLMRouter(
            default=OpenAIClient(model="gpt-4o-mini"),
            planner=OpenAIClient(model="gpt-4o"),
            analyzer=OpenAIClient(model="gpt-4o-mini"),
        )
        pipeline = ResearchAgentPipeline(workdir=..., llm=router)

    Backwards compatibility: passing a plain :class:`LLMClient`
    (``MockLLMClient``, ``OpenAIClient``, …) to
    :class:`ResearchAgentPipeline` continues to work because the
    pipeline asks the router for ``for_role(role)`` only when the
    object actually has the method.
    """

    name = "router"

    def __init__(
        self,
        *,
        default: Optional[Any] = None,
        planner: Optional[Any] = None,
        coder: Optional[Any] = None,
        analyzer: Optional[Any] = None,
        writer: Optional[Any] = None,
        literature: Optional[Any] = None,
    ) -> None:
        self._default = default
        self._roles: Dict[str, Optional[Any]] = {
            "planner": planner,
            "coder": coder,
            "analyzer": analyzer,
            "writer": writer,
            "literature": literature,
        }
        if default is None and all(v is None for v in self._roles.values()):
            raise ValueError(
                "LLMRouter needs at least one client. Pass a `default=` "
                "and/or any subset of role-specific clients."
            )

    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------

    def for_role(self, role: str) -> Any:
        """Return the client to use for ``role``.

        Falls back to the ``default`` client when a role-specific
        client is not configured. Raises ``KeyError`` if neither is
        available.
        """
        if role not in self._roles:
            raise KeyError(
                f"unknown role {role!r}; expected one of {list(self._roles)}"
            )
        client = self._roles[role] or self._default
        if client is None:
            raise KeyError(
                f"LLMRouter has no client for role {role!r} and no default."
            )
        return client

    def iter_clients(self):
        """Yield every distinct underlying client.

        Used by the pipeline to bind ``ResearchContext`` onto every
        :class:`MockLLMClient` reachable through the router so the
        canned responses pick up the cohort that's actually being
        analysed.
        """
        seen = set()
        for client in (self._default, *self._roles.values()):
            if client is None:
                continue
            ident = id(client)
            if ident in seen:
                continue
            seen.add(ident)
            yield client

    # ------------------------------------------------------------------
    # Pass-through ``complete``
    # ------------------------------------------------------------------

    def complete(self, messages: Sequence["LLMMessage"], *, max_tokens: int = 2048,
                 temperature: float = 0.2) -> str:
        """Route to the default client.

        This bridge exists so a router can be passed to legacy code
        paths that haven't been updated to call :meth:`for_role`.
        Prefer ``router.for_role(role).complete(...)`` in new code.
        """
        if self._default is None:
            raise RuntimeError(
                "LLMRouter.complete() called but no `default` client is "
                "configured; use ``router.for_role(role).complete(...)`` "
                "or set ``default=...`` at construction."
            )
        return self._default.complete(messages, max_tokens=max_tokens, temperature=temperature)


def resolve_role_client(llm: Any, role: str) -> Any:
    """Return the client to use for ``role``.

    If ``llm`` exposes ``for_role`` (i.e. it is an :class:`LLMRouter`),
    we delegate; otherwise the same ``llm`` is returned for every role
    — preserving the pre-T2.3 single-client semantics.
    """
    if llm is None:
        return None
    if hasattr(llm, "for_role"):
        return llm.for_role(role)
    return llm


__all__ = [
    "LLMMessage",
    "LLMClient",
    "MockLLMClient",
    "OpenAIClient",
    "LLMRouter",
    "resolve_role_client",
]
