"""ICU-aware mock LLM client + deterministic response generators.

This module hosts the deterministic mock-LLM machinery: the
``MockLLMClient`` class and the canned ``_mock_*`` response
generators that produce ICU-shaped plan/replan/code/manuscript
responses without calling out to a real model.

Split out of :mod:`easyicu.research_agent.providers.llm` so that file stays
focused on production provider integrations (OpenAIClient,
FallbackLLMClient, LLMRouter). The mock layer is ~1.5k lines of
canned content and is what the unit tests and the offline demo
exercise.

The mock client is intentionally imported from this module rather than from
the production provider module, keeping offline fixtures out of production
provider initialization.
"""

from __future__ import annotations

import json
import re
import textwrap
from typing import Any, Dict, List, Optional, Sequence

from ..planning.analysis_types import infer_analysis_type
from .protocol import LLMMessage
from ..skills import build_dynamic_core_plan_steps
from ..schema import (
    AnalysisPlan,
    AnalysisStep,
    ConceptDescriptor,
    ResearchContext,
    VariableRole,
)
from ..planning.robustness_contract import RobustnessSpec

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
        from .factory import register_offline_test_client

        register_offline_test_client(self)
        self.context = context
        self.calls: list[tuple[list[LLMMessage], dict[str, Any]]] = []
        # Populated by :meth:`complete` so a wrapping ``MeteredClient``
        # picks up deterministic token counts in tests / offline demo
        # without falling back to the chars/4 heuristic.
        self.last_usage: Optional[Dict[str, int]] = None

    def complete(
        self,
        messages: Sequence[LLMMessage],
        *,
        max_tokens: int = 2048,
        temperature: float = 0.2,
        seed: Optional[int] = None,
    ) -> str:
        # ``seed`` is accepted for signature parity with OpenAIClient so
        # the reproducibility envelope (O20) can forward it uniformly.
        # The mock is deterministic regardless of seed.
        _ = seed
        self.calls.append(
            (
                list(messages),
                {
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "seed": seed,
                },
            )
        )
        user_messages = [m.content for m in messages if m.role == "user"]
        last_user = user_messages[-1] if user_messages else ""
        # Structured retries append a validator-feedback user turn after the
        # original request. Classify the request from the complete immutable
        # user-message history so the deterministic mock keeps serving the
        # same agent role on retry. Use the same combined text for generators
        # that need step details from the original prompt.
        response = _contextual_mock_response(
            context=self.context,
            user_messages=user_messages,
            last_user=last_user,
        )

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


class ScriptedMockLLMClient:
    """Built-in deterministic mock returning a closed response sequence."""

    name = "scripted-mock"

    def __init__(
        self,
        responses: Sequence[str | BaseException],
        *,
        repeat_last: bool = False,
    ) -> None:
        from .factory import register_offline_test_client

        self.responses = list(responses)
        self._repeat_last = bool(repeat_last)
        self._last_response: str | BaseException | None = None
        self.calls: list[tuple[list[LLMMessage], dict[str, Any]]] = []
        self.messages: list[LLMMessage] = []
        register_offline_test_client(self)

    def complete(self, messages: Sequence[LLMMessage], **kwargs: Any) -> str:
        self.messages = list(messages)
        self.calls.append((list(self.messages), dict(kwargs)))
        if self.responses:
            response = self.responses.pop(0)
            self._last_response = response
        elif self._repeat_last and self._last_response is not None:
            response = self._last_response
        else:
            raise RuntimeError("scripted mock response sequence exhausted")
        if isinstance(response, BaseException):
            raise response
        return str(response)


class ScriptedVisionMockLLMClient(ScriptedMockLLMClient):
    """Built-in static-response vision mock with no transport callback."""

    name = "scripted-vision-mock"
    supports_vision = True

    def __init__(self, responses: Sequence[str | BaseException]) -> None:
        # Register only after construction as this exact reviewed type; calling
        # the parent constructor would attempt to register the subclass there.
        from .factory import register_offline_test_client

        self.responses = list(responses)
        self._repeat_last = False
        self._last_response: str | BaseException | None = None
        self.calls: list[tuple[list[LLMMessage], dict[str, Any]]] = []
        self.image_calls: list[dict[str, Any]] = []
        register_offline_test_client(self)

    def complete_with_images(self, **kwargs: Any) -> str:
        self.image_calls.append(dict(kwargs))
        if not self.responses:
            raise RuntimeError("scripted vision mock response sequence exhausted")
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return str(response)


class BudgetAwareScriptedMockLLMClient(ScriptedMockLLMClient):
    """Closed scripted mock that owns the active provider-attempt charge."""

    name = "budget-aware-scripted-mock"
    provider_attempt_budget_aware = True

    def __init__(self, responses: Sequence[str | Exception]) -> None:
        from .factory import register_offline_test_client

        self.responses = list(responses)
        self._repeat_last = False
        self._last_response: str | Exception | None = None
        self.calls: list[tuple[list[LLMMessage], dict[str, Any]]] = []
        register_offline_test_client(self)

    def complete(self, messages: Sequence[LLMMessage], **kwargs: Any) -> str:
        from ..authority.provider_budget import consume_active_transport_attempt

        consume_active_transport_attempt()
        self.calls.append((list(messages), dict(kwargs)))
        if not self.responses:
            raise RuntimeError("scripted mock response sequence exhausted")
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return str(response)


class PatternScriptedMockLLMClient:
    """Closed, non-network prompt router with static literal response queues.

    This is intentionally less expressive than a callback mock: tests may map
    literal prompt markers to fixed strings or exceptions, but cannot execute
    arbitrary code at the trusted delivery boundary. Rules are ordered from
    generic to specific; a later matching rule has explicit priority.
    """

    name = "pattern-scripted-mock"

    def __init__(
        self,
        rules: Sequence[tuple[str, Sequence[str | BaseException]]],
        *,
        default: str | BaseException = "{}",
        contextual_default: bool = False,
    ) -> None:
        from .factory import register_offline_test_client

        self._rules = [(str(marker), list(responses)) for marker, responses in rules]
        self._default = default
        self._contextual_default = bool(contextual_default)
        self.context: Optional[ResearchContext] = None
        self.calls: list[tuple[list[LLMMessage], dict[str, Any]]] = []
        register_offline_test_client(self)

    def complete(self, messages: Sequence[LLMMessage], **kwargs: Any) -> str:
        copied = list(messages)
        self.calls.append((copied, dict(kwargs)))
        prompt = "\n".join(str(message.content or "") for message in copied)
        folded_prompt = prompt.casefold()
        response: str | BaseException = self._default
        matches = [
            (index, marker, responses)
            for index, (marker, responses) in enumerate(self._rules)
            if marker.casefold() in folded_prompt
        ]
        if matches:
            _priority, marker, responses = matches[-1]
            if not responses:
                raise RuntimeError(
                    f"pattern mock response sequence exhausted for {marker!r}"
                )
            response = responses.pop(0)
        elif self._contextual_default:
            user_messages = [
                message.content for message in copied if message.role == "user"
            ]
            response = _contextual_mock_response(
                context=self.context,
                user_messages=user_messages,
                last_user=user_messages[-1] if user_messages else "",
            )
        if isinstance(response, BaseException):
            raise response
        return str(response)


def _contextual_mock_response(
    *,
    context: Optional[ResearchContext],
    user_messages: Sequence[str],
    last_user: str,
) -> str:
    """Return the built-in contextual mock response without a callback seam."""

    if context is None:
        return _mock_generic_response(last_user)

    request_text = "\n\n".join(user_messages)
    # Match on unique anchor phrases each agent injects, in order of
    # specificity. Order matters: the coder prompt may include step intents
    # that mention the word "plan", so plan matching must come last.
    upper = request_text.upper()
    if (
        "WRITE THE PYTHON CODE FOR STEP" in upper
        or "WRITE THE PYTHON CODE" in upper
        or "REPAIR THE PYTHON CODE FOR STEP" in upper
        or "REPAIR THE PYTHON CODE" in upper
    ):
        return _mock_code_for_step(context, request_text)
    if "INTERPRET THE RESULTS OF STEP" in upper or "INTERPRET THE RESULTS" in upper:
        return _mock_interpretation(context, request_text)
    if "WRITE ONLY THE **" in upper and "CITATION RULE" in upper:
        language = (
            "zh"
            if ("OUTPUT LANGUAGE: ZH" in upper or "SIMPLIFIED CHINESE" in upper)
            else "en"
        )
        return _mock_writer_section(context, request_text, language=language)
    if (
        "WRITE A MANUSCRIPT SCAFFOLD" in upper
        or "MANUSCRIPT SCAFFOLD" in upper
        or "WRITE METHODS" in upper
    ):
        language = (
            "zh"
            if ("OUTPUT LANGUAGE: ZH" in upper or "SIMPLIFIED CHINESE" in upper)
            else "en"
        )
        return _mock_manuscript_scaffold(context, language=language)
    if (
        "REVISE THE ICU-AWARE RESEARCH PLAN" in upper
        or "REVISE THE RESEARCH PLAN" in upper
        or "COMPLETED STEP RECORDS" in upper
        and "CURRENT PLAN" in upper
    ):
        return _mock_replan_json(context, request_text)
    if (
        "ICU-AWARE RESEARCH PLAN" in upper
        or "RESEARCH PLAN AS JSON" in upper
        or "ANALYSISPLAN SCHEMA" in upper
    ):
        return _mock_plan_json(context)
    if "LITERATURE" in upper and ("REVIEW" in upper or "CITATION" in upper):
        return _mock_literature(context)
    return _mock_generic_response(last_user)


class ExternalCaptureMockLLMClient:
    """Non-network capture mock that exercises the external outbound path."""

    name = "external-capture-mock"

    def __init__(self, responses: Sequence[str | Exception]) -> None:
        from .factory import _register_external_capture_test_client

        self.responses = list(responses)
        self.calls: list[tuple[list[LLMMessage], dict[str, Any]]] = []
        _register_external_capture_test_client(self)

    def complete(self, messages: Sequence[LLMMessage], **kwargs: Any) -> str:
        self.calls.append((list(messages), dict(kwargs)))
        if not self.responses:
            raise RuntimeError("external capture response sequence exhausted")
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return str(response)


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
    aki_in_scope = any(
        v.name.lower() in {"creat", "kdigo", "aki"} for v in ctx.variables
    )
    seps_in_scope = any(
        v.name.lower() in {"sep3", "sepsis", "lact"} for v in ctx.variables
    )
    citations: List[Dict[str, str]] = []
    if sofa_in_scope:
        citations.append(
            {
                "key": "vincent_sofa_1996",
                "title": "The SOFA (Sepsis-related Organ Failure Assessment) score to describe organ dysfunction/failure.",
                "year": "1996",
                "venue": "Intensive Care Medicine",
                "relevance": "Defines the SOFA score and its 0-4 ordinal components used here.",
            }
        )
    if seps_in_scope:
        citations.append(
            {
                "key": "singer_sepsis3_2016",
                "title": "The Third International Consensus Definitions for Sepsis and Septic Shock (Sepsis-3).",
                "year": "2016",
                "venue": "JAMA",
                "relevance": "Sepsis-3 reframes sepsis around SOFA-defined organ dysfunction; underpins outcome interpretation.",
            }
        )
    if aki_in_scope:
        citations.append(
            {
                "key": "kdigo_aki_2012",
                "title": "KDIGO Clinical Practice Guideline for Acute Kidney Injury.",
                "year": "2012",
                "venue": "Kidney International Supplements",
                "relevance": "Defines KDIGO AKI staging used by the EasyICU AKI module.",
            }
        )
    citations.append(
        {
            "key": "easyicu_2026",
            "title": "EasyICU: a Python toolkit for ICU dataset standardisation, inspired by ricu.",
            "year": "2026",
            "venue": "Software (this work)",
            "relevance": "Source of the cohort and concept dictionary used in the analysis.",
        }
    )
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
    # The deterministic mock coder implements the generic association branch
    # as logistic regression.  Keep the mock plan's method contract equally
    # specific so an effect-producing step is owned by an authorised method
    # instead of the old ambiguous ``logistic_or_KM`` placeholder.
    steps = [
        (
            step.model_copy(update={"method": "logistic_regression"})
            if step.step_id == "04_primary_association"
            else (
                step.model_copy(
                    update={
                        "expected_outputs": [
                            *step.expected_outputs,
                            "table:calibration",
                        ]
                    }
                )
                if step.step_id == "04_prediction_model_analysis"
                and "table:calibration" not in step.expected_outputs
                else step
            )
        )
        for step in steps
    ]
    # Mock plans make the data dependency explicit before the shared plan
    # shaper runs.  This lets render-only children consume the exact typed
    # table registered by their science parent through resolved_inputs.
    separated_steps: List[AnalysisStep] = []
    for step in steps:
        outputs = list(step.expected_outputs or [])
        figure_outputs = [
            output for output in outputs if str(output).lower().startswith("figure:")
        ]
        non_figure_outputs = [
            output for output in outputs if output not in figure_outputs
        ]
        split_mock_step = bool(figure_outputs and non_figure_outputs) and (
            step.step_id == "04_primary_association" or "missingness" in step.step_id
        )
        if not split_mock_step:
            separated_steps.append(step)
            continue
        separated_steps.append(
            step.model_copy(update={"expected_outputs": non_figure_outputs})
        )
        typed_table_inputs = [
            output
            for output in non_figure_outputs
            if str(output)
            .lower()
            .startswith(("table:", "statistic:", "artifact:", "dataset:", "model:"))
        ]
        separated_steps.append(
            AnalysisStep(
                step_id=f"{step.step_id}_figure",
                planned_analysis_role="auxiliary",
                intent=(
                    f"Render {', '.join(figure_outputs)} from the registered "
                    f"typed outputs of '{step.step_id}' without recomputing science."
                ),
                inputs=typed_table_inputs,
                expected_outputs=figure_outputs,
                method="visualization",
                icu_rule_refs=list(step.icu_rule_refs or []) + ["visualization_rule"],
            )
        )
    steps = separated_steps

    if ctx.cross_database_validation:
        steps.append(
            AnalysisStep(
                step_id="06_cross_database_protocol",
                planned_analysis_role="auxiliary",
                intent=(
                    "Document a replication protocol for: "
                    + ", ".join(ctx.cross_database_validation)
                    + ". Run the same pipeline with the same research_context schema; "
                    "compare cohort sizes, missingness profiles and primary-association "
                    "effect estimates."
                ),
                inputs=[],
                expected_outputs=["log:cross_database_protocol"],
                method="replication_protocol",
            )
        )

    plan = AnalysisPlan(
        research_question=ctx.research_question,
        analysis_type=analysis_type.key,
        steps=steps,
        rationale=(
            f"Mock plan generated from ResearchContext for analysis type "
            f"'{analysis_type.key}'. The outer loop stays stable, while inner "
            "analysis steps are selected from the task family, variable roles "
            "and missingness metadata instead of being forced as a one-size-fits-all checklist."
        ),
    )
    # The production Planner is required to satisfy the article-level contract
    # before execution. Keep the built-in offline Planner on that same schema:
    # deterministic augmentation may add reporting/display roles, but never
    # invents a missing Planner-owned headline result.
    from ..reporting.article_contract import (
        augment_plan_for_article_contract,
        build_article_analysis_contract,
    )

    article_contract = build_article_analysis_contract(
        ctx,
        analysis_type=plan.analysis_type,
    )
    if "robustness" in article_contract.required_roles and not plan.robustness_specs:
        variables = [
            value
            for value in (primary_pred, outcome)
            if value and ctx.variable(value) is not None
        ]
        plan = plan.model_copy(
            update={
                "robustness_specs": [
                    RobustnessSpec(
                        spec_id="mock_complete_case",
                        axis="missing",
                        description=(
                            "Deterministic offline complete-case sensitivity over "
                            "the Planner-selected analysis variables."
                        ),
                        missing_override={
                            "strategy": "complete_case",
                            "variables": variables,
                        },
                    )
                ]
            }
        )
    pre_augmentation_step_ids = {step.step_id for step in plan.steps}
    plan, _ = augment_plan_for_article_contract(
        plan=plan,
        contract=article_contract,
    )
    # Deterministically added display-only figure steps still need an exact
    # upstream table. The real Planner must declare that dependency itself;
    # the built-in offline mock wires it here so its generated code exercises
    # the same resolved-input and provenance boundary instead of rereading the
    # cohort or fabricating a source table.
    primary_tables = [
        output
        for step in plan.steps
        if step.step_id in pre_augmentation_step_ids
        and step.planned_analysis_role == "primary"
        for output in step.expected_outputs
        if str(output).lower().startswith("table:")
    ]
    all_tables = [
        output
        for step in plan.steps
        if step.step_id in pre_augmentation_step_ids
        for output in step.expected_outputs
        if str(output).lower().startswith("table:")
    ]
    table_one = next(
        (
            output
            for output in all_tables
            if "table_one" in str(output).lower() or "baseline" in str(output).lower()
        ),
        None,
    )
    wired_steps: List[AnalysisStep] = []
    for step in plan.steps:
        is_added_display = (
            step.step_id not in pre_augmentation_step_ids
            and step.method == "article_contract_display"
        )
        figure_outputs = [
            output
            for output in step.expected_outputs
            if str(output).lower().startswith("figure:")
        ]
        if not is_added_display or not figure_outputs or step.inputs:
            wired_steps.append(step)
            continue
        wants_cohort_source = any(
            "cohort" in str(output).lower() for output in figure_outputs
        )
        source = (
            table_one
            if wants_cohort_source and table_one is not None
            else next(iter(primary_tables or all_tables), None)
        )
        wired_steps.append(
            step.model_copy(update={"inputs": [source] if source is not None else []})
        )
    plan = plan.model_copy(update={"steps": wired_steps})
    return plan.model_dump_json(indent=2)


def _mock_replan_json(ctx: ResearchContext, prompt: str) -> str:
    """Deterministic replan: preserve completed steps, adjust remaining plan conservatively."""
    plan = AnalysisPlan.model_validate_json(_mock_plan_json(ctx))
    try:
        current_match = re.search(
            r"CURRENT PLAN:\n(\{.*?\})\n\nPROBE SUMMARY:", prompt, flags=re.DOTALL
        )
        if current_match:
            current = AnalysisPlan.model_validate_json(current_match.group(1))
            plan = current
    except Exception:
        pass
    return plan.model_copy(update={"revision": plan.revision + 1}).model_dump_json(
        indent=2
    )


def _pick_outcome(ctx: ResearchContext) -> Optional[str]:
    for v in ctx.variables:
        if v.role == VariableRole.OUTCOME and v.name.lower() in {
            "death",
            "death_icu",
            "death_hosp",
            "mortality",
        }:
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
    # Prefer the more specific variable name when multiple candidates match the
    # same question text (e.g. ``score2`` should beat ``score``).
    return (mentioned_rank, -len(_normalise_for_question_match(name)), lower)


def _pick_score(ctx: ResearchContext) -> Optional[str]:
    """Choose a composite/ordinal score without naming a particular score."""
    candidates = [
        v.name
        for v in ctx.variables
        if v.role in {VariableRole.COMPOSITE_SCORE, VariableRole.ORDINAL_SCORE}
    ]
    if not candidates:
        return None
    return sorted(candidates, key=lambda name: _score_preference_key(ctx, name))[0]


def _pick_primary_predictor(
    ctx: ResearchContext, outcome: Optional[str]
) -> Optional[str]:
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
        v.name
        for v in ctx.variables
        if v.name != outcome
        and v.role in eligible_roles
        and _question_mentions_variable(ctx, v.name)
    ]
    if mentioned:
        return sorted(mentioned, key=lambda name: _score_preference_key(ctx, name))[0]

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
    expected_outputs = _extract_expected_outputs(prompt)
    protocol_output = next(
        (output for output in expected_outputs if output.lower().startswith("log:")),
        "log:protocol_notes",
    )
    outcome = ctx.target_outcome or _pick_outcome(ctx) or "death"
    score_var = _pick_score(ctx)

    # Split figure steps inherit the parent id plus ``_figure``.  Route them
    # before any science-step matcher so a primary-association figure cannot
    # accidentally refit the model or reread the cohort.
    if step_id.endswith("_figure") or (
        expected_outputs
        and all(
            str(output).lower().startswith("figure:") for output in expected_outputs
        )
    ):
        return _mock_code_declared_figure(step_id=step_id, prompt=prompt)
    if "publication_figure_generation" in step_id:
        return _mock_code_publication_figure(
            ctx=ctx,
            step_id=step_id,
            outcome=outcome,
        )
    if re.search(r"(?:^|_)primary_association$", step_id):
        primary_pred = (
            _pick_primary_predictor(ctx, outcome=outcome) or score_var or "age"
        )
        return _mock_code_primary_association(
            ctx=ctx,
            step_id=step_id,
            outcome=outcome,
            predictor=primary_pred,
        )
    if "prediction_model_analysis" in step_id:
        return _mock_code_prediction_model(
            ctx=ctx,
            step_id=step_id,
            outcome=outcome,
        )
    if "trajectory_clustering_analysis" in step_id:
        return _mock_code_trajectory_clustering(
            ctx=ctx,
            step_id=step_id,
            outcome=outcome,
        )
    # Inline script as a triple-quoted heredoc — note: keep this tight; the
    # runner persists it byte-for-byte and hashes it as evidence.
    code = (
        textwrap.dedent(
            f"""
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

        cohort_path = os.environ["COHORT_PARQUET"]
        out_dir = Path(os.environ["STEP_OUT_DIR"])
        out_dir.mkdir(parents=True, exist_ok=True)

        df = pd.read_parquet(cohort_path)
        step_kind = {step_id!r}.lower()
        do_table_one = "table_one" in step_kind
        do_outcome_incidence = "outcome_incidence" in step_kind
        do_missingness = "missingness" in step_kind
        do_protocol_only = any(token in step_kind for token in ("protocol", "plan"))

        outcome_col = {outcome!r} if {outcome!r} in df.columns else None
        score_col = {score_var!r} if {score_var!r} else None
        if score_col and score_col not in df.columns:
            score_col = None
        if not any((do_table_one, do_outcome_incidence, do_missingness)):
            if do_protocol_only:
                do_table_one = False
                do_outcome_incidence = False
                do_missingness = False
            else:
                do_table_one = True
                do_outcome_incidence = True
                do_missingness = True

        summary = {{"output_files": {{}}}}

        if do_protocol_only:
            protocol_lines = [
                f"# Protocol note for {{step_kind}}",
                f"- Research question: {ctx.research_question}",
                "- Available variables: " + ", ".join(df.columns.astype(str).tolist()),
                "- This is a task-family planning/protocol step rather than a finished effect estimate.",
            ]
            (out_dir / "protocol_notes.md").write_text("\\n".join(protocol_lines), encoding="utf-8")
            summary["protocol_notes_path"] = "protocol_notes.md"
            summary["output_files"][{protocol_output!r}] = "protocol_notes.md"

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
                    if score_col is not None and col == score_col:
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
            summary["output_files"]["table:table_one"] = "table_one.csv"

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
            summary["output_files"]["table:outcome_incidence"] = "outcome_incidence.csv"

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
            summary["output_files"]["table:missingness"] = "missingness.csv"

        # ---- Persist machine-readable summary ----
        with open(out_dir / "step_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
        print(json.dumps(summary, indent=2, ensure_ascii=False, default=str))
        """
        ).strip()
        + "\n"
    )
    return code


def _mock_code_declared_figure(*, step_id: str, prompt: str) -> str:
    """Render a split figure step from registered upstream products only."""

    products = [
        output.partition(":")[2].lower()
        for output in _extract_expected_outputs(prompt)
        if output.lower().startswith("figure:")
    ]
    products = list(dict.fromkeys(product for product in products if product))
    if not products:
        products = ["publication_figure"]
    template = r"""
    # AUTO-GENERATED by easyicu.research_agent.MockLLMClient
    # rendering-only figure step; never reads COHORT_PARQUET or refits a model
    from __future__ import annotations
    import json
    import math
    import os
    import shutil
    from pathlib import Path

    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    step_id = __STEP_ID__
    figure_products = __FIGURE_PRODUCTS__
    out_dir = Path(os.environ["STEP_OUT_DIR"])
    out_dir.mkdir(parents=True, exist_ok=True)
    run_dir = Path(os.environ.get("EASYICU_RUN_DIR") or out_dir.parents[2])

    manifest_path = Path(os.environ["EASYICU_RESOLVED_INPUTS_JSON"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source_candidates = []
    for declared_product, binding in manifest.get("inputs", {}).items():
        relative_path = binding.get("relative_path")
        if relative_path:
            source_candidates.append({
                "declared_product": declared_product,
                "path": run_dir / relative_path,
                "evidence_id": binding.get("evidence_id"),
                "sha256": binding.get("sha256"),
                "evidence_kind": binding.get("evidence_kind"),
                "product": binding.get("product"),
                "produced_by_step": binding.get("produced_by_step"),
            })

    csv_sources = list(dict.fromkeys(
        (item["path"], item["evidence_id"])
        for item in source_candidates
        if item["path"].is_file() and item["path"].suffix.lower() == ".csv"
    ))
    if len(csv_sources) != 1:
        raise RuntimeError(
            f"Expected exactly one registered upstream CSV for render-only step "
            f"{step_id}; found {len(csv_sources)}"
        )
    source_path, source_evidence_id = csv_sources[0]
    if not source_evidence_id:
        raise RuntimeError(
            f"Registered upstream CSV has no evidence id for render-only step {step_id}"
        )
    source = pd.read_csv(source_path)
    if source.empty:
        raise RuntimeError(f"Registered upstream table is empty: {source_path}")

    def _normalise(value):
        return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")

    def _named_scalars(payload, target):
        values = []
        if isinstance(payload, dict):
            declared_name = payload.get("name") or payload.get("statistic")
            if _normalise(declared_name) == target:
                for field in ("value", "estimate", "result"):
                    scalar = payload.get(field)
                    if isinstance(scalar, (int, float)) and not isinstance(scalar, bool):
                        values.append(float(scalar))
            for key, child in payload.items():
                if _normalise(key) == target:
                    if isinstance(child, (int, float)) and not isinstance(child, bool):
                        values.append(float(child))
                if isinstance(child, (dict, list, tuple)):
                    values.extend(_named_scalars(child, target))
        elif isinstance(payload, (list, tuple)):
            for child in payload:
                values.extend(_named_scalars(child, target))
        return values

    statistic_rows = []
    for item in source_candidates:
        declared_product = str(item["declared_product"] or "")
        if not declared_product.lower().startswith("statistic:"):
            continue
        statistic_path = item["path"]
        if not statistic_path.is_file() or statistic_path.suffix.lower() != ".json":
            raise RuntimeError(
                f"Bound statistic {declared_product} has no readable JSON evidence"
            )
        product_name = str(
            item.get("product") or declared_product.partition(":")[2]
        ).strip()
        payload = json.loads(statistic_path.read_text(encoding="utf-8"))
        scalars = [
            value
            for value in _named_scalars(payload, _normalise(product_name))
            if math.isfinite(value)
        ]
        unique_scalars = []
        for value in scalars:
            if not any(math.isclose(value, seen, rel_tol=1e-9, abs_tol=1e-9) for seen in unique_scalars):
                unique_scalars.append(value)
        if len(unique_scalars) != 1:
            raise RuntimeError(
                f"Bound statistic {declared_product} must resolve to one finite scalar; "
                f"found {len(unique_scalars)}"
            )
        statistic_rows.append({
            "statistic": product_name,
            "value": unique_scalars[0],
            "source_step_id": item.get("produced_by_step"),
            "source_evidence_id": item.get("evidence_id"),
        })

    source_evidence_ids = list(dict.fromkeys(
        str(item["evidence_id"])
        for item in source_candidates
        if item.get("evidence_id")
    ))
    input_bindings = []
    for item in source_candidates:
        receipt = {
            "input_key": str(item["declared_product"]),
            "loaded": True,
            "evidence_id": item.get("evidence_id"),
            "sha256": item.get("sha256"),
        }
        if item["path"] == source_path:
            receipt["row_count"] = int(len(source))
        elif not str(item["declared_product"]).lower().startswith("statistic:"):
            raise RuntimeError(
                f"Unsupported bound input for render-only step {step_id}: "
                f"{item['declared_product']}"
            )
        input_bindings.append(receipt)

    output_files = {}
    figure_files = []
    contract_files = []
    source_data_files = []
    for product in figure_products:
        source_copy = out_dir / f"{product}.source.csv"
        shutil.copy2(source_path, source_copy)
        source_data_files.append(source_copy.name)
        product_source_files = [source_copy.name]
        if statistic_rows:
            statistic_source = out_dir / f"{product}.statistics.source.csv"
            pd.DataFrame(statistic_rows).to_csv(statistic_source, index=False)
            source_data_files.append(statistic_source.name)
            product_source_files.append(statistic_source.name)

        fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.0))
        ax, audit_ax = axes
        if {"variable", "odds_ratio"} <= set(source.columns):
            plot = source[source["variable"].astype(str) != "intercept"].copy()
            y = np.arange(len(plot))
            if {"or_lower", "or_upper"} <= set(plot.columns):
                estimate = pd.to_numeric(plot["odds_ratio"], errors="coerce")
                lower = pd.to_numeric(plot["or_lower"], errors="coerce")
                upper = pd.to_numeric(plot["or_upper"], errors="coerce")
                xerr = [np.maximum(0, estimate - lower), np.maximum(0, upper - estimate)]
            else:
                estimate = pd.to_numeric(plot["odds_ratio"], errors="coerce")
                xerr = None
            ax.errorbar(estimate, y, xerr=xerr, fmt="o", color="#2a6f97")
            ax.axvline(1.0, linestyle="--", color="grey", linewidth=0.8)
            ax.set_yticks(y, plot["variable"].astype(str))
            ax.set_xlabel("Odds ratio")
            ax.set_title("Adjusted association")
            audit_values = pd.to_numeric(
                plot.get("p_value", pd.Series(np.nan, index=plot.index)),
                errors="coerce",
            )
            audit_ax.barh(plot["variable"].astype(str), audit_values, color="#c97c5d")
            audit_ax.axvline(0.05, linestyle="--", color="grey", linewidth=0.8)
            audit_ax.set_xlabel("P value")
            audit_ax.set_title("Model-table audit context")
        elif {"variable", "frac_missing"} <= set(source.columns):
            plot = source.sort_values("frac_missing", ascending=True).tail(12)
            ax.barh(
                plot["variable"].astype(str),
                pd.to_numeric(plot["frac_missing"], errors="coerce"),
                color="#7aa6d1",
            )
            ax.set_xlabel("Fraction missing")
            ax.set_title("Missingness audit")
            count_values = pd.to_numeric(
                plot.get("n_missing", pd.Series(0, index=plot.index)),
                errors="coerce",
            )
            audit_ax.barh(plot["variable"].astype(str), count_values, color="#c97c5d")
            audit_ax.set_xlabel("Missing rows")
            audit_ax.set_title("Missing-count context")
        else:
            numeric = source.select_dtypes(include=["number"])
            if numeric.empty:
                raise RuntimeError(
                    f"Registered upstream table has no plottable numeric column: {source_path}"
                )
            values = pd.to_numeric(numeric.iloc[:20, 0], errors="coerce")
            ax.plot(np.arange(len(values)), values, marker="o", color="#2a6f97")
            ax.set_xlabel("Row")
            ax.set_ylabel(str(numeric.columns[0]))
            ax.set_title(product.replace("_", " ").title())
            audit_column = numeric.columns[min(1, len(numeric.columns) - 1)]
            audit_values = pd.to_numeric(
                numeric[audit_column].iloc[:20], errors="coerce"
            )
            audit_ax.plot(
                np.arange(len(audit_values)), audit_values, marker="o", color="#c97c5d"
            )
            audit_ax.set_xlabel("Row")
            audit_ax.set_ylabel(str(audit_column))
            audit_ax.set_title("Registered-table context")
        fig.tight_layout()

        png_path = out_dir / f"{product}.png"
        svg_path = out_dir / f"{product}.svg"
        fig.savefig(png_path, dpi=200)
        fig.savefig(svg_path)
        plt.close(fig)

        contract = {
            "figure_id": product,
            "title": product.replace("_", " ").title(),
            "core_claim": "Rendering of the registered upstream table without scientific recomputation.",
            "panels": [
                {
                    "panel_id": "A",
                    "title": product.replace("_", " ").title(),
                    "role": "descriptive_result",
                    "claim": "Values are rendered directly from the registered upstream table.",
                    "evidence_ids": source_evidence_ids,
                    "review_risk": "Interpretation remains owned by the upstream analysis step.",
                },
                {
                    "panel_id": "B",
                    "title": "Registered-table audit context",
                    "role": "audit",
                    "claim": "A second view exposes supporting values from the same registered table.",
                    "evidence_ids": source_evidence_ids,
                    "review_risk": "This panel adds no new estimand or model fit.",
                }
            ],
            "export_formats": ["png", "svg"],
            "source_data": product_source_files,
            "statistics_note": "No model fitting or cohort transformation occurs in this step.",
            "image_integrity_note": "The figure is drawn directly from copied upstream source data.",
        }
        contract_path = out_dir / f"{product}.figure_contract.json"
        contract_path.write_text(
            json.dumps(contract, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        output_files[f"figure:{product}"] = png_path.name
        figure_files.extend([png_path.name, svg_path.name])
        contract_files.append(contract_path.name)

    summary = {
        "method": "registered_product_rendering",
        "render_only": True,
        "upstream_source": str(source_path.relative_to(run_dir)),
        "upstream_sources": [
            str(item["path"].relative_to(run_dir)) for item in source_candidates
        ],
        "output_files": output_files,
        "figure_files": figure_files,
        "figure_contract_files": contract_files,
        "source_data_files": source_data_files,
        "input_bindings": input_bindings,
    }
    (out_dir / "step_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    """
    return (
        textwrap.dedent(template)
        .replace("__STEP_ID__", json.dumps(step_id))
        .replace("__FIGURE_PRODUCTS__", json.dumps(products))
    )


def _mock_code_primary_association(
    *,
    ctx: ResearchContext,
    step_id: str,
    outcome: str,
    predictor: str,
    adjust: Sequence[str] | None = None,
    typed_model_contract: bool = False,
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
    adjustment_candidates = list(adjust) if adjust is not None else ["age", "sex"]
    output_name = (
        "adjusted_association_estimates.csv"
        if typed_model_contract
        else "primary_association.csv"
    )
    output_product = (
        "table:adjusted_association_estimates"
        if typed_model_contract
        else "table:primary_association"
    )
    requirement_id = f"primary_{predictor}_{outcome}"
    code = (
        textwrap.dedent(
            f"""
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
        adjustment_candidates = {adjustment_candidates!r}
        cols_needed = [
            c
            for c in [predictor_col, outcome_col, *adjustment_candidates]
            if c in df.columns
        ]
        sub = df[cols_needed].dropna().copy()

        # one-hot the binary sex column (M -> 1, F -> 0); leave any other
        # categorical sex coding alone with a warning written to summary.
        sex_used = False
        if "sex" in adjustment_candidates and "sex" in sub.columns:
            uniq = set(map(str, sub["sex"].dropna().unique()))
            if uniq <= {{"M", "F"}}:
                sub["sex_M"] = (sub["sex"].astype(str) == "M").astype(int)
                sex_used = True
            sub = sub.drop(columns=["sex"])

        terms = [predictor_col]
        for candidate in adjustment_candidates:
            if candidate == "sex" and sex_used:
                terms.append("sex_M")
            elif candidate in sub.columns and candidate not in terms:
                terms.append(candidate)

        y = sub[outcome_col].astype(int).to_numpy()
        X = np.column_stack([np.ones(len(sub))] + [sub[t].astype(float).to_numpy() for t in terms])
        names = ["intercept"] + terms

        # ---- Fit ----
        coefs = None
        cov = None
        backend = "manual"
        optimizer_success = False
        try:
            import statsmodels.api as sm  # type: ignore
            res = sm.Logit(y, X).fit(disp=0, method="newton", maxiter=200)
            coefs = np.asarray(res.params, dtype=float)
            cov = np.asarray(res.cov_params(), dtype=float)
            backend = "statsmodels"
            optimizer_success = bool(
                getattr(res, "mle_retvals", {{}}).get("converged", True)
            )
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
                optimizer_success = bool(opt.success)
            except Exception as exc:
                # Last-ditch: skip but still write a parseable artefact so the
                # pipeline doesn't crash; downstream validator will flag it.
                pd.DataFrame([{{"variable": "(skipped)", "reason": str(exc)}}]).to_csv(
                    out_dir / {output_name!r}, index=False)
                with open(out_dir / "step_summary.json", "w", encoding="utf-8") as f:
                    json.dump({{
                        "method": "logistic_regression",
                        "predictor": predictor_col,
                        "outcome": outcome_col,
                        "primary_or": None,
                        "skipped": True,
                        "reason": str(exc),
                        "output_files": {{
                            {output_product!r}: {output_name!r}
                        }},
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
            source_variable = (
                predictor_col
                if nm == predictor_col
                else "sex"
                if nm == "sex_M"
                else nm
            )
            rows.append({{
                "variable": nm,
                "model_id": {requirement_id!r},
                "term": nm,
                "term_role": (
                    "intercept"
                    if nm == "intercept"
                    else "exposure"
                    if nm == predictor_col
                    else "adjustment"
                ),
                "source_variable": source_variable,
                "coef": float(b),
                "std_err": float(s),
                "standard_error": float(s),
                "ci_lower": float(lo),
                "ci_upper": float(hi),
                "ci_low": float(np.exp(lo)),
                "ci_high": float(np.exp(hi)),
                "odds_ratio": float(np.exp(b)),
                "or_lower": float(np.exp(lo)),
                "or_upper": float(np.exp(hi)),
                "p_value": float(pv) if not np.isnan(pv) else None,
                "effect_scale": "odds_ratio",
            }})
        coef_df = pd.DataFrame(rows)
        coef_df.to_csv(out_dir / {output_name!r}, index=False)

        primary_or = float(np.exp(coefs[names.index(predictor_col)]))
        primary_or_lo = float(np.exp(ci_lo[names.index(predictor_col)]))
        primary_or_hi = float(np.exp(ci_hi[names.index(predictor_col)]))

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
            "primary_association_path": {output_name!r},
            "output_files": {{
                {output_product!r}: {output_name!r}
            }},
        }}
        if {typed_model_contract!r}:
            summary["model_contracts"] = [{{
                "model_id": {requirement_id!r},
                "requirement_id": {requirement_id!r},
                "outcome": outcome_col,
                "outcome_type": "binary",
                "method_family": "logistic_regression",
                "model_family": "logistic_regression",
                "exposure_source": predictor_col,
                "exposure_expression": predictor_col,
                "exposure_role": "primary",
                "analysis_role": "primary",
                "analysis_set": "complete_case",
                "baseline_missing_policy": "drop_missing_baseline",
                "n": int(len(sub)),
                "event_n": int(np.sum(y)),
                "fit_status": "fitted",
                "converged": bool(optimizer_success),
                "separation_detected": False,
                "penalized": False,
                "fit_method": backend,
                "convergence_method": "optimizer_success",
                "optimizer_success": bool(optimizer_success),
            }}]
        with open(out_dir / "step_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
        print(json.dumps(summary, indent=2, ensure_ascii=False, default=str))
        """
        ).strip()
        + "\n"
    )
    return code


def _mock_code_prediction_model(
    *, ctx: ResearchContext, step_id: str, outcome: str
) -> str:
    template = r"""
    # AUTO-GENERATED by easyicu.research_agent.MockLLMClient
    # step_id: __STEP_ID__
    # research_question: __QUESTION__
    from __future__ import annotations
    import json
    import math
    import os
    from pathlib import Path

    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    outcome_col = __OUTCOME__
    out_dir = Path(os.environ["STEP_OUT_DIR"])
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(os.environ["COHORT_PARQUET"])

    def to_jsonable(x):
        if isinstance(x, (np.integer, )):
            return int(x)
        if isinstance(x, (np.floating, )):
            v = float(x)
            return v if math.isfinite(v) else None
        if isinstance(x, (np.bool_, )):
            return bool(x)
        if isinstance(x, np.ndarray):
            return x.tolist()
        try:
            if pd.isna(x):
                return None
        except Exception:
            pass
        return x

    def sigmoid(z):
        return 1.0 / (1.0 + np.exp(-z))

    def fit_logit(model_df, y_col, x_cols):
        y = model_df[y_col].astype(float).to_numpy()
        X = model_df[x_cols].astype(float).to_numpy()
        X = np.column_stack([np.ones(len(X)), X])
        names = ["intercept"] + list(x_cols)
        backend = "scipy_bfgs"
        try:
            import statsmodels.api as sm  # type: ignore
            res = sm.Logit(y, X).fit(disp=0, method="newton", maxiter=200)
            coef = np.asarray(res.params, dtype=float)
            cov = np.asarray(res.cov_params(), dtype=float)
            backend = "statsmodels"
            return coef, cov, names, backend
        except Exception:
            from scipy import optimize  # type: ignore

            def neg_ll(beta):
                z = X @ beta
                log_ll = np.where(z >= 0, np.log1p(np.exp(-z)), -z + np.log1p(np.exp(z)))
                return float(np.sum((1 - y) * z + log_ll))

            def grad(beta):
                p = sigmoid(X @ beta)
                return X.T @ (p - y)

            beta0 = np.zeros(X.shape[1], dtype=float)
            opt = optimize.minimize(neg_ll, beta0, jac=grad, method="BFGS")
            coef = np.asarray(opt.x, dtype=float)
            p = sigmoid(X @ coef)
            W = p * (1 - p)
            fisher = (X.T * W) @ X
            cov = np.linalg.pinv(fisher)
            return coef, cov, names, backend

    def auc_rank(y_true, scores):
        y_true = np.asarray(y_true).astype(int)
        scores = np.asarray(scores, dtype=float)
        pos = int(y_true.sum())
        neg = int(len(y_true) - pos)
        if pos == 0 or neg == 0:
            return None
        order = np.argsort(scores)
        ranks = np.empty(len(scores), dtype=float)
        ranks[order] = np.arange(1, len(scores) + 1)
        sum_ranks_pos = float(ranks[y_true == 1].sum())
        auc = (sum_ranks_pos - pos * (pos + 1) / 2.0) / (pos * neg)
        return float(auc)

    def roc_curve_points(y_true, scores):
        y_true = np.asarray(y_true).astype(int)
        scores = np.asarray(scores, dtype=float)
        order = np.argsort(-scores)
        y_sorted = y_true[order]
        tp = np.cumsum(y_sorted == 1)
        fp = np.cumsum(y_sorted == 0)
        pos = max(int((y_true == 1).sum()), 1)
        neg = max(int((y_true == 0).sum()), 1)
        tpr = np.concatenate([[0.0], tp / pos, [1.0]])
        fpr = np.concatenate([[0.0], fp / neg, [1.0]])
        thr = np.concatenate([[scores.max() + 1e-6], scores[order], [scores.min() - 1e-6]])
        return pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": thr})

    if "patient_stay_id" in df.columns:
        patient_groups = df["patient_stay_id"].astype(str).str.split(":s").str[0]
        patient_group_source = "patient_stay_id_prefix_before_:s"
    elif "subject_id" in df.columns:
        patient_groups = df["subject_id"].astype(str)
        patient_group_source = "subject_id"
    elif "patient_id" in df.columns:
        patient_groups = df["patient_id"].astype(str)
        patient_group_source = "patient_id"
    else:
        raise SystemExit(
            "Prediction preflight requires a patient-level grouping column; "
            "row-level or stay-level splitting is forbidden."
        )

    feature_order = ["sofa2", "lact", "creat", "map", "hr", "resp", "spo2", "vaso", "age", "sex"]
    features = [c for c in feature_order if c in df.columns and c != outcome_col]
    model_df = df[[outcome_col] + features].copy()
    model_df["_patient_group"] = patient_groups
    if "sex" in model_df.columns:
        model_df["sex_M"] = (model_df["sex"].astype(str) == "M").astype(int)
        model_df = model_df.drop(columns=["sex"])
        features = ["sex_M" if c == "sex" else c for c in features]
    model_df[[outcome_col] + features] = model_df[
        [outcome_col] + features
    ].apply(pd.to_numeric, errors="coerce")
    model_df = model_df.replace([np.inf, -np.inf], np.nan).dropna()
    model_df[outcome_col] = model_df[outcome_col].astype(int)

    if len(model_df) < 80:
        raise SystemExit("Not enough complete cases for prediction-model example.")

    rng = np.random.default_rng(7)
    unique_groups = np.asarray(sorted(model_df["_patient_group"].unique()))
    if len(unique_groups) < 10:
        raise SystemExit("Prediction preflight requires at least 10 patient groups.")
    shuffled_groups = rng.permutation(unique_groups)
    split = min(max(int(0.7 * len(shuffled_groups)), 2), len(shuffled_groups) - 1)
    train_group_set = set(shuffled_groups[:split].tolist())
    test_group_set = set(shuffled_groups[split:].tolist())
    overlap = sorted(train_group_set & test_group_set)
    if overlap:
        raise RuntimeError("patient-level train/test split overlap detected")
    train = model_df[model_df["_patient_group"].isin(train_group_set)].copy()
    test = model_df[model_df["_patient_group"].isin(test_group_set)].copy()
    if train.empty or test.empty:
        raise RuntimeError("patient-grouped split produced an empty partition")

    coef, cov, names, backend = fit_logit(train, outcome_col, features)
    X_test = np.column_stack([np.ones(len(test)), test[features].astype(float).to_numpy()])
    risk = sigmoid(X_test @ coef)
    y_test = test[outcome_col].astype(int).to_numpy()

    auc = auc_rank(y_test, risk)
    brier = float(np.mean((risk - y_test) ** 2))
    predicted_class = (risk >= 0.5).astype(int)
    true_positive = int(np.sum((predicted_class == 1) & (y_test == 1)))
    false_positive = int(np.sum((predicted_class == 1) & (y_test == 0)))
    false_negative = int(np.sum((predicted_class == 0) & (y_test == 1)))
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    score_order = np.argsort(-risk)
    sorted_outcome = y_test[score_order]
    cumulative_precision = np.cumsum(sorted_outcome) / np.arange(1, len(y_test) + 1)
    average_precision = float(
        np.sum(cumulative_precision * sorted_outcome) / max(int(y_test.sum()), 1)
    )
    logit_pred = np.log(np.clip(risk, 1e-6, 1 - 1e-6) / np.clip(1 - risk, 1e-6, 1 - 1e-6))
    cal_df = pd.DataFrame({"death": y_test, "logit_pred": logit_pred})
    cal_slope = None
    try:
        if cal_df["death"].nunique() > 1:
            cal_coef, _, cal_names, _ = fit_logit(cal_df, "death", ["logit_pred"])
            cal_slope = float(cal_coef[cal_names.index("logit_pred")])
    except Exception:
        cal_slope = None

    se = np.sqrt(np.maximum(np.diag(cov), 0.0))
    ci_lo = coef - 1.959963984540054 * se
    ci_hi = coef + 1.959963984540054 * se
    coef_rows = []
    for name, beta, lo, hi in zip(names, coef, ci_lo, ci_hi):
        coef_rows.append({
            "variable": name,
            "coef": float(beta),
            "odds_ratio": float(np.exp(beta)),
            "or_lower": float(np.exp(lo)),
            "or_upper": float(np.exp(hi)),
        })
    coef_df = pd.DataFrame(coef_rows)
    coef_df.to_csv(out_dir / "model_coefficients.csv", index=False)

    perf_df = pd.DataFrame([{
        "model": "logistic_regression",
        "backend": backend,
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "auc": auc,
        "auroc": auc,
        "average_precision": average_precision,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "brier": brier,
        "calibration_slope": cal_slope,
    }])
    perf_df.to_csv(out_dir / "model_performance_train_test.csv", index=False)

    risk_df = test[["_patient_group", *features, outcome_col]].copy()
    risk_df = risk_df.rename(columns={"_patient_group": "patient_group"})
    risk_df["predicted_risk"] = risk
    risk_df.to_csv(out_dir / "risk_predictions_test.csv", index=False)

    roc_df = roc_curve_points(y_test, risk)
    roc_df.to_csv(out_dir / "roc_curve.csv", index=False)

    cal_bins = pd.qcut(pd.Series(risk), q=min(10, max(3, len(risk) // 30)), duplicates="drop")
    calibration = pd.DataFrame({"predicted_risk": risk, "death": y_test, "bin": cal_bins})
    cal_curve = calibration.groupby("bin", observed=False).agg(
        predicted_mean=("predicted_risk", "mean"),
        observed_rate=("death", "mean"),
        n_bin=("death", "size"),
    ).reset_index(drop=True)
    cal_curve.to_csv(out_dir / "calibration_curve.csv", index=False)

    thresholds = np.linspace(0.05, 0.50, 10)
    decision_rows = []
    prevalence = float(y_test.mean())
    for threshold in thresholds:
        predicted_positive = risk >= threshold
        tp = float(np.sum(predicted_positive & (y_test == 1)))
        fp = float(np.sum(predicted_positive & (y_test == 0)))
        odds = threshold / (1.0 - threshold)
        decision_rows.append({
            "threshold": float(threshold),
            "net_benefit_model": (tp / len(y_test)) - (fp / len(y_test)) * odds,
            "net_benefit_all": prevalence - (1.0 - prevalence) * odds,
            "net_benefit_none": 0.0,
        })
    pd.DataFrame(decision_rows).to_csv(
        out_dir / "decision_curve.csv", index=False
    )
    pd.DataFrame([{
        "patient_group_source": patient_group_source,
        "n_train_rows": int(len(train)),
        "n_test_rows": int(len(test)),
        "n_train_patients": int(len(train_group_set)),
        "n_test_patients": int(len(test_group_set)),
        "patient_overlap_n": int(len(overlap)),
        "preprocessing_fit_scope": "training_partition_only",
    }]).to_csv(out_dir / "split_definition.csv", index=False)

    fig, ax = plt.subplots(figsize=(4.0, 3.2))
    ax.plot(roc_df["fpr"], roc_df["tpr"], color="#1f77b4", linewidth=1.6)
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", linewidth=0.8)
    ax.set_xlabel("False-positive rate")
    ax.set_ylabel("True-positive rate")
    ax.set_title("ROC curve")
    fig.tight_layout()
    fig.savefig(out_dir / "roc_curve.png", dpi=160)
    fig.savefig(out_dir / "roc_curve.svg")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.0, 3.2))
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", linewidth=0.8)
    ax.plot(cal_curve["predicted_mean"], cal_curve["observed_rate"], marker="o", color="#d1495b", linewidth=1.3)
    ax.set_xlabel("Predicted risk")
    ax.set_ylabel("Observed risk")
    ax.set_title("Calibration")
    fig.tight_layout()
    fig.savefig(out_dir / "calibration_curve.png", dpi=160)
    fig.savefig(out_dir / "calibration_curve.svg")
    plt.close(fig)

    summary = {
        "method": "prediction_model_analysis",
        "backend": backend,
        "target_outcome": outcome_col,
        "features": features,
        "n_complete_cases": int(len(model_df)),
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "patient_group_source": patient_group_source,
        "patient_overlap_n": int(len(overlap)),
        "auc": auc,
        "auroc": auc,
        "average_precision": average_precision,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "brier": brier,
        "calibration_slope": cal_slope,
        "output_files": {
            "table:model_performance_train_test": "model_performance_train_test.csv",
            "table:model_coefficients": "model_coefficients.csv",
            "table:risk_predictions_test": "risk_predictions_test.csv",
            "table:roc_curve": "roc_curve.csv",
            "table:calibration_curve": "calibration_curve.csv",
            "table:decision_curve": "decision_curve.csv",
            "table:split_definition": "split_definition.csv",
            "figure:roc_curve": "roc_curve.png",
            "figure:calibration_curve": "calibration_curve.png",
            "statistic:auc": "model_performance_train_test.csv",
        },
        "outputs": {
            "performance_table": "model_performance_train_test.csv",
            "coefficients_table": "model_coefficients.csv",
            "risk_predictions": "risk_predictions_test.csv",
            "roc_curve": "roc_curve.png",
            "calibration_curve": "calibration_curve.png",
            "decision_curve": "decision_curve.csv",
            "split_definition": "split_definition.csv",
        },
    }
    with open(out_dir / "step_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=to_jsonable)
    print(json.dumps(summary, indent=2, ensure_ascii=False, default=to_jsonable))
    """
    return (
        textwrap.dedent(template)
        .replace("__STEP_ID__", step_id)
        .replace("__QUESTION__", json.dumps(ctx.research_question))
        .replace("__OUTCOME__", json.dumps(outcome))
    )


def _mock_code_trajectory_clustering(
    *, ctx: ResearchContext, step_id: str, outcome: str
) -> str:
    template = r"""
    # AUTO-GENERATED by easyicu.research_agent.MockLLMClient
    # step_id: __STEP_ID__
    # research_question: __QUESTION__
    from __future__ import annotations
    import json
    import math
    import os
    import re
    from pathlib import Path

    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    outcome_col = __OUTCOME__
    out_dir = Path(os.environ["STEP_OUT_DIR"])
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(os.environ["COHORT_PARQUET"])

    def to_jsonable(x):
        if isinstance(x, (np.integer, )):
            return int(x)
        if isinstance(x, (np.floating, )):
            v = float(x)
            return v if math.isfinite(v) else None
        if isinstance(x, (np.bool_, )):
            return bool(x)
        if isinstance(x, np.ndarray):
            return x.tolist()
        try:
            if pd.isna(x):
                return None
        except Exception:
            pass
        return x

    def suffix_key(name):
        m = re.search(r"_t(\d+)$", str(name))
        if m:
            return float(m.group(1))
        m = re.search(r"_h(\d+(?:p\d+)?)_(\d+(?:p\d+)?)$", str(name))
        return float(m.group(1).replace("p", ".")) if m else 0.0

    window_suffix = r"(?:t\d+|h\d+(?:p\d+)?_\d+(?:p\d+)?)"
    lact_cols = sorted(
        [c for c in df.columns if re.fullmatch(rf"lact_{window_suffix}", str(c))],
        key=suffix_key,
    )
    map_cols = sorted(
        [c for c in df.columns if re.fullmatch(rf"map_{window_suffix}", str(c))],
        key=suffix_key,
    )
    if not lact_cols or not map_cols:
        raise SystemExit("Trajectory clustering example requires lact_t* and map_t* columns.")

    panel = df[[outcome_col] + lact_cols + map_cols].dropna().copy()
    panel["lact_mean"] = panel[lact_cols].mean(axis=1)
    panel["lact_slope"] = panel[lact_cols[-1]] - panel[lact_cols[0]]
    panel["map_mean"] = panel[map_cols].mean(axis=1)
    panel["map_slope"] = panel[map_cols[-1]] - panel[map_cols[0]]
    feat_cols = ["lact_mean", "lact_slope", "map_mean", "map_slope"]
    feat = panel[feat_cols].copy()
    feat = (feat - feat.mean()) / feat.std(ddof=0).replace(0, 1)

    labels = None
    method = "rule_based_fallback"
    try:
        from scipy.cluster.vq import kmeans2  # type: ignore
        np.random.seed(7)
        _, labels = kmeans2(feat.to_numpy(), 3, minit="points", iter=30)
        method = "scipy_kmeans2"
    except Exception:
        high_lact = panel["lact_mean"] >= panel["lact_mean"].quantile(0.67)
        low_map = panel["map_mean"] <= panel["map_mean"].quantile(0.33)
        labels = np.where(high_lact & low_map, 2, np.where(high_lact | low_map, 1, 0))

    panel["cluster_raw"] = labels.astype(int)
    cluster_outcomes = panel.groupby("cluster_raw", observed=True).agg(
        n=("cluster_raw", "size"),
        mortality_rate=(outcome_col, "mean"),
        lact_mean=("lact_mean", "mean"),
        map_mean=("map_mean", "mean"),
    ).reset_index()
    order = cluster_outcomes.sort_values(["mortality_rate", "lact_mean"]).reset_index(drop=True)
    remap = {int(old): int(new) for new, old in enumerate(order["cluster_raw"].tolist())}
    panel["cluster"] = panel["cluster_raw"].map(remap).astype(int)

    cluster_outcomes = panel.groupby("cluster", observed=True).agg(
        n=("cluster", "size"),
        mortality_rate=(outcome_col, "mean"),
        lact_mean=("lact_mean", "mean"),
        map_mean=("map_mean", "mean"),
        lact_slope_mean=("lact_slope", "mean"),
        map_slope_mean=("map_slope", "mean"),
    ).reset_index()
    cluster_outcomes.to_csv(out_dir / "cluster_outcomes.csv", index=False)

    assign = panel[["cluster", outcome_col] + lact_cols + map_cols].copy()
    assign.to_csv(out_dir / "cluster_assignments.csv", index=False)

    traj_rows = []
    for cluster_id, sub in panel.groupby("cluster", observed=True):
        for col in lact_cols:
            traj_rows.append({
                "cluster": int(cluster_id),
                "domain": "lactate",
                "timepoint": col,
                "value_mean": float(sub[col].mean()),
            })
        for col in map_cols:
            traj_rows.append({
                "cluster": int(cluster_id),
                "domain": "map",
                "timepoint": col,
                "value_mean": float(sub[col].mean()),
            })
    traj_df = pd.DataFrame(traj_rows)
    traj_df.to_csv(out_dir / "cluster_trajectory_means.csv", index=False)

    centroids = panel.groupby("cluster", observed=True)[feat_cols].mean().to_numpy()
    within = 0.0
    count = 0
    for cluster_id, sub in panel.groupby("cluster", observed=True):
        c = sub[feat_cols].mean().to_numpy()
        within += float(((sub[feat_cols].to_numpy() - c) ** 2).sum())
        count += len(sub)
    within = within / max(count, 1)
    between = 0.0
    if len(centroids) > 1:
        dists = []
        for i in range(len(centroids)):
            for j in range(i + 1, len(centroids)):
                dists.append(float(np.linalg.norm(centroids[i] - centroids[j])))
        between = float(np.mean(dists)) if dists else 0.0
    stability_proxy = between / max(within, 1e-6)

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0), constrained_layout=True)
    colors = ["#2a6f97", "#c97c5d", "#6a994e", "#7b5ea7"]
    for idx, (cluster_id, sub) in enumerate(panel.groupby("cluster", observed=True)):
        c = colors[idx % len(colors)]
        axes[0].plot(range(len(lact_cols)), [float(sub[col].mean()) for col in lact_cols], marker="o", color=c, label=f"Cluster {int(cluster_id)}")
        axes[1].plot(range(len(map_cols)), [float(sub[col].mean()) for col in map_cols], marker="o", color=c, label=f"Cluster {int(cluster_id)}")
    axes[0].set_xticks(range(len(lact_cols)), lact_cols, rotation=0)
    axes[1].set_xticks(range(len(map_cols)), map_cols, rotation=0)
    axes[0].set_ylabel("Mean lactate")
    axes[1].set_ylabel("Mean MAP")
    axes[0].set_title("Lactate trajectories")
    axes[1].set_title("MAP trajectories")
    axes[1].legend(frameon=False, fontsize=7)
    fig.savefig(out_dir / "trajectory_clusters.png", dpi=160)
    fig.savefig(out_dir / "trajectory_clusters.svg")
    plt.close(fig)

    summary = {
        "method": "trajectory_clustering_analysis",
        "backend": method,
        "target_outcome": outcome_col,
        "n_clusters": int(panel["cluster"].nunique()),
        "n_complete_cases": int(len(panel)),
        "stability_proxy": stability_proxy,
        "cluster_sizes": {f"cluster_{int(row.cluster)}": int(row.n) for row in cluster_outcomes.itertuples()},
        "cluster_mortality": {f"cluster_{int(row.cluster)}": float(row.mortality_rate) for row in cluster_outcomes.itertuples()},
        "outputs": {
            "cluster_assignments": "cluster_assignments.csv",
            "cluster_outcomes": "cluster_outcomes.csv",
            "cluster_trajectory_means": "cluster_trajectory_means.csv",
            "trajectory_clusters_figure": "trajectory_clusters.png",
        },
    }
    with open(out_dir / "step_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=to_jsonable)
    print(json.dumps(summary, indent=2, ensure_ascii=False, default=to_jsonable))
    """
    return (
        textwrap.dedent(template)
        .replace("__STEP_ID__", step_id)
        .replace("__QUESTION__", json.dumps(ctx.research_question))
        .replace("__OUTCOME__", json.dumps(outcome))
    )


def _mock_code_publication_figure(
    *, ctx: ResearchContext, step_id: str, outcome: str
) -> str:
    analysis_type = infer_analysis_type(
        ctx,
        primary_predictor=_pick_primary_predictor(ctx, outcome=outcome),
        target_outcome=outcome,
    ).key
    template = r"""
    # AUTO-GENERATED by easyicu.research_agent.MockLLMClient
    # step_id: __STEP_ID__
    # research_question: __QUESTION__
    from __future__ import annotations
    import json
    import math
    import os
    import shutil
    from pathlib import Path

    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from easyicu.research_agent.figures.publication import (
        apply_publication_style,
        add_panel_label,
        audit_publication_exports,
        make_figure_contract,
        save_publication_figure,
    )

    out_dir = Path(os.environ["STEP_OUT_DIR"])
    out_dir.mkdir(parents=True, exist_ok=True)
    run_dir = Path(os.environ["EASYICU_RUN_DIR"])
    source_dir = out_dir / "publication_figure_source_tables"
    source_dir.mkdir(parents=True, exist_ok=True)

    def to_jsonable(x):
        if isinstance(x, (np.integer, )):
            return int(x)
        if isinstance(x, (np.floating, )):
            v = float(x)
            return v if math.isfinite(v) else None
        if isinstance(x, (np.bool_, )):
            return bool(x)
        if isinstance(x, np.ndarray):
            return x.tolist()
        try:
            if pd.isna(x):
                return None
        except Exception:
            pass
        return x

    def finding_to_dict(f):
        if hasattr(f, "model_dump"):
            return f.model_dump(mode="json")
        return {"message": str(f)}

    family = "__ANALYSIS_TYPE__"
    apply_publication_style()
    summary = {"step": "__STEP_ID__", "status": "completed", "analysis_type": family}

    if family == "prediction_model":
        analysis_dir = run_dir / "steps" / "04_prediction_model_analysis" / "outputs"
        perf = pd.read_csv(analysis_dir / "model_performance_train_test.csv")
        coef = pd.read_csv(analysis_dir / "model_coefficients.csv")
        risk = pd.read_csv(analysis_dir / "risk_predictions_test.csv")
        roc = pd.read_csv(analysis_dir / "roc_curve.csv")
        cal = pd.read_csv(analysis_dir / "calibration_curve.csv")

        for name in ["model_performance_train_test.csv", "model_coefficients.csv", "risk_predictions_test.csv", "roc_curve.csv", "calibration_curve.csv"]:
            shutil.copy2(analysis_dir / name, source_dir / name)

        fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.8), constrained_layout=True)
        ax = axes[0, 0]
        ax.plot(roc["fpr"], roc["tpr"], color="#1f77b4", linewidth=1.5)
        ax.plot([0, 1], [0, 1], linestyle="--", color="grey", linewidth=0.8)
        ax.set_xlabel("False-positive rate")
        ax.set_ylabel("True-positive rate")
        add_panel_label(ax, "A")
        ax.text(0.04, 0.08, f"AUC={float(perf['auc'].iloc[0]):.3f}", transform=ax.transAxes, fontsize=7)

        ax = axes[0, 1]
        ax.plot([0, 1], [0, 1], linestyle="--", color="grey", linewidth=0.8)
        ax.plot(cal["predicted_mean"], cal["observed_rate"], marker="o", color="#d1495b", linewidth=1.3)
        ax.set_xlabel("Predicted risk")
        ax.set_ylabel("Observed risk")
        add_panel_label(ax, "B")
        slope = perf["calibration_slope"].iloc[0]
        if pd.notna(slope):
            ax.text(0.04, 0.08, f"Slope={float(slope):.2f}", transform=ax.transAxes, fontsize=7)

        ax = axes[1, 0]
        bins = np.linspace(0, 1, 16)
        ax.hist(risk.loc[risk["death"] == 0, "predicted_risk"], bins=bins, alpha=0.7, color="#8fb9e0", label="Survived")
        ax.hist(risk.loc[risk["death"] == 1, "predicted_risk"], bins=bins, alpha=0.7, color="#d96c6c", label="Died")
        ax.set_xlabel("Predicted risk")
        ax.set_ylabel("Count")
        ax.legend(frameon=False, fontsize=7)
        add_panel_label(ax, "C")

        ax = axes[1, 1]
        plot_coef = coef[coef["variable"] != "intercept"].copy().sort_values("odds_ratio")
        ys = np.arange(len(plot_coef))
        ax.errorbar(
            plot_coef["odds_ratio"], ys,
            xerr=[
                np.maximum(0, plot_coef["odds_ratio"] - plot_coef["or_lower"]),
                np.maximum(0, plot_coef["or_upper"] - plot_coef["odds_ratio"]),
            ],
            fmt="o",
            color="#2a6f97",
        )
        ax.axvline(1.0, linestyle="--", color="grey", linewidth=0.8)
        ax.set_yticks(ys, plot_coef["variable"])
        ax.set_xlabel("Odds ratio")
        add_panel_label(ax, "D")

        contract = make_figure_contract(
            figure_id="prediction_publication_figure",
            core_claim="The latest EasyICU prediction-model workflow reports held-out discrimination, calibration, risk separation, and model coefficients in a claim-first publication figure.",
            panels=[
                {"panel_id": "A", "title": "", "role": "overview", "claim": "Held-out ROC discrimination is reported explicitly.", "evidence_ids": ["roc_curve.csv", "model_performance_train_test.csv"]},
                {"panel_id": "B", "title": "", "role": "robustness", "claim": "Calibration is shown against the identity line with the reported slope.", "evidence_ids": ["calibration_curve.csv", "model_performance_train_test.csv"]},
                {"panel_id": "C", "title": "", "role": "distribution", "claim": "Predicted risk separates deaths from survivors on the held-out test set.", "evidence_ids": ["risk_predictions_test.csv"]},
                {"panel_id": "D", "title": "", "role": "association", "claim": "Coefficient directions and uncertainty are preserved as model outputs, not prose-only claims.", "evidence_ids": ["model_coefficients.csv"]},
            ],
            source_data=[
                "roc_curve.csv",
                "calibration_curve.csv",
                "risk_predictions_test.csv",
                "model_coefficients.csv",
                "model_performance_train_test.csv",
            ],
        )
        stem = out_dir / "prediction_publication_figure"
        paths = save_publication_figure(fig, stem, contract=contract, dpi=300)
        plt.close(fig)
        audit = [finding_to_dict(f) for f in audit_publication_exports(paths)]
        summary["figure_id"] = "prediction_publication_figure"
        summary["core_claim"] = contract.core_claim
        summary["outputs"] = {k: str(v.name) for k, v in paths.items()}
        summary["source_tables"] = {p.name: f"publication_figure_source_tables/{p.name}" for p in sorted(source_dir.glob("*.csv"))}
        summary["numeric_statistics"] = {
            "n_train": int(perf["n_train"].iloc[0]),
            "n_test": int(perf["n_test"].iloc[0]),
            "auc": float(perf["auc"].iloc[0]) if pd.notna(perf["auc"].iloc[0]) else None,
            "brier": float(perf["brier"].iloc[0]) if pd.notna(perf["brier"].iloc[0]) else None,
            "calibration_slope": float(perf["calibration_slope"].iloc[0]) if pd.notna(perf["calibration_slope"].iloc[0]) else None,
        }
        summary["publication_export_qa"] = {"audit_result": audit}
    else:
        analysis_dir = run_dir / "steps" / "04_trajectory_clustering_analysis" / "outputs"
        outcomes = pd.read_csv(analysis_dir / "cluster_outcomes.csv")
        traj = pd.read_csv(analysis_dir / "cluster_trajectory_means.csv")
        assign = pd.read_csv(analysis_dir / "cluster_assignments.csv")

        for name in ["cluster_outcomes.csv", "cluster_trajectory_means.csv", "cluster_assignments.csv"]:
            shutil.copy2(analysis_dir / name, source_dir / name)

        fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.8), constrained_layout=True)
        colors = ["#2a6f97", "#c97c5d", "#6a994e", "#7b5ea7"]
        ax = axes[0, 0]
        lact = traj[traj["domain"] == "lactate"].copy()
        for idx, cluster_id in enumerate(sorted(lact["cluster"].unique())):
            sub = lact[lact["cluster"] == cluster_id]
            ax.plot(range(len(sub)), sub["value_mean"], marker="o", color=colors[idx % len(colors)])
        ax.set_xticks(range(len(sub)), sub["timepoint"].tolist())
        ax.set_ylabel("Mean lactate")
        add_panel_label(ax, "A")

        ax = axes[0, 1]
        map_df = traj[traj["domain"] == "map"].copy()
        for idx, cluster_id in enumerate(sorted(map_df["cluster"].unique())):
            sub = map_df[map_df["cluster"] == cluster_id]
            ax.plot(range(len(sub)), sub["value_mean"], marker="o", color=colors[idx % len(colors)])
        ax.set_xticks(range(len(sub)), sub["timepoint"].tolist())
        ax.set_ylabel("Mean MAP")
        add_panel_label(ax, "B")

        ax = axes[1, 0]
        ax.bar(outcomes["cluster"].astype(str), outcomes["n"], color="#8fb9e0")
        ax.set_xlabel("Cluster")
        ax.set_ylabel("Patients")
        add_panel_label(ax, "C")

        ax = axes[1, 1]
        ax.bar(outcomes["cluster"].astype(str), outcomes["mortality_rate"], color="#d96c6c")
        ax.set_xlabel("Cluster")
        ax.set_ylabel("Mortality rate")
        add_panel_label(ax, "D")

        contract = make_figure_contract(
            figure_id="trajectory_clustering_publication_figure",
            core_claim="The latest EasyICU trajectory-clustering workflow yields interpretable hemodynamic subphenotypes with distinct lactate/MAP trajectories and outcome rates.",
            panels=[
                {"panel_id": "A", "title": "", "role": "overview", "claim": "Clusters separate by mean lactate trajectories.", "evidence_ids": ["cluster_trajectory_means.csv"]},
                {"panel_id": "B", "title": "", "role": "relationship", "claim": "Clusters also separate by mean MAP trajectories.", "evidence_ids": ["cluster_trajectory_means.csv"]},
                {"panel_id": "C", "title": "", "role": "distribution", "claim": "Cluster sizes are explicit rather than implied.", "evidence_ids": ["cluster_outcomes.csv"]},
                {"panel_id": "D", "title": "", "role": "audit", "claim": "Mortality differences across clusters are reported directly.", "evidence_ids": ["cluster_outcomes.csv"]},
            ],
            source_data=[
                "cluster_outcomes.csv",
                "cluster_trajectory_means.csv",
                "cluster_assignments.csv",
            ],
        )
        stem = out_dir / "trajectory_clustering_publication_figure"
        paths = save_publication_figure(fig, stem, contract=contract, dpi=300)
        plt.close(fig)
        audit = [finding_to_dict(f) for f in audit_publication_exports(paths)]
        summary["figure_id"] = "trajectory_clustering_publication_figure"
        summary["core_claim"] = contract.core_claim
        summary["outputs"] = {k: str(v.name) for k, v in paths.items()}
        summary["source_tables"] = {p.name: f"publication_figure_source_tables/{p.name}" for p in sorted(source_dir.glob("*.csv"))}
        summary["numeric_statistics"] = {
            "n_clusters": int(outcomes["cluster"].nunique()),
            "largest_cluster_n": int(outcomes["n"].max()),
            "highest_cluster_mortality_rate": float(outcomes["mortality_rate"].max()),
        }
        summary["publication_export_qa"] = {"audit_result": audit}

    with open(out_dir / "step_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=to_jsonable)
    print(json.dumps(summary, indent=2, ensure_ascii=False, default=to_jsonable))
    """
    return (
        textwrap.dedent(template)
        .replace("__STEP_ID__", step_id)
        .replace("__QUESTION__", json.dumps(ctx.research_question))
        .replace("__ANALYSIS_TYPE__", analysis_type)
        .replace("__OUTCOME__", json.dumps(outcome))
    )


def _extract_step_id(prompt: str) -> Optional[str]:
    m = re.search(r"step_id\s*[:=]\s*([\w-]+)", prompt)
    if m:
        return m.group(1)
    m = re.search(r"\b(step\s+)?(\d{2,3}_[\w-]+)\b", prompt)
    return m.group(2) if m else None


def _extract_expected_outputs(prompt: str) -> List[str]:
    """Read the coder prompt's exact typed output declarations."""

    match = re.search(r"Expected outputs:\s*\[([^\n]*)\]", prompt)
    if match is None:
        return []
    return re.findall(r"['\"]([^'\"]+:[^'\"]+)['\"]", match.group(1))


def _mock_interpretation(ctx: ResearchContext, prompt: str) -> str:
    """Brief, evidence-grounded interpretation paragraph."""
    outcome = ctx.target_outcome or _pick_outcome(ctx) or "the primary outcome"
    parts: List[str] = []
    parts.append(
        f"The cohort of {ctx.cohort.n_stays:,} ICU stays from {ctx.cohort.database} "
        f"was analysed against the question: '{ctx.research_question}'."
    )
    parts.append(
        "All numerical claims in this paragraph are bound to entries in the evidence "
        "store; reviewers can verify each value against its generating script and run log."
    )
    return " ".join(parts)


def _mock_writer_section(
    ctx: ResearchContext,
    prompt: str,
    *,
    language: str = "en",
) -> str:
    """Evidence-bound section stub for WriterAgent unit/smoke tests.

    The production WriterAgent writes each manuscript section in a
    separate prompt. The older mock only knew the legacy one-shot
    scaffold prompt, which meant strict evidence smoke tests received
    a generic "no live LLM configured" paragraph and correctly failed.
    This helper keeps the mock narrow: it emits short, citation-shaped
    prose that exercises the binder without pretending to write a
    submission-ready manuscript.
    """
    m = re.search(r"Write ONLY the \*\*(.*?)\*\* section", prompt, flags=re.I | re.S)
    section = (m.group(1) if m else "Section").strip()
    ids_match = re.search(
        r"Only use ids from this list:\s*(.*?)\n\nLANGUAGE POLICY:",
        prompt,
        flags=re.I | re.S,
    )
    ids = [
        token.strip()
        for token in (ids_match.group(1) if ids_match else "").split(",")
        if token.strip() and token.strip() != "(none)"
    ]

    def cite(*preferred: str) -> str:
        for candidate in preferred:
            if candidate in ids:
                return f"{{evidence:{candidate}}}"
        return f"{{evidence:{ids[0]}}}" if ids else "{evidence:research_context}"

    table = cite("table_one", "cohort_summary", "research_context")
    outcome = cite("outcome_rate", "outcome_incidence", "mortality_rate", "table_one")
    association = cite("primary_association", "model_performance", "table_one")
    missingness = cite("missingness", "research_context", "table_one")
    context = cite("research_context", "architecture_profile", "table_one")

    if language == "zh":
        if section.startswith("Title"):
            return "# EasyICU ICU 关联分析\n\n**关键词：** ICU，队列，关联，证据追踪，EasyICU"
        heading = "## " + section.replace(
            "Conclusion, Data availability, Funding, COI", "结论"
        )
        return textwrap.dedent(
            f"""
        {heading}

        本节为确定性 mock writer 生成的证据绑定占位文本。队列和变量定义见 {context}。
        结局、缺失和主要关联分别记录于 {outcome}、{missingness} 和 {association}。
        这些表述仅用于测试证据绑定流程，不构成新的临床结论。
        """
        ).strip()

    if section.startswith("Title"):
        return (
            "# EasyICU evidence-bound ICU association analysis\n"
            "**Keywords:** ICU, cohort, association, provenance, EasyICU"
        )
    if section.startswith("Abstract"):
        return textwrap.dedent(
            f"""
        ## Abstract

        **Background:** This evidence-bound EasyICU analysis evaluates the requested ICU association.
        **Methods:** Cohort construction and variable semantics are documented in {context}.
        **Results:** Outcome incidence, missingness, and the primary association are recorded in {outcome}, {missingness}, and {association}.
        **Conclusions:** Findings are associational and require external validation.
        """
        ).strip()
    if section.startswith("Methods"):
        return textwrap.dedent(
            f"""
        ## Methods

        The cohort, variable definitions, and analysis context were taken from the EasyICU research context {context}.
        Descriptive cohort evidence is available in {table}, and missingness handling is documented in {missingness}.
        """
        ).strip()
    if section.startswith("Results"):
        return textwrap.dedent(
            f"""
        ## Results

        Cohort characteristics are summarized in {table}.
        Outcome incidence is reported in {outcome}.
        Missingness diagnostics are reported in {missingness}.
        The primary association is reported in {association}.
        """
        ).strip()
    if section.startswith("Discussion"):
        return textwrap.dedent(
            f"""
        ## Discussion

        The analysis should be interpreted as an observed association grounded in the registered evidence {association}.
        EasyICU's concept layer and evidence registry document how the cohort and variables were constructed {context}.
        """
        ).strip()
    if section.startswith("Limitations"):
        return textwrap.dedent(
            f"""
        ## Limitations

        This is an observational, agent-assisted analysis, so residual confounding and limited external generalisability remain possible {context}.
        The mock writer records these limitations while leaving clinical interpretation to the final human-authored manuscript {context}.
        """
        ).strip()
    return textwrap.dedent(
        f"""
    ## Conclusion

    The registered evidence supports an associational summary only {association}.

    ## Data and code availability
    Generated scripts and evidence artefacts are recorded in the EasyICU run manifest {context}.

    ## Funding
    Funding information was not assessed in this mock smoke run.

    ## Conflicts of interest
    The authors declare no conflicts of interest.
    """
    ).strip()


def _mock_manuscript_scaffold(ctx: ResearchContext, *, language: str = "en") -> str:
    """Return a minimal manuscript scaffold in markdown.

    The scaffold is deliberately *thin*: title, methods, results
    skeleton, all referencing evidence ids that the writer will inject
    from the evidence store. Discussion and clinical claims are left
    blank — that is policy, not laziness.
    """
    outcome = ctx.target_outcome or _pick_outcome(ctx) or "the primary outcome"
    predictor = _pick_primary_predictor(ctx, outcome=outcome) or "the primary predictor"
    cross_db = (
        ", ".join(ctx.cross_database_validation)
        if ctx.cross_database_validation
        else "(none planned)"
    )
    if language == "zh":
        return (
            textwrap.dedent(
                f"""
        # 手稿脚手架

        > 由 easyicu.research_agent 生成。以下每个数值性主张都必须带有
        > `{{evidence:<id>}}` 证据占位符；未绑定证据的句子会被后处理拦截。

        ## 标题
        {ctx.cohort.database} ICU 患者中 {predictor} 与 {outcome} 的关系：
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

        ## 讨论
        *(留给人类作者；writer agent 不在没有人工确认的情况下生成临床主张或建议。)*
        """
            ).strip()
            + "\n"
        )
    return (
        textwrap.dedent(
            f"""
    # Manuscript scaffold

    > Generated by easyicu.research_agent. Every numeric claim below is an
    > `{{evidence_id}}` placeholder filled in from the evidence store.
    > Sentences without an evidence id are blocked by the writer.

    ## Title
    Association between {predictor} and {outcome}
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

    ## Discussion
    *(left to the human author; the writer agent declines to generate clinical
    claims and recommendations without explicit human sign-off.)*
    """
        ).strip()
        + "\n"
    )
