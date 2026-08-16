"""Progressive Planner v2: retrieve, skeletonize, compile, revise a suffix."""

from __future__ import annotations

import json
import re
from typing import Any, Callable, Mapping, Optional, Sequence

from pydantic import ValidationError

from ..canonical_json import canonical_sha256
from ..cohort.schema import validate_plan_typed_bindings_against_context
from ..contracts.primary_cohort import primary_analysis_cohort_plan_findings
from ..planning.adjustment_authority import validate_plan_against_adjustment_authority
from ..planning.analysis_types import (
    infer_analysis_type,
    list_analysis_types,
    validate_host_authorized_analysis_family,
)
from ..planning.literature_bindings import validate_literature_citation_bindings
from ..planning.planner_output_contract import (
    validate_fresh_planner_typed_product_specs,
)
from ..planning.preplan_know_how import verify_know_how_decisions
from ..planning.primary_result_contract import validate_required_primary_result
from ..planning.progressive_compiler import (
    assert_immutable_prefix,
    compile_progressive_plan,
)
from ..planning.progressive_contract import (
    ProgressivePlanCompileError,
    ProgressivePlanCompileReceipt,
    ProgressivePlanSkeleton,
    ProgressiveSuffixRevision,
)
from ..planning.robustness_contract import validate_planner_robustness_specs
from ..planning.scientific_action_catalog import scientific_actions_for_analysis_type
from ..planning.scientific_review import required_method_layers_for_context
from ..providers.capabilities import llm_supports_strict_json_schema
from ..providers.llm import llm_is_mockish
from ..providers.prompt_budget import DEFAULT_MAX_PROMPT_TOKENS
from ..providers.prompts import load_prompt_pack
from ..providers.protocol import LLMClient, LLMMessage, StructuredOutputRequest
from ..providers.structured_retry import call_llm_with_structured_retry
from ..reporting.article_contract import (
    build_article_analysis_contract,
    validate_plan_against_article_contract,
)
from ..research_context.outbound import format_outbound_safe_context
from ..schema import AnalysisPlan, ResearchContext
from .progressive_payload import progressive_structured_output_request
from .plan_payload import bind_literature_citation_authority


_GUIDE = load_prompt_pack()["progressive_planner"]
_MAX_INITIAL_PARSE_RETRIES = 2
_MAX_COMPILE_REVISIONS = 4
_MAX_OUTPUT_TOKENS = 20_000
_MAX_REQUEST_BYTES = DEFAULT_MAX_PROMPT_TOKENS * 4


def _tokens(value: object) -> set[str]:
    folded = str(value or "").casefold()
    compounds = re.findall(r"[a-z0-9_]+", folded)
    pieces = re.findall(r"[a-z0-9]+", folded.replace("_", " "))
    return {token for token in (*compounds, *pieces) if len(token) >= 2}


def candidate_analysis_types(
    context: ResearchContext,
    *,
    max_candidates: int = 4,
) -> tuple[str, ...]:
    """Retrieve a generous, question-relevant family subset.

    Scoring uses the research question only.  Case notes can contain required
    audits and sensitivities without changing the headline study family, which
    is exactly how a measurement-audit keyword displaced E1's association.
    """

    question = str(context.research_question or "").casefold()
    question_tokens = _tokens(question)
    scored: list[tuple[int, int, str]] = []
    for position, spec in enumerate(list_analysis_types()):
        score = 0
        for trigger in spec.trigger_terms:
            phrase = str(trigger).casefold().strip()
            if not phrase:
                continue
            if phrase in question:
                score += 5 + len(phrase.split())
            else:
                score += len(question_tokens & _tokens(phrase))
        if score:
            scored.append((score, -position, spec.key))
    scored.sort(reverse=True)
    candidates = [key for _score, _position, key in scored]
    inferred = infer_analysis_type(context).key
    if context.primary_exposure and context.target_outcome:
        candidates.insert(0, "association_study")
    candidates.extend([inferred, "descriptive_epidemiology"])
    authorized: list[str] = []
    for key in candidates:
        if key in authorized:
            continue
        try:
            validate_host_authorized_analysis_family(context, key)
        except ValueError:
            continue
        authorized.append(key)
        if len(authorized) >= max(1, int(max_candidates)):
            break
    if not authorized:
        raise ProgressivePlanCompileError(
            "progressive_no_authorized_analysis_type",
            "retrieval found no analysis family authorized by ResearchContext",
            path="analysis_type",
        )
    return tuple(authorized)


def select_progressive_variables(
    context: ResearchContext,
    *,
    max_variables: int = 48,
) -> tuple[str, ...]:
    """Retrieve a bounded, run-specific variable set without dropping anchors."""

    preferences = context.user_preferences
    search_text = "\n".join(
        value
        for value in (
            context.research_question,
            context.notes,
            getattr(preferences, "evaluation_focus", None),
            getattr(preferences, "data_constraints", None),
            getattr(preferences, "must_have_outputs", None),
            getattr(preferences, "timing_and_design", None),
            getattr(preferences, "subgroup_sensitivity", None),
        )
        if value
    ).casefold()
    text_tokens = _tokens(search_text)
    exact = {
        str(value).strip()
        for value in (
            context.primary_exposure,
            context.target_outcome,
            *context.cohort.outcome_columns,
            *context.cohort.time_columns,
            *(getattr(preferences, "covariates", ()) or ()),
        )
        if str(value or "").strip()
    }
    primary = context.variable(str(context.primary_exposure or ""))
    primary_concepts = {
        str(value).casefold()
        for value in (
            getattr(primary, "source_concept", None),
            *(getattr(primary, "derived_from_concepts", ()) or ()),
        )
        if str(value or "").strip()
    }
    scored: list[tuple[int, int, str]] = []
    for position, variable in enumerate(context.variables):
        name = variable.name
        folded_name = name.casefold()
        source = str(variable.source_concept or "").casefold()
        description_tokens = _tokens(variable.description)
        score = 0
        if name in exact:
            score += 10_000
        if folded_name and folded_name in search_text:
            score += 2_000
        if source and source in search_text:
            score += 1_000
        name_tokens = _tokens(name) - {
            "icu",
            "max",
            "min",
            "mean",
            "first",
            "last",
            "time",
            "measured",
            "value",
            "flag",
            "code",
        }
        score += 3_000 * len(name_tokens & text_tokens)
        score += 20 * len(description_tokens & text_tokens)
        if variable.role.value in {"outcome", "intervention", "treatment"}:
            score += 300
        if variable.role.value == "demographic":
            score += 2_500
        if (
            source in primary_concepts
            or {str(item).casefold() for item in variable.derived_from_concepts}
            & primary_concepts
        ):
            score += 500
        if any(
            folded_name.startswith(f"{concept}_")
            for concept in primary_concepts
            if concept
        ):
            score += 400
        # Prefer the compact audit-bearing representatives of a repeated
        # concept family before its alternate summaries. This keeps breadth
        # across components without encoding any ICU variable name.
        for suffix, bonus in (
            ("_max", 120),
            ("_measured", 110),
            ("_n", 100),
            ("_first_time", 90),
            ("_last_time", 80),
        ):
            if folded_name.endswith(suffix):
                score += bonus
                break
        scored.append((score, -position, name))
    scored.sort(reverse=True)
    limit = max(1, int(max_variables))
    variable_by_name = {variable.name: variable for variable in context.variables}
    source_counts: dict[str, int] = {}
    selected: list[str] = []
    for _score, _position, name in scored:
        variable = variable_by_name[name]
        source = str(variable.source_concept or name).casefold()
        if source in primary_concepts:
            source_limit = 5
        elif any(
            source.startswith(f"{concept}_") for concept in primary_concepts if concept
        ):
            source_limit = 3
        else:
            source_limit = 4
        if name not in exact and source_counts.get(source, 0) >= source_limit:
            continue
        selected.append(name)
        source_counts[source] = source_counts.get(source, 0) + 1
        if len(selected) >= limit:
            break
    # Restore ResearchContext order so opaque level indices and prompt reviews
    # are deterministic and easy to compare across runs.
    selected_set = set(selected)
    ordered = tuple(
        variable.name for variable in context.variables if variable.name in selected_set
    )
    if not ordered:
        raise ProgressivePlanCompileError(
            "progressive_variable_retrieval_empty",
            "ResearchContext exposes no variables to the progressive planner",
            path="variables",
        )
    return ordered


def progressive_cohort_concept_ids(
    context: ResearchContext,
    variable_names: Sequence[str],
) -> tuple[str, ...]:
    """Expose sealed source concepts without conflating them with columns."""

    selected = set(variable_names)
    values: list[str] = list(variable_names)
    for variable in context.variables:
        if variable.name not in selected:
            continue
        values.extend(
            str(value).strip()
            for value in (
                variable.source_concept,
                *variable.derived_from_concepts,
            )
            if str(value or "").strip()
        )
    return tuple(dict.fromkeys(values))


def _action_catalog(
    analysis_types: Sequence[str],
) -> tuple[tuple[str, ...], list[dict[str, Any]]]:
    action_ids: list[str] = []
    rows: list[dict[str, Any]] = []
    for analysis_type in analysis_types:
        catalog = scientific_actions_for_analysis_type(analysis_type)
        for action in catalog.actions:
            if action.execution_mode == "not_available":
                continue
            if action.action_id not in action_ids:
                action_ids.append(action.action_id)
            rows.append(
                {
                    "analysis_type": analysis_type,
                    "action_id": action.action_id,
                    "execution_mode": action.execution_mode,
                    "produces": action.produces,
                    "required_inputs": list(action.required_inputs),
                }
            )
    return tuple(action_ids), rows


def _parse_model(raw: str, model: type[Any]) -> Any:
    payload = json.loads(str(raw or "").strip())
    if not isinstance(payload, dict):
        raise ValueError("progressive Planner response root must be an object")
    return model.model_validate(payload)


def _parse_suffix_revision(
    raw: str,
    *,
    expected_step_id: str,
) -> ProgressiveSuffixRevision:
    revision = _parse_model(raw, ProgressiveSuffixRevision)
    if revision.replace_from_step_id != expected_step_id:
        raise ValueError(
            "suffix revision must begin at the host-rejected coordinate "
            f"{expected_step_id!r}; received {revision.replace_from_step_id!r}"
        )
    return revision


def _primary_or_first_step_index(plan: AnalysisPlan) -> int:
    for index, step in enumerate(plan.steps):
        if step.planned_analysis_role == "primary":
            return index
    return 0


def _step_index_from_error(plan: AnalysisPlan, message: str) -> int:
    for index, step in enumerate(plan.steps):
        if step.step_id and step.step_id in message:
            return index
    return _primary_or_first_step_index(plan)


def _accept_compiled_plan(
    *,
    plan: AnalysisPlan,
    agent_context: ResearchContext,
    article_context: ResearchContext,
    allowed_literature_citation_keys: Sequence[str],
    direct_comparator_literature_keys: Sequence[str],
    allowed_know_how_decisions: Mapping[str, Mapping[str, Any]] | None,
    enforce_article_contract: bool,
    llm: LLMClient,
) -> None:
    """Apply the same fresh-plan authorities after host compilation."""

    try:
        if plan.robustness_specs:
            validate_planner_robustness_specs(plan.robustness_specs)
        validate_literature_citation_bindings(
            plan,
            allowed_literature_citation_keys,
            context=agent_context,
            direct_comparator_keys=direct_comparator_literature_keys,
        )
        if allowed_know_how_decisions is not None:
            verify_know_how_decisions(
                plan.know_how_decisions,
                allowed_know_how_decisions,
            )
        if enforce_article_contract:
            contract = build_article_analysis_contract(
                article_context,
                analysis_type=plan.analysis_type,
            )
            findings = validate_plan_against_article_contract(
                plan=plan,
                contract=contract,
            )
            missing_roles = sorted(
                {
                    str(role)
                    for finding in findings
                    for role in (finding.detail or {}).get("missing_roles", [])
                    if str(role).strip()
                }
            )
            if "robustness" in contract.required_roles and not plan.robustness_specs:
                missing_roles = sorted({*missing_roles, "robustness_specs"})
            if missing_roles:
                raise ValueError(
                    "progressive article contract is missing required role(s): "
                    + ", ".join(missing_roles)
                )
        if not llm_is_mockish(llm):
            validate_fresh_planner_typed_product_specs(
                plan,
                context=agent_context,
            )
        validate_plan_typed_bindings_against_context(
            plan=plan,
            context=agent_context,
        )
        validate_plan_against_adjustment_authority(
            plan=plan,
            context=agent_context,
        )
        cohort_findings = primary_analysis_cohort_plan_findings(plan=plan)
        if cohort_findings:
            raise ValueError(
                "primary cohort contract findings: "
                + json.dumps(
                    [item.detail for item in cohort_findings],
                    ensure_ascii=False,
                    default=str,
                )
            )
        validate_required_primary_result(plan=plan, context=agent_context)
    except ProgressivePlanCompileError:
        raise
    except ValueError as exc:
        message = str(exc)
        index = _step_index_from_error(plan, message)
        step = plan.steps[index] if plan.steps else None
        raise ProgressivePlanCompileError(
            "progressive_fresh_plan_gate_failed",
            message,
            step_id=step.step_id if step is not None else None,
            step_index=index if step is not None else None,
            path="fresh_plan_acceptance",
        ) from exc


class ProgressivePlannerAgent:
    """Emit a small plan skeleton, compile it, and revise only failed suffixes."""

    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm
        self.last_prompt_metrics: dict[str, Any] = {}
        self.last_compile_receipt: Optional[ProgressivePlanCompileReceipt] = None
        self.last_skeleton: Optional[ProgressivePlanSkeleton] = None
        self.last_dropped_plan_keys: dict[str, list[str]] = {
            "top_level": [],
            "steps": [],
        }

    @staticmethod
    def _request_authorities(
        context: ResearchContext,
    ) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...], list[dict[str, Any]]]:
        analysis_types = candidate_analysis_types(context)
        variables = select_progressive_variables(context)
        action_ids, action_rows = _action_catalog(analysis_types)
        return analysis_types, variables, action_ids, action_rows

    @staticmethod
    def _user_prompt(
        context: ResearchContext,
        *,
        article_context: ResearchContext | None = None,
        analysis_types: Sequence[str],
        variables: Sequence[str],
        action_rows: Sequence[Mapping[str, Any]],
        allowed_literature_citation_keys: Sequence[str] = (),
        know_how_context: str = "",
        planning_contract_context: str = "",
    ) -> str:
        contract_context = article_context or context
        article_contracts = []
        for analysis_type in analysis_types:
            contract = build_article_analysis_contract(
                contract_context,
                analysis_type=analysis_type,
            )
            article_contracts.append(
                {
                    "analysis_type": analysis_type,
                    "analysis_family": contract.analysis_family,
                    "required_roles": list(contract.required_roles),
                    "planner_owned_result_roles": list(
                        contract.planner_owned_result_roles
                    ),
                    "requirements": [
                        {
                            "module_id": item.module_id,
                            "role": item.role,
                            "required": item.required,
                        }
                        for item in contract.requirements
                    ],
                }
            )
        blocks = [
            "PROGRESSIVE PLANNER RUN AUTHORITY",
            "Candidate analysis families (choose exactly one):\n"
            + json.dumps(list(analysis_types), ensure_ascii=False),
            "Retrieved scientific actions (only these may be selected):\n"
            + json.dumps(list(action_rows), ensure_ascii=False, separators=(",", ":")),
            "Sealed literature citation keys:\n"
            + json.dumps(list(allowed_literature_citation_keys), ensure_ascii=False),
            "Candidate-specific host article role contracts:\n"
            + json.dumps(article_contracts, ensure_ascii=False, separators=(",", ":")),
            "Outbound-safe ResearchContext for the retrieved variable subset:\n"
            + format_outbound_safe_context(context, variable_names=variables),
        ]
        if planning_contract_context:
            blocks.append(
                "Additional run-specific article/task contract (binding; never "
                "global). Its pre-plan family classification is provisional; "
                "the candidate-specific contracts above govern the family you "
                "select, while every explicit task requirement remains binding:\n"
                + planning_contract_context
            )
        if know_how_context:
            blocks.append(
                "Retrieved protocol know-how (record every required claim disposition):\n"
                + know_how_context
            )
        blocks.append(
            "Return the compact skeleton only. Standard modules have canonical "
            "host-derived outputs; use outputs=[] for cohort_definition, "
            "table_one, exposure_outcome_distribution, and adjusted_association."
        )
        return "\n\n".join(blocks)

    @classmethod
    def request_messages(
        cls,
        context: ResearchContext,
        *,
        know_how_context: str = "",
        planning_contract_context: str = "",
        strict_transport_schema: bool = False,
        structured_output: StructuredOutputRequest | None = None,
        allowed_literature_citation_keys: Sequence[str] = (),
    ) -> list[LLMMessage]:
        del strict_transport_schema, structured_output
        analysis_types, variables, _action_ids, action_rows = cls._request_authorities(
            context
        )
        return [
            LLMMessage(role="system", content=_GUIDE),
            LLMMessage(
                role="user",
                content=cls._user_prompt(
                    context,
                    article_context=context,
                    analysis_types=analysis_types,
                    variables=variables,
                    action_rows=action_rows,
                    allowed_literature_citation_keys=allowed_literature_citation_keys,
                    know_how_context=know_how_context,
                    planning_contract_context=planning_contract_context,
                ),
            ),
        ]

    @classmethod
    def request_metrics(
        cls,
        context: ResearchContext,
        *,
        know_how_context: str = "",
        planning_contract_context: str = "",
        strict_transport_schema: bool = False,
        structured_output: StructuredOutputRequest | None = None,
    ) -> dict[str, Any]:
        analysis_types, variables, action_ids, _rows = cls._request_authorities(context)
        request = structured_output
        if request is None and strict_transport_schema:
            request = progressive_structured_output_request(
                analysis_types=analysis_types,
                variable_names=variables,
                cohort_concept_ids=progressive_cohort_concept_ids(
                    context,
                    variables,
                ),
                scientific_action_ids=action_ids,
            )
        messages = cls.request_messages(
            context,
            know_how_context=know_how_context,
            planning_contract_context=planning_contract_context,
            strict_transport_schema=strict_transport_schema,
            structured_output=request,
        )
        message_bytes = sum(len(item.content.encode("utf-8")) for item in messages)
        schema_bytes = request.payload_bytes if request is not None else 0
        return {
            "message_payload_bytes": message_bytes,
            "structured_output_payload_bytes": schema_bytes,
            "structured_output_authority_sha256": (
                request.authority_sha256 if request is not None else None
            ),
            "total_bytes": message_bytes + schema_bytes,
            "selected_variable_count": len(variables),
            "selected_variable_roster_sha256": canonical_sha256(list(variables)),
            "candidate_analysis_types": list(analysis_types),
            "selected_scientific_action_ids": list(action_ids),
            "planner_strategy": "progressive_v2",
        }

    def _compile_and_accept(
        self,
        skeleton: ProgressivePlanSkeleton,
        *,
        agent_context: ResearchContext,
        article_context: ResearchContext,
        allowed_literature_citation_keys: Sequence[str],
        direct_comparator_literature_keys: Sequence[str],
        allowed_know_how_decisions: Mapping[str, Mapping[str, Any]] | None,
        enforce_article_contract: bool,
    ) -> tuple[AnalysisPlan, ProgressivePlanCompileReceipt]:
        plan, receipt = compile_progressive_plan(
            skeleton=skeleton,
            context=agent_context,
            allowed_literature_citation_keys=allowed_literature_citation_keys,
            allowed_know_how_decisions=allowed_know_how_decisions,
        )
        _accept_compiled_plan(
            plan=plan,
            agent_context=agent_context,
            article_context=article_context,
            allowed_literature_citation_keys=allowed_literature_citation_keys,
            direct_comparator_literature_keys=direct_comparator_literature_keys,
            allowed_know_how_decisions=allowed_know_how_decisions,
            enforce_article_contract=enforce_article_contract,
            llm=self.llm,
        )
        return plan, receipt

    def run(
        self,
        context: ResearchContext,
        *,
        allowed_know_how_decisions: Mapping[str, Mapping[str, Any]] | None = None,
        allowed_literature_citation_keys: Sequence[str] | None = None,
        direct_comparator_literature_keys: Sequence[str] | None = None,
        know_how_context: str = "",
        enforce_article_contract: bool = False,
        article_contract_context: Optional[ResearchContext] = None,
        planning_contract_context: str = "",
        progress_callback: Optional[Callable[[Any], None]] = None,
    ) -> AnalysisPlan:
        if bool(allowed_know_how_decisions) != bool(know_how_context):
            raise ValueError(
                "Progressive Planner know-how authority and prompt must be supplied together"
            )
        article_context = article_contract_context or context
        allowed_citations = tuple(
            dict.fromkeys(
                str(value).strip()
                for value in (allowed_literature_citation_keys or ())
                if str(value).strip()
            )
        )
        direct_keys = tuple(
            dict.fromkeys(
                str(value).strip()
                for value in (direct_comparator_literature_keys or ())
                if str(value).strip()
            )
        )
        analysis_types, variables, action_ids, action_rows = self._request_authorities(
            context
        )
        resolved_planning_contract_context = bind_literature_citation_authority(
            planning_contract_context,
            allowed_citations,
            direct_comparator_keys=direct_keys,
            required_method_layers=required_method_layers_for_context(context),
        )
        structured_output = None
        if llm_supports_strict_json_schema(self.llm):
            structured_output = progressive_structured_output_request(
                analysis_types=analysis_types,
                variable_names=variables,
                cohort_concept_ids=progressive_cohort_concept_ids(
                    context,
                    variables,
                ),
                scientific_action_ids=action_ids,
                allowed_literature_citation_keys=allowed_citations,
                allowed_know_how_decisions=allowed_know_how_decisions,
            )
        user_prompt = self._user_prompt(
            context,
            article_context=article_context,
            analysis_types=analysis_types,
            variables=variables,
            action_rows=action_rows,
            allowed_literature_citation_keys=allowed_citations,
            know_how_context=know_how_context,
            planning_contract_context=resolved_planning_contract_context,
        )
        user_prompt_without_know_how = self._user_prompt(
            context,
            article_context=article_context,
            analysis_types=analysis_types,
            variables=variables,
            action_rows=action_rows,
            allowed_literature_citation_keys=allowed_citations,
            planning_contract_context=resolved_planning_contract_context,
        )
        messages = [
            LLMMessage(role="system", content=_GUIDE),
            LLMMessage(role="user", content=user_prompt),
        ]
        message_bytes = sum(len(item.content.encode("utf-8")) for item in messages)
        schema_bytes = structured_output.payload_bytes if structured_output else 0
        total_bytes = message_bytes + schema_bytes
        if total_bytes > _MAX_REQUEST_BYTES:
            raise ProgressivePlanCompileError(
                "progressive_prompt_budget_exceeded",
                f"initial request uses {total_bytes} bytes; limit={_MAX_REQUEST_BYTES}",
                path="planner_request",
            )
        self.last_prompt_metrics = {
            "message_payload_bytes": message_bytes,
            "structured_output_payload_bytes": schema_bytes,
            "structured_output_authority_sha256": (
                structured_output.authority_sha256 if structured_output else None
            ),
            "total_bytes": total_bytes,
            "without_know_how_total_bytes": (
                len(_GUIDE.encode("utf-8"))
                + len(user_prompt_without_know_how.encode("utf-8"))
                + schema_bytes
            ),
            "selected_variable_count": len(variables),
            "selected_variable_roster": list(variables),
            "selected_variable_roster_sha256": canonical_sha256(list(variables)),
            "candidate_analysis_types": list(analysis_types),
            "selected_scientific_action_ids": list(action_ids),
            "planner_strategy": "progressive_v2",
            "compile_revision_count": 0,
            "suffix_revision_count": 0,
            "full_revision_count": 0,
        }
        skeleton = call_llm_with_structured_retry(
            self.llm,
            messages,
            parser=lambda raw: _parse_model(raw, ProgressivePlanSkeleton),
            role="progressive_planner_skeleton",
            max_retries=_MAX_INITIAL_PARSE_RETRIES,
            max_tokens=_MAX_OUTPUT_TOKENS,
            temperature=0.2,
            include_failed_response_on_retry=False,
            progress_callback=progress_callback,
            structured_output=structured_output,
            format_reminder=(
                "Return the complete ProgressivePlanSkeleton. Emit null and [] "
                "for inactive fields required by the strict transport schema."
            ),
        )
        current = skeleton
        last_error: Optional[ProgressivePlanCompileError] = None
        for revision in range(_MAX_COMPILE_REVISIONS + 1):
            try:
                plan, receipt = self._compile_and_accept(
                    current,
                    agent_context=context,
                    article_context=article_context,
                    allowed_literature_citation_keys=allowed_citations,
                    direct_comparator_literature_keys=direct_keys,
                    allowed_know_how_decisions=allowed_know_how_decisions,
                    enforce_article_contract=enforce_article_contract,
                )
            except ProgressivePlanCompileError as exc:
                last_error = exc
                if revision >= _MAX_COMPILE_REVISIONS:
                    raise
                self.last_prompt_metrics["compile_revision_count"] = revision + 1
                if exc.step_index is None or exc.step_id is None:
                    self.last_prompt_metrics["full_revision_count"] += 1
                    repair_messages = [
                        *messages,
                        LLMMessage(
                            role="user",
                            content=(
                                "HOST COMPILER OBSERVATION:\n"
                                + json.dumps(exc.details, ensure_ascii=False)
                                + "\n\nCurrent skeleton:\n"
                                + current.model_dump_json()
                                + "\n\nReturn a corrected complete skeleton."
                            ),
                        ),
                    ]
                    current = call_llm_with_structured_retry(
                        self.llm,
                        repair_messages,
                        parser=lambda raw: _parse_model(raw, ProgressivePlanSkeleton),
                        role="progressive_planner_full_revision",
                        max_retries=1,
                        max_tokens=_MAX_OUTPUT_TOKENS,
                        temperature=0.2,
                        include_failed_response_on_retry=False,
                        progress_callback=progress_callback,
                        structured_output=structured_output,
                    )
                    continue
                replace_index = int(exc.step_index)
                rejected_step_id = exc.step_id
                prefix = list(current.steps[:replace_index])
                prior_receipt: Optional[ProgressivePlanCompileReceipt] = None
                if prefix:
                    prefix_skeleton = ProgressivePlanSkeleton.model_validate(
                        {
                            **current.model_dump(mode="json"),
                            "steps": [item.model_dump(mode="json") for item in prefix],
                        }
                    )
                    _prefix_plan, prior_receipt = compile_progressive_plan(
                        skeleton=prefix_skeleton,
                        context=context,
                        allowed_literature_citation_keys=allowed_citations,
                        allowed_know_how_decisions=allowed_know_how_decisions,
                    )
                suffix_schema = None
                if structured_output is not None:
                    suffix_schema = progressive_structured_output_request(
                        analysis_types=analysis_types,
                        variable_names=variables,
                        scientific_action_ids=action_ids,
                        allowed_literature_citation_keys=allowed_citations,
                        suffix=True,
                    )
                suffix_prompt = (
                    user_prompt
                    + "\n\nIMMUTABLE COMPILED PREFIX (do not return or change):\n"
                    + json.dumps(
                        [
                            {
                                "step_id": item.step_id,
                                "module_id": item.module_id,
                                "skeleton_sha256": (
                                    prior_receipt.compiled_steps[index].skeleton_sha256
                                    if prior_receipt is not None
                                    else None
                                ),
                            }
                            for index, item in enumerate(prefix)
                        ],
                        ensure_ascii=False,
                    )
                    + "\n\nCURRENT UNLOCKED SUFFIX:\n"
                    + json.dumps(
                        [
                            item.model_dump(mode="json")
                            for item in current.steps[replace_index:]
                        ],
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                    + "\n\nHOST COMPILER OBSERVATION:\n"
                    + json.dumps(exc.details, ensure_ascii=False)
                    + "\n\nReturn only a ProgressiveSuffixRevision beginning at "
                    + repr(exc.step_id)
                    + ". Preserve every satisfied requirement in the suffix."
                )
                revision_obj = call_llm_with_structured_retry(
                    self.llm,
                    [
                        LLMMessage(role="system", content=_GUIDE),
                        LLMMessage(role="user", content=suffix_prompt),
                    ],
                    parser=lambda raw: _parse_suffix_revision(
                        raw,
                        expected_step_id=rejected_step_id,
                    ),
                    role="progressive_planner_suffix_revision",
                    max_retries=1,
                    max_tokens=_MAX_OUTPUT_TOKENS,
                    temperature=0.2,
                    include_failed_response_on_retry=False,
                    progress_callback=progress_callback,
                    structured_output=suffix_schema,
                )
                merged_payload = current.model_dump(mode="json")
                merged_payload["steps"] = [
                    *[item.model_dump(mode="json") for item in prefix],
                    *[
                        item.model_dump(mode="json")
                        for item in revision_obj.replacement_steps
                    ],
                ]
                try:
                    revised = ProgressivePlanSkeleton.model_validate(merged_payload)
                except ValidationError as merge_error:
                    raise ProgressivePlanCompileError(
                        "progressive_suffix_merge_invalid",
                        str(merge_error),
                        step_id=exc.step_id,
                        step_index=replace_index,
                        path="replacement_steps",
                    ) from merge_error
                if prior_receipt is not None:
                    assert_immutable_prefix(
                        prior_receipt=prior_receipt,
                        revised_skeleton=revised,
                        locked_step_count=replace_index,
                    )
                current = revised
                self.last_prompt_metrics["suffix_revision_count"] += 1
                continue
            self.last_skeleton = current
            self.last_compile_receipt = receipt
            self.last_prompt_metrics["final_skeleton_sha256"] = receipt.skeleton_sha256
            self.last_prompt_metrics["compiled_plan_sha256"] = (
                receipt.analysis_plan_sha256
            )
            return plan
        if last_error is not None:  # pragma: no cover - loop raises at the cap
            raise last_error
        raise RuntimeError("progressive planner reached an impossible empty state")


__all__ = [
    "ProgressivePlannerAgent",
    "candidate_analysis_types",
    "progressive_cohort_concept_ids",
    "select_progressive_variables",
]
