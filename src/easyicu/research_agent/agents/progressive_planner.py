"""Progressive Planner v2: retrieve, skeletonize, compile, revise a suffix."""

from __future__ import annotations

import json
import re
from typing import Any, Callable, Mapping, Optional, Sequence

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
from ..planning.method_literature import (
    reporting_method_source_keys_for_guidelines,
)
from ..planning.planner_output_contract import (
    validate_fresh_planner_typed_product_specs,
)
from ..planning.preplan_know_how import verify_know_how_decisions
from ..planning.primary_result_contract import validate_required_primary_result
from ..planning.progressive_compiler import (
    compile_progressive_plan,
)
from ..planning.progressive_contract import (
    ProgressiveCohortIntent,
    ProgressiveFoundationMaterialization,
    ProgressiveOutlineStep,
    ProgressivePlanCompileError,
    ProgressivePlanCompileReceipt,
    ProgressivePlanOutline,
    ProgressivePlannerCheckpoint,
    ProgressivePlanSkeleton,
    ProgressiveStepMaterialization,
)
from ..planning.progressive_artifacts import ProgressivePlannerCheckpointEmitter
from ..planning.progressive_resume import (
    ProgressivePrefixState,
    assemble_progressive_skeleton,
    build_progressive_checkpoint_authorities,
    compile_progressive_prefix,
    restore_progressive_resume_foundation,
    restore_progressive_resume_prefix,
    restore_progressive_resume_prompt_metrics,
    validate_progressive_materialization_coordinate,
    validate_progressive_resume_runtime_dependencies,
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
from .progressive_payload import (
    progressive_foundation_structured_output_request,
    progressive_outline_structured_output_request,
    progressive_step_materialization_request,
)
from .plan_payload import bind_literature_citation_authority


_GUIDE = load_prompt_pack()["progressive_planner"]
_MAX_INITIAL_PARSE_RETRIES = 2
_MAX_COMPILE_REVISIONS = 4
_MAX_OUTLINE_OUTPUT_TOKENS = 4_000
_MAX_FOUNDATION_OUTPUT_TOKENS = 4_000
_MAX_STEP_OUTPUT_TOKENS = 8_000
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


def _parse_foundation_materialization(
    raw: str,
    *,
    host_cohort: ProgressiveCohortIntent | None,
) -> ProgressiveFoundationMaterialization:
    payload = json.loads(str(raw or "").strip())
    if not isinstance(payload, dict):
        raise ValueError("progressive Planner response root must be an object")
    if host_cohort is not None:
        foundation = payload.get("foundation")
        if not isinstance(foundation, dict):
            raise ValueError("progressive Planner foundation must be an object")
        payload = dict(payload)
        payload["foundation"] = {
            **foundation,
            "cohort": host_cohort.model_dump(mode="json"),
        }
    return ProgressiveFoundationMaterialization.model_validate(payload)


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


def _article_reporting_source_keys(
    *,
    article_context: ResearchContext,
    analysis_type: str,
    enforce_article_contract: bool,
) -> tuple[str, ...]:
    if not enforce_article_contract:
        return ()
    contract = build_article_analysis_contract(
        article_context,
        analysis_type=analysis_type,
    )
    return reporting_method_source_keys_for_guidelines(
        contract.reporting_guidelines
    )


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
        self.last_outline: Optional[ProgressivePlanOutline] = None
        self.last_foundation: Optional[ProgressiveFoundationMaterialization] = None
        self.last_materializations: list[ProgressiveStepMaterialization] = []
        self.last_skeleton: Optional[ProgressivePlanSkeleton] = None
        self.last_resume_validated = False
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
    def _retrieved_data_cards(
        context: ResearchContext,
        variables: Sequence[str],
    ) -> list[dict[str, Any]]:
        selected = set(variables)
        return [
            variable.model_dump(
                mode="json",
                include={
                    "name",
                    "role",
                    "dtype",
                    "source_concept",
                    "derived_from_concepts",
                },
            )
            for variable in context.variables
            if variable.name in selected
        ]

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
            "Research question and sealed study anchors:\n"
            + json.dumps(
                {
                    "research_question": context.research_question,
                    "primary_exposure": context.primary_exposure,
                    "target_outcome": context.target_outcome,
                    "cohort": context.cohort.model_dump(
                        mode="json",
                        include={
                            "cohort_name",
                            "database",
                            "n_stays",
                            "id_columns",
                            "outcome_columns",
                        },
                    ),
                },
                ensure_ascii=False,
                separators=(",", ":"),
            ),
            "Retrieved data cards (high-level fields only):\n"
            + json.dumps(
                ProgressivePlannerAgent._retrieved_data_cards(context, variables),
                ensure_ascii=False,
                separators=(",", ":"),
            ),
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
            "Return only a concise ProgressivePlanOutline. Do not return raw or "
            "product inputs, output tokens, Table 1 fields, model terms, level "
            "indices, denominator or missingness policies, or literature "
            "applications. For each step select only the retrieved variable "
            "names and sealed literature keys that its later materialization "
            "needs. The host requests and validates executable details only "
            "when it materializes the current step."
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
            request = progressive_outline_structured_output_request(
                analysis_types=analysis_types,
                variable_names=variables,
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

    @staticmethod
    def _validate_outline_authority(
        outline: ProgressivePlanOutline,
        *,
        analysis_types: Sequence[str],
        variable_names: Sequence[str],
        allowed_literature_citation_keys: Sequence[str],
    ) -> None:
        if outline.analysis_type not in set(analysis_types):
            raise ProgressivePlanCompileError(
                "progressive_outline_analysis_type_unavailable",
                f"outline selected unavailable analysis type {outline.analysis_type!r}",
                path="analysis_type",
            )
        allowed_actions, _rows = _action_catalog((outline.analysis_type,))
        allowed = set(allowed_actions)
        available_variables = set(variable_names)
        available_citations = set(allowed_literature_citation_keys)
        for index, step in enumerate(outline.steps):
            unknown_variables = sorted(set(step.variable_names) - available_variables)
            if unknown_variables:
                raise ProgressivePlanCompileError(
                    "progressive_outline_variable_unavailable",
                    "outline selected variable(s) outside the retrieved roster: "
                    + ", ".join(unknown_variables),
                    step_id=step.step_id,
                    step_index=index,
                    path="variable_names",
                )
            unknown_citations = sorted(
                set(step.literature_citation_keys) - available_citations
            )
            if unknown_citations:
                raise ProgressivePlanCompileError(
                    "progressive_outline_citation_unavailable",
                    "outline selected citation(s) outside the sealed roster: "
                    + ", ".join(unknown_citations),
                    step_id=step.step_id,
                    step_index=index,
                    path="literature_citation_keys",
                )
            action = step.scientific_action_id
            if action is not None and action not in allowed:
                raise ProgressivePlanCompileError(
                    "progressive_outline_action_unavailable",
                    f"outline action {action!r} is not available for "
                    f"analysis type {outline.analysis_type!r}",
                    step_id=step.step_id,
                    step_index=index,
                    path="scientific_action_id",
                )

    @staticmethod
    def _foundation_prompt(
        *,
        context: ResearchContext,
        outline: ProgressivePlanOutline,
        outline_sha256: str,
        variables: Sequence[str],
        know_how_context: str,
        planning_contract_context: str,
        host_cohort: ProgressiveCohortIntent | None,
    ) -> str:
        blocks = [
            "PROGRESSIVE PLAN-FOUNDATION AUTHORITY",
            "Host-validated outline and digest:\n"
            + json.dumps(
                {
                    "outline": outline.model_dump(mode="json"),
                    "outline_sha256": outline_sha256,
                },
                ensure_ascii=False,
                separators=(",", ":"),
            ),
            "Outbound-safe data authority for plan-wide choices:\n"
            + format_outbound_safe_context(context, variable_names=variables),
        ]
        if planning_contract_context:
            blocks.append(
                "Run-specific article/task contract (binding):\n"
                + planning_contract_context
            )
        if know_how_context:
            blocks.append("Retrieved protocol know-how (binding):\n" + know_how_context)
        if host_cohort is not None:
            blocks.append(
                "Caller-bound cohort authority (host owned; copy exactly and do "
                "not reinterpret):\n"
                + host_cohort.model_dump_json()
            )
        blocks.append(
            "Return one ProgressiveFoundationMaterialization only. Bind cohort "
            "selection, display labels, robustness intents, and any authorized "
            "know-how decisions. Do not return executable step fields."
        )
        return "\n\n".join(blocks)

    @staticmethod
    def _materialization_prompt(
        *,
        context: ResearchContext,
        outline: ProgressivePlanOutline,
        outline_step: ProgressiveOutlineStep,
        outline_step_sha256: str,
        variables: Sequence[str],
        action_rows: Sequence[Mapping[str, Any]],
        allowed_literature_citation_keys: Sequence[str],
        know_how_context: str,
        planning_contract_context: str,
        prefix_summary: Sequence[Mapping[str, Any]],
        available_product_refs: Sequence[tuple[str, str]],
        compiler_observation: Mapping[str, Any] | None = None,
    ) -> str:
        blocks = [
            "PROGRESSIVE CURRENT-STEP MATERIALIZATION AUTHORITY",
            "Selected analysis family:\n" + outline.analysis_type,
            "High-level cohort objective:\n" + outline.cohort_objective,
            "Current outline step and host digest:\n"
            + json.dumps(
                {
                    "outline_step": outline_step.model_dump(mode="json"),
                    "outline_step_sha256": outline_step_sha256,
                },
                ensure_ascii=False,
                separators=(",", ":"),
            ),
            "Immutable materialized prefix summary (not editable):\n"
            + json.dumps(
                list(prefix_summary),
                ensure_ascii=False,
                separators=(",", ":"),
            ),
            "Host product registry visible to this materialization. These are "
            "planning edges only; runtime consumption still requires verified "
            "executed evidence/capsule authority:\n"
            + json.dumps(
                [
                    {"producer_step_id": producer, "product_id": product}
                    for producer, product in available_product_refs
                ],
                ensure_ascii=False,
                separators=(",", ":"),
            ),
            "Retrieved scientific action/method cards for the selected family:\n"
            + json.dumps(
                list(action_rows),
                ensure_ascii=False,
                separators=(",", ":"),
            ),
            "Sealed literature citation keys:\n"
            + json.dumps(
                list(allowed_literature_citation_keys),
                ensure_ascii=False,
            ),
            "Outbound-safe data authority for this current step:\n"
            + format_outbound_safe_context(context, variable_names=variables),
        ]
        if planning_contract_context:
            blocks.append(
                "Run-specific article/task contract (binding):\n"
                + planning_contract_context
            )
        if know_how_context:
            blocks.append(
                "Retrieved protocol know-how (binding):\n" + know_how_context
            )
        if compiler_observation:
            blocks.append(
                "HOST COMPILER OBSERVATION FOR THIS CURRENT STEP:\n"
                + json.dumps(
                    dict(compiler_observation),
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            )
        blocks.append(
            "Return one ProgressiveStepMaterialization only. Copy every "
            "outline-owned coordinate exactly. Return foundation=null; the host "
            "already sealed it. Do not return or rewrite any prefix or future step."
        )
        return "\n\n".join(blocks)

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
        reporting_source_keys = _article_reporting_source_keys(
            article_context=article_context,
            analysis_type=skeleton.analysis_type,
            enforce_article_contract=enforce_article_contract,
        )
        plan, receipt = compile_progressive_plan(
            skeleton=skeleton,
            context=agent_context,
            allowed_literature_citation_keys=allowed_literature_citation_keys,
            allowed_know_how_decisions=allowed_know_how_decisions,
            host_reporting_method_source_keys=reporting_source_keys,
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

    def _materialize_remaining_steps(
        self,
        prefix_state: ProgressivePrefixState,
        *,
        context: ResearchContext,
        outline: ProgressivePlanOutline,
        foundation_materialization: ProgressiveFoundationMaterialization,
        scientific_action_ids: Sequence[str],
        action_rows: Sequence[Mapping[str, Any]],
        allowed_literature_citation_keys: Sequence[str],
        allowed_know_how_decisions: Mapping[str, Mapping[str, Any]] | None,
        reporting_method_source_keys: Sequence[str],
        planning_contract_context: str,
        progress_callback: Optional[Callable[[Any], None]],
        checkpoint_emitter: ProgressivePlannerCheckpointEmitter,
        resumed: bool,
    ) -> ProgressivePrefixState:
        """Materialize and locally repair only the uncompiled suffix."""

        foundation = foundation_materialization.foundation
        for step_index in range(len(prefix_state.steps), len(outline.steps)):
            outline_step = outline.steps[step_index]
            step_variables = tuple(outline_step.variable_names)
            step_citations = tuple(outline_step.literature_citation_keys)
            outline_step_sha256 = canonical_sha256(
                outline_step.model_dump(mode="json")
            )
            step_schema = None
            if llm_supports_strict_json_schema(self.llm):
                step_schema = progressive_step_materialization_request(
                    outline_step=outline_step,
                    outline_step_sha256=outline_step_sha256,
                    variable_names=step_variables,
                    scientific_action_ids=scientific_action_ids,
                    allowed_literature_citation_keys=step_citations,
                    available_product_refs=prefix_state.available_product_refs,
                )
            compiler_observation: Mapping[str, Any] | None = None
            for revision in range(_MAX_COMPILE_REVISIONS + 1):
                materialization_prompt = self._materialization_prompt(
                    context=context,
                    outline=outline,
                    outline_step=outline_step,
                    outline_step_sha256=outline_step_sha256,
                    variables=step_variables,
                    action_rows=action_rows,
                    allowed_literature_citation_keys=step_citations,
                    know_how_context="",
                    planning_contract_context=planning_contract_context,
                    prefix_summary=prefix_state.prompt_summary,
                    available_product_refs=prefix_state.available_product_refs,
                    compiler_observation=compiler_observation,
                )
                step_messages = [
                    LLMMessage(role="system", content=_GUIDE),
                    LLMMessage(role="user", content=materialization_prompt),
                ]
                step_payload_bytes = sum(
                    len(item.content.encode("utf-8")) for item in step_messages
                ) + (step_schema.payload_bytes if step_schema is not None else 0)
                if step_payload_bytes > _MAX_REQUEST_BYTES:
                    raise ProgressivePlanCompileError(
                        "progressive_step_prompt_budget_exceeded",
                        f"current-step request uses {step_payload_bytes} bytes; "
                        f"limit={_MAX_REQUEST_BYTES}",
                        step_id=outline_step.step_id,
                        step_index=step_index,
                        path="planner_request",
                    )
                self.last_prompt_metrics[
                    "step_materialization_attempt_payload_bytes"
                ].append(step_payload_bytes)
                self.last_prompt_metrics[
                    "step_materialization_attempt_schema_sha256"
                ].append(step_schema.authority_sha256 if step_schema else None)
                if resumed:
                    self.last_prompt_metrics[
                        "current_run_step_materialization_attempt_payload_bytes"
                    ].append(step_payload_bytes)
                    self.last_prompt_metrics[
                        "current_run_step_materialization_attempt_schema_sha256"
                    ].append(
                        step_schema.authority_sha256 if step_schema else None
                    )
                materialization = call_llm_with_structured_retry(
                    self.llm,
                    step_messages,
                    parser=lambda raw: _parse_model(
                        raw,
                        ProgressiveStepMaterialization,
                    ),
                    role="progressive_planner_step_materialization",
                    max_retries=1,
                    max_tokens=_MAX_STEP_OUTPUT_TOKENS,
                    temperature=0.2,
                    include_failed_response_on_retry=False,
                    progress_callback=progress_callback,
                    structured_output=step_schema,
                    format_reminder=(
                        "Return exactly one ProgressiveStepMaterialization for "
                        "the current outline coordinate. Never return other steps."
                    ),
                )
                validate_progressive_materialization_coordinate(
                    materialization,
                    outline_step=outline_step,
                    outline_step_sha256=outline_step_sha256,
                    step_index=step_index,
                )
                try:
                    candidate_state = compile_progressive_prefix(
                        prefix_state,
                        materialization,
                        outline=outline,
                        foundation=foundation,
                        context=context,
                        allowed_literature_citation_keys=(
                            allowed_literature_citation_keys
                        ),
                        allowed_know_how_decisions=allowed_know_how_decisions,
                        reporting_method_source_keys=reporting_method_source_keys,
                    )
                except ProgressivePlanCompileError as exc:
                    if revision >= _MAX_COMPILE_REVISIONS:
                        raise
                    if exc.step_index is not None and int(exc.step_index) < step_index:
                        raise ProgressivePlanCompileError(
                            "progressive_materialization_invalidated_prefix",
                            "current-step materialization exposed a finding in the "
                            "already compiled prefix",
                            step_id=exc.step_id,
                            step_index=exc.step_index,
                            path=exc.path,
                        ) from exc
                    self.last_prompt_metrics["compile_revision_count"] += 1
                    if resumed:
                        self.last_prompt_metrics[
                            "current_run_compile_revision_count"
                        ] += 1
                    compiler_observation = exc.details
                    continue
                prefix_state = candidate_state
                self.last_materializations = list(prefix_state.materializations)
                self.last_prompt_metrics[
                    "step_materialization_payload_bytes"
                ].append(step_payload_bytes)
                self.last_prompt_metrics[
                    "step_materialization_schema_sha256"
                ].append(step_schema.authority_sha256 if step_schema else None)
                self.last_prompt_metrics["step_materialization_count"] += 1
                if resumed:
                    self.last_prompt_metrics[
                        "current_run_step_materialization_count"
                    ] += 1
                checkpoint_emitter.emit(
                    stage="step",
                    outline=outline,
                    foundation=foundation_materialization,
                    materializations=self.last_materializations,
                    prompt_metrics=self.last_prompt_metrics,
                )
                break
        return prefix_state

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
        checkpoint_callback: Optional[
            Callable[[ProgressivePlannerCheckpoint], None]
        ] = None,
        resume_checkpoint: ProgressivePlannerCheckpoint | None = None,
        resume_dependency_context: Mapping[str, Any] | None = None,
        required_primary_cohort_selection_mode: str | None = None,
    ) -> AnalysisPlan:
        self.last_resume_validated = False
        if bool(allowed_know_how_decisions) != bool(know_how_context):
            raise ValueError(
                "Progressive Planner know-how authority and prompt must be supplied together"
            )
        article_context = article_contract_context or context
        if required_primary_cohort_selection_mode not in {
            None,
            "all_input_rows",
            "predicate_filtered",
        }:
            raise ValueError(
                "required_primary_cohort_selection_mode is unavailable"
            )
        host_cohort = (
            ProgressiveCohortIntent(
                name=context.cohort.cohort_name,
                selection_mode="all_input_rows",
                inclusion=[],
                exclusion=[],
            )
            if required_primary_cohort_selection_mode == "all_input_rows"
            else None
        )
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
        if resume_checkpoint is not None:
            validate_progressive_resume_runtime_dependencies(
                resume_dependency_context
            )
        scientific_authority = {
            "analysis_types": list(analysis_types),
            "variables": list(variables),
            "scientific_action_ids": list(action_ids),
            "allowed_literature_citation_keys": list(allowed_citations),
            "direct_comparator_literature_keys": list(direct_keys),
            "allowed_know_how_decisions": dict(
                allowed_know_how_decisions or {}
            ),
            "know_how_context": know_how_context,
            "planning_contract_context": resolved_planning_contract_context,
            "required_primary_cohort_selection_mode": (
                required_primary_cohort_selection_mode
            ),
            "host_cohort": (
                host_cohort.model_dump(mode="json")
                if host_cohort is not None
                else None
            ),
        }
        checkpoint_authorities = build_progressive_checkpoint_authorities(
            context=context,
            article_context=article_context,
            scientific_authority=scientific_authority,
            runtime_dependency_authority=resume_dependency_context,
        )
        checkpoint_emitter = ProgressivePlannerCheckpointEmitter(
            callback=checkpoint_callback,
            request_authority_sha256=(
                resume_checkpoint.request_authority_sha256
                if resume_checkpoint is not None
                else checkpoint_authorities.request_authority_sha256
            ),
            source_checkpoint=resume_checkpoint,
        )
        outline_schema = None
        if llm_supports_strict_json_schema(self.llm):
            outline_schema = progressive_outline_structured_output_request(
                analysis_types=analysis_types,
                variable_names=variables,
                scientific_action_ids=action_ids,
                allowed_literature_citation_keys=allowed_citations,
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
        schema_bytes = outline_schema.payload_bytes if outline_schema else 0
        total_bytes = message_bytes + schema_bytes
        if total_bytes > _MAX_REQUEST_BYTES:
            raise ProgressivePlanCompileError(
                "progressive_prompt_budget_exceeded",
                f"initial request uses {total_bytes} bytes; limit={_MAX_REQUEST_BYTES}",
                path="planner_request",
            )
        current_prompt_metrics = {
            "message_payload_bytes": message_bytes,
            "structured_output_payload_bytes": schema_bytes,
            "structured_output_authority_sha256": (
                outline_schema.authority_sha256 if outline_schema else None
            ),
            "total_bytes": total_bytes,
            "outline_request_payload_bytes": total_bytes,
            "outline_schema_bytes": schema_bytes,
            "foundation_request_payload_bytes": 0,
            "foundation_schema_bytes": 0,
            "foundation_structured_output_authority_sha256": None,
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
            "foundation_cohort_owner": (
                "host_required_primary_cohort"
                if host_cohort is not None
                else "planner"
            ),
            "required_primary_cohort_selection_mode": (
                required_primary_cohort_selection_mode
            ),
            "resume_dependency_authority_sha256": (
                checkpoint_authorities.resume_dependency_authority_sha256
            ),
            "compile_revision_count": 0,
            "step_materialization_count": 0,
            "step_materialization_payload_bytes": [],
            "step_materialization_schema_sha256": [],
            "step_materialization_attempt_payload_bytes": [],
            "step_materialization_attempt_schema_sha256": [],
            "suffix_revision_count": 0,
            "full_revision_count": 0,
            "suffix_request_payload_bytes": [],
        }
        if resume_checkpoint is not None:
            self.last_prompt_metrics = restore_progressive_resume_prompt_metrics(
                checkpoint=resume_checkpoint,
                current_prompt_metrics=current_prompt_metrics,
                expected_dependency_sha256=(
                    checkpoint_authorities.resume_dependency_authority_sha256
                ),
            )
            outline = resume_checkpoint.outline
        else:
            self.last_prompt_metrics = current_prompt_metrics
            outline = call_llm_with_structured_retry(
                self.llm,
                messages,
                parser=lambda raw: _parse_model(raw, ProgressivePlanOutline),
                role="progressive_planner_outline",
                max_retries=_MAX_INITIAL_PARSE_RETRIES,
                max_tokens=_MAX_OUTLINE_OUTPUT_TOKENS,
                temperature=0.2,
                include_failed_response_on_retry=False,
                progress_callback=progress_callback,
                structured_output=outline_schema,
                format_reminder=(
                    "Return one concise ProgressivePlanOutline only. Do not include "
                    "any executable step-detail fields."
                ),
            )
        self._validate_outline_authority(
            outline,
            analysis_types=analysis_types,
            variable_names=variables,
            allowed_literature_citation_keys=allowed_citations,
        )
        self.last_outline = outline
        self.last_foundation = None
        self.last_materializations = []
        outline_sha256 = canonical_sha256(outline.model_dump(mode="json"))
        if resume_checkpoint is not None:
            if self.last_prompt_metrics.get("outline_sha256") != outline_sha256:
                raise ProgressivePlanCompileError(
                    "progressive_resume_outline_digest_mismatch",
                    "development checkpoint metrics identify another outline",
                    path="resume_checkpoint.prompt_metrics.outline_sha256",
                )
        else:
            self.last_prompt_metrics["outline_sha256"] = outline_sha256
            checkpoint_emitter.emit(
                stage="outline",
                outline=outline,
                foundation=None,
                materializations=self.last_materializations,
                prompt_metrics=self.last_prompt_metrics,
            )

        foundation_schema = None
        if llm_supports_strict_json_schema(self.llm):
            foundation_schema = progressive_foundation_structured_output_request(
                outline_sha256=outline_sha256,
                variable_names=variables,
                cohort_concept_ids=progressive_cohort_concept_ids(context, variables),
                allowed_know_how_decisions=allowed_know_how_decisions,
                required_cohort_selection_mode=(
                    required_primary_cohort_selection_mode
                ),
                required_cohort_name=(
                    context.cohort.cohort_name
                    if required_primary_cohort_selection_mode is not None
                    else None
                ),
            )
        foundation_prompt = self._foundation_prompt(
            context=context,
            outline=outline,
            outline_sha256=outline_sha256,
            variables=variables,
            know_how_context=know_how_context,
            planning_contract_context=resolved_planning_contract_context,
            host_cohort=host_cohort,
        )
        foundation_messages = [
            LLMMessage(role="system", content=_GUIDE),
            LLMMessage(role="user", content=foundation_prompt),
        ]
        foundation_message_bytes = sum(
            len(item.content.encode("utf-8")) for item in foundation_messages
        )
        foundation_schema_bytes = (
            foundation_schema.payload_bytes if foundation_schema is not None else 0
        )
        foundation_total_bytes = foundation_message_bytes + foundation_schema_bytes
        if foundation_total_bytes > _MAX_REQUEST_BYTES:
            raise ProgressivePlanCompileError(
                "progressive_foundation_prompt_budget_exceeded",
                f"foundation request uses {foundation_total_bytes} bytes; "
                f"limit={_MAX_REQUEST_BYTES}",
                path="planner_request",
            )
        current_foundation_authority = (
            foundation_schema.authority_sha256 if foundation_schema else None
        )
        foundation_materialization = (
            restore_progressive_resume_foundation(
                checkpoint=resume_checkpoint,
                prompt_metrics=self.last_prompt_metrics,
                request_payload_bytes=foundation_total_bytes,
                schema_bytes=foundation_schema_bytes,
                schema_authority_sha256=current_foundation_authority,
            )
            if resume_checkpoint is not None
            else None
        )
        resume_foundation_reused = foundation_materialization is not None
        if foundation_materialization is None:
            self.last_prompt_metrics["foundation_request_payload_bytes"] = (
                foundation_total_bytes
            )
            self.last_prompt_metrics["foundation_schema_bytes"] = (
                foundation_schema_bytes
            )
            self.last_prompt_metrics[
                "foundation_structured_output_authority_sha256"
            ] = current_foundation_authority
            foundation_materialization = call_llm_with_structured_retry(
                self.llm,
                foundation_messages,
                parser=lambda raw: _parse_foundation_materialization(
                    raw,
                    host_cohort=host_cohort,
                ),
                role="progressive_planner_foundation",
                max_retries=1,
                max_tokens=_MAX_FOUNDATION_OUTPUT_TOKENS,
                temperature=0.2,
                include_failed_response_on_retry=False,
                progress_callback=progress_callback,
                structured_output=foundation_schema,
                format_reminder=(
                    "Return exactly one ProgressiveFoundationMaterialization bound "
                    "to the supplied outline digest."
                ),
            )
        assert foundation_materialization is not None
        if foundation_materialization.outline_sha256 != outline_sha256:
            raise ProgressivePlanCompileError(
                "progressive_foundation_outline_digest_mismatch",
                "plan foundation did not bind the host-validated outline digest",
                path="outline_sha256",
            )
        foundation = foundation_materialization.foundation
        if (
            required_primary_cohort_selection_mode is not None
            and foundation.cohort.selection_mode
            != required_primary_cohort_selection_mode
        ):
            raise ProgressivePlanCompileError(
                "progressive_foundation_cohort_mode_mismatch",
                "plan foundation did not preserve the caller-bound primary "
                "cohort selection mode",
                path="cohort.selection_mode",
            )
        self.last_foundation = foundation_materialization
        if not resume_foundation_reused:
            checkpoint_emitter.emit(
                stage="foundation",
                outline=outline,
                foundation=foundation_materialization,
                materializations=self.last_materializations,
                prompt_metrics=self.last_prompt_metrics,
            )

        selected_action_ids, selected_action_rows = _action_catalog(
            (outline.analysis_type,)
        )
        reporting_source_keys = _article_reporting_source_keys(
            article_context=article_context,
            analysis_type=outline.analysis_type,
            enforce_article_contract=enforce_article_contract,
        )

        def current_step_schema_authority(
            outline_step: ProgressiveOutlineStep,
            outline_step_sha256: str,
            available_product_refs: Sequence[tuple[str, str]],
        ) -> str | None:
            if not llm_supports_strict_json_schema(self.llm):
                return None
            request = progressive_step_materialization_request(
                outline_step=outline_step,
                outline_step_sha256=outline_step_sha256,
                variable_names=tuple(outline_step.variable_names),
                scientific_action_ids=selected_action_ids,
                allowed_literature_citation_keys=tuple(
                    outline_step.literature_citation_keys
                ),
                available_product_refs=available_product_refs,
            )
            return request.authority_sha256

        prefix_state = (
            restore_progressive_resume_prefix(
                checkpoint=resume_checkpoint,
                outline=outline,
                foundation=foundation,
                context=context,
                step_schema_authority=current_step_schema_authority,
                allowed_literature_citation_keys=allowed_citations,
                allowed_know_how_decisions=allowed_know_how_decisions,
                reporting_method_source_keys=reporting_source_keys,
            )
            if resume_checkpoint is not None
            else ProgressivePrefixState()
        )
        self.last_materializations = list(prefix_state.materializations)
        self.last_resume_validated = resume_checkpoint is not None

        prefix_state = self._materialize_remaining_steps(
            prefix_state,
            context=context,
            outline=outline,
            foundation_materialization=foundation_materialization,
            scientific_action_ids=selected_action_ids,
            action_rows=selected_action_rows,
            allowed_literature_citation_keys=allowed_citations,
            allowed_know_how_decisions=allowed_know_how_decisions,
            reporting_method_source_keys=reporting_source_keys,
            planning_contract_context=resolved_planning_contract_context,
            progress_callback=progress_callback,
            checkpoint_emitter=checkpoint_emitter,
            resumed=resume_checkpoint is not None,
        )

        if prefix_state.plan is None or prefix_state.receipt is None:
            raise RuntimeError("progressive outline produced no materialized steps")
        skeleton = assemble_progressive_skeleton(
            outline=outline,
            foundation=foundation,
            steps=prefix_state.steps,
        )
        plan, receipt = self._compile_and_accept(
            skeleton,
            agent_context=context,
            article_context=article_context,
            allowed_literature_citation_keys=allowed_citations,
            direct_comparator_literature_keys=direct_keys,
            allowed_know_how_decisions=allowed_know_how_decisions,
            enforce_article_contract=enforce_article_contract,
        )
        self.last_skeleton = skeleton
        self.last_compile_receipt = receipt
        self.last_prompt_metrics["final_skeleton_sha256"] = receipt.skeleton_sha256
        self.last_prompt_metrics["compiled_plan_sha256"] = receipt.analysis_plan_sha256
        return plan


__all__ = [
    "ProgressivePlannerAgent",
    "candidate_analysis_types",
    "progressive_cohort_concept_ids",
    "select_progressive_variables",
]
