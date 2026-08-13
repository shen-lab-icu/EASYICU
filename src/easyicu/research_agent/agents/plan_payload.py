"""Schema-driven projection of raw Planner payloads.

This module owns the boundary between untrusted model JSON and the strict
``AnalysisPlan`` schema. It may discard presentation-only top/step chatter,
but an unknown key inside a scientific contract is a structured retry: silently
projecting it away could turn the Planner's intended design into a different,
valid-looking design. It normalizes only representation-level aliases.
"""

from __future__ import annotations

import dataclasses
import json
from typing import Any, Dict, List, Sequence, Tuple

from ..contracts.declared_product import (
    PLAN_MATERIALIZABLE_TYPED_OUTPUT_KINDS,
    RUNTIME_BINDABLE_TYPED_INPUT_KINDS,
    typed_product as _canonical_typed_product,
)
from ..planning.method_literature import METHOD_CARDS
from ..planning.literature_bindings import (
    allowed_method_source_keys,
    normalize_literature_citation_keys,
    validate_literature_citation_bindings,
)
from ..planning.primary_result_contract import model_terms_retry_guide
from ..planning.robustness_contract import RobustnessSpec
from ..planning.scientific_review import required_method_layers_for_plan
from ..schema import (
    AnalysisPlan,
    AnalysisStep,
    ArtifactConsumptionContract,
    PlannedModelRequirement,
    TableOneSpec,
    TableOneVariableSpec,
)


def planner_descriptive_method_guidance(analysis_type: str) -> str:
    """Return the exact contracts for compact descriptive host owners."""

    if str(analysis_type).strip().casefold() != "descriptive_epidemiology":
        return ""
    return (
        "Two compact descriptive methods have exact host contracts. For "
        "`method='descriptive_distribution'`, declare exactly one typed cohort "
        "input followed by exactly one categorical grouping column and exactly "
        "one continuous value column, in that order; declare only "
        "`table:distribution_prevalence`. Do not add a third column or an "
        "association to that step. For a non-causal two-continuous-variable "
        "association, use a separate `method='descriptive_association'` step "
        "with exactly one typed cohort input followed by the predictor and "
        "outcome columns, in that order, and exactly one "
        "`statistic:<descriptive_name>` output. This contract computes a "
        "complete-case Spearman rho without adjustment or imputation. A figure "
        "of the grouped distribution consumes only the distribution table; a "
        "figure of the association scalar consumes only its statistic. Never "
        "bundle the grouped distribution and the association into one step.\n\n"
    )


def planner_descriptive_robustness_guidance(analysis_type: str) -> str:
    """Keep effect robustness out of descriptive-only analysis families."""

    if str(analysis_type).strip().casefold() != "descriptive_epidemiology":
        return ""
    return (
        " This replay contract applies only when a primary fitted "
        "effect and its uncertainty already exist. For "
        "`analysis_type='descriptive_epidemiology'`, do NOT declare "
        "`robustness_specs`, a `robustness_sensitivity` step, effect-style "
        "products such as `primary_or`, or a robustness forest plot. Use the "
        "typed measurement/missingness audits above for denominator and "
        "complete-case availability checks; any additional descriptive "
        "summary must remain a separately declared descriptive method.\n\n"
    )


def planner_adjusted_association_owner_guidance() -> str:
    """Name the one method family implemented by the sealed host owner."""

    return (
        "`table:adjusted_association_estimates` requires "
        "`method_family='statsmodels_logit_mle'`; no sealed executor owns "
        "`statsmodels_glm_binomial`, which needs an agent-coded step and a "
        "different output. "
    )


def planner_endpoint_and_optional_science_guidance() -> str:
    """Render host-owned endpoint and optional post-analysis boundaries."""

    return (
        "`ResearchContext.endpoint` is sealed host authority: copy exactly; never "
        "infer or repair. A required missing endpoint blocks execution.\n\n"
        "Leave `evalue_conversion_spec` null unless requested; it requires a "
        "baseline-risk evidence id, rate and population columns, and exact "
        "population. Leave `subgroup_analysis_spec` null unless requested; then "
        "bind a primary model requirement and declare predictor, outcome, subgroup "
        "columns, quantile buckets, minimum sizes, effect scale, adjustment roster, "
        "and multiplicity family. The Planner chooses these fields.\n\n"
    )


def render_methodological_principles(principles: Sequence[Any]) -> str:
    """Project case-neutral methodological principles into Planner guidance."""

    errors = [principle for principle in principles if principle.kind == "error"]
    cautions = [
        principle for principle in principles if principle.kind == "caution"
    ]
    lines = [
        "\n\nCROSS-CUTTING ICU METHODOLOGY (case-neutral; apply when planning):",
        "Objective errors to avoid — wrong under any study design:",
    ]
    lines.extend(
        f"- [{principle.phase}] {principle.principle}" for principle in errors
    )
    lines.append(
        "Defensible choices — state and justify in the plan; do not let them "
        "pass silently, but the analyst, not these rules, decides:"
    )
    lines.extend(
        f"- [{principle.phase}] {principle.principle}" for principle in cautions
    )
    return "\n".join(lines)


def bind_literature_citation_authority(
    planning_contract_context: str,
    allowed_keys: Sequence[str],
    *,
    direct_comparator_keys: Sequence[str] = (),
) -> str:
    """Append role-bound LiteratureBundle authority to the Planner profile.

    Citation keys alone are not a scientific design aid: the model also needs
    to know which source supports which methodological decision, and which
    retrieved records survived the direct-comparator screen.  This projection
    is deliberately assembled by the host from the sealed pre-plan bundle.
    """

    if not allowed_keys:
        return planning_contract_context
    method_source_keys = allowed_method_source_keys(allowed_keys)
    direct_keys = tuple(
        key
        for key in normalize_literature_citation_keys(direct_comparator_keys)
        if key in set(allowed_keys)
    )
    method_cards = [
        card for card in METHOD_CARDS if card.source_key in method_source_keys
    ]
    authority = (
        "PRE-PLAN LITERATURE CITATION AUTHORITY (exact, run-bound):\n"
        "- allowed_literature_citation_keys: "
        + json.dumps(list(allowed_keys), ensure_ascii=False)
        + (
            "\n- allowed_method_source_keys: "
            + json.dumps(list(method_source_keys), ensure_ascii=False)
            if method_source_keys
            else ""
        )
        + (
            "\n- screened_direct_comparator_keys: "
            + json.dumps(list(direct_keys), ensure_ascii=False)
            if direct_keys
            else "\n- screened_direct_comparator_keys: []"
        )
        + "\n- Every primary, secondary, and sensitivity step MUST bind one or "
        "more exact values from this list in literature_citation_keys. Do not cite "
        "an evidence artifact, analysis contract, study-design brief, or invented "
        "semantic label in that field. Auxiliary steps may use an empty list."
        + " Every scientific step MUST also emit literature_design_bindings. "
        "Each record must use a bound citation_key, select one or more exact "
        "design_elements (population, time_zero, exposure, outcome, estimand, "
        "adjustment, dependence, missing_data, robustness, reporting), and state what was "
        "applied plus any deliberate divergence. Do not invent or copy a source "
        "quotation; the host joins the sealed excerpt."
        + (
            " Every scientific step MUST include at least one exact "
            "allowed_method_source_key that supports its design or method; a "
            "disease-definition or database paper alone is insufficient. Add "
            "topic/direct-comparator keys when they support the step's population, "
            "exposure, outcome, or interpretation."
            if method_source_keys
            else ""
        )
        + (
            " When screened_direct_comparator_keys is non-empty, at least one "
            "primary analysis step MUST additionally bind one of those keys. "
            "Use its source excerpt only to compare population, time zero, "
            "exposure, outcome/estimand, and analysis choices. It is not "
            "automatic authority to copy eligibility criteria or change this "
            "study's sealed ResearchContext."
            if direct_keys
            else ""
        )
        + (
            "\n- method_decision_cards (host-curated; id|layer|supported_design_elements|question|requirement|source):\n"
            + "\n".join(
                "  - "
                + " | ".join(
                    (
                        card.id,
                        card.layer,
                        ",".join(card.design_elements),
                        card.question,
                        card.requirement,
                        card.source_key,
                    )
                )
                for card in method_cards
            )
            if method_cards
            else ""
        )
    )
    return "\n\n".join(
        value for value in (planning_contract_context, authority) if value
    )


def literature_citation_retry_suffix(
    allowed_keys: Sequence[str],
    *,
    direct_comparator_keys: Sequence[str] = (),
) -> str:
    """Render exact citation keys in the structured-retry reminder."""

    if not allowed_keys:
        return ""
    method_source_keys = allowed_method_source_keys(allowed_keys)
    direct_keys = tuple(
        key
        for key in normalize_literature_citation_keys(direct_comparator_keys)
        if key in set(allowed_keys)
    )
    method_element_map = {
        key: sorted(
            {
                element
                for card in METHOD_CARDS
                if card.source_key == key
                for element in card.design_elements
            }
        )
        for key in method_source_keys
    }
    return (
        " Allowed literature_citation_keys for this run are exactly: "
        + json.dumps(list(allowed_keys), ensure_ascii=False)
        + "."
        + " Each scientific step must also include literature_design_bindings "
        "with citation_key, design_elements, application, and optional divergence."
        + (
            " Every scientific step must include at least one method-source key "
            "from: "
            + json.dumps(list(method_source_keys), ensure_ascii=False)
            + ". Method-source bindings may use ONLY the design elements in "
            "this exact host-owned map (use another source or omit an "
            "unsupported element; never broaden a method card): "
            + json.dumps(method_element_map, ensure_ascii=False, sort_keys=True)
            + "."
            if method_source_keys
            else ""
        )
        + (
            " At least one primary step must also cite a screened direct "
            "comparator from: "
            + json.dumps(list(direct_keys), ensure_ascii=False)
            + "."
            if direct_keys
            else ""
        )
    )


def planner_science_retry_guide() -> str:
    """Return schema-owned retry guidance outside the Planner god module."""

    optional_fields = (
        "Contract applicability is exact: `family_primary_result_requirement` is "
        "legal only on the primary step when `analysis_type` is `causal_inference` "
        "or `survival`. An `association_study` must omit that field and declare its "
        "supported adjusted model through `model_requirements`. Optional "
        "collections such as `know_how_decisions` use JSON arrays, never `null`. "
        "`input_consumption_contracts` accepts `input_key`, `mode`, and optional "
        "`role_column`, `expected_roles`, or `schema_version`. "
        "Omit `AnalysisPlan.endpoint` or emit null."
    )
    representation_fields = (
        "Use only schema product kinds: `protocol` is not a product kind. A "
        "non-executable future/feasibility step uses method "
        "`feasibility_protocol` and output `report:<name>`; never consume that "
        "terminal report downstream. "
        "`TableOneSpec` already requires "
        "`missingness_display='n_percent_by_group'`; do not add undeclared "
        "`report_missing_by_group`. An `exposure_outcome_distribution_spec` "
        "step must list its exact exposure and outcome in `inputs`. Preserve "
        "observed scalar types (JSON numbers remain numbers). On retry retain "
        "every article role and required `robustness_specs`; do not fix one "
        "error by dropping an already-satisfied requirement."
    )
    return (
        "\n\n"
        + model_terms_retry_guide()
        + "\n\n"
        + optional_fields
        + "\n\n"
        + representation_fields
    )


def _canonicalise_figure_output_alias(token: object) -> object:
    """Canonicalize only a colon-typed alias for a declared figure product."""

    if not isinstance(token, str):
        return token
    parsed = _canonical_typed_product(token)
    if parsed is None or parsed[0] != "figure":
        return token
    _kind, _separator, name = token.partition(":")
    return f"figure:{name.strip()}"


def _is_untyped_figure_alias_output(token: object) -> bool:
    """Reject an underscore figure alias before it silently loses its role."""

    if not isinstance(token, str):
        return False
    text = token.strip()
    if not text or ":" in text:
        return False
    if text.lower().endswith((".png", ".svg", ".pdf", ".tif", ".tiff")):
        return False
    head, separator, _rest = text.partition("_")
    if not separator:
        return False
    probe = _canonical_typed_product(f"{head}:probe")
    return probe is not None and probe[0] == "figure"


def _canonicalise_planned_analysis_role(
    value: object,
    *,
    method: object,
) -> object:
    """Normalize a closed set of representation-only planner role variants."""

    if not isinstance(value, str):
        return value
    token = value.strip().casefold()
    if token in {"primary", "secondary", "sensitivity", "auxiliary"}:
        return token
    method_token = str(method or "").strip().casefold()
    if token == "robustness" and method_token == "robustness_sensitivity":
        return "sensitivity"
    return value


def _declared_field_names(model: type) -> set:
    """Read accepted fields from the declaring Pydantic model or dataclass."""

    fields = getattr(model, "model_fields", None)
    if fields is not None:
        return set(fields)
    if dataclasses.is_dataclass(model):
        return {field.name for field in dataclasses.fields(model)}
    raise TypeError(
        f"{model.__name__} declares neither pydantic model_fields nor "
        "dataclass fields, so its accepted Planner keys cannot be read "
        "from the schema; do not transcribe them by hand."
    )


class PlannerScientificProjectionError(ValueError):
    """The Planner emitted an unknown key inside a scientific value object."""

    issue_code = "planner_scientific_contract_unknown_key"
    owner = "easyicu.planning.plan_payload_projection_v1"

    def __init__(self, *, path: str, unknown_keys: List[str]) -> None:
        self.path = path
        self.unknown_keys = tuple(sorted(unknown_keys))
        super().__init__(
            f"{self.issue_code}: unknown key(s) at {path}: "
            + ", ".join(repr(key) for key in self.unknown_keys)
            + "; re-emit the scientific object using only its declared schema"
        )


def _require_exact_scientific_keys(
    raw: Dict[str, Any],
    *,
    allowed: set,
    path: str,
) -> None:
    unknown = [str(key) for key in raw if key not in allowed]
    if unknown:
        raise PlannerScientificProjectionError(path=path, unknown_keys=unknown)


def _require_runtime_supported_product_kinds(
    *,
    step_id: str,
    inputs: object,
    expected_outputs: object,
) -> None:
    """Reject typed product spellings the runtime cannot honour.

    Product *names* and the scientific dependency graph remain Planner-owned.
    The closed kind vocabulary is a representation/runtime contract: accepting
    ``text:x`` here only to reject it after the paid probe/replan cycle cannot
    make the plan more expressive.  A terminal ``report`` is intentionally a
    valid output but not a consumable input; the writer materialises it after
    the evidence-producing analysis steps have completed.
    """

    for field, values, supported in (
        ("inputs", inputs, RUNTIME_BINDABLE_TYPED_INPUT_KINDS),
        (
            "expected_outputs",
            expected_outputs,
            PLAN_MATERIALIZABLE_TYPED_OUTPUT_KINDS,
        ),
    ):
        if not isinstance(values, list):
            continue
        for index, raw in enumerate(values):
            product = _canonical_typed_product(raw) if isinstance(raw, str) else None
            if product is None or product[0] in supported:
                continue
            terminal_report_note = (
                " A report product is terminal writer output and cannot be "
                "consumed by another analysis step."
                if field == "inputs" and product[0] == "report"
                else ""
            )
            raise ValueError(
                f"Planner step {step_id!r} declares unsupported typed product "
                f"kind {product[0]!r} at {field}[{index}]. Supported kinds are "
                f"{sorted(supported)!r}.{terminal_report_note} Re-emit the same "
                "scientific plan with a runtime-supported product kind."
            )


def _normalise_plan_payload(
    data: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, List[str]]]:
    """Drop invented keys while preserving every field declared by the schema."""

    allowed_plan = _declared_field_names(AnalysisPlan)
    allowed_step = _declared_field_names(AnalysisStep)
    allowed_model_requirement = _declared_field_names(PlannedModelRequirement)
    allowed_consumption_contract = _declared_field_names(ArtifactConsumptionContract)
    allowed_table_one_spec = _declared_field_names(TableOneSpec)
    allowed_table_one_variable = _declared_field_names(TableOneVariableSpec)
    allowed_robustness_spec = _declared_field_names(RobustnessSpec)
    dropped: Dict[str, List[str]] = {
        "top_level": [],
        "steps": [],
        "model_requirements": [],
        "input_consumption_contracts": [],
        "table_one_spec": [],
        "robustness_specs": [],
    }
    out = {key: value for key, value in data.items() if key in allowed_plan}
    dropped["top_level"] = [str(key) for key in data if key not in allowed_plan]
    steps = []
    for idx, raw_step in enumerate(out.get("steps", []) or []):
        if not isinstance(raw_step, dict):
            continue
        step_payload = {
            key: value for key, value in raw_step.items() if key in allowed_step
        }
        step_id = raw_step.get("step_id") or f"step[{idx}]"
        dropped["steps"].extend(
            f"{step_id}:{key}" for key in raw_step if key not in allowed_step
        )
        if "planned_analysis_role" in step_payload:
            step_payload["planned_analysis_role"] = _canonicalise_planned_analysis_role(
                step_payload["planned_analysis_role"],
                method=step_payload.get("method"),
            )
        requirements = []
        for req_idx, raw_requirement in enumerate(
            step_payload.get("model_requirements", []) or []
        ):
            if not isinstance(raw_requirement, dict):
                requirements.append(raw_requirement)
                continue
            requirement_payload = {
                key: value
                for key, value in raw_requirement.items()
                if key in allowed_model_requirement
            }
            requirement_id = (
                raw_requirement.get("requirement_id")
                or f"step[{idx}].model_requirements[{req_idx}]"
            )
            _require_exact_scientific_keys(
                raw_requirement,
                allowed=allowed_model_requirement,
                path=f"steps[{idx}].model_requirements[{req_idx}]({requirement_id})",
            )
            if "analysis_role" in requirement_payload:
                requirement_payload["analysis_role"] = (
                    _canonicalise_planned_analysis_role(
                        requirement_payload["analysis_role"],
                        method=step_payload.get("method"),
                    )
                )
            requirements.append(requirement_payload)
        if "model_requirements" in step_payload:
            step_payload["model_requirements"] = requirements
        consumption_contracts = []
        for contract_idx, raw_contract in enumerate(
            step_payload.get("input_consumption_contracts", []) or []
        ):
            if not isinstance(raw_contract, dict):
                consumption_contracts.append(raw_contract)
                continue
            contract_payload = {
                key: value
                for key, value in raw_contract.items()
                if key in allowed_consumption_contract
            }
            contract_id = (
                raw_contract.get("input_key")
                or f"step[{idx}].input_consumption_contracts[{contract_idx}]"
            )
            _require_exact_scientific_keys(
                raw_contract,
                allowed=allowed_consumption_contract,
                path=(
                    f"steps[{idx}].input_consumption_contracts[{contract_idx}]"
                    f"({contract_id})"
                ),
            )
            if contract_payload:
                consumption_contracts.append(contract_payload)
            else:
                dropped["input_consumption_contracts"].append(
                    f"{contract_id}:empty_after_normalization"
                )
        if "input_consumption_contracts" in step_payload:
            step_payload["input_consumption_contracts"] = consumption_contracts
        raw_table_one = step_payload.get("table_one_spec")
        if isinstance(raw_table_one, dict):
            _require_exact_scientific_keys(
                raw_table_one,
                allowed=allowed_table_one_spec,
                path=f"steps[{idx}].table_one_spec",
            )
            table_one_payload = {
                key: value
                for key, value in raw_table_one.items()
                if key in allowed_table_one_spec
            }
            variables = []
            for variable_index, raw_variable in enumerate(
                table_one_payload.get("variables", []) or []
            ):
                if not isinstance(raw_variable, dict):
                    variables.append(raw_variable)
                    continue
                variable_payload = {
                    key: value
                    for key, value in raw_variable.items()
                    if key in allowed_table_one_variable
                }
                _require_exact_scientific_keys(
                    raw_variable,
                    allowed=allowed_table_one_variable,
                    path=(f"steps[{idx}].table_one_spec.variables[{variable_index}]"),
                )
                variables.append(variable_payload)
            table_one_payload["variables"] = variables
            step_payload["table_one_spec"] = table_one_payload
        raw_outputs = step_payload.get("expected_outputs")
        if isinstance(raw_outputs, list):
            normalised_outputs: List[Any] = []
            for item in raw_outputs:
                if _is_untyped_figure_alias_output(item):
                    suggested = str(item).strip().partition("_")[2]
                    raise ValueError(
                        f"Planner step {step_id!r} declares figure output "
                        f"{item!r} with an underscore instead of the typed "
                        "'figure:' separator; re-emit it as "
                        f"'figure:{suggested}' so the declared figure binds "
                        "to an exact output file."
                    )
                normalised_outputs.append(_canonicalise_figure_output_alias(item))
            figure_identity_aliases: Dict[Tuple[str, str], List[str]] = {}
            for candidate in normalised_outputs:
                if not isinstance(candidate, str):
                    continue
                identity = _canonical_typed_product(candidate)
                if identity is None or identity[0] != "figure":
                    continue
                figure_identity_aliases.setdefault(identity, []).append(candidate)
            collisions = {
                identity: aliases
                for identity, aliases in figure_identity_aliases.items()
                if len(aliases) > 1
            }
            if collisions:
                detail = "; ".join(
                    f"figure:{product} declared as {sorted(set(aliases))}"
                    for (_kind, product), aliases in sorted(collisions.items())
                )
                raise ValueError(
                    f"Planner step {step_id!r} declares the same figure "
                    f"product under more than one output alias ({detail}); "
                    "declare each figure exactly once as 'figure:<name>'."
                )
            step_payload["expected_outputs"] = normalised_outputs
        method_head = (
            str(step_payload.get("method") or "")
            .strip()
            .casefold()
            .split(" with ", 1)[0]
        )
        if method_head == "visualization" and not (
            step_payload.get("expected_outputs") or []
        ):
            raise ValueError(
                f"Planner step {step_id!r} is a visualization but declares no "
                "typed figure output; either drop the redundant step or re-emit "
                "it with exactly the intended 'figure:<name>' product."
            )
        _require_runtime_supported_product_kinds(
            step_id=str(step_id),
            inputs=step_payload.get("inputs"),
            expected_outputs=step_payload.get("expected_outputs"),
        )
        steps.append(step_payload)
    out["steps"] = steps
    specs = []
    for idx, raw_spec in enumerate(out.get("robustness_specs", []) or []):
        if not isinstance(raw_spec, dict):
            specs.append(raw_spec)
            continue
        spec_payload = {
            key: value
            for key, value in raw_spec.items()
            if key in allowed_robustness_spec
        }
        spec_id = raw_spec.get("spec_id") or f"robustness_specs[{idx}]"
        _require_exact_scientific_keys(
            raw_spec,
            allowed=allowed_robustness_spec,
            path=f"robustness_specs[{idx}]({spec_id})",
        )
        specs.append(spec_payload)
    if "robustness_specs" in out:
        out["robustness_specs"] = specs
    analysis_type = str(out.get("analysis_type") or "").strip().casefold()
    descriptive_robustness_steps = [
        str(step.get("step_id") or "")
        for step in steps
        if str(step.get("method") or "").strip().casefold().split(" with ", 1)[0]
        == "robustness_sensitivity"
    ]
    if analysis_type == "descriptive_epidemiology" and descriptive_robustness_steps:
        raise ValueError(
            "A descriptive_epidemiology plan cannot route "
            f"{descriptive_robustness_steps!r} through method "
            "'robustness_sensitivity': that executor re-estimates an already "
            "fitted primary effect with an interval. Re-emit the descriptive "
            "plan without robustness_specs/robustness_sensitivity, and use "
            "typed missingness or denominator audits for descriptive "
            "sensitivity instead."
        )
    narrative_execution_steps = [
        str(step.get("step_id") or "")
        for step in steps
        if str(step.get("method") or "").strip().casefold().split(" with ", 1)[0]
        in {
            "descriptive_interpretation",
            "result_interpretation",
            "report_writing",
            "manuscript_writing",
        }
    ]
    if narrative_execution_steps:
        raise ValueError(
            "Analysis steps cannot execute narrative interpretation or writing "
            f"methods for {narrative_execution_steps!r}. Re-emit only the "
            "evidence-producing statistical, audit, and figure steps. The "
            "gate-bound result interpreter and manuscript writer consume the "
            "verified products after analysis execution; do not generate "
            "Python code to narrate or draft them."
        )
    return out, dropped


__all__ = [
    "_canonicalise_figure_output_alias",
    "_canonicalise_planned_analysis_role",
    "_declared_field_names",
    "_is_untyped_figure_alias_output",
    "_normalise_plan_payload",
    "PlannerScientificProjectionError",
    "planner_descriptive_method_guidance",
    "planner_descriptive_robustness_guidance",
    "planner_adjusted_association_owner_guidance",
    "planner_endpoint_and_optional_science_guidance",
    "render_methodological_principles",
    "normalize_literature_citation_keys",
    "bind_literature_citation_authority",
    "literature_citation_retry_suffix",
    "required_method_layers_for_plan",
    "validate_literature_citation_bindings",
]
