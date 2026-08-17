"""Response-free diagnostics for structured model validation failures.

This owner is deliberately independent of Planner, Web, and artifact modules.
It turns an exception into a small closed projection: contract stage, schema
field coordinates, coarse issue types, and a one-way violation fingerprint.
Validator prose, rejected inputs, model output, prompts, and secrets never
cross this boundary.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List, Mapping, Optional

_SAFE_VALIDATION_STAGES = frozenset(
    {
        "json_shape",
        "schema_validation",
        "dependence_authority",
        "literature_authority",
        "robustness_contract",
        "article_contract",
        "know_how_authority",
        "table_one_contract",
        "distribution_contract",
        "analysis_family",
        "typed_context_binding",
        "adjustment_authority",
        "cohort_output_contract",
        "primary_result_contract",
    }
)

_SAFE_VALIDATION_FIELDS = frozenset(
    {
        "research_question",
        "analysis_type",
        "cohort",
        "display_labels",
        "robustness_specs",
        "robustness_intents",
        "know_how_decisions",
        "steps",
        "rationale",
        "step_id",
        "planned_analysis_role",
        "intent",
        "inputs",
        "expected_outputs",
        "method",
        "icu_rule_refs",
        "sensitivity_spec_ids",
        "literature_citation_keys",
        "literature_design_bindings",
        "model_requirements",
        "family_primary_result_requirement",
        "input_consumption_contracts",
        "figure_panels",
        "table_one_spec",
        "trajectory_stability_spec",
        "exposure_outcome_distribution_spec",
        "cohort_definition_spec",
        "descriptive_claim",
        "schema_version",
        "panel_id",
        "figure_output",
        "article_role",
        "chart_type",
        "source_products",
        "variables",
        "levels",
        "reference_level",
        "comparison_level",
        "dependence",
    }
)


def safe_validation_stage(value: Any) -> Optional[str]:
    """Return a closed validation-stage label or ``None``."""

    text = str(value or "").strip()
    return text if text in _SAFE_VALIDATION_STAGES else None


def infer_validation_stage(exc: BaseException) -> Optional[str]:
    """Locate a contract owner from closed exception/traceback coordinates.

    Exact prefixes are compared only in memory for checks raised directly by
    the Planner acceptance boundary; exception text itself is never returned.
    """

    explicit = safe_validation_stage(
        getattr(exc, "easyicu_structured_validation_stage", None)
    )
    if explicit is not None:
        return explicit
    if isinstance(exc, json.JSONDecodeError):
        return "json_shape"
    if callable(getattr(exc, "errors", None)):
        return "schema_validation"
    if type(exc).__name__ == "PlannerArticleContractError":
        return "article_contract"

    function_names: set[str] = set()
    current = exc.__traceback__
    while current is not None and len(function_names) < 32:
        function_names.add(current.tb_frame.f_code.co_name)
        current = current.tb_next
    stage_functions = (
        ("dependence_authority", {"bind_context_dependence_authority"}),
        ("literature_authority", {"validate_literature_citation_bindings"}),
        ("robustness_contract", {"validate_planner_robustness_specs"}),
        ("table_one_contract", {"_validate_table_one_observed_levels"}),
        (
            "typed_context_binding",
            {"validate_plan_typed_bindings_against_context"},
        ),
        ("adjustment_authority", {"validate_plan_against_adjustment_authority"}),
        ("cohort_output_contract", {"primary_analysis_cohort_plan_findings"}),
        ("primary_result_contract", {"validate_required_primary_result"}),
    )
    for stage, owner_functions in stage_functions:
        if function_names.intersection(owner_functions):
            return stage

    message = str(exc)
    direct_prefixes = (
        ("json_shape", "Planner LLM did not return parseable JSON"),
        ("json_shape", "Planner JSON root must be an object"),
        ("json_shape", "Planner step "),
        ("robustness_contract", "robustness spec"),
        ("know_how_authority", "Planner know_how_decisions"),
        ("know_how_authority", "Planner must record at least one claim-level"),
        (
            "distribution_contract",
            "Planner exposure/outcome distribution steps must declare",
        ),
        ("table_one_contract", "Planner Table 1 steps must declare"),
        ("analysis_family", "Unknown analysis_type declaration"),
        (
            "cohort_output_contract",
            "Planner primary-cohort output contract is not executable",
        ),
    )
    for stage, prefix in direct_prefixes:
        if message.startswith(prefix):
            return stage
    return None


def _safe_validation_issue_type(value: Any) -> str:
    text = str(value or "").strip().casefold()
    if text in {"type_error", "constraint_error", "union_error", "other"}:
        return text
    if text in {"missing", "extra_forbidden", "literal_error", "json_invalid"}:
        return text
    if text.startswith(
        ("string_", "list_", "dict_", "bool_", "int_", "float_", "model_")
    ):
        return "type_error"
    if text.startswith(("too_short", "too_long", "greater_than", "less_than")):
        return "constraint_error"
    if text.startswith("union_"):
        return "union_error"
    if text in {"value_error", "assertion_error"}:
        return text
    return "other"


def safe_validation_issues(exc: BaseException) -> List[Dict[str, Any]]:
    """Project schema coordinates without validator prose or rejected inputs."""

    errors = getattr(exc, "errors", None)
    if not callable(errors):
        return []
    try:
        records = list(errors())
    except Exception:  # noqa: BLE001 - diagnostics must not affect execution
        return []
    projected: List[Dict[str, Any]] = []
    for record in records[:40]:
        if not isinstance(record, Mapping):
            continue
        location: List[Any] = []
        raw_location = record.get("loc", ())
        if isinstance(raw_location, (list, tuple)):
            for part in raw_location[:8]:
                if isinstance(part, int) and not isinstance(part, bool):
                    location.append(max(0, min(9999, int(part))))
                elif str(part) in _SAFE_VALIDATION_FIELDS:
                    location.append(str(part))
                else:
                    location.append("<other>")
        projected.append(
            {
                "location": location or ["<root>"],
                "issue_type": _safe_validation_issue_type(record.get("type")),
            }
        )
    return projected


def safe_projected_validation_issues(value: Any) -> List[Dict[str, Any]]:
    """Revalidate an exception-attached issue projection as untrusted input."""

    if not isinstance(value, list):
        return []
    projected: List[Dict[str, Any]] = []
    for record in value[:40]:
        if not isinstance(record, Mapping):
            continue
        location: List[Any] = []
        for part in list(record.get("location") or [])[:8]:
            if isinstance(part, int) and not isinstance(part, bool):
                location.append(max(0, min(9999, int(part))))
            elif str(part) in _SAFE_VALIDATION_FIELDS or part in {"<root>", "<other>"}:
                location.append(str(part))
            else:
                location.append("<other>")
        projected.append(
            {
                "location": location or ["<root>"],
                "issue_type": _safe_validation_issue_type(record.get("issue_type")),
            }
        )
    return projected


def violation_sha256(rendered_failure: str) -> str:
    """Fingerprint one private violation without exposing its text."""

    return hashlib.sha256(rendered_failure.encode("utf-8")).hexdigest()


__all__ = [
    "infer_validation_stage",
    "safe_projected_validation_issues",
    "safe_validation_issues",
    "safe_validation_stage",
    "violation_sha256",
]
